"""
main.py
=======
Orchestrates Stage 1 (U-Net segmentation) and Stage 2 (EfficientNet-B0
classification) as separate, independently controllable training phases.

The spectrogram always flows live through the pipeline — no patches are
written to disk. Stage separation means:
  1. You can pretrain the U-Net alone, inspect mask quality, then hand off.
  2. You can retrain the classifier with frozen U-Net weights without
     re-running Stage 1.
  3. Phase 2 of Stage 2 fine-tunes both jointly at different LRs.

Usage
-----
# Full pipeline (Stage 1 then Stage 2)
python main.py

# Skip Stage 1, use existing U-Net checkpoint for Stage 2
python main.py --skip_stage1 --unet_ckpt checkpoints/unet_best.pt

# Tune training length / LR
python main.py --s1_epochs 30 --s2_phase1_epochs 15 --s2_phase2_epochs 40

# Resume Stage 2 from a classifier checkpoint
python main.py --skip_stage1 --resume checkpoints/classifier_best.pt
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from drone_dataloader import build_dataloaders
from RoiExtractor import (
    DroneROIUNet,
    ROIExtractor,
    build_proxy_mask,
    train_unet,
    load_unet,
)
from EfficientNetB0_Classification import (
    DroneCLSNet,
    DronePipelineLoss,
    train_classifier,
    load_classifier,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Two-stage Drone Detection Pipeline")

    # Data
    p.add_argument("--root",               default="output_spectrograms/")
    p.add_argument("--subsets",            nargs="+", default=["BOTH", "CLEAN"])
    p.add_argument("--img_size",           nargs=2, type=int, default=[256, 512])
    p.add_argument("--batch_size",         type=int,   default=16)
    p.add_argument("--workers",            type=int,   default=4)
    p.add_argument("--seed",               type=int,   default=42)

    # Stage control
    p.add_argument("--skip_stage1",        action="store_true",
                   help="Skip Stage-1 training and load --unet_ckpt instead.")
    p.add_argument("--unet_ckpt",          default="checkpoints/unet_best.pt",
                   help="Stage-1 checkpoint to load (used when --skip_stage1).")
    p.add_argument("--resume",             default=None,
                   help="Resume Stage-2 from this classifier checkpoint.")
    p.add_argument("--checkpoint_dir",     default="checkpoints/")

    # Stage 1 — U-Net
    p.add_argument("--s1_epochs",          type=int,   default=30)
    p.add_argument("--s1_lr",              type=float, default=1e-3)
    p.add_argument("--unet_base_filters",  type=int,   default=32)

    # Stage 2 — Classifier
    p.add_argument("--s2_phase1_epochs",   type=int,   default=15)
    p.add_argument("--s2_phase1_lr",       type=float, default=1e-3)
    p.add_argument("--s2_phase2_epochs",   type=int,   default=40)
    p.add_argument("--s2_phase2_lr",       type=float, default=1e-4)
    p.add_argument("--unfreeze_block",     type=int,   default=5)
    p.add_argument("--unet_lr_scale",      type=float, default=0.1)
    p.add_argument("--seg_weight",         type=float, default=1.0)
    p.add_argument("--cls_weight",         type=float, default=1.0)
    p.add_argument("--cls_dropout",        type=float, default=0.2)
    p.add_argument("--cls_drop_connect",   type=float, default=0.2)
    p.add_argument("--roi_strategy",       default="multiply",
                   choices=["multiply", "bbox"])
    p.add_argument("--weight_decay",       type=float, default=1e-4)
    p.add_argument("--log_interval",       type=int,   default=50)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random; random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def format_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"


def save_test_file_list(records: list, checkpoint_dir: str):
    out_dir    = Path(checkpoint_dir)
    all_path   = out_dir / "test_files.txt"
    wrong_path = out_dir / "test_wrong.txt"

    header  = f"{'#':>5}  {'Filename':<55}  {'True':<12}  {'Pred':<12}  {'OK':>4}  {'Conf':>7}\n"
    divider = "-" * 105 + "\n"

    with open(all_path, "w") as fa, open(wrong_path, "w") as fw:
        fa.write(header); fa.write(divider)
        fw.write("Misclassified samples\n"); fw.write(header); fw.write(divider)
        wrong = 0
        for i, r in enumerate(records, 1):
            ok  = "OK" if r["correct"] else "FAIL"
            row = (f"{i:>5}  {r['filename']:<55}  {r['true']:<12}  "
                   f"{r['pred']:<12}  {ok:>4}  {r['confidence']:>6.1%}\n")
            fa.write(row)
            if not r["correct"]:
                fw.write(row); wrong += 1

    print(f"  Test results  → {all_path}")
    print(f"  Wrong samples → {wrong_path}  ({wrong} samples)")


# ─────────────────────────────────────────────────────────────────────────────
#  Final test evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(
    unet        : DroneROIUNet,
    extractor   : ROIExtractor,
    cls_net     : DroneCLSNet,
    test_loader,
    criterion   : DronePipelineLoss,
    device,
    no_drone_idx: int,
    class_names : list,
    proxy_threshold: float = 0.5,
):
    unet.eval(); cls_net.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct, class_total = {}, {}
    records = []
    sample_offset = 0
    test_dataset  = test_loader.dataset   # _TransformSubset

    pbar = tqdm(test_loader, desc="[Test]", ascii=True, ncols=100)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        gt_mask   = build_proxy_mask(images, labels, no_drone_idx,
                                     proxy_threshold)
        pred_mask = unet(images)
        roi       = extractor(images, pred_mask)
        logits    = cls_net(roi)
        probs     = torch.softmax(logits, dim=1)
        preds     = logits.argmax(dim=1)

        losses = criterion(pred_mask, gt_mask, logits, labels)
        B = images.size(0)
        total_loss += losses["total"].item() * B
        correct    += (preds == labels).sum().item()
        total      += B

        for i, (pred, lbl) in enumerate(
            zip(preds.cpu().tolist(), labels.cpu().tolist())
        ):
            class_total[lbl]   = class_total.get(lbl, 0) + 1
            class_correct[lbl] = class_correct.get(lbl, 0) + int(pred == lbl)

            global_idx = test_dataset.subset.indices[sample_offset + i]
            img_path, _ = test_dataset.subset.dataset.samples[global_idx]
            records.append({
                "filename"  : Path(img_path).name,
                "true"      : class_names[lbl],
                "pred"      : class_names[pred],
                "correct"   : pred == lbl,
                "confidence": float(probs[i][pred].item()),
            })

        sample_offset += B
        pbar.set_postfix({
            "loss": f"{total_loss/total:.4f}",
            "acc" : f"{100.*correct/total:.1f}%",
        })

    per_class_acc = {
        k: 100.0 * class_correct.get(k, 0) / v
        for k, v in class_total.items()
    }
    return {
        "loss"         : total_loss / total,
        "acc"          : 100.0 * correct / total,
        "per_class_acc": per_class_acc,
        "records"      : records,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    sep = "=" * 65
    print(f"\n{sep}")
    print("  Two-Stage Drone Detection Pipeline")
    print(sep)
    print(f"  Device   : {device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if device.type == "cuda" else ""))
    print(f"  Subsets  : {args.subsets}")
    print(f"  Img size : {tuple(args.img_size)}")
    print(f"{sep}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("► Loading dataset ...")
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        root        = args.root,
        subsets     = args.subsets,
        img_size    = tuple(args.img_size),
        batch_size  = args.batch_size,
        num_workers = args.workers,
        seed        = args.seed,
    )

    class_names  = meta["class_names"]
    class_to_idx = meta["class_to_idx"]
    no_drone_idx = class_to_idx.get("NO_DRONE", -1)

    print(f"  Classes ({meta['num_classes']}): {class_names}")
    print(f"  NO_DRONE index : {no_drone_idx}")
    print(f"  Train: {meta['n_train']}  Val: {meta['n_val']}  "
          f"Test: {meta['n_test']}\n")

    # Print test manifest upfront (matches original main.py behaviour)
    print("  Test set:")
    test_ds = test_loader.dataset
    for i, global_idx in enumerate(test_ds.subset.indices):
        img_path, label = test_ds.subset.dataset.samples[global_idx]
        print(f"    {i+1:>4}  {Path(img_path).name:<55}  {class_names[label]}")
    print()

    # ── Stage 1: U-Net ───────────────────────────────────────────────────────
    print(f"{sep}")
    if args.skip_stage1:
        print("► Stage 1: Skipped — loading checkpoint")
        unet_ckpt = args.unet_ckpt
        print(f"  U-Net checkpoint: {unet_ckpt}")
    else:
        print("► Stage 1: Training U-Net segmentation model")
        print(sep)
        unet_ckpt = train_unet(
            train_loader    = train_loader,
            val_loader      = val_loader,
            no_drone_idx    = no_drone_idx,
            ckpt_dir        = args.checkpoint_dir,
            base_filters    = args.unet_base_filters,
            in_channels     = 3,
            img_h           = args.img_size[0],
            img_w           = args.img_size[1],
            epochs          = args.s1_epochs,
            lr              = args.s1_lr,
            weight_decay    = args.weight_decay,
            log_interval    = args.log_interval,
        )
    print()

    # ── Stage 2: Classifier ──────────────────────────────────────────────────
    print(f"{sep}")
    print("► Stage 2: Fine-tuning EfficientNet-B0 classifier")
    print(sep)

    if args.resume:
        # Resume from a full Stage-2 checkpoint (has both unet + cls states)
        unet, extractor, cls_net, _ = load_classifier(args.resume, device)
        cls_ckpt = args.resume
        print(f"  Resumed from: {args.resume}")
    else:
        cls_ckpt = train_classifier(
            train_loader      = train_loader,
            val_loader        = val_loader,
            no_drone_idx      = no_drone_idx,
            class_names       = class_names,
            unet_ckpt         = unet_ckpt,
            ckpt_dir          = args.checkpoint_dir,
            num_classes       = meta["num_classes"],
            in_channels       = 3,
            unet_base_filters = args.unet_base_filters,
            img_h             = args.img_size[0],
            img_w             = args.img_size[1],
            roi_strategy      = args.roi_strategy,
            seg_weight        = args.seg_weight,
            cls_weight        = args.cls_weight,
            cls_dropout       = args.cls_dropout,
            cls_drop_connect  = args.cls_drop_connect,
            phase1_epochs     = args.s2_phase1_epochs,
            phase1_lr         = args.s2_phase1_lr,
            phase2_epochs     = args.s2_phase2_epochs,
            phase2_lr         = args.s2_phase2_lr,
            unfreeze_block    = args.unfreeze_block,
            unet_lr_scale     = args.unet_lr_scale,
            weight_decay      = args.weight_decay,
            log_interval      = args.log_interval,
        )
        unet, extractor, cls_net, _ = load_classifier(cls_ckpt, device)

    # ── Final Test ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("► Final Test Evaluation  (best model)")
    print(sep)

    criterion = DronePipelineLoss(args.seg_weight, args.cls_weight)
    test_metrics = evaluate_test(
        unet=unet, extractor=extractor, cls_net=cls_net,
        test_loader=test_loader, criterion=criterion, device=device,
        no_drone_idx=no_drone_idx, class_names=class_names,
    )

    # Per-sample table
    records = test_metrics["records"]
    print(f"\n  {'#':>5}  {'File':<50}  {'True':<12}  {'Pred':<12}  {'OK':>4}  {'Conf':>7}")
    print(f"  {'-'*5}  {'-'*50}  {'-'*12}  {'-'*12}  {'-'*4}  {'-'*7}")
    for i, r in enumerate(records, 1):
        ok = "OK" if r["correct"] else "FAIL"
        print(f"  {i:>5}  {r['filename']:<50}  {r['true']:<12}  "
              f"{r['pred']:<12}  {ok:>4}  {r['confidence']:>6.1%}")

    print(f"\n  Test Loss     : {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy : {test_metrics['acc']:.2f}%")
    print(f"\n  Per-class Test Accuracy:")
    for cls_idx, acc in sorted(test_metrics["per_class_acc"].items()):
        bar = "#" * int(acc / 5)
        print(f"    {class_names[cls_idx]:12s}: {acc:5.1f}%  {bar}")

    save_test_file_list(records, args.checkpoint_dir)

    print(f"\n{sep}")
    print("  Done")
    print(f"  Test acc: {test_metrics['acc']:.2f}%")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()