"""
train.py
========
Two-phase training for FusedDronePipeline.

Phase 1 — Classifier head warmup  (U-Net frozen)
    U-Net weights frozen, only EfficientNet linear head trains.
    Uses classification loss only.
    Protects pretrained U-Net features while the head adapts to 8 classes.

Phase 2 — Full end-to-end  (everything trains jointly)
    All weights unfrozen with per-component learning rates:
        U-Net          :  lr × 0.1   (low — already partially trained)
        EfficientNet b1-b4: lr × 0.1 (low — transfer well from ImageNet)
        EfficientNet b5-b7: lr × 0.5 (mid — task-specific, needs adaptation)
        Classifier head:    lr × 1.0 (full)
    Uses FusedPipelineLoss: seg (BCE) + cls (CrossEntropy).
    Classification gradient flows back through ROI extraction into U-Net —
    the U-Net learns masks that directly help classification, not just masks
    that match the energy proxy target.

Usage
-----
    python train.py
    python train.py --epochs 60 --batch_size 16
    python train.py --resume checkpoints/fused_best.pt
"""

import os
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from drone_dataloader import build_dataloaders
from fused_model import (
    FusedDronePipeline,
    FusedPipelineLoss,
    build_proxy_mask,
    save_checkpoint,
    load_pipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Train FusedDronePipeline")

    # Data
    p.add_argument("--root",           default="../output_spectrograms/")
    p.add_argument("--subsets",        nargs="+", default=["BLUE","BOTH", "CLEAN","WIFI"])
    p.add_argument("--img_size",       nargs=2, type=int, default=[256, 512])
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--seed",           type=int,   default=42)

    # Model
    p.add_argument("--base_filters",   type=int,   default=32)
    p.add_argument("--cls_dropout",    type=float, default=0.3)
    p.add_argument("--cls_drop_connect",type=float,default=0.2)

    # Phase 1 — head warmup
    p.add_argument("--p1_epochs",      type=int,   default=10)
    p.add_argument("--p1_lr",          type=float, default=1e-3)

    # Phase 2 — full end-to-end
    p.add_argument("--p2_epochs",      type=int,   default=50)
    p.add_argument("--p2_lr",          type=float, default=1e-4)

    # Loss weights
    p.add_argument("--seg_weight",     type=float, default=0.3,
                   help="Weight for BCE segmentation loss (keep low — cls drives training)")
    p.add_argument("--cls_weight",     type=float, default=1.0)
    p.add_argument("--label_smoothing",type=float, default=0.1)

    # Other
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--proxy_threshold",type=float, default=0.5)
    p.add_argument("--ckpt_dir",       default="checkpoints/")
    p.add_argument("--log_interval",   type=int,   default=50)
    p.add_argument("--resume",         default=None,
                   help="Resume from checkpoint (skips Phase 1)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def fmt_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


# ─────────────────────────────────────────────────────────────────────────────
#  One training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model, loader, criterion, optimizer,
    device, no_drone_idx, proxy_threshold,
    epoch, total_epochs, log_interval,
    use_seg_loss: bool = True,
):
    model.train()
    total_loss = seg_sum = cls_sum = correct = total = 0
    t0 = time.time()

    pbar = tqdm(loader, desc=f"Ep {epoch:03d}/{total_epochs} [train]",
                ascii=True, ncols=110, leave=True)

    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            proxy = build_proxy_mask(images, labels, no_drone_idx,
                                     proxy_threshold)

        optimizer.zero_grad(set_to_none=True)

        logits, mask_logit = model(images, return_mask=True)
        losses = criterion(mask_logit, proxy, logits, labels)

        # In Phase 1 (use_seg_loss=False) only cls loss drives gradients
        loss = losses["cls"] if not use_seg_loss else losses["total"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        B = images.size(0)
        total_loss += losses["total"].item() * B
        seg_sum    += losses["seg"].item()   * B
        cls_sum    += losses["cls"].item()   * B
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += B

        if (i + 1) % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{total_loss/total:.4f}",
                "seg" : f"{seg_sum/total:.4f}",
                "cls" : f"{cls_sum/total:.4f}",
                "acc" : f"{100.*correct/total:.1f}%",
            })

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "seg" : seg_sum    / n,
        "cls" : cls_sum    / n,
        "acc" : 100. * correct / n,
        "time": time.time() - t0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, no_drone_idx,
             proxy_threshold, class_names, split="Val"):
    model.eval()
    total_loss = correct = total = 0
    class_correct = {}
    class_total   = {}

    pbar = tqdm(loader, desc=f"           [{split:5s}]",
                ascii=True, ncols=110, leave=True)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        proxy  = build_proxy_mask(images, labels, no_drone_idx,
                                  proxy_threshold)
        logits, mask_logit = model(images, return_mask=True)
        losses = criterion(mask_logit, proxy, logits, labels)

        preds   = logits.argmax(1)
        B       = images.size(0)
        total_loss += losses["total"].item() * B
        correct    += (preds == labels).sum().item()
        total      += B

        for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            class_total[l]   = class_total.get(l, 0) + 1
            class_correct[l] = class_correct.get(l, 0) + int(p == l)

        pbar.set_postfix({
            "loss": f"{total_loss/total:.4f}",
            "acc" : f"{100.*correct/total:.1f}%",
        })

    n = len(loader.dataset)
    per_class = {
        class_names[k]: 100. * class_correct.get(k, 0) / v
        for k, v in class_total.items()
    }
    return {
        "loss"     : total_loss / n,
        "acc"      : 100. * correct / n,
        "per_class": per_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    sep = "=" * 68
    print(f"\n{sep}")
    print("  Fused Drone Detection Pipeline — Training")
    print(f"{sep}")
    print(f"  Device     : {device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if device.type == "cuda" else ""))
    print(f"  Image size : {tuple(args.img_size)}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Subsets    : {args.subsets}")
    print(f"{sep}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("► Loading data ...")
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        root        = args.root,
        subsets     = args.subsets,
        img_size    = tuple(args.img_size),
        batch_size  = args.batch_size,
        num_workers = args.workers,
        seed        = args.seed,
    )
    class_names  = meta["class_names"]
    no_drone_idx = meta["class_to_idx"].get("NO_DRONE", -1)
    print(f"  Classes ({meta['num_classes']}): {class_names}")
    print(f"  Train: {meta['n_train']}  Val: {meta['n_val']}  "
          f"Test: {meta['n_test']}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_config = dict(
        num_classes      = meta["num_classes"],
        in_channels      = 3,
        base_filters     = args.base_filters,
        cls_dropout      = args.cls_dropout,
        cls_drop_connect = args.cls_drop_connect,
        roi_size         = 224,
    )

    if args.resume:
        print(f"► Resuming from {args.resume} (skipping Phase 1)")
        model = load_pipeline(args.resume, device)
    else:
        model = FusedDronePipeline(**model_config).to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_p:,}")
    print(f"  Trainable params: {trainable_p:,}\n")

    criterion = FusedPipelineLoss(
        seg_weight     = args.seg_weight,
        cls_weight     = args.cls_weight,
        label_smoothing= args.label_smoothing,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_ckpt    = os.path.join(args.ckpt_dir, "fused_best.pt")
    best_val_acc = 0.0
    history      = {"train_loss": [], "val_loss": [],
                    "train_acc": [],  "val_acc": [],  "lr": []}

    def maybe_save(epoch, val_acc, phase, extra=None):
        nonlocal best_val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {
                "epoch"       : epoch,
                "phase"       : phase,
                "model_state" : model.state_dict(),
                "model_config": model_config,
                "val_acc"     : val_acc,
                "class_names" : class_names,
            }
            if extra:
                state.update(extra)
            save_checkpoint(state, best_ckpt)
            print(f"  ✓ Best checkpoint saved  (val_acc={val_acc:.2f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 1 — classifier head warmup, U-Net frozen
    # ─────────────────────────────────────────────────────────────────────────
    if not args.resume:
        print(f"{sep}")
        print(f"► Phase 1 — Classifier head warmup  "
              f"({args.p1_epochs} epochs, lr={args.p1_lr})")
        print(f"  U-Net FROZEN — only linear head trains")
        print(f"{sep}\n")

        model.freeze_unet()
        model.freeze_cls_backbone()

        opt1 = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.p1_lr, weight_decay=args.weight_decay,
        )
        sch1 = CosineAnnealingLR(opt1, T_max=args.p1_epochs,
                                 eta_min=args.p1_lr * 0.01)

        for epoch in range(1, args.p1_epochs + 1):
            tr = train_epoch(
                model, train_loader, criterion, opt1,
                device, no_drone_idx, args.proxy_threshold,
                epoch, args.p1_epochs, args.log_interval,
                use_seg_loss=False,   # only cls loss in Phase 1
            )
            vl = validate(model, val_loader, criterion, device,
                          no_drone_idx, args.proxy_threshold, class_names)
            sch1.step()
            lr_now = sch1.get_last_lr()[0]

            history["train_loss"].append(tr["loss"])
            history["val_loss"].append(vl["loss"])
            history["train_acc"].append(tr["acc"])
            history["val_acc"].append(vl["acc"])
            history["lr"].append(lr_now)

            print(f"P1 Ep {epoch:03d}/{args.p1_epochs} | "
                  f"train_loss={tr['loss']:.4f} acc={tr['acc']:.1f}% | "
                  f"val_loss={vl['loss']:.4f} acc={vl['acc']:.1f}% | "
                  f"lr={lr_now:.2e} | {fmt_time(tr['time'])}")

            maybe_save(epoch, vl["acc"], phase=1)
        print()

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 2 — full end-to-end, all components
    # ─────────────────────────────────────────────────────────────────────────
    print(f"{sep}")
    print(f"► Phase 2 — Full end-to-end fine-tune  "
          f"({args.p2_epochs} epochs, cls_lr={args.p2_lr})")
    print(f"  U-Net lr={args.p2_lr*0.1:.1e}  "
          f"EfficientNet b5-7 lr={args.p2_lr*0.5:.1e}  "
          f"Head lr={args.p2_lr:.1e}")
    print(f"  Seg loss weight={args.seg_weight}  "
          f"Cls loss weight={args.cls_weight}")
    print(f"{sep}\n")

    model.unfreeze_unet()
    model.unfreeze_cls_from_block(block=5)

    opt2 = torch.optim.Adam(
        model.param_groups(
            unet_lr = args.p2_lr * 0.1,
            cls_lr  = args.p2_lr,
        ),
        weight_decay=args.weight_decay,
    )
    sch2 = CosineAnnealingLR(opt2, T_max=args.p2_epochs,
                             eta_min=args.p2_lr * 0.001)

    for epoch in range(1, args.p2_epochs + 1):
        tr = train_epoch(
            model, train_loader, criterion, opt2,
            device, no_drone_idx, args.proxy_threshold,
            epoch, args.p2_epochs, args.log_interval,
            use_seg_loss=True,   # both seg + cls losses in Phase 2
        )
        vl = validate(model, val_loader, criterion, device,
                      no_drone_idx, args.proxy_threshold, class_names)
        sch2.step()
        lr_now = sch2.get_last_lr()[0]

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_acc"].append(vl["acc"])
        history["lr"].append(lr_now)

        print(f"P2 Ep {epoch:03d}/{args.p2_epochs} | "
              f"train_loss={tr['loss']:.4f} seg={tr['seg']:.4f} "
              f"cls={tr['cls']:.4f} acc={tr['acc']:.1f}% | "
              f"val_loss={vl['loss']:.4f} acc={vl['acc']:.1f}% | "
              f"lr={lr_now:.2e} | {fmt_time(tr['time'])}")

        # Per-class accuracy every 10 epochs
        if epoch % 10 == 0:
            print("  Per-class val accuracy:")
            for cls_name, acc in sorted(vl["per_class"].items()):
                bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
                print(f"    {cls_name:12s} [{bar}] {acc:.1f}%")

        maybe_save(epoch, vl["acc"], phase=2)
        print()

    # ── Final test ────────────────────────────────────────────────────────────
    print(f"{sep}")
    print("► Final Test Evaluation")
    print(f"{sep}")

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded best model from epoch {ckpt['epoch']}")

    test_metrics = validate(model, test_loader, criterion, device,
                            no_drone_idx, args.proxy_threshold,
                            class_names, split="Test")

    print(f"\n  Test Loss     : {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy : {test_metrics['acc']:.2f}%")
    print(f"\n  Per-class Test Accuracy:")
    for cls_name, acc in sorted(test_metrics["per_class"].items()):
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"    {cls_name:12s} [{bar}] {acc:.1f}%")

    # Save history
    np.save(os.path.join(args.ckpt_dir, "history.npy"), history)

    print(f"\n{sep}")
    print(f"  Best val acc  : {best_val_acc:.2f}%")
    print(f"  Test acc      : {test_metrics['acc']:.2f}%")
    print(f"  Checkpoint    : {best_ckpt}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()