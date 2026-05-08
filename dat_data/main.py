"""
main.py  (refactored)
======================
Changes from previous version
------------------------------
Strategy 4 — Proxy mask is now NO_DRONE-aware.
    When the ground-truth label is NO_DRONE (index looked up from meta),
    the gt_mask is forced to all-zeros so the U-Net learns that there is
    no drone ROI to segment in background-only samples.

    Additionally, model() now returns raw logits (Softmax removed from
    DroneCLSNet), so argmax and CrossEntropyLoss are applied to logits
    directly — no double-softmax.

All other training logic (cosine LR, gradient clipping, checkpointing,
per-class accuracy, history saving) is unchanged.

Usage:
    python main.py
    python main.py --epochs 50 --batch_size 16 --lr 1e-4
    python main.py --resume checkpoints/best_model.pth
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from drone_dataloader import build_dataloaders
from roi import DronePipeline, PipelineLoss


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description="Train Drone Detection Pipeline")
    p.add_argument("--root",              default="output_spectrograms/")
    p.add_argument("--subsets",           nargs="+", default=["BOTH"])
    p.add_argument("--img_size",          nargs=2, type=int, default=[256, 512])
    p.add_argument("--batch_size",        type=int,   default=16)
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--weight_decay",      type=float, default=1e-4)
    p.add_argument("--seg_weight",        type=float, default=1.0)
    p.add_argument("--cls_weight",        type=float, default=1.0)
    p.add_argument("--workers",           type=int,   default=4)
    p.add_argument("--unet_base_filters", type=int,   default=32,
                   help="U-Net base filter count. 32≈8M params (default), 64≈31M params.")
    p.add_argument("--resume",            default=None)
    p.add_argument("--checkpoint_dir",    default="checkpoints/")
    p.add_argument("--log_interval",      type=int,   default=10)
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Full determinism on GPU (slight speed cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"    [✓] Checkpoint saved → {path}")


def load_checkpoint(path: str, model, optimizer, scheduler, device):
    print(f"  Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch  = ckpt["epoch"] + 1
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    print(f"  Resumed from epoch {ckpt['epoch']}  "
          f"best_val_acc={best_val_acc:.2f}%")
    return start_epoch, best_val_acc


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"


# ═════════════════════════════════════════════════════════════════════════════
# ░░  Strategy 4 — NO_DRONE-aware proxy mask builder                       ░░
# ═════════════════════════════════════════════════════════════════════════════

def build_proxy_mask(
    images: torch.Tensor,
    labels: torch.Tensor,
    no_drone_idx: int,
    threshold: float = 0.7,
) -> torch.Tensor:
    """
    Build the weak-supervision binary mask used as U-Net gt_mask.

    For DRONE samples   : energy-threshold proxy (same as before)
    For NO_DRONE samples: all-zeros mask  ← Strategy 4

    Args:
        images       : (B, C, H, W) normalised float tensor
        labels       : (B,)         integer class labels
        no_drone_idx : integer label index for the NO_DRONE class
        threshold    : energy binarisation cutoff (default 0.5)

    Returns:
        gt_mask : (B, 1, H, W) binary float32 tensor
    """
    # ── Energy-threshold proxy (applied to all samples first) ─────────────────
    energy  = images.mean(dim=1, keepdim=True)                # (B, 1, H, W)
    e_min   = energy.flatten(1).min(1)[0].view(-1, 1, 1, 1)
    e_max   = energy.flatten(1).max(1)[0].view(-1, 1, 1, 1)
    gt_mask = (energy - e_min) / (e_max - e_min + 1e-8)
    gt_mask = (gt_mask > threshold).float()

    # ── Strategy 4: zero out mask for NO_DRONE samples ────────────────────────
    # Where the label is NO_DRONE, there is no drone RF region to segment.
    # Teaching the U-Net to predict an empty mask for these samples makes the
    # segmentation semantically consistent with the classification task.
    no_drone_mask = (labels == no_drone_idx)                  # (B,) bool
    if no_drone_mask.any():
        gt_mask[no_drone_mask] = 0.0

    return gt_mask


# ═════════════════════════════════════════════════════════════════════════════
# Training epoch
# ═════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model, loader, criterion, optimizer, device,
    epoch, total_epochs, log_interval,
    no_drone_idx: int,
):
    model.train()

    total_loss    = 0.0
    total_seg     = 0.0
    total_cls     = 0.0
    total_correct = 0
    total_samples = 0
    t0 = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{total_epochs} [Train]",
                leave=True, dynamic_ncols=True)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)   # (B, 3, H, W)
        labels = labels.to(device, non_blocking=True)   # (B,)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward ──────────────────────────────────────────────────────────
        logits, mask = model(images, return_mask=True)
        # logits : (B, num_classes)  raw scores — NO softmax
        # mask   : (B, 1, H, W)     sigmoid output ∈ [0, 1]

        # ── Proxy gt_mask (Strategy 4: NO_DRONE-aware) ───────────────────────
        with torch.no_grad():
            gt_mask = build_proxy_mask(
                images, labels, no_drone_idx, threshold=0.7
            )

        # ── Loss ─────────────────────────────────────────────────────────────
        losses = criterion(mask, gt_mask, logits, labels)
        loss   = losses["total"]

        # ── Backward + clip + step ────────────────────────────────────────────
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Metrics (argmax on logits — no softmax needed for accuracy) ───────
        B = images.size(0)
        total_loss    += loss.item()          * B
        total_seg     += losses["seg"].item() * B
        total_cls     += losses["cls"].item() * B
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += B

        pbar.set_postfix({
            "loss": f"{total_loss/total_samples:.4f}",
            "acc" : f"{100.*total_correct/total_samples:.1f}%",
            "seg" : f"{total_seg/total_samples:.4f}",
            "cls" : f"{total_cls/total_samples:.4f}",
        })

    epoch_time = time.time() - t0
    return {
        "loss"     : total_loss    / total_samples,
        "seg_loss" : total_seg     / total_samples,
        "cls_loss" : total_cls     / total_samples,
        "acc"      : 100.0 * total_correct / total_samples,
        "time"     : epoch_time,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Validation / Test
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model, loader, criterion, device,
    no_drone_idx: int,
    split: str = "Val",
):
    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    class_correct: dict[int, int] = {}
    class_total:   dict[int, int] = {}

    pbar = tqdm(loader, desc=f"           [{split:5s}]",
                leave=True, dynamic_ncols=True)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, mask = model(images, return_mask=True)

        # Strategy 4: NO_DRONE-aware proxy mask in val/test too (for seg loss)
        gt_mask = build_proxy_mask(images, labels, no_drone_idx, threshold=0.7)
        losses  = criterion(mask, gt_mask, logits, labels)

        B             = images.size(0)
        total_loss   += losses["total"].item() * B
        preds         = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += B

        for pred, lbl in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            class_total[lbl]   = class_total.get(lbl, 0) + 1
            class_correct[lbl] = class_correct.get(lbl, 0) + int(pred == lbl)

        pbar.set_postfix({
            "loss": f"{total_loss/total_samples:.4f}",
            "acc" : f"{100.*total_correct/total_samples:.1f}%",
        })

    per_class_acc = {
        k: 100.0 * class_correct.get(k, 0) / v
        for k, v in class_total.items()
    }
    return {
        "loss"          : total_loss / total_samples,
        "acc"           : 100.0 * total_correct / total_samples,
        "per_class_acc" : per_class_acc,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    sep = "═" * 65
    print(f"\n{sep}")
    print("  Drone Detection Pipeline — Training  (refactored)")
    print(f"{sep}")
    print(f"  Device           : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})"
             if device.type == "cuda" else ""))
    print(f"  Subsets          : {args.subsets}")
    print(f"  Image size       : {tuple(args.img_size)}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  LR               : {args.lr}")
    print(f"  UNet base filters: {args.unet_base_filters}  "
          f"(~{8 if args.unet_base_filters==32 else 31}M params)")
    print(f"{sep}\n")

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    print("► Step 1: Loading dataset …")
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        root        = args.root,
        subsets     = args.subsets,
        img_size    = tuple(args.img_size),
        batch_size  = args.batch_size,
        num_workers = args.workers,
        seed        = args.seed,
    )

    # ── Resolve NO_DRONE label index ─────────────────────────────
    class_to_idx  = meta["class_to_idx"]
    no_drone_idx  = class_to_idx.get("NO_DRONE", -1)
    if no_drone_idx == -1:
        print("  WARNING: 'NO_DRONE' class not found in dataset. "
              "NO_DRONE mask zeroing will be skipped.")
    else:
        print(f"  NO_DRONE label index : {no_drone_idx}  "
              f"(proxy mask will be zeroed for this class)")

    print(f"  Classes      : {meta['num_classes']}  →  {meta['class_names']}")
    print(f"  Train        : {meta['n_train']} samples  "
          f"({len(train_loader)} batches)")
    print(f"  Val          : {meta['n_val']} samples  "
          f"({len(val_loader)} batches)")
    print(f"  Test         : {meta['n_test']} samples  "
          f"({len(test_loader)} batches)")

    # ── Step 2: Model ─────────────────────────────────────────────────────────
    print("\n► Step 2: Building model …")
    model = DronePipeline(
        num_classes       = meta["num_classes"],
        in_channels       = 3,
        unet_base_filters = args.unet_base_filters,
        # 64: 31M  parameters
        # 32: 8M parameters
        roi_output_size   = (224, 224),
        mask_threshold    = 0.7,
        roi_strategy      = "multiply",
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")

    # ── Step 3: Optimizer + Scheduler + Loss ──────────────────────────────────
    print("\n► Step 3: Setting up optimizer …")
    optimizer = optim.Adam(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max   = args.epochs,
        eta_min = 1e-6,
    )
    criterion = PipelineLoss(
        seg_weight = args.seg_weight,
        cls_weight = args.cls_weight,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 1
    best_val_acc = 0.0
    if args.resume:
        start_epoch, best_val_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    # ── History ───────────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }

    # ── Step 4: Training Loop ─────────────────────────────────────────────────
    print(f"\n► Step 4: Training for {args.epochs} epochs …\n")
    total_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.epochs, args.log_interval,
            no_drone_idx=no_drone_idx,
        )

        val_metrics = evaluate(
            model, val_loader, criterion, device,
            no_drone_idx=no_drone_idx,
            split="Val",
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["lr"].append(current_lr)

        print(
            f"  Epoch {epoch:3d}/{args.epochs}"
            f"  train_loss={train_metrics['loss']:.4f}"
            f"  train_acc={train_metrics['acc']:5.1f}%"
            f"  val_loss={val_metrics['loss']:.4f}"
            f"  val_acc={val_metrics['acc']:5.1f}%"
            f"  lr={current_lr:.2e}"
            f"  time={format_time(train_metrics['time'])}"
        )

        # Per-class accuracy every 10 epochs
        if epoch % 10 == 0:
            print("  Per-class val accuracy:")
            for cls_idx, acc in sorted(val_metrics["per_class_acc"].items()):
                cls_name = meta["class_names"][cls_idx]
                bar = "█" * int(acc / 5)
                print(f"    {cls_name:12s}: {acc:5.1f}%  {bar}")

        # Save best checkpoint
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            save_checkpoint(
                {
                    "epoch"           : epoch,
                    "model_state"     : model.state_dict(),
                    "optimizer_state" : optimizer.state_dict(),
                    "scheduler_state" : scheduler.state_dict(),
                    "best_val_acc"    : best_val_acc,
                    "meta"            : meta,
                    "args"            : vars(args),
                },
                path=os.path.join(args.checkpoint_dir, "best_model.pth"),
            )

    total_time = time.time() - total_start

    # ── Step 5: Final Test Evaluation ─────────────────────────────────────────
    print(f"\n{sep}")
    print("► Step 5: Final Test Evaluation  (best model)")
    print(sep)

    best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded best model from epoch {ckpt['epoch']}")

    test_metrics = evaluate(
        model, test_loader, criterion, device,
        no_drone_idx=no_drone_idx,
        split="Test",
    )

    print(f"\n  Test Loss     : {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy : {test_metrics['acc']:.2f}%")
    print("\n  Per-class Test Accuracy:")
    for cls_idx, acc in sorted(test_metrics["per_class_acc"].items()):
        cls_name = meta["class_names"][cls_idx]
        bar = "█" * int(acc / 5)
        print(f"    {cls_name:12s}: {acc:5.1f}%  {bar}")

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Training Complete")
    print(sep)
    print(f"  Total time    : {format_time(total_time)}")
    print(f"  Best val acc  : {best_val_acc:.2f}%")
    print(f"  Test acc      : {test_metrics['acc']:.2f}%")
    print(f"  Best model    : {best_ckpt}")
    print(f"\n  Loss curve (last 5 epochs):")
    print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}"
          f"  {'Train Acc':>10}  {'Val Acc':>10}")
    for i in range(max(0, len(history["train_loss"]) - 5),
                   len(history["train_loss"])):
        ep = i + 1
        print(f"  {ep:6d}  {history['train_loss'][i]:10.4f}"
              f"  {history['val_loss'][i]:10.4f}"
              f"  {history['train_acc'][i]:9.1f}%"
              f"  {history['val_acc'][i]:9.1f}%")
    print(f"{sep}\n")

    np.save(
        os.path.join(args.checkpoint_dir, "history.npy"),
        history,
    )
    print(f"  History saved → {args.checkpoint_dir}/history.npy")


if __name__ == "__main__":
    main()