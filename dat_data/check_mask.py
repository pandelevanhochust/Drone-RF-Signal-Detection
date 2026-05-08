"""
inspect_single_spectrogram.py
==============================
Takes ONE spectrogram PNG and returns three outputs saved as PNGs:

    1. pre_norm.png        — after resize + ToTensor, before ImageNet normalisation
                             (the "raw float" view, values in [0,1])
    2. proxy_energy_mask.png — binary proxy mask used as U-Net gt_mask during training
                             (mean-channel energy → min-max norm → threshold 0.5)
    3. roi_masked.png      — mask × pre-norm tensor  (what ROIExtractor feeds to the
                             classifier, before the final 224×224 resize)

Usage:
    python inspect_single_spectrogram.py --img path/to/spectrogram.png
    python inspect_single_spectrogram.py --img path/to/spectrogram.png --img_size 256 512 --threshold 0.5 --out_dir ./outputs
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# ImageNet stats (same as drone_dataloader.py)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    Convert a (C, H, W) float32 tensor in [0, 1] to a PIL RGB image.
    Values are clamped so nothing clips.
    """
    t = t.clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    """
    Convert a (1, H, W) or (H, W) binary float mask to a grayscale PIL image.
    White = mask ON (kept), black = mask OFF (zeroed).
    """
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    arr = (mask.numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def compute_proxy_mask(tensor: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    """
    Replicates the exact proxy-mask logic from main.py's train_one_epoch().

    Args:
        tensor    : (C, H, W) float32 tensor, values in [0, 1] — pre-normalisation
        threshold : binarisation cutoff (default 0.5, same as training)

    Returns:
        mask : (1, H, W) binary float32 tensor  (1 = keep, 0 = zero out)

    Logic (copied verbatim from main.py):
        energy  = images.mean(dim=1, keepdim=True)               # mean across channels
        e_min   = energy.flatten(1).min(1)[0].view(-1,1,1,1)
        e_max   = energy.flatten(1).max(1)[0].view(-1,1,1,1)
        gt_mask = (energy - e_min) / (e_max - e_min + 1e-8)      # min-max normalise
        gt_mask = (gt_mask > threshold).float()                   # binarise
    """
    # Add a batch dim so we match the batched shapes in main.py exactly
    x = tensor.unsqueeze(0)                                   # (1, C, H, W)
    energy = x.mean(dim=1, keepdim=True)                      # (1, 1, H, W)
    e_min  = energy.flatten(1).min(1)[0].view(-1, 1, 1, 1)   # (1, 1, 1, 1)
    e_max  = energy.flatten(1).max(1)[0].view(-1, 1, 1, 1)   # (1, 1, 1, 1)
    norm   = (energy - e_min) / (e_max - e_min + 1e-8)       # (1, 1, H, W)
    mask   = (norm > threshold).float()                       # (1, 1, H, W)
    return mask.squeeze(0)                                    # (1, H, W)


def apply_roi_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Replicates Stage-2 ROIExtractor (strategy='multiply') from roi.py.

    Args:
        tensor : (C, H, W) pre-norm float32 tensor
        mask   : (1, H, W) binary float32 mask

    Returns:
        roi : (C, H, W) masked tensor — background pixels are 0
    """
    return tensor * mask   # broadcast across C channels


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def inspect(
    img_path: str,
    img_size: tuple[int, int] = (256, 512),
    threshold: float = 0.5,
    out_dir: str = ".",
) -> dict[str, torch.Tensor]:
    """
    Full inspection pipeline for a single spectrogram PNG.

    Args:
        img_path  : path to input .png spectrogram
        img_size  : (H, W) — must both be divisible by 16 (U-Net constraint)
        threshold : proxy mask binarisation threshold (default 0.5)
        out_dir   : directory to save output PNGs

    Returns:
        dict with keys:
            "pre_norm"     : (3, H, W) tensor, values in [0, 1]
            "proxy_mask"   : (1, H, W) binary tensor
            "roi_masked"   : (3, H, W) tensor, background zeroed
    """
    H, W = img_size
    assert H % 16 == 0 and W % 16 == 0, (
        f"img_size ({H}, {W}) must both be divisible by 16 for the U-Net."
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load PNG as RGB (viridis colormap → 3 channels) ──────────────
    print(f"\n[1] Loading image: {img_path}")
    pil_img = Image.open(img_path).convert("RGB")
    print(f"    Original size : {pil_img.size}  (W×H)")

    # ── Step 2: Resize + ToTensor  (matches get_transforms("val")) ───────────
    resize_and_to_tensor = transforms.Compose([
        transforms.Resize(
            (H, W),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),   # PIL [0,255] uint8  →  float32 [0,1] (C,H,W)
    ])
    pre_norm: torch.Tensor = resize_and_to_tensor(pil_img)   # (3, H, W)
    print(f"    Pre-norm tensor : shape={tuple(pre_norm.shape)}"
          f"  min={pre_norm.min():.4f}  max={pre_norm.max():.4f}")

    # ── Step 3: Proxy energy mask (gt_mask from main.py) ─────────────────────
    print(f"\n[2] Computing proxy energy mask  (threshold={threshold}) …")
    proxy_mask: torch.Tensor = compute_proxy_mask(pre_norm, threshold)
    kept_pct = proxy_mask.mean().item() * 100
    print(f"    Mask shape  : {tuple(proxy_mask.shape)}")
    print(f"    Pixels kept : {kept_pct:.1f}%  ({proxy_mask.sum().int().item()} / {H*W})")

    # ── Step 4: ROI masked tensor (Stage-2 multiply strategy) ────────────────
    print(f"\n[3] Applying ROI mask (multiply strategy) …")
    roi_masked: torch.Tensor = apply_roi_mask(pre_norm, proxy_mask)
    nonzero_pct = (roi_masked.sum(dim=0) > 0).float().mean().item() * 100
    print(f"    ROI tensor shape    : {tuple(roi_masked.shape)}")
    print(f"    Non-zero pixel cols : {nonzero_pct:.1f}% of spatial positions")

    # ── Step 5: Save outputs ──────────────────────────────────────────────────
    stem = Path(img_path).stem

    pre_norm_path  = out_dir / f"{stem}__pre_norm.png"
    mask_path      = out_dir / f"{stem}__proxy_energy_mask.png"
    roi_path       = out_dir / f"{stem}__roi_masked.png"

    tensor_to_pil(pre_norm).save(pre_norm_path)
    mask_to_pil(proxy_mask).save(mask_path)
    tensor_to_pil(roi_masked).save(roi_path)

    print(f"\n[✓] Saved outputs to: {out_dir.resolve()}")
    print(f"    pre_norm          → {pre_norm_path.name}")
    print(f"    proxy_energy_mask → {mask_path.name}")
    print(f"    roi_masked        → {roi_path.name}")

    # ── Step 6: Print full tensor stats table ─────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'Tensor':<22}  {'Shape':<18}  {'Min':>7}  {'Max':>7}  {'Mean':>7}")
    print(f"{'─'*60}")
    for name, t in [
        ("pre_norm",          pre_norm),
        ("proxy_mask (float)", proxy_mask),
        ("roi_masked",         roi_masked),
    ]:
        print(
            f"  {name:<22}  {str(tuple(t.shape)):<18}"
            f"  {t.min().item():>7.4f}  {t.max().item():>7.4f}  {t.mean().item():>7.4f}"
        )
    print(f"{'─'*60}")

    # ── Optional: also print ImageNet-normalized stats (for reference) ────────
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    normed = normalize(pre_norm)
    print(f"\n  (For reference — after ImageNet normalization:)")
    print(
        f"  {'normed_tensor':<22}  {str(tuple(normed.shape)):<18}"
        f"  {normed.min().item():>7.4f}  {normed.max().item():>7.4f}  {normed.mean().item():>7.4f}"
    )

    return {
        "pre_norm"   : pre_norm,
        "proxy_mask" : proxy_mask,
        "roi_masked" : roi_masked,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Inspect a single spectrogram through the drone dataloader pipeline."
    )
    p.add_argument(
        "--img", required=True,
        help="Path to the input spectrogram PNG."
    )
    p.add_argument(
        "--img_size", nargs=2, type=int, default=[256, 512],
        metavar=("H", "W"),
        help="Resize target (H W). Both must be divisible by 16. Default: 256 512"
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Proxy mask binarisation threshold (default 0.5, same as training)."
    )
    p.add_argument(
        "--out_dir", default=".",
        help="Output directory for saved PNGs. Default: current directory."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    inspect(
        img_path  = args.img,
        img_size  = tuple(args.img_size),
        threshold = args.threshold,
        out_dir   = args.out_dir,
    )