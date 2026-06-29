"""
inspect_single_spectrogram.py
==============================
Takes ONE spectrogram PNG and saves three output PNGs:

    1. {stem}__pre_norm.png   -- after resize + ToTensor, before ImageNet
                                 normalisation. Values in [0, 1].
    2. {stem}__unet_mask.png  -- real U-Net binary mask from the trained
                                 DronePipeline (not the proxy energy mask).
    3. {stem}__roi_masked.png -- mask x pre-norm tensor, exactly what
                                 ROIExtractor feeds to the classifier.

Usage:
    python inspect_single_spectrogram.py \
        --img        path/to/spectrogram.png \
        --checkpoint checkpoints/best_model.pth

    python inspect_single_spectrogram.py \
        --img        path/to/spectrogram.png \
        --checkpoint checkpoints/best_model.pth \
        --img_size   256 512 \
        --threshold  0.7 \
        --out_dir    ./outputs
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from roi import DronePipeline


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =============================================================================
# Image / tensor utilities
# =============================================================================

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(C, H, W) float32 [0,1] -> PIL RGB image."""
    arr = (t.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    """(1, H, W) or (H, W) binary float -> PIL grayscale. White=ON, Black=OFF."""
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    arr = (mask.numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


# =============================================================================
# Model loader
# =============================================================================

def load_pipeline(
    checkpoint_path: str,
    device: torch.device,
    threshold: float,
) -> DronePipeline:
    """
    Restore DronePipeline from a checkpoint saved by main.py.

    Args:
        checkpoint_path : path to best_model.pth
        device          : torch device
        threshold       : U-Net sigmoid binarisation threshold

    Returns:
        DronePipeline in eval mode on device
    """
    print(f"  Loading checkpoint : {checkpoint_path}")
    ckpt       = torch.load(checkpoint_path, map_location=device)
    meta       = ckpt.get("meta", {})
    train_args = ckpt.get("args", {})

    model = DronePipeline(
        num_classes       = meta.get("num_classes", train_args.get("num_classes", 8)),
        in_channels       = 3,
        unet_base_filters = train_args.get("unet_base_filters", 32),
        roi_output_size   = (224, 224),
        mask_threshold    = threshold,
        roi_strategy      = "multiply",
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  Restored from epoch : {ckpt.get('epoch', '?')}")
    print(f"  Best val accuracy   : {ckpt.get('best_val_acc', 0.0):.2f}%")
    print(f"  Classes             : {meta.get('class_names', [])}")
    return model


# =============================================================================
# Core pipeline steps
# =============================================================================

def load_and_resize(
    img_path: str,
    img_size: tuple,
) -> torch.Tensor:
    """
    Open PNG as RGB, resize to img_size, convert to float32 tensor [0, 1].

    Returns:
        pre_norm : (3, H, W) float32 tensor
    """
    pil_img = Image.open(img_path).convert("RGB")
    print(f"    Original size : {pil_img.size}  (W x H)")

    transform = transforms.Compose([
        transforms.Resize(
            img_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),   # [0,255] uint8 -> [0,1] float32 (C,H,W)
    ])
    pre_norm = transform(pil_img)
    print(f"    Pre-norm tensor : shape={tuple(pre_norm.shape)}"
          f"  min={pre_norm.min():.4f}  max={pre_norm.max():.4f}")
    return pre_norm


def compute_unet_mask(
    pre_norm  : torch.Tensor,
    model     : DronePipeline,
    device    : torch.device,
    threshold : float,
) -> torch.Tensor:
    """
    ImageNet-normalise pre_norm, run the U-Net, return a binary mask.

    Matches exactly what happens inside DronePipeline during training
    and inference — the model never sees un-normalised tensors.

    Args:
        pre_norm  : (3, H, W) float32 in [0, 1] — NOT yet normalised
        model     : loaded DronePipeline in eval mode
        device    : torch device
        threshold : sigmoid binarisation cutoff

    Returns:
        binary_mask : (1, H, W) float32  (1 = kept, 0 = zeroed)
    """
    norm   = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    normed = norm(pre_norm).unsqueeze(0).to(device)   # (1, 3, H, W)

    with torch.no_grad():
        raw_mask = model.unet(normed)                 # (1, 1, H, W) sigmoid
        binary   = (raw_mask >= threshold).float()    # (1, 1, H, W) binary

    return binary.squeeze(0).cpu()                    # (1, H, W)


def apply_roi_mask(
    pre_norm  : torch.Tensor,
    mask      : torch.Tensor,
) -> torch.Tensor:
    """
    Multiply pre_norm by the binary mask (ROIExtractor multiply strategy).

    Args:
        pre_norm : (3, H, W) float32 in [0, 1]
        mask     : (1, H, W) binary float32

    Returns:
        roi : (3, H, W) — background pixels are exactly 0
    """
    return pre_norm * mask   # mask broadcasts across C channels


# =============================================================================
# Main inspection function
# =============================================================================

def inspect(
    img_path        : str,
    checkpoint_path : str,
    img_size        : tuple  = (256, 512),
    threshold       : float  = 0.7,
    out_dir         : str    = ".",
) -> dict:
    """
    Full inspection pipeline for a single spectrogram PNG.

    Steps
    -----
    0. Load DronePipeline from checkpoint
    1. Load PNG -> resize -> ToTensor  (pre_norm)
    2. ImageNet-normalise -> U-Net forward -> binary mask
    3. mask x pre_norm -> ROI tensor
    4. Save all three as PNGs and print a stats table

    Args:
        img_path        : input spectrogram PNG
        checkpoint_path : best_model.pth from main.py
        img_size        : (H, W) — both must be divisible by 16
        threshold       : U-Net sigmoid binarisation threshold
        out_dir         : directory for output PNGs

    Returns:
        dict with keys "pre_norm", "unet_mask", "roi_masked"
    """
    H, W = img_size
    assert H % 16 == 0 and W % 16 == 0, (
        f"img_size ({H}, {W}) — both values must be divisible by 16 "
        f"(U-Net has 4x MaxPool2d stride-2 layers)."
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # ── Step 0: Load model ────────────────────────────────────────────────────
    print(f"\n[0] Loading model ...")
    model = load_pipeline(checkpoint_path, device, threshold)

    # ── Step 1: Load + resize image ───────────────────────────────────────────
    print(f"\n[1] Loading image: {img_path}")
    pre_norm = load_and_resize(img_path, img_size)    # (3, H, W)

    # ── Step 2: U-Net mask ────────────────────────────────────────────────────
    print(f"\n[2] Running U-Net  (threshold={threshold}) ...")
    unet_mask = compute_unet_mask(pre_norm, model, device, threshold)
    kept_pct  = unet_mask.mean().item() * 100
    print(f"    Mask shape  : {tuple(unet_mask.shape)}")
    print(f"    Pixels kept : {kept_pct:.1f}%"
          f"  ({unet_mask.sum().int().item()} / {H * W})")

    # ── Step 3: ROI masked tensor ─────────────────────────────────────────────
    print(f"\n[3] Applying ROI mask (multiply strategy) ...")
    roi_masked  = apply_roi_mask(pre_norm, unet_mask)
    nonzero_pct = (roi_masked.sum(dim=0) > 0).float().mean().item() * 100
    print(f"    ROI tensor shape     : {tuple(roi_masked.shape)}")
    print(f"    Non-zero spatial pos : {nonzero_pct:.1f}%")

    # ── Step 4: Save outputs ──────────────────────────────────────────────────
    stem          = Path(img_path).stem
    pre_norm_path = out_dir / f"{stem}__pre_norm.png"
    mask_path     = out_dir / f"{stem}__unet_mask.png"
    roi_path      = out_dir / f"{stem}__roi_masked.png"

    tensor_to_pil(pre_norm).save(pre_norm_path)
    mask_to_pil(unet_mask).save(mask_path)
    tensor_to_pil(roi_masked).save(roi_path)

    print(f"\n[OK] Saved outputs -> {out_dir.resolve()}")
    print(f"     pre_norm   -> {pre_norm_path.name}")
    print(f"     unet_mask  -> {mask_path.name}")
    print(f"     roi_masked -> {roi_path.name}")

    # ── Step 5: Stats table ───────────────────────────────────────────────────
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  {'Tensor':<24}  {'Shape':<18}  {'Min':>7}  {'Max':>7}  {'Mean':>7}")
    print(sep)
    for name, t in [
        ("pre_norm",           pre_norm),
        ("unet_mask (binary)", unet_mask),
        ("roi_masked",         roi_masked),
    ]:
        print(
            f"  {name:<24}  {str(tuple(t.shape)):<18}"
            f"  {t.min().item():>7.4f}"
            f"  {t.max().item():>7.4f}"
            f"  {t.mean().item():>7.4f}"
        )
    print(sep)

    # ImageNet-normalised stats for reference
    normed = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(pre_norm)
    print(f"\n  (After ImageNet normalisation -- what the model actually receives:)")
    print(
        f"  {'normed_tensor':<24}  {str(tuple(normed.shape)):<18}"
        f"  {normed.min().item():>7.4f}"
        f"  {normed.max().item():>7.4f}"
        f"  {normed.mean().item():>7.4f}"
    )

    return {
        "pre_norm"  : pre_norm,
        "unet_mask" : unet_mask,
        "roi_masked": roi_masked,
    }


# =============================================================================
# CLI
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="Inspect a single spectrogram through the drone pipeline."
    )
    p.add_argument(
        "--img", required=True,
        help="Path to the input spectrogram PNG.",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to best_model.pth saved by main.py.",
    )
    p.add_argument(
        "--img_size", nargs=2, type=int, default=[256, 512],
        metavar=("H", "W"),
        help="Resize target. Both must be divisible by 16. Default: 256 512",
    )
    p.add_argument(
        "--threshold", type=float, default=0.7,
        help="U-Net sigmoid binarisation threshold (default 0.7).",
    )
    p.add_argument(
        "--out_dir", default=".",
        help="Output directory for saved PNGs. Default: current directory.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    inspect(
        img_path        = args.img,
        checkpoint_path = args.checkpoint,
        img_size        = tuple(args.img_size),
        threshold       = args.threshold,
        out_dir         = args.out_dir,
    )