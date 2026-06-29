"""
roi_extraction_debug.py
=======================
Takes your existing step4_rgb.png spectrogram images, loads the trained
U-Net from the Stage-2 checkpoint, runs the ROI extraction pipeline, and
saves debug images at every intermediate step.

This lets you verify ROI extraction independently — no BladeRF needed.

What this script replicates from FusedDronePipeline.forward()
-------------------------------------------------------------
    spectrogram (1, 3, 256, 512)
        │
        ▼  DroneROIUNet
    pred_mask   (1, 1, 256, 512)  sigmoid, values in [0, 1]
        │
        ▼  torch.ge(pred_mask, 0.5).float()
    binary_mask (1, 1, 256, 512)  hard {0, 1}
        │
        ▼  spectrogram × binary_mask
    roi         (1, 3, 256, 512)  background energy zeroed
        │
        ▼  F.interpolate(bilinear, align_corners=False)
    roi_patch   (1, 3, 224, 224)  classifier input

Output per image
----------------
    debug_roi/<stem>/
        00_input.png          the step4 RGB image you gave as input
        01_pred_mask.png      raw U-Net sigmoid output (greyscale, [0,1])
        02_binary_mask.png    hard threshold at 0.5 (pure black/white)
        03_roi_masked.png     spectrogram × binary_mask (background zeroed)
        04_roi_patch.png      resized to (224, 224) — classifier input
        05_overlay.png        input with mask overlaid in red

Usage
-----
    # Single image
    python3 roi_extraction_debug.py \\
        --ckpt checkpoints/classifier_best.pt \\
        --image debug_output/seg_000/step4_rgb.png

    # All step4 images in a debug_output tree
    python3 roi_extraction_debug.py \\
        --ckpt checkpoints/classifier_best.pt \\
        --folder debug_output/

    # Also run classification on each roi_patch
    python3 roi_extraction_debug.py \\
        --ckpt checkpoints/classifier_best.pt \\
        --folder debug_output/ \\
        --classify

Requirements
------------
    pip install torch torchvision pillow numpy
"""

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import sys
import os

# 1. Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up two levels to reach the 'SpectrumAnalyzer' root folder
# (From device/pipeline -> device -> SpectrumAnalyzer)
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

# 3. Add the root directory to Python's system path
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants — must match training pipeline
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SPEC_H, SPEC_W  = 256, 512     # U-Net input size
ROI_H,  ROI_W   = 224, 224     # Classifier input size
MASK_THRESHOLD  = 0.7


# ─────────────────────────────────────────────────────────────────────────────
#  Load model from Stage-2 checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_unet_from_checkpoint(ckpt_path: str, device: torch.device):
    """
    Load DroneROIUNet from the Stage-2 checkpoint.

    The Stage-2 checkpoint (classifier_best.pt) contains both the U-Net
    and classifier weights under keys:
        ckpt["unet_config"]  — architecture params
        ckpt["unet_state"]   — U-Net state_dict (post fine-tune)
    """
    from dat_data.EfficientNet_B0.RoiExtractor import DroneROIUNet

    print(f"[Load] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    unet_config = ckpt["unet_config"]
    unet = DroneROIUNet(**unet_config).to(device)
    unet.load_state_dict(ckpt["unet_state"])
    unet.eval()

    epoch   = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_acc", "?")
    print(f"  ✓ U-Net loaded  (epoch={epoch}  val_acc={val_acc})")
    print(f"  ✓ Params: {sum(p.numel() for p in unet.parameters()):,}\n")
    return unet


def load_classifier_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load DroneCLSNet from the same Stage-2 checkpoint."""
    from dat_data.EfficientNet_B0.EfficientNetB0_Classification import DroneCLSNet

    ckpt    = torch.load(ckpt_path, map_location=device)
    cls_net = DroneCLSNet(num_classes=ckpt["num_classes"]).to(device)
    cls_net.load_state_dict(ckpt["cls_state"])
    cls_net.eval()
    return cls_net, ckpt.get("class_names", [])


# ─────────────────────────────────────────────────────────────────────────────
#  Image I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_spectrogram_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """
    Load a step4_rgb.png and convert to an ImageNet-normalised
    (1, 3, 256, 512) float32 tensor — the same format as iq_to_spectrogram().

    step4_rgb.png is already the viridis-coloured RGB image BEFORE normalisation.
    We apply normalisation here to reproduce the model input exactly.
    """
    img = Image.open(image_path).convert("RGB").resize(
        (SPEC_W, SPEC_H), Image.BILINEAR
    )
    arr = np.array(img, dtype=np.float32) / 255.0       # HWC [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD           # HWC normalised
    arr = arr.transpose(2, 0, 1)                         # CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)          # (1, 3, H, W)
    return tensor.to(device)


def tensor_to_rgb_array(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, 3, H, W) ImageNet-normalised tensor back to uint8 HWC RGB.
    Used for saving debug images.
    """
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = arr * IMAGENET_STD + IMAGENET_MEAN            # undo normalise
    return (arr * 255).clip(0, 255).astype(np.uint8)


def mask_to_array(mask: torch.Tensor) -> np.ndarray:
    """Convert (1, 1, H, W) sigmoid mask → (H, W) uint8 [0..255] greyscale."""
    m = mask.squeeze().cpu().numpy()                    # (H, W) float [0, 1]
    return (m * 255).clip(0, 255).astype(np.uint8)


def save_overlay(input_rgb: np.ndarray, binary_mask: np.ndarray, path: str):
    """
    Save the input spectrogram with the binary mask overlaid in semi-transparent red.
    Red = U-Net said 'drone signal here', black = background.

    input_rgb    : (H, W, 3) uint8
    binary_mask  : (H, W) uint8, values 0 or 255
    """
    overlay = input_rgb.copy()
    # Where mask is 1, tint red
    mask_bool = binary_mask > 127
    overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0].astype(int) + 80, 0, 255)
    overlay[mask_bool, 1] = (overlay[mask_bool, 1] * 0.5).astype(np.uint8)
    overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(path)


# ─────────────────────────────────────────────────────────────────────────────
#  Core ROI extraction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def extract_roi(
    image_path  : str,
    unet        : torch.nn.Module,
    out_dir     : str,
    cls_net     = None,
    class_names : list = None,
    device      : torch.device = None,
) -> dict:
    """
    Run the full ROI extraction pipeline on one step4_rgb.png image.

    Parameters
    ----------
    image_path  : path to a step4_rgb.png (or any 3-channel spectrogram PNG)
    unet        : loaded DroneROIUNet in eval mode
    out_dir     : output directory for debug images
    cls_net     : optional DroneCLSNet for classification after ROI extraction
    class_names : list of class name strings

    Returns
    -------
    dict with keys: pred_mask_min, pred_mask_max, mask_coverage,
                    predicted_class (if cls_net provided), confidence
    """
    stem = Path(image_path).stem
    save_dir = os.path.join(out_dir, stem)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[ROI]  {Path(image_path).name}")

    # ── Step 0: load input ────────────────────────────────────────────────────
    spec_tensor = load_spectrogram_tensor(image_path, device)
    input_rgb   = tensor_to_rgb_array(spec_tensor)

    p0 = os.path.join(save_dir, "00_input.png")
    Image.fromarray(input_rgb).save(p0)
    print(f"  00_input.png       shape={spec_tensor.shape}  → {p0}")

    # ── Step 1: U-Net forward — predict sigmoid mask ──────────────────────────
    with torch.no_grad():
        pred_mask = unet(spec_tensor)               # (1, 1, 256, 512) [0, 1]

    mask_arr = mask_to_array(pred_mask)             # (256, 512) uint8
    p1 = os.path.join(save_dir, "01_pred_mask.png")
    Image.fromarray(mask_arr, mode="L").save(p1)
    print(f"  01_pred_mask.png   range=[{pred_mask.min():.3f}, {pred_mask.max():.3f}]  → {p1}")

    # ── Step 2: binarise at threshold ────────────────────────────────────────
    binary_mask   = (pred_mask >= MASK_THRESHOLD).float()  # (1, 1, 256, 512)
    binary_arr    = mask_to_array(binary_mask)             # 0 or 255
    coverage      = float(binary_mask.mean()) * 100        # % of pixels = 1

    p2 = os.path.join(save_dir, "02_binary_mask.png")
    Image.fromarray(binary_arr, mode="L").save(p2)
    print(f"  02_binary_mask.png coverage={coverage:.1f}%  → {p2}")

    # ── Step 3: spectrogram × binary_mask (background → 0) ───────────────────
    roi       = spec_tensor * binary_mask            # (1, 3, 256, 512)
    roi_rgb   = tensor_to_rgb_array(roi)

    p3 = os.path.join(save_dir, "03_roi_masked.png")
    Image.fromarray(roi_rgb).save(p3)
    print(f"  03_roi_masked.png  background energy zeroed  → {p3}")

    # ── Step 4: bilinear resize to (224, 224) — classifier input ─────────────
    roi_patch = F.interpolate(
        roi,
        size          = (ROI_H, ROI_W),
        mode          = "bilinear",
        align_corners = False,
    )                                               # (1, 3, 224, 224)
    roi_patch_rgb = tensor_to_rgb_array(roi_patch)

    p4 = os.path.join(save_dir, "04_roi_patch.png")
    Image.fromarray(roi_patch_rgb).save(p4)
    print(f"  04_roi_patch.png   shape={roi_patch.shape}  → {p4}")

    # ── Step 5: overlay — mask on input ──────────────────────────────────────
    p5 = os.path.join(save_dir, "05_overlay.png")
    save_overlay(input_rgb, binary_arr, p5)
    print(f"  05_overlay.png     red = drone signal region  → {p5}")

    result = {
        "pred_mask_min" : float(pred_mask.min()),
        "pred_mask_max" : float(pred_mask.max()),
        "mask_coverage" : coverage,
    }

    # ── Optional: classify the roi_patch ─────────────────────────────────────
    if cls_net is not None and class_names:
        with torch.no_grad():
            logits = cls_net(roi_patch)             # (1, 8)
        logits_np = logits.cpu().numpy()[0]
        logits_np -= logits_np.max()
        exp   = np.exp(np.clip(logits_np, -500, 500))
        probs = exp / exp.sum()
        pred_idx  = int(np.argmax(probs))
        pred_cls  = class_names[pred_idx]
        confidence= float(probs[pred_idx])

        result["predicted_class"] = pred_cls
        result["confidence"]      = confidence
        result["probs"]           = probs

        print(f"\n  ▶  {pred_cls}  ({confidence*100:.1f}%)")
        print(f"  {'Class':<12} {'Prob':>7}")
        print(f"  {'─'*20}")
        for name, p in zip(class_names, probs):
            marker = " ◀" if name == pred_cls else ""
            print(f"  {name:<12} {p*100:>6.2f}%{marker}")

    print()
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="ROI extraction debug — takes step4 PNGs, saves U-Net mask + ROI images"
    )
    p.add_argument("--ckpt",     required=True,
                   help="Stage-2 checkpoint path (checkpoints/classifier_best.pt)")
    p.add_argument("--image",    default=None,
                   help="Single step4_rgb.png to process")
    p.add_argument("--folder",   default=None,
                   help="Folder to scan for step4_rgb.png files (recursive)")
    p.add_argument("--out_dir",  default="debug_roi",
                   help="Output directory for debug images (default: debug_roi/)")
    p.add_argument("--classify", action="store_true",
                   help="Also run the classifier on each roi_patch")
    p.add_argument("--cpu",      action="store_true",
                   help="Force CPU even if CUDA is available")
    return p.parse_args()


def main():
    args = get_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")

    sep = "=" * 58
    print(f"\n{sep}")
    print("  ROI extraction debug")
    print(sep)
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Output dir : {args.out_dir}")
    print(f"  Device     : {device}")
    print(f"  Classify   : {args.classify}")
    print(f"{sep}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    unet = load_unet_from_checkpoint(args.ckpt, device)
    cls_net, class_names = None, []
    if args.classify:
        cls_net, class_names = load_classifier_from_checkpoint(args.ckpt, device)
        print(f"[Load] Classifier loaded  classes={class_names}\n")

    # ── Collect input images ──────────────────────────────────────────────────
    images = []
    if args.image:
        images = [args.image]
    elif args.folder:
        # Scan recursively for step4_rgb.png — also accepts any PNG
        images = sorted(glob.glob(
            os.path.join(args.folder, "**", "step4_rgb.png"), recursive=True
        ))
        if not images:
            # Fallback: any PNG in the folder
            images = sorted(glob.glob(
                os.path.join(args.folder, "**", "*.png"), recursive=True
            ))
        print(f"[Scan] Found {len(images)} image(s) in {args.folder}\n")
    else:
        print("ERROR: provide --image <path> or --folder <dir>")
        return

    if not images:
        print("ERROR: no images found.")
        return

    # ── Process each image ────────────────────────────────────────────────────
    results = []
    for img_path in images:
        result = extract_roi(
            image_path  = img_path,
            unet        = unet,
            out_dir     = args.out_dir,
            cls_net     = cls_net,
            class_names = class_names,
            device      = device,
        )
        results.append((Path(img_path).name, result))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(sep)
    print(f"  Done — {len(results)} image(s) processed")
    print(f"  Output: {os.path.abspath(args.out_dir)}/")
    print(f"\n  Per-image output:")
    print(f"    00_input.png       original spectrogram")
    print(f"    01_pred_mask.png   U-Net sigmoid (greyscale)")
    print(f"    02_binary_mask.png hard threshold (black/white)")
    print(f"    03_roi_masked.png  spectrogram × mask")
    print(f"    04_roi_patch.png   resized to 224×224 (classifier input)")
    print(f"    05_overlay.png     mask overlaid red on input")

    if results and "mask_coverage" in results[0][1]:
        print(f"\n  Mask coverage summary:")
        for name, r in results:
            cov = r["mask_coverage"]
            cls = r.get("predicted_class", "")
            conf = r.get("confidence", 0)
            cls_str = f"  →  {cls} ({conf*100:.0f}%)" if cls else ""
            print(f"    {name:<30} coverage={cov:5.1f}%{cls_str}")

    print(sep)


if __name__ == "__main__":
    main()