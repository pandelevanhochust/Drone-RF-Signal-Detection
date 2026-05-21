"""
inference.py
=============
Run the trained DronePipeline on a single spectrogram image or a folder
of images and print / save predictions.

Compatible with the refactored roi.py where DroneCLSNet returns raw
logits (no Softmax) — torch.softmax is applied here at inference time.

Usage:
    # Single image
    python inference.py --checkpoint checkpoints/best_model.pth \
                        --img path/to/spectrogram.png

    # Folder of images
    python inference.py --checkpoint checkpoints/best_model.pth \
                        --img_dir path/to/folder/

    # Save visualisation (pre-norm + mask + ROI side-by-side)
    python inference.py --checkpoint checkpoints/best_model.pth \
                        --img path/to/spectrogram.png \
                        --save_vis --out_dir ./inference_outputs
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from roi import DronePipeline


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# =============================================================================
# Model loader
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device):
    print(f"  Loading checkpoint : {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    meta = ckpt.get("meta", {})
    args = ckpt.get("args", {})

    num_classes       = meta.get("num_classes", args.get("num_classes", 8))
    img_size          = meta.get("img_size", tuple(args.get("img_size", [256, 512])))
    unet_base_filters = args.get("unet_base_filters", 32)

    model = DronePipeline(
        num_classes       = num_classes,
        in_channels       = 3,
        unet_base_filters = unet_base_filters,
        roi_output_size   = (224, 224),
        mask_threshold    = 0.5,
        roi_strategy      = "multiply",
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  Restored from epoch : {ckpt.get('epoch', '?')}")
    print(f"  Best val accuracy   : {ckpt.get('best_val_acc', 0.0):.2f}%")
    print(f"  Classes             : {num_classes}  ->  {meta.get('class_names', [])}")
    print(f"  Image size          : {img_size}")
    print(f"  UNet base filters   : {unet_base_filters}")

    meta["img_size"] = img_size
    return model, meta


# =============================================================================
# Transforms
# =============================================================================

def build_transform(img_size):
    H, W = img_size
    return transforms.Compose([
        transforms.Resize(
            (H, W),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_pretensor_transform(img_size):
    H, W = img_size
    return transforms.Compose([
        transforms.Resize(
            (H, W),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
    ])


# =============================================================================
# Single-image inference
# =============================================================================

@torch.no_grad()
def infer_single(model, img_path, transform, device, class_names, top_k=3):
    """
    Run inference on one spectrogram PNG.

    Model returns raw logits — torch.softmax applied here for probabilities.

    Returns dict with: img_path, pred_class, pred_idx, confidence,
                       top_k_classes, probs, logits, mask, latency_ms
    """
    pil_img = Image.open(img_path).convert("RGB")
    tensor  = transform(pil_img).unsqueeze(0).to(device)   # (1, 3, H, W)

    t0 = time.perf_counter()

    # model returns raw logits (Softmax removed from DroneCLSNet)
    logits, mask = model(tensor, return_mask=True)          # (1,C), (1,1,H,W)

    # convert logits -> probabilities here at inference time
    probs = torch.softmax(logits, dim=1)                    # (1, num_classes)

    latency_ms = (time.perf_counter() - t0) * 1000

    probs_np  = probs[0].cpu().numpy()
    logits_np = logits[0].cpu().numpy()
    mask_np   = mask[0].cpu().numpy()                       # (1, H, W)

    pred_idx   = int(probs_np.argmax())
    confidence = float(probs_np[pred_idx])
    pred_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    top_k_indices = probs_np.argsort()[::-1][:top_k]
    top_k_classes = [
        (class_names[i] if i < len(class_names) else str(i), float(probs_np[i]))
        for i in top_k_indices
    ]

    return {
        "img_path"     : str(img_path),
        "pred_class"   : pred_class,
        "pred_idx"     : pred_idx,
        "confidence"   : confidence,
        "top_k_classes": top_k_classes,
        "probs"        : probs_np,
        "logits"       : logits_np,
        "mask"         : mask_np,
        "latency_ms"   : latency_ms,
    }


# =============================================================================
# Print result
# =============================================================================

def print_result(result, top_k=3):
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  File       : {Path(result['img_path']).name}")
    print(f"  Prediction : {result['pred_class']}")
    print(f"  Confidence : {result['confidence']*100:.2f}%")
    print(f"  Latency    : {result['latency_ms']:.2f} ms")
    print(f"\n  Top-{top_k} predictions:")
    for rank, (cls, prob) in enumerate(result["top_k_classes"], 1):
        bar    = "#" * int(prob * 30)
        marker = " <--" if rank == 1 else ""
        print(f"    {rank}. {cls:<12}  {prob*100:5.2f}%  {bar}{marker}")
    print(sep)


# =============================================================================
# Visualisation
# =============================================================================

def save_visualisation(result, img_size, out_dir, pre_transform):
    """
    Save a 4-panel PNG:
        Panel 1 - Original spectrogram (pre-norm)
        Panel 2 - U-Net ROI mask (grayscale)
        Panel 3 - Masked ROI (mask x original)
        Panel 4 - Top-K probability bar chart
    Falls back to 3 separate PIL PNGs if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result["img_path"]).stem

    pil_img  = Image.open(result["img_path"]).convert("RGB")
    pre_norm = pre_transform(pil_img)
    mask_t   = torch.from_numpy(result["mask"])
    roi      = (pre_norm * mask_t).clamp(0, 1)

    def t2np(t):
        return t.permute(1, 2, 0).numpy()

    if HAS_MPL:
        fig = plt.figure(figsize=(20, 4))
        gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.08)

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(t2np(pre_norm))
        ax0.set_title("Original (pre-norm)", fontsize=10, fontweight="bold")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(result["mask"][0], cmap="gray", vmin=0, vmax=1)
        ax1.set_title("U-Net ROI Mask", fontsize=10, fontweight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(t2np(roi))
        ax2.set_title("Masked ROI", fontsize=10, fontweight="bold")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[3])
        names  = [c for c, _ in result["top_k_classes"]]
        vals   = [p for _, p in result["top_k_classes"]]
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(names))]
        bars   = ax3.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Probability", fontsize=9)
        ax3.set_title("Top-K Predictions", fontsize=10, fontweight="bold")
        for bar, val in zip(bars, vals[::-1]):
            ax3.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=8,
            )
        ax3.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            f"Pred: {result['pred_class']}  "
            f"({result['confidence']*100:.1f}% confidence)  "
            f"| {result['latency_ms']:.1f} ms",
            fontsize=11, fontweight="bold", y=1.02,
        )

        out_path = out_dir / f"{stem}__inference_vis.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Visualisation saved -> {out_path}")

    else:
        def arr_to_pil(t):
            return Image.fromarray(
                (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
        arr_to_pil(pre_norm).save(out_dir / f"{stem}__pre_norm.png")
        mask_disp = (result["mask"][0] * 255).astype(np.uint8)
        Image.fromarray(mask_disp, mode="L").save(out_dir / f"{stem}__mask.png")
        arr_to_pil(roi).save(out_dir / f"{stem}__roi_masked.png")
        print(f"  matplotlib not found -- saved 3 separate PNGs to {out_dir}")


# =============================================================================
# CLI
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="DronePipeline inference -- single image or folder."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--img",     help="Path to a single spectrogram PNG.")
    src.add_argument("--img_dir", help="Path to a folder of spectrogram PNGs.")

    p.add_argument("--checkpoint", required=True,
                   help="Path to best_model.pth saved by main.py.")
    p.add_argument("--img_size", nargs=2, type=int, default=None,
                   metavar=("H", "W"),
                   help="Override image size from checkpoint.")
    p.add_argument("--top_k", type=int, default=3,
                   help="Number of top predictions to display (default 3).")
    p.add_argument("--save_vis", action="store_true",
                   help="Save 4-panel visualisation PNG (requires matplotlib).")
    p.add_argument("--out_dir", default="inference_outputs",
                   help="Output directory for visualisations.")
    p.add_argument("--device", default=None,
                   help="Force device: 'cpu' or 'cuda'. Auto-detected if omitted.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sep = "=" * 55
    print(f"\n{sep}")
    print("  DronePipeline -- Inference")
    print(f"{sep}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")

    model, meta = load_model(args.checkpoint, device)

    class_names = meta.get("class_names", [])
    img_size    = tuple(args.img_size) if args.img_size else \
                  meta.get("img_size", (256, 512))

    if not class_names:
        print("  WARNING: No class names in checkpoint. Using indices.")
        class_names = [str(i) for i in range(meta.get("num_classes", 8))]

    transform     = build_transform(img_size)
    pre_transform = build_pretensor_transform(img_size)

    # collect image paths
    if args.img:
        img_paths = [Path(args.img)]
    else:
        img_paths = sorted([
            p for p in Path(args.img_dir).iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS
        ])
        if not img_paths:
            raise FileNotFoundError(f"No images found in: {args.img_dir}")
        print(f"  Found {len(img_paths)} images in {args.img_dir}")

    print(f"{sep}\n")

    all_results = []
    for img_path in img_paths:
        result = infer_single(
            model       = model,
            img_path    = str(img_path),
            transform   = transform,
            device      = device,
            class_names = class_names,
            top_k       = args.top_k,
        )
        print_result(result, top_k=args.top_k)
        all_results.append(result)

        if args.save_vis:
            save_visualisation(result, img_size, args.out_dir, pre_transform)

    # folder-mode summary
    if len(all_results) > 1:
        from collections import Counter
        print(f"\n{sep}")
        print(f"  Summary -- {len(all_results)} images")
        print(sep)
        print(f"  Avg confidence : {np.mean([r['confidence'] for r in all_results])*100:.2f}%")
        print(f"  Avg latency    : {np.mean([r['latency_ms'] for r in all_results]):.2f} ms")
        print(f"\n  Prediction distribution:")
        for cls, cnt in Counter(r["pred_class"] for r in all_results).most_common():
            bar = "#" * cnt
            print(f"    {cls:<12}  {cnt:4d}  {bar}")
        print(sep)

        if args.save_vis:
            out_path = Path(args.out_dir) / "inference_results.npy"
            np.save(out_path, [
                {k: v for k, v in r.items() if k != "mask"}
                for r in all_results
            ])
            print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()

"""
inference.py
=============
Run the trained DronePipeline on a single spectrogram image or a folder
of images and print / save predictions.

Compatible with the refactored roi.py where DroneCLSNet returns raw
logits (no Softmax) — torch.softmax is applied here at inference time.

Usage:
    # Single image
    python inference.py --checkpoint checkpoints/best_model.pth \
                        --img path/to/spectrogram.png

    # Folder of images
    python inference.py --checkpoint checkpoints/best_model.pth \
                        --img_dir path/to/folder/

    # Save visualisation (pre-norm + mask + ROI side-by-side)
    python inference.py --checkpoint checkpoints/best_model.pth \
                        --img path/to/spectrogram.png \
                        --save_vis --out_dir ./inference_outputs
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from roi import DronePipeline


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# =============================================================================
# Model loader
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device):
    print(f"  Loading checkpoint : {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    meta = ckpt.get("meta", {})
    args = ckpt.get("args", {})

    num_classes       = meta.get("num_classes", args.get("num_classes", 8))
    img_size          = meta.get("img_size", tuple(args.get("img_size", [256, 512])))
    unet_base_filters = args.get("unet_base_filters", 32)

    model = DronePipeline(
        num_classes       = num_classes,
        in_channels       = 3,
        unet_base_filters = unet_base_filters,
        roi_output_size   = (224, 224),
        mask_threshold    = 0.7,
        roi_strategy      = "multiply",
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  Restored from epoch : {ckpt.get('epoch', '?')}")
    print(f"  Best val accuracy   : {ckpt.get('best_val_acc', 0.0):.2f}%")
    print(f"  Classes             : {num_classes}  ->  {meta.get('class_names', [])}")
    print(f"  Image size          : {img_size}")
    print(f"  UNet base filters   : {unet_base_filters}")

    meta["img_size"] = img_size
    return model, meta


# =============================================================================
# Transforms
# =============================================================================

def build_transform(img_size):
    H, W = img_size
    return transforms.Compose([
        transforms.Resize(
            (H, W),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_pretensor_transform(img_size):
    H, W = img_size
    return transforms.Compose([
        transforms.Resize(
            (H, W),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
    ])


# =============================================================================
# Single-image inference
# =============================================================================

@torch.no_grad()
def infer_single(model, img_path, transform, device, class_names, top_k=3):
    """
    Run inference on one spectrogram PNG.

    Model returns raw logits — torch.softmax applied here for probabilities.

    Returns dict with: img_path, pred_class, pred_idx, confidence,
                       top_k_classes, probs, logits, mask, latency_ms
    """
    pil_img = Image.open(img_path).convert("RGB")
    tensor  = transform(pil_img).unsqueeze(0).to(device)   # (1, 3, H, W)

    t0 = time.perf_counter()

    # model returns raw logits (Softmax removed from DroneCLSNet)
    logits, mask = model(tensor, return_mask=True)          # (1,C), (1,1,H,W)

    # convert logits -> probabilities here at inference time
    probs = torch.softmax(logits, dim=1)                    # (1, num_classes)

    latency_ms = (time.perf_counter() - t0) * 1000

    probs_np  = probs[0].cpu().numpy()
    logits_np = logits[0].cpu().numpy()
    mask_np   = mask[0].cpu().numpy()                       # (1, H, W)

    pred_idx   = int(probs_np.argmax())
    confidence = float(probs_np[pred_idx])
    pred_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    top_k_indices = probs_np.argsort()[::-1][:top_k]
    top_k_classes = [
        (class_names[i] if i < len(class_names) else str(i), float(probs_np[i]))
        for i in top_k_indices
    ]

    return {
        "img_path"     : str(img_path),
        "pred_class"   : pred_class,
        "pred_idx"     : pred_idx,
        "confidence"   : confidence,
        "top_k_classes": top_k_classes,
        "probs"        : probs_np,
        "logits"       : logits_np,
        "mask"         : mask_np,
        "latency_ms"   : latency_ms,
    }


# =============================================================================
# Print result
# =============================================================================

def print_result(result, top_k=3):
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  File       : {Path(result['img_path']).name}")
    print(f"  Prediction : {result['pred_class']}")
    print(f"  Confidence : {result['confidence']*100:.2f}%")
    print(f"  Latency    : {result['latency_ms']:.2f} ms")
    print(f"\n  Top-{top_k} predictions:")
    for rank, (cls, prob) in enumerate(result["top_k_classes"], 1):
        bar    = "#" * int(prob * 30)
        marker = " <--" if rank == 1 else ""
        print(f"    {rank}. {cls:<12}  {prob*100:5.2f}%  {bar}{marker}")
    print(sep)


# =============================================================================
# Visualisation
# =============================================================================

def save_visualisation(result, img_size, out_dir, pre_transform):
    """
    Save a 4-panel PNG:
        Panel 1 - Original spectrogram (pre-norm)
        Panel 2 - U-Net ROI mask (grayscale)
        Panel 3 - Masked ROI (mask x original)
        Panel 4 - Top-K probability bar chart
    Falls back to 3 separate PIL PNGs if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result["img_path"]).stem

    pil_img  = Image.open(result["img_path"]).convert("RGB")
    pre_norm = pre_transform(pil_img)
    mask_t   = torch.from_numpy(result["mask"])
    roi      = (pre_norm * mask_t).clamp(0, 1)

    def t2np(t):
        return t.permute(1, 2, 0).numpy()

    if HAS_MPL:
        fig = plt.figure(figsize=(20, 4))
        gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.08)

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(t2np(pre_norm))
        ax0.set_title("Original (pre-norm)", fontsize=10, fontweight="bold")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(result["mask"][0], cmap="gray", vmin=0, vmax=1)
        ax1.set_title("U-Net ROI Mask", fontsize=10, fontweight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(t2np(roi))
        ax2.set_title("Masked ROI", fontsize=10, fontweight="bold")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[3])
        names  = [c for c, _ in result["top_k_classes"]]
        vals   = [p for _, p in result["top_k_classes"]]
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(names))]
        bars   = ax3.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Probability", fontsize=9)
        ax3.set_title("Top-K Predictions", fontsize=10, fontweight="bold")
        for bar, val in zip(bars, vals[::-1]):
            ax3.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=8,
            )
        ax3.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            f"Pred: {result['pred_class']}  "
            f"({result['confidence']*100:.1f}% confidence)  "
            f"| {result['latency_ms']:.1f} ms",
            fontsize=11, fontweight="bold", y=1.02,
        )

        out_path = out_dir / f"{stem}__inference_vis.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Visualisation saved -> {out_path}")

    else:
        def arr_to_pil(t):
            return Image.fromarray(
                (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
        arr_to_pil(pre_norm).save(out_dir / f"{stem}__pre_norm.png")
        mask_disp = (result["mask"][0] * 255).astype(np.uint8)
        Image.fromarray(mask_disp, mode="L").save(out_dir / f"{stem}__mask.png")
        arr_to_pil(roi).save(out_dir / f"{stem}__roi_masked.png")
        print(f"  matplotlib not found -- saved 3 separate PNGs to {out_dir}")


# =============================================================================
# CLI
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="DronePipeline inference -- single image or folder."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--img",     help="Path to a single spectrogram PNG.")
    src.add_argument("--img_dir", help="Path to a folder of spectrogram PNGs.")

    p.add_argument("--checkpoint", default="best_model.pth", required=True,
                   help="Path to best_model.pth saved by main.py.")
    p.add_argument("--img_size", nargs=2, type=int, default=None,
                   metavar=("H", "W"),
                   help="Override image size from checkpoint.")
    p.add_argument("--top_k", type=int, default=3,
                   help="Number of top predictions to display (default 3).")
    p.add_argument("--save_vis", action="store_true",
                   help="Save 4-panel visualisation PNG (requires matplotlib).")
    p.add_argument("--out_dir", default="inference_outputs",
                   help="Output directory for visualisations.")
    p.add_argument("--device", default=None,
                   help="Force device: 'cpu' or 'cuda'. Auto-detected if omitted.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sep = "=" * 55
    print(f"\n{sep}")
    print("  DronePipeline -- Inference")
    print(f"{sep}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")

    model, meta = load_model(args.checkpoint, device)

    class_names = meta.get("class_names", [])
    img_size    = tuple(args.img_size) if args.img_size else \
                  meta.get("img_size", (256, 512))

    if not class_names:
        print("  WARNING: No class names in checkpoint. Using indices.")
        class_names = [str(i) for i in range(meta.get("num_classes", 8))]

    transform     = build_transform(img_size)
    pre_transform = build_pretensor_transform(img_size)

    # collect image paths
    if args.img:
        img_paths = [Path(args.img)]
    else:
        img_paths = sorted([
            p for p in Path(args.img_dir).iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS
        ])
        if not img_paths:
            raise FileNotFoundError(f"No images found in: {args.img_dir}")
        print(f"  Found {len(img_paths)} images in {args.img_dir}")

    print(f"{sep}\n")

    all_results = []
    for img_path in img_paths:
        result = infer_single(
            model       = model,
            img_path    = str(img_path),
            transform   = transform,
            device      = device,
            class_names = class_names,
            top_k       = args.top_k,
        )
        print_result(result, top_k=args.top_k)
        all_results.append(result)

        if args.save_vis:
            save_visualisation(result, img_size, args.out_dir, pre_transform)

    # folder-mode summary
    if len(all_results) > 1:
        from collections import Counter
        print(f"\n{sep}")
        print(f"  Summary -- {len(all_results)} images")
        print(sep)
        print(f"  Avg confidence : {np.mean([r['confidence'] for r in all_results])*100:.2f}%")
        print(f"  Avg latency    : {np.mean([r['latency_ms'] for r in all_results]):.2f} ms")
        print(f"\n  Prediction distribution:")
        for cls, cnt in Counter(r["pred_class"] for r in all_results).most_common():
            bar = "#" * cnt
            print(f"    {cls:<12}  {cnt:4d}  {bar}")
        print(sep)

        if args.save_vis:
            out_path = Path(args.out_dir) / "inference_results.npy"
            np.save(out_path, [
                {k: v for k, v in r.items() if k != "mask"}
                for r in all_results
            ])
            print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()



# # Single image
# python inference.py --checkpoint best_model.pth \
#                     --img INS_1110_01__seg01_start1498ms.png
#
# # Single image + save 4-panel visualisation
# python inference.py --checkpoint checkpoints/best_model.pth \
#                     --img INS_1110_01__seg01_start1498ms.png \
#                     --save_vis --out_dir ./outputs
#
# # Entire folder
# python inference.py --checkpoint checkpoints/best_model.pth \
#                     --img_dir output_spectrograms/BOTH/INS/