"""
visualize_roi.py
================
Visualise what happens to an input spectrogram at each stage of the
DronePipeline: raw input → U-Net mask → masked ROI → resized ROI patch.

Outputs one PNG per image with 4 panels side-by-side:
    Panel 1 — Original spectrogram (after val-transform resize, before norm)
    Panel 2 — U-Net binary mask  (0/1, thresholded)
    Panel 3 — Masked ROI  (spectrogram × mask, before resize)
    Panel 4 — Final ROI patch fed to classifier  (224×224)

Usage
-----
# Single image
python visualize_roi.py --image path/to/spectrogram.png \
                        --checkpoint checkpoints/best_model.pth

# All images in a folder (saves one PNG per image)
python visualize_roi.py --folder output_spectrograms/BOTH/AIR \
                        --checkpoint checkpoints/best_model.pth \
                        --max_images 16

# Use a random sample from the dataset splits
python visualize_roi.py --from_dataset \
                        --root output_spectrograms/ \
                        --checkpoint checkpoints/best_model.pth \
                        --n_samples 8 \
                        --split test

Options
-------
--checkpoint    Path to best_model.pth  (required)
--image         Single PNG to visualise
--folder        Directory of PNGs to visualise
--from_dataset  Draw samples from the train/val/test split
--root          Dataset root (used with --from_dataset)
--split         train | val | test  (used with --from_dataset, default: test)
--n_samples     How many samples to draw (used with --from_dataset, default: 8)
--max_images    Cap on images when using --folder (default: 32)
--img_size      H W of the model input (default: 256 512)
--threshold     Mask binarisation threshold (default: 0.5)
--out_dir       Where to save visualisation PNGs (default: ./roi_vis)
--no_save       Display with plt.show() instead of saving
--cols          Number of image columns per figure row (default: 4 panels fixed)
--device        cpu | cuda  (default: auto-detect)
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless-safe; override below if --no_save
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── Local imports ─────────────────────────────────────────────────────────────
from roi import DronePipeline

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =============================================================================
# Helpers
# =============================================================================

def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation → [0, 1] float tensor."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device)
    std  = torch.tensor(IMAGENET_STD,  dtype=t.dtype, device=t.device)
    return (t * std[:, None, None] + mean[:, None, None]).clamp(0.0, 1.0)


def to_np(t: torch.Tensor) -> np.ndarray:
    """CHW float tensor → HWC uint8 numpy array."""
    return (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def load_image_tensor(
    path: str,
    img_size: tuple,
    normalize: bool = True,
) -> tuple:
    """
    Load a PNG and return:
        normed_tensor : (1, 3, H, W)  normalised, ready for model
        raw_tensor    : (3, H, W)     resized only, for display
    """
    H, W = img_size
    resize    = transforms.Resize((H, W),
                                   interpolation=transforms.InterpolationMode.BILINEAR,
                                   antialias=True)
    to_tensor = transforms.ToTensor()
    norm      = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    img = Image.open(path).convert("RGB")
    raw = to_tensor(resize(img))          # [0,1], no norm
    normed = norm(raw) if normalize else raw
    return normed.unsqueeze(0), raw       # (1,3,H,W), (3,H,W)


def load_model(checkpoint: str, img_size: tuple, device: torch.device) -> tuple:
    """Load DronePipeline from checkpoint; return (model, class_names)."""
    ckpt       = torch.load(checkpoint, map_location=device)
    meta       = ckpt.get("meta", {})
    train_args = ckpt.get("args", {})

    num_classes       = meta.get("num_classes", 8)
    unet_base_filters = train_args.get("unet_base_filters", 32)
    class_names       = meta.get("class_names", [str(i) for i in range(num_classes)])

    model = DronePipeline(
        num_classes       = num_classes,
        in_channels       = 3,
        unet_base_filters = unet_base_filters,
        roi_output_size   = (224, 224),
        mask_threshold    = train_args.get("mask_threshold", 0.5),
        roi_strategy      = train_args.get("roi_strategy", "multiply"),
        img_h             = img_size[0],
        img_w             = img_size[1],
    ).to(device)

    state_dict = ckpt["model_state"]
    new_state  = {}
    for k, v in state_dict.items():
        nk = k
        if ".se.1." in nk:
            nk = nk.replace(".se.1.", ".fc1.")
        elif ".se.3." in nk:
            nk = nk.replace(".se.3.", ".fc2.")
        if "classifier.classifier.3." in nk:
            nk = nk.replace("classifier.classifier.3.", "classifier.classifier.1.")
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  [load] Missing keys (new SNPE-compat layers): {len(missing)}")
    if unexpected:
        print(f"  [load] Unexpected keys: {unexpected}")

    model.eval()
    return model, class_names


# =============================================================================
# Core: extract intermediate tensors
# =============================================================================

@torch.no_grad()
def get_pipeline_stages(
    model     : DronePipeline,
    normed    : torch.Tensor,   # (1, 3, H, W)  normalised
    raw       : torch.Tensor,   # (3, H, W)     un-normalised, for display
    threshold : float = 0.5,
    device    : torch.device = torch.device("cpu"),
) -> dict:
    """
    Run the pipeline and return intermediate tensors for display.

    Returns dict with keys:
        original   (3, H, W)      raw resized spectrogram  [0,1]
        mask_soft  (1, H, W)      U-Net sigmoid output     [0,1]
        mask_bin   (1, H, W)      thresholded binary mask  {0,1}
        roi_masked (3, H, W)      spectrogram × binary mask (before resize)
        roi_patch  (3, 224, 224)  final input to classifier (after resize)
        logits     (num_classes,) raw class scores
        pred_idx   int            argmax class index
        pred_conf  float          softmax confidence of top class
    """
    normed = normed.to(device)

    # ── Stage 1: U-Net mask ───────────────────────────────────────────────────
    mask_soft = model.unet(normed)                        # (1, 1, H, W) sigmoid

    # ── Stage 2: ROI extraction ───────────────────────────────────────────────
    mask_bin  = (mask_soft >= threshold).float()          # (1, 1, H, W)
    roi_raw   = normed * mask_bin                         # masked, still normalised
    roi_patch = F.interpolate(roi_raw, size=(224, 224),
                              mode="bilinear", align_corners=False)  # (1,3,224,224)

    # ── Stage 3: classifier ───────────────────────────────────────────────────
    logits = model.classifier(roi_patch)                  # (1, num_classes)
    probs  = torch.softmax(logits, dim=1)

    pred_idx  = int(logits.argmax(dim=1).item())
    pred_conf = float(probs[0, pred_idx].item())

    # ── Denorm for display ────────────────────────────────────────────────────
    roi_display = denormalize(roi_patch[0])               # (3, 224, 224)

    return {
        "original"   : raw.cpu(),                         # (3, H, W)
        "mask_soft"  : mask_soft[0].cpu(),                # (1, H, W)
        "mask_bin"   : mask_bin[0].cpu(),                 # (1, H, W)
        "roi_masked" : denormalize(roi_raw[0]).cpu(),     # (3, H, W)
        "roi_patch"  : roi_display.cpu(),                 # (3, 224, 224)
        "logits"     : logits[0].cpu(),
        "pred_idx"   : pred_idx,
        "pred_conf"  : pred_conf,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_stages(
    stages      : dict,
    title       : str,
    class_names : list,
    save_path   : str = None,
    show        : bool = False,
):
    """
    Draw a 1×5 figure:
        [Original] [Soft Mask] [Binary Mask] [Masked ROI] [ROI Patch 224²]
    """
    pred_name = class_names[stages["pred_idx"]] if class_names else str(stages["pred_idx"])

    fig = plt.figure(figsize=(22, 4.5))
    fig.suptitle(
        f"{title}\nPrediction: {pred_name}  ({stages['pred_conf']:.1%} confidence)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.04)

    panels = [
        (stages["original"],                     "1. Original\n(resized input)",    "viridis"),
        (stages["mask_soft"].repeat(3, 1, 1),    "2. U-Net mask\n(soft, sigmoid)",  "plasma"),
        (stages["mask_bin"].repeat(3, 1, 1),     "3. Binary mask\n(thresholded)",   "gray"),
        (stages["roi_masked"],                   "4. Masked ROI\n(before resize)",  "viridis"),
        (stages["roi_patch"],                    "5. ROI patch\n(224×224 → cls)",   "viridis"),
    ]

    for col, (tensor, panel_title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        img = to_np(tensor.clamp(0, 1))

        if cmap == "gray":
            ax.imshow(img[:, :, 0], cmap="gray", vmin=0, vmax=255)
        elif cmap == "plasma":
            ax.imshow(img[:, :, 0], cmap="plasma", vmin=0, vmax=255)
        else:
            ax.imshow(img)

        ax.set_title(panel_title, fontsize=9)
        ax.axis("off")

        # Annotate mask coverage on the binary mask panel
        if col == 2:
            coverage = float(stages["mask_bin"].mean().item()) * 100
            ax.set_xlabel(f"Coverage: {coverage:.1f}%", fontsize=8)

        # Annotate size
        h, w = tensor.shape[1], tensor.shape[2]
        ax.text(0.01, 0.01, f"{w}×{h}", transform=ax.transAxes,
                fontsize=7, color="white",
                bbox=dict(facecolor="black", alpha=0.45, pad=1, edgecolor="none"))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved -> {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_logit_bar(
    stages      : dict,
    class_names : list,
    save_path   : str = None,
    show        : bool = False,
):
    """Separate bar chart of softmax probabilities for all classes."""
    probs = torch.softmax(stages["logits"], dim=0).numpy()
    pred  = stages["pred_idx"]

    colors = ["#e74c3c" if i == pred else "#3498db" for i in range(len(probs))]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(class_names, probs * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Softmax probability (%)")
    ax.set_title("Classifier output probabilities")
    ax.set_ylim(0, 105)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{p:.1%}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


# =============================================================================
# Image source helpers
# =============================================================================

def collect_from_folder(folder: str, max_images: int) -> list:
    paths = sorted(Path(folder).glob("*.png"))
    if len(paths) > max_images:
        paths = random.sample(paths, max_images)
    return [(str(p), None) for p in paths]   # (path, true_label_or_None)


def collect_from_dataset(
    root: str, subsets: list, split: str, n_samples: int, seed: int = 42
) -> list:
    """Return list of (path, true_class_name) sampled from the given split."""
    from drone_dataloader import (
        DroneSpectrogramDataset, split_dataset, CLASS_NAMES,
    )

    full_ds = DroneSpectrogramDataset(root=root, subsets=subsets)
    train_sub, val_sub, test_sub = split_dataset(full_ds, seed=seed)

    split_map = {"train": train_sub, "val": val_sub, "test": test_sub}
    subset = split_map[split]

    indices = list(subset.indices)
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:n_samples]

    result = []
    for idx in indices:
        path, label = full_ds.samples[idx]
        result.append((path, full_ds.class_names[label]))
    return result


# =============================================================================
# Main
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(description="Visualise DronePipeline ROI stages")

    # Source
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",        help="Single PNG file to visualise")
    src.add_argument("--folder",       help="Directory of PNG files")
    src.add_argument("--from_dataset", action="store_true",
                     help="Sample from dataset splits")

    # Dataset options (used with --from_dataset)
    p.add_argument("--root",      default="output_spectrograms/")
    p.add_argument("--subsets",   nargs="+", default=["BOTH"])
    p.add_argument("--split",     default="test", choices=["train", "val", "test"])
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--seed",      type=int, default=42)

    # Folder options
    p.add_argument("--max_images", type=int, default=32)

    # Model
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    p.add_argument("--img_size",   nargs=2, type=int, default=[256, 512],
                   metavar=("H", "W"))
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="Mask binarisation threshold")

    # Output
    p.add_argument("--out_dir",  default="roi_vis",
                   help="Output directory for saved PNGs")
    p.add_argument("--no_save",  action="store_true",
                   help="Show with plt.show() instead of saving")
    p.add_argument("--device",   default=None, help="cpu | cuda")

    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    img_size = tuple(args.img_size)

    if args.no_save:
        matplotlib.use("TkAgg")   # switch to interactive backend

    print(f"\n  Device    : {device}")
    print(f"  Img size  : {img_size}")
    print(f"  Threshold : {args.threshold}")
    print(f"  Checkpoint: {args.checkpoint}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, class_names = load_model(args.checkpoint, img_size, device)
    print(f"  Classes   : {class_names}\n")

    # ── Collect image paths ───────────────────────────────────────────────────
    if args.image:
        image_list = [(args.image, None)]

    elif args.folder:
        image_list = collect_from_folder(args.folder, args.max_images)
        print(f"  Found {len(image_list)} images in {args.folder}")

    else:  # --from_dataset
        image_list = collect_from_dataset(
            root     = args.root,
            subsets  = args.subsets,
            split    = args.split,
            n_samples= args.n_samples,
            seed     = args.seed,
        )
        print(f"  Sampled {len(image_list)} images from {args.split} split")

    # ── Process each image ────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (img_path, true_label) in enumerate(image_list):
        fname = Path(img_path).stem
        print(f"  [{i+1:3d}/{len(image_list)}] {fname}")

        # load
        normed, raw = load_image_tensor(img_path, img_size, normalize=True)

        # forward through pipeline stages
        stages = get_pipeline_stages(
            model, normed, raw,
            threshold=args.threshold,
            device=device,
        )

        pred_name = class_names[stages["pred_idx"]]
        label_str = f"true={true_label}" if true_label else ""
        correct   = (true_label == pred_name) if true_label else None
        correct_str = "" if correct is None else ("✓" if correct else "✗")

        panel_title = (
            f"{fname}\n"
            f"{label_str}  pred={pred_name} {correct_str}"
            if true_label else fname
        )

        # Save stage panel
        stage_path = str(out_dir / f"{fname}_stages.png") if not args.no_save else None
        plot_stages(stages, panel_title, class_names,
                    save_path=stage_path, show=args.no_save)

        # Save logit bar
        bar_path = str(out_dir / f"{fname}_logits.png") if not args.no_save else None
        plot_logit_bar(stages, class_names,
                       save_path=bar_path, show=args.no_save)

        # Print stage summary to stdout
        coverage = float(stages["mask_bin"].mean().item()) * 100
        print(f"         mask coverage: {coverage:.1f}%  "
              f"pred: {pred_name} ({stages['pred_conf']:.1%})")

    print(f"\n  Done. Output written to: {out_dir}/")
    print(f"  Each image produces two files:")
    print(f"    <name>_stages.png  — 5-panel pipeline visualisation")
    print(f"    <name>_logits.png  — classifier probability bar chart")


if __name__ == "__main__":
    main()