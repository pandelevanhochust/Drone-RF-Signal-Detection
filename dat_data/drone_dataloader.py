"""
drone_dataloader.py
====================
Transforms the DroneDetect_V2 STFT spectrogram PNGs produced by
segment_dataset.py into model-ready tensors for the DronePipeline.

Data flow:
    segment_dataset.py output/
        BOTH/AIR_FY/AIR_1110_00__seg00_start42ms.png   ← viridis RGB PNG
        BOTH/AIR_HO/...
        CLEAN/AIR_FY/...
        ...
            │
            ▼  DroneSpectrogramDataset  (this file)
            │
            ├─ Image load          PIL.Image (RGB)
            ├─ Resize              → (H_model, W_model)  divisible by 16
            ├─ ToTensor            → float32 [0,1]  shape (3, H, W)
            ├─ Normalize           → mean/std per-channel (ImageNet stats)
            ├─ [optional augment]  RandomHFlip, FreqShift, TimeMask
            │
            ▼
        tensor (3, 256, 512)  +  label int
            │
            ▼  DataLoader  (batch_size=B)
            │
        batch tensor (B, 3, 256, 512)
            │
            ▼  DronePipeline.forward(x)
            │
        probs  (B, num_classes)
        mask   (B, 1, 256, 512)

Dataset tree understood by this loader:
    <root>/
        <SUBSET>/          BOTH | CLEAN | BLUE | WIFI  (top-level split)
            <CLASS>/       AIR_FY | AIR_HO | … (20 drone-class folders)
                *.png

Label encoding:
    Class names are sorted alphabetically → integer index 0–N-1.
    The full 20-class list from the dataset is pre-defined in CLASS_NAMES.
    You can pass a custom subset to DroneSpectrogramDataset(classes=[...]).

Usage (minimal):
    from drone_dataloader import build_dataloaders
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        root="output_spectrograms/", batch_size=16
    )
    for images, labels in train_loader:
        probs = pipeline(images.to(device))

Usage (advanced / custom):
    from drone_dataloader import DroneSpectrogramDataset, get_transforms
    ds = DroneSpectrogramDataset(
        root="output_spectrograms/",
        subsets=["BOTH"],
        img_size=(256, 512),
        transform=get_transforms("train"),
    )
"""

from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# Known class names (20 folders inside each subset)
# Source: DroneDetect_V2 README / screenshot
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = sorted([
    "AIR_FY", "AIR_HO", "AIR_ON",
    "DIS_FY", "DIS_HO", "DIS_ON",      # DIS = Disco / Dissonance variant
    "INS_FY", "INS_HO", "INS_ON",
    "MIN_FY", "MIN_HO", "MIN_ON",
    "MP1_FY", "MP1_HO", "MP1_ON",
    "MP2_FY", "MP2_HO", "MP2_ON",
    "PHA_FY", "PHA_HO", "PHA_ON",
])

# Top-level subset folders available in the dataset
ALL_SUBSETS: list[str] = ["BLUE", "BOTH", "CLEAN", "WIFI"]

# ─────────────────────────────────────────────────────────────────────────────
# ImageNet normalisation statistics
# Using these because the spectrograms are viridis-coloured RGB images and
# the EfficientNet-B0 backbone was pretrained on ImageNet.
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ═════════════════════════════════════════════════════════════════════════════
# ░░  Custom augmentation transforms                                        ░░
# ═════════════════════════════════════════════════════════════════════════════

class FrequencyMask(nn.Module):
    """
    SpecAugment-style frequency masking.
    Randomly zeros out `max_f` consecutive frequency rows (height axis).
    Applied BEFORE normalisation on a PIL image via ToTensor first,
    or directly on a float tensor (C, H, W).
    """
    def __init__(self, max_f: int = 20):
        super().__init__()
        self.max_f = max_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, H, _ = x.shape
        f  = random.randint(0, self.max_f)
        f0 = random.randint(0, H - f) if f < H else 0
        x[:, f0:f0 + f, :] = 0.0
        return x


class TimeMask(nn.Module):
    """
    SpecAugment-style time masking.
    Randomly zeros out `max_t` consecutive time columns (width axis).
    """
    def __init__(self, max_t: int = 40):
        super().__init__()
        self.max_t = max_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, W = x.shape
        t  = random.randint(0, self.max_t)
        t0 = random.randint(0, W - t) if t < W else 0
        x[:, :, t0:t0 + t] = 0.0
        return x


class RandomBrightnessJitter(nn.Module):
    """
    Adds small random brightness perturbation to simulate power variations.
    Operates on float tensors (C, H, W) in [0, 1].
    """
    def __init__(self, delta: float = 0.05):
        super().__init__()
        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = random.uniform(-self.delta, self.delta)
        return (x + shift).clamp(0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# ░░  Transform factory                                                     ░░
# ═════════════════════════════════════════════════════════════════════════════

def get_transforms(
    mode: str,                              # "train" | "val" | "test"
    img_size: tuple[int, int] = (256, 512), # (H, W) — both must be div-by-16
    normalize: bool = True,
) -> transforms.Compose:
    """
    Build the torchvision transform pipeline for each split.

    Transformation order (train):
        1. Resize          → (H, W) using bilinear interpolation
        2. RandomHFlip     → p=0.5 (time-axis reflection, physically valid)
        3. ToTensor        → (C, H, W) float32 in [0, 1]
        4. BrightnessJitter→ ±5 % amplitude shift
        5. FrequencyMask   → SpecAugment freq masking (max 20 rows)
        6. TimeMask        → SpecAugment time masking (max 40 cols)
        7. Normalize       → ImageNet mean/std

    Transformation order (val / test):
        1. Resize
        2. ToTensor
        3. Normalize

    Args:
        mode       : split name — controls augmentation on/off
        img_size   : (H, W) fed to the model; H and W must be divisible by 16
        normalize  : set False to inspect raw pixel values during debugging

    Returns:
        torchvision.transforms.Compose pipeline
    """
    H, W = img_size
    assert H % 16 == 0 and W % 16 == 0, (
        f"img_size ({H}, {W}) must both be divisible by 16 "
        f"(U-Net applies 4× MaxPool2d with stride 2)."
    )

    resize = transforms.Resize((H, W),
                                interpolation=transforms.InterpolationMode.BILINEAR,
                                antialias=True)
    to_tensor = transforms.ToTensor()   # PIL [0,255] → float [0,1] + CHW
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if mode == "train":
        pipeline = [
            resize,
            transforms.RandomHorizontalFlip(p=0.5),   # time-flip augment
            to_tensor,
            RandomBrightnessJitter(delta=0.05),
            FrequencyMask(max_f=20),
            TimeMask(max_t=40),
        ]
    else:   # "val" or "test" — deterministic, no augmentation
        pipeline = [resize, to_tensor]

    if normalize:
        pipeline.append(norm)

    return transforms.Compose(pipeline)


# ═════════════════════════════════════════════════════════════════════════════
# ░░  Dataset                                                               ░░
# ═════════════════════════════════════════════════════════════════════════════

class DroneSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for DroneDetect_V2 STFT spectrogram PNG images.

    Scans <root>/<subset>/<class>/*.png and returns (tensor, label) pairs.

    Directory structure expected:
        root/
            BOTH/
                AIR_FY/
                    AIR_1110_00__seg00_start42ms.png
                    AIR_1110_00__seg01_start193ms.png
                    ...
                AIR_HO/
                    ...
            CLEAN/
                AIR_FY/
                    ...

    Args:
        root        : path to output_spectrograms/ (segment_dataset.py output)
        subsets     : which top-level folders to include (default: ["BOTH"])
                      Use ["BOTH", "CLEAN"] to merge interference conditions.
        classes     : optional explicit class list; inferred from disk if None
        transform   : torchvision transform pipeline (use get_transforms())
        img_size    : (H, W) resize target — only used if transform is None
        cache_paths : if True, pre-loads all file paths at init time (fast)

    Attributes:
        class_names : sorted list of class name strings
        class_to_idx: dict mapping class name → int label
        samples     : list of (image_path, label_int) tuples
        targets     : list of int labels (for WeightedRandomSampler)
    """

    def __init__(
        self,
        root: str,
        subsets: list[str]             = None,
        classes: Optional[list[str]]   = None,
        transform: Optional[Callable]  = None,
        img_size: tuple[int, int]      = (256, 512),
        cache_paths: bool              = True,
    ):
        super().__init__()
        self.root      = Path(root).expanduser().resolve()
        self.subsets   = subsets or ["BOTH"]
        self.transform = transform or get_transforms("val", img_size)
        self.img_size  = img_size

        # ── Discover class names ─────────────────────────────────────────────
        if classes is not None:
            self.class_names = sorted(classes)
        else:
            found = set()
            for subset in self.subsets:
                subset_dir = self.root / subset
                if subset_dir.is_dir():
                    found.update(
                        p.name for p in subset_dir.iterdir() if p.is_dir()
                    )
            self.class_names = sorted(found) if found else CLASS_NAMES

        self.class_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(self.class_names)
        }

        # ── Scan all PNG paths ───────────────────────────────────────────────
        self.samples: list[tuple[str, int]] = []
        if cache_paths:
            self._scan()

        self.targets: list[int] = [lbl for _, lbl in self.samples]

    # ─────────────────────────────────────────────────────────────────────────
    def _scan(self) -> None:
        """Walk the directory tree and collect (path, label) pairs."""
        for subset in self.subsets:
            subset_dir = self.root / subset
            if not subset_dir.is_dir():
                print(f"[DroneDataset] WARNING: subset folder not found: {subset_dir}")
                continue
            for class_name, label in self.class_to_idx.items():
                class_dir = subset_dir / class_name
                if not class_dir.is_dir():
                    continue
                for png in sorted(class_dir.glob("*.png")):
                    self.samples.append((str(png), label))

        if not self.samples:
            raise FileNotFoundError(
                f"No PNG files found under {self.root} "
                f"for subsets={self.subsets}. "
                f"Run segment_dataset.py first."
            )

    # ─────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # ── Load PNG as RGB (viridis colourmap → 3 channels) ─────────────────
        image = Image.open(img_path).convert("RGB")

        # ── Apply transform pipeline ─────────────────────────────────────────
        tensor = self.transform(image)   # → (3, H, W)  float32

        return tensor, label

    # ─────────────────────────────────────────────────────────────────────────
    def class_distribution(self) -> dict[str, int]:
        """Return sample counts per class (useful to detect imbalance)."""
        counts = Counter(self.targets)
        return {self.class_names[k]: v for k, v in sorted(counts.items())}

    def __repr__(self) -> str:
        dist = self.class_distribution()
        lines = [
            f"DroneSpectrogramDataset",
            f"  root    : {self.root}",
            f"  subsets : {self.subsets}",
            f"  classes : {len(self.class_names)}",
            f"  samples : {len(self.samples)}",
            f"  img_size: {self.img_size}",
            f"  per-class counts:",
        ]
        for cls, cnt in dist.items():
            lines.append(f"    {cls:12s}: {cnt:5d}")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# ░░  Train / Val / Test splitter                                           ░░
# ═════════════════════════════════════════════════════════════════════════════

def split_dataset(
    dataset: DroneSpectrogramDataset,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Stratified split of a DroneSpectrogramDataset into train / val / test.

    Stratification ensures each class is proportionally represented in
    every split (important since segments come from same .dat files).

    Args:
        dataset     : full dataset to split
        train_ratio : fraction for training   (default 0.70)
        val_ratio   : fraction for validation (default 0.15)
                      test_ratio = 1 - train - val = 0.15
        seed        : random seed for reproducibility

    Returns:
        (train_ds, val_ds, test_ds) — each a torch.utils.data.Subset
    """
    from torch.utils.data import Subset

    rng = random.Random(seed)

    # Group sample indices by class label
    class_indices: dict[int, list[int]] = {}
    for i, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(i)

    train_idx, val_idx, test_idx = [], [], []

    for label, indices in class_indices.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train:n_train + n_val])
        test_idx.extend(shuffled[n_train + n_val:])

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ═════════════════════════════════════════════════════════════════════════════
# ░░  DataLoader factory                                                    ░░
# ═════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    root: str,
    subsets: list[str]        = None,
    classes: list[str]        = None,
    img_size: tuple[int, int] = (256, 512),
    batch_size: int           = 16,
    num_workers: int          = 4,
    train_ratio: float        = 0.70,
    val_ratio: float          = 0.15,
    seed: int                 = 42,
    use_weighted_sampler: bool = True,
    pin_memory: bool          = True,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    One-call factory: scan → split → transform → DataLoader.

    The train loader uses separate augmentation transforms from val/test.
    Weighted random sampling is applied to the train loader to handle
    class imbalance (common when some drone types have fewer recordings).

    Args:
        root                 : output_spectrograms/ root directory
        subsets              : subset folders to include (default: ["BOTH"])
        classes              : explicit class list (inferred from disk if None)
        img_size             : (H, W) — both must be divisible by 16
        batch_size           : images per batch
        num_workers          : DataLoader worker processes
        train_ratio          : train split fraction (default 0.70)
        val_ratio            : validation split fraction (default 0.15)
        seed                 : reproducibility seed
        use_weighted_sampler : balance class frequency in training batches
        pin_memory           : pin tensors to CPU memory for faster GPU transfer

    Returns:
        train_loader, val_loader, test_loader, meta_dict

        meta_dict keys:
            "class_names"  : list[str]
            "class_to_idx" : dict[str, int]
            "num_classes"  : int
            "img_size"     : (H, W)
            "n_train"      : int
            "n_val"        : int
            "n_test"       : int
            "distribution" : dict[str, int]  (full dataset counts)
    """
    subsets = subsets or ["BOTH"]

    # ── Build three separate dataset objects with correct transforms ──────────
    # We build a full dataset first just to discover class names & split indices,
    # then attach proper per-split transforms to Subset wrappers.

    full_ds = DroneSpectrogramDataset(
        root        = root,
        subsets     = subsets,
        classes     = classes,
        transform   = get_transforms("val", img_size),   # neutral for splitting
        img_size    = img_size,
    )

    train_sub, val_sub, test_sub = split_dataset(
        full_ds, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    # Wrap Subsets so each sees the right transform
    train_ds = _TransformSubset(train_sub, get_transforms("train", img_size))
    val_ds   = _TransformSubset(val_sub,   get_transforms("val",   img_size))
    test_ds  = _TransformSubset(test_sub,  get_transforms("test",  img_size))

    # ── Weighted sampler (balances class frequencies in each training batch) ──
    sampler = None
    if use_weighted_sampler:
        train_labels = [full_ds.samples[i][1] for i in train_sub.indices]
        class_counts = Counter(train_labels)
        n_classes    = len(full_ds.class_names)
        weights_per_class = {
            c: 1.0 / (class_counts[c] + 1e-6) for c in range(n_classes)
        }
        sample_weights = torch.tensor(
            [weights_per_class[lbl] for lbl in train_labels],
            dtype=torch.float32,
        )
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True,
        )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = sampler,          # replaces shuffle when set
        shuffle     = (sampler is None),
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,             # avoid incomplete last batch
        persistent_workers = num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        persistent_workers = num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        persistent_workers = num_workers > 0,
    )

    meta = {
        "class_names"  : full_ds.class_names,
        "class_to_idx" : full_ds.class_to_idx,
        "num_classes"  : len(full_ds.class_names),
        "img_size"     : img_size,
        "n_train"      : len(train_ds),
        "n_val"        : len(val_ds),
        "n_test"       : len(test_ds),
        "distribution" : full_ds.class_distribution(),
    }

    return train_loader, val_loader, test_loader, meta


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Subset with overridden transform
# ─────────────────────────────────────────────────────────────────────────────

class _TransformSubset(Dataset):
    """Wraps a torch.utils.data.Subset and overrides its transform."""

    def __init__(self, subset, transform: Callable):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Load from original dataset (returns PIL-based tensor via val transform)
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image  = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


# ═════════════════════════════════════════════════════════════════════════════
# ░░  Normalisation inverse (for visualisation)                             ░░
# ═════════════════════════════════════════════════════════════════════════════

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation for display.
    tensor : (C, H, W) or (B, C, H, W) normalised float32
    Returns : same shape, values clamped to [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)
    std  = torch.tensor(IMAGENET_STD,  dtype=tensor.dtype, device=tensor.device)
    if tensor.ndim == 4:           # batch
        mean = mean[None, :, None, None]
        std  = std[None,  :, None, None]
    else:                          # single image
        mean = mean[:, None, None]
        std  = std[:,  None, None]
    return (tensor * std + mean).clamp(0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# ░░  Verification / demo                                                   ░░
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Verify DroneDetect_V2 data loading pipeline."
    )
    parser.add_argument("--root",       required=True,
                        help="Path to output_spectrograms/ directory")
    parser.add_argument("--subsets",    nargs="+", default=["BOTH"],
                        help="Subset folders to use (e.g. BOTH CLEAN)")
    parser.add_argument("--img_size",   nargs=2, type=int, default=[256, 512],
                        metavar=("H", "W"),
                        help="Resize target. Both must be divisible by 16.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers",    type=int, default=0,
                        help="DataLoader num_workers (0=main process)")
    args = parser.parse_args()

    img_size = tuple(args.img_size)   # (H, W)

    print(f"\n{'═'*65}")
    print("  DroneDetect_V2  →  DronePipeline  Data Loading Verification")
    print(f"{'═'*65}")
    print(f"  Root    : {args.root}")
    print(f"  Subsets : {args.subsets}")
    print(f"  img_size: {img_size}  (H × W)")
    print(f"  Batch   : {args.batch_size}")

    # ── Build loaders ──────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        root        = args.root,
        subsets     = args.subsets,
        img_size    = img_size,
        batch_size  = args.batch_size,
        num_workers = args.workers,
    )

    print(f"\n  Classes ({meta['num_classes']})  : {meta['class_names']}")
    print(f"  Train samples : {meta['n_train']}")
    print(f"  Val   samples : {meta['n_val']}")
    print(f"  Test  samples : {meta['n_test']}")

    print("\n  Class distribution (full dataset):")
    for cls, cnt in meta["distribution"].items():
        bar = "█" * (cnt // max(1, max(meta["distribution"].values()) // 30))
        print(f"    {cls:12s} {cnt:5d}  {bar}")

    # ── Inspect one training batch ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Inspecting one training batch …")
    images, labels = next(iter(train_loader))

    print(f"\n  ✓ images.shape  : {tuple(images.shape)}")
    print(f"  ✓ images.dtype  : {images.dtype}")
    print(f"  ✓ images.min()  : {images.min():.4f}  (after normalisation)")
    print(f"  ✓ images.max()  : {images.max():.4f}")
    print(f"  ✓ labels.shape  : {tuple(labels.shape)}")
    print(f"  ✓ label sample  : {labels[:8].tolist()}")
    print(f"  ✓ class sample  : {[meta['class_names'][l] for l in labels[:8].tolist()]}")

    # ── Confirm shape is pipeline-compatible ──────────────────────────────
    B, C, H, W = images.shape
    assert C == 3,       f"Expected 3 channels, got {C}"
    assert H % 16 == 0,  f"H={H} must be divisible by 16 for the U-Net"
    assert W % 16 == 0,  f"W={W} must be divisible by 16 for the U-Net"

    print(f"\n  ✓ Shape check passed: (B={B}, C={C}, H={H}, W={W})")
    print(f"  ✓ H%16={H%16}, W%16={W%16}  — U-Net MaxPool constraint satisfied")

    # ── Optional: run through DronePipeline if available ──────────────────
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from drone_full_pipeline import DronePipeline

        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = DronePipeline(
            num_classes = meta["num_classes"],
            in_channels = 3,
        ).to(device)
        pipeline.eval()

        with torch.no_grad():
            probs, mask = pipeline(images.to(device), return_mask=True)

        print(f"\n  ✓ DronePipeline forward pass OK")
        print(f"    probs shape : {tuple(probs.shape)}")
        print(f"    mask  shape : {tuple(mask.shape)}")
        print(f"    probs sum   : {probs[0].sum():.6f}  (≈ 1.0)")

    except ImportError:
        print("\n  [INFO] drone_full_pipeline.py not found — skipping pipeline test.")

    print(f"\n{'═'*65}")
    print("  ✓ Data loading pipeline verified. Ready for training.")
    print(f"{'═'*65}\n")