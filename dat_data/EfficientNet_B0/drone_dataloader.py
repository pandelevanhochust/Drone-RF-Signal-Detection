from __future__ import annotations
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# Class names  (8 merged drone-type folders + NO_DRONE)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = sorted([
    "AIR", "DIS", "INS", "MIN",
    "MP1", "MP2", "NO_DRONE", "PHA",
])

ALL_SUBSETS: list[str] = ["BLUE", "BOTH", "CLEAN", "WIFI"]
# ALL_SUBSETS: list[str] = ["BOTH"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ═════════════════════════════════════════════════════════════════════════════
# Custom augmentation transforms  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

class FrequencyMask(nn.Module):
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
    def __init__(self, delta: float = 0.05):
        super().__init__()
        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = random.uniform(-self.delta, self.delta)
        return (x + shift).clamp(0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# Transform factory  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def get_transforms(
    mode: str,
    img_size: tuple[int, int] = (256, 512),
    normalize: bool = True,
) -> transforms.Compose:
    H, W = img_size
    assert H % 16 == 0 and W % 16 == 0, (
        f"img_size ({H}, {W}) must both be divisible by 16."
    )

    resize    = transforms.Resize((H, W),
                                   interpolation=transforms.InterpolationMode.BILINEAR,
                                   antialias=True)
    to_tensor = transforms.ToTensor()
    norm      = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if mode == "train":
        pipeline = [
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            to_tensor,
            RandomBrightnessJitter(delta=0.05),
            FrequencyMask(max_f=20),
            TimeMask(max_t=40),
        ]
    else:
        pipeline = [resize, to_tensor]

    if normalize:
        pipeline.append(norm)

    return transforms.Compose(pipeline)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset  (unchanged except _scan keeps full path for recording-ID parsing)
# ═════════════════════════════════════════════════════════════════════════════

class DroneSpectrogramDataset(Dataset):
    """
    Scans <root>/<subset>/<class>/*.png and returns (tensor, label) pairs.

    self.samples : list of (abs_path_str, label_int)
    self.targets : list of label_int  (for WeightedRandomSampler)
    """

    def __init__(
        self,
        root: str,
        subsets: list[str]            = None,
        classes: Optional[list[str]]  = None,
        transform: Optional[Callable] = None,
        img_size: tuple[int, int]     = (256, 512),
        cache_paths: bool             = True,
    ):
        super().__init__()
        self.root      = Path(root).expanduser().resolve()
        self.subsets   = subsets or ["BLUE", "BOTH", "CLEAN", "WIFI"]
        self.transform = transform or get_transforms("val", img_size)
        self.img_size  = img_size

        if classes is not None:
            self.class_names = sorted(classes)
        else:
            found = set()
            for subset in self.subsets:
                subset_dir = self.root / subset
                if subset_dir.is_dir():
                    found.update(p.name for p in subset_dir.iterdir() if p.is_dir())
            self.class_names = sorted(found) if found else CLASS_NAMES

        self.class_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(self.class_names)
        }

        self.samples: list[tuple[str, int]] = []
        if cache_paths:
            self._scan()
        self.targets: list[int] = [lbl for _, lbl in self.samples]

    def _scan(self) -> None:
        for subset in self.subsets:
            subset_dir = self.root / subset
            if not subset_dir.is_dir():
                print(f"[DroneDataset] WARNING: subset not found: {subset_dir}")
                continue
            for class_name, label in self.class_to_idx.items():
                class_dir = subset_dir / class_name
                if not class_dir.is_dir():
                    continue
                for png in sorted(class_dir.glob("*.png")):
                    self.samples.append((str(png), label))

        if not self.samples:
            raise FileNotFoundError(
                f"No PNG files found under {self.root} for subsets={self.subsets}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image  = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label

    def class_distribution(self) -> dict[str, int]:
        counts = Counter(self.targets)
        return {self.class_names[k]: v for k, v in sorted(counts.items())}

    def __repr__(self) -> str:
        dist  = self.class_distribution()
        lines = [
            "DroneSpectrogramDataset",
            f"  root    : {self.root}",
            f"  subsets : {self.subsets}",
            f"  classes : {len(self.class_names)}",
            f"  samples : {len(self.samples)}",
            f"  img_size: {self.img_size}",
            "  per-class counts:",
        ]
        for cls, cnt in dist.items():
            lines.append(f"    {cls:12s}: {cnt:5d}")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# ░░  Strategy 1 — Recording-ID-level stratified split                     ░░
# ═════════════════════════════════════════════════════════════════════════════

def _recording_id(path: str) -> str:
    """
    Extract the recording ID from a segment filename.

    Examples
    --------
    'INS_1110_01__seg00_start42ms.png'   → 'INS_1110_01'
    'AIR_2210_03__seg12_start1498ms.png' → 'AIR_2210_03'
    'NO_DRONE_001__seg00_start0ms.png'   → 'NO_DRONE_001'

    The split key is everything before '__seg', which uniquely identifies
    the original .dat recording file the segments came from.
    Falls back to the full stem if '__seg' is not found.
    """
    stem = Path(path).stem          # drop .png
    if "__seg" in stem:
        return stem.split("__seg")[0]
    return stem                     # fallback: treat whole name as unique ID

def split_dataset(
    dataset: DroneSpectrogramDataset,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Stratified split at the RECORDING level (not segment level).

    All segments that share a recording ID are kept in the same split,
    preventing near-duplicate frames from leaking between train and val/test.

    Steps
    -----
    1. Group sample indices by (label, recording_id).
    2. For each class, collect unique recording IDs and shuffle them.
    3. Assign recording IDs to train/val/test by count ratio.
    4. Expand recording-ID groups back to individual sample indices.

    Returns
    -------
    (train_subset, val_subset, test_subset)  — torch.utils.data.Subset
    """
    rng = random.Random(seed)

    # ── Step 1: Group indices by (label, recording_id) ────────────────────────
    # rec_groups[(label, rec_id)] = [idx0, idx1, …]
    rec_groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, (path, label) in enumerate(dataset.samples):
        rec_id = _recording_id(path)
        rec_groups[(label, rec_id)].append(i)

    # ── Step 2: For each class, split at recording-ID level ───────────────────
    train_idx, val_idx, test_idx = [], [], []

    # Collect unique recording IDs per class
    class_recs: dict[int, list[str]] = defaultdict(list)
    for (label, rec_id) in rec_groups:
        class_recs[label].append(rec_id)

    for label, rec_ids in class_recs.items():
        shuffled_recs = rec_ids[:]
        rng.shuffle(shuffled_recs)

        n       = len(shuffled_recs)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        # test gets the remainder

        train_recs = shuffled_recs[:n_train]
        val_recs   = shuffled_recs[n_train : n_train + n_val]
        test_recs  = shuffled_recs[n_train + n_val :]

        for rec_id in train_recs:
            train_idx.extend(rec_groups[(label, rec_id)])
        for rec_id in val_recs:
            val_idx.extend(rec_groups[(label, rec_id)])
        for rec_id in test_recs:
            test_idx.extend(rec_groups[(label, rec_id)])

    # ── Step 3: Report split sizes ────────────────────────────────────────────
    total = len(dataset.samples)
    print(f"  [split] Recording-level split  (seed={seed})")
    print(f"    Unique recordings : "
          f"{len(rec_groups)} across {len(class_recs)} classes")
    print(f"    Train segments    : {len(train_idx):6d}  "
          f"({100*len(train_idx)/total:.1f}%)")
    print(f"    Val   segments    : {len(val_idx):6d}  "
          f"({100*len(val_idx)/total:.1f}%)")
    print(f"    Test  segments    : {len(test_idx):6d}  "
          f"({100*len(test_idx)/total:.1f}%)")

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ═════════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ═════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    root: str,
    subsets: list[str]         = None,
    classes: list[str]         = None,
    img_size: tuple[int, int]  = (256, 512),
    batch_size: int            = 16,
    num_workers: int           = 4,
    train_ratio: float         = 0.70,
    val_ratio: float           = 0.15,
    seed: int                  = 42,
    use_weighted_sampler: bool = True,
    pin_memory: bool           = True,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:

    subsets = subsets or ["BLUE", "BOTH", "CLEAN", "WIFI"]

    full_ds = DroneSpectrogramDataset(
        root      = root,
        subsets   = subsets,
        classes   = classes,
        transform = get_transforms("val", img_size),
        img_size  = img_size,
    )

    train_sub, val_sub, test_sub = split_dataset(
        full_ds,
        train_ratio = train_ratio,
        val_ratio   = val_ratio,
        seed        = seed,
    )

    train_ds = _TransformSubset(train_sub, get_transforms("train", img_size))
    val_ds   = _TransformSubset(val_sub,   get_transforms("val",   img_size))
    test_ds  = _TransformSubset(test_sub,  get_transforms("test",  img_size))

    # Weighted sampler — uses train split labels
    sampler = None
    if use_weighted_sampler:
        train_labels  = [full_ds.samples[i][1] for i in train_sub.indices]
        class_counts  = Counter(train_labels)
        n_classes     = len(full_ds.class_names)
        w_per_class   = {c: 1.0 / (class_counts[c] + 1e-6) for c in range(n_classes)}
        sample_weights = torch.tensor(
            [w_per_class[lbl] for lbl in train_labels], dtype=torch.float32
        )
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size         = batch_size,
        sampler            = sampler,
        shuffle            = (sampler is None),
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        drop_last          = True,
        persistent_workers = num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        persistent_workers = num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
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
    def __init__(self, subset, transform: Callable):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image  = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


# ─────────────────────────────────────────────────────────────────────────────
# Denormalize utility
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)
    std  = torch.tensor(IMAGENET_STD,  dtype=tensor.dtype, device=tensor.device)
    if tensor.ndim == 4:
        mean = mean[None, :, None, None]
        std  = std[None,  :, None, None]
    else:
        mean = mean[:, None, None]
        std  = std[:,  None, None]
    return (tensor * std + mean).clamp(0.0, 1.0)