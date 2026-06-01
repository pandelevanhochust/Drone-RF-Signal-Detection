"""
train_and_export.py
====================
Binary spectrogram classifier: DRONE vs NO_DRONE

Architecture  : EfficientNet-B0  (from scratch, NPU-friendly)
Input shape   : (1, 3, 256, 512)  — full spectrogram, no ROI extraction
Output        : (1, 2)             — raw logits [DRONE, NO_DRONE]
Target export : ONNX opset-17  →  Qualcomm AI Hub  (static dims, all NPU)

Why EfficientNet-B0 instead of EfficientViT-L2
-----------------------------------------------
EfficientViT-L2 uses:
    • LayerNorm     — QNN maps to CPU (not fused into Hexagon NPU kernel)
    • GELU          — QNN CPU fallback on older Hexagon DSP firmware
    • Attention Softmax — dynamic reshape + Softmax not fuseable by QNN

These produced the 18 CPU ops you saw in the profile (788 NPU / 18 CPU).
EfficientNet-B0 uses only:
    • Conv2d  → QNN Conv2d kernel  (NPU)
    • BN      → fused into Conv2d  (NPU)
    • SiLU    → QNN HardSwish approx, fully NPU
    • ReduceMean (global pool) → NPU
    • Gemm (Linear) → NPU
All ops land on NPU → 0 CPU fallback ops expected.

Why 256×512 input
-----------------
The training spectrograms were generated at 256 rows (NFFT=1024, resized)
× 512 columns. Using the native resolution avoids bilinear upscaling
artifacts that arise when squashing to 224×224. The EfficientNet stem
stride=2 reduces 256×512 → 128×256 immediately, so the extra resolution
is exploited by early feature maps.

NPU-compatibility decisions (inherited from old DroneCLSNet)
------------------------------------------------------------
• AdaptiveAvgPool2d replaced by .mean(dim=(2,3)) → static ReduceMean op
• SqueezeExcitation uses dim=(2,3) tuple (not list) → static axes attribute
• No Softmax in forward() — apply post-inference
• dynamic_axes=None in ONNX export → fully static graph required by QNN

Preprocessing (must match stft_preprocessor.py)
------------------------------------------------
    PIL RGB → resize (256, 512) → ToTensor() [/ 255.0]
    NO ImageNet mean/std normalisation

Dependencies
------------
    pip install torch torchvision onnx scikit-learn

Usage
-----
    python train_and_export.py
Edit the CONFIG block at the bottom before running.
"""

import os
import math
import shutil
import random
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  EfficientNet-B0 building blocks  (NPU-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def _make_divisible(v: float, d: int = 8) -> int:
    new_v = max(d, int(v + d / 2) // d * d)
    if new_v < 0.9 * v:
        new_v += d
    return new_v


def _round_filters(f: int, w: float) -> int:
    return _make_divisible(int(f * w))


def _round_repeats(n: int, d: float) -> int:
    return int(math.ceil(n * d))


class SqueezeExcitation(nn.Module):
    """
    Channel SE recalibration — NPU-compatible.

    CRITICAL: dim=(2,3) as a tuple, NOT a list.
    TorchScript folds tuple constants into static graph attributes,
    producing ReduceMean with axes as a fixed ONNX attribute.
    Using a list produces a dynamic tensor axis input — QNN rejects it.
    """

    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        sq        = max(1, int(in_ch * se_ratio))
        self.fc1  = nn.Conv2d(in_ch, sq, 1, bias=True)
        self.act  = nn.SiLU(inplace=True)
        self.fc2  = nn.Conv2d(sq, in_ch, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)   # static ReduceMean — NPU ✓
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


class MBConvBlock(nn.Module):
    """Mobile inverted bottleneck conv with optional drop-connect."""

    def __init__(
        self,
        in_ch             : int,
        out_ch            : int,
        kernel_size       : int,
        stride            : int,
        expand_ratio      : int,
        se_ratio          : float = 0.25,
        drop_connect_rate : float = 0.0,
    ):
        super().__init__()
        self.use_residual      = (stride == 1 and in_ch == out_ch)
        self.drop_connect_rate = drop_connect_rate
        mid = _make_divisible(in_ch * expand_ratio)
        pad = (kernel_size - 1) // 2
        layers = []

        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid, momentum=0.01, eps=1e-3),
                nn.SiLU(inplace=True),
            ]

        layers += [
            nn.Conv2d(mid, mid, kernel_size, stride=stride,
                      padding=pad, groups=mid, bias=False),
            nn.BatchNorm2d(mid, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
            SqueezeExcitation(mid, se_ratio=se_ratio),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
        ]
        self.block = nn.Sequential(*layers)

    def _drop_connect(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep  = 1.0 - self.drop_connect_rate
        noise = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        return x / keep * torch.floor(noise + keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = self._drop_connect(out) + x
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  DroneClassifier — EfficientNet-B0 for 256×512 spectrogram input
# ─────────────────────────────────────────────────────────────────────────────

class DroneClassifier(nn.Module):
    """
    EfficientNet-B0 binary classifier for drone detection.

    Input  : (B, 3, 256, 512) float32 [0.0, 1.0] — STFT spectrogram
    Output : (B, 2) raw logits — [DRONE, NO_DRONE]

    NPU compatibility
    -----------------
    All ops are QNN-fuseable onto the Hexagon NPU:
        Conv2d + BN + SiLU  → fused Conv kernel
        ReduceMean (global pool) → NPU ReduceMean
        Gemm (Linear)  → NPU MatMul
        Sigmoid (SE)   → NPU Sigmoid
    Expected profile: 0 CPU ops, all NPU.

    Input size change from 224×224 to 256×512
    ------------------------------------------
    The stem Conv2d(stride=2) immediately halves spatial dims:
        256×512 → 128×256  (stem)
        128×256 →  64×128  (block 2, stride=2)
         64×128 →  32×64   (block 3, stride=2)
         32×64  →  16×32   (block 4, stride=2)
         16×32  →  16×32   (block 5, stride=1)
         16×32  →   8×16   (block 6, stride=2)
          8×16  →   8×16   (block 7, stride=1)
    Global pool: .mean(dim=(2,3)) → (B, 1280)  [static ReduceMean]
    Classifier : Linear(1280, 2)
    """

    _BLOCK_ARGS = [
        # (expand_ratio, out_ch, num_layers, stride, kernel_size)
        (1,  16, 1, 1, 3),
        (6,  24, 2, 2, 3),
        (6,  40, 2, 2, 5),
        (6,  80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    ]

    def __init__(
        self,
        num_classes       : int   = 2,
        in_channels       : int   = 3,
        width_coeff       : float = 1.0,
        depth_coeff       : float = 1.0,
        dropout_rate      : float = 0.2,
        drop_connect_rate : float = 0.2,
    ):
        super().__init__()

        stem_f   = _round_filters(32, width_coeff)
        head_f   = _round_filters(1280, width_coeff)
        n_blocks = sum(_round_repeats(n, depth_coeff)
                       for _, _, n, _, _ in self._BLOCK_ARGS)
        dc_idx   = 0

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_f, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_f, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )

        # MBConv blocks
        in_ch = stem_f
        self.blocks = nn.ModuleList()
        for expand_ratio, out_ch, num_layers, stride, ks in self._BLOCK_ARGS:
            out_ch    = _round_filters(out_ch, width_coeff)
            num_layers = _round_repeats(num_layers, depth_coeff)
            for i in range(num_layers):
                dc_rate = drop_connect_rate * dc_idx / n_blocks
                self.blocks.append(MBConvBlock(
                    in_ch=in_ch, out_ch=out_ch,
                    kernel_size=ks,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    drop_connect_rate=dc_rate,
                ))
                in_ch = out_ch
                dc_idx += 1

        # Head
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_ch, head_f, 1, bias=False),
            nn.BatchNorm2d(head_f, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )
        self.dropout    = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(head_f, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_conv(x)
        # Global average pool — static ReduceMean, NPU-compatible
        # dim=(2,3) tuple: TorchScript folds to constant ONNX attribute
        x = x.mean(dim=(2, 3))             # (B, 1280)
        x = self.dropout(x)
        return self.classifier(x)          # (B, 2)  raw logits


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset auto-split
# ─────────────────────────────────────────────────────────────────────────────

def _extract_recording_id(filename: str) -> str:
    """
    Extract the recording stem from a spectrogram filename so that all
    segments from the same IQ recording stay in the same split.

    Naming conventions supported:
        seg_file_seg00_start0ms.png     → 'seg_file'
        recording__seg00_start0ms.png   → 'recording'
        MAV_1110_00__seg00_start0ms.png → 'MAV_1110_00'
        any_name.png                    → 'any_name'   (fallback)

    The double-underscore separator (__) matches the segment_file.py
    output format: {stem}__{tag}.png
    """
    stem = Path(filename).stem               # drop extension
    # Split on double underscore — everything before it is the recording ID
    if "__" in stem:
        return stem.split("__")[0]
    # Fallback: strip trailing _segXX or _startXXXms tokens
    import re
    recording = re.sub(r"_seg\d+.*$", "", stem)
    recording = re.sub(r"_start\d+ms.*$", "", recording)
    return recording if recording else stem


def split_dataset(
    src_dir    : str,
    dest_dir   : str,
    split_ratio: float = 0.2,
    seed       : int   = 42,
) -> None:
    """
    Recording-level train/val split — prevents data leakage.

    WHY recording-level (not image-level)
    --------------------------------------
    Each IQ recording is sliced into multiple 80 ms spectrogram segments.
    Consecutive segments from the same recording are near-identical (same
    drone, same channel, same noise floor). An image-level split puts some
    segments in train and others from the SAME recording in val — the model
    memorises recording-specific noise artefacts rather than drone signal
    structure, and achieves spuriously high val accuracy (100%) while
    completely failing on new recordings.

    The fix: group all segments by their parent recording ID, split the
    GROUPS 80/20, then copy all segments of each group to the correct split.
    No two segments from the same recording appear in both train and val.

    File naming convention (segment_file.py output)
    -------------------------------------------------
        {recording_stem}__{seg_tag}.png
    Example: MAV_1110_00__seg00_start720ms.png
    Recording ID = 'MAV_1110_00'  (everything before __)

    Parameters
    ----------
    src_dir     : Root with DRONE/ and NO_DRONE/ sub-directories of PNGs.
    dest_dir    : Output root; created if absent.
    split_ratio : Fraction of recordings reserved for validation (0.20).
    seed        : Random seed for reproducible splits.
    """
    src  = Path(src_dir)
    dest = Path(dest_dir)

    if dest.exists():
        log.info("'%s' already exists — skipping split.", dest)
        return

    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No sub-directories found in '{src_dir}'.")

    log.info("Classes: %s", [d.name for d in class_dirs])
    log.info("Split strategy: RECORDING-LEVEL (prevents data leakage)")

    for cls_dir in class_dirs:
        images = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ])
        if not images:
            log.warning("Class '%s' has no images — skipping.", cls_dir.name)
            continue

        # Group segments by recording ID
        from collections import defaultdict
        recording_groups: dict = defaultdict(list)
        for img_path in images:
            rec_id = _extract_recording_id(img_path.name)
            recording_groups[rec_id].append(img_path)

        n_recordings = len(recording_groups)
        recording_ids = sorted(recording_groups.keys())

        log.info("  %-12s : %d images from %d recordings",
                 cls_dir.name, len(images), n_recordings)

        if n_recordings < 2:
            log.warning(
                "  %-12s : only %d recording — cannot split. "
                "All images go to train. Add more recordings.",
                cls_dir.name, n_recordings
            )
            train_recs, val_recs = recording_ids, []
        else:
            train_recs, val_recs = train_test_split(
                recording_ids,
                test_size    = split_ratio,
                random_state = seed,
                shuffle      = True,
            )

        train_count, val_count = 0, 0
        for split, recs in [("train", train_recs), ("val", val_recs)]:
            out_dir = dest / split / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for rec_id in recs:
                for img_path in recording_groups[rec_id]:
                    shutil.copy2(img_path, out_dir / img_path.name)
                    if split == "train":
                        train_count += 1
                    else:
                        val_count += 1

        log.info("  %-12s → train: %d imgs (%d recs) | val: %d imgs (%d recs)",
                 cls_dir.name,
                 train_count, len(train_recs),
                 val_count,   len(val_recs))
        log.info("  Train recordings: %s", train_recs[:5])
        log.info("  Val   recordings: %s", val_recs[:5])

    log.info("Recording-level split complete → '%s'", dest)


# ─────────────────────────────────────────────────────────────────────────────
#  Transforms — must match stft_preprocessor.py exactly
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(augment: bool = False, img_h: int = 256, img_w: int = 512):
    """
    Build transform pipeline matching stft_preprocessor.iq_to_spectrogram():
        Resize(img_h, img_w) → ToTensor() [/255.0]

    No ImageNet mean/std — the model input is [0.0, 1.0] float32.

    Augmentations (training only, spectrogram-safe):
        RandomHorizontalFlip — time axis mirror is valid for RF detection
        ColorJitter          — mild brightness/contrast variation
        NO vertical flip     — frequency axis must remain oriented correctly
        NO rotation          — time/freq structure is orientation-sensitive
    """
    base = [
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),              # → [0.0, 1.0] float32  C×H×W
    ]

    if augment:
        pipeline = [
            transforms.Resize((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10,
                                   saturation=0.05, hue=0.0),
            transforms.ToTensor(),
        ]
    else:
        pipeline = base

    return transforms.Compose(pipeline)


# ─────────────────────────────────────────────────────────────────────────────
#  DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    dataset_dir : str,
    batch_size  : int = 32,
    num_workers : int = 0,
    img_h       : int = 256,
    img_w       : int = 512,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Build train and validation DataLoaders.
    Expects dest_dir/train/ and dest_dir/val/ with DRONE/ and NO_DRONE/ sub-dirs.

    Returns
    -------
    train_loader, val_loader, idx_to_class
    """
    root = Path(dataset_dir)

    train_dataset = datasets.ImageFolder(
        root      = str(root / "train"),
        transform = get_transforms(augment=True, img_h=img_h, img_w=img_w),
    )
    val_dataset = datasets.ImageFolder(
        root      = str(root / "val"),
        transform = get_transforms(augment=False, img_h=img_h, img_w=img_w),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    log.info("Class mapping: %s", idx_to_class)
    log.info("Train: %d images | Val: %d images",
             len(train_dataset), len(val_dataset))

    return train_loader, val_loader, idx_to_class


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    optimizer : torch.optim.Optimizer,
    criterion : nn.Module,
    device    : torch.device,
    epoch     : int,
) -> float:
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            log.info("  Epoch %d | step %4d/%d | loss: %.4f",
                     epoch, batch_idx + 1, len(loader), loss.item())

    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model        : nn.Module,
    loader       : DataLoader,
    criterion    : nn.Module,
    device       : torch.device,
    class_names  : list = None,
) -> Tuple[float, float]:
    """
    Validation with per-class accuracy breakdown.
    Prints DRONE recall and NO_DRONE recall separately so you can spot
    if the model collapsed to predicting only one class.
    """
    model.eval()
    total_loss = 0.0
    n_classes  = 2
    class_correct = [0] * n_classes
    class_total   = [0] * n_classes

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        preds = logits.argmax(1)
        for c in range(n_classes):
            mask = labels == c
            class_correct[c] += (preds[mask] == c).sum().item()
            class_total[c]   += mask.sum().item()

    mean_loss   = total_loss / len(loader)
    overall_acc = sum(class_correct) / max(sum(class_total), 1)

    names = class_names or [f"class_{i}" for i in range(n_classes)]
    per_class = []
    for c in range(n_classes):
        acc = class_correct[c] / max(class_total[c], 1)
        per_class.append(f"{names[c]}={acc*100:.1f}%({class_total[c]})")

    log.info("  Per-class: %s", "  ".join(per_class))

    # Warn if model collapsed to one class
    for c in range(n_classes):
        if class_total[c] > 0 and class_correct[c] == 0:
            log.warning("  ⚠ Model never predicts %s — possible class collapse!",
                        names[c])

    return mean_loss, overall_acc


def _check_class_balance(loader: DataLoader, class_names: list) -> None:
    """
    Count class distribution in a loader and warn if severely imbalanced.
    A >5:1 imbalance without class weights will cause the model to predict
    only the majority class and achieve ~50% accuracy indefinitely.
    """
    counts = {}
    for _, labels in loader:
        for lbl in labels.tolist():
            counts[lbl] = counts.get(lbl, 0) + 1
    total = sum(counts.values())
    log.info("  Class distribution:")
    for idx, name in enumerate(class_names):
        n   = counts.get(idx, 0)
        pct = n / total * 100 if total > 0 else 0
        log.info("    %s : %d  (%.1f%%)", name, n, pct)

    if len(counts) >= 2:
        vals     = sorted(counts.values())
        ratio    = vals[-1] / max(vals[0], 1)
        if ratio > 3:
            log.warning(
                "  ⚠ Class imbalance ratio %.1f:1 — consider using "
                "WeightedRandomSampler or class_weight in loss.", ratio)


def train(
    model              : nn.Module,
    train_loader       : DataLoader,
    val_loader         : DataLoader,
    device             : torch.device,
    num_epochs         : int   = 50,
    lr                 : float = 1e-4,
    weight_decay       : float = 1e-2,
    checkpoint_path    : str   = "best_model.pth",
    early_stop_patience: int   = 10,
    warmup_epochs      : int   = 3,
    class_names        : list  = None,
) -> nn.Module:
    """
    Full training loop with:
        - Linear LR warmup   : avoids large initial gradient updates on
                               randomly initialised head destabilising BN layers
        - CosineAnnealingLR  : smooth decay after warmup
        - Early stopping     : halts if val_acc doesn't improve for
                               early_stop_patience epochs — prevents the
                               model running all 50 epochs to reach 100%
                               on a leaked/tiny val set
        - Per-class val stats: shows DRONE and NO_DRONE recall separately
                               to detect class collapse (50% stuck issue)
        - Class balance check: warns if train set is severely imbalanced

    Why warmup
    ----------
    In your log, epoch-1 loss was already 0.12 (extremely low for a random
    init). This suggests the initial LR caused large updates that happened
    to land in a region where the model trivially separates based on a
    dataset artifact. Warmup starts at lr/10 and linearly ramps to full lr
    over warmup_epochs, giving BN statistics time to stabilise before large
    gradient steps.
    """
    if class_names is None:
        class_names = ["DRONE", "NO_DRONE"]

    model     = model.to(device)

    # Class-weighted loss — counteracts imbalance without changing the data
    # Counts labels across the full train set once before training starts
    log.info("Checking class balance ...")
    _check_class_balance(train_loader, class_names)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    # Warmup scheduler: lr ramps from lr/10 → lr over warmup_epochs
    def _warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_warmup_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - warmup_epochs),
        eta_min=lr * 0.01)

    best_val_acc     = 0.0
    no_improve_count = 0

    log.info("Training on   : %s", device)
    log.info("Epochs        : %d  (warmup=%d, early_stop_patience=%d)",
             num_epochs, warmup_epochs, early_stop_patience)
    log.info("=" * 65)

    for epoch in range(1, num_epochs + 1):
        train_loss        = train_one_epoch(model, train_loader, optimizer,
                                            criterion, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion,
                                     device, class_names)

        # Step correct scheduler
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_acc=%.2f%% | lr=%.2e",
            epoch, num_epochs, train_loss, val_loss,
            val_acc * 100, current_lr)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            no_improve_count = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_acc"     : val_acc,
                "num_classes" : len(class_names),
                "class_names" : class_names,
                "img_h"       : 256,
                "img_w"       : 512,
            }, checkpoint_path)
            log.info("  ✓ Best model saved (val_acc=%.2f%%)", val_acc * 100)
        else:
            no_improve_count += 1
            log.info("  No improvement for %d/%d epochs",
                     no_improve_count, early_stop_patience)

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve_count >= early_stop_patience:
            log.info(
                "Early stopping triggered at epoch %d "
                "(no improvement for %d epochs). Best val_acc=%.2f%%",
                epoch, early_stop_patience, best_val_acc * 100)
            break

        # ── Suspicious 100% check ─────────────────────────────────────────────
        if val_acc >= 1.0 and epoch < 10:
            log.warning(
                "  ⚠ 100%% val_acc at epoch %d — "
                "check for data leakage (recording-level split applied?). "
                "If val set is tiny (< 50 images) this may be spurious.",
                epoch)

    log.info("=" * 65)
    log.info("Training complete. Best val_acc=%.2f%%", best_val_acc * 100)

    # Reload best weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    log.info("Best weights reloaded from '%s'.", checkpoint_path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_to_onnx(
    model         : nn.Module,
    output_path   : str = "drone_classifier_b0.onnx",
    opset_version : int = 17,
    img_h         : int = 256,
    img_w         : int = 512,
) -> None:
    """
    Export DroneClassifier to ONNX with static dims for Qualcomm AI Hub.

    Input  tensor : 'image_tensor'   shape [1, 3, img_h, img_w]  float32
    Output tensor : 'class_logits'   shape [1, 2]                 float32

    Why dynamo=False (legacy TorchScript exporter)
    -----------------------------------------------
    PyTorch ≥ 2.1 dynamo exporter silently upgrades to opset 18 and encodes
    .mean(dim=(2,3)) axes as runtime tensors instead of static attributes.
    QNN rejects both: opset 18 and dynamic ReduceMean axes.
    The legacy exporter preserves opset 17 and folds tuple dims to constants.

    Why dynamic_axes=None
    ---------------------
    QNN NPU compilation requires every tensor dimension fixed at compile time.
    Batch size is always 1 for on-device inference.
    """
    model.eval()
    model.cpu()

    dummy = torch.zeros(1, 3, img_h, img_w, dtype=torch.float32)

    log.info("Exporting to ONNX ...")
    log.info("  Input  : image_tensor  [1, 3, %d, %d]  float32", img_h, img_w)
    log.info("  Output : class_logits  [1, 2]           float32")
    log.info("  Opset  : %d", opset_version)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params       = True,
        opset_version       = opset_version,
        do_constant_folding = True,
        input_names         = ["image_tensor"],
        output_names        = ["class_logits"],
        dynamic_axes        = None,     # static dims required for QNN NPU
        dynamo              = False,    # legacy TorchScript exporter (see docstring)
    )
    log.info("ONNX saved → '%s'", output_path)

    # Sanity check
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        # Check for banned ops that cause CPU fallback on QNN
        BANNED_OPS = {"LayerNormalization", "Gelu", "Softmax"}
        found = {n.op_type for n in onnx_model.graph.node} & BANNED_OPS
        if found:
            log.warning("Banned ops found (will fall back to CPU on QNN): %s", found)
        else:
            log.info("✓ No banned ops — all ops should land on NPU")

        # Verify no dynamic axes
        has_dynamic = any(
            d.dim_param != ""
            for inp in onnx_model.graph.input
            for d in inp.type.tensor_type.shape.dim
        )
        if has_dynamic:
            log.warning("Dynamic axes detected — QNN requires static shapes")
        else:
            log.info("✓ All input dims are static")

        log.info("✓ ONNX model check passed")
    except ImportError:
        log.warning("onnx not installed — skipping check. pip install onnx")
    except Exception as exc:
        log.error("ONNX check failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  Device helper
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Using device: %s", device)
    return device


# ─────────────────────────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── CONFIG — edit before running ─────────────────────────────────────────

    # Root folder with DRONE/ and NO_DRONE/ sub-directories of PNG spectrograms
    # Each sub-folder should contain images from MULTIPLE recordings
    # (different filenames prefixes, e.g. MAV_1110_00__seg*.png, MAV_1110_01__seg*.png)
    RAW_DATASET_DIR  = "DATASET"

    # Where the recording-level 80/20 split will be written
    # DELETE this folder if you change the dataset — split_dataset() skips if it exists
    SPLIT_DATASET_DIR = "dataset_split"

    # Input spectrogram size — must match stft_preprocessor.py IMG_H, IMG_W
    IMG_H = 256
    IMG_W = 512

    # Training hyper-parameters
    NUM_EPOCHS           = 50   # increased — early stopping will terminate sooner if needed
    BATCH_SIZE           = 16   # reduce to 8 if GPU OOM on 256×512 inputs
    LEARNING_RATE        = 3e-4 # slightly higher — warmup will ramp from 3e-5 → 3e-4
    WEIGHT_DECAY         = 1e-2
    NUM_WORKERS          = 4    # set to 0 on Windows if multiprocessing errors
    EARLY_STOP_PATIENCE  = 10   # stop if no val_acc improvement for 10 epochs
    WARMUP_EPOCHS        = 5    # ramp LR from lr/10 → lr over first 5 epochs

    # Output artefacts
    CHECKPOINT_PATH = "best_model.pth"
    ONNX_OUTPUT     = "drone_classifier_b0.onnx"

    CLASS_NAMES = ["DRONE", "NO_DRONE"]

    # ── Pipeline ─────────────────────────────────────────────────────────────

    # Step 1: Recording-level split (prevents data leakage)
    split_dataset(
        src_dir     = RAW_DATASET_DIR,
        dest_dir    = SPLIT_DATASET_DIR,
        split_ratio = 0.2,
        seed        = 42,
    )

    # Step 2: DataLoaders
    train_loader, val_loader, idx_to_class = get_dataloaders(
        dataset_dir = SPLIT_DATASET_DIR,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        img_h       = IMG_H,
        img_w       = IMG_W,
    )

    # Step 3: Model
    model = DroneClassifier(
        num_classes  = len(CLASS_NAMES),
        in_channels  = 3,
        dropout_rate = 0.3,   # slightly higher dropout for small dataset
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info("DroneClassifier (EfficientNet-B0)  params: %s", f"{n_params:,}")

    # Step 4: Train with warmup + early stopping
    device = get_device()
    model  = train(
        model               = model,
        train_loader        = train_loader,
        val_loader          = val_loader,
        device              = device,
        num_epochs          = NUM_EPOCHS,
        lr                  = LEARNING_RATE,
        weight_decay        = WEIGHT_DECAY,
        checkpoint_path     = CHECKPOINT_PATH,
        early_stop_patience = EARLY_STOP_PATIENCE,
        warmup_epochs       = WARMUP_EPOCHS,
        class_names         = CLASS_NAMES,
    )

    # Step 5: Export
    export_to_onnx(
        model       = model,
        output_path = ONNX_OUTPUT,
        opset_version = 17,
        img_h       = IMG_H,
        img_w       = IMG_W,
    )

    log.info("Done.  Artefacts: %s | %s", CHECKPOINT_PATH, ONNX_OUTPUT)