"""
new_train_with_graph.py
=============================================================================
3-Class Spectrogram Classifier: DRONE, DRONE_SIGNAL, and NO_DRONE
Architecture  : EfficientNet-B0 (From Scratch, 100% NPU Compliant)
Input Shape   : (1, 3, 256, 512) -> Full Wide Spectrogram
Target Export : ONNX Opset 17 -> Optimized for Qualcomm AI Hub
Split Logic   : Stratified Recording-Level Isolation (No Data Leakage)
"""

import os
import math
import shutil
import random
import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────────────────
#  Logging Configuration
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. Chia dataset thành các tập train / val / test
# ===========================================================================

def _extract_recording_id(filename: str) -> str:
    stem = Path(filename).stem

    if "__" in stem:
        return stem.split("__")[0]

    if "_" in stem:
        parts = stem.split("_")
        if parts[-1].isdigit():
            return "_".join(parts[:-1])

    return stem


def split_dataset(
        src_dir: str,
        dest_dir: str,
        split_ratio: float = 0.2,
        seed: int = 42,
) -> None:
    src = Path(src_dir)
    dest = Path(dest_dir)

    if dest.exists():
        log.info("Destination directory '%s' already exists — skipping partition creation.", dest)
        return

    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No operational class sub-folders found under '{src_dir}'.")

    random.seed(seed)
    log.info("Executing 3-Class Stratified Group Split Matrix...")
    log.info("Strategy: RECORDING-LEVEL ISOLATION (Bypasses Data Leakage)")

    for cls_dir in class_dirs:
        images = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ])
        if not images:
            log.warning("Class subdirectory '%s' contains no valid frames — bypassing.", cls_dir.name)
            continue

        from collections import defaultdict
        recording_groups = defaultdict(list)
        for img_path in images:
            rec_id = _extract_recording_id(img_path.name)
            recording_groups[rec_id].append(img_path)

        sorted_recs = sorted(
            recording_groups.items(),
            key=lambda item: len(item[1]),
            reverse=True
        )

        train_recs, val_recs = [], []
        train_count, val_count = 0, 0

        for rec_id, frame_list in sorted_recs:
            rec_size = len(frame_list)
            total_current = train_count + val_count

            if total_current == 0:
                train_recs.append((rec_id, frame_list))
                train_count += rec_size
            else:
                if (val_count / total_current) < split_ratio:
                    val_recs.append((rec_id, frame_list))
                    val_count += rec_size
                else:
                    train_recs.append((rec_id, frame_list))
                    train_count += rec_size

        for split, rec_list in [("train", train_recs), ("val", val_recs)]:
            split_class_dir = dest / split / cls_dir.name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for rec_id, frame_list in rec_list:
                for img_path in frame_list:
                    shutil.copy2(img_path, split_class_dir / img_path.name)

        log.info("  %-15s → train: %d imgs (%d recs) | val: %d imgs (%d recs) | val_ratio=%.1f%%",
                 cls_dir.name, train_count, len(train_recs), val_count, len(val_recs),
                 (val_count / (train_count + val_count)) * 100)

    log.info("Stratified recording-level split generation complete → '%s'", dest)


# ===========================================================================
# 2. Transforms & Data Streaming Loaders
# ===========================================================================

def get_transforms(augment: bool = False, img_h: int = 256, img_w: int = 512) -> transforms.Compose:
    base = [
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
    ]
    if augment:
        pipeline = [
            transforms.Resize((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.0),
            transforms.ToTensor(),
        ]
    else:
        pipeline = base
    return transforms.Compose(pipeline)


def get_dataloaders(
        dataset_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        img_h: int = 256,
        img_w: int = 512,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    root = Path(dataset_dir)

    train_dataset = datasets.ImageFolder(
        root=str(root / "train"),
        transform=get_transforms(augment=True, img_h=img_h, img_w=img_w),
    )
    val_dataset = datasets.ImageFolder(
        root=str(root / "val"),
        transform=get_transforms(augment=False, img_h=img_h, img_w=img_w),
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
    return train_loader, val_loader, idx_to_class


# ===========================================================================
# 3. Kiến trúc EfficientNet-B0 (Tối ưu 100% cho NPU))
# ===========================================================================

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
    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        sq = max(1, int(in_ch * se_ratio))
        self.fc1 = nn.Conv2d(in_ch, sq, 1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(sq, in_ch, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


class MBConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, expand_ratio: int,
                 se_ratio: float = 0.25, drop_connect_rate: float = 0.0):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
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
            nn.Conv2d(mid, mid, kernel_size, stride=stride, padding=pad, groups=mid, bias=False),
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
        keep = 1.0 - self.drop_connect_rate
        noise = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        return x / keep * torch.floor(noise + keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = self._drop_connect(out) + x
        return out


class DroneClassifier(nn.Module):
    _BLOCK_ARGS = [
        (1, 16, 1, 1, 3), (6, 24, 2, 2, 3), (6, 40, 2, 2, 5),
        (6, 80, 3, 2, 3), (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3),
    ]

    def __init__(self, num_classes: int = 3, in_channels: int = 3, width_coeff: float = 1.0, depth_coeff: float = 1.0,
                 dropout_rate: float = 0.3, drop_connect_rate: float = 0.2):
        super().__init__()
        stem_f = _round_filters(32, width_coeff)
        head_f = _round_filters(1280, width_coeff)
        n_blocks = sum(_round_repeats(n, depth_coeff) for _, _, n, _, _ in self._BLOCK_ARGS)
        dc_idx = 0

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_f, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_f, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )

        in_ch = stem_f
        self.blocks = nn.ModuleList()
        for expand_ratio, out_ch, num_layers, stride, ks in self._BLOCK_ARGS:
            out_ch = _round_filters(out_ch, width_coeff)
            num_layers = _round_repeats(num_layers, depth_coeff)
            for i in range(num_layers):
                dc_rate = drop_connect_rate * dc_idx / n_blocks
                self.blocks.append(MBConvBlock(
                    in_ch=in_ch, out_ch=out_ch, kernel_size=ks,
                    stride=stride if i == 0 else 1, expand_ratio=expand_ratio,
                    drop_connect_rate=dc_rate,
                ))
                in_ch = out_ch
                dc_idx += 1

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_ch, head_f, 1, bias=False),
            nn.BatchNorm2d(head_f, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(head_f, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_conv(x)
        x = x.mean(dim=(2, 3))
        x = self.dropout(x)
        return self.classifier(x)


# ===========================================================================
# 4. Training Engine
# ===========================================================================

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                    device: torch.device, epoch: int) -> float:
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int = 3,
             class_names: list = None) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item()

        preds = logits.argmax(1)
        for c in range(num_classes):
            mask = (labels == c)
            class_correct[c] += (preds[mask] == c).sum().item()
            class_total[c] += mask.sum().item()

    mean_loss = total_loss / len(loader)
    overall_acc = sum(class_correct) / max(sum(class_total), 1)
    return mean_loss, overall_acc


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
          num_classes: int = 3, num_epochs: int = 50, lr: float = 3e-4, weight_decay: float = 1e-2,
          checkpoint_path: str = "best_model.pth", early_stop_patience: int = 10, warmup_epochs: int = 5,
          class_names: list = None) -> nn.Module:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _warmup_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs),
                                                                  eta_min=lr * 0.01)

    best_val_acc = 0.0
    no_improve_count = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, num_classes, class_names)

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "class_names": class_names,
            }, checkpoint_path)
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            log.info("Early stopping barrier triggered at epoch %d.", epoch)
            break

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


# ===========================================================================
# 5. Graph Export Drivers (Standard ONNX & RoCC Visualizer Graph)
# ===========================================================================

#Convert model sang ONNX
def export_to_onnx(model: nn.Module, output_path: str = "drone_classifier_b0.onnx", num_classes: int = 3,
                   opset_version: int = 17, img_h: int = 256, img_w: int = 512) -> None:
    model.eval()
    model.cpu()
    dummy = torch.zeros(1, 3, img_h, img_w, dtype=torch.float32)

    log.info("Exporting target computation path graph to static ONNX container payload...")
    torch.onnx.export(
        model, dummy, output_path, export_params=True, opset_version=opset_version,
        do_constant_folding=True, input_names=["image_tensor"], output_names=["class_logits"],
        dynamic_axes=None, dynamo=False,
    )
    log.info("ONNX graph successfully committed to disk storage link → '%s'", output_path)


def export_to_rocc_graph(model: nn.Module, output_path: str = "drone_classifier_rocc.onnx", img_h: int = 256,
                         img_w: int = 512) -> None:

    model.eval()
    model.cpu()
    dummy = torch.zeros(1, 3, img_h, img_w, dtype=torch.float32)

    log.info("Generating hardware-aligned representation graph for custom compiler/visualization target...")

    # Using Opset 15/17 ensures standard convolutions, pooling, and matrix operations
    # extract as un-fused atomic components easily mapped to physical accelerator layouts
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["accelerator_input_tensor"],
        output_names=["accelerator_output_logits"],
        dynamic_axes=None
    )
    log.info("✓ Structural design graph successfully written to link → '%s'", output_path)
    log.info("  Tip: Open this file in Netron to check layer dimensions and kernel allocations.")


# ===========================================================================
# 6. Global Main Orchestrator Pipeline
# ===========================================================================

if __name__ == "__main__":
    RAW_DATASET_DIR = "NEW_DATASET"
    SPLIT_DATASET_DIR = "dataset_split"

    IMG_H, IMG_W = 256, 512
    CLASS_NAMES = ["DRONE", "DRONE_SIGNAL", "NO_DRONE"]
    NUM_CLASSES = len(CLASS_NAMES)

    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 4
    EARLY_STOP_PATIENCE = 10
    WARMUP_EPOCHS = 5

    CHECKPOINT_PATH = "best_model.pth"
    ONNX_OUTPUT = "drone_classifier_b0.onnx"
    ROCC_OUTPUT = "drone_classifier_rocc.onnx"

    if os.path.exists(SPLIT_DATASET_DIR):
        shutil.rmtree(SPLIT_DATASET_DIR)

    split_dataset(src_dir=RAW_DATASET_DIR, dest_dir=SPLIT_DATASET_DIR, split_ratio=0.2, seed=42)

    train_loader, val_loader, idx_to_class = get_dataloaders(
        dataset_dir=SPLIT_DATASET_DIR, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, img_h=IMG_H, img_w=IMG_W,
    )

    model = DroneClassifier(num_classes=NUM_CLASSES, in_channels=3, dropout_rate=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("DroneClassifier Model initialized footprint parameter weight count: %s", f"{n_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device,
        num_classes=NUM_CLASSES, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        checkpoint_path=CHECKPOINT_PATH, early_stop_patience=EARLY_STOP_PATIENCE,
        warmup_epochs=WARMUP_EPOCHS, class_names=CLASS_NAMES,
    )

    # Export production payload
    export_to_onnx(model=model, output_path=ONNX_OUTPUT, num_classes=NUM_CLASSES, opset_version=17, img_h=IMG_H,
                   img_w=IMG_W)

    # Export structural graph target for architecture visualization and compiler design pipelines
    export_to_rocc_graph(model=model, output_path=ROCC_OUTPUT, img_h=IMG_H, img_w=IMG_W)

    log.info("All pipeline processes completed successfully.")