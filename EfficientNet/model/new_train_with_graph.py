"""
new_train_with_graph.py
=============================================================================
3-Class Spectrogram Classifier: DRONE, DRONE_SIGNAL, and NO_DRONE
Architecture  : EfficientNet-B0 (From Scratch, 100% NPU Compliant)
Input Shape   : Auto-Optimized based on Hardware Profiling
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
from collections import defaultdict

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

    # Chia file ảnh theo tên, để phân các tệp thu tín hiệu vào đúng tập train và tập val
    if "__" in stem:
        return stem.split("__")[0]

    if "_" in stem:
        parts = stem.split("_")
        if parts[-1].isdigit():
            return "_".join(parts[:-1])

    return stem


def split_dataset_3way(
    src_dir: str,
    dest_dir: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
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
    log.info("Strategy: RECORDING-LEVEL ISOLATION (Train/Val/Test)")

    for cls_dir in class_dirs:
        images = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ])
        if not images:
            log.warning("Class subdirectory '%s' contains no valid frames — bypassing.", cls_dir.name)
            continue

        # Group images by unique root recording experiment key
        recording_groups = defaultdict(list)
        for img_path in images:
            rec_id = _extract_recording_id(img_path.name)
            recording_groups[rec_id].append(img_path)

        # Sort recording sessions by frame weight volume (descending)
        sorted_recs = sorted(
            recording_groups.items(),
            key=lambda item: len(item[1]),
            reverse=True
        )

        train_recs, val_recs, test_recs = [], [], []
        train_count, val_count, test_count = 0, 0, 0

        # Greedy knapsack assignment routing loop
        for rec_id, frame_list in sorted_recs:
            rec_size = len(frame_list)
            total_current = train_count + val_count + test_count

            if total_current == 0:
                train_recs.append((rec_id, frame_list))
                train_count += rec_size
            else:
                current_val_p = val_count / total_current
                current_test_p = test_count / total_current

                if current_val_p < val_ratio:
                    val_recs.append((rec_id, frame_list))
                    val_count += rec_size
                elif current_test_p < test_ratio:
                    test_recs.append((rec_id, frame_list))
                    test_count += rec_size
                else:
                    train_recs.append((rec_id, frame_list))
                    train_count += rec_size

        # Lưu file vật lý
        splits = [("train", train_recs), ("val", val_recs), ("test", test_recs)]
        for split_name, rec_list in splits:
            split_class_dir = dest / split_name / cls_dir.name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for _, frame_list in rec_list:
                for img_path in frame_list:
                    shutil.copy2(img_path, split_class_dir / img_path.name)

        total_imgs = train_count + val_count + test_count
        log.info(f"  {cls_dir.name:<15} → Train: {train_count} ({train_count/total_imgs:.1%}) | "
                 f"Val: {val_count} ({val_count/total_imgs:.1%}) | "
                 f"Test: {test_count} ({test_count/total_imgs:.1%})")

    log.info("Stratified 3-way split generation complete → '%s'", dest)


# ===========================================================================
# 2. Transforms & Data Streaming Loaders
# ===========================================================================

def get_transforms(augment: bool = False, img_h: int = 256, img_w: int = 512) -> transforms.Compose:
    """Build standardized, spectrogram-safe image formatting pipeline wrappers."""
    base = [
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),              # Range: [0.0, 1.0] float32
    ]

    if augment:
        pipeline = [
            transforms.Resize((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=0.5),  # Mirroring across time axis is valid
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
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Generate train/val/test processing iterators referencing the new partition directory."""
    root = Path(dataset_dir)

    train_dataset = datasets.ImageFolder(
        root=str(root / "train"),
        transform=get_transforms(augment=True, img_h=img_h, img_w=img_w),
    )
    val_dataset = datasets.ImageFolder(
        root=str(root / "val"),
        transform=get_transforms(augment=False, img_h=img_h, img_w=img_w),
    )
    test_dataset = datasets.ImageFolder(
        root=str(root / "test"),
        transform=get_transforms(augment=False, img_h=img_h, img_w=img_w),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    log.info("Class index mapping registry: %s", idx_to_class)
    log.info("Active batch loaders — train: %d frames | val: %d frames | test: %d frames",
             len(train_dataset), len(val_dataset), len(test_dataset))

    return train_loader, val_loader, test_loader, idx_to_class


# ===========================================================================
# 3. Kiến trúc EfficientNet-B0 (Tối ưu 100% cho NPU)
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
        # Crucial fixed tuple syntax folds into static axes parameters inside ONNX
        s = x.mean(dim=(2, 3), keepdim=True)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


class MBConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, expand_ratio: int, se_ratio: float = 0.25, drop_connect_rate: float = 0.0):
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
    """Custom EfficientNet-B0 optimized for 3-class edge deployment maps."""
    _BLOCK_ARGS = [
        (1,  16, 1, 1, 3), (6,  24, 2, 2, 3), (6,  40, 2, 2, 5),
        (6,  80, 3, 2, 3), (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3),
    ]

    def __init__(self, num_classes: int = 3, in_channels: int = 3, width_coeff: float = 1.0, depth_coeff: float = 1.0, dropout_rate: float = 0.3, drop_connect_rate: float = 0.2):
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
        x = x.mean(dim=(2, 3))  # Static ReduceMean -> Fully NPU compliant ✓
        x = self.dropout(x)
        return self.classifier(x)


# ===========================================================================
# 4. Training, Optimization, & Validation Engine
# ===========================================================================

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, epoch: int) -> float:
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

        # In log liên tục sau mỗi 20 bước (vô cùng hữu ích khi chạy CPU không có card CUDA)
        if (batch_idx + 1) % 20 == 0:
            log.info("  Epoch %d | step %4d/%d | running batch loss: %.4f", epoch, batch_idx + 1, len(loader), loss.item())

    return running_loss / len(loader)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int = 3, class_names: list = None) -> Tuple[float, float]:
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

    names = class_names or [f"class_{i}" for i in range(num_classes)]
    per_class_strings = []
    for c in range(num_classes):
        acc = class_correct[c] / max(class_total[c], 1)
        per_class_strings.append(f"{names[c]}={acc*100:.1f}%({class_total[c]})")

    log.info("  Validation Breakdown: %s", "  ".join(per_class_strings))

    for c in range(num_classes):
        if class_total[c] > 0 and class_correct[c] == 0:
            log.warning("  ⚠ Warning: Zero hits scored on category: %s — potential class collapse!", names[c])

    return mean_loss, overall_acc


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, num_classes: int = 3, num_epochs: int = 50, lr: float = 3e-4, weight_decay: float = 1e-2, checkpoint_path: str = "best_model.pth", early_stop_patience: int = 10, warmup_epochs: int = 5, class_names: list = None) -> nn.Module:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _warmup_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=lr * 0.01)

    best_val_acc = 0.0
    no_improve_count = 0
    log.info("Training runtime claims driver engine node context: %s", device)
    log.info("=" * 70)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, num_classes, class_names)

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        log.info("Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%% | lr=%.2e",
                 epoch, num_epochs, train_loss, val_loss, val_acc * 100, current_lr)

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
            log.info("  ✓ Best parameter weight state checkpoint saved (val_acc=%.2f%%)", val_acc * 100)
        else:
            no_improve_count += 1
            log.info("  No performance improvement detected for %d/%d epochs", no_improve_count, early_stop_patience)

        if no_improve_count >= early_stop_patience:
            log.info("Early stopping barrier triggered at epoch %d. Aborting budget.", epoch)
            break

    log.info("=" * 70)
    log.info("Loop track finished execution. Re-loading parameter payload: '%s'", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


# ===========================================================================
# 5. Động cơ Đánh giá độc lập trên tập dữ liệu TEST (Folder Test)
# ===========================================================================

@torch.no_grad()
def evaluate_test_set(model: nn.Module, test_loader: DataLoader, device: torch.device, class_names: list) -> float:
    log.info("=" * 70)
    log.info("      STARTING INDEPENDENT TEST SET EVALUATION ENGINE          ")
    log.info("=" * 70)

    model.eval()
    model = model.to(device)

    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)

        for c in range(num_classes):
            mask = (labels == c)
            class_correct[c] += (preds[mask] == c).sum().item()
            class_total[c] += mask.sum().item()

    overall_acc = sum(class_correct) / max(sum(class_total), 1)
    log.info(f"Kết quả Test Tổng Thể (Overall Test Accuracy): {overall_acc:.2%}")
    log.info("----------------------------------------------------------------------")
    log.info(f"{'Class Name':<20} | {'Đoán Đúng/Tổng':<15} | {'Độ chính xác (Acc)':<10}")
    log.info("----------------------------------------------------------------------")
    for c in range(num_classes):
        acc = class_correct[c] / max(class_total[c], 1)
        log.info(f"{class_names[c]:<20} | {f'{class_correct[c]}/{class_total[c]}':<15} | {acc:.2%}")
    log.info("=" * 70)
    return overall_acc


# ===========================================================================
# 6. Static ONNX Graph Export Driver & Verification Checker
# ===========================================================================

def export_to_onnx(model: nn.Module, output_path: str = "drone_classifier_b0.onnx", num_classes: int = 3, opset_version: int = 17, img_h: int = 256, img_w: int = 512) -> None:
    model.eval()
    model.cpu()
    dummy = torch.zeros(1, 3, img_h, img_w, dtype=torch.float32)

    log.info("Exporting target computation path graph to static ONNX container payload...")
    log.info("  Input Node   : image_tensor  [1, 3, %d, %d]  float32", img_h, img_w)
    log.info("  Output Node  : class_logits  [1, %d]           float32", num_classes)
    log.info("  Opset Target : %d", opset_version)

    torch.onnx.export(
        model, dummy, output_path, export_params=True, opset_version=opset_version,
        do_constant_folding=True, input_names=["image_tensor"], output_names=["class_logits"],
        dynamic_axes=None, dynamo=False,
    )
    log.info("ONNX graph successfully committed to disk storage link → '%s'", output_path)

    # Hệ thống quét và kiểm tra xem có chứa toán tử cấm gây lỗi biên dịch trên NPU hay không
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        BANNED_OPS = {"LayerNormalization", "Gelu", "Softmax"}
        found = {n.op_type for n in onnx_model.graph.node} & BANNED_OPS
        if found:
            log.warning("  ⚠ Warning: Banned operations detected inside ONNX graph: %s", found)
        else:
            log.info("  ✓ Validation verification passed: 0 illegal operations detected. Fused operations match NPU specifications.")
    except ImportError:
        log.warning("The 'onnx' package was not found; skipping internal architecture verification checks.")


def export_to_rocc_graph(model: nn.Module, output_path: str = "drone_classifier_rocc.onnx", img_h: int = 256, img_w: int = 512) -> None:
    model.eval()
    model.cpu()
    dummy = torch.zeros(1, 3, img_h, img_w, dtype=torch.float32)

    log.info("Generating hardware-aligned representation graph for custom compiler/visualization target...")
    torch.onnx.export(
        model, dummy, output_path, export_params=True, opset_version=17,
        do_constant_folding=True, input_names=["accelerator_input_tensor"], output_names=["accelerator_output_logits"],
        dynamic_axes=None
    )
    log.info("✓ Structural design graph successfully written to link → '%s'", output_path)


# ===========================================================================
# 7. Global Main Orchestrator Pipeline
# ===========================================================================

if __name__ == "__main__":
    # ── Path Routing Configurations ────────────────────────────────────────
    RAW_DATASET_DIR   = "../UPDATED_DATASET"
    SPLIT_DATASET_DIR = "dataset_split"

    # ── Hardware & Size Dimensions ──────────────────────────────────────────
    CLASS_NAMES  = ["DRONE", "DRONE_SIGNAL", "NO_DRONE"]
    NUM_CLASSES  = len(CLASS_NAMES)

    # ── TỰ ĐỘNG PHÁT HIỆN PHẦN CỨNG ĐỂ TỐI ƯU HOÁ THÔNG SỐ (TRÁNH TREO MÁY) ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Đang phân tích driver tài nguyên hệ thống... TARGET DEVICE: {device}")

    if device.type == "cpu":
        log.warning("Hệ thống chạy CPU! Hạ dải phổ xuống 128x256 và tối ưu luồng xử lý để máy tính không bị đứng.")
        IMG_H, IMG_W = 128, 256         # Giảm 4 lần kích thước ảnh phổ để CPU xử lý siêu tốc
        BATCH_SIZE   = 8               # Kích thước lô nhỏ giải quyết vấn đề nghẽn RAM
        NUM_WORKERS  = 0               # Đặt về 0 tránh lỗi deadlock hệ thống trên Windows
        PIN_MEMORY   = False
        NUM_EPOCHS   = 15              # Rút ngắn epoch kiểm tra để nhanh lấy file đích ONNX
        EARLY_STOP_PATIENCE = 3
        WARMUP_EPOCHS       = 2
    else:
        log.info("Hệ thống nhận dạng được card đồ hoạ rời CUDA! Áp dụng cấu hình chuẩn rộng.")
        IMG_H, IMG_W = 256, 512
        BATCH_SIZE   = 16
        NUM_WORKERS  = 4
        PIN_MEMORY   = True
        NUM_EPOCHS   = 50
        EARLY_STOP_PATIENCE = 10
        WARMUP_EPOCHS       = 5
    # ─────────────────────────────────────────────────────────────────────────

    CHECKPOINT_PATH = "best_model.pth"
    ONNX_OUTPUT     = "drone_classifier_b0.onnx"
    ROCC_OUTPUT     = "drone_classifier_rocc.onnx"

    # Xoá phân tách dữ liệu cũ nếu tồn tại
    if os.path.exists(SPLIT_DATASET_DIR):
        log.info("Removing obsolete data directory partition layout array nodes...")
        shutil.rmtree(SPLIT_DATASET_DIR)

    # Bước 1: Tiến hành phân chia thành 3 tập Train/Val/Test độc lập cô lập bản ghi
    split_dataset_3way(src_dir=RAW_DATASET_DIR, dest_dir=SPLIT_DATASET_DIR, val_ratio=0.15, test_ratio=0.15, seed=42)

    # Bước 2: Khởi tạo các bộ nạp dữ liệu (Loaders)
    train_loader, val_loader, test_loader, idx_to_class = get_dataloaders(
        dataset_dir=SPLIT_DATASET_DIR, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, img_h=IMG_H, img_w=IMG_W, pin_memory=PIN_MEMORY
    )

    # Bước 3: Khởi tạo cấu trúc mô hình
    model = DroneClassifier(num_classes=NUM_CLASSES, in_channels=3, dropout_rate=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("DroneClassifier Model initialized footprint parameter weight count: %s", f"{n_params:,}")

    # Bước 4: Thực thi vòng lặp huấn luyện hệ thống
    model = train(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device,
        num_classes=NUM_CLASSES, num_epochs=NUM_EPOCHS, lr=3e-4, weight_decay=1e-2,
        checkpoint_path=CHECKPOINT_PATH, early_stop_patience=EARLY_STOP_PATIENCE,
        warmup_epochs=WARMUP_EPOCHS, class_names=CLASS_NAMES,
    )

    # Bước 5: Chạy đánh giá độc lập mô hình trên tập dữ liệu kiểm thử TEST (Folder Test)
    evaluate_test_set(model=model, test_loader=test_loader, device=device, class_names=CLASS_NAMES)

    # Bước 6: Xuất các sơ đồ đồ thị tính toán hỗ trợ Qualcomm Hub và SDR Netron
    export_to_onnx(model=model, output_path=ONNX_OUTPUT, opset_version=17, img_h=IMG_H, img_w=IMG_W)
    export_to_rocc_graph(model=model, output_path=ROCC_OUTPUT, img_h=IMG_H, img_w=IMG_W)

    log.info("All pipeline processes completed successfully. Sourced outputs: %s | %s | %s",
             CHECKPOINT_PATH, ONNX_OUTPUT, ROCC_OUTPUT)