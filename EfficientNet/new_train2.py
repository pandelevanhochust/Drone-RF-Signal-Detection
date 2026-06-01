"""
train_and_export.py
====================
Binary spectrogram classifier: DRONE vs NO_DRONE
Architecture : efficientvit_l2  (MIT Han Lab via timm)
Target export: ONNX opset-17  →  Qualcomm AI Hub (static [1,3,224,224])

Dependencies
------------
    pip install torch torchvision timm onnx scikit-learn

Usage
-----
    python train_and_export.py
Edit the CONFIG block at the bottom before running.
"""

import os
import shutil
import random
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. Dataset Auto-Split Utility
# ===========================================================================

def split_dataset(
    src_dir: str,
    dest_dir: str,
    split_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Walk *src_dir* (two class sub-folders) and create:

        dest_dir/
            train/
                class_A/
                class_B/
            val/
                class_A/
                class_B/

    Parameters
    ----------
    src_dir     : Root folder with per-class sub-directories.
    dest_dir    : Destination root; will be created if absent.
    split_ratio : Fraction of images reserved for validation (default 0.20).
    seed        : Random seed for reproducible splits.
    """
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    if dest_path.exists():
        log.info("Destination '%s' already exists — skipping split.", dest_path)
        return

    class_dirs = sorted([d for d in src_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No sub-directories found in '{src_dir}'.")

    log.info("Found %d classes: %s", len(class_dirs), [d.name for d in class_dirs])

    for cls_dir in class_dirs:
        images = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        ])

        if len(images) == 0:
            log.warning("Class '%s' has no images — skipping.", cls_dir.name)
            continue

        train_imgs, val_imgs = train_test_split(
            images,
            test_size=split_ratio,
            random_state=seed,
            shuffle=True,
        )

        for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
            split_class_dir = dest_path / split / cls_dir.name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for img_path in img_list:
                shutil.copy2(img_path, split_class_dir / img_path.name)

        log.info(
            "  %-15s → train: %d | val: %d",
            cls_dir.name,
            len(train_imgs),
            len(val_imgs),
        )

    log.info("Dataset split complete. Saved to '%s'.", dest_path)


# ===========================================================================
# 2. Transforms & DataLoaders
# ===========================================================================

def get_transforms(augment: bool = False) -> transforms.Compose:
    base = [
        transforms.Lambda(lambda img: img.convert("RGB")),  # ← ADD THIS LINE
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]

    if augment:
        augmentation = [
            transforms.Lambda(lambda img: img.convert("RGB")),  # ← ADD HERE TOO
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.05,
                hue=0.0,
            ),
            transforms.ToTensor(),
        ]
        pipeline = augmentation
    else:
        pipeline = base

    return transforms.Compose(pipeline)


def get_dataloaders(
    dataset_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Build train and validation DataLoaders from a pre-split directory.

    Parameters
    ----------
    dataset_dir : Root of the split dataset (contains 'train/' and 'val/').
    batch_size  : Mini-batch size.
    num_workers : CPU workers for data loading.

    Returns
    -------
    train_loader, val_loader, idx_to_class
    """
    dataset_path = Path(dataset_dir)

    train_dataset = datasets.ImageFolder(
        root=str(dataset_path / "train"),
        transform=get_transforms(augment=True),
    )
    val_dataset = datasets.ImageFolder(
        root=str(dataset_path / "val"),
        transform=get_transforms(augment=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    log.info("Class mapping (index → label): %s", idx_to_class)
    log.info(
        "Splits — train: %d images | val: %d images",
        len(train_dataset),
        len(val_dataset),
    )

    return train_loader, val_loader, idx_to_class


# ===========================================================================
# 3. Model Factory
# ===========================================================================

def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Instantiate EfficientViT-L2 from timm with a custom classification head.

    Parameters
    ----------
    num_classes : Number of output logits (2 for DRONE / NO_DRONE).
    pretrained  : Load ImageNet-pretrained backbone weights when True.

    Returns
    -------
    torch.nn.Module
    """
    model = timm.create_model(
        "efficientvit_l2",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    log.info(
        "Built efficientvit_l2 — output head: %d logits | pretrained: %s",
        num_classes,
        pretrained,
    )
    return model


# ===========================================================================
# 4. Training Loop
# ===========================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one full training pass; return mean loss."""
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)                  # → [B, 2]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            log.info(
                "  Epoch %d | step %4d/%d | batch loss: %.4f",
                epoch,
                batch_idx + 1,
                len(loader),
                loss.item(),
            )

    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run validation; return (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    mean_loss = total_loss / len(loader)
    accuracy = correct / total
    return mean_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    checkpoint_path: str = "best_model.pth",
) -> nn.Module:
    """
    Full training + validation loop with best-model checkpointing.

    Parameters
    ----------
    model           : Network to train.
    train_loader    : Training DataLoader.
    val_loader      : Validation DataLoader.
    device          : Target device (CPU / CUDA / MPS).
    num_epochs      : Total training epochs.
    lr              : Initial AdamW learning rate.
    weight_decay    : AdamW L2 regularisation coefficient.
    checkpoint_path : File path where the best weights are saved.

    Returns
    -------
    model with best validation weights loaded.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing over full training budget
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    best_val_acc = 0.0
    log.info("Starting training on device: %s", device)
    log.info("=" * 60)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        log.info(
            "Epoch %2d/%d | train_loss: %.4f | val_loss: %.4f | val_acc: %.2f%%",
            epoch,
            num_epochs,
            train_loss,
            val_loss,
            val_acc * 100,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            log.info("  ✓ New best model saved (val_acc: %.2f%%)", best_val_acc * 100)

    log.info("=" * 60)
    log.info("Training complete. Best val accuracy: %.2f%%", best_val_acc * 100)

    # Reload best weights before returning
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    log.info("Best weights reloaded from '%s'.", checkpoint_path)

    return model


# ===========================================================================
# 5. ONNX Export
# ===========================================================================

def export_to_onnx(
    model: nn.Module,
    output_path: str = "efficientvit_l2_drone.onnx",
    opset_version: int = 17,
) -> None:
    """
    Export the trained model to ONNX with static dimensions for Qualcomm AI Hub.

    Input  tensor name : 'image_tensor'   shape [1, 3, 224, 224]  float32
    Output tensor name : 'class_logits'   shape [1, 2]             float32

    Dynamic axes are intentionally disabled: Qualcomm AI Hub compilation
    requires fixed static batch and spatial dimensions.

    Parameters
    ----------
    model        : Trained nn.Module (will be set to .eval() internally).
    output_path  : File path for the exported .onnx file.
    opset_version: ONNX opset (17 recommended for TFLite downstream compat).
    """
    model.eval()
    model.cpu()

    # Dummy input matching the exact deployment shape
    dummy_input = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

    log.info("Exporting model to ONNX …")
    log.info("  Input  : image_tensor  [1, 3, 224, 224]  float32")
    log.info("  Output : class_logits  [1, 2]             float32")
    log.info("  Opset  : %d", opset_version)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image_tensor"],
        output_names=["class_logits"],
        dynamic_axes=None,          # Static dims required for Qualcomm AI Hub
        verbose=False,
    )

    log.info("ONNX export saved to '%s'.", output_path)

    # Quick shape sanity check
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        log.info("ONNX model check passed.")
    except ImportError:
        log.warning("onnx package not found; skipping model check.")
    except Exception as exc:
        log.error("ONNX check failed: %s", exc)


# ===========================================================================
# 6. Device Helper
# ===========================================================================

def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Using device: %s", device)
    return device


# ===========================================================================
# 7. Entrypoint
# ===========================================================================

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # CONFIG — edit these paths before running
    # -----------------------------------------------------------------------

    # Root folder with two class sub-folders (DRONE/ and NO_DRONE/)
    RAW_DATASET_DIR = "DATASET"

    # Where the 80/20 split will be written
    SPLIT_DATASET_DIR = "dataset"

    # Training hyper-parameters
    NUM_EPOCHS    = 20
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-2
    NUM_WORKERS   = 4           # set to 0 on Windows if multiprocessing errors occur

    # Output artefacts
    CHECKPOINT_PATH = "best_model.pth"
    ONNX_OUTPUT     = "efficientvit_l2_drone.onnx"

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------

    # Step 1: Auto-split raw dataset into train / val
    split_dataset(
        src_dir=RAW_DATASET_DIR,
        dest_dir=SPLIT_DATASET_DIR,
        split_ratio=0.2,
        seed=42,
    )

    # Step 2: Build DataLoaders
    train_loader, val_loader, idx_to_class = get_dataloaders(
        dataset_dir=SPLIT_DATASET_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Step 3: Instantiate model
    model = build_model(num_classes=2, pretrained=True)

    # Step 4: Train
    device = get_device()
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # Step 5: Export to ONNX
    export_to_onnx(
        model=model,
        output_path=ONNX_OUTPUT,
        opset_version=17,
    )

    log.info("All done. Artefacts: %s | %s", CHECKPOINT_PATH, ONNX_OUTPUT)