"""
eval_roc.py
=============================================================================
Evaluates the trained 3-Class model and plots a Receiver Operating Characteristic
(ROC) curve using One-vs-Rest (OvR) multiclass tracking metrics.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Reuse your exact architecture definition from the training script
from new_train2 import DroneClassifier, get_dataloaders

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

def generate_roc_evaluation(checkpoint_path="best_model.pth", dataset_dir="dataset_split"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading checkpoint context using: {device}")

    # 1. Load the parameters and data targets
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing state checkpoint binary: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get("num_classes", 3)
    class_names = ckpt.get("class_names", ["DRONE", "DRONE_SIGNAL", "NO_DRONE"])

    # Get clean, un-augmented validation dataloader
    _, val_loader, _ = get_dataloaders(dataset_dir=dataset_dir, batch_size=16, img_h=256, img_w=512)

    # 2. Instantiate and mount model weights
    model = DroneClassifier(num_classes=num_classes, in_channels=3)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    all_labels = []
    all_probs = []

    # 3. Extract model raw probabilities over validation sweep arrays
    log.info("Gathering model predictions across validation dataset arrays...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1) # Convert outputs to valid probabilities

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_score = np.array(all_probs)

    # Binarize labels for One-vs-Rest multiclass tracking evaluation
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

    # 4. Compute ROC metrics and draw curves
    plt.figure(figsize=(9, 7), dpi=300)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr,
            color=colors[i],
            lw=2.5,
            label=f"Class {class_names[i]} (AUC = {roc_auc:.4f})"
        )
        log.info(f"  Class {class_names[i]} -> Computed Area Under Curve (AUC) = {roc_auc:.4f}")

    # Plot baseline reference line diagonal
    plt.plot([0, 1], [0, 1], color="darkgrey", lw=1.5, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("True Positive Rate (TPR)", fontsize=11, fontweight="bold", labelpad=10)
    plt.title("Multiclass Receiver Operating Characteristic (ROC) — 3-Class Update", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle=":", alpha=0.6)

    # Save image plot to disk storage channel link
    output_image = "roc_curve.png"
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()

    log.info(f"✓ ROC curve evaluation image plot successfully written to disk → '{output_image}'")

if __name__ == "__main__":
    generate_roc_evaluation()