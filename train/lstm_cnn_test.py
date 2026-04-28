"""
lstm_cnn_test.py
=================
Loads the trained CNN+LSTM model (lstm_cnn_model.keras) and the fitted
StandardScaler (lstm_cnn_scaler.pkl), then:

  1. Re-evaluates on the SAME held-out test split used during training
     (reproducible via the fixed RANDOM_SEED in lstm_cnn_train.py).
  2. Runs per-sample inference on a single sample or a small batch.
  3. Prints a full classification report and displays a confusion matrix.

Usage
─────
  # Full test-set evaluation
  python lstm_cnn_test.py

  # Inference on a specific sample index
  python lstm_cnn_test.py --sample 42
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← must match lstm_cnn_train.py
# ─────────────────────────────────────────────────────────────────────────────

PCA_PATH    = "/mnt/c/Users/navis/toanlv/core/pca_features.h5"   # same HDF5 used for training
MODEL_PATH  = "lstm_cnn_model.keras"                # saved by ModelCheckpoint
SCALER_PATH = "lstm_cnn_scaler.pkl"                 # fitted StandardScaler

TEST_SIZE   = 0.20
RANDOM_SEED = 42   # MUST match the value used in lstm_cnn_train.py

CLASS_NAMES = [
    "AIR_FY", "AIR_HO", "AIR_ON", "DIS_FY", "DIS_ON",
    "INS_FY", "INS_HO", "INS_ON", "MIN_FY", "MIN_HO",
    "MIN_ON", "MP1_FY", "MP1_HO", "MP1_ON", "MP2_FY",
    "MP2_HO", "MP2_ON", "PHA_FY", "PHA_HO", "PHA_ON",
]

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="CNN+LSTM inference / evaluation")
parser.add_argument(
    "--sample", type=int, default=None,
    help="If set, run inference on this single test-set index (0-based)."
)
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# 1. SANITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

for p, label in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler"), (PCA_PATH, "PCA HDF5")]:
    if not os.path.exists(p):
        print(f"[ERROR] {label} file not found: {p}")
        print("  → Run lstm_cnn_train.py first to produce the required files.")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODEL + SCALER
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading model and scaler …")
print("=" * 60)

model  = keras.models.load_model(MODEL_PATH)
model.summary()

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print(f"\n  Model  loaded  ← {MODEL_PATH}")
print(f"  Scaler loaded  ← {SCALER_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD & PREPARE DATA (identical split to training)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Loading PCA features …")
print("=" * 60)

df = pd.read_hdf(PCA_PATH, key="data")
X  = df.drop(columns=["label"]).values.astype(np.float32)
y  = df["label"].values.astype(np.int32)
del df

n_features = X.shape[1]   # 399
n_classes  = len(np.unique(y))

print(f"  X shape: {X.shape}  |  Classes: {n_classes}")

# Reproduce the exact same split
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)
del X

# Scale using the SAME scaler fitted during training
X_test_scaled = scaler.transform(X_test)

# Reshape for Conv1D input
X_test_3d = X_test_scaled.reshape(-1, n_features, 1)   # (N_test, 399, 1)

print(f"  X_test shape   : {X_test_3d.shape}")

# One-hot encode
y_test_cat = keras.utils.to_categorical(y_test, n_classes)

# ─────────────────────────────────────────────────────────────────────────────
# 4A.  SINGLE SAMPLE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

if args.sample is not None:
    idx = args.sample
    if idx < 0 or idx >= len(y_test):
        print(f"[ERROR] --sample {idx} is out of range (test set has {len(y_test)} samples).")
        sys.exit(1)

    sample = X_test_3d[idx : idx + 1]          # shape (1, 399, 1)
    probs  = model.predict(sample, verbose=0)   # shape (1, n_classes)
    pred   = int(np.argmax(probs, axis=1)[0])
    true   = int(y_test[idx])
    conf   = float(probs[0, pred]) * 100

    print("\n" + "=" * 60)
    print(f"  Sample index  : {idx}")
    print(f"  True label    : {true}  ({CLASS_NAMES[true] if true < len(CLASS_NAMES) else 'unknown'})")
    print(f"  Predicted     : {pred}  ({CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else 'unknown'})")
    print(f"  Confidence    : {conf:.2f}%")
    print(f"  Result        : {'✅ CORRECT' if pred == true else '❌ WRONG'}")
    print("=" * 60)

    # Show top-5 predictions
    top5_idx  = np.argsort(probs[0])[::-1][:5]
    print("\n  Top-5 predictions:")
    for rank, ci in enumerate(top5_idx, 1):
        name = CLASS_NAMES[ci] if ci < len(CLASS_NAMES) else str(ci)
        print(f"    {rank}. {name:12s}  {probs[0, ci] * 100:6.2f}%")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# 4B.  FULL TEST-SET EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Full test-set evaluation …")
print("=" * 60)

score = model.evaluate(X_test_3d, y_test_cat, verbose=1)
print(f"\n  Test loss     : {score[0]:.4f}")
print(f"  Test accuracy : {score[1] * 100:.2f}%")

y_pred      = model.predict(X_test_3d, verbose=1)
pred_labels = np.argmax(y_pred,    axis=1)
true_labels = np.argmax(y_test_cat, axis=1)

# Use class names that actually appear in the test set
present     = sorted(np.unique(true_labels))
target_names = [CLASS_NAMES[i] for i in present] if len(CLASS_NAMES) >= n_classes else None

print("\n  Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=target_names))

# ─────────────────────────────────────────────────────────────────────────────
# 5. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

cm   = confusion_matrix(true_labels, pred_labels)
fig, ax = plt.subplots(figsize=(13, 11))
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
ax.set_title(f"Confusion Matrix — CNN+LSTM  (Test Accuracy: {score[1]*100:.2f}%)")
plt.tight_layout()
plt.savefig("lstm_cnn_test_confusion_matrix.png", dpi=150)
plt.show()
print("\n  Confusion matrix saved → lstm_cnn_test_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PER-CLASS ACCURACY BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

per_class_acc = []
for ci in present:
    mask = true_labels == ci
    if mask.sum() > 0:
        per_class_acc.append(accuracy_score(true_labels[mask], pred_labels[mask]) * 100)
    else:
        per_class_acc.append(0.0)

fig2, ax2 = plt.subplots(figsize=(14, 5))
bars = ax2.bar(
    [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in present],
    per_class_acc,
    color="steelblue", edgecolor="white"
)
ax2.axhline(y=score[1] * 100, color="tomato", linestyle="--", label=f"Overall: {score[1]*100:.2f}%")
ax2.set_xlabel("Class")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Per-class Accuracy — CNN+LSTM (PCA Features)")
ax2.set_ylim(0, 105)
ax2.tick_params(axis="x", rotation=45)
ax2.legend()
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("lstm_cnn_per_class_accuracy.png", dpi=150)
plt.show()
print("  Per-class accuracy chart saved → lstm_cnn_per_class_accuracy.png")

print("\n✅ Evaluation complete.")
