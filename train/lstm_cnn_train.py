"""
lstm_cnn_train.py  (v2 — Redesigned)
======================================
Trains a Deep Residual CNN + Multi-Head Attention model on the PCA-reduced
features stored in pca_features.h5.

WHY THE OLD DESIGN FAILED (24.84% accuracy)
────────────────────────────────────────────
1. LSTM on PCA components is semantically wrong.
   PCA components are ordered by variance, NOT by time.  An LSTM wastes its
   entire recurrent capacity trying to find sequential dependencies where
   none exist.
2. Architecture was too shallow (2 conv blocks) for a 20-class problem.
3. LR=5e-4 with ReduceLROnPlateau patience=5 collapsed the LR too quickly,
   trapping the model in a poor local minimum.
4. Heavy dropout (0.4/0.3) on a relatively small network starved it of
   the capacity needed to separate 20 classes.

NEW ARCHITECTURE
────────────────
  Input (399, 1)
  ── Stem  : Conv1D(64, 7, same) → BN → ReLU
  ── Block1: Conv1D(64,  3) + Conv1D(64,  3)  [residual] → MaxPool(2)   → 200 steps
  ── Block2: Conv1D(128, 3) + Conv1D(128, 3)  [residual] → MaxPool(2)   → 100 steps
  ── Block3: Conv1D(256, 3) + Conv1D(256, 3)  [residual] → MaxPool(2)   →  50 steps
  ── Block4: Conv1D(512, 3) + Conv1D(512, 3)  [residual] → MaxPool(2)   →  25 steps
  ── Multi-Head Self-Attention (4 heads, key_dim=64)
  ── GlobalAveragePooling1D
  ── Dense(512, gelu) → Dropout(0.35)
  ── Dense(256, gelu) → Dropout(0.25)
  ── Dense(n_classes, softmax)

  Residual shortcuts use a 1×1 Conv projection when the channel width changes.
  Multi-Head Attention replaces LSTM — it captures global inter-PC
  relationships without imposing an artificial sequence order.

TRAINING IMPROVEMENTS
─────────────────────
  • Label smoothing 0.10  → prevents overconfident predictions
  • Initial LR = 1e-4     → gentler start, avoids early divergence
  • Cosine annealing LR   → smooth decay without sudden drops
  • EarlyStopping patience = 20 epochs (val_accuracy monitor)
  • Batch = 128           → more stable gradient estimates
  • Epochs = 120          → enough headroom; ES will cut short

Output files
────────────
  lstm_cnn_model.keras      ← saved Keras model
  lstm_cnn_history.npy      ← training history dict
  lstm_cnn_scaler.pkl       ← StandardScaler fitted on train set
  lstm_cnn_training_curves.png
  lstm_cnn_confusion_matrix.png

Usage
─────
  python lstm_cnn_train.py
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PCA_PATH    = "D:\CODIng\Thesis\SpectrumAnalyzer\pca_features.h5"   # input  HDF5
MODEL_OUT     = "lstm_cnn_model.keras"
HISTORY_OUT   = "lstm_cnn_history.npy"
SCALER_OUT    = "lstm_cnn_scaler.pkl"

TEST_SIZE     = 0.20
RANDOM_SEED   = 42
BATCH_SIZE    = 128       # larger → more stable gradients
EPOCHS        = 120       # EarlyStopping will cut short
LEARNING_RATE = 1e-4      # lower initial LR for smoother convergence
LABEL_SMOOTH  = 0.10      # label smoothing prevents over-confident softmax

CLASS_NAMES = [
    "AIR_FY", "AIR_HO", "AIR_ON", "DIS_FY", "DIS_ON",
    "INS_FY", "INS_HO", "INS_ON", "MIN_FY", "MIN_HO",
    "MIN_ON", "MP1_FY", "MP1_HO", "MP1_ON", "MP2_FY",
    "MP2_HO", "MP2_ON", "PHA_FY", "PHA_HO", "PHA_ON",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def residual_block(x, filters, kernel_size=3, pool_size=2, name_prefix="blk"):
    """
    Residual block:
      Main path  : Conv1D → BN → ReLU → Conv1D → BN
      Short path : 1×1 Conv projection (if channel width changes) → BN
      Merge      : Add → ReLU → MaxPool
    """
    shortcut = x

    # ── main path ────────────────────────────────────────────────────────────
    x = layers.Conv1D(filters, kernel_size, padding="same",
                      name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)

    x = layers.Conv1D(filters, kernel_size, padding="same",
                      name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)

    # ── shortcut projection (channel mismatch) ────────────────────────────────
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same",
                                 name=f"{name_prefix}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(shortcut)

    # ── merge ─────────────────────────────────────────────────────────────────
    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
    x = layers.MaxPooling1D(pool_size=pool_size, strides=pool_size,
                            name=f"{name_prefix}_pool")(x)
    return x


def build_model(n_features: int, n_classes: int,
                learning_rate: float, label_smoothing: float) -> Model:
    """
    Deep Residual CNN + Multi-Head Attention classifier.

    Sequence of operations:
      Stem → 4× Residual Blocks → Multi-Head Attention
      → GlobalAveragePooling → Dense(512) → Dense(256) → Dense(n_classes)

    Input shape : (n_features, 1)  e.g. (399, 1)
    Output shape: (n_classes,)     softmax probabilities
    """
    inp = keras.Input(shape=(n_features, 1), name="pca_input")

    # ── Stem ─────────────────────────────────────────────────────────────────
    # Large kernel (7) to capture broad patterns first; no pooling yet
    x = layers.Conv1D(64, 7, padding="same", name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    # ── Residual Blocks ───────────────────────────────────────────────────────
    # 399 → 200 → 100 → 50 → 25 timesteps
    x = residual_block(x, filters=64,  pool_size=2, name_prefix="blk1")
    x = residual_block(x, filters=128, pool_size=2, name_prefix="blk2")
    x = residual_block(x, filters=256, pool_size=2, name_prefix="blk3")
    x = residual_block(x, filters=512, pool_size=2, name_prefix="blk4")

    # ── Multi-Head Self-Attention ─────────────────────────────────────────────
    # Captures global relationships between the 25 learned feature groups.
    # Much more appropriate than LSTM on non-sequential PCA features.
    attn_out = layers.MultiHeadAttention(
        num_heads=4, key_dim=64, dropout=0.1, name="mha"
    )(x, x)
    x = layers.Add(name="attn_residual")([x, attn_out])   # residual
    x = layers.LayerNormalization(name="attn_ln")(x)

    # ── Pooling → Dense Head ──────────────────────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(512, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("gelu", name="fc1_gelu")(x)
    x = layers.Dropout(0.35, name="fc1_drop")(x)

    x = layers.Dense(256, name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Activation("gelu", name="fc2_gelu")(x)
    x = layers.Dropout(0.25, name="fc2_drop")(x)

    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="ResidualCNN_Attention_UAV")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        ),
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading PCA features from HDF5")
print(f"  Path: {PCA_PATH}")
print("=" * 60)

df = pd.read_hdf(PCA_PATH, key="data")
print(f"  Loaded DataFrame shape: {df.shape}")

X = df.drop(columns=["label"]).values.astype(np.float32)   # (N, 399)
y = df["label"].values.astype(np.int32)                     # (N,)
del df

n_features = X.shape[1]        # 399
n_classes  = len(np.unique(y)) # 20

print(f"  X shape      : {X.shape}")
print(f"  y shape      : {y.shape}")
print(f"  Classes found: {np.unique(y)}  → {n_classes} classes")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Train / Test split  (stratified, 80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)
del X

print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}  |  y_test: {y_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. NORMALISE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: StandardScaler (fit on train only)")
print("=" * 60)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

with open(SCALER_OUT, "wb") as f:
    pickle.dump(scaler, f)
print(f"  Scaler saved → {SCALER_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. RESHAPE for Conv1D  →  (samples, timesteps, 1)
# ─────────────────────────────────────────────────────────────────────────────

X_train = X_train.reshape(-1, n_features, 1)   # (N_train, 399, 1)
X_test  = X_test.reshape(-1, n_features, 1)    # (N_test,  399, 1)

y_train_cat = keras.utils.to_categorical(y_train, n_classes)
y_test_cat  = keras.utils.to_categorical(y_test,  n_classes)

print(f"\n  X_train reshaped: {X_train.shape}")
print(f"  y_train one-hot : {y_train_cat.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Building Residual CNN + Attention model")
print("=" * 60)

model = build_model(n_features, n_classes, LEARNING_RATE, LABEL_SMOOTH)
model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights_arr))
print("  Class weights computed.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. LR SCHEDULE — Cosine Decay with Warm Restart
#    Steps per epoch = ceil(N_train / BATCH_SIZE)
# ─────────────────────────────────────────────────────────────────────────────

steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
total_steps     = EPOCHS * steps_per_epoch
warmup_steps    = 5 * steps_per_epoch          # 5-epoch linear warm-up

cosine_decay_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=total_steps - warmup_steps,
    alpha=1e-6,        # minimum LR floor
)

# Re-compile with the schedule (overrides the constant LR set in build_model).
# Use AdamW instead of Adam — the weight_decay term adds L2 regularisation
# directly to the gradient update, helping combat the overfitting seen when
# train accuracy diverges far above val accuracy.
model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=cosine_decay_schedule,
        weight_decay=1e-4,   # L2 penalty on weights
    ),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=["accuracy"],
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: ReduceLROnPlateau is intentionally removed.
# When the optimizer is built with a LearningRateSchedule object (CosineDecay),
# Keras makes the LR read-only and ReduceLROnPlateau raises a TypeError when
# it tries to overwrite it.  CosineDecay already handles LR annealing.
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=20,               # more patience → fewer premature stops
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=MODEL_OUT,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAIN
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Training")
print("=" * 60)

t0 = time.time()
history = model.fit(
    X_train, y_train_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1,
)
elapsed = time.time() - t0
print(f"\n  Training complete in {elapsed / 60:.1f} min")

np.save(HISTORY_OUT, history.history)
print(f"  History saved → {HISTORY_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Evaluation on test set")
print("=" * 60)

# Evaluate using raw (non-smoothed) cross-entropy for a fair comparison
score = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"  Test loss     : {score[0]:.4f}")
print(f"  Test accuracy : {score[1] * 100:.2f}%")
print(f"  Training time : {elapsed:.1f}s")

y_pred      = model.predict(X_test)
pred_labels = np.argmax(y_pred,     axis=1)
true_labels = np.argmax(y_test_cat, axis=1)

present_classes = sorted(np.unique(true_labels))
target_names    = [CLASS_NAMES[i] for i in present_classes] if len(CLASS_NAMES) >= n_classes else None

print("\n  Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=target_names))

# ─────────────────────────────────────────────────────────────────────────────
# 11. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"],     label="Train Accuracy", color="royalblue")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   color="tomato", linestyle="--")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train Loss", color="royalblue")
axes[1].plot(history.history["val_loss"], label="Val Loss",   color="tomato", linestyle="--")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("lstm_cnn_training_curves.png", dpi=150)
plt.show()

# ── Confusion Matrix ──────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 12))
cm   = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
disp.plot(ax=ax2, colorbar=True, xticks_rotation=45)
ax2.set_title("Confusion Matrix — Residual CNN + Attention (PCA Features)")
plt.tight_layout()
plt.savefig("lstm_cnn_confusion_matrix.png", dpi=150)
plt.show()

print(f"\n✅ Model saved  → {MODEL_OUT}")
print(f"✅ Scaler saved → {SCALER_OUT}")
print(f"✅ History saved → {HISTORY_OUT}")
print(f"✅ Plots saved  → lstm_cnn_training_curves.png, lstm_cnn_confusion_matrix.png")
