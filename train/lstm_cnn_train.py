"""
lstm_cnn_train.py
==================
Trains a CNN + LSTM model on the PCA-reduced features stored in pca_features.h5.

Architecture (mirrors the "LSTM + CNN based model" from Classification.ipynb,
adapted for 399 PCA features instead of 6 000 raw features):

    Input (399, 1)
    ─── LFLB1: Conv1D(128, 3) → BN → ELU → MaxPool1D(2)
    ─── LFLB2: Conv1D(128, 3) → BN → ELU → MaxPool1D(2)
    ─── LFLB3: Conv1D(128, 3) → BN → ELU → MaxPool1D(2)
    ─── LSTM(64)
    ─── Dropout(0.3)
    ─── Dense(64, relu)
    ─── Dropout(0.3)
    ─── Dense(n_classes, softmax)

Output files
────────────
  lstm_cnn_model.keras      ← saved Keras model (for inference)
  lstm_cnn_history.npy      ← training history dict  (for later plotting)
  lstm_cnn_scaler.pkl       ← StandardScaler fitted on train set

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
from keras.models import Sequential
from keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization,
    LSTM, Dense, Dropout, Input
)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these paths to match your environment
# ─────────────────────────────────────────────────────────────────────────────

PCA_PATH       = "/mnt/c/Users/navis/toanlv/core/pca_features.h5"  # input  HDF5
MODEL_OUT      = "lstm_cnn_model.keras"                # saved model
HISTORY_OUT    = "lstm_cnn_history.npy"                # training history
SCALER_OUT     = "lstm_cnn_scaler.pkl"                 # StandardScaler

# Training hyper-parameters
TEST_SIZE      = 0.20
RANDOM_SEED    = 42
BATCH_SIZE     = 32
EPOCHS         = 30
LEARNING_RATE  = 1e-3

# Class names matching the FOLDERS list in pca_build_dataset.py
CLASS_NAMES = [
    "AIR_FY", "AIR_HO", "AIR_ON", "DIS_FY", "DIS_ON",
    "INS_FY", "INS_HO", "INS_ON", "MIN_FY", "MIN_HO",
    "MIN_ON", "MP1_FY", "MP1_HO", "MP1_ON", "MP2_FY",
    "MP2_HO", "MP2_ON", "PHA_FY", "PHA_HO", "PHA_ON",
]

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

n_features  = X.shape[1]          # 399
n_classes   = len(np.unique(y))   # should be 20

print(f"  X shape      : {X.shape}")
print(f"  y shape      : {y.shape}")
print(f"  Classes found: {np.unique(y)}  → {n_classes} classes")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Train / Test split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y      # keeps class proportions equal in both splits
)
del X

print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}  |  y_test: {y_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. NORMALISE (StandardScaler)
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

# One-hot encode labels
y_train_cat = keras.utils.to_categorical(y_train, n_classes)
y_test_cat  = keras.utils.to_categorical(y_test,  n_classes)

print(f"\n  X_train reshaped: {X_train.shape}")
print(f"  y_train one-hot : {y_train_cat.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD MODEL — CNN (4× LFLB) + LSTM
# ─────────────────────────────────────────────────────────────────────────────
#
#  We use 4 Local Feature Learning Blocks (LFLB) each containing:
#      Conv1D → BatchNorm → ELU activation → MaxPool1D
#  followed by an LSTM layer to capture sequential dependencies,
#  then fully-connected head.
#
#  Pool sizes are halved (2 instead of 4) because the PCA input (399)
#  is much shorter than the original 6 000-feature sequences.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Building CNN+LSTM model")
print("=" * 60)

inp_shape = (n_features, 1)   # (399, 1)

model = Sequential(name="CNN_LSTM_UAV")

# ── LFLB 1 ──────────────────────────────────────────────────────────────────
model.add(Input(shape=inp_shape))
model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding="same"))
model.add(BatchNormalization())
model.add(keras.layers.Activation("elu"))
model.add(MaxPooling1D(pool_size=2, strides=2))

# ── LFLB 2 ──────────────────────────────────────────────────────────────────
model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding="same"))
model.add(BatchNormalization())
model.add(keras.layers.Activation("elu"))
model.add(MaxPooling1D(pool_size=2, strides=2))

# ── LFLB 3 ──────────────────────────────────────────────────────────────────
model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding="same"))
model.add(BatchNormalization())
model.add(keras.layers.Activation("elu"))
model.add(MaxPooling1D(pool_size=2, strides=2))

# ── LFLB 4 ──────────────────────────────────────────────────────────────────
model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding="same"))
model.add(BatchNormalization())
model.add(keras.layers.Activation("elu"))
model.add(MaxPooling1D(pool_size=2, strides=2))

# ── LSTM ─────────────────────────────────────────────────────────────────────
model.add(LSTM(units=64))
model.add(Dropout(0.3))

# ── Fully Connected Head ──────────────────────────────────────────────────────
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(n_classes, activation="softmax"))

opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)
model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=MODEL_OUT,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAIN
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
    verbose=1
)
elapsed = time.time() - t0
print(f"\n  Training complete in {elapsed / 60:.1f} min")

# Save history for later plotting
np.save(HISTORY_OUT, history.history)
print(f"  History saved → {HISTORY_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. QUICK EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Evaluation on test set")
print("=" * 60)

score = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"  Test loss     : {score[0]:.4f}")
print(f"  Test accuracy : {score[1] * 100:.2f}%")
print(f"  Training time : {elapsed:.1f}s")

y_pred     = model.predict(X_test)
pred_labels = np.argmax(y_pred,      axis=1)
true_labels = np.argmax(y_test_cat,  axis=1)

# Use only class names that actually appear in y_test
present_classes = sorted(np.unique(true_labels))
target_names    = [CLASS_NAMES[i] for i in present_classes] if len(CLASS_NAMES) >= n_classes else None

print("\n  Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=target_names))

# ─────────────────────────────────────────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Accuracy curve
axes[0].plot(history.history["accuracy"],     label="Train Accuracy", color="royalblue")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   color="tomato",  linestyle="--")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

# ── Loss curve
axes[1].plot(history.history["loss"],     label="Train Loss", color="royalblue")
axes[1].plot(history.history["val_loss"], label="Val Loss",   color="tomato",  linestyle="--")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("lstm_cnn_training_curves.png", dpi=150)
plt.show()

# ── Confusion Matrix
fig2, ax2 = plt.subplots(figsize=(12, 10))
cm   = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
disp.plot(ax=ax2, colorbar=True, xticks_rotation=45)
ax2.set_title("Confusion Matrix — CNN+LSTM (PCA Features)")
plt.tight_layout()
plt.savefig("lstm_cnn_confusion_matrix.png", dpi=150)
plt.show()

print(f"\n✅ Model saved  → {MODEL_OUT}")
print(f"✅ Scaler saved → {SCALER_OUT}")
print(f"✅ History saved → {HISTORY_OUT}")
print(f"✅ Plots saved  → lstm_cnn_training_curves.png, lstm_cnn_confusion_matrix.png")
