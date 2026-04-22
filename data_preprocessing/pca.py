"""
pca_build_dataset.py
====================
This script fills the gap between data_preprocessing.ipynb and Classification.ipynb.

It:
  1. Loops over every class folder in your HDF5 output directory
  2. Loads each .h5 file and stacks the samples into one big matrix X
  3. Assigns an integer label y to each sample based on its folder (class)
  4. Applies PCA to compress 600,000 features → 6,000 components
  5. Saves the PCA-reduced X and labels y to a single file: pca_features.h5

Run this AFTER data_preprocessing.ipynb has finished writing all .h5 files.
Then open Classification.ipynb and load pca_features.h5.

Author: Auto-generated gap-fill for DroneDetect_V2 pipeline
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import time

# ─────────────────────────────────────────────────
# CONFIG — adjust these two paths for your machine
# ─────────────────────────────────────────────────

# Path where data_preprocessing.ipynb saved the .h5 files
HDF5_INPUT_DIR = "C:/Users/navis/toanlv/OutputHdf5/"
PCA_OUTPUT_PATH = "C:/Users/navis/toanlv/core/pca_features.h5"

# Number of PCA components — must match n_outputs logic in Classification.ipynb
N_PCA_COMPONENTS = 6000


FOLDERS = [
    "AIR_FY", "AIR_HO", "AIR_ON", "DIS_FY", "DIS_ON", "INS_FY", "INS_HO", "INS_ON",
    "MIN_FY", "MIN_HO", "MIN_ON", "MP1_FY", "MP1_HO", "MP1_ON", "MP2_FY", "MP2_HO",
    "MP2_ON", "PHA_FY", "PHA_HO", "PHA_ON"
]

# ─────────────────────────────────────────────────
# STEP 1 — Load all .h5 files and build X, y
# ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading HDF5 files")
print("=" * 60)

all_X = []   # list of numpy arrays, each (400, 600000)
all_y = []   # list of integer labels

for label_idx, folder in enumerate(FOLDERS):
    folder_path = os.path.join(HDF5_INPUT_DIR, folder)

    if not os.path.exists(folder_path):
        print(f"  [SKIP] Folder not found: {folder_path}")
        continue

    h5_files = [f for f in os.listdir(folder_path) if f.endswith(".h5")]
    if not h5_files:
        print(f"  [SKIP] No .h5 files in: {folder_path}")
        continue

    print(f"  [{label_idx:02d}] {folder} — {len(h5_files)} file(s)")

    for h5_file in sorted(h5_files):
        full_path = os.path.join(folder_path, h5_file)
        df = pd.read_hdf(full_path, key="droneV2_data")   # shape: (400, 600000)
        all_X.append(df.values)                            # numpy array
        all_y.extend([label_idx] * len(df))               # 400 copies of this label

print()
print(f"  Total files loaded: {sum(len([f for f in os.listdir(os.path.join(HDF5_INPUT_DIR, folder)) if f.endswith('.h5')]) for folder in FOLDERS if os.path.exists(os.path.join(HDF5_INPUT_DIR, folder)))}")

# Stack into one matrix
X = np.vstack(all_X)          # shape: (N_total_samples, 600000)
y = np.array(all_y)           # shape: (N_total_samples,)

print(f"\n  X shape: {X.shape}  — samples × raw features")
print(f"  y shape: {y.shape}  — integer class labels")
print(f"  Unique classes: {np.unique(y)}")

# ─────────────────────────────────────────────────
# STEP 2 — Apply PCA
# ─────────────────────────────────────────────────
print()
print("=" * 60)
print(f"STEP 2: Applying PCA (n_components={N_PCA_COMPONENTS})")
print("        This may take several minutes for large datasets…")
print("=" * 60)

t0 = time.time()
pca = PCA(n_components=N_PCA_COMPONENTS)
X_pca = pca.fit_transform(X)   # shape: (N_total_samples, 6000)
elapsed = time.time() - t0

variance_explained = pca.explained_variance_ratio_.sum()
print(f"\n  Done in {elapsed:.1f}s")
print(f"  X_pca shape:          {X_pca.shape}")
print(f"  Variance explained:   {variance_explained * 100:.2f}%")

# ─────────────────────────────────────────────────
# STEP 3 — Save to HDF5
# ─────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3: Saving PCA features to HDF5")
print("=" * 60)

# Build a clean DataFrame: columns 0..5999 are PCA components, 'label' is class
df_pca = pd.DataFrame(X_pca, columns=[f"pc{i}" for i in range(N_PCA_COMPONENTS)])
df_pca["label"] = y

print(f"  Output DataFrame shape: {df_pca.shape}")
print(f"  Saving to: {PCA_OUTPUT_PATH}")

df_pca.to_hdf(PCA_OUTPUT_PATH, key="data", mode="w")

print("\n  ✅ Saved successfully!")


