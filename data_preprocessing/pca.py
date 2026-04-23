import os
import h5py
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

# ─────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────

# Linux path
HDF5_INPUT_DIR  = "/mnt/c/Users/navis/toanlv/OutputHdf5/"
PCA_OUTPUT_PATH = "/mnt/c/Users/navis/toanlv/core/pca_features.h5"

# Windows path (uncomment if running on Windows directly)
# HDF5_INPUT_DIR  = "C:/Users/navis/toanlv/OutputHdf5/"
# PCA_OUTPUT_PATH = "C:/Users/navis/toanlv/core/pca_features.h5"

# PCA SETTINGS
# Since each file has 400 rows, n_components must be <= 400.
# 399 allows us to process one file at a time without crashing.
N_PCA_COMPONENTS = 399
ROWS_PER_FILE = 400

FOLDERS = [
    "AIR_FY", "AIR_HO", "AIR_ON", "DIS_FY", "DIS_ON",
    "INS_FY", "INS_HO", "INS_ON", "MIN_FY", "MIN_HO",
    "MIN_ON", "MP1_FY", "MP1_HO", "MP1_ON", "MP2_FY",
    "MP2_HO", "MP2_ON", "PHA_FY", "PHA_HO", "PHA_ON",
]


# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────

def get_hdf5_key(path):
    """Detects the internal key of the HDF5 file automatically."""
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
    if not keys:
        raise ValueError(f"No keys found in {path}")
    return keys[0]


# ─────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────

file_label_pairs = []

# Collect all files and assign labels based on folder index
for label_idx, folder in enumerate(FOLDERS):
    folder_path = os.path.join(HDF5_INPUT_DIR, folder)
    if not os.path.exists(folder_path):
        print(f"[SKIP] Folder not found: {folder_path}")
        continue

    h5_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".h5")])
    if not h5_files:
        print(f"[SKIP] No .h5 files in: {folder_path}")
        continue

    for h5_file in h5_files:
        file_label_pairs.append((os.path.join(folder_path, h5_file), label_idx))
    print(f"  [label {label_idx:02d}] {folder} — {len(h5_files)} file(s)")

total_files = len(file_label_pairs)
print(f"\nTotal files:     {total_files}")
print(f"Total samples:   {total_files * ROWS_PER_FILE}")
print(f"PCA components:  {N_PCA_COMPONENTS}")

# ─────────────────────────────────────────────────
# STEP 1: Fit IncrementalPCA (Pass 1)
# ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Fitting IncrementalPCA (Learning the patterns)")
print("=" * 60)

ipca = IncrementalPCA(n_components=N_PCA_COMPONENTS)

t0 = time.time()
for i, (fpath, label_idx) in enumerate(file_label_pairs):
    try:
        key = get_hdf5_key(fpath)
        df = pd.read_hdf(fpath, key=key)

        # Fit incrementally using current file data
        ipca.partial_fit(df.values)

        del df  # Clear RAM immediately
        print(f"  [{i + 1:03d}/{total_files}] fitted: {os.path.basename(fpath)}")
    except Exception as e:
        print(f"  [!] Error fitting {os.path.basename(fpath)}: {e}")

print(f"\nFit complete in {time.time() - t0:.1f}s")
print(f"Variance explained: {ipca.explained_variance_ratio_.sum() * 100:.2f}%")

# ─────────────────────────────────────────────────
# STEP 2: Transform and Save (Pass 2)
# ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Transforming and saving to HDF5")
print(f"Output: {PCA_OUTPUT_PATH}")
print("=" * 60)

col_names = [f"pc{i}" for i in range(N_PCA_COMPONENTS)]
first_write = True

t0 = time.time()
for i, (fpath, label_idx) in enumerate(file_label_pairs):
    try:
        key = get_hdf5_key(fpath)
        df = pd.read_hdf(fpath, key=key)

        # Transform 600,000 features down to 399
        X_chunk = ipca.transform(df.values)
        del df

        # Create output DataFrame with label
        df_out = pd.DataFrame(X_chunk, columns=col_names)
        df_out["label"] = label_idx
        del X_chunk

        if first_write:
            # Create fresh file
            df_out.to_hdf(PCA_OUTPUT_PATH, key="data", mode="w",
                          format="table", data_columns=["label"])
            first_write = False
        else:
            # Append to existing file
            df_out.to_hdf(PCA_OUTPUT_PATH, key="data", mode="a",
                          append=True, format="table", data_columns=["label"])

        del df_out
        print(f"  [{i + 1:03d}/{total_files}] saved: {os.path.basename(fpath)}")

    except Exception as e:
        print(f"  [!] Error transforming {os.path.basename(fpath)}: {e}")

print(f"\n✅ All done! PCA features saved to: {PCA_OUTPUT_PATH}")