import os
import h5py
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

# ─────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────

# Linux path
HDF5_INPUT_DIR  = "/mnt/c/Users/navis/toanlv/OutputHdf5/"
PCA_OUTPUT_PATH = "/mnt/c/Users/navis/toanlv/core/pca_features.h5"

# Windows path (uncomment if running on Windows directly)
# HDF5_INPUT_DIR  = "C:/Users/navis/toanlv/OutputHdf5/"
# PCA_OUTPUT_PATH = "C:/Users/navis/toanlv/core/pca_features.h5"

# ─────────────────────────────────────────────────
# MEMORY NOTE:
#   Each .h5 file = 400 rows × 600,000 cols × 4 bytes ≈ 960 MB
#   This script loads ONE file at a time → peak RAM ≈ ~2 GB ✅
#
#   IncrementalPCA constraint: n_components < rows per partial_fit call
#   One file = 400 rows → N_PCA_COMPONENTS must be < 400
# ─────────────────────────────────────────────────
N_PCA_COMPONENTS = 399   # max safe value for 400-row files
ROWS_PER_FILE    = 400

FOLDERS = [
    "AIR_FY", "AIR_HO", "AIR_ON", "DIS_FY", "DIS_ON",
    "INS_FY", "INS_HO", "INS_ON", "MIN_FY", "MIN_HO",
    "MIN_ON", "MP1_FY", "MP1_HO", "MP1_ON", "MP2_FY",
    "MP2_HO", "MP2_ON", "PHA_FY", "PHA_HO", "PHA_ON",
]

# ─────────────────────────────────────────────────
# HELPER — auto-detect the key stored in an .h5 file
# ─────────────────────────────────────────────────
def get_hdf5_key(path):
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
    if not keys:
        raise ValueError(f"No keys found in {path}")
    return keys[0]

# ─────────────────────────────────────────────────
# Build the ordered list of (file_path, label_idx)
# ─────────────────────────────────────────────────
file_label_pairs = []

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
print(f"Peak RAM/step:   ~{ROWS_PER_FILE * 600000 * 4 / 1e9 * 2:.1f} GB  (one file at a time)")

# ─────────────────────────────────────────────────
# STEP 1 — PASS 1: Fit IncrementalPCA
#
#   ONE file loaded → partial_fit() called → file deleted → next file
#   No stacking, no doubling. Peak RAM stays at ~2 GB throughout.
# ─────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 1 (Pass 1 of 2): Fitting IncrementalPCA")
print(f"  One file at a time — peak RAM ≈ ~2 GB")
print("=" * 60)

ipca = IncrementalPCA(n_components=N_PCA_COMPONENTS)

t0 = time.time()
for i, (fpath, label_idx) in enumerate(file_label_pairs):
    key = get_hdf5_key(fpath)           # auto-detect key (e.g. 'droneV2_data')
    df  = pd.read_hdf(fpath, key=key)  # load one file: (400, 600000) ≈ 960 MB
    ipca.partial_fit(df.values)         # update PCA model incrementally
    del df                              # free RAM before loading next file
    print(f"  [{i+1:03d}/{total_files}] fitted: {os.path.basename(fpath)}"
          f"  (label {label_idx})")

print(f"\nFit complete in {time.time()-t0:.1f}s")
print(f"Variance explained: {ipca.explained_variance_ratio_.sum()*100:.2f}%")

# ─────────────────────────────────────────────────
# STEP 2 — PASS 2: Transform each file and save
#
#   Again, one file at a time → (400, 600000) → (400, 399)
#   Append each chunk to the output HDF5 table.
#   Peak RAM ≈ ~2 GB.
# ─────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2 (Pass 2 of 2): Transforming and saving to HDF5")
print(f"  Output: {PCA_OUTPUT_PATH}")
print("=" * 60)

col_names   = [f"pc{i}" for i in range(N_PCA_COMPONENTS)]
first_write = True

t0 = time.time()
for i, (fpath, label_idx) in enumerate(file_label_pairs):
    key     = get_hdf5_key(fpath)
    df      = pd.read_hdf(fpath, key=key)      # (400, 600000)
    X_chunk = ipca.transform(df.values)        # (400, 399)
    del df                                     # free big matrix immediately

    df_out          = pd.DataFrame(X_chunk, columns=col_names)
    df_out["label"] = label_idx
    del X_chunk

    if first_write:
        df_out.to_hdf(PCA_OUTPUT_PATH, key="data", mode="w",
                      format="table", data_columns=["label"])
        first_write = False
    else:
        df_out.to_hdf(PCA_OUTPUT_PATH, key="data", mode="a",
                      append=True, format="table", data_columns=["label"])
    del df_out

    print(f"  [{i+1:03d}/{total_files}] saved: {os.path.basename(fpath)}"
          f"  rows {i*ROWS_PER_FILE}–{(i+1)*ROWS_PER_FILE-1}  (label {label_idx})")

print(f"\nTransform + save done in {time.time()-t0:.1f}s")
print(f"\n✅ Done!  {total_files * ROWS_PER_FILE} samples × {N_PCA_COMPONENTS} components")
print(f"   Saved to: {PCA_OUTPUT_PATH}")
print()
print("=" * 60)
print("Update Classification.ipynb:")
print(f"  inp_shape  = ({N_PCA_COMPONENTS},)")
print(f"  n_features = {N_PCA_COMPONENTS}")
print("=" * 60)