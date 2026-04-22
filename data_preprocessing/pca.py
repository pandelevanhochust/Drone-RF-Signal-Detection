import os
import h5py
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

# ─────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────

# Linux path (comment out and use Windows path below if needed)
HDF5_INPUT_DIR  = "/mnt/c/Users/navis/toanlv/OutputHdf5/"
PCA_OUTPUT_PATH = "/mnt/c/Users/navis/toanlv/core/pca_features.h5"

# Windows path (uncomment if running on Windows directly)
# HDF5_INPUT_DIR  = "C:/Users/navis/toanlv/OutputHdf5/"
# PCA_OUTPUT_PATH = "C:/Users/navis/toanlv/core/pca_features.h5"

N_PCA_COMPONENTS = 6000   # compressed feature size — must match Classification.ipynb

FOLDERS = [
    "AIR_FY", "AIR_HO", "AIR_ON", "DIS_FY", "DIS_ON",
    "INS_FY", "INS_HO", "INS_ON", "MIN_FY", "MIN_HO",
    "MIN_ON", "MP1_FY", "MP1_HO", "MP1_ON", "MP2_FY",
    "MP2_HO", "MP2_ON", "PHA_FY", "PHA_HO", "PHA_ON",
]

# ─────────────────────────────────────────────────
# Build the full ordered list of (file_path, label_idx)
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
print(f"\nTotal files to process: {total_files}")
print(f"Estimated samples:      {total_files * 400}  (400 per file)")

# ─────────────────────────────────────────────────
# STEP 1 — PASS 1: Fit IncrementalPCA file by file
#
#   partial_fit() updates the PCA model with one batch at a time
#   and immediately discards the data from RAM (del df).
#   Peak RAM ≈ ~960 MB (one file).
# ─────────────────────────────────────────────────
print()
print("=" * 60)
print(f"STEP 1 (Pass 1 of 2): Fitting IncrementalPCA")
print(f"  n_components = {N_PCA_COMPONENTS}")
print("=" * 60)

ipca = IncrementalPCA(n_components=N_PCA_COMPONENTS)

t0 = time.time()
for i, (fpath, label_idx) in enumerate(file_label_pairs):
    df = pd.read_hdf(fpath, key="data")     # (400, 600000)
    ipca.partial_fit(df.values)          # update PCA incrementally — no accumulation
    del df                               # free ~960 MB immediately
    print(f"  [{i+1:03d}/{total_files}] fitted: {os.path.basename(fpath)}  (label {label_idx})")

print(f"\nFit complete in {time.time()-t0:.1f}s")
print(f"Variance explained: {ipca.explained_variance_ratio_.sum()*100:.2f}%")

# ─────────────────────────────────────────────────
# STEP 2 — PASS 2: Transform each file and append
#           results to the output HDF5 one chunk at a time.
#
#   Each file is loaded, transformed to (400, 6000), saved,
#   then discarded — peak RAM ≈ ~2 × 960 MB.
# ─────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2 (Pass 2 of 2): Transforming and saving to HDF5")
print(f"  Output: {PCA_OUTPUT_PATH}")
print("=" * 60)

col_names = [f"pc{i}" for i in range(N_PCA_COMPONENTS)]
first_write = True

t0 = time.time()
for i, (fpath, label_idx) in enumerate(file_label_pairs):
    df = pd.read_hdf(fpath, key="data")          # (400, 600000)
    X_chunk = ipca.transform(df.values)       # (400, 6000)
    del df

    df_chunk = pd.DataFrame(X_chunk, columns=col_names)
    df_chunk["label"] = label_idx
    del X_chunk

    if first_write:
        # mode="w" creates (or overwrites) the output file
        df_chunk.to_hdf(PCA_OUTPUT_PATH, key="data", mode="w",
                        format="table", data_columns=["label"])
        first_write = False
    else:
        # mode="a" + append=True adds rows to the existing table
        df_chunk.to_hdf(PCA_OUTPUT_PATH, key="data", mode="a",
                        append=True, format="table", data_columns=["label"])

    del df_chunk
    print(f"  [{i+1:03d}/{total_files}] saved: {os.path.basename(fpath)}"
          f"  rows {i*400}–{(i+1)*400-1}  (label {label_idx})")

print(f"\nTransform + save done in {time.time()-t0:.1f}s")
print(f"\n✅ Done! PCA features saved to: {PCA_OUTPUT_PATH}")