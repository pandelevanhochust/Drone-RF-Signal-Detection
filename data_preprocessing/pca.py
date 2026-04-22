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

# ── KEY SETTINGS ──────────────────────────────────
FILES_PER_BATCH  = 5      # How many .h5 files to stack per partial_fit call
                           # RAM per batch ≈ FILES_PER_BATCH × 960 MB
                           # Increase if you have more RAM

N_PCA_COMPONENTS = 1999   # Must be < FILES_PER_BATCH × 400
                           # (5 files × 400 rows = 2000 → max is 1999)
                           # Update inp_shape in Classification.ipynb to match!
# ─────────────────────────────────────────────────

ROWS_PER_FILE = 400   # each .h5 file has 400 samples (5ms clips)

assert N_PCA_COMPONENTS < FILES_PER_BATCH * ROWS_PER_FILE, (
    f"N_PCA_COMPONENTS={N_PCA_COMPONENTS} must be < "
    f"FILES_PER_BATCH({FILES_PER_BATCH}) × ROWS_PER_FILE({ROWS_PER_FILE}) "
    f"= {FILES_PER_BATCH * ROWS_PER_FILE}"
)

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
total_samples = total_files * ROWS_PER_FILE
n_batches = (total_files + FILES_PER_BATCH - 1) // FILES_PER_BATCH  # ceiling division

print(f"\nTotal files:        {total_files}")
print(f"Total samples:      {total_samples}")
print(f"Files per batch:    {FILES_PER_BATCH}  (~{FILES_PER_BATCH * 960} MB RAM per batch)")
print(f"Total batches:      {n_batches}")
print(f"PCA components:     {N_PCA_COMPONENTS}")

# ─────────────────────────────────────────────────
# STEP 1 — PASS 1: Fit IncrementalPCA batch by batch
#
#   Accumulate FILES_PER_BATCH files into one array,
#   call partial_fit(), then discard the batch.
#   Peak RAM ≈ FILES_PER_BATCH × 960 MB
# ─────────────────────────────────────────────────
print()
print("=" * 60)
print(f"STEP 1 (Pass 1 of 2): Fitting IncrementalPCA")
print("=" * 60)

ipca = IncrementalPCA(n_components=N_PCA_COMPONENTS)

t0 = time.time()
for batch_start in range(0, total_files, FILES_PER_BATCH):
    batch = file_label_pairs[batch_start : batch_start + FILES_PER_BATCH]
    batch_num = batch_start // FILES_PER_BATCH + 1

    # Load and stack FILES_PER_BATCH files into one array
    chunks = []
    for fpath, label_idx in batch:
        df = pd.read_hdf(fpath, key='data')
        chunks.append(df.values)
        del df
    batch_X = np.vstack(chunks)   # shape: (FILES_PER_BATCH×400, 600000)
    del chunks

    ipca.partial_fit(batch_X)
    del batch_X

    fnames = [os.path.basename(p) for p, _ in batch]
    print(f"  [batch {batch_num:03d}/{n_batches}] fitted {len(batch)} files: {fnames[0]} … {fnames[-1]}")

print(f"\nFit complete in {time.time()-t0:.1f}s")
print(f"Variance explained: {ipca.explained_variance_ratio_.sum()*100:.2f}%")

# ─────────────────────────────────────────────────
# STEP 2 — PASS 2: Transform each file individually
#           and append results to output HDF5.
#
#   Transform is done one file at a time — minimal RAM.
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
    df = pd.read_hdf(fpath, key="data")           # (400, 600000)
    X_chunk = ipca.transform(df.values)        # (400, N_PCA_COMPONENTS)
    del df

    df_chunk = pd.DataFrame(X_chunk, columns=col_names)
    df_chunk["label"] = label_idx
    del X_chunk

    if first_write:
        df_chunk.to_hdf(PCA_OUTPUT_PATH, key="data", mode="w",
                        format="table", data_columns=["label"])
        first_write = False
    else:
        df_chunk.to_hdf(PCA_OUTPUT_PATH, key="data", mode="a",
                        append=True, format="table", data_columns=["label"])

    del df_chunk
    print(f"  [{i+1:03d}/{total_files}] saved: {os.path.basename(fpath)}"
          f"  rows {i*ROWS_PER_FILE}–{(i+1)*ROWS_PER_FILE-1}  (label {label_idx})")

print(f"\nTransform + save done in {time.time()-t0:.1f}s")
print(f"\n✅ Done!  {total_samples} samples × {N_PCA_COMPONENTS} components")
print(f"   Saved to: {PCA_OUTPUT_PATH}")
print()