#!/usr/bin/env bash
# =============================================================================
# Alternative to writing a C++ app: use the qnn-net-run CLI tool that ships
# with QAIRT SDK to validate inference directly on the QCS6490 board.
#
# Run on: QCS6490 board (copy qnn-net-run + libs to the board first)
#
# QCS6490 HTP details:
#   Hexagon arch : v68
#   soc_id       : 35
# =============================================================================

set -euo pipefail

MODEL_DIR="."
BACKEND_LIB="${MODEL_DIR}/libQnnHtp.so"
CONTEXT_BIN="${MODEL_DIR}/resnet50_qcs6490.bin"
INPUT_LIST="${MODEL_DIR}/test_list.txt"
OUTPUT_DIR="${MODEL_DIR}/output_bin"

mkdir -p "$OUTPUT_DIR"

# adsprpcd (FastRPC daemon) must be running:
#   adsprpcd &
# On most QCS6490 builds it auto-starts via init.

echo "=== Running ResNet50 inference via qnn-net-run (HTP/NPU) ==="
./qnn-net-run \
    --backend "$BACKEND_LIB" \
    --retrieve_context "$CONTEXT_BIN" \
    --input_list "$INPUT_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --duration 0 \
    --perf_profile burst

echo ""
echo "Outputs written to $OUTPUT_DIR"
echo "Each .raw file = 1000 float32 logits (one per ImageNet class)"

# ── Post-process outputs with Python ─────────────────────────────────────────
python3 - <<'PY'
import os, glob, struct, numpy as np

output_dir   = "./output_bin"
labels_path  = "./imagenet_classes.txt"
labels       = open(labels_path).read().splitlines()

for raw_path in sorted(glob.glob(os.path.join(output_dir, "*.raw")))[:5]:
    data  = np.frombuffer(open(raw_path, "rb").read(), dtype=np.float32)
    # softmax
    exp   = np.exp(data - data.max())
    probs = exp / exp.sum()
    top5  = np.argsort(probs)[::-1][:5]

    print(f"\n── {os.path.basename(raw_path)} ──────────")
    for rank, idx in enumerate(top5, 1):
        lbl = labels[idx] if idx < len(labels) else "?"
        print(f"  {rank}. [{idx:4d}] {lbl:<40s} {probs[idx]:.4f}")
PY