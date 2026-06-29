#!/usr/bin/env bash
# =============================================================================
# Step 3: Convert split YOLOv8 ONNX → quantized DLC → Context-Binary (.bin)
#         for QCS6490 Hexagon HTP v68
#
# Run on: x86 Linux host (Ubuntu 22.04) with QAIRT SDK installed
# Prereqs: source $QAIRT_ROOT/bin/envsetup.sh
# =============================================================================

set -euo pipefail

QAIRT_ROOT="${QAIRT_ROOT:?Set QAIRT_ROOT to your SDK root}"
MODEL_NAME="yolov8n"
ONNX_MODEL="${MODEL_NAME}_split.onnx"   # output of 2_split_yolo_output.py
CALIB_DIR="./data/calib_yolo"           # calibration .raw images (1×3×640×640 f32)

echo "=== QAIRT_ROOT: $QAIRT_ROOT ==="
source "$QAIRT_ROOT/bin/envsetup.sh"

# ── 3a. ONNX → float DLC ─────────────────────────────────────────────────────
echo "[3a] Converting ONNX → DLC (float) ..."
qairt-converter \
    --input_network "$ONNX_MODEL" \
    --input_dim images "1,3,640,640" \
    --output_path "${MODEL_NAME}.dlc" \
    --float_bias_bw 32

# ── 3b. Calibration list ─────────────────────────────────────────────────────
echo "[3b] Generating calibration file list ..."
find "$CALIB_DIR" -name "*.raw" | sort > calib_list_yolo.txt
echo "    Found $(wc -l < calib_list_yolo.txt) calibration samples"

# ── 3c. Quantize → INT8 (w8a8) for HTP ───────────────────────────────────────
echo "[3c] Quantizing DLC → INT8 ..."
qairt-quantizer \
    --input_dlc "${MODEL_NAME}.dlc" \
    --input_list calib_list_yolo.txt \
    --output_dlc "${MODEL_NAME}_quantized.dlc" \
    --act_bitwidth 8 \
    --weights_bitwidth 8 \
    --bias_bitwidth 32 \
    --use_per_channel_quantization \
    --use_native_input_files \
    --algorithms cle                       # cross-layer equalisation — improves
                                            # YOLO accuracy post-quantization

# ── 3d. Context-Binary for QCS6490 (v68, soc_id=35) ──────────────────────────
echo "[3d] Generating Context-Binary for QCS6490 ..."

cat > htp_config_qcs6490.json <<'JSON'
{
  "graphs": [
    {
      "graph_names": ["yolov8n"],
      "vtcm_mb": 4
    }
  ],
  "devices": [
    {
      "dsp_arch": "v68",
      "soc_id": 35
    }
  ]
}
JSON

mkdir -p context_bin
qnn-context-binary-generator \
    --backend "$QAIRT_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --model   "${MODEL_NAME}_quantized.dlc" \
    --output_dir context_bin/ \
    --binary_file "${MODEL_NAME}_qcs6490" \
    --backend_config htp_config_qcs6490.json

echo ""
echo "✓ Context-Binary ready: context_bin/${MODEL_NAME}_qcs6490.bin"
echo "  Deploy with: libQnnHtp.so, libQnnHtpV68Stub.so, libQnnHtpV68Skel.so"