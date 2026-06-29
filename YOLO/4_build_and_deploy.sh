#!/usr/bin/env bash
# =============================================================================
# Step 4: Cross-compile yolo_tracker for aarch64 and deploy to QCS6490
#
# Run on: x86 Linux host with QAIRT SDK + FastCV SDK + aarch64 toolchain
# =============================================================================

set -euo pipefail

# ── Edit these ────────────────────────────────────────────────────────────────
QAIRT_ROOT="${QAIRT_ROOT:?Set QAIRT_ROOT}"
FASTCV_ROOT="${FASTCV_ROOT:?Set FASTCV_ROOT}"
BOARD_IP="192.168.1.100"
BOARD_USER="root"
REMOTE_DIR="/home/root/yolo_tracker"
DSP_ARCH="v68"
# ─────────────────────────────────────────────────────────────────────────────

TOOLCHAIN="$QAIRT_ROOT/cmake/toolchains/aarch64-oe-linux.cmake"

echo "=== Cross-compiling for aarch64 ==="
mkdir -p build_aarch64 && cd build_aarch64

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DQAIRT_ROOT="$QAIRT_ROOT" \
    -DFASTCV_ROOT="$FASTCV_ROOT" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . --parallel "$(nproc)"
cd ..

echo "=== Deploying to QCS6490 @ $BOARD_IP ==="
ssh "${BOARD_USER}@${BOARD_IP}" "mkdir -p ${REMOTE_DIR}"

# ── Binary ────────────────────────────────────────────────────────────────────
scp build_aarch64/yolo_tracker "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"

# ── QNN HTP runtime libs ─────────────────────────────────────────────────────
AARCH64_QNN_LIB="$QAIRT_ROOT/lib/aarch64-oe-linux-gcc11.2"
HEX_LIB="$QAIRT_ROOT/lib/hexagon-${DSP_ARCH}/unsigned"

scp \
    "${AARCH64_QNN_LIB}/libQnnHtp.so" \
    "${AARCH64_QNN_LIB}/libQnnHtpV${DSP_ARCH}Stub.so" \
    "${HEX_LIB}/libQnnHtpV${DSP_ARCH}Skel.so" \
    "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"

# ── FastCV runtime lib ────────────────────────────────────────────────────────
scp \
    "${FASTCV_ROOT}/libs/aarch64-linux-gcc/libFastCV_Interface.so" \
    "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"

# ── Model + labels ────────────────────────────────────────────────────────────
scp \
    "context_bin/yolov8n_qcs6490.bin" \
    "coco.txt" \
    "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"

echo ""
echo "=== Starting pipeline on board ==="
echo "SSH in and run:"
echo ""
echo "  ssh ${BOARD_USER}@${BOARD_IP}"
echo "  cd ${REMOTE_DIR}"
echo "  export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH"
echo "  chmod +x yolo_tracker"
echo "  ./yolo_tracker \\"
echo "      --model   yolov8n_qcs6490.bin \\"
echo "      --labels  coco.txt \\"
echo "      --device  /dev/video0 \\"
echo "      --width   1280 --height 720 --fps 30 \\"
echo "      --mode    http --port 8080"
echo ""
echo "Then view the stream at: http://${BOARD_IP}:8080/stream"
echo ""
echo "✓ Deployment files copied."