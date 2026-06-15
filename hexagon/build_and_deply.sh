#!/usr/bin/env bash
# =============================================================================
# Step 3: Cross-compile qcs6490_classify for aarch64 and deploy to QCS6490
#
# Run on: x86 Linux host with QAIRT SDK + aarch64 toolchain installed
# =============================================================================

set -euo pipefail

# ── Edit these ────────────────────────────────────────────────────────────────
QAIRT_ROOT="${QAIRT_ROOT:?}"
BOARD_IP="192.168.1.100"               # QCS6490 board IP (SSH)
BOARD_USER="root"
REMOTE_DIR="/home/root/classify_demo"
DSP_ARCH="v68"                         # QCS6490 = Hexagon v68
# ─────────────────────────────────────────────────────────────────────────────

TOOLCHAIN="$QAIRT_ROOT/cmake/toolchains/aarch64-oe-linux.cmake"

echo "=== Cross-compiling for aarch64 ==="
mkdir -p build_aarch64 && cd build_aarch64

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DQAIRT_ROOT="$QAIRT_ROOT" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . --parallel "$(nproc)"
cd ..

echo "=== Deploying to QCS6490 @ $BOARD_IP ==="

# Create remote directory
ssh "${BOARD_USER}@${BOARD_IP}" "mkdir -p ${REMOTE_DIR}"

# Binary
scp build_aarch64/qcs6490_classify "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"

# QNN HTP runtime libs needed on the board
AARCH64_LIB="$QAIRT_ROOT/lib/aarch64-oe-linux-gcc11.2"
HEX_LIB="$QAIRT_ROOT/lib/hexagon-${DSP_ARCH}/unsigned"

scp \
    "${AARCH64_LIB}/libQnnHtp.so" \
    "${AARCH64_LIB}/libQnnHtpV${DSP_ARCH}Stub.so" \
    "${HEX_LIB}/libQnnHtpV${DSP_ARCH}Skel.so" \
    "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"

# Model + inputs
scp \
    "context_bin/resnet50_qcs6490.bin" \
    "data/test/sample_input.raw" \
    "imagenet_classes.txt" \
    "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"

echo ""
echo "=== Running inference on board ==="
ssh "${BOARD_USER}@${BOARD_IP}" bash -s <<REMOTE
cd ${REMOTE_DIR}
export LD_LIBRARY_PATH=.:${REMOTE_DIR}:\$LD_LIBRARY_PATH
# FastRPC must be running (adsprpcd) — usually started by init on QCS6490
chmod +x qcs6490_classify
./qcs6490_classify \
    resnet50_qcs6490.bin \
    sample_input.raw \
    imagenet_classes.txt \
    libQnnHtp.so
REMOTE

echo ""
echo "✓ Done."