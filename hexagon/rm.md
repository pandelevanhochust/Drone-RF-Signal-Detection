# QCS6490 Image Classification — NPU/HTP via QAIRT/QNN SDK

End-to-end pipeline for running ResNet50 (or MobileNetV2) image classification
on the **QCS6490 Hexagon HTP v68** NPU using the Qualcomm AI Runtime (QAIRT) SDK.

---

## QCS6490 HTP Details

| Property       | Value              |
|----------------|--------------------|
| Hexagon arch   | **v68**            |
| soc_id         | **35**             |
| Supported dtypes | INT8, INT16, FP16  |
| VTCM           | 4 MB               |

---

## Full Pipeline

```
[Host: x86 Ubuntu 22.04]                   [Board: QCS6490]
─────────────────────────────────────────  ──────────────────────────
1. Export PyTorch → ONNX                   5. Deploy .bin + libs
   scripts/1_export_onnx.py                   scripts/3_build_and_deploy.sh

2. ONNX → DLC → quantized DLC → .bin      6a. C++ inference app
   scripts/2_convert_and_quantize.sh           src/qcs6490_classify.cpp

3. Cross-compile C++ app                   6b. OR: CLI quick-test
   scripts/3_build_and_deploy.sh               scripts/4_run_on_board_cli.sh

4. Prepare calibration data
   scripts/prepare_calibration_data.py
```

---

## Prerequisites

### Host
```bash
# QAIRT SDK (download from Qualcomm Software Center)
unzip v2.42.0.251225.zip
export QAIRT_ROOT=~/qairt/2.42.0.251225
source $QAIRT_ROOT/bin/envsetup.sh

pip install torch torchvision onnx onnxruntime Pillow numpy
```

### Board
- QCS6490 board running Ubuntu / Qualcomm Linux
- `adsprpcd` daemon running (FastRPC for HTP communication)
- `libQnnHtp.so`, `libQnnHtpV68Stub.so`, `libQnnHtpV68Skel.so` deployed

---

## Step-by-Step

### 1. Export ONNX model
```bash
python3 scripts/1_export_onnx.py
# → resnet50.onnx
```

### 2. Prepare calibration images
```bash
# Download ~100 ImageNet validation images into images/calib/
python3 scripts/prepare_calibration_data.py \
    --img_dir images/calib/ \
    --out_dir data/calib/ \
    --num_imgs 100
```

### 3. Convert → quantize → context binary
```bash
bash scripts/2_convert_and_quantize.sh
# → context_bin/resnet50_qcs6490.bin
```

Key parameters for QCS6490:
- `dsp_arch = v68`  (Hexagon v68)
- `soc_id = 35`
- Quantization: `w8a8` (INT8 weights + INT8 activations) — best performance on HTP

### 4. Build and deploy the C++ app
```bash
export BOARD_IP=192.168.1.100
bash scripts/3_build_and_deploy.sh
```

### 5. (Alternative) CLI test using qnn-net-run
```bash
# On the board:
bash scripts/4_run_on_board_cli.sh
```

---

## Project Structure

```
qcs6490_classification/
├── README.md
├── CMakeLists.txt
├── src/
│   └── qcs6490_classify.cpp      ← C++ inference app (QNN API)
├── scripts/
│   ├── 1_export_onnx.py          ← PyTorch → ONNX
│   ├── 2_convert_and_quantize.sh ← ONNX → DLC → Context-Binary
│   ├── 3_build_and_deploy.sh     ← Cross-compile + push to board
│   ├── 4_run_on_board_cli.sh     ← qnn-net-run CLI quick test
│   └── prepare_calibration_data.py
└── configs/
    ├── htp_config_qcs6490.json   ← dsp_arch + soc_id for context-binary gen
    └── backend_extensions.json   ← backend extensions for qnn-net-run
```

---

## Common Issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `dlopen libQnnHtpV68Skel.so failed` | Skel lib not on board LD path | Copy to same dir, set `LD_LIBRARY_PATH` |
| `FastRPC error` | `adsprpcd` not running | `adsprpcd &` or check init |
| Low accuracy after quantization | Poor calibration data | Use real ImageNet images, not random |
| `Graph name not found` | Graph name mismatch | Match `--binary_file` name in converter to `graphRetrieve()` call |
| Slow HTP inference | Perf mode not set | Use `QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_BURST_MODE` or `--perf_profile burst` |

---

## Useful QAIRT CLI Reference

```bash
# Inspect a DLC model
snpe-dlc-info -i resnet50.dlc

# Benchmark on-device
qnn-net-run --backend libQnnHtp.so \
            --retrieve_context resnet50_qcs6490.bin \
            --input_list test_list.txt \
            --output_dir output/ \
            --perf_profile burst \
            --duration 10    # run for 10 seconds, report avg latency
```