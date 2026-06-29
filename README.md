# 🛸 Drone RF Signal Detection

Hệ thống phát hiện UAV/drone qua phân tích tín hiệu RF thu từ thiết bị SDR.

## Tổng quan

```
SDR (USRP X300 / BladeRF 2.0)
        │  Thu tín hiệu IQ thô
        ▼
Xử lý STFT  (bin_data / dat_data)
        │  Chuyển IQ → ảnh spectrogram
        ▼
EfficientNet-B0  (EfficientNet/model)
        │  Huấn luyện từ spectrogram → .tflite
        ▼
Pipeline Inference  (EfficientNet/pipeline)
        │  USRP → STFT → NPU → Telemetry API
        ▼
Dashboard / Server
```

## Cấu trúc thư mục

```
SpectrumAnalyzer/
├── usrp.py              # Thu IQ cố định tần số (USRP X300)
├── usrp_sweep.py        # Quét IQ đa tần số (USRP X300)
├── spectrum_analyzer.py # GUI phân tích phổ BladeRF 2.0
│
├── bin_data/            # Dữ liệu thô .bin (SC16) + script STFT
├── dat_data/            # Dữ liệu thô .dat (fc32) + script STFT
│
└── EfficientNet/
    ├── model/           # Huấn luyện và export mô hình
    └── pipeline/        # Pipeline inference thời gian thực
```

## Thiết bị hỗ trợ

| Thiết bị | Script | Ghi chú |
|----------|--------|---------|
| USRP X300 | `usrp.py`, `usrp_sweep.py` | Thu IQ qua 10GbE, UHD driver |
| BladeRF 2.0 | `spectrum_analyzer.py` | GUI desktop, PySide6 |
| Qualcomm RB3 Gen 2 | `EfficientNet/pipeline/` | NPU inference (QNN HTP) |

## Chi tiết từng thành phần

- [`bin_data/README.md`](bin_data/README.md) — Dữ liệu .bin và script STFT
- [`dat_data/README.md`](dat_data/README.md) — Dữ liệu .dat và script STFT
- [`EfficientNet/model/README.md`](EfficientNet/model/README.md) — Huấn luyện mô hình
- [`EfficientNet/pipeline/README.md`](EfficientNet/pipeline/README.md) — Pipeline inference

## Cài đặt nhanh

```bash
# Thu tín hiệu USRP
pip install uhd numpy

# BladeRF GUI
pip install bladerf PySide6 pyqtgraph numpy

# Xử lý dữ liệu (STFT)
pip install numpy scipy matplotlib

# Huấn luyện mô hình
pip install torch torchvision onnx

# Pipeline inference (RB3 Gen 2)
pip install numpy scipy pillow matplotlib ai-edge-litert
```
