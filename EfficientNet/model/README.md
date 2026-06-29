# EfficientNet/model — Huấn luyện mô hình phân loại drone

## Kiến trúc mô hình

`DroneClassifier` là bản **EfficientNet-B0 tùy chỉnh từ đầu** (from scratch), tối ưu cho NPU Qualcomm.

| Thuộc tính | Giá trị |
|-----------|---------|
| Input | `(1, 3, 256, 512)` — ảnh spectrogram STFT màu viridis |
| Output | `(1, 3)` logits cho 3 lớp |
| Backbone | 7 stage MBConv + Squeeze-and-Excitation |
| Activation | SiLU (NPU-friendly) |
| Export | ONNX Opset 17 → TFLite INT8 |

**3 lớp phân loại:**
- `DRONE` — tín hiệu drone rõ ràng (dải dọc đậm trên spectrogram)
- `DRONE_SIGNAL` — tín hiệu drone yếu / thoáng qua
- `NO_DRONE` — nhiễu nền

## Chuẩn bị dataset

```
NEW_DATASET/               ← hoặc UPDATED_DATASET/
├── DRONE/
│   ├── recording1__001.png
│   └── recording1__002.png
├── DRONE_SIGNAL/
│   └── ...
└── NO_DRONE/
    └── ...
```

> **Quy tắc đặt tên file:** các frame cùng một lần thu phải có cùng prefix (vd: `dji_hover__001.png`, `dji_hover__002.png`). Script sẽ tự nhóm theo recording để tránh data leakage.

## Các file

### `new_train.py` — Huấn luyện + export ONNX

```bash
cd EfficientNet/model
python new_train.py
```

**Thông số chỉnh trong `if __name__ == "__main__"`:**

```python
RAW_DATASET_DIR     = "UPDATED_DATASET"  # thư mục dataset gốc
SPLIT_DATASET_DIR   = "dataset_split"    # thư mục sau khi chia train/val
IMG_H, IMG_W        = 256, 512
NUM_EPOCHS          = 50
BATCH_SIZE          = 16                 # giảm xuống 8 nếu hết VRAM
LEARNING_RATE       = 3e-4
EARLY_STOP_PATIENCE = 10
WARMUP_EPOCHS       = 5
```

**File đầu ra:**
- `best_model.pth` — PyTorch checkpoint tốt nhất
- `drone_classifier_b0.onnx` — Model ONNX production (Opset 17)

---

### `new_train_with_graph.py` — Huấn luyện + export ONNX + RoCC graph

Giống `new_train.py` nhưng xuất thêm file graph cho Netron / compiler.

```bash
python new_train_with_graph.py
```

**File đầu ra bổ sung:**
- `drone_classifier_rocc.onnx` — Graph để xem trong [Netron](https://netron.app)

---

### `new_quantize.py` — Quantize INT8 qua Qualcomm AI Hub

Nhận file ONNX, calibrate bằng ảnh dataset thực, export `.tflite` INT8.

```bash
python new_quantize.py
```

## Quy trình đầy đủ

```
Dataset ảnh PNG
    │
    ▼  new_train.py  (hoặc new_train_with_graph.py)
best_model.pth  +  drone_classifier_b0.onnx
    │
    ▼  new_quantize.py  (Qualcomm AI Hub)
new_three_classes.tflite  (INT8, chạy trên NPU RB3 Gen 2)
    │
    ▼  EfficientNet/pipeline/run_pipeline.py
Phát hiện drone thời gian thực
```

## Lưu ý quan trọng

- Preprocessing lúc **train** và lúc **inference** phải khớp hoàn toàn: `nfft=1024`, `hamming window`, `viridis colormap`, resize `256×512`, **không** dùng ImageNet normalization
- Chỉ dùng `ToTensor()` → pixel chia 255, giá trị [0.0, 1.0]
