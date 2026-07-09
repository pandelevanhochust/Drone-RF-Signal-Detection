# 🛸 Drone RF Signal Detection — SpectrumAnalyzer

Hệ thống thu thập tín hiệu RF từ thiết bị SDR (USRP X300 hoặc BladeRF 2.0), xử lý thành ảnh STFT spectrogram, và phân loại tín hiệu UAV/drone bằng mô hình EfficientNet-B0 tùy chỉnh.

---

## 📁 Cấu trúc thư mục

```
SpectrumAnalyzer/
│
├── usrp.py                  # Thu IQ tại một tần số cố định (USRP X300)
├── usrp_sweep.py            # Quét IQ đa tần số (sweep) với USRP X300
├── spectrum_analyzer.py     # GUI phân tích phổ BladeRF 2.0 (PySide6)
│
├── bin_data/                # Dữ liệu thô .bin ( định dạng SC16)
│   ├── bin_read.py          # Đọc và hiển thị STFT từ file .bin
│   └── normalize_bin_read.py# Đọc .bin có chuẩn hoá (DC offset + scale)
│
├── dat_data/                # Dữ liệu thô .dat ( định dạng float32)
│   └── dat_read.py          # Đọc và hiển thị STFT từ file .dat
│
└── EfficientNet/
    ├── model/               # Huấn luyện và export mô hình
    │   ├── new_train.py         # Huấn luyện → export ONNX
    │   └── new_train_with_graph.py  # Huấn luyện → export ONNX + RoCC graph
    │
    └── pipeline/            # Pipeline inference thời gian thực
        ├── run_pipeline.py      # Điểm khởi chạy chính (USRP → STFT → NPU)
        ├── usrp_capture.py      # Thread thu IQ từ USRP
        ├── stft_preprocessor.py # Chuyển IQ → tensor spectrogram
        ├── drone_inference.py   # Inference TFLite trên NPU/CPU
        ├── telemetry_sender.py  # Gửi kết quả lên API qua HTTP POST
        └── class_names.txt      # Danh sách nhãn lớp
```

---

## ⚙️ Thiết bị thu tín hiệu

### `usrp.py` — Thu IQ cố định tần số (USRP X300)

Script đơn giản dùng để thu một lượng lớn mẫu IQ tại **một tần số duy nhất** và lưu ra file `.bin` định dạng `fc32` (float32 complex — interleaved I/Q).

**Yêu cầu:**

```bash
pip install uhd numpy
```

**Cách chạy:**

```bash
# Thu 100 triệu mẫu tại 2.375 GHz, gain 20 dB
python usrp.py -f 2375000000 -r 25000000 -g 20 -n 100000000 -o capture.bin
```

| Tham số | Ý nghĩa               | Mặc định     |
| ------- | --------------------- | ------------ |
| `-f`    | Tần số trung tâm (Hz) | 2.375 GHz    |
| `-r`    | Sample rate (Hz)      | 25 MHz       |
| `-g`    | RX Gain (dB)          | 20           |
| `-n`    | Số mẫu cần thu        | 100,000,000  |
| `-o`    | File đầu ra           | _(bắt buộc)_ |

> **Lưu ý:** Địa chỉ IP của USRP được hardcode là `192.168.5.111`. Chỉnh trong code nếu cần.

**Định dạng file đầu ra:**

```
[I0_f32, Q0_f32, I1_f32, Q1_f32, ...]   # interleaved float32
```

Đọc lại bằng numpy:

```python
iq = np.fromfile("capture.bin", dtype=np.complex64)
```

---

### `usrp_sweep.py` — Quét IQ đa tần số (USRP X300)

Script quét (sweep) tần số theo từng bước 25 MHz để bao phủ toàn bộ dải ISM 2.4 GHz (85 MHz rộng). Mỗi hop được lưu ra một file `.bin` và một file `_meta.txt` kèm theo.

**Tại sao cần sweep?**
USRP X300 với card UBX-160 chỉ thu được **25 MHz** tức thời. Dải ISM 2.4 GHz rộng 85 MHz. Sweep hop tần số giúp bao phủ toàn bộ băng tần mà không vượt quá giới hạn 10 GbE.

**Cách chạy:**

```bash
# Quét mặc định: 2.440 → 2.460 GHz, 80ms/hop, 1 lần
python usrp_sweep.py -o captures/

# Quét toàn bộ ISM 2.4 GHz, 5 lần quét
python usrp_sweep.py \
    --start_freq 2.4e9 \
    --stop_freq  2.485e9 \
    --step       25e6 \
    --duration_ms 80 \
    --passes 5 \
    --gain 30 \
    --addr 192.168.5.111 \
    -o captures/
```

| Tham số            | Ý nghĩa                           | Mặc định        |
| ------------------ | --------------------------------- | --------------- |
| `--addr`           | IP của USRP X300                  | `192.168.5.111` |
| `--start_freq`     | Tần số bắt đầu (Hz)               | 2.440 GHz       |
| `--stop_freq`      | Tần số kết thúc (Hz)              | 2.460 GHz       |
| `--step`           | Bước nhảy tần số (Hz)             | 25 MHz          |
| `-r/--rate`        | Sample rate (Hz)                  | 25 MHz          |
| `-g/--gain`        | RX Gain (dB)                      | 30              |
| `-d/--duration_ms` | Thời gian thu mỗi hop (ms)        | 80 ms           |
| `--passes`         | Số lần quét toàn bộ dải           | 1               |
| `--settle_ms`      | Thời gian chờ PLL lock sau retune | 20 ms           |
| `-o/--out_dir`     | Thư mục đầu ra                    | _(bắt buộc)_    |

**Cấu trúc đầu ra:**

```
captures/
    seg_0000_2440MHz.bin        # IQ samples hop 0 (fc32)
    seg_0000_2440MHz_meta.txt   # Metadata: freq, rate, samples, gain, timestamp
    seg_0001_2460MHz.bin
    seg_0001_2460MHz_meta.txt
    ...
```

---

### `spectrum_analyzer.py` — GUI Phân tích phổ BladeRF 2.0

Ứng dụng desktop (PySide6 + PyQtGraph) cung cấp giao diện trực quan để phân tích phổ tần số thời gian thực với BladeRF 2.0.

**Yêu cầu:**

```bash
pip install bladerf PySide6 pyqtgraph numpy
```

**Cách chạy:**

```bash
python spectrum_analyzer.py
```

**Tính năng chính:**

| Tính năng                | Mô tả                                        |
| ------------------------ | -------------------------------------------- |
| **Spectrum + Waterfall** | Hiển thị phổ dBm và waterfall theo thời gian |
| **Quét composite**       | Ghép nhiều đoạn phổ thành dải rộng           |
| **Max Hold**             | Giữ đỉnh tín hiệu mạnh nhất theo thời gian   |
| **Calibration**          | Tạo và áp dụng profile hiệu chỉnh biên độ    |
| **TX**                   | Phát tín hiệu CW qua kênh TX1/TX2            |
| **Sweep TX**             | Quét tần số phát tự động theo bước           |
| **IQ Recording**         | Lưu mẫu IQ thô ra file                       |

**Luồng hoạt động:**

1. Nhấn **Connect to BladeRF** → khởi tạo device
2. Chọn **RX Channel** (RX1 hoặc RX2), cấu hình tần số quét
3. Nhấn **Start Scanning** → thu IQ liên tục, hiển thị phổ
4. (Tùy chọn) Nhấn **Calibrate Profile** để tạo baseline nhiễu nền

---

## 📊 Xử lý dữ liệu tín hiệu thành STFT

### `bin_data/` — Dữ liệu BladeRF (định dạng SC16 — int16)

Chứa các file `.bin` thu từ BladeRF với định dạng **SC16**: mỗi mẫu IQ gồm 2 số nguyên 16-bit (I, Q).

**`bin_read.py`** — Đọc và hiển thị STFT từ file `.bin`:

```python
# Thay đổi các thông số này:
FILE_PATH = r"1toan.bin"
FS = 60e6           # Sample rate (Hz)
CENTER_FREQ = 2.375e9  # Tần số trung tâm (Hz)

python bin_data/bin_read.py
```

**Pipeline xử lý:**

1. `np.memmap` → mở file không load vào RAM
2. Đọc slice cần thiết (`duration_ms` ms đầu tiên)
3. Tách I/Q (interleaved int16) → chuyển sang complex float32
4. Tính STFT bằng `scipy.signal.stft` (Hamming window, 2048-point)
5. `fftshift` để đưa DC vào giữa
6. Hiển thị spectrogram dBm

**`normalize_bin_read.py`** — Phiên bản có chuẩn hoá:

- Chia cho 32768 (chuẩn hoá về [-1, 1])
- Trừ DC offset (mean subtraction)
- Hỗ trợ tham số `start_ms` để đọc từ vị trí bất kỳ trong file

---

### `dat_data/` — Dữ liệu GNU Radio / USRP (định dạng float32)

Chứa các file `.dat` thu từ GNU Radio hoặc `usrp_sweep.py` với định dạng **fc32**: mỗi mẫu IQ gồm 2 số float32 (I, Q).

**`dat_read.py`** — Đọc và hiển thị STFT từ file `.dat`:

```python
# Thay đổi các thông số này:
FILE_PATH = r"stft_segmentation/Drone/noise.dat"
FS = 25e6           # Sample rate (Hz)
CENTER_FREQ = 2.440e9  # Tần số trung tâm (Hz)

python dat_data/dat_read.py
```

**Pipeline xử lý:**

1. `np.memmap` với `dtype=np.float32` → không cần chuyển đổi kiểu
2. Hỗ trợ `start_ms` để bỏ qua phần đầu file (loại bỏ transient)
3. Tách I/Q → complex → STFT (Hamming, 1024-point)
4. `fftshift` và hiển thị spectrogram

**So sánh hai định dạng:**

|               | `bin_data` (.bin SC16) | `dat_data` (.dat fc32) |
| ------------- | ---------------------- | ---------------------- |
| Nguồn         | BladeRF 2.0            | GNU Radio / USRP sweep |
| dtype         | `int16`                | `float32`              |
| Bytes/mẫu     | 4 bytes                | 8 bytes                |
| nfft mặc định | 2048                   | 1024                   |

---

## 🧠 EfficientNet — Huấn luyện & Triển khai

### `EfficientNet/model/` — Huấn luyện mô hình

#### Kiến trúc mô hình

Mô hình `DroneClassifier` là bản EfficientNet-B0 **tùy chỉnh từ đầu** (from scratch), tối ưu hoá để export sang ONNX và chạy trên NPU Qualcomm (RB3 Gen 2).

- **Input:** `(1, 3, 256, 512)` — ảnh spectrogram STFT màu viridis
- **Output:** `(1, 3)` logits cho 3 lớp: `DRONE`, `DRONE_SIGNAL`, `NO_DRONE`
- **Backbone:** 7 stage MBConv blocks với Squeeze-and-Excitation
- **NPU compliance:** Dùng `SiLU`, `ReduceMean` (static axes) — không có `LayerNorm`, `Gelu`, `Softmax`

#### Chuẩn bị dataset

Dataset phải có cấu trúc:

```
NEW_DATASET/          # (hoặc UPDATED_DATASET/ tuỳ file)
├── DRONE/
│   ├── recording1__001.png
│   └── recording1__002.png
├── DRONE_SIGNAL/
│   └── ...
└── NO_DRONE/
    └── ...
```

Script tự động chia train/val theo **recording-level isolation** (không để frame cùng một lần thu xuất hiện ở cả train lẫn val — tránh data leakage).

#### `new_train.py` — Huấn luyện và export ONNX

```bash
cd EfficientNet/model
python new_train.py
```

**Các bước thực hiện:**

1. Xoá và tạo lại `dataset_split/` (train 80%, val 20%)
2. Khởi tạo `DroneClassifier` (~5.3M tham số)
3. Huấn luyện tối đa **50 epochs** với:
   - AdamW optimizer, lr=3e-4, weight_decay=1e-2
   - Warmup cosine learning rate schedule (5 epoch warmup)
   - CrossEntropyLoss với label smoothing 0.05
   - Early stopping sau 10 epoch không cải thiện
4. Lưu checkpoint tốt nhất → `best_model.pth`
5. Export sang **`drone_classifier_b0.onnx`** (Opset 17)

**Thay đổi thông số trong phần `if __name__ == "__main__"`:**

```python
RAW_DATASET_DIR   = "UPDATED_DATASET"   # Đường dẫn dataset
BATCH_SIZE        = 16                  # Giảm nếu OOM GPU
NUM_EPOCHS        = 50
LEARNING_RATE     = 3e-4
EARLY_STOP_PATIENCE = 10
```

#### `new_train_with_graph.py` — Huấn luyện + export đồ thị RoCC

Giống `new_train.py` nhưng bổ sung thêm bước export **RoCC graph** dùng cho Netron và custom compiler.

```bash
cd EfficientNet/model
python new_train_with_graph.py
```

**Điểm khác biệt so với `new_train.py`:**

|                | `new_train.py`             | `new_train_with_graph.py`       |
| -------------- | -------------------------- | ------------------------------- |
| Dataset dir    | `UPDATED_DATASET`          | `NEW_DATASET`                   |
| ONNX output    | `drone_classifier_b0.onnx` | `drone_classifier_b0.onnx`      |
| Graph thêm     | ❌                         | ✅ `drone_classifier_rocc.onnx` |
| Validation log | Chi tiết per-class         | Tổng hợp                        |

**File đầu ra:**

- `best_model.pth` — PyTorch checkpoint
- `drone_classifier_b0.onnx` — Model production
- `drone_classifier_rocc.onnx` — Graph cho Netron / compiler (chỉ `new_train_with_graph.py`)

---

### `EfficientNet/pipeline/` — Pipeline Inference Thời Gian Thực

#### Luồng xử lý tổng quan

```
[USRP X300]
    │  IQ frames (complex64, 25 MHz, 2.44 GHz)
    ▼
[usrp_capture.py]  ──────────────── thread nền (daemon)
    │  push vào frame_queue (maxsize=4)
    ▼
[run_pipeline.py]  ──────────────── main thread
    │
    ├─► [stft_preprocessor.py]
    │       IQ → scipy STFT → fftshift → dB
    │       → crop skirt → min-max norm → uint8
    │       → resize 256×512 → viridis LUT
    │       → /255.0 → tensor (1,3,256,512) float32
    │       ≈ 12 ms/frame
    │
    ├─► [drone_inference.py]
    │       tensor → TFLite interpreter
    │       → softmax → top-1 class + confidence
    │       ≈ 22 ms/frame (NPU HTP)
    │
    └─► [telemetry_sender.py]
            kết quả → build_payload() → HTTP POST
            → /api/v1/telemetry/log (async, non-blocking)
```

#### Cách chạy pipeline

**Bước 1: Cấu hình `.env`**

```ini
# EfficientNet/pipeline/.env
API_URL=http://192.168.5.75:80
API_KEY=YOUR_API_KEY_HERE
DEVICE_ID=101
```

**Bước 2: Đảm bảo có file model và class names**

```
EfficientNet/
├── new_three_classes.tflite   # model đã convert sang TFLite
└── pipeline/
    └── class_names.txt        # nội dung: DRONE\nDRONE_SIGNAL\nNO_DRONE
```

**Bước 3: Chạy pipeline**

```bash
cd EfficientNet/pipeline

# Chạy đầy đủ (NPU + telemetry)
python run_pipeline.py

# Tuỳ chọn thêm
python run_pipeline.py --gain 30                        # Tăng gain RX
python run_pipeline.py --cpu                            # Dùng CPU thay NPU
python run_pipeline.py --threshold 0.80                 # Tăng ngưỡng tin cậy
python run_pipeline.py --no_telemetry                   # Tắt gửi API
python run_pipeline.py --save_dir debug_specs/          # Lưu spectrogram debug
python run_pipeline.py --no_infer                       # Chỉ chạy STFT, không inference
```

| Tham số          | Ý nghĩa                       | Mặc định                      |
| ---------------- | ----------------------------- | ----------------------------- |
| `--model`        | Đường dẫn file `.tflite`      | `../new_three_classes.tflite` |
| `--addr`         | IP USRP X300                  | `192.168.5.111`               |
| `--gain`         | RX gain (dB)                  | 35.0                          |
| `--cpu`          | Dùng CPU (bỏ NPU)             | False                         |
| `--threshold`    | Ngưỡng confidence tối thiểu   | 0.70                          |
| `--queue_size`   | Kích thước hàng đợi IQ frames | 4                             |
| `--save_dir`     | Thư mục lưu spectrogram PNG   | None                          |
| `--no_infer`     | Bỏ qua inference              | False                         |
| `--no_telemetry` | Bỏ qua gửi API                | False                         |

#### Mô tả các module trong pipeline

**`stft_preprocessor.py`** — Chuyển đổi IQ → tensor model

- Các thông số phải **khớp chính xác** với quá trình tạo dataset khi train: `nfft=1024`, `hamming window`, `fftshift`, `viridis colormap`, `resize 256×512`
- Hàm `iq_to_spectrogram_debug()` lưu từng bước ra file PNG để kiểm tra

**`drone_inference.py`** — Inference TFLite

- Hỗ trợ **NPU Qualcomm Hexagon** (qua `libQnnTFLiteDelegate.so`) và CPU fallback
- Áp dụng **confidence threshold** (mặc định 0.70): nếu model không đủ tự tin → kết quả bị ép về `NO_DRONE`
- Kết quả trả về gồm: `class`, `confidence`, `probs` (3 giá trị), `latency_ms`, `suppressed`

**`telemetry_sender.py`** — Gửi kết quả lên server

- Gửi **async** qua hàng đợi nội bộ (không block inference)
- Retry tự động với exponential backoff (tối đa 3 lần)
- Đọc cấu hình từ file `.env` hoặc biến môi trường

---

## 📦 Yêu cầu cài đặt

```bash
# Thu tín hiệu USRP
pip install uhd numpy

# BladeRF GUI
pip install bladerf PySide6 pyqtgraph numpy

# Xử lý dữ liệu
pip install numpy scipy matplotlib

# Huấn luyện mô hình
pip install torch torchvision onnx

# Pipeline inference
pip install numpy scipy pillow matplotlib ai-edge-litert
```

---

## 📌 Ghi chú nhanh

- File `.bin` từ BladeRF dùng `dtype=np.int16` (SC16 format)
- File `.dat` từ GNU Radio / USRP dùng `dtype=np.float32` (fc32 format)
- Preprocessing pipeline inference **phải khớp** với preprocessing khi tạo dataset: cùng `nfft`, cùng `window`, cùng `colormap`, **không** dùng ImageNet normalization
- Model TFLite chạy NPU yêu cầu board Qualcomm RB3 Gen 2 với `libQnnTFLiteDelegate.so`
