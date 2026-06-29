# EfficientNet/pipeline — Pipeline inference thời gian thực

## Luồng xử lý

```
[USRP X300]
    │  IQ frames (complex64, 25 MHz)
    ▼
[usrp_capture.py]  ── thread nền (daemon)
    │  đẩy (freq_hz, iq) vào frame_queue
    ▼
[run_pipeline_sweep.py]  ── main thread
    │
    ├─► [stft_preprocessor.py]   ~12 ms/frame
    │       IQ → STFT → viridis → tensor (1,3,256,512) float32
    │
    ├─► [drone_inference.py]     ~22 ms/frame  (NPU HTP)
    │       tensor → TFLite → softmax → class + confidence
    │
    └─► [telemetry_sender.py]    async, non-blocking
            kết quả → HTTP POST → /api/v1/telemetry/log
```

## Cấu hình

**Bước 1: Tạo file `.env`**
```ini
API_URL=http://192.168.5.75:80
API_KEY=YOUR_API_KEY_HERE
DEVICE_ID=101
```

**Bước 2: Đảm bảo có model và class names**
```
EfficientNet/
├── new_three_classes.tflite
└── pipeline/
    └── class_names.txt    # nội dung: DRONE\nDRONE_SIGNAL\nNO_DRONE
```

## Cách chạy

### Pipeline quét tần số (khuyến nghị)

```bash
cd EfficientNet/pipeline

# Chạy đầy đủ: sweep 2.400–2.475 GHz + NPU + telemetry
python run_pipeline_sweep.py

# Các tuỳ chọn thường dùng
python run_pipeline_sweep.py --gain 30           # tăng gain RX
python run_pipeline_sweep.py --cpu               # dùng CPU thay NPU
python run_pipeline_sweep.py --threshold 0.80    # tăng ngưỡng tin cậy
python run_pipeline_sweep.py --no_telemetry      # tắt gửi API
python run_pipeline_sweep.py --no_sweep          # cố định 2.400 GHz (debug)
python run_pipeline_sweep.py --save_dir specs/   # lưu spectrogram PNG
python run_pipeline_sweep.py --no_infer          # chỉ chạy STFT
```

### Pipeline tần số cố định

```bash
python run_pipeline.py
python run_pipeline.py --gain 30 --threshold 0.75
```

### Kiểm tra telemetry

```bash
python test_telemetry.py    # gửi payload giả lên server
```

## Tham số CLI

| Tham số | Ý nghĩa | Mặc định |
|---------|---------|---------|
| `--model` | Đường dẫn file `.tflite` | `../new_three_classes.tflite` |
| `--addr` | IP USRP X300 | `192.168.5.111` |
| `--gain` | RX gain (dB) | 35.0 |
| `--cpu` | Dùng CPU, bỏ NPU | False |
| `--threshold` | Ngưỡng confidence tối thiểu | 0.70 |
| `--queue_size` | Kích thước hàng đợi IQ frame | 4 |
| `--save_dir` | Thư mục lưu spectrogram PNG | None |
| `--no_infer` | Bỏ qua inference | False |
| `--no_telemetry` | Bỏ qua gửi API | False |
| `--no_sweep` | Cố định tần số *(sweep only)* | False |

## Mô tả các file

| File | Vai trò |
|------|---------|
| `run_pipeline_sweep.py` | Entry point — pipeline quét tần số |
| `run_pipeline.py` | Entry point — pipeline tần số cố định |
| `usrp_capture.py` | Thread thu IQ, quản lý sweep tần số |
| `stft_preprocessor.py` | Chuyển IQ → tensor spectrogram |
| `drone_inference.py` | Inference TFLite (NPU / CPU) |
| `telemetry_sender.py` | Gửi kết quả lên API (async) |
| `test_telemetry.py` | Test gửi API với payload giả |
| `multiple_test_telemetry.py` | Test đồng thời nhiều thiết bị |
| `class_names.txt` | Danh sách nhãn lớp |
| `.env` | Cấu hình API (không commit lên git) |

## Thông số sweep mặc định

| Thông số | Giá trị |
|---------|---------|
| Dải tần | 2.400 → 2.475 GHz |
| Bước nhảy | 25 MHz (4 tần số) |
| Frame/bước | 5 frame (~575 ms/kênh) |
| Chu kỳ | ~2.3 s / vòng quét |
| Sample rate | 25 MHz |
| Frame size | 2,000,000 mẫu (80 ms) |
