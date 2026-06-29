"""
test_telemetry.py (Multi-Device Edition)
=============================================================================
Kịch bản giả lập nhiều thiết bị biên đồng thời gửi dữ liệu về Telemetry API.
Hỗ trợ stress-test hệ thống backend với cấu hình số lượng thiết bị linh hoạt.
"""

import json
import os
import queue
import threading
import time
import random
from datetime import datetime, timezone
from pathlib import Path

import urllib.request
import urllib.error


# ─────────────────────────────────────────────────────────────────────────────
#  1. Các hàm cấu trúc lõi (Giữ nguyên từ hệ thống thật)
# ─────────────────────────────────────────────────────────────────────────────

def _load_env(env_path: str = ".env") -> dict:
    env = {}
    path = Path(env_path)
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


MAX_RETRIES = 3
RETRY_BASE_S = 1.0
QUEUE_MAXSIZE = 64
REQUEST_TIMEOUT = 1  # Giới hạn timeout 1s để tránh nghẽn luồng khi test số lượng lớn

DRONE_TYPE_MAP = {
    "DRONE": "DJI Mavic 3",
    "DRONE_SIGNAL": "RF Transmission",
    "NO_DRONE": "None",
}


def build_payload(result: dict, device_id: int) -> dict:
    pred_class = result["class"]
    is_drone = pred_class != "NO_DRONE"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "deviceId": int(device_id),
        "timestamp": timestamp,
        "status": "Online",
        "detected": 1 if is_drone else 0,   # ← Fix: int, not bool
        "droneType": DRONE_TYPE_MAP.get(pred_class, "None"),
        "accuracy": round(float(result["confidence"]), 4),
        "controlState": "Active" if is_drone else "None",
        "latency": round(float(result.get("latency_ms", 12.5)), 1),
        "frequency": float(result.get("frequency_hz", 0))  # ← Also float, not int
    }


def _post(url: str, api_key: str, payload: dict) -> tuple:
    body = json.dumps(payload, default=str).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Device-API-Key": str(api_key).strip(),
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Edge/104.0.101"
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return True, resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return False, exc.code, exc.read().decode()
    except Exception as exc:
        return False, 0, str(exc)


def _post_with_retry(url: str, api_key: str, payload: dict) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        ok, code, body = _post(url, api_key, payload)
        if ok:
            return True
        wait = RETRY_BASE_S * (2 ** (attempt - 1))
        if attempt < MAX_RETRIES:
            time.sleep(wait)
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  2. Lớp TelemetrySender (Quản lý độc lập theo từng Device ID)
# ─────────────────────────────────────────────────────────────────────────────

class TelemetrySender:
    def __init__(self, endpoint: str, api_key: str, device_id: int):
        self._endpoint = endpoint
        self.api_key = api_key
        self.device_id = device_id

        self._queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._sent = 0
        self._failed = 0

        # Mỗi thiết bị sẽ có một luồng worker gửi mạng riêng biệt
        self._thread = threading.Thread(
            target=self._worker,
            name=f"Sender-Device-{device_id}",
            daemon=True
        )
        self._thread.start()

    def send(self, result: dict) -> None:
        payload = build_payload(result, self.device_id)
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(payload)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def _worker(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            ok = _post_with_retry(self._endpoint, self.api_key, payload)
            if ok:
                self._sent += 1
                # Log ra màn hình kèm theo Device ID để dễ theo dõi
                print(f"[Device {self.device_id}] ✓ Gửi thành công! (tx_q={self._queue.qsize()})")
            else:
                self._failed += 1
            self._queue.task_done()


# ─────────────────────────────────────────────────────────────────────────────
#  3. Luồng điều khiển cấu hình giả lập Multi-Device
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Hệ Thống Giả Lập Đa Thiết Bị Biên (Multi-Device Telemetry Simulator)")
    print("=" * 70)

    # Đọc cấu hình chung từ .env
    env = _load_env(".env")
    api_url = os.environ.get("API_URL", env.get("API_URL", "http://localhost:80"))
    api_key = os.environ.get("API_KEY", env.get("API_KEY", "YOUR_KEY"))

    base = api_url.rstrip("/")
    path = "/api/v1/telemetry/log"
    endpoint = base if base.endswith(path) else base + path

    # ⚙️ CẤU HÌNH SỐ LƯỢNG THIẾT BỊ TẠI ĐÂY
    START_DEVICE_ID = 1001
    NUM_DEVICES = 10  # Giả lập 10 thiết bị chạy song song cùng lúc
    LOOP_INTERVAL = 1.0  # Mỗi thiết bị gửi 1 gói/giây (Tương đương tổng 10 rps)

    devices = []
    print(f"[Khởi tạo] Đang tạo {NUM_DEVICES} thiết bị biên giả lập...")

    for i in range(NUM_DEVICES):
        dev_id = START_DEVICE_ID + i
        sender = TelemetrySender(endpoint=endpoint, api_key=api_key, device_id=dev_id)
        devices.append(sender)

    print(f"[Sẵn sàng] Mục tiêu: {endpoint}")
    print("[Chạy] Bắt đầu kích hoạt vòng lặp gửi dữ liệu. Nhấn Ctrl + C để dừng...\n")

    try:
        while True:
            # Vòng lặp kích hoạt tất cả các thiết bị cùng gửi dữ liệu trong chu kỳ
            for sender in devices:
                # Tạo kết quả mô phỏng (Có thể đổi thành ngẫu nhiên ngẫu nhiên hoặc chỉ cố định NO_DRONE)
                choice = random.choice(["DRONE", "DRONE_SIGNAL", "NO_DRONE"])

                mock_result = {
                    "class": choice,
                    "confidence": random.uniform(0.7500, 0.9900),
                    "latency_ms": random.uniform(10.0, 15.0),
                    "frequency_hz": 0 if choice == "NO_DRONE" else random.choice([2412, 2437, 2462])
                }

                # Đẩy lệnh gửi bất đồng bộ cho từng thiết bị
                sender.send(mock_result)

            # Nghỉ theo chu kỳ cấu hình trước khi lặp lại lượt tiếp theo
            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n[Dừng hệ thống] Đang giải phóng tài nguyên và ngắt luồng kết nối mạng...")
    finally:
        total_sent = 0
        total_failed = 0
        for sender in devices:
            sender.stop(timeout=1.0)
            total_sent += sender._sent
            total_failed += sender._failed

        print("=" * 70)
        print(f"[KẾT QUẢ TEST] Tổng số thiết bị: {NUM_DEVICES}")
        print(f"Tổng gói tin gửi thành công : {total_sent}")
        print(f"Tổng gói tin thất bại        : {total_failed}")
        print("=" * 70)