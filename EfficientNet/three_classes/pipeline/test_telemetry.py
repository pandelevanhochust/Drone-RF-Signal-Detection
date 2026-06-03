"""
test_telemetry.py
=============================================================================
Standalone infinite loop script to continuously test your Telemetry API connection.
Runs indefinitely, sending randomized mock 3-class data payloads until Ctrl+C.
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
#  1. Copied Core Engine Functions from telemetry_sender.py
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


MAX_RETRIES     = 3
RETRY_BASE_S    = 1.0
QUEUE_MAXSIZE   = 64
REQUEST_TIMEOUT = 5

DRONE_TYPE_MAP = {
    "DRONE"        : "Detected",
    "DRONE_SIGNAL" : "DroneSignal",
    "NO_DRONE"     : "None",
}


def build_payload(result: dict, device_id: int) -> dict:
    """
    Convert a DroneInferencer result dict into the strict API body schema.

    Expected Server Format:
    {
      "deviceId": 1001,
      "timestamp": "2026-06-03T02:17:14Z",
      "status": "Online",
      "detected": 1,
      "droneType": "DJI Mavic 3" | "None",
      "accuracy": 0.98,
      "controlState": "Active" | "None",
      "latency": 12.5
    }
    """
    pred_class = result["class"]
    is_drone = pred_class != "NO_DRONE"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Dynamic Drone Type Assignment to prevent server validation drops
    if pred_class == "DRONE":
        drone_type = "DRONE"  # Or use result.get("drone_model", "Unknown Drone")
    elif pred_class == "DRONE_SIGNAL":
        drone_type = "RF Transmission"
    else:
        drone_type = "None"

    return {
        "deviceId": device_id,
        "timestamp": timestamp,
        "status": "Online",
        "detected": 1 if is_drone else 0,
        "droneType": drone_type,
        "accuracy": round(float(result["confidence"]), 2) if is_drone else 0.0,
        "controlState": "Active" if is_drone else "None",
        "latency": round(float(result.get("latency_ms", 12.5)), 1)  # Maps to server latency expectations
    }


def _post(url: str, api_key: str, payload: dict) -> tuple:
    """
    Executes a secure HTTP POST request against the remote telemetry gateway.
    Injects required cryptographic API access tokens and masks runtime fingerprints.
    """
    # Convert JSON payload dictionary to raw UTF-8 bytes
    body = json.dumps(payload, default=str).encode("utf-8")

    # Absolute strict header configuration matching server security specs
    headers = {
        "Content-Type": "application/json",
        "X-Device-API-Key": str(api_key).strip(),
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Edge/104.0.101"  # Overrides Python signature
    }

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            # Code 200/201 Success path
            return True, resp.status, resp.read().decode()

    except urllib.error.HTTPError as exc:
        # Handles server-side rejections (e.g., 401 Unauthorized Key, 400 Bad Payload)
        return False, exc.code, exc.read().decode()

    except urllib.error.URLError as exc:
        # Handles lower-level infrastructure routing drops (e.g., wrong IP port, DNS timeout)
        return False, 0, f"Network unreachable: {exc.reason}"

    except Exception as exc:
        return False, 0, f"Unexpected pipeline anomaly: {str(exc)}"


def _post_with_retry(url: str, api_key: str, payload: dict) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        ok, code, body = _post(url, api_key, payload)
        if ok:
            return True
        wait = RETRY_BASE_S * (2 ** (attempt - 1))
        print(f"[Telemetry] POST failed attempt={attempt}/{MAX_RETRIES} status={code}")
        if attempt < MAX_RETRIES:
            time.sleep(wait)
    return False


class TelemetrySender:
    def __init__(self, env_path: str = ".env"):
        env = _load_env(env_path)
        self.api_url   = os.environ.get("API_URL",   env.get("API_URL"))
        self.api_key   = os.environ.get("API_KEY",   env.get("API_KEY"))
        self.device_id = int(os.environ.get("DEVICE_ID", env.get("DEVICE_ID", 1001)))

        if not self.api_url or not self.api_key:
            raise ValueError("Missing API_URL or API_KEY in environment configuration.")

        base           = self.api_url.rstrip("/")
        path           = "/api/v1/telemetry/log"
        self._endpoint = base if base.endswith(path) else base + path
        self._queue    = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._stop     = threading.Event()
        self._sent     = 0
        self._failed   = 0

        self._thread = threading.Thread(target=self._worker, name="Telemetry-Sender", daemon=True)
        self._thread.start()

    def send(self, result: dict) -> None:
        payload = build_payload(result, self.device_id)
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(payload)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
        print(f"[Telemetry] Stopped. Sent successfully: {self._sent} | Failed: {self._failed}")

    def _worker(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            ok = _post_with_retry(self._endpoint, self.api_key, payload)
            if ok:
                self._sent += 1
                label = f"{payload['droneType']} {payload['accuracy']*100:.1f}%" if payload["detected"] else "NO_DRONE"
                print(f"[Telemetry] ✓ Sent packet: {label}")
            else:
                self._failed += 1


# ===========================================================================
# 2. Continuous Loop Infinite Harness
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  TelemetrySender — Infinite Stress-Testing Continuous Loop")
    print("=" * 65)

    # 1. Enforce a backup structural configuration path if missing
    if not os.path.exists(".env"):
        Path(".env").write_text("API_URL=http://localhost:8082\nAPI_KEY=YOUR_KEY\nDEVICE_ID=101\n")

    try:
        sender = TelemetrySender(env_path=".env")
    except ValueError as err:
        print(f"[Initialization Fatal Error] {err}")
        exit(1)

    print(f"[Running] Target Endpoint: {sender._endpoint}")
    print("[Running] Starting continuous loop transmission layout. Press Ctrl + C to exit safely...\n")

    loop_idx = 1

    try:
        while True:
            # 2. Randomly synthesize a realistic 3-class model classification output dictionary
            choice = random.choice(["DRONE", "DRONE_SIGNAL", "NO_DRONE"])
            confidence = random.uniform(0.75, 0.99)

            mock_result = {
                "class": choice,
                "confidence": confidence,
                "latency_ms": random.uniform(10.0, 25.0)
            }

            # Optional extra parameters from your updated server schema spec block
            if choice == "DRONE":
                mock_result["controlState"] = random.choice(["Approaching", "Hovering", "Tracking"])
            else:
                mock_result["controlState"] = None

            print(f"[Packet {loop_idx:04d}] Enqueuing random mock telemetry layout: {choice}")
            sender.send(mock_result)

            # 3. Time separation gap delay (change 2.0 to make it faster or slower)
            time.sleep(2.0)
            loop_idx += 1

    except KeyboardInterrupt:
        print("\n\n[OS Intercept] Ctrl + C detected. Gracefully flushing pipelines and closing socket tunnels...")
    finally:
        sender.stop(timeout=3.0)
        print("=" * 65)