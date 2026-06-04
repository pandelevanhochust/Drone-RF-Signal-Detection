"""
test_telemetry.py
=============================================================================
Standalone infinite loop script to continuously test your Telemetry API connection.
Runs indefinitely, sending randomized mock 3-class data payloads until Ctrl+C.

Refactored Updates:
-------------------
  - Accuracy field now reflects true background classification confidence
    when a drone is NOT detected instead of wiping it to 0.0.
  - Dynamically assigns valid drone strings ("DJI Mavic 3", "RF Transmission", "None")
    and explicit control states matching server ingestion rules.
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
REQUEST_TIMEOUT = 2  # Lowered to match production specs and avoid thread blockages


def build_payload(result: dict, device_id: int) -> dict:
    """
    Convert a DroneInferencer result dict into the strict API body schema.

    Refactored Logic:
    -----------------
    Always maps the explicit model confidence rating directly to the accuracy field,
    guaranteeing true metadata tracking even under negative classifications.
    """
    pred_class = result["class"]
    is_drone = pred_class != "NO_DRONE"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Map classifications to production lookup targets expected by the microservice
    if pred_class == "DRONE":
        drone_type = "DJI Mavic 3"
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
        # Refactored: Preserves raw floating-point accuracy scores across all states
        "accuracy": round(float(result["confidence"]), 4),
        "controlState": "Active" if is_drone else "None",
        "latency": round(float(result.get("latency_ms", 12.5)), 1)
    }


def _post(url: str, api_key: str, payload: dict) -> tuple:
    """
    Executes a secure HTTP POST request against the remote telemetry gateway.
    Injects required cryptographic API access tokens and masks runtime fingerprints.
    """
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
        print(f"[Telemetry] POST failed attempt={attempt}/{MAX_RETRIES} status={code} body={body[:100]}")
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
                print("[Telemetry] WARNING: Simulation buffer full — dropping stale message thread context.")
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
                label = f"{payload['droneType']} ({payload['accuracy']*100:.1f}%)" if payload["detected"] else f"NO_DRONE ({payload['accuracy']*100:.1f}%)"
                print(f"[Telemetry] ✓ Sent packet: {label}")
            else:
                self._failed += 1
            self._queue.task_done()


# ===========================================================================
# 2. Continuous Loop Infinite Harness
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  TelemetrySender — Infinite Stress-Testing Continuous Loop")
    print("=" * 65)

    # 1. Enforce a backup structural configuration path if missing
    if not os.path.exists(".env"):
        Path(".env").write_text("API_URL=http://localhost:80\nAPI_KEY=YOUR_KEY\nDEVICE_ID=1001\n")

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

            print(f"[Packet {loop_idx:04d}] Enqueuing random mock telemetry layout: {choice}")
            sender.send(mock_result)

            # 3. Time separation gap delay (2-second baseline)
            time.sleep(2.0)
            loop_idx += 1

    except KeyboardInterrupt:
        print("\n\n[OS Intercept] Ctrl + C detected. Gracefully flushing pipelines and closing socket tunnels...")
    finally:
        sender.stop(timeout=3.0)
        print("=" * 65)