"""
telemetry_sender.py
=============================================================================
Sends drone detection results to the telemetry API via HTTP POST.

Refactored Updates
------------------
    - Accuracy payload field now passes the exact raw model confidence score
      for NO_DRONE instead of wiping it to 0.0.
"""

import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import urllib.request
import urllib.error

# ─────────────────────────────────────────────────────────────────────────────
#  .env loader
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


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_RETRIES     = 3
RETRY_BASE_S    = 1.0
QUEUE_MAXSIZE   = 64
REQUEST_TIMEOUT = 5

# 3-class → droneType string mapping
DRONE_TYPE_MAP = {
    "DRONE"        : "Detected",
    "DRONE_SIGNAL" : "DroneSignal",
    "NO_DRONE"     : "None",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Payload builder
# ─────────────────────────────────────────────────────────────────────────────

def build_payload(result: dict, device_id: int) -> dict:
    """
    Convert a DroneInferencer result dict into the API body schema.

    Refactored Logic:
    -----------------
    Always rounds and includes the exact model 'confidence' value into the
    'accuracy' payload field, even when detected is False.
    """
    pred_class = result["class"]
    is_drone   = pred_class != "NO_DRONE"
    timestamp  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "deviceId"     : device_id,
        "timestamp"    : timestamp,
        "status"       : "Online",
        "detected"     : 1 if is_drone else 0,
        "droneType"    : DRONE_TYPE_MAP.get(pred_class, "None"),
        # Fixed: Always map the exact float value from your NPU prediction
        "accuracy"     : round(float(result.get("confidence", 0.0)), 4),
        "controlState" : result.get("controlState", "None" if not is_drone else "Active"),
        "latency"      : round(float(result.get("latency_ms", 12.5)), 1)
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP POST
# ─────────────────────────────────────────────────────────────────────────────

def _post(url: str, api_key: str, payload: dict) -> tuple:
    body    = json.dumps(payload, default=str).encode("utf-8")
    headers = {
        "Content-Type"     : "application/json",
        "X-Device-API-Key" : api_key,
        "User-Agent"       : "Mozilla/5.0 (X11; Linux x86_64) Edge/104.0.101"
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
        print(
            f"[Telemetry] POST failed  attempt={attempt}/{MAX_RETRIES}"
            f"  status={code}  body={body[:120]}"
            + (f"  retrying in {wait:.0f}s ..." if attempt < MAX_RETRIES else "  giving up.")
        )
        if attempt < MAX_RETRIES:
            time.sleep(wait)
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  TelemetrySender
# ─────────────────────────────────────────────────────────────────────────────

class TelemetrySender:
    """
    Non-blocking telemetry dispatcher for 3-class inference results.

    .send(result) enqueues immediately and returns.
    Background daemon thread POSTs with retry logic.
    """

    def __init__(self, env_path: str = ".env"):
        env = _load_env(env_path)

        self.api_url   = os.environ.get("API_URL",   env.get("API_URL"))
        self.api_key   = os.environ.get("API_KEY",   env.get("API_KEY"))
        self.device_id = int(os.environ.get("DEVICE_ID", env.get("DEVICE_ID", 101)))

        if not self.api_url:
            raise ValueError("API_URL not set. Add to .env: API_URL=http://<host>:<port>")
        if not self.api_key:
            raise ValueError("API_KEY not set. Add to .env: API_KEY=YOUR_RAW_API_KEY_HERE")

        base           = self.api_url.rstrip("/")
        path           = "/api/v1/telemetry/log"
        self._endpoint = base if base.endswith(path) else base + path
        self._queue    = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._stop     = threading.Event()
        self._sent     = 0
        self._failed   = 0

        self._thread = threading.Thread(
            target=self._worker, name="Telemetry-Sender", daemon=True
        )
        self._thread.start()

        print(f"[Telemetry] Endpoint  : {self._endpoint}")
        print(f"[Telemetry] Device ID : {self.device_id}")
        print(f"[Telemetry] Queue     : {QUEUE_MAXSIZE}  retries={MAX_RETRIES}\n")

    def send(self, result: dict) -> None:
        """Enqueue one inference result for async POST. Never blocks."""
        payload = build_payload(result, self.device_id)
        if self._queue.full():
            try:
                self._queue.get_nowait()
                print("[Telemetry] WARNING: queue full — oldest frame dropped")
            except queue.Empty:
                pass
        self._queue.put_nowait(payload)

    def stop(self, timeout: float = 5.0) -> None:
        print(f"[Telemetry] Flushing queue ({self._queue.qsize()} pending) ...")
        self._stop.set()
        self._thread.join(timeout=timeout)
        print(f"[Telemetry] Done  sent={self._sent}  failed={self._failed}")

    @property
    def stats(self) -> dict:
        return {"sent": self._sent, "failed": self._failed,
                "queued": self._queue.qsize()}

    def _worker(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            ok = _post_with_retry(self._endpoint, self.api_key, payload)
            if ok:
                self._sent += 1
                label = f"{payload['droneType']} {payload['accuracy']*100:.1f}%" \
                        if payload["detected"] else f"NO_DRONE ({payload['accuracy']*100:.1f}%)"
                print(f"[Telemetry] ✓  ts={payload['timestamp']}  "
                      f"detected={payload['detected']}  {label}")
            else:
                self._failed += 1