"""
telemetry_sender.py
===================
Sends drone detection results to the telemetry API via HTTP POST.

Endpoint
--------
    POST {API_URL}/api/v1/telemetry/log
    Header: Content-Type: application/json
    Header: X-Device-API-Key: {API_KEY}

Body schema
-----------
    {
        "deviceId"  : 101,
        "timestamp" : "2026-05-21T12:00:00Z",   # ISO-8601 UTC
        "status"    : "Online",
        "detected"  : 1,                          # 1 = drone, 0 = no drone
        "droneType" : "MP1",                      # null if no drone
        "accuracy"  : 0.94                        # null if no drone
    }

.env file (place next to this script)
--------------------------------------
    API_URL=http://localhost:8082
    API_KEY=YOUR_RAW_API_KEY_HERE
    DEVICE_ID=101

NO_DRONE handling
-----------------
    When the classifier returns NO_DRONE:
        detected  = 0
        droneType = null
        accuracy  = null

Retry policy
------------
    Failed POST (network error or non-2xx response) is retried up to
    MAX_RETRIES times with exponential back-off (1 s, 2 s, 4 s).
    If all retries fail, the frame is logged and discarded — the pipeline
    never blocks waiting for the API.

Threading
---------
    TelemetrySender runs a background daemon thread with an internal queue.
    .send(result) is non-blocking — it enqueues and returns immediately.
    The background thread drains the queue and handles retries.

Usage as a library
------------------
    from telemetry_sender import TelemetrySender

    sender = TelemetrySender()          # loads .env automatically
    sender.send(inference_result)       # non-blocking

Standalone test
---------------
    python3 telemetry_sender.py
    # Sends two test payloads (drone detected + no drone) to the API
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
#  .env loader  (no third-party dependency — pure stdlib)
# ─────────────────────────────────────────────────────────────────────────────

def _load_env(env_path: str = ".env") -> dict:
    """
    Parse a simple KEY=VALUE .env file.
    Lines starting with # are comments. Inline comments are not supported.
    Values are stripped of surrounding whitespace and optional quotes.
    """
    env = {}
    path = Path(env_path)
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        env[key.strip()] = value
    return env


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

NO_DRONE_CLASS  = "NO_DRONE"
MAX_RETRIES     = 3             # max POST attempts per frame
RETRY_BASE_S    = 1.0           # exponential back-off base (1 s, 2 s, 4 s)
QUEUE_MAXSIZE   = 64            # frames buffered before oldest is dropped
REQUEST_TIMEOUT = 5             # HTTP timeout in seconds

# Human-readable drone type labels — extend as needed
DRONE_TYPE_MAP = {
    "AIR" : "Airo NX",
    "DIS" : "Disco",
    "INS" : "Inspire",
    "MIN" : "Mini",
    "MP1" : "Mavic Pro 1",
    "MP2" : "Mavic Pro 2",
    "PHA" : "Phantom 4",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Payload builder
# ─────────────────────────────────────────────────────────────────────────────

def build_payload(result: dict, device_id: int) -> dict:
    """
    Convert a DroneInferencer result dict into the API body schema.

    Parameters
    ----------
    result : dict from DroneInferencer.run()
        Keys: class (str), confidence (float), probs (ndarray), latency_ms (float)
    device_id : int
        Device identifier from .env (DEVICE_ID)

    Returns
    -------
    dict matching the API schema:
        deviceId  : int
        timestamp : ISO-8601 UTC string
        status    : "Online"
        detected  : 1 | 0
        droneType : str | null
        accuracy  : float (2 d.p.) | null

    NO_DRONE mapping
    ----------------
    When result["class"] == "NO_DRONE":
        detected  = 0
        droneType = null   (JSON null / Python None)
        accuracy  = null
    """
    is_drone  = result["class"] != NO_DRONE_CLASS
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "deviceId"  : device_id,
        "timestamp" : timestamp,
        "status"    : "Online",
        "detected"  : 1 if is_drone else 0,
        "droneType" : DRONE_TYPE_MAP.get(result["class"], result["class"])
                      if is_drone else None,
        "accuracy"  : round(float(result["confidence"]), 4) if is_drone else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP POST  (stdlib urllib — no requests dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _post(url: str, api_key: str, payload: dict) -> tuple:
    """
    Send one POST request.

    Returns
    -------
    (success: bool, status_code: int, body: str)
    """
    body    = json.dumps(payload, default=str).encode("utf-8")
    headers = {
        "Content-Type"    : "application/json",
        "X-Device-API-Key": api_key,
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
    """
    POST with exponential back-off retry.

    Returns True if any attempt succeeded (2xx), False after all retries fail.
    """
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
    Non-blocking telemetry dispatcher.

    .send(result) enqueues the inference result immediately and returns.
    A background daemon thread picks entries from the queue and POSTs them
    to the API with retry logic.

    If the queue is full (API is unreachable for a long time), the oldest
    entry is dropped so the pipeline never stalls.

    Parameters
    ----------
    env_path : str
        Path to .env file.  Defaults to ".env" next to this script.
        Required keys: API_URL, API_KEY, DEVICE_ID

    Example .env
    ------------
        API_URL=http://localhost:8082
        API_KEY=YOUR_RAW_API_KEY_HERE
        DEVICE_ID=101
    """

    def __init__(self, env_path: str = ".env"):
        env = _load_env(env_path)

        self.api_url   = os.environ.get("API_URL",   env.get("API_URL"))
        self.api_key   = os.environ.get("API_KEY",   env.get("API_KEY"))
        self.device_id = int(os.environ.get("DEVICE_ID", env.get("DEVICE_ID", 101)))

        if not self.api_url:
            raise ValueError(
                "API_URL not set.\n"
                "Add it to .env:  API_URL=http://<host>:<port>"
            )
        if not self.api_key:
            raise ValueError(
                "API_KEY not set.\n"
                "Add it to .env:  API_KEY=YOUR_RAW_API_KEY_HERE"
            )

        self._endpoint = f"{self.api_url.rstrip('/')}/api/v1/telemetry/log"
        self._queue    = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._stop     = threading.Event()
        self._sent     = 0
        self._failed   = 0

        self._thread   = threading.Thread(
            target=self._worker, name="Telemetry-Sender", daemon=True
        )
        self._thread.start()

        print(f"[Telemetry] Endpoint  : {self._endpoint}")
        print(f"[Telemetry] Device ID : {self.device_id}")
        print(f"[Telemetry] Queue size: {QUEUE_MAXSIZE}  retries={MAX_RETRIES}\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self, result: dict) -> None:
        """
        Enqueue one inference result for async POST.  Never blocks.

        Parameters
        ----------
        result : dict from DroneInferencer.run()
            Must have keys: class (str), confidence (float)
        """
        payload = build_payload(result, self.device_id)
        if self._queue.full():
            try:
                self._queue.get_nowait()    # drop oldest
                print("[Telemetry] WARNING: queue full — oldest frame dropped")
            except queue.Empty:
                pass
        self._queue.put_nowait(payload)

    def stop(self, timeout: float = 5.0) -> None:
        """
        Flush remaining queue entries and stop the background thread.
        Call this on shutdown before closing the BladeRF device.
        """
        print(f"[Telemetry] Flushing queue ({self._queue.qsize()} pending) ...")
        self._stop.set()
        self._thread.join(timeout=timeout)
        print(f"[Telemetry] Done  sent={self._sent}  failed={self._failed}")

    @property
    def stats(self) -> dict:
        return {"sent": self._sent, "failed": self._failed,
                "queued": self._queue.qsize()}

    # ── Background worker ─────────────────────────────────────────────────────

    def _worker(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            ok = _post_with_retry(self._endpoint, self.api_key, payload)

            if ok:
                self._sent += 1
                drone_info = (
                    f"{payload['droneType']} {payload['accuracy']*100:.1f}%"
                    if payload["detected"] else "NO_DRONE"
                )
                print(
                    f"[Telemetry] ✓ sent"
                    f"  ts={payload['timestamp']}"
                    f"  detected={payload['detected']}"
                    f"  {drone_info}"
                )
            else:
                self._failed += 1


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("  TelemetrySender — standalone test")
    print("=" * 55)

    # Check .env exists
    if not Path(".env").exists():
        print("\nERROR: .env not found in current directory.")
        print("Create it with:\n")
        print("  API_URL=http://localhost:8082")
        print("  API_KEY=YOUR_RAW_API_KEY_HERE")
        print("  DEVICE_ID=101\n")
        sys.exit(1)

    sender = TelemetrySender()

    # ── Test 1: drone detected ────────────────────────────────────────────────
    import numpy as np
    result_drone = {
        "class"      : "MP1",
        "confidence" : 0.9431,
        "probs"      : np.array([0.002, 0.001, 0.011, 0.004, 0.943, 0.027, 0.007, 0.004]),
        "latency_ms" : 22.1,
    }
    print("\nTest 1: drone detected (MP1) ...")
    sender.send(result_drone)

    # ── Test 2: no drone ─────────────────────────────────────────────────────
    result_no_drone = {
        "class"      : "NO_DRONE",
        "confidence" : 0.8831,
        "probs"      : np.array([0.003, 0.002, 0.008, 0.004, 0.007, 0.012, 0.883, 0.002]),
        "latency_ms" : 21.8,
    }
    print("Test 2: no drone detected ...")
    sender.send(result_no_drone)

    # Give worker thread time to flush
    time.sleep(3.0)
    sender.stop()

    print(f"\nFinal stats: {sender.stats}")