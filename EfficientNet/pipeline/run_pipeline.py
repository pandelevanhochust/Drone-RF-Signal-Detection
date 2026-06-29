"""
run_pipeline.py
===============
Điểm khởi chạy: kết nối usrp_capture, stft_preprocessor và drone_inference
thành pipeline phát hiện drone 3 lớp liên tục theo thời gian thực.

Các lớp phân loại:
    DRONE        : Tín hiệu drone rõ ràng (dải dọc đậm trên spectrogram)
    DRONE_SIGNAL : Tín hiệu drone yếu / thoáng qua
    NO_DRONE     : Nhiễu nền không có drone

Mô hình thread:
    Thread USRP-Capture (daemon nền)
        capture_frame() trong vòng lặp liên tục
        đẩy IQ array complex64 vào frame_queue
        bỏ entry cũ nhất khi queue đầy (không bao giờ block capture)

    Main thread
        kéo IQ từ frame_queue
        gọi iq_to_spectrogram()   [CPU, ~12 ms]
        gọi inferencer.run()      [NPU, ~22 ms]
        in một dòng kết quả mỗi frame

Usage
-----
    python3 run_pipeline.py
    python3 run_pipeline.py --gain 30
    python3 run_pipeline.py --cpu
    python3 run_pipeline.py --save_dir debug_specs/ --no_infer
    python3 run_pipeline.py --queue_size 8
"""

import argparse
import os
import queue
import signal
import sys
import threading
import time

import numpy as np
from PIL import Image

from usrp_capture      import open_usrp, close_usrp, start_capture_thread
from stft_preprocessor import iq_to_spectrogram, iq_to_spectrogram_debug, save_spectrogram_png
from drone_inference   import DroneInferencer
from telemetry_sender  import TelemetrySender

IMG_H, IMG_W = 256, 512

# Nhãn cảnh báo hiển thị trên console theo từng lớp
_ALERT = {
    "DRONE"        : "  ⚠⚠ DRONE DETECTED",
    "DRONE_SIGNAL" : "  ~  DRONE SIGNAL",
    "NO_DRONE"     : "",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Main processing loop
# ─────────────────────────────────────────────────────────────────────────────

def processing_loop(
    frame_queue : queue.Queue,
    inferencer  : DroneInferencer,
    stop_event  : threading.Event,
    sender      : TelemetrySender = None,
    save_dir    : str  = None,
    no_infer    : bool = False,
) -> None:
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"[Pipeline] Saving spectrograms to {save_dir}/\n")

    frame_idx = 0
    while not stop_event.is_set():

        # ── Pull next IQ frame ────────────────────────────────────────────────
        try:
            iq = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # ── STFT on CPU ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if frame_idx == 0:
            print("[Pipeline] First frame — saving debug intermediates to debug_stft/")
            tensor = iq_to_spectrogram_debug(iq, debug_dir="debug_stft")
        else:
            tensor = iq_to_spectrogram(iq)
        stft_ms = (time.perf_counter() - t0) * 1000

        # ── Optional debug save ───────────────────────────────────────────────
        if save_dir:
            path = os.path.join(save_dir, f"frame_{frame_idx:06d}.png")
            save_spectrogram_png(tensor, path)

        # ── Inference disabled ────────────────────────────────────────────────
        if no_infer or inferencer is None:
            print(f"[Frame {frame_idx:05d}]  STFT {stft_ms:5.1f} ms  "
                  f"(inference disabled)")
            frame_idx += 1
            continue

        # ── NPU inference ─────────────────────────────────────────────────────
        result = inferencer.run(tensor)

        # ── POST to telemetry API (non-blocking) ──────────────────────────────
        if sender is not None:
            sender.send(result)

        # ── One-line result per frame ─────────────────────────────────────────
        queued = sender.stats["queued"] if sender else 0
        probs  = result["probs"]
        names  = inferencer.class_names   # ["DRONE", "DRONE_SIGNAL", "NO_DRONE"]
        bar    = "  ".join(f"{n}:{p*100:4.1f}%" for n, p in zip(names, probs))
        alert  = _ALERT.get(result["class"], "")
        supp   = "  [suppressed]" if result.get("suppressed") else ""

        print(
            f"[Frame {frame_idx:05d}]"
            f"  STFT {stft_ms:5.1f} ms"
            f"  NPU {result['latency_ms']:5.1f} ms"
            f"  ▶  {result['class']:<14s} {result['confidence']*100:5.1f}%"
            f"  [{bar}]"
            f"  [tx_q={queued}]"
            f"{alert}{supp}"
        )
        frame_idx += 1


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="USRP X300 → STFT → 3-class Drone Detection (RB3 Gen 2 NPU)"
    )
    p.add_argument("--model",        default="../new_three_classes.tflite")
    p.add_argument("--labels",       default="class_names.txt",
                   help="class_names.txt — must contain DRONE, DRONE_SIGNAL, NO_DRONE")
    p.add_argument("--addr",         default="192.168.5.111")
    p.add_argument("--gain",         type=float, default=35.0)
    p.add_argument("--cpu",          action="store_true",
                   help="Disable NPU delegate, run on CPU only")
    p.add_argument("--threshold",    type=float, default=0.70,
                   help="Confidence threshold — below this → forced NO_DRONE (default 0.70)")
    p.add_argument("--save_dir",     default=None)
    p.add_argument("--no_infer",     action="store_true")
    p.add_argument("--no_telemetry", action="store_true")
    p.add_argument("--env",          default=".env")
    p.add_argument("--queue_size",   type=int, default=4)
    return p.parse_args()


def main():
    args = get_args()

    sep = "=" * 64
    print(f"\n{sep}")
    print("  USRP X300 → STFT → 3-Class Drone Detection  (RB3 Gen 2)")
    print(sep)
    print(f"  Model     : {args.model}")
    print(f"  Labels    : {args.labels}  (DRONE | DRONE_SIGNAL | NO_DRONE)")
    print(f"  USRP addr : {args.addr}")
    print(f"  Backend   : {'CPU only' if args.cpu else 'NPU (QNN HTP delegate)'}")
    print(f"  Gain      : {args.gain} dB")
    print(f"  Threshold : {args.threshold}")
    print(f"  Queue     : {args.queue_size} frames")
    print(f"  Telemetry : {'disabled' if args.no_telemetry else f'enabled  env={args.env}'}")
    if args.save_dir:
        print(f"  Save dir  : {args.save_dir}")
    print(f"{sep}\n")

    # ── Telemetry sender ──────────────────────────────────────────────────────
    sender = None
    if not args.no_telemetry and not args.no_infer:
        sender = TelemetrySender(env_path=args.env)

    # ── Inference setup ───────────────────────────────────────────────────────
    inferencer = None
    if not args.no_infer:
        inferencer = DroneInferencer(
            model_path           = args.model,
            labels_path          = args.labels,
            use_npu              = not args.cpu,
            confidence_threshold = args.threshold,
        )

    # ── USRP setup ────────────────────────────────────────────────────────────
    dev, streamer, metadata = open_usrp(addr=args.addr, gain=args.gain)

    # ── Shared state ──────────────────────────────────────────────────────────
    frame_queue = queue.Queue(maxsize=args.queue_size)
    stop_event  = threading.Event()

    # ── Graceful Ctrl+C ───────────────────────────────────────────────────────
    def _sigint(sig, frame):
        print("\n[Shutdown] Ctrl+C received — stopping ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint)

    # ── Start capture thread ──────────────────────────────────────────────────
    cap_thread = start_capture_thread(streamer, metadata, frame_queue, stop_event)

    print("[Running] Ctrl+C to stop\n")

    # ── Processing loop ───────────────────────────────────────────────────────
    try:
        processing_loop(
            frame_queue = frame_queue,
            inferencer  = inferencer,
            stop_event  = stop_event,
            sender      = sender,
            save_dir    = args.save_dir,
            no_infer    = args.no_infer,
        )
    finally:
        stop_event.set()
        cap_thread.join(timeout=3.0)
        if sender:
            sender.stop()
        close_usrp(dev, streamer)
        print("[Done]")


if __name__ == "__main__":
    main()