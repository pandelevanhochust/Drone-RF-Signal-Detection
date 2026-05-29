"""
run_pipeline.py
===============
Entry point: wires bladerf_capture, stft_preprocessor, and drone_inference
together into the continuous live detection pipeline.

Module responsibilities
-----------------------
    bladerf_capture.py    — BladeRF USB open / configure / capture IQ frames
    stft_preprocessor.py  — IQ → STFT log-power spectrogram → model tensor
    drone_inference.py    — TFLite NPU inference → class + probabilities
    run_pipeline.py       — threads, queue, Ctrl+C shutdown, CLI  ← this file

Threading model
---------------
    BladeRF-Capture thread  (background daemon)
        capture_frame() in a tight loop
        pushes complex64 IQ arrays onto frame_queue
        drops oldest entry when queue is full (never blocks capture)

    Main thread
        pulls IQ from frame_queue
        calls iq_to_spectrogram()  [CPU, ~12 ms]
        calls inferencer.run()     [NPU, ~22 ms]
        prints one result line per frame

The queue decouples the USB transfer timing from inference timing.
With maxsize=4 the pipeline tolerates a 4-frame burst without dropping,
and always serves the most recent signal when inference catches up.

Copy files to device
--------------------
    scp bladerf_capture.py    ubuntu@<IP>:/home/ubuntu/
    scp stft_preprocessor.py  ubuntu@<IP>:/home/ubuntu/
    scp drone_inference.py    ubuntu@<IP>:/home/ubuntu/
    scp run_pipeline.py       ubuntu@<IP>:/home/ubuntu/
    scp exports/drone_pipeline_fused_quantized.tflite ubuntu@<IP>:/home/ubuntu/
    scp exports/class_names.txt                        ubuntu@<IP>:/home/ubuntu/

Usage
-----
    source .venv-drone/bin/activate

    # Full pipeline — BladeRF → STFT on CPU → NPU inference
    python3 run_pipeline.py

    # Manual gain instead of AGC
    python3 run_pipeline.py --gain 30

    # CPU inference only (no QNN delegate — useful for debugging)
    python3 run_pipeline.py --cpu

    # Save debug_old spectrogram PNGs without running inference
    python3 run_pipeline.py --save_dir debug_specs/ --no_infer

    # Increase frame buffer if USB and NPU are both under load
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
from stft_preprocessor import iq_to_spectrogram, iq_to_spectrogram_debug, save_spectrogram_png, IMAGENET_MEAN, IMAGENET_STD
from drone_inference   import DroneInferencer
from telemetry_sender  import TelemetrySender

IMG_H, IMG_W = 256, 512


# ─────────────────────────────────────────────────────────────────────────────
#  Main processing loop (runs in main thread)
# ─────────────────────────────────────────────────────────────────────────────

def processing_loop(
    frame_queue : queue.Queue,
    inferencer  : DroneInferencer,
    stop_event  : threading.Event,
    sender      : TelemetrySender = None,
    save_dir    : str  = None,
    no_infer    : bool = False,
) -> None:
    """
    Pull IQ frames from the queue, run STFT on CPU, run inference on NPU,
    and POST results to the telemetry API.

    Parameters
    ----------
    frame_queue : shared queue fed by the BladeRF capture thread
    inferencer  : DroneInferencer instance (None if no_infer=True)
    stop_event  : set by SIGINT handler to exit cleanly
    sender      : TelemetrySender instance (None to skip telemetry)
    save_dir    : if set, saves each spectrogram as a debug_old PNG
    no_infer    : if True, only runs STFT (skips inference and telemetry)
    """
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
            print("[Pipeline] First frame — saving debug_old intermediates to debug_stft/")
            tensor = iq_to_spectrogram_debug(iq, debug_dir="debug_stft")
        else:
            tensor = iq_to_spectrogram(iq)
        stft_ms = (time.perf_counter() - t0) * 1000

        # ── Optional debug_old save ───────────────────────────────────────────────
        if save_dir:
            path = os.path.join(save_dir, f"frame_{frame_idx:06d}.png")
            save_spectrogram_png(tensor, path)

        # ── NPU inference ─────────────────────────────────────────────────────
        if no_infer or inferencer is None:
            print(f"[Frame {frame_idx:05d}]  STFT {stft_ms:5.1f} ms  "
                  f"(inference disabled)")
            frame_idx += 1
            continue

        result = inferencer.run(tensor)

        # ── POST to telemetry API (non-blocking) ──────────────────────────────
        if sender is not None:
            sender.send(result)

        # ── One-line result per frame ─────────────────────────────────────────
        queued = sender.stats["queued"] if sender else 0
        bar = "  ".join(
            f"{n}:{p*100:4.1f}%"
            for n, p in zip(inferencer.class_names, result["probs"])
        )
        print(
            f"[Frame {frame_idx:05d}]"
            f"  STFT {stft_ms:5.1f} ms"
            f"  NPU {result['latency_ms']:5.1f} ms"
            f"  ▶  {result['class']:10s} {result['confidence']*100:5.1f}%"
            f"  [tx_q={queued}]"
            f"  [ {bar} ]"
        )
        frame_idx += 1


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="BladeRF → STFT → Drone detection (RB3 Gen 2 NPU)"
    )
    p.add_argument("--model",      default="../quantize_model/drone_fused.tflite",
                   help="Path to fused quantized TFLite model")
    p.add_argument("--labels",     default="../quantize_model/class_names.txt",
                   help="Path to class_names.txt")
    p.add_argument("--addr",       default="192.168.10.2",
                   help="USRP X300 IP address (default: 192.168.10.2)")
    p.add_argument("--gain",       type=float, default=30.0,
                   help="USRP RX gain in dB, range 0-31.5 (default: 30.0)")
    p.add_argument("--cpu",        action="store_true",
                   help="Disable NPU delegate, run inference on CPU only")
    p.add_argument("--threshold",  type=float, default=0.70,
                   help="Confidence threshold for drone detection (default 0.70)")
    p.add_argument("--save_dir",   default=None,
                   help="Save debug_old spectrogram PNGs to this folder")
    p.add_argument("--no_infer",   action="store_true",
                   help="STFT only — skip inference (useful for spectrogram debugging)")
    p.add_argument("--no_telemetry", action="store_true",
                   help="Disable HTTP telemetry POST (inference still runs)")
    p.add_argument("--env",          default=".env",
                   help="Path to .env file (default: .env)")
    p.add_argument("--queue_size",   type=int, default=4,
                   help="IQ frame queue depth between capture and inference (default 4)")
    return p.parse_args()


def main():
    args = get_args()

    sep = "=" * 62
    print(f"\n{sep}")
    print("  USRP X300 → STFT → Drone Detection  (RB3 Gen 2)")
    print(sep)
    print(f"  Model     : {args.model}")
    print(f"  USRP addr : {args.addr}")
    print(f"  Backend   : {'CPU only' if args.cpu else 'NPU (QNN HTP delegate)'}")
    print(f"  Gain      : {args.gain} dB")
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

    # ── BladeRF setup ─────────────────────────────────────────────────────────
    dev, streamer, metadata = open_usrp(addr=args.addr, gain=args.gain)

    # ── Shared state ──────────────────────────────────────────────────────────
    frame_queue = queue.Queue(maxsize=args.queue_size)
    stop_event  = threading.Event()

    # ── Graceful Ctrl+C shutdown ──────────────────────────────────────────────
    def _sigint(sig, frame):
        print("\n[Shutdown] Ctrl+C received — stopping ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint)

    # ── Start capture thread ──────────────────────────────────────────────────
    cap_thread = start_capture_thread(streamer, metadata, frame_queue, stop_event)

    print("[Running] Ctrl+C to stop\n")

    # ── Processing loop in main thread ────────────────────────────────────────
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
    # try:
    #     import bladerf  # noqa: F401
    # except ImportError:
    #     print("ERROR: bladerf not installed.\n  pip3 install bladerf")
    #     sys.exit(1)
    main()