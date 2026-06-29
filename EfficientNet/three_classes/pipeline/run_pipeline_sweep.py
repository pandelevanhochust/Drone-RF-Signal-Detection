"""
run_pipeline.py
===============
Entry point: wires usrp_capture, stft_preprocessor, and drone_inference
together into the continuous live 3-class detection pipeline with
frequency sweep.

Sweep behaviour
---------------
    2.400 → 2.425 → 2.450 → 2.475 → 2.500 →
    2.525 → 2.550 → 2.575 → 2.600 → (wrap back to 2.400)

    5 frames captured per step before retuning.
    Sweep always continues regardless of detection result.
    Each frame printed includes its capture frequency.

Threading model
---------------
    USRP-Sweep-Capture thread  (background daemon)
        Manages retune loop + capture_frame() in a tight loop.
        Pushes (freq_hz, iq) tuples onto frame_queue.

    Main thread
        Pulls (freq_hz, iq) from frame_queue.
        Calls iq_to_spectrogram()   [CPU, ~12 ms]
        Calls inferencer.run()      [NPU, ~22 ms]
        Prints one result line per frame including frequency.

Usage
-----
    python3 run_pipeline.py
    python3 run_pipeline.py --gain 30
    python3 run_pipeline.py --cpu
    python3 run_pipeline.py --save_dir debug_specs/ --no_infer
    python3 run_pipeline.py --no_sweep   # fixed at 2.400 GHz (debug)
"""

import argparse
import os
import queue
import signal
import threading
import time

import numpy as np

from usrp_capture      import (open_usrp, close_usrp, start_capture_thread,
                                SWEEP_FREQS_HZ, FRAMES_PER_STEP, NUM_SWEEP_STEPS)
from stft_preprocessor import iq_to_spectrogram, iq_to_spectrogram_debug, save_spectrogram_png
from drone_inference   import DroneInferencer
from telemetry_sender  import TelemetrySender

IMG_H, IMG_W = 256, 512

# Per-class console alert tags
_ALERT = {
    "DRONE"        : "  ⚠⚠ DRONE VIDEO",
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
    """
    Pull (freq_hz, iq) frames → STFT → 3-class NPU inference → telemetry.

    Each console line includes the capture frequency so you can correlate
    detections to specific channels across the sweep cycle.

    Parameters
    ----------
    frame_queue : Queue of (freq_hz: int, iq: np.ndarray) tuples
    inferencer  : DroneInferencer instance (None if no_infer=True)
    stop_event  : set by SIGINT handler to exit cleanly
    sender      : TelemetrySender instance (None to skip telemetry)
    save_dir    : if set, saves each spectrogram PNG named by freq + frame idx
    no_infer    : if True, only runs STFT (skips inference and telemetry)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"[Pipeline] Saving spectrograms to {save_dir}/\n")

    frame_idx = 0

    while not stop_event.is_set():

        # ── Pull next (freq_hz, iq) from sweep capture thread ─────────────────
        try:
            item = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # Unpack — freq_hz tells us which channel this frame belongs to
        freq_hz, iq = item
        freq_ghz    = freq_hz / 1e9

        # ── STFT on CPU ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if frame_idx == 0:
            print("[Pipeline] First frame — saving debug intermediates to debug_stft/")
            tensor = iq_to_spectrogram_debug(iq, debug_dir="debug_stft")
        else:
            tensor = iq_to_spectrogram(iq)
        stft_ms = (time.perf_counter() - t0) * 1000

        # ── Optional debug save — named by frequency and frame index ──────────
        if save_dir:
            fname = f"frame_{frame_idx:06d}_{freq_ghz:.4f}GHz.png"
            save_spectrogram_png(tensor, os.path.join(save_dir, fname))

        # ── Inference disabled ────────────────────────────────────────────────
        if no_infer or inferencer is None:
            print(
                f"[Frame {frame_idx:05d}]"
                f"  {freq_ghz:.4f} GHz"
                f"  STFT {stft_ms:5.1f} ms"
                f"  (inference disabled)"
            )
            frame_idx += 1
            continue

        # ── NPU inference ─────────────────────────────────────────────────────
        result = inferencer.run(tensor)

        # Attach capture frequency to result for telemetry context
        result["freq_hz"] = freq_hz

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
            f"  {freq_ghz:.4f} GHz"
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
        description="USRP X300 → STFT → 3-class Drone Detection with frequency sweep"
    )
    p.add_argument("--model",        default="../new_three_classes.tflite")
    p.add_argument("--labels",       default="class_names.txt",
                   help="One class per line: DRONE, DRONE_SIGNAL, NO_DRONE")
    p.add_argument("--addr",         default="192.168.5.115")
    p.add_argument("--gain",         type=float, default=35.0)
    p.add_argument("--cpu",          action="store_true",
                   help="Disable NPU delegate, run on CPU only")
    p.add_argument("--threshold",    type=float, default=0.70)
    p.add_argument("--save_dir",     default=None,
                   help="Save debug spectrogram PNGs (named by freq + frame idx)")
    p.add_argument("--no_infer",     action="store_true",
                   help="STFT only — skip inference (spectrogram debugging)")
    p.add_argument("--no_telemetry", action="store_true")
    p.add_argument("--no_sweep",     action="store_true",
                   help="Disable sweep — fixed at 2.400 GHz (useful for debugging)")
    p.add_argument("--env",          default=".env")
    p.add_argument("--queue_size",   type=int, default=4)
    return p.parse_args()


def main():
    args = get_args()

    sweep_active = not args.no_sweep
    cycle_s      = NUM_SWEEP_STEPS * FRAMES_PER_STEP * 0.115   # approx

    sep = "=" * 68
    print(f"\n{sep}")
    print("  USRP X300 → STFT → 3-Class Drone Detection  (RB3 Gen 2 NPU)")
    print(sep)
    print(f"  Model      : {args.model}")
    print(f"  Labels     : {args.labels}  (DRONE | DRONE_SIGNAL | NO_DRONE)")
    print(f"  USRP addr  : {args.addr}")
    print(f"  Backend    : {'CPU only' if args.cpu else 'NPU (QNN HTP delegate)'}")
    print(f"  Gain       : {args.gain} dB")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Telemetry  : {'disabled' if args.no_telemetry else f'enabled  env={args.env}'}")
    if sweep_active:
        freqs_str = " → ".join(f"{f/1e9:.3f}" for f in SWEEP_FREQS_HZ)
        print(f"  Sweep      : {freqs_str} GHz")
        print(f"             : {FRAMES_PER_STEP} frames/step × "
              f"{NUM_SWEEP_STEPS} steps ≈ {cycle_s:.1f} s/cycle")
    else:
        print(f"  Sweep      : DISABLED — fixed at {SWEEP_FREQS_HZ[0]/1e9:.3f} GHz")
    if args.save_dir:
        print(f"  Save dir   : {args.save_dir}")
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

    # ── Start sweep capture thread ────────────────────────────────────────────
    # Pass usrp=dev to enable sweep; pass usrp=None to stay fixed (--no_sweep)
    cap_thread = start_capture_thread(
        streamer    = streamer,
        metadata    = metadata,
        frame_queue = frame_queue,
        stop_event  = stop_event,
        usrp        = dev if sweep_active else None,
    )

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