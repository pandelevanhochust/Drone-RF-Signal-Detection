"""
capture_and_infer.py
====================
Continuously captures RF signal from a Nuand BladeRF via USB, converts each
80 ms frame into an STFT spectrogram on CPU, and feeds it to the fused drone
detection TFLite pipeline on the RB3 Gen 2 NPU.

Full pipeline per frame
-----------------------
    BladeRF RX  →  IQ samples (int16 SC16Q11)
        │
        ▼  CPU
    Convert to complex float32  →  apply Hann window
        │
        ▼  numpy.fft.fft (STFT, overlap-add)
    power spectrogram  (dB scale, freq × time)
        │
        ▼  resize to (256, 512)  →  convert to 3-ch RGB PNG-style tensor
        │  →  ImageNet normalise  →  NCHW float32 (1, 3, 256, 512)
        │
        ▼  NPU (TFLite / QNN delegate)
    class_logits (1, 8)  →  softmax  →  top-1 class

RF parameters (must match training dataset collection)
------------------------------------------------------
    Center frequency : 2.375 GHz
    Sample rate      : 60 MHz
    Bandwidth        : 60 MHz
    Frame duration   : 80 ms  →  num_samples = 60e6 × 0.080 = 4,800,000
    Gain             : auto (AGC) — override with --gain if needed

STFT parameters  (must match drone_dataloader STFT transform)
-------------------------------------------------------------
    Window size  : 256 samples  (nfft = 256)
    Hop size     : 128 samples  (50% overlap)
    Window type  : Hann
    Output       : log10 power (dB), shape (freq_bins, time_steps)
                   → resized to (256, 512) to match model input

Device setup (run once on RB3 Gen 2)
--------------------------------------
    sudo apt update
    sudo apt install cmake python3-pip libusb-1.0-0 -y
    cd ~ && git clone --depth 1 https://github.com/Nuand/bladeRF.git
    cd bladeRF && mkdir host/build && cd host/build
    cmake ../ -DINSTALL_UDEV_RULES=ON
    make -j4 && sudo make install && sudo ldconfig

    # Add yourself to plugdev so you don't need sudo for USB access
    sudo adduser $USER plugdev
    # Reconnect USB or reboot, then verify:
    bladeRF-cli -p        # should list your device
    bladeRF-cli -e "version"

    # Python bindings
    pip3 install bladerf numpy scipy pillow ai-edge-litert

Usage
-----
    # NPU inference (default)
    python3 capture_and_infer.py

    # CPU inference only (no QNN delegate)
    python3 capture_and_infer.py --cpu

    # Override gain (dB), adjust STFT params, change model path
    python3 capture_and_infer.py --gain 30 --model my_model.tflite

    # Dry run: capture and save spectrograms to disk, skip inference
    python3 capture_and_infer.py --save_dir spectrograms/ --no_infer
"""

import argparse
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter, load_delegate

# ─────────────────────────────────────────────────────────────────────────────
#  RF / STFT constants  (must match training data collection params)
# ─────────────────────────────────────────────────────────────────────────────

CENTER_FREQ_HZ  = 2_375_000_000     # 2.375 GHz
SAMPLE_RATE_HZ  = 60_000_000        # 60 MHz
BANDWIDTH_HZ    = 60_000_000        # 60 MHz
FRAME_DURATION  = 0.080             # 80 ms per frame
NUM_SAMPLES     = int(SAMPLE_RATE_HZ * FRAME_DURATION)   # 4,800,000 per frame

# STFT parameters
NFFT            = 256               # FFT window size (freq resolution)
HOP             = 128               # hop size → 50% overlap
WINDOW          = np.hanning(NFFT).astype(np.float32)

# Model input size
IMG_H, IMG_W    = 256, 512          # spectrogram resize target

# ImageNet normalisation (must match drone_dataloader.get_transforms)
IMAGENET_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Class labels (will be loaded from class_names.txt)
CLASS_NAMES_DEFAULT = ["AIR", "DIS", "INS", "MIN", "MP1", "MP2", "NO_DRONE", "PHA"]

DELEGATE_LIB    = "libQnnTFLiteDelegate.so"


# ─────────────────────────────────────────────────────────────────────────────
#  BladeRF capture
# ─────────────────────────────────────────────────────────────────────────────

def open_bladerf(gain: int = None):
    try:
        import bladerf
        from bladerf import _bladerf  # Clean access to core C-bindings
    except ImportError:
        raise ImportError(
            "bladerf Python bindings not found.\n"
            "  pip3 install bladerf"
        )

    print("[BladeRF] Opening device ...")
    dev = bladerf.BladeRF()
    info = dev.get_devinfo()
    print(f"  ✓ Device  : {info}")

    ch = _bladerf.CHANNEL_RX(0)

    # Frequency
    dev.set_frequency(ch, CENTER_FREQ_HZ)
    print(f"  Center freq : {dev.get_frequency(ch)/1e9:.4f} GHz")

    # Sample rate
    dev.set_sample_rate(ch, SAMPLE_RATE_HZ)
    print(f"  Sample rate : {dev.get_sample_rate(ch)/1e6:.1f} MHz")

    # Bandwidth
    dev.set_bandwidth(ch, BANDWIDTH_HZ)
    print(f"  Bandwidth   : {dev.get_bandwidth(ch)/1e6:.1f} MHz")

    # Gain configuration (Forced manual fallback because hardware lacks calibration tables)
    if gain is not None:
        dev.set_gain_mode(ch, _bladerf.GainMode.Manual)
        dev.set_gain(ch, gain)
        print(f"  Gain        : {gain} dB (manual)")
    else:
        # Default safety fallback to manual mode to circumvent missing AGC table warnings
        fallback_gain = 30
        dev.set_gain_mode(ch, _bladerf.GainMode.Manual)
        dev.set_gain(ch, fallback_gain)
        print(f"  Gain        : {fallback_gain} dB (Forced Manual Fallback - No Device AGC Table)")

    # Configure sync interface
    dev.sync_config(
        layout          = _bladerf.ChannelLayout.RX_X1,
        fmt             = _bladerf.Format.SC16_Q11,
        num_buffers     = 16,
        buffer_size     = 8192,
        num_transfers   = 8,
        stream_timeout  = 3500,
    )

    dev.enable_module(ch, True)
    print(f"  ✓ RX channel enabled\n")
    return dev


def capture_frame(dev) -> np.ndarray:
    """
    Capture exactly NUM_SAMPLES IQ samples (one 80 ms frame) from BladeRF.

    Returns
    -------
    iq : complex64 ndarray of shape (NUM_SAMPLES,)
        Normalised to approximately [-1, 1].

    BladeRF sync_rx reads raw bytes in SC16Q11:
        each sample = [I_int16, Q_int16], values in [-2048, 2047]
    We interleave as int16 and view as complex via:
        (I + jQ) / 2048.0  →  complex float32
    """
    # Buffer: 2 int16 per sample (I and Q), each 2 bytes → 4 bytes per sample
    buf = np.zeros(NUM_SAMPLES * 2, dtype=np.int16)
    dev.sync_rx(buf, NUM_SAMPLES, timeout_ms=5000)

    # Deinterleave I/Q and convert to complex float32
    iq = (buf[0::2].astype(np.float32) +
          1j * buf[1::2].astype(np.float32)) / 2048.0
    return iq.astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
#  STFT → spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def iq_to_spectrogram(iq: np.ndarray) -> np.ndarray:
    """
    Optimized STFT computation. Bypasses the 37,497 column bottleneck
    by downsampling the time axis before running the FFT math.
    """
    # Instead of hopping by 128 samples, calculate a hop size that
    # yields EXACTLY 512 time steps natively!
    # 4,800,000 samples / 512 columns = ~9375 hop size
    TARGET_WIDTH = 512
    OPT_HOP = (len(iq) - NFFT) // (TARGET_WIDTH - 1)

    # Build a tight, tiny index grid of exactly 512 columns instead of 37,497
    idx = np.arange(NFFT)[None, :] + OPT_HOP * np.arange(TARGET_WIDTH)[:, None]
    frames = iq[idx]  # Small matrix: (512, 256) instead of (37497, 256)

    # Apply Hann window and run only 512 FFTs
    frames = frames * WINDOW[None, :]
    spectrum = np.fft.fft(frames, axis=1)
    spectrum = np.fft.fftshift(spectrum, axes=1)

    # Log power calculation
    power = 10.0 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    power = power.T  # Shape: (256, 512) - Already perfectly sized!

    # Local Min/Max normalisation
    p_min, p_max = power.min(), power.max()
    if p_max > p_min:
        power = (power - p_min) / (p_max - p_min)
    else:
        power = np.zeros_like(power)

        # Convert directly to standard RGB format without utilizing PIL resize loops
    uint8_img = (power * 255).clip(0, 255).astype(np.uint8)

    # Fast grayscale to 3-Channel RGB stack using numpy views
    rgb_arr = np.stack([uint8_img] * 3, axis=-1).astype(np.float32) / 255.0

    # ImageNet normalisation
    arr = (rgb_arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC to CHW
    arr = arr[np.newaxis]  # CHW to NCHW
    return arr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  TFLite interpreter
# ─────────────────────────────────────────────────────────────────────────────

def build_interpreter(model_path: str, use_npu: bool):
    """
    Load the LiteRT interpreter with or without the QNN NPU delegate.
    """
    delegates = []
    if use_npu:
        try:
            # Explicitly point to the Hexagon Tensor Processor (HTP) core and skel libraries
            delegate_options = {
                "backend_type": "htp",
                "library_path": "/usr/lib/libQnnHtp.so",
                "skel_library_dir": "/usr/lib/rfsa/adsp"
            }

            qnn = load_delegate(DELEGATE_LIB, options=delegate_options)
            delegates = [qnn]
            print(f"[Setup] QNN delegate loaded successfully (Backend: HTP / Hexagon NPU)")
        except Exception as exc:
            print(f"[Setup] WARNING: could not load QNN delegate: {exc}")
            print(f"[Setup] Falling back to CPU-only inference.")

    interp = Interpreter(
        model_path=model_path,
        experimental_delegates=delegates)
    interp.allocate_tensors()
    return interp


def run_inference(interp, tensor: np.ndarray, class_names: list) -> dict:
    """
    Feed one (1,3,256,512) NCHW float32 tensor, invoke, dequantise, return result.
    """
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    interp.set_tensor(inp_det["index"], tensor)
    t0 = time.perf_counter()
    interp.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = interp.get_tensor(out_det["index"])           # (1, 8) int8 or f32
    scale, zp = out_det["quantization"]
    logits = (raw.astype(np.float32) - zp) * scale if scale != 0.0 else raw.astype(np.float32)

    # Softmax
    logits  -= logits.max()
    exp      = np.exp(np.clip(logits, -500, 500))
    probs    = (exp / exp.sum())[0]                     # (8,)

    pred_idx = int(np.argmax(probs))
    return {
        "class"      : class_names[pred_idx],
        "confidence" : float(probs[pred_idx]),
        "probs"      : probs,
        "latency_ms" : latency_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Producer / consumer threads
# ─────────────────────────────────────────────────────────────────────────────

def capture_thread(dev, frame_queue: queue.Queue, stop_event: threading.Event):
    """
    Runs in a background thread.
    Captures frames continuously and puts them on frame_queue.
    Drops frames if the queue is full (inference is too slow to keep up).
    """
    frame_idx = 0
    while not stop_event.is_set():
        try:
            iq = capture_frame(dev)
            if frame_queue.full():
                frame_queue.get_nowait()    # drop oldest if inference can't keep up
            frame_queue.put_nowait(iq)
            frame_idx += 1
        except Exception as exc:
            print(f"[Capture] Error on frame {frame_idx}: {exc}")
            time.sleep(0.01)


def infer_thread(
    frame_queue  : queue.Queue,
    interp,
    class_names  : list,
    stop_event   : threading.Event,
    save_dir     : str = None,
    no_infer     : bool = False,
):
    """
    Runs in the main thread (or a second thread).
    Pulls IQ frames from frame_queue, converts to spectrogram, runs inference.
    """
    frame_idx = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"[Infer ] Saving spectrograms to {save_dir}/")

    while not stop_event.is_set():
        try:
            iq = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        t_stft0  = time.perf_counter()
        tensor   = iq_to_spectrogram(iq)
        stft_ms  = (time.perf_counter() - t_stft0) * 1000

        # Optionally save spectrogram as PNG for debugging
        if save_dir:
            # tensor is (1,3,H,W) normalised — undo normalise for saving
            arr_save = tensor[0].transpose(1, 2, 0)           # HWC
            arr_save = arr_save * IMAGENET_STD + IMAGENET_MEAN
            arr_save = (arr_save * 255).clip(0, 255).astype(np.uint8)
            path     = os.path.join(save_dir, f"frame_{frame_idx:06d}.png")
            Image.fromarray(arr_save).save(path)

        if no_infer or interp is None:
            print(f"[Frame {frame_idx:05d}]  STFT {stft_ms:.1f} ms  "
                  f"(inference disabled)")
            frame_idx += 1
            continue

        result = run_inference(interp, tensor, class_names)

        # Print one-line summary per frame
        bar = " ".join(
            f"{n}:{p*100:4.1f}%" for n, p in zip(class_names, result["probs"])
        )
        print(
            f"[Frame {frame_idx:05d}]"
            f"  STFT {stft_ms:5.1f}ms"
            f"  NPU {result['latency_ms']:5.1f}ms"
            f"  ▶ {result['class']:8s} {result['confidence']*100:5.1f}%"
            f"  [{bar}]"
        )
        frame_idx += 1


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="BladeRF capture → STFT → drone TFLite inference (RB3 Gen 2)"
    )
    p.add_argument("--model",    default="quantize_model/drone_pipeline_fused_quantized.tflite")
    p.add_argument("--labels",   default="quantize_model/class_names.txt")
    p.add_argument("--gain",     type=int, default=None,
                   help="RX gain in dB (default: AGC)")
    p.add_argument("--cpu",      action="store_true",
                   help="Disable NPU delegate, run inference on CPU")
    p.add_argument("--save_dir", default=None,
                   help="Save debug_old spectrogram PNGs to this folder")
    p.add_argument("--no_infer", action="store_true",
                   help="Capture and build spectrograms only, skip inference")
    p.add_argument("--queue_size", type=int, default=4,
                   help="Max frames buffered between capture and inference (default 4)")
    return p.parse_args()


def main():
    args = get_args()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  BladeRF → STFT → Drone Detection Pipeline")
    print(sep)
    print(f"  Center freq  : {CENTER_FREQ_HZ/1e9:.3f} GHz")
    print(f"  Sample rate  : {SAMPLE_RATE_HZ/1e6:.0f} MHz")
    print(f"  Bandwidth    : {BANDWIDTH_HZ/1e6:.0f} MHz")
    print(f"  Frame length : {FRAME_DURATION*1000:.0f} ms  "
          f"({NUM_SAMPLES:,} samples)")
    print(f"  STFT nfft    : {NFFT}  hop={HOP}  window=Hann")
    print(f"  Model input  : (1, 3, {IMG_H}, {IMG_W})")
    print(f"  Backend      : {'CPU' if args.cpu else 'NPU (QNN HTP delegate)'}")
    print(f"{sep}\n")

    # ── Load class names ──────────────────────────────────────────────────────
    if os.path.exists(args.labels):
        with open(args.labels) as f:
            class_names = [l.strip() for l in f if l.strip()]
    else:
        print(f"[Warning] {args.labels} not found — using defaults")
        class_names = CLASS_NAMES_DEFAULT
    print(f"[Setup] Classes: {class_names}\n")

    # ── Build TFLite interpreter ──────────────────────────────────────────────
    interp = None
    if not args.no_infer:
        if not os.path.exists(args.model):
            raise FileNotFoundError(
                f"Model not found: {args.model}\n"
                "Copy from dev machine:\n"
                "  scp exports/drone_pipeline_fused_quantized.tflite ubuntu@<IP>:~/"
            )
        interp = build_interpreter(args.model, use_npu=not args.cpu)

        # Warmup invoke
        dummy = np.zeros((1, 3, IMG_H, IMG_W), dtype=np.float32)
        inp_idx = interp.get_input_details()[0]["index"]
        interp.set_tensor(inp_idx, dummy)
        interp.invoke()
        print("[TFLite] Warmup invoke done\n")

    # ── Open BladeRF ──────────────────────────────────────────────────────────
    dev = open_bladerf(gain=args.gain)

    # ── Threading: capture in background, infer in foreground ─────────────────
    frame_queue = queue.Queue(maxsize=args.queue_size)
    stop_event  = threading.Event()

    cap_thread = threading.Thread(
        target=capture_thread,
        args=(dev, frame_queue, stop_event),
        daemon=True,
        name="BladeRF-Capture",
    )

    # Graceful shutdown on Ctrl+C
    def _sigint(sig, frame):
        print("\n[Shutdown] Stopping ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint)

    print("[Running] Press Ctrl+C to stop\n")
    cap_thread.start()

    # Inference runs in main thread
    infer_thread(
        frame_queue = frame_queue,
        interp      = interp,
        class_names = class_names,
        stop_event  = stop_event,
        save_dir    = args.save_dir,
        no_infer    = args.no_infer,
    )

    cap_thread.join(timeout=3.0)
    from bladerf import _bladerf
    dev.enable_module(_bladerf.CHANNEL_RX(0), False)
    dev.close()
    print("[Done]")


if __name__ == "__main__":
    # Late import so the main() error message is clean if bladerf is missing
    try:
        import bladerf
    except ImportError:
        print("ERROR: bladerf Python package not installed.")
        print("  pip3 install bladerf")
        sys.exit(1)
    main()