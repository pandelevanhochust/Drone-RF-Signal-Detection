"""
bin_file_debug.py
=================
Reads a captured .bin IQ file, slices it into consecutive 80 ms segments,
runs the full STFT preprocessing pipeline on each segment, and saves debug
images at every intermediate step.

Supported file formats
-----------------------
    .bin / .dat — interleaved int16 SC16Q11  (BladeRF default USB capture)
                  OR interleaved float32      (GNU Radio / SigMF convention)

    The script auto-detects the format based on --dtype argument.
    BladeRF CLI captures as int16. GNU Radio saves as float32.

Output per segment
------------------
    debug_output/seg_000/
        step1_raw_db.png     raw dB spectrogram via matplotlib viridis
                             (pixel-identical to training script output)
        step2_norm8.png      after min-max normalise + freq flip (greyscale)
        step3_small.png      after resize to (256×512) (greyscale)
        step4_rgb.png        after viridis LUT — true model input colours
        step5_final.png      after ImageNet normalise → undo (== step4)
    debug_output/seg_001/
        ...

Usage
-----
    # BladeRF .bin file (int16 SC16Q11, default)
    python3 bin_file_debug.py --file capture.bin

    # GNU Radio .dat file (float32)
    python3 bin_file_debug.py --file capture.dat --dtype float32

    # Limit to first 5 segments
    python3 bin_file_debug.py --file capture.bin --max_segments 5

    # Custom output directory
    python3 bin_file_debug.py --file capture.bin --out_dir my_debug/

    # Also run inference on each segment (needs TFLite model)
    python3 bin_file_debug.py --file capture.bin --infer

Requirements
------------
    pip install numpy scipy pillow matplotlib
    pip install ai-edge-litert   # only needed with --infer
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import stft as scipy_stft
from scipy.signal.windows import hamming

# ─────────────────────────────────────────────────────────────────────────────
#  Constants — must match training script and stft_preprocessor.py
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE_HZ  = 60_000_000
FRAME_DURATION  = 0.080                             # 80 ms per segment
SAMPLES_PER_FRAME = int(SAMPLE_RATE_HZ * FRAME_DURATION)  # 4,800,000

NFFT            = 1024
NOVERLAP        = 0
WINDOW          = hamming(NFFT)

IMG_H, IMG_W    = 256, 512

IMAGENET_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_VIRIDIS_LUT = (
    matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255
).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  File reader
# ─────────────────────────────────────────────────────────────────────────────

def read_iq_file(file_path: str, dtype: str) -> np.ndarray:
    """
    Read a raw IQ capture file and return a complex64 array.

    Parameters
    ----------
    file_path : path to .bin or .dat file
    dtype     : 'int16'   → BladeRF SC16Q11 (interleaved int16, values ÷ 2048)
                'float32' → GNU Radio / SigMF (interleaved float32)

    Returns
    -------
    iq : complex64 ndarray, shape (N,)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)

    if dtype == "int16":
        # Each IQ sample = 2 × int16 = 4 bytes
        n_values  = file_size // 2          # total int16 values
        n_samples = n_values  // 2          # total IQ pairs
        raw = np.memmap(file_path, dtype=np.int16, mode="r")
        iq  = (raw[0::2].astype(np.float32) +
               1j * raw[1::2].astype(np.float32)) / 2048.0
        print(f"[Read] int16 SC16Q11  {file_size/1e6:.1f} MB  "
              f"→  {n_samples:,} IQ samples  "
              f"({n_samples/SAMPLE_RATE_HZ*1000:.1f} ms)")

    elif dtype == "float32":
        # Each IQ sample = 2 × float32 = 8 bytes
        n_values  = file_size // 4          # total float32 values
        n_samples = n_values  // 2          # total IQ pairs
        raw = np.memmap(file_path, dtype=np.float32, mode="r")
        iq  = raw[0::2] + 1j * raw[1::2]
        iq  = iq.astype(np.complex64)
        print(f"[Read] float32 GNU Radio  {file_size/1e6:.1f} MB  "
              f"→  {n_samples:,} IQ samples  "
              f"({n_samples/SAMPLE_RATE_HZ*1000:.1f} ms)")

    else:
        raise ValueError(f"Unknown dtype: {dtype}. Use 'int16' or 'float32'.")

    return iq.astype(np.complex64)


def slice_segments(iq: np.ndarray) -> list:
    """
    Slice a full IQ array into consecutive non-overlapping 80 ms segments.
    Any trailing samples that don't fill a complete segment are discarded.

    Returns
    -------
    list of complex64 arrays, each of shape (SAMPLES_PER_FRAME,)
    """
    n_segments = len(iq) // SAMPLES_PER_FRAME
    segments   = [
        iq[i * SAMPLES_PER_FRAME : (i + 1) * SAMPLES_PER_FRAME]
        for i in range(n_segments)
    ]
    trailing = len(iq) - n_segments * SAMPLES_PER_FRAME
    print(f"[Slice] {len(iq):,} samples  →  {n_segments} × 80 ms segments  "
          f"({trailing:,} trailing samples discarded)\n")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
#  Full debug pipeline for one segment
# ─────────────────────────────────────────────────────────────────────────────

def process_segment_debug(
    iq       : np.ndarray,
    seg_idx  : int,
    out_dir  : str,
) -> np.ndarray:
    """
    Run the full STFT preprocessing pipeline on one 80 ms IQ segment and
    save a debug PNG at every intermediate step.

    Parameters
    ----------
    iq      : complex64 ndarray (SAMPLES_PER_FRAME,)
    seg_idx : segment index used for folder name and print prefix
    out_dir : root output directory; creates out_dir/seg_NNN/ per segment

    Returns
    -------
    tensor : float32 ndarray (1, 3, 256, 512) — model-ready input
    """
    seg_dir = os.path.join(out_dir, f"seg_{seg_idx:03d}")
    os.makedirs(seg_dir, exist_ok=True)
    prefix  = f"  [seg {seg_idx:03d}]"

    t_total = time.perf_counter()

    # ── Step 1: scipy STFT ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    _, _, Zxx = scipy_stft(
        iq,
        fs             = SAMPLE_RATE_HZ,
        window         = WINDOW,
        nperseg        = NFFT,
        noverlap       = NOVERLAP,
        return_onesided= False,
    )
    Zxx     = np.fft.fftshift(Zxx, axes=0)
    spec_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-10)
    stft_ms = (time.perf_counter() - t0) * 1000

    # Save step1 — via matplotlib (pixel-identical to training script)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    p1 = os.path.join(seg_dir, "step1_raw_db.png")
    plt.savefig(p1, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"{prefix}  step1_raw_db   dB=[{spec_db.min():.1f}, {spec_db.max():.1f}]  "
          f"stft={stft_ms:.0f}ms  → {p1}")

    # ── Step 2: normalise + freq flip ─────────────────────────────────────────
    s_min, s_max = float(spec_db.min()), float(spec_db.max())
    denom = s_max - s_min if s_max > s_min else 1.0
    norm8 = ((spec_db[::-1] - s_min) / denom * 255).clip(0, 255).astype(np.uint8)
    p2    = os.path.join(seg_dir, "step2_norm8.png")
    Image.fromarray(norm8, mode="L").save(p2)
    print(f"{prefix}  step2_norm8    shape={norm8.shape}  → {p2}")

    # ── Step 3: resize to (IMG_H, IMG_W) ─────────────────────────────────────
    small = np.array(
        Image.fromarray(norm8, mode="L").resize((IMG_W, IMG_H), Image.BILINEAR)
    )
    p3    = os.path.join(seg_dir, "step3_small.png")
    Image.fromarray(small, mode="L").save(p3)
    print(f"{prefix}  step3_small    shape={small.shape}  → {p3}")

    # ── Step 4: viridis colormap ──────────────────────────────────────────────
    rgb = _VIRIDIS_LUT[small]
    p4  = os.path.join(seg_dir, "step4_rgb.png")
    Image.fromarray(rgb, mode="RGB").save(p4)
    print(f"{prefix}  step4_rgb      viridis applied  → {p4}")

    # ── Step 5: ImageNet normalise → model tensor ─────────────────────────────
    arr    = rgb.astype(np.float32) / 255.0
    arr    = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    # Save step5 — undo normalise to verify round-trip is lossless
    arr_back = tensor[0].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    p5       = os.path.join(seg_dir, "step5_final.png")
    Image.fromarray((arr_back * 255).clip(0, 255).astype(np.uint8)).save(p5)
    print(f"{prefix}  step5_final    tensor={tensor.shape}  "
          f"range=[{tensor.min():.3f}, {tensor.max():.3f}]  → {p5}")

    total_ms = (time.perf_counter() - t_total) * 1000
    print(f"{prefix}  total={total_ms:.0f}ms\n")

    return tensor


# ─────────────────────────────────────────────────────────────────────────────
#  Optional inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(tensor, model_path, labels_path, use_npu=False):
    """Run TFLite inference on one tensor. Returns result dict."""
    from drone_inference import DroneInferencer
    if not hasattr(run_inference, "_inferencer"):
        run_inference._inferencer = DroneInferencer(
            model_path  = model_path,
            labels_path = labels_path,
            use_npu     = use_npu,
        )
    return run_inference._inferencer.run(tensor)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Read a .bin IQ capture, process each 80ms segment, save debug images"
    )
    p.add_argument("--file",         required=True,
                   help="Path to .bin or .dat IQ capture file")
    p.add_argument("--dtype",        default="int16",
                   choices=["int16", "float32"],
                   help="Sample dtype: 'int16' for BladeRF SC16Q11 (default), "
                        "'float32' for GNU Radio")
    p.add_argument("--out_dir",      default="debug_output",
                   help="Root output directory (default: debug_output/)")
    p.add_argument("--max_segments", type=int, default=None,
                   help="Limit number of segments to process (default: all)")
    p.add_argument("--infer",        action="store_true",
                   help="Run TFLite inference on each segment (needs drone_inference.py)")
    p.add_argument("--model",        default="../quantize_model/drone_pipeline_fused_quantized.tflite",
                   help="TFLite model path (only used with --infer)")
    p.add_argument("--labels",       default="class_names.txt",
                   help="class_names.txt path (only used with --infer)")
    p.add_argument("--cpu",          action="store_true",
                   help="CPU-only inference (only used with --infer)")
    return p.parse_args()


def main():
    args = get_args()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  .bin File Debug — STFT pipeline per segment")
    print(sep)
    print(f"  File     : {args.file}")
    print(f"  Dtype    : {args.dtype}")
    print(f"  Out dir  : {args.out_dir}")
    print(f"  Segments : {args.max_segments or 'all'}")
    print(f"  Infer    : {args.infer}")
    print(f"{sep}\n")

    # ── Read file ─────────────────────────────────────────────────────────────
    iq = read_iq_file(args.file, args.dtype)

    # ── Slice into 80 ms segments ─────────────────────────────────────────────
    segments = slice_segments(iq)

    if args.max_segments is not None:
        segments = segments[:args.max_segments]
        print(f"[Info] Processing first {len(segments)} segment(s)\n")

    # ── Process each segment ──────────────────────────────────────────────────
    results = []
    for i, seg in enumerate(segments):
        tensor = process_segment_debug(seg, i, args.out_dir)

        if args.infer:
            result = run_inference(
                tensor,
                model_path  = args.model,
                labels_path = args.labels,
                use_npu     = not args.cpu,
            )
            results.append(result)
            bar = "  ".join(
                f"{n}:{p*100:4.1f}%"
                for n, p in zip(result.get("probs", []), []) or []
            )
            print(f"  [seg {i:03d}]  ▶  {result['class']:10s}  "
                  f"{result['confidence']*100:.1f}%  "
                  f"NPU={result['latency_ms']:.1f}ms\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(sep)
    print(f"  Done  —  {len(segments)} segment(s) processed")
    print(f"  Debug images saved to: {os.path.abspath(args.out_dir)}/")
    print(f"\n  Folder layout:")
    print(f"    seg_000/step1_raw_db.png   ← compare with training spectrograms")
    print(f"    seg_000/step2_norm8.png    ← greyscale after normalise+flip")
    print(f"    seg_000/step3_small.png    ← greyscale after resize (256×512)")
    print(f"    seg_000/step4_rgb.png      ← viridis RGB (true model input)")
    print(f"    seg_000/step5_final.png    ← round-trip verify (== step4)")

    if args.infer and results:
        from collections import Counter
        dist = Counter(r["class"] for r in results)
        print(f"\n  Inference summary ({len(results)} segments):")
        for cls, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {cls:<12}  {cnt:>3}  ({100*cnt/len(results):.0f}%)")
    print(sep)


if __name__ == "__main__":
    main()