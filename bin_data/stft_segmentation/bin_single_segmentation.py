"""
Sliding-window STFT spectrogram export from a single .bin (SC16) IQ file.

Examples:
    # 1/8 overlap (default, 70 ms step for 80 ms window)
    python bin_single_segmentation.py --file recording.bin

    # No overlap / consecutive
    python bin_single_segmentation.py --file recording.bin --overlap 0

    # Manual step
    python bin_single_segmentation.py --file recording.bin --step_ms 40
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft, windows


# ── Core DSP ──────────────────────────────────────────────────────────────────

def file_duration_ms(file_path: str, sample_rate: float) -> float:
    """Total recording length in ms for SC16 .bin files (2 × int16 per sample)."""
    n_ints    = os.path.getsize(file_path) // 2   # total int16 values
    n_samples = n_ints // 2                        # I+Q pairs
    return (n_samples / sample_rate) * 1000.0


def compute_spectrogram(file_path: str,
                        sample_rate: float,
                        start_ms: float,
                        duration_ms: float = 80.0,
                        nfft: int = 1024) -> np.ndarray:
    """Read one window from the SC16 .bin file. Returns spec_db."""
    samples_to_skip = int(sample_rate * (start_ms   / 1000.0))
    samples_to_read = int(sample_rate * (duration_ms / 1000.0))

    start_idx = 2 * samples_to_skip
    end_idx   = start_idx + 2 * samples_to_read

    data_map  = np.memmap(file_path, dtype=np.int16, mode='r')
    raw_chunk = data_map[start_idx:end_idx]

    # SC16 → float32 [-1, 1], remove DC offset
    i_ch = raw_chunk[0::2].astype(np.float32) / 32768.0
    q_ch = raw_chunk[1::2].astype(np.float32) / 32768.0
    i_ch -= np.mean(i_ch)
    q_ch -= np.mean(q_ch)

    iq = i_ch + 1j * q_ch

    _, _, Zxx = stft(iq, sample_rate,
                     return_onesided=False,
                     window=windows.hamming(nfft),
                     nperseg=nfft)

    Zxx     = np.fft.fftshift(Zxx, axes=0)
    spec_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-12)

    return spec_db


def save_spectrogram(spec_db: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ── Sliding-window segmentation ───────────────────────────────────────────────

def segment_file_sliding(file_path: str,
                         out_dir: str,
                         sample_rate: float = 60e6,
                         duration_ms: float = 80.0,
                         step_ms: float     = 70.0,
                         nfft: int          = 1024) -> None:

    total_ms = file_duration_ms(file_path, sample_rate)

    if total_ms < duration_ms:
        print(f"File too short ({total_ms:.1f} ms) for a {duration_ms} ms window — aborted.")
        sys.exit(1)

    starts_ms = np.arange(0.0, total_ms - duration_ms + step_ms, step_ms)
    starts_ms = starts_ms[starts_ms + duration_ms <= total_ms]

    overlap_frac = 1.0 - (step_ms / duration_ms)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(file_path))[0]

    print(f"File     : {file_path}")
    print(f"Duration : {total_ms:.1f} ms")
    print(f"Window   : {duration_ms} ms,  step {step_ms} ms  ({overlap_frac:.1%} overlap)")
    print(f"Segments : {len(starts_ms)}")
    print(f"Output   : {out_dir}")
    print("─" * 50)

    for idx, start in enumerate(starts_ms):
        spec_db  = compute_spectrogram(file_path, sample_rate,
                                       float(start), duration_ms, nfft)
        tag      = f"seg{idx:04d}_start{start:.1f}ms"
        out_name = f"{stem}__{tag}.png"
        out_path = os.path.join(out_dir, out_name)
        save_spectrogram(spec_db, out_path)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(starts_ms):
            print(f"  [{idx+1}/{len(starts_ms)}]  {out_name}")

    print("─" * 50)
    print(f"Done. {len(starts_ms)} images saved to '{out_dir}'.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Sliding-window STFT spectrogram export from a single .bin (SC16) file."
    )
    p.add_argument("--file",        required=True,           help="Input .bin file")
    p.add_argument("--out",         default="output_images", help="Output directory")
    p.add_argument("--fs",          type=float, default=60e6, help="Sample rate (Hz)")
    p.add_argument("--nfft",        type=int,   default=1024, help="FFT size")
    p.add_argument("--duration_ms", type=float, default=80.0, help="Window length (ms)")

    group = p.add_mutually_exclusive_group()
    group.add_argument("--overlap", type=float, default=None,
                       help="Overlap as a fraction 0–<1 (e.g. 0.125 = 1/8, 0 = no overlap).")
    group.add_argument("--step_ms", type=float, default=None,
                       help="Slide step in ms (overrides --overlap).")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.step_ms is not None:
        step_ms = args.step_ms
    elif args.overlap is not None:
        if not (0.0 <= args.overlap < 1.0):
            print("ERROR: --overlap must be in [0, 1).")
            sys.exit(1)
        step_ms = args.duration_ms * (1.0 - args.overlap)
    else:
        step_ms = args.duration_ms * (1.0 - 0.125)  # default: 1/8 overlap → 70 ms

    segment_file_sliding(
        file_path   = args.file,
        out_dir     = args.out,
        sample_rate = args.fs,
        duration_ms = args.duration_ms,
        step_ms     = step_ms,
        nfft        = args.nfft,
    )