"""
segment_single_file.py
----------------------
Segments a single .dat IQ file into 2–3 random 80 ms STFT spectrograms
and saves each as a PNG image.

Usage:
    python segment_single_file.py \
        --file path/to/AIR_1110_00.dat \
        --out  output_images/ \
        [--fs 60e6] [--center_freq 2.4375e9] [--nfft 1024]
        [--duration_ms 80] [--n_segments 3] [--seed 42]
"""

import argparse
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
from scipy.signal import stft, windows


# ── core DSP ──────────────────────────────────────────────────────────────────

def file_duration_ms(file_path: str, sample_rate: float) -> float:
    """Return the total recording length in milliseconds."""
    n_floats = os.path.getsize(file_path) // 4   # float32 = 4 bytes
    n_samples = n_floats // 2                     # interleaved I/Q
    return (n_samples / sample_rate) * 1000.0


def compute_spectrogram(file_path: str,
                        sample_rate: float,
                        start_ms: float,
                        duration_ms: float = 80.0,
                        nfft: int = 1024):
    """
    Read one time slice of an IQ .dat file and return (f, t, Zxx).
    The file stores interleaved float32 I/Q samples (GNU Radio convention).
    """
    skip_samples = int(sample_rate * (start_ms / 1000.0))
    num_samples  = int(sample_rate * (duration_ms / 1000.0))

    start_idx = 2 * skip_samples
    end_idx   = start_idx + 2 * num_samples

    data_map  = np.memmap(file_path, dtype=np.float32, mode='r')
    raw_chunk = data_map[start_idx:end_idx]

    i_ch = raw_chunk[0::2]
    q_ch = raw_chunk[1::2]
    iq   = i_ch + 1j * q_ch

    f, t, Zxx = stft(iq, sample_rate,
                     return_onesided=False,
                     window=windows.hamming(nfft),
                     nperseg=nfft)

    f    = np.fft.fftshift(f)
    Zxx  = np.fft.fftshift(Zxx, axes=0)

    return f, t + (start_ms / 1000.0), Zxx


# ── plotting ──────────────────────────────────────────────────────────────────

def save_spectrogram(f, t, Zxx, center_freq: float,
                     out_path: str, title: str = "") -> None:
    spec_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def segment_file(file_path: str,
                 out_dir: str,
                 sample_rate: float = 60e6,
                 center_freq: float = 2.4375e9,
                 duration_ms: float = 80.0,
                 nfft: int = 1024,
                 n_segments: int | None = None,
                 seed: int | None = None) -> None:

    if seed is not None:
        random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    total_ms   = file_duration_ms(file_path, sample_rate)
    max_start  = total_ms - duration_ms

    if max_start <= 0:
        raise ValueError(
            f"File is only {total_ms:.1f} ms long; "
            f"cannot extract a {duration_ms} ms segment."
        )

    k = n_segments if n_segments is not None else random.randint(2, 3)

    # Draw k non-overlapping start positions
    starts_ms: list[float] = []
    attempts  = 0
    while len(starts_ms) < k and attempts < 10_000:
        attempts += 1
        candidate = random.uniform(0, max_start)
        # Reject if it overlaps any accepted start
        if all(abs(candidate - s) >= duration_ms for s in starts_ms):
            starts_ms.append(candidate)

    if len(starts_ms) < k:
        # Fall back to evenly spaced starts if random search failed
        starts_ms = [i * (max_start / k) for i in range(k)]

    stem    = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n{stem}  ({total_ms:.1f} ms)  →  {k} segment(s)")

    for idx, start in enumerate(sorted(starts_ms)):
        f, t, Zxx = compute_spectrogram(file_path, sample_rate,
                                        start, duration_ms, nfft)
        tag      = f"seg{idx:02d}_start{start:.0f}ms"
        out_name = f"{stem}__{tag}.png"
        out_path = os.path.join(out_dir, out_name)
        title    = (f"{stem}  |  segment {idx}  |  "
                    f"start={start:.1f} ms  |  {center_freq/1e9:.4f} GHz")
        save_spectrogram(f, t, Zxx, center_freq, out_path, title)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Segment a single .dat IQ file into STFT PNG images.")
    p.add_argument("--file",        required=True,        help="Path to the .dat file")
    p.add_argument("--out",         default="output_images", help="Output directory")
    p.add_argument("--fs",          type=float, default=60e6,      help="Sample rate (Hz)")
    p.add_argument("--center_freq", type=float, default=2.4375e9,  help="Centre frequency (Hz)")
    p.add_argument("--nfft",        type=int,   default=1024,      help="FFT size")
    p.add_argument("--duration_ms", type=float, default=80.0,      help="Segment length (ms)")
    p.add_argument("--n_segments",  type=int,   default=None,      help="Fixed # of segments (default: random 2–3)")
    p.add_argument("--seed",        type=int,   default=None,      help="Random seed for reproducibility")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    segment_file(
        file_path   = args.file,
        out_dir     = args.out,
        sample_rate = args.fs,
        center_freq = args.center_freq,
        duration_ms = args.duration_ms,
        nfft        = args.nfft,
        n_segments  = args.n_segments,
        seed        = args.seed,
    )

# C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe single_segmentation.py --file droneV2_data/MAV_1110_00.dat  --out  output_images/
