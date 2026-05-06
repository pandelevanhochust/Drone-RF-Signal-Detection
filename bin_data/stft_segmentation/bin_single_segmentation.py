"""
segment_bin_file.py
-------------------
Segments a single .bin (SC16) IQ file into random 80 ms STFT spectrograms.
Applies normalization (/32768.0) and DC offset removal for clean signals.
"""

import argparse
import os
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Headless mode for server/batch processing
import matplotlib.pyplot as plt
from scipy.signal import stft, windows


# ── Core DSP logic using finalized bin transform ──────────────────────────────

def file_duration_ms(file_path: str, sample_rate: float) -> float:
    """Return the total recording length in milliseconds for SC16 files."""
    # Each sample is 4 bytes (2 bytes I, 2 bytes Q as int16)
    n_ints = os.path.getsize(file_path) // 2
    n_samples = n_ints // 2
    return (n_samples / sample_rate) * 1000.0


def compute_spectrogram_final(file_path: str,
                              sample_rate: float,
                              start_ms: float,
                              duration_ms: float = 80.0,
                              nfft: int = 1024):
    """
    Read a time slice of an SC16 .bin file and return (f, t, Zxx).
    Applies normalization and DC removal to match visualization logic.
    """
    samples_to_skip = int(sample_rate * (start_ms / 1000.0))
    samples_to_read = int(sample_rate * (duration_ms / 1000.0))

    # Calculate indices for the int16 array (2 integers per complex sample)
    start_idx = 2 * samples_to_skip
    end_idx = start_idx + (2 * samples_to_read)

    data_map = np.memmap(file_path, dtype=np.int16, mode='r')
    raw_chunk = data_map[start_idx:end_idx]

    # Convert to float32 and normalize to range [-1.0, 1.0]
    i_ch = raw_chunk[0::2].astype(np.float32) / 32768.0
    q_ch = raw_chunk[1::2].astype(np.float32) / 32768.0

    # DC Offset removal: subtract the mean
    i_ch -= np.mean(i_ch)
    q_ch -= np.mean(q_ch)

    complex_chunk = i_ch + 1j * q_ch

    f, t, Zxx = stft(complex_chunk, sample_rate,
                     return_onesided=False,
                     window=windows.hamming(nfft),
                     nperseg=nfft)

    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    # Shift time vector to the absolute start time
    return f, t + (start_ms / 1000.0), Zxx


# ── Plotting with Vmin/Vmax for cleaner background ───────────────────────────

def save_spectrogram(f, t, Zxx, out_path: str) -> None:
    # Convert to dB with the updated epsilon
    spec_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 6))

    # vmin/vmax set to -120 and -40 to remove "green" noise and blue the background
    ax.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis')

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Segmentation logic ────────────────────────────────────────────────────────

def segment_file(file_path: str,
                 out_dir: str,
                 sample_rate: float = 60e6,
                 center_freq: float = 2.4375e9,
                 duration_ms: float = 80.0,
                 nfft: int = 1024,
                 n_segments: int | None = None,   # None → random 2 or 3
                 seed: int | None = None) -> None:

    if seed is not None:
        random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    total_ms = file_duration_ms(file_path, sample_rate)
    max_start = total_ms - duration_ms

    if max_start <= 0:
        print(f"Skipping {file_path}: File duration too short for {duration_ms}ms segment.")
        return

    # Randomly extract 2 or 3 segments
    k = n_segments if n_segments is not None else random.randint(2, 3)

    starts_ms = []
    attempts = 0
    while len(starts_ms) < k and attempts < 5000:
        attempts += 1
        candidate = random.uniform(0, max_start)
        # Ensure no overlapping segments
        if all(abs(candidate - s) >= duration_ms for s in starts_ms):
            starts_ms.append(candidate)

    if len(starts_ms) < k:
        starts_ms = [i * (max_start / k) for i in range(k)]

    stem = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\nProcessing: {stem} ({total_ms:.1f} ms) → {k} segments")

    for idx, start in enumerate(sorted(starts_ms)):
        f, t, Zxx = compute_spectrogram_final(file_path, sample_rate, start, duration_ms, nfft)
        tag = f"seg{idx:02d}_start{int(start)}ms"
        out_name = f"{stem}__{tag}.png"
        out_path = os.path.join(out_dir, out_name)
        save_spectrogram(f, t, Zxx, out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment .bin IQ files using finalized DSP logic.")
    parser.add_argument("--file", required=True, help="Path to input .bin file")
    parser.add_argument("--out", default="output_images", help="Output directory")
    parser.add_argument("--fs", type=float, default=60e6, help="Sample rate in Hz")
    parser.add_argument("--nfft", type=int, default=1024, help="FFT size")
    parser.add_argument("--duration_ms", type=float, default=80.0, help="Segment length in ms")
    parser.add_argument("--n_segments", type=int, default=None, help="Force fixed number of segments")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    segment_file(
        file_path=args.file,
        out_dir=args.out,
        sample_rate=args.fs,
        duration_ms=args.duration_ms,
        nfft=args.nfft,
        n_segments=args.n_segments,
        seed=args.seed
    )