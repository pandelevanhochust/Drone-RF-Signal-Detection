"""
segment_bin_sliding.py
----------------------
Segments a single .bin (SC16) IQ file into overlapping 80 ms STFT spectrograms
using a sliding window: [0→80ms], [step→step+80ms], [2*step→2*step+80ms], …

Usage:
    python segment_bin_sliding.py \
        --file  path/to/file.bin \
        --out   output_images/ \
        [--fs 60e6] [--nfft 1024] [--duration_ms 80] [--step_ms 1]
"""

import argparse
import os
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
                        nfft: int = 1024):
    """Read one window from the .bin file and return (f, t, Zxx)."""
    samples_to_skip = int(sample_rate * (start_ms   / 1000.0))
    samples_to_read = int(sample_rate * (duration_ms / 1000.0))

    start_idx = 2 * samples_to_skip
    end_idx   = start_idx + 2 * samples_to_read

    data_map  = np.memmap(file_path, dtype=np.int16, mode='r')
    raw_chunk = data_map[start_idx:end_idx]

    # Normalize SC16 → float32 [-1, 1] and remove DC offset
    i_ch = raw_chunk[0::2].astype(np.float32) / 32768.0
    q_ch = raw_chunk[1::2].astype(np.float32) / 32768.0
    i_ch -= np.mean(i_ch)
    q_ch -= np.mean(q_ch)

    iq = i_ch + 1j * q_ch

    f, t, Zxx = stft(iq, sample_rate,
                     return_onesided=False,
                     window=windows.hamming(nfft),
                     nperseg=nfft)

    f   = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    return f, t + (start_ms / 1000.0), Zxx


def save_spectrogram(f, t, Zxx, out_path: str) -> None:
    spec_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-12)

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
                         step_ms: float     = 1.0,
                         nfft: int          = 1024) -> None:

    total_ms = file_duration_ms(file_path, sample_rate)

    if total_ms < duration_ms:
        print(f"File too short ({total_ms:.1f} ms) for a {duration_ms} ms window — aborted.")
        return

    # All valid start positions: 0, step_ms, 2*step_ms, … where start+duration <= total
    starts_ms = np.arange(0.0, total_ms - duration_ms + step_ms, step_ms)
    starts_ms = starts_ms[starts_ms + duration_ms <= total_ms]

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(file_path))[0]

    print(f"File     : {file_path}")
    print(f"Duration : {total_ms:.1f} ms")
    print(f"Window   : {duration_ms} ms,  step {step_ms} ms")
    print(f"Segments : {len(starts_ms)}")
    print(f"Output   : {out_dir}")
    print("─" * 50)

    for idx, start in enumerate(starts_ms):
        plt.close('all')
        f, t, Zxx = compute_spectrogram(file_path, sample_rate,
                                        float(start), duration_ms, nfft)
        tag      = f"seg{idx:04d}_start{start:.1f}ms"
        out_name = f"{stem}__{tag}.png"
        out_path = os.path.join(out_dir, out_name)
        save_spectrogram(f, t, Zxx, out_path)

        # Progress every 100 segments to avoid flooding stdout
        if (idx + 1) % 100 == 0 or (idx + 1) == len(starts_ms):
            print(f"  [{idx+1}/{len(starts_ms)}]  {out_name}")

    print("─" * 50)
    print(f"Done. {len(starts_ms)} images saved to '{out_dir}'.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sliding-window STFT spectrogram export from a single .bin (SC16) file."
    )
    parser.add_argument("--file",        required=True,       help="Input .bin file")
    parser.add_argument("--out",         default="output_images", help="Output directory")
    parser.add_argument("--fs",          type=float, default=60e6,  help="Sample rate (Hz)")
    parser.add_argument("--nfft",        type=int,   default=1024,  help="FFT size")
    parser.add_argument("--duration_ms", type=float, default=80.0,  help="Window length (ms)")
    parser.add_argument("--step_ms",     type=float, default=1.0,
                        help="Slide step in ms (default 1 ms)")
    args = parser.parse_args()

    segment_file_sliding(
        file_path   = args.file,
        out_dir     = args.out,
        sample_rate = args.fs,
        duration_ms = args.duration_ms,
        step_ms     = args.step_ms,
        nfft        = args.nfft,
    )