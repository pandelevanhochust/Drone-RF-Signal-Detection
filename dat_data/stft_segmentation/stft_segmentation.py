"""
segment_dataset.py
------------------
Walks the DroneDetect_V2 dataset tree and converts every .dat file into
overlapping 80 ms STFT spectrogram images using a sliding window.

Window:  80 ms, step = --step_ms (default 1 ms)
Segments: [0→80], [1→81], [2→82], … until start + 80 ms > file length

Usage:
    python segment_dataset.py \
        --root  ~/toanlv/DroneDetect_V2 \
        --out   output_spectrograms/ \
        [--fs 60e6] [--center_freq 2.4375e9] [--nfft 1024]
        [--duration_ms 80] [--step_ms 1] [--workers 4]
"""

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft, windows


# ── core DSP ──────────────────────────────────────────────────────────────────

def file_duration_ms(file_path: str, sample_rate: float) -> float:
    n_floats  = os.path.getsize(file_path) // 4
    n_samples = n_floats // 2
    return (n_samples / sample_rate) * 1000.0


def compute_spectrogram(file_path: str,
                        sample_rate: float,
                        start_ms: float,
                        duration_ms: float = 80.0,
                        nfft: int = 1024):
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

    f   = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    return f, t + (start_ms / 1000.0), Zxx


def save_spectrogram(f, t, Zxx, center_freq: float, out_path: str) -> None:
    spec_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ── per-file worker ────────────────────────────────────────────────────────────

def process_one_file(task: dict) -> tuple[str, list[str], str | None]:
    """
    Slides an 80 ms window across the file in steps of `step_ms`.
    Produces floor((total_ms - duration_ms) / step_ms) + 1 images.
    """
    file_path   = task["file_path"]
    out_dir     = task["out_dir"]
    sample_rate = task["sample_rate"]
    center_freq = task["center_freq"]
    duration_ms = task["duration_ms"]
    step_ms     = task["step_ms"]
    nfft        = task["nfft"]

    saved = []
    try:
        total_ms = file_duration_ms(file_path, sample_rate)

        if total_ms < duration_ms:
            return file_path, [], (
                f"File too short ({total_ms:.1f} ms) for a {duration_ms} ms window — skipped."
            )

        # Build the list of start positions: 0, step_ms, 2*step_ms, …
        # Last valid start is the largest value where start + duration_ms <= total_ms
        starts_ms = np.arange(0.0, total_ms - duration_ms + step_ms, step_ms)
        starts_ms = starts_ms[starts_ms + duration_ms <= total_ms]

        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(file_path))[0]

        for idx, start in enumerate(starts_ms):
            plt.close('all')
            f, t, Zxx = compute_spectrogram(file_path, sample_rate,
                                            float(start), duration_ms, nfft)
            tag      = f"seg{idx:04d}_start{start:.1f}ms"
            out_name = f"{stem}__{tag}.png"
            out_path = os.path.join(out_dir, out_name)
            save_spectrogram(f, t, Zxx, center_freq, out_path)
            saved.append(out_path)

    except Exception:
        return file_path, saved, traceback.format_exc()

    return file_path, saved, None


# ── dataset walker ─────────────────────────────────────────────────────────────

def collect_tasks(root_dir: str, out_root: str,
                  sample_rate: float, center_freq: float,
                  duration_ms: float, step_ms: float,
                  nfft: int) -> list[dict]:

    root  = Path(root_dir).expanduser().resolve()
    tasks = []

    SKIP_FOLDERS = {"BLUE", "WIFI","CLEAN"}

    for dat_file in sorted(root.rglob("*.dat")):
        rel_parts = dat_file.relative_to(root).parts
        if rel_parts[0] in SKIP_FOLDERS:
            continue
        out_dir = os.path.join(out_root, *rel_parts[:-1])

        tasks.append({
            "file_path":   str(dat_file),
            "out_dir":     out_dir,
            "sample_rate": sample_rate,
            "center_freq": center_freq,
            "duration_ms": duration_ms,
            "step_ms":     step_ms,
            "nfft":        nfft,
        })

    return tasks


def run_dataset(root_dir: str,
                out_root: str  = "output_spectrograms",
                sample_rate: float = 60e6,
                center_freq: float = 2.4375e9,
                duration_ms: float = 80.0,
                step_ms: float     = 1.0,
                nfft: int          = 1024,
                workers: int       = 1) -> None:

    tasks = collect_tasks(root_dir, out_root, sample_rate, center_freq,
                          duration_ms, step_ms, nfft)

    if not tasks:
        print(f"No .dat files found under '{root_dir}'. Check the path.")
        sys.exit(1)

    # Estimate segments per file for display
    est_segs = int((2000.0 - duration_ms) / step_ms) + 1  # rough estimate for a 2 s file
    print(f"Found {len(tasks)} .dat file(s) under '{root_dir}'")
    print(f"Output root   : {out_root}")
    print(f"Workers       : {workers}")
    print(f"Window        : {duration_ms} ms,  step {step_ms} ms")
    print(f"Est. segs/file: ~{est_segs}  (exact count depends on file length)")
    print("─" * 60)

    ok_count  = 0
    err_count = 0
    img_count = 0

    if workers <= 1:
        for i, task in enumerate(tasks):
            file_path, saved, err = process_one_file(task)
            rel = os.path.relpath(file_path, root_dir)
            if err:
                print(f"[{i+1}/{len(tasks)}] ERROR  {rel}\n  {err}")
                err_count += 1
            else:
                print(f"[{i+1}/{len(tasks)}] OK     {rel}  ({len(saved)} images)")
                ok_count  += 1
                img_count += len(saved)
    else:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(process_one_file, t): t for t in tasks}
            done    = 0
            for fut in as_completed(futures):
                done += 1
                file_path, saved, err = fut.result()
                rel = os.path.relpath(file_path, root_dir)
                if err:
                    print(f"[{done}/{len(tasks)}] ERROR  {rel}\n  {err}")
                    err_count += 1
                else:
                    print(f"[{done}/{len(tasks)}] OK     {rel}  ({len(saved)} images)")
                    ok_count  += 1
                    img_count += len(saved)

    print("─" * 60)
    print(f"Done.  {ok_count} files OK, {err_count} errors, {img_count} images saved.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Sliding-window STFT spectrogram export from DroneDetect_V2 .dat files."
    )
    p.add_argument("--root",        required=True,
                   help="Root directory of the dataset")
    p.add_argument("--out",         default="output_spectrograms",
                   help="Root output directory")
    p.add_argument("--fs",          type=float, default=60e6,     help="Sample rate (Hz)")
    p.add_argument("--center_freq", type=float, default=2.4375e9, help="Centre frequency (Hz)")
    p.add_argument("--nfft",        type=int,   default=1024,     help="FFT size")
    p.add_argument("--duration_ms", type=float, default=80.0,     help="Window length (ms)")
    p.add_argument("--step_ms",     type=float, default=1.0,
                   help="Slide step in ms (default 1 ms → ~1921 segs for a 2 s file)")
    p.add_argument("--workers",     type=int,   default=1,
                   help="Parallel worker processes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dataset(
        root_dir    = args.root,
        out_root    = args.out,
        sample_rate = args.fs,
        center_freq = args.center_freq,
        duration_ms = args.duration_ms,
        step_ms     = args.step_ms,
        nfft        = args.nfft,
        workers     = args.workers,
    )