"""
segment_dataset.py
------------------
Walks the DroneDetect_V2 dataset tree and converts every .dat file into
2–3 randomly-placed 80 ms STFT spectrogram images.

Expected tree (from the README / screenshot):
    DroneDetect_V2/
        BLUE/   BOTH/   CLEAN/   WIFI/
            AIR_FY/ AIR_HO/ AIR_ON/ DIS_FY/ … (20 class folders)
                *.dat

Output mirrors the source tree under <out_root>:
    output_spectrograms/
        BLUE/AIR_FY/AIR_1110_00__seg00_start42ms.png
        …

Usage:
    python segment_dataset.py \
        --root  ~/toanlv/DroneDetect_V2 \
        --out   output_spectrograms/ \
        [--fs 60e6] [--center_freq 2.4375e9] [--nfft 1024]
        [--duration_ms 80] [--n_segments 3] [--seed 42]
        [--workers 4]
"""

import argparse
import os
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # must be before pyplot import
import matplotlib.pyplot as plt
from scipy.signal import stft, windows


# ── core DSP (same as segment_single_file.py) ─────────────────────────────────

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

    f    = np.fft.fftshift(f)
    Zxx  = np.fft.fftshift(Zxx, axes=0)

    return f, t + (start_ms / 1000.0), Zxx


def save_spectrogram(f, t, Zxx, center_freq: float,
                     out_path: str, title: str = "") -> None:
    spec_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ── per-file worker (runs in a subprocess when workers > 1) ───────────────────

def process_one_file(task: dict) -> tuple[str, list[str], str | None]:
    """
    task keys: file_path, out_dir, sample_rate, center_freq,
               duration_ms, nfft, n_segments, seed_offset
    Returns (file_path, [saved_paths], error_message_or_None)
    """
    file_path   = task["file_path"]
    out_dir     = task["out_dir"]
    sample_rate = task["sample_rate"]
    center_freq = task["center_freq"]
    duration_ms = task["duration_ms"]
    nfft        = task["nfft"]
    n_segments  = task["n_segments"]
    seed_offset = task["seed_offset"]

    saved = []
    try:
        rng = random.Random(seed_offset)

        total_ms  = file_duration_ms(file_path, sample_rate)
        max_start = total_ms - duration_ms

        if max_start <= 0:
            return file_path, [], (
                f"File too short ({total_ms:.1f} ms) for a {duration_ms} ms segment — skipped."
            )

        k = n_segments if n_segments is not None else rng.randint(2, 3)

        starts_ms: list[float] = []
        attempts  = 0
        while len(starts_ms) < k and attempts < 10_000:
            attempts += 1
            candidate = rng.uniform(0, max_start)
            if all(abs(candidate - s) >= duration_ms for s in starts_ms):
                starts_ms.append(candidate)

        if len(starts_ms) < k:
            starts_ms = [i * (max_start / k) for i in range(k)]

        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(file_path))[0]

        for idx, start in enumerate(sorted(starts_ms)):
            f, t, Zxx = compute_spectrogram(file_path, sample_rate,
                                            start, duration_ms, nfft)
            tag      = f"seg{idx:02d}_start{start:.0f}ms"
            out_name = f"{stem}__{tag}.png"
            out_path = os.path.join(out_dir, out_name)
            title    = (f"{stem}  |  seg {idx}  |  "
                        f"start={start:.1f} ms  |  {center_freq/1e9:.4f} GHz")
            save_spectrogram(f, t, Zxx, center_freq, out_path, title)
            saved.append(out_path)

    except Exception:
        return file_path, saved, traceback.format_exc()

    return file_path, saved, None


# ── dataset walker ─────────────────────────────────────────────────────────────

def collect_tasks(root_dir: str, out_root: str,
                  sample_rate: float, center_freq: float,
                  duration_ms: float, nfft: int,
                  n_segments: int | None,
                  base_seed: int | None) -> list[dict]:
    """
    Walk root_dir, find every .dat file, build a task dict for each.
    Output path mirrors the source tree under out_root.
    """
    root = Path(root_dir).expanduser().resolve()
    tasks = []

    SKIP_FOLDERS = {"BLUE", "WIFI"}

    for dat_file in sorted(root.rglob("*.dat")):
        # Relative path from root → reconstruct under out_root
        rel_parts = dat_file.relative_to(root).parts   # e.g. ('BOTH','AIR_FY','AIR_1110_00.dat')
        if rel_parts[0] in SKIP_FOLDERS:
            continue
        out_dir   = os.path.join(out_root, *rel_parts[:-1])  # drop the filename

        seed_offset = None
        if base_seed is not None:
            # Deterministic per-file seed derived from the global seed + file index
            seed_offset = base_seed + len(tasks)

        tasks.append({
            "file_path":   str(dat_file),
            "out_dir":     out_dir,
            "sample_rate": sample_rate,
            "center_freq": center_freq,
            "duration_ms": duration_ms,
            "nfft":        nfft,
            "n_segments":  n_segments,
            "seed_offset": seed_offset,
        })

    return tasks


def run_dataset(root_dir: str,
                out_root: str = "output_spectrograms",
                sample_rate: float = 60e6,
                center_freq: float = 2.4375e9,
                duration_ms: float = 80.0,
                nfft: int = 1024,
                n_segments: int | None = None,
                seed: int | None = None,
                workers: int = 1) -> None:

    tasks = collect_tasks(root_dir, out_root, sample_rate, center_freq,
                          duration_ms, nfft, n_segments, seed)

    if not tasks:
        print(f"No .dat files found under '{root_dir}'. Check the path.")
        sys.exit(1)

    print(f"Found {len(tasks)} .dat file(s) under '{root_dir}'")
    print(f"Output root : {out_root}")
    print(f"Workers     : {workers}")
    print(f"Segments/file: {'random 2–3' if n_segments is None else n_segments}")
    print("─" * 60)

    ok_count  = 0
    err_count = 0
    img_count = 0

    if workers <= 1:
        # Single-process path (easier to debug)
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
        description="Batch-convert DroneDetect_V2 .dat files into STFT PNG images."
    )
    p.add_argument("--root",        required=True,
                   help="Root directory of the dataset (e.g. ~/toanlv/DroneDetect_V2)")
    p.add_argument("--out",         default="output_spectrograms",
                   help="Root output directory (mirrors source tree)")
    p.add_argument("--fs",          type=float, default=60e6,     help="Sample rate (Hz)")
    p.add_argument("--center_freq", type=float, default=2.4375e9, help="Centre frequency (Hz)")
    p.add_argument("--nfft",        type=int,   default=1024,     help="FFT size")
    p.add_argument("--duration_ms", type=float, default=80.0,     help="Segment length (ms)")
    p.add_argument("--n_segments",  type=int,   default=None,
                   help="Fixed # of segments per file (default: random 2–3)")
    p.add_argument("--seed",        type=int,   default=None,
                   help="Global random seed for reproducibility")
    p.add_argument("--workers",     type=int,   default=1,
                   help="Number of parallel worker processes (default: 1)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dataset(
        root_dir    = args.root,
        out_root    = args.out,
        sample_rate = args.fs,
        center_freq = args.center_freq,
        duration_ms = args.duration_ms,
        nfft        = args.nfft,
        n_segments  = args.n_segments,
        seed        = args.seed,
        workers     = args.workers,
    )