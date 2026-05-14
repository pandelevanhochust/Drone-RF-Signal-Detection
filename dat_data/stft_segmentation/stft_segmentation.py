"""
Sliding-window STFT spectrogram export from DroneDetect_V2 .dat files.

Examples:
    # 1/8 overlap (default, 70 ms step for 80 ms window)
    python stft_segmentation.py --root ~/DroneDetect_V2 --out output_spectrograms/

    # No overlap / consecutive
    python stft_segmentation.py --root ~/DroneDetect_V2 --overlap 0

    # Manual step
    python stft_segmentation.py --root ~/DroneDetect_V2 --step_ms 40
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


# ── Core DSP ──────────────────────────────────────────────────────────────────

def file_duration_ms(file_path: str, sample_rate: float) -> float:
    n_floats  = os.path.getsize(file_path) // 4   # float32 = 4 bytes
    n_samples = n_floats // 2                      # I+Q pairs
    return (n_samples / sample_rate) * 1000.0


def compute_spectrogram(file_path: str,
                        sample_rate: float,
                        start_ms: float,
                        duration_ms: float = 80.0,
                        nfft: int = 1024) -> np.ndarray:
    """Read one window from a float32 IQ .dat file. Returns spec_db."""
    skip_samples = int(sample_rate * (start_ms   / 1000.0))
    num_samples  = int(sample_rate * (duration_ms / 1000.0))

    data_map  = np.memmap(file_path, dtype=np.float32, mode='r')
    raw_chunk = data_map[2 * skip_samples: 2 * skip_samples + 2 * num_samples]

    iq = raw_chunk[0::2] + 1j * raw_chunk[1::2]

    _, _, Zxx = stft(iq, sample_rate,
                     return_onesided=False,
                     window=windows.hamming(nfft),
                     nperseg=nfft)

    Zxx     = np.fft.fftshift(Zxx, axes=0)
    spec_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    return spec_db


def save_spectrogram(spec_db: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ── Per-file worker ────────────────────────────────────────────────────────────

def process_one_file(task: dict) -> tuple[str, list[str], str | None]:
    file_path   = task["file_path"]
    out_dir     = task["out_dir"]
    sample_rate = task["sample_rate"]
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

        starts_ms = np.arange(0.0, total_ms - duration_ms + step_ms, step_ms)
        starts_ms = starts_ms[starts_ms + duration_ms <= total_ms]

        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(file_path))[0]

        for idx, start in enumerate(starts_ms):
            spec_db  = compute_spectrogram(file_path, sample_rate,
                                           float(start), duration_ms, nfft)
            tag      = f"seg{idx:04d}_start{start:.1f}ms"
            out_name = f"{stem}__{tag}.png"
            out_path = os.path.join(out_dir, out_name)
            save_spectrogram(spec_db, out_path)
            saved.append(out_path)

    except Exception:
        return file_path, saved, traceback.format_exc()

    return file_path, saved, None


# ── Dataset walker ─────────────────────────────────────────────────────────────

def collect_tasks(root_dir: str, out_root: str,
                  sample_rate: float,
                  duration_ms: float, step_ms: float,
                  nfft: int) -> list[dict]:

    root  = Path(root_dir).expanduser().resolve()
    tasks = []

    SKIP_FOLDERS = {"BLUE", "WIFI", "CLEAN"}

    for dat_file in sorted(root.rglob("*.dat")):
        rel_parts = dat_file.relative_to(root).parts
        if rel_parts[0] in SKIP_FOLDERS:
            continue
        out_dir = os.path.join(out_root, *rel_parts[:-1])
        tasks.append({
            "file_path":   str(dat_file),
            "out_dir":     out_dir,
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
            "step_ms":     step_ms,
            "nfft":        nfft,
        })

    return tasks


def run_dataset(root_dir: str,
                out_root: str      = "output_spectrograms",
                sample_rate: float = 60e6,
                duration_ms: float = 80.0,
                step_ms: float     = 70.0,
                nfft: int          = 1024,
                workers: int       = 1) -> None:

    tasks = collect_tasks(root_dir, out_root, sample_rate,
                          duration_ms, step_ms, nfft)

    if not tasks:
        print(f"No .dat files found under '{root_dir}'. Check the path.")
        sys.exit(1)

    overlap_frac = 1.0 - (step_ms / duration_ms)
    est_segs     = int((2000.0 - duration_ms) / step_ms) + 1

    print(f"Found {len(tasks)} .dat file(s) under '{root_dir}'")
    print(f"Output root   : {out_root}")
    print(f"Workers       : {workers}")
    print(f"Window        : {duration_ms} ms")
    print(f"Step          : {step_ms} ms  ({overlap_frac:.1%} overlap)")
    print(f"Est. segs/file: ~{est_segs}  (for a 2 s file)")
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
    p.add_argument("--root",        required=True,                  help="Dataset root directory")
    p.add_argument("--out",         default="output_spectrograms",  help="Output root directory")
    p.add_argument("--fs",          type=float, default=60e6,       help="Sample rate (Hz)")
    p.add_argument("--nfft",        type=int,   default=1024,       help="FFT size")
    p.add_argument("--duration_ms", type=float, default=80.0,       help="Window length (ms)")
    p.add_argument("--workers",     type=int,   default=1,          help="Parallel worker processes")

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

    run_dataset(
        root_dir    = args.root,
        out_root    = args.out,
        sample_rate = args.fs,
        duration_ms = args.duration_ms,
        step_ms     = step_ms,
        nfft        = args.nfft,
        workers     = args.workers,
    )