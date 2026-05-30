"""
usrp_sweep_capture.py
=====================
High-speed sweeping IQ recorder for USRP X300.
Sweeps the centre frequency across a configurable range (default 2.440–2.460 GHz)
in 25 MHz steps, capturing one IQ frame per hop and saving every segment to
its own .bin file (fc32 interleaved float32).

Why sweep instead of one wide capture
--------------------------------------
The X300/UBX-160 captures 25 MHz of instantaneous bandwidth.
The 2.4 GHz ISM band is 85 MHz wide (2.400–2.485 GHz).
By hopping the LO in 25 MHz steps we cover the full band (or any sub-range)
without needing a wider sample rate that would cause overflows on 10 GbE.

Default sweep: 2.440 → 2.460 GHz (two 25 MHz hops)
    Hop 0: centre=2.440 GHz  →  covers 2.4275–2.4525 GHz
    Hop 1: centre=2.460 GHz  →  covers 2.4475–2.4725 GHz

Output files
------------
    output_dir/
        seg_000_2440MHz.bin        fc32 IQ samples for hop 0
        seg_000_2440MHz_meta.txt   metadata (freq, rate, samples, timestamp)
        seg_001_2460MHz.bin
        seg_001_2460MHz_meta.txt
        ...

Each .bin file contains `samples_per_hop` fc32 complex samples:
    [I0_f32, Q0_f32, I1_f32, Q1_f32, ...]  (interleaved, same as GNU Radio)

Usage
-----
    # Default: 2.440–2.460 GHz, 80 ms per hop, 1 sweep pass
    python3 usrp_sweep_capture.py -o captures/

    # Custom range and duration
    python3 usrp_sweep_capture.py \\
        --start_freq 2.4e9 \\
        --stop_freq  2.485e9 \\
        --step       25e6 \\
        --duration_ms 80 \\
        --passes 5 \\
        --gain 30 \\
        --addr 192.168.10.2 \\
        -o captures/

    # Read back a saved segment with numpy
    import numpy as np
    iq = np.fromfile("seg_000_2440MHz.bin", dtype=np.complex64)
    # iq is now shape (samples,), real=I, imag=Q

Requirements
------------
    pip install uhd numpy
    sudo apt-get install python3-uhd
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import uhd


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_stream_modes():
    """
    Return (start_mode, stop_mode) enums — handles both naming conventions
    across different pyuhd wheel versions.
    """
    try:
        return uhd.types.StreamMode.start_cont, uhd.types.StreamMode.stop_cont
    except AttributeError:
        return uhd.types.StreamMode.START_CONT, uhd.types.StreamMode.STOP_CONT


def _build_sweep_centres(start_hz: float, stop_hz: float,
                          step_hz: float) -> list:
    """
    Build a list of centre frequencies for the sweep.
    Includes start; stops before exceeding stop_hz.

    Example: start=2.44e9, stop=2.46e9, step=25e6
        → [2.44e9, 2.46e9]   (two hops covering 2.4275–2.4725 GHz)
    """
    centres = []
    f = start_hz
    while f <= stop_hz + 1:      # +1 to include exact stop_hz
        centres.append(f)
        f += step_hz
    return centres


def _settle_after_retune(usrp, settle_ms: float = 20.0):
    """
    Brief sleep after retuning LO to let PLL lock and transient settle.
    20 ms is conservative; 10 ms usually sufficient for X300/UBX-160.
    """
    time.sleep(settle_ms / 1000.0)


def _save_metadata(path: str, centre_hz: float, sample_rate: float,
                   num_samples: int, gain: float, fmt: str,
                   overflow_count: int):
    with open(path, "w") as f:
        f.write(f"center_freq_hz={centre_hz}\n")
        f.write(f"sample_rate_hz={sample_rate}\n")
        f.write(f"num_samples={num_samples}\n")
        f.write(f"format={fmt}\n")
        f.write(f"gain_db={gain}\n")
        f.write(f"overflows={overflow_count}\n")
        f.write(f"recorded_at={datetime.now().isoformat()}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Single-hop capture
# ─────────────────────────────────────────────────────────────────────────────

def capture_hop(
    usrp,
    streamer,
    centre_hz       : float,
    sample_rate     : float,
    samples_per_hop : int,
    out_path        : str,
    settle_ms       : float = 20.0,
) -> tuple:
    """
    Retune to centre_hz, capture samples_per_hop fc32 samples, save to out_path.

    Returns
    -------
    (samples_captured: int, overflow_count: int)

    File format
    -----------
    Raw fc32 = interleaved float32 [I0, Q0, I1, Q1, ...].
    Read back with:
        iq = np.fromfile(out_path, dtype=np.complex64)
    or equivalently:
        raw = np.fromfile(out_path, dtype=np.float32)
        iq  = raw[0::2] + 1j * raw[1::2]
    """
    start_mode, stop_mode = _get_stream_modes()

    # ── Stop any running stream ───────────────────────────────────────────────
    stop_cmd            = uhd.types.StreamCMD(stop_mode)
    stop_cmd.stream_now = True
    stop_cmd.time_spec  = uhd.types.TimeSpec(0.0)
    usrp.issue_stream_cmd(stop_cmd)
    time.sleep(0.01)

    # ── Retune LO ─────────────────────────────────────────────────────────────
    usrp.set_rx_freq(uhd.types.TuneRequest(centre_hz), 0)
    _settle_after_retune(usrp, settle_ms)

    # ── Start stream ──────────────────────────────────────────────────────────
    start_cmd            = uhd.types.StreamCMD(start_mode)
    start_cmd.stream_now = True
    start_cmd.num_samps  = 0
    start_cmd.time_spec  = uhd.types.TimeSpec(0.0)
    usrp.issue_stream_cmd(start_cmd)

    # ── Receive and write to file ─────────────────────────────────────────────
    buf_size    = streamer.get_max_num_samps()
    if buf_size <= 0:
        buf_size = 4096
    recv_buf    = np.zeros((1, buf_size), dtype=np.complex64)
    metadata    = uhd.types.RXMetadata()

    samples_collected = 0
    overflow_count    = 0

    with open(out_path, "wb") as fh:
        while samples_collected < samples_per_hop:
            to_recv  = min(buf_size, samples_per_hop - samples_collected)
            n_recvd  = streamer.recv(recv_buf[:, :to_recv], metadata, 1.0)

            ec = metadata.error_code
            if ec != uhd.types.RXMetadataErrorCode.none:
                if ec == uhd.types.RXMetadataErrorCode.overflow:
                    overflow_count += 1
                    print("O", end="", flush=True)
                else:
                    print(f"\n[WARN] Stream error: {metadata.strerror()}")
                    if samples_collected > 0:
                        break

            if n_recvd > 0:
                recv_buf[0, :n_recvd].tofile(fh)
                samples_collected += n_recvd

    # ── Stop stream ───────────────────────────────────────────────────────────
    usrp.issue_stream_cmd(stop_cmd)

    return samples_collected, overflow_count


# ─────────────────────────────────────────────────────────────────────────────
#  Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    addr            : str,
    start_freq      : float,
    stop_freq       : float,
    step            : float,
    sample_rate     : float,
    duration_ms     : float,
    gain            : float,
    passes          : int,
    out_dir         : str,
    settle_ms       : float,
):
    centres = _build_sweep_centres(start_freq, stop_freq, step)
    samples_per_hop = int(sample_rate * duration_ms / 1000.0)

    print("=" * 60)
    print("  USRP X300 Sweeping IQ Recorder")
    print("=" * 60)
    print(f"  Address       : {addr}")
    print(f"  Sweep range   : {start_freq/1e9:.4f} – {stop_freq/1e9:.4f} GHz")
    print(f"  Step          : {step/1e6:.1f} MHz")
    print(f"  Hops          : {len(centres)}")
    print(f"  Centres (GHz) : {[f'{c/1e9:.4f}' for c in centres]}")
    print(f"  Sample rate   : {sample_rate/1e6:.1f} MHz")
    print(f"  Duration/hop  : {duration_ms:.0f} ms  ({samples_per_hop:,} samples)")
    print(f"  Gain          : {gain} dB")
    print(f"  Passes        : {passes}")
    print(f"  Output dir    : {out_dir}")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)

    # ── Connect ───────────────────────────────────────────────────────────────
    print(f"\n[USRP] Connecting to addr={addr} ...")
    try:
        usrp = uhd.usrp.MultiUSRP(f"addr={addr}")
    except Exception as exc:
        print(f"[USRP] Connection failed: {exc}")
        sys.exit(1)

    # ── Configure ─────────────────────────────────────────────────────────────
    usrp.set_rx_rate(sample_rate, 0)
    actual_rate = usrp.get_rx_rate(0)
    if abs(actual_rate - sample_rate) > 1e3:
        print(f"[WARN] Requested {sample_rate/1e6:.3f} MHz, "
              f"got {actual_rate/1e6:.3f} MHz.")
        print(f"       X300 valid rates: 200/N MHz  (100, 50, 40, 25, 20, 10 ...)")
        print(f"       Proceeding with actual rate.")
    sample_rate = actual_rate

    usrp.set_rx_gain(gain, 0)
    usrp.set_rx_antenna("RX2", 0)
    usrp.set_rx_bandwidth(min(sample_rate, 40e6), 0)
    time.sleep(0.2)

    print(f"  Actual rate   : {usrp.get_rx_rate(0)/1e6:.3f} MHz")
    print(f"  Actual gain   : {usrp.get_rx_gain(0):.1f} dB")
    print(f"  Antenna       : {usrp.get_rx_antenna(0)}")

    # ── Stream ────────────────────────────────────────────────────────────────
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_rx_stream(st_args)

    # ── Sweep loop ────────────────────────────────────────────────────────────
    total_hops      = 0
    total_overflows = 0
    t_sweep_start   = time.perf_counter()

    try:
        for pass_idx in range(passes):
            print(f"\n[Pass {pass_idx+1}/{passes}]")

            for hop_idx, centre in enumerate(centres):
                seg_name  = (f"seg_{total_hops:04d}_"
                             f"{int(centre/1e6)}MHz")
                bin_path  = os.path.join(out_dir, seg_name + ".bin")
                meta_path = os.path.join(out_dir, seg_name + "_meta.txt")

                t0 = time.perf_counter()
                n, ov = capture_hop(
                    usrp        = usrp,
                    streamer    = streamer,
                    centre_hz   = centre,
                    sample_rate = sample_rate,
                    samples_per_hop = samples_per_hop,
                    out_path    = bin_path,
                    settle_ms   = settle_ms,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

                _save_metadata(
                    path        = meta_path,
                    centre_hz   = centre,
                    sample_rate = sample_rate,
                    num_samples = n,
                    gain        = gain,
                    fmt         = "fc32",
                    overflow_count = ov,
                )

                size_kb = os.path.getsize(bin_path) / 1024
                print(f"  hop {hop_idx:02d}  {centre/1e9:.4f} GHz  "
                      f"{n:>9,} samp  {elapsed_ms:6.0f} ms  "
                      f"{size_kb:7.0f} KB  "
                      f"{'OVF='+str(ov) if ov else 'ok':>6}  "
                      f"→ {seg_name}.bin")

                total_overflows += ov
                total_hops      += 1

    except KeyboardInterrupt:
        print("\n[Interrupted] Ctrl+C received — stopping sweep.")

    elapsed_total = time.perf_counter() - t_sweep_start
    print(f"\n{'='*60}")
    print(f"  Sweep complete")
    print(f"  Total hops      : {total_hops}")
    print(f"  Total overflows : {total_overflows}")
    print(f"  Total time      : {elapsed_total:.1f} s")
    print(f"  Output dir      : {os.path.abspath(out_dir)}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="USRP X300 sweeping IQ recorder (25 MHz hops)"
    )
    p.add_argument("--addr",        default="192.168.10.2",
                   help="USRP X300 IP address (default: 192.168.10.2)")
    p.add_argument("--start_freq",  type=float, default=2.440e9,
                   help="Sweep start centre frequency in Hz (default: 2.44 GHz)")
    p.add_argument("--stop_freq",   type=float, default=2.460e9,
                   help="Sweep stop  centre frequency in Hz (default: 2.46 GHz)")
    p.add_argument("--step",        type=float, default=25e6,
                   help="Frequency step in Hz (default: 25 MHz)")
    p.add_argument("-r", "--rate",  type=float, default=25e6,
                   help="Sample rate in Hz — must be 200 MHz/N (default: 25 MHz)")
    p.add_argument("-g", "--gain",  type=float, default=30.0,
                   help="RX gain in dB (default: 30)")
    p.add_argument("-d", "--duration_ms", type=float, default=80.0,
                   help="Capture duration per hop in ms (default: 80)")
    p.add_argument("--passes",      type=int,   default=1,
                   help="Number of full sweep passes (default: 1)")
    p.add_argument("--settle_ms",   type=float, default=20.0,
                   help="LO settle time after retune in ms (default: 20)")
    p.add_argument("-o", "--out_dir", required=True,
                   help="Output directory for .bin and _meta.txt files")
    args = p.parse_args()

    run_sweep(
        addr        = args.addr,
        start_freq  = args.start_freq,
        stop_freq   = args.stop_freq,
        step        = args.step,
        sample_rate = args.rate,
        duration_ms = args.duration_ms,
        gain        = args.gain,
        passes      = args.passes,
        out_dir     = args.out_dir,
        settle_ms   = args.settle_ms,
    )


if __name__ == "__main__":
    main()

    # # 10 full sweep passes = 40 segments
    # python3
    # usrp_sweep_capture.py \
    # - -addr
    # 192.168
    # .10
    # .2 \
    # - -start_freq
    # 2.4e9 \
    # - -stop_freq
    # 2.485e9 \
    # - -step
    # 25e6 \
    # - -passes
    # 10 \
    # - o
    # captures /

    # python3
    # usrp_sweep_capture.py \
    # - -addr
    # 192.168
    # .10
    # .2 \
    # - -start_freq
    # 2.4e9 - -stop_freq
    # 2.485e9 - -step
    # 25e6 \
    # - -continuous \
    # - o
    # captures /