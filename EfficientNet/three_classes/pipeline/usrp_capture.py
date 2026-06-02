"""
usrp_capture.py
===============
USRP X300 IQ frame capture with frequency sweep support.

Sweep parameters
----------------
    Start  : 2.400 GHz
    Stop   : 2.600 GHz
    Step   : 25 MHz
    Steps  : 2.400 → 2.425 → 2.450 → 2.475 → 2.500 →
             2.525 → 2.550 → 2.575 → 2.600  (9 frequencies)
    Dwell  : 5 frames per step  (5 × ~115 ms ≈ 575 ms per channel)
    Cycle  : 9 × 575 ms ≈ ~5.2 s per full sweep

    Sweep always continues regardless of detection result.
    DroneInferencer downstream handles class decisions per frame.

RF parameters (matched to training data collection)
----------------------------------------------------
    Sample rate  : 25 MHz   (X300 decimation=8, 3 halfband filters)
    Bandwidth    : 30 MHz
    Frame size   : 2,000,000 samples  (80 ms @ 25 MHz)
    Stream fmt   : fc32 CPU / sc16 wire

Threading model
---------------
    FrequencySweeper (background daemon)
        Owns the retune loop — calls _retune() every FRAMES_PER_STEP frames.
        Pushes (freq_hz, iq_frame) tuples onto frame_queue so the
        processing loop knows which frequency produced each tensor.

    Main thread (run_pipeline.py)
        Pulls (freq_hz, iq) from frame_queue.
        Passes iq to stft_preprocessor, freq_hz to console/telemetry.

Standalone test
---------------
    python3 usrp_capture.py --addr 192.168.10.2 --gain 30
"""

import queue
import threading
import time

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Sweep configuration
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_START_HZ   = 2_400_000_000    # 2.400 GHz
SWEEP_STOP_HZ    = 2_475_000_000    # 2.600 GHz
SWEEP_STEP_HZ    =    25_000_000    # 25 MHz per hop
FRAMES_PER_STEP  = 5                # frames captured before retuning

# Build ordered frequency list: [2.400, 2.425, ..., 2.600] GHz
SWEEP_FREQS_HZ = list(range(SWEEP_START_HZ,
                             SWEEP_STOP_HZ + SWEEP_STEP_HZ,
                             SWEEP_STEP_HZ))
NUM_SWEEP_STEPS = len(SWEEP_FREQS_HZ)   # 9

# ─────────────────────────────────────────────────────────────────────────────
#  RF constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE_HZ  = 25_000_000        # 25 MHz — matches training --fs 25e6
BANDWIDTH_HZ    = 30_000_000
FRAME_DURATION  = 0.080             # 80 ms
NUM_SAMPLES     = int(SAMPLE_RATE_HZ * FRAME_DURATION)   # 2,000,000

RECV_CHUNK      = 100_000
RECV_TIMEOUT    = 3.0
MAX_OVERFLOWS   = 3
DEFAULT_GAIN_DB = 30.0

STREAM_CPU_FMT  = "fc32"
STREAM_WIRE_FMT = "sc16"

# PLL settle time after retune — X300 PLL locks in ~10 ms; 50 ms is safe
RETUNE_SETTLE_S = 0.050


# ─────────────────────────────────────────────────────────────────────────────
#  Device open / close
# ─────────────────────────────────────────────────────────────────────────────

def open_usrp(addr: str = "192.168.10.2", gain: float = DEFAULT_GAIN_DB):
    """
    Open and configure USRP X300 for continuous RX on channel 0.
    Initial centre frequency is set to SWEEP_START_HZ (2.400 GHz).

    Returns
    -------
    usrp     : uhd.usrp.MultiUSRP handle
    streamer : uhd.usrp.RxStreamer
    metadata : uhd.types.RXMetadata
    """
    try:
        import uhd
    except ImportError:
        raise ImportError(
            "UHD Python bindings not installed.\n"
            "  sudo apt-get install python3-uhd"
        )

    device_args = f"addr={addr},type=x300" if addr else ""
    print(f"[USRP] Opening  args='{device_args}' ...")
    usrp = uhd.usrp.MultiUSRP(device_args)

    # Sample rate
    usrp.set_rx_rate(SAMPLE_RATE_HZ, 0)
    actual_rate = usrp.get_rx_rate(0)
    print(f"  Sample rate : {actual_rate/1e6:.3f} MHz  "
          f"(decimation={int(200e6/actual_rate)})")

    if abs(actual_rate - SAMPLE_RATE_HZ) > 1e3:
        print(f"  ⚠  Rate mismatch: requested {SAMPLE_RATE_HZ/1e6} MHz, "
              f"got {actual_rate/1e6:.3f} MHz — STFT will NOT match training.")

    # Initial centre frequency — sweep starts here
    tune_req = uhd.libpyuhd.types.tune_request(SWEEP_START_HZ)
    usrp.set_rx_freq(tune_req, 0)
    actual_freq = usrp.get_rx_freq(0)
    print(f"  Centre freq : {actual_freq/1e9:.4f} GHz  (sweep start)")

    # Bandwidth
    usrp.set_rx_bandwidth(BANDWIDTH_HZ, 0)
    actual_bw = usrp.get_rx_bandwidth(0)
    print(f"  Bandwidth   : {actual_bw/1e6:.1f} MHz")

    # Gain
    usrp.set_rx_gain(gain, 0)
    actual_gain = usrp.get_rx_gain(0)
    print(f"  Gain        : {actual_gain:.1f} dB")

    # Antenna
    usrp.set_rx_antenna("RX2", 0)
    print(f"  Antenna     : {usrp.get_rx_antenna(0)}")

    # Stream
    st_args          = uhd.usrp.StreamArgs(STREAM_CPU_FMT, STREAM_WIRE_FMT)
    st_args.channels = [0]
    streamer         = usrp.get_rx_stream(st_args)
    metadata         = uhd.types.RXMetadata()

    # Start continuous stream
    stream_cmd            = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)

    time.sleep(0.2)   # PLL lock + FIFO fill

    # Print sweep plan
    print(f"\n  Sweep plan ({NUM_SWEEP_STEPS} steps × {FRAMES_PER_STEP} frames):")
    for i, f in enumerate(SWEEP_FREQS_HZ):
        print(f"    step {i+1:>2d} : {f/1e9:.4f} GHz")
    cycle_s = NUM_SWEEP_STEPS * FRAMES_PER_STEP * (FRAME_DURATION + RETUNE_SETTLE_S / FRAMES_PER_STEP)
    print(f"  Est. cycle  : ~{cycle_s:.1f} s per full sweep\n")

    return usrp, streamer, metadata


def close_usrp(usrp, streamer) -> None:
    """Stop stream and release device cleanly."""
    try:
        import uhd
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stop_cmd)
        time.sleep(0.05)
    except Exception:
        pass
    print("[USRP] Stream stopped and device released.")


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _retune(usrp, streamer, freq_hz: int) -> None:
    """
    Retune to freq_hz and restart the RX stream cleanly.

    Steps
    -----
    1. Stop continuous stream
    2. Set new centre frequency
    3. Wait RETUNE_SETTLE_S for PLL lock
    4. Restart continuous stream
    5. Discard one dummy recv() to flush stale samples from FIFO

    The dummy flush is critical — without it the first frame after a retune
    contains residual IQ from the previous frequency, producing a mixed
    spectrogram that confuses the classifier.
    """
    try:
        import uhd

        # Stop
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stop_cmd)
        time.sleep(0.01)

        # Retune
        tune_req = uhd.libpyuhd.types.tune_request(freq_hz)
        usrp.set_rx_freq(tune_req, 0)
        time.sleep(RETUNE_SETTLE_S)   # PLL settle

        # Restart
        start_cmd            = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        start_cmd.stream_now = True
        streamer.issue_stream_cmd(start_cmd)
        time.sleep(0.01)

        # Flush one chunk of stale samples from FIFO
        flush_buf = np.zeros((1, RECV_CHUNK), dtype=np.complex64)
        meta      = uhd.types.RXMetadata()
        streamer.recv(flush_buf, meta, timeout=0.5)

    except Exception as exc:
        print(f"[USRP] Retune to {freq_hz/1e9:.4f} GHz failed: {exc}")


def _restart_stream(streamer) -> None:
    """Recover from overflow without full retune."""
    try:
        import uhd
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stop_cmd)
        time.sleep(0.1)
        start_cmd            = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        start_cmd.stream_now = True
        streamer.issue_stream_cmd(start_cmd)
        time.sleep(0.05)
        print("[USRP] Stream restarted after overflow")
    except Exception as exc:
        print(f"[USRP] Stream restart failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Single frame capture
# ─────────────────────────────────────────────────────────────────────────────

def capture_frame(streamer, metadata) -> np.ndarray:
    """
    Capture exactly NUM_SAMPLES IQ samples (one 80 ms frame).

    Returns
    -------
    iq : complex64 ndarray, shape (NUM_SAMPLES,)
    """
    buf            = np.zeros((1, RECV_CHUNK), dtype=np.complex64)
    frame          = np.empty(NUM_SAMPLES, dtype=np.complex64)
    total          = 0
    overflow_count = 0

    while total < NUM_SAMPLES:
        chunk = min(RECV_CHUNK, NUM_SAMPLES - total)
        try:
            n_recvd = streamer.recv(buf[:, :chunk], metadata, RECV_TIMEOUT)
        except Exception as exc:
            print(f"[USRP] recv() error: {exc} — restarting stream")
            _restart_stream(streamer)
            total = overflow_count = 0
            continue

        ec = metadata.error_code
        if ec == 1:    # TIMEOUT
            print(f"[USRP] Timeout at sample {total} — retrying")
            continue
        if ec == 4:    # OVERFLOW
            overflow_count += 1
            print(f"[USRP] Overflow ({overflow_count}/{MAX_OVERFLOWS}) at sample {total}")
            if overflow_count >= MAX_OVERFLOWS:
                _restart_stream(streamer)
                total = overflow_count = 0
            continue
        if n_recvd == 0:
            continue

        frame[total : total + n_recvd] = buf[0, :n_recvd]
        total          += n_recvd
        overflow_count  = 0

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Frequency sweep capture thread
# ─────────────────────────────────────────────────────────────────────────────

def start_capture_thread(
    streamer,
    metadata,
    frame_queue : queue.Queue,
    stop_event  : threading.Event,
    usrp        = None,
) -> threading.Thread:
    """
    Sweep across SWEEP_FREQS_HZ continuously, pushing (freq_hz, iq) tuples.

    Sweep behaviour
    ---------------
    - Captures FRAMES_PER_STEP frames at each frequency before retuning.
    - After the last frequency (2.600 GHz) wraps back to 2.400 GHz.
    - Always sweeps — never pauses on detection.
    - If frame_queue is full, drops the oldest entry so inference always
      sees the most recent frame.

    Parameters
    ----------
    usrp : MultiUSRP handle — required for retuning. Pass None to disable
           sweep (fixed frequency, backward-compatible with old callers).

    Queue item format
    -----------------
    (freq_hz: int, iq: np.ndarray)
        freq_hz — centre frequency this frame was captured at
        iq      — complex64 (NUM_SAMPLES,)
    """
    sweep_enabled = usrp is not None

    def _run():
        step_idx    = 0    # current position in SWEEP_FREQS_HZ
        frame_count = 0    # frames captured at current step

        while not stop_event.is_set():
            current_freq = SWEEP_FREQS_HZ[step_idx]

            try:
                iq = capture_frame(streamer, metadata)

                # Pack freq alongside IQ so processing loop can label results
                item = (current_freq, iq)

                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put_nowait(item)

                frame_count += 1

                # ── Advance to next frequency after FRAMES_PER_STEP ──────────
                if sweep_enabled and frame_count >= FRAMES_PER_STEP:
                    step_idx    = (step_idx + 1) % NUM_SWEEP_STEPS
                    frame_count = 0
                    next_freq   = SWEEP_FREQS_HZ[step_idx]
                    print(
                        f"[Sweep] → {next_freq/1e9:.4f} GHz  "
                        f"(step {step_idx + 1}/{NUM_SWEEP_STEPS})"
                    )
                    _retune(usrp, streamer, next_freq)

            except Exception as exc:
                print(f"[USRP] Thread error: {exc}")
                time.sleep(0.01)

    t = threading.Thread(target=_run, name="USRP-Sweep-Capture", daemon=True)
    t.start()
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="USRP X300 sweep capture test")
    p.add_argument("--addr",   default="192.168.10.2")
    p.add_argument("--gain",   type=float, default=DEFAULT_GAIN_DB)
    p.add_argument("--frames", type=int,   default=3,
                   help="Test frames to capture per frequency step")
    args = p.parse_args()

    try:
        import uhd
    except ImportError:
        print("ERROR: uhd not installed.  sudo apt-get install python3-uhd")
        sys.exit(1)

    usrp, streamer, metadata = open_usrp(addr=args.addr, gain=args.gain)

    print(f"Sweep test — {args.frames} frames per step ...\n")
    for step_idx, freq_hz in enumerate(SWEEP_FREQS_HZ):
        if step_idx > 0:
            print(f"  Retuning to {freq_hz/1e9:.4f} GHz ...")
            _retune(usrp, streamer, freq_hz)

        for f in range(args.frames):
            t0  = time.perf_counter()
            iq  = capture_frame(streamer, metadata)
            ms  = (time.perf_counter() - t0) * 1000
            pwr = 10 * np.log10(np.mean(np.abs(iq) ** 2) + 1e-12)
            print(
                f"  [{freq_hz/1e9:.4f} GHz]  frame {f+1}/{args.frames}"
                f"  power={pwr:+6.2f} dBFS  time={ms:.1f} ms"
            )

    close_usrp(usrp, streamer)
    print("\nSweep test complete.")