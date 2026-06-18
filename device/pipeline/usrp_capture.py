"""
usrp_capture.py
===============
USRP X300 IQ frame capture — drop-in replacement for bladerf_capture.py.

RF parameters matched to training data collection script
---------------------------------------------------------
The training script (segment_file) defaults:
    --fs          25e6   (25 MHz sample rate)
    --center_freq 2.4e9  (2.4 GHz)
2_3    --duration_ms 80.0   (80 ms frames)
    --nfft        1024

Therefore this capture module uses the SAME parameters so that the
live STFT spectrogram is pixel-identical to the training spectrograms.

X300 master clock = 200 MHz. Valid sample rates = 200 MHz / N:
    200 / 8  =  25.0 MHz  ← matches training exactly, decimation=8 (3 halfbands)
    200 / 4  =  50.0 MHz
    200 / 5  =  40.0 MHz

25 MHz with decimation=8 enables 3 halfband filters → flattest passband,
best alias rejection. Wire rate = 25M × 8 bytes = 200 MB/s — very safe
on 10 GbE with no overflow risk.

UHD fc32 stream format
-----------------------
recv() fills a (n_channels, n_samples) complex64 buffer directly.
Each complex64 sample = float32 real (I) + float32 imag (Q).
This is identical to what np.memmap reads from a GNU Radio fc32 bin file:
    bin file: [I0_f32, Q0_f32, I1_f32, Q1_f32, ...]
    fc32 recv: complex64 array where real=I, imag=Q
No manual interleaving or /2048 conversion needed.

Connection
----------
    X300 via 10 GbE: default IP 192.168.40.2
    Verify: uhd_find_devices --args="addr=192.168.40.2"

Setup (run once)
----------------
    sudo apt-get install libuhd-dev uhd-host python3-uhd
    sudo uhd_images_downloader -t x3xx
    pip3 install uhd numpy

Standalone test
---------------
    python3 usrp_capture.py --addr 192.168.40.2 --gain 30
"""

import queue
import threading
import time

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  RF constants — matched to training data collection
# ─────────────────────────────────────────────────────────────────────────────

# Must match training script default --fs
# X300: 200 MHz / 8 = 25 MHz, decimation=8 → 3 halfband filters, flat passband
CENTER_FREQ_HZ  = 2_400_000_000     # 2.4 GHz  (matches training --center_freq 2.4e9)
SAMPLE_RATE_HZ  = 25_000_000        # 25 MHz   (matches training --fs 25e6)
BANDWIDTH_HZ    = 30_000_000        # 25 MHz   (set equal to sample rate)
FRAME_DURATION  = 0.080             # 80 ms    (matches training --duration_ms 80)
NUM_SAMPLES     = int(SAMPLE_RATE_HZ * FRAME_DURATION)  # 2,000,000 samples

# recv() chunk size — at 25 MHz wire rate = 200 MB/s, very safe on 10 GbE
# 100K chunks = 20 recv() calls per frame, low Python overhead
RECV_CHUNK      = 100_000
RECV_TIMEOUT    = 3.0

# Max consecutive overflows before stream restart
MAX_OVERFLOWS   = 3

DEFAULT_GAIN_DB = 30.0

# UHD stream format
# fc32 = complex float32 on CPU side (identical layout to GNU Radio fc32 files)
# sc16 = int16 on wire, UHD driver converts to fc32 internally
STREAM_CPU_FMT  = "fc32"
STREAM_WIRE_FMT = "sc16"


# ─────────────────────────────────────────────────────────────────────────────
#  Device open / close
# ─────────────────────────────────────────────────────────────────────────────

def open_usrp(addr: str = "192.168.5.111", gain: float = DEFAULT_GAIN_DB):
    """
    Open and configure USRP X300 for continuous RX on channel 0.

    Returns
    -------
    usrp     : uhd.usrp.MultiUSRP handle
    streamer : uhd.usrp.RxStreamer  — reuse across all frames
    metadata : uhd.types.RXMetadata — reused recv buffer
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

    # Verify rate matches training — warn if it differs
    if abs(actual_rate - SAMPLE_RATE_HZ) > 1e3:
        print(f"  ⚠  Rate mismatch: requested {SAMPLE_RATE_HZ/1e6} MHz, "
              f"got {actual_rate/1e6:.3f} MHz")
        print(f"     STFT spectrograms will NOT match training data.")

    # Centre frequency
    tune_req = uhd.libpyuhd.types.tune_request(CENTER_FREQ_HZ)
    usrp.set_rx_freq(tune_req, 0)
    actual_freq = usrp.get_rx_freq(0)
    print(f"  Centre freq : {actual_freq/1e9:.4f} GHz")

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

    # Create stream — fc32 CPU format, sc16 wire format
    # UHD converts sc16 → complex float32 internally; output is identical
    # in layout to GNU Radio fc32 files: real=I, imag=Q per sample
    st_args          = uhd.usrp.StreamArgs(STREAM_CPU_FMT, STREAM_WIRE_FMT)
    st_args.channels = [0]
    streamer         = usrp.get_rx_stream(st_args)
    metadata         = uhd.types.RXMetadata()

    # Start continuous stream
    stream_cmd            = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)

    # Wait for PLL lock and FIFO fill
    time.sleep(0.2)
    print(f"  ✓ Streaming  {NUM_SAMPLES:,} samples / frame  "
          f"({FRAME_DURATION*1000:.0f} ms @ {SAMPLE_RATE_HZ/1e6:.0f} MHz)\n")

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
#  Stream restart helper
# ─────────────────────────────────────────────────────────────────────────────

def _restart_stream(streamer) -> None:
    """
    Stop and restart the continuous RX stream.
    Called after overflow or socket-closed to recover without re-opening device.
    """
    try:
        import uhd
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stop_cmd)
        time.sleep(0.1)
        start_cmd            = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        start_cmd.stream_now = True
        streamer.issue_stream_cmd(start_cmd)
        time.sleep(0.05)
        print("[USRP] Stream restarted")
    except Exception as exc:
        print(f"[USRP] Stream restart failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Single frame capture
# ─────────────────────────────────────────────────────────────────────────────

def capture_frame(streamer, metadata) -> np.ndarray:
    """
    Capture exactly NUM_SAMPLES IQ samples (one 80 ms frame).

    UHD fc32 recv() delivers complex64 directly:
        buf shape: (1, RECV_CHUNK)   — (n_channels, n_samples)
        buf[0]   : complex64 array   — real=I, imag=Q, normalised float32

    This is the same IQ layout as a GNU Radio fc32 .bin file read with:
        raw = np.memmap(path, dtype=np.float32)
        iq  = raw[0::2] + 1j * raw[1::2]

    Returns
    -------
    iq : complex64 ndarray, shape (NUM_SAMPLES,)  values approx [-1, 1]
    """
    buf            = np.zeros((1, RECV_CHUNK), dtype=np.complex64)
    frame          = np.empty(NUM_SAMPLES, dtype=np.complex64)
    total          = 0
    overflow_count = 0

    while total < NUM_SAMPLES:
        chunk = min(RECV_CHUNK, NUM_SAMPLES - total)

        # ── recv() ───────────────────────────────────────────────────────────
        try:
            n_recvd = streamer.recv(buf[:, :chunk], metadata, RECV_TIMEOUT)
        except Exception as exc:
            print(f"[USRP] recv() error: {exc} — restarting stream")
            _restart_stream(streamer)
            total          = 0
            overflow_count = 0
            continue

        # ── Error codes ───────────────────────────────────────────────────────
        ec = metadata.error_code

        if ec == 1:    # TIMEOUT
            print(f"[USRP] Timeout at sample {total} — retrying")
            continue

        if ec == 4:    # OVERFLOW — host couldn't drain fast enough
            overflow_count += 1
            print(f"[USRP] Overflow ({overflow_count}/{MAX_OVERFLOWS}) "
                  f"at sample {total}")
            if overflow_count >= MAX_OVERFLOWS:
                _restart_stream(streamer)
                total          = 0
                overflow_count = 0
            continue

        if n_recvd == 0:
            continue

        # ── Accumulate samples ────────────────────────────────────────────────
        # buf[0, :n_recvd] is complex64: real=I float32, imag=Q float32
        # Identical layout to: raw[0::2] + 1j * raw[1::2] from a .bin file
        frame[total : total + n_recvd] = buf[0, :n_recvd]
        total          += n_recvd
        overflow_count  = 0

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Background capture thread
# ─────────────────────────────────────────────────────────────────────────────

def start_capture_thread(
    streamer,
    metadata,
    frame_queue : queue.Queue,
    stop_event  : threading.Event,
) -> threading.Thread:
    """
    Capture continuously in background, push frames onto frame_queue.
    Drops oldest frame if queue is full so inference always sees latest signal.
    """
    def _run():
        frame_idx = 0
        while not stop_event.is_set():
            try:
                iq = capture_frame(streamer, metadata)
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put_nowait(iq)
                frame_idx += 1
            except Exception as exc:
                print(f"[USRP] Thread error on frame {frame_idx}: {exc}")
                time.sleep(0.01)

    t = threading.Thread(target=_run, name="USRP-Capture", daemon=True)
    t.start()
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="USRP X300 capture test")
    p.add_argument("--addr", default="192.168.5.111")
    p.add_argument("--gain", type=float, default=DEFAULT_GAIN_DB)
    args = p.parse_args()

    try:
        import uhd
    except ImportError:
        print("ERROR: uhd not installed.  sudo apt-get install python3-uhd")
        sys.exit(1)

    usrp, streamer, metadata = open_usrp(addr=args.addr, gain=args.gain)

    print(f"Capturing 3 test frames ({NUM_SAMPLES:,} samples each) ...\n")
    for i in range(3):
        t0  = time.perf_counter()
        iq  = capture_frame(streamer, metadata)
        ms  = (time.perf_counter() - t0) * 1000
        pwr = 10 * np.log10(np.mean(np.abs(iq)**2) + 1e-12)
        print(f"  Frame {i}: shape={iq.shape}  dtype={iq.dtype}  "
              f"power={pwr:.2f} dBFS  time={ms:.1f} ms")

    close_usrp(usrp, streamer)