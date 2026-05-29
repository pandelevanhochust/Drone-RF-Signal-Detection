"""
usrp_capture.py
===============
USRP X300 IQ frame capture — drop-in replacement for bladerf_capture.py.
Keeps the same public interface: open_usrp(), close_usrp(),
capture_frame(), start_capture_thread().

Why USRP X300 over BladeRF for FHSS drone detection
-----------------------------------------------------
The drone uses FHSS (Frequency Hopping Spread Spectrum) — its signal
jumps rapidly across sub-channels within the 2.4 GHz ISM band. Catching
it requires:

  1. Wide instantaneous bandwidth (60 MHz) to see all hops simultaneously
  2. High ADC dynamic range (14-bit on X300 vs 12-bit on BladeRF) so weak
     hops above the noise floor are resolved
  3. Flat passband — X300's daughterboard (UBX-160) has much flatter
     in-band response than BladeRF's fixed filters, eliminating the skirt
     artefact that triggered false U-Net activations

RF parameters (must match training data collection)
----------------------------------------------------
    Center frequency : 2.375 GHz
    Sample rate      : 60 MHz   (requires 10 GbE or PCIe backhaul on X300)
    Bandwidth        : 60 MHz
    Frame duration   : 80 ms  →  4,800,000 samples per frame
    Gain             : manual 30 dB recommended as starting point
                       (X300 total gain range: 0–31.5 dB on UBX-160)

Connection
----------
    X300 connects via 10 GbE (default IP 192.168.40.2) or PCIe.
    Pass --addr 192.168.40.2 or set USRP_ADDR in .env.

Setup (run once on host)
------------------------
    # Install UHD
    sudo apt-get install libuhd-dev uhd-host python3-uhd

    # Or build from source for latest X300 FPGA support
    git clone https://github.com/EttusResearch/uhd.git
    cd uhd/host && mkdir build && cd build
    cmake .. -DENABLE_PYTHON_API=ON && make -j4 && sudo make install

    # Download FPGA image for X300
    sudo uhd_images_downloader -t x3xx

    # Verify device is found
    uhd_find_devices --args="addr=192.168.40.2"
    uhd_usrp_probe --args="addr=192.168.40.2"

    pip3 install uhd numpy

Standalone test
---------------
    python3 usrp_capture.py --addr 192.168.40.2
    # Opens device, captures 3 frames, prints shape and power
"""

import queue
import threading
import time

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  RF constants — edit here if your collection parameters differ
# ─────────────────────────────────────────────────────────────────────────────

CENTER_FREQ_HZ  = 2_375_000_000     # 2.375 GHz

# X300 master clock = 200 MHz. Sample rate must be 200 MHz / N (integer N).
# Valid rates near 2.4 GHz ISM band:
#   200/2 = 100 MHz, 200/4 = 50 MHz ← use this, 200/5 = 40 MHz
# 50 MHz gives decimation=4 → 2 halfband filters → flat passband
# 60 MHz gave decimation=3 (odd) → CIC rolloff artefacts + socket overflow
SAMPLE_RATE_HZ  = 10_000_000        # 50 MHz  (200 MHz / 4, even decimation)

# UBX-160 max analog bandwidth = 40 MHz per channel.
# Requesting more than 40 MHz clips silently. Set to 40 MHz explicitly.
BANDWIDTH_HZ    = 40_000_000        # 40 MHz  (UBX-160 hardware maximum)

FRAME_DURATION  = 0.080             # 80 ms per frame
NUM_SAMPLES     = int(SAMPLE_RATE_HZ * FRAME_DURATION)   # 4,000,000

# Receive buffer chunk size — UHD recv() is called in chunks to avoid
# timeout on large single-call requests.
# At 50 MHz / fc32: wire rate = 50M × 8 bytes = 400 MB/s — well within 10 GbE.
# Larger chunks reduce Python call overhead; 500K is safe at 50 MHz.
RECV_CHUNK      = 500_000           # samples per recv() call
RECV_TIMEOUT    = 3.0               # seconds per recv() call

# Max consecutive overflows before stream is restarted
MAX_OVERFLOWS   = 3

# X300 / UBX-160 gain range: 0 – 31.5 dB
# 30 dB is a safe starting point for 2.4 GHz ISM band
DEFAULT_GAIN_DB = 30.0

# Stream format: fc32 = complex float32 (native Python/numpy format)
# Wire format: sc16 = int16 on the wire, converted to fc32 in UHD driver
STREAM_CPU_FMT  = "fc32"
STREAM_WIRE_FMT = "sc16"


# ─────────────────────────────────────────────────────────────────────────────
#  Device open / close
# ─────────────────────────────────────────────────────────────────────────────

def open_usrp(addr: str = "192.168.40.2", gain: float = DEFAULT_GAIN_DB):
    """
    Open and configure the USRP X300 for continuous RX on channel 0.

    Parameters
    ----------
    addr : str
        IP address of the X300 (default 192.168.40.2).
        For PCIe connection use addr="" and let UHD auto-detect.
    gain : float
        RX gain in dB. X300/UBX-160 range: 0 – 31.5 dB.
        30 dB is recommended for 2.4 GHz drone detection.

    Returns
    -------
    usrp     : uhd.usrp.MultiUSRP handle
    streamer : uhd.usrp.RxStreamer — reuse across frames for efficiency
    metadata : uhd.types.RXMetadata — reused buffer

    Notes
    -----
    The streamer is created once here and reused in capture_frame().
    Creating a new streamer per frame is expensive (~100 ms overhead).
    The stream is left running continuously between frames — call
    close_usrp() to stop it cleanly on shutdown.
    """
    try:
        import uhd
    except ImportError:
        raise ImportError(
            "UHD Python bindings not installed.\n"
            "  sudo apt-get install python3-uhd\n"
            "  or: pip3 install uhd"
        )

    device_args = f"addr={addr},type=x300" if addr else ""
    print(f"[USRP] Opening device  args='{device_args}' ...")
    usrp = uhd.usrp.MultiUSRP(device_args)

    # ── Sample rate ───────────────────────────────────────────────────────────
    usrp.set_rx_rate(SAMPLE_RATE_HZ, 0)
    actual_rate = usrp.get_rx_rate(0)
    print(f"  Sample rate : {actual_rate/1e6:.3f} MHz")

    # ── Centre frequency ──────────────────────────────────────────────────────
    tune_req = uhd.libpyuhd.types.tune_request(CENTER_FREQ_HZ)
    usrp.set_rx_freq(tune_req, 0)
    actual_freq = usrp.get_rx_freq(0)
    print(f"  Centre freq : {actual_freq/1e9:.4f} GHz")

    # ── Bandwidth ─────────────────────────────────────────────────────────────
    usrp.set_rx_bandwidth(BANDWIDTH_HZ, 0)
    actual_bw = usrp.get_rx_bandwidth(0)
    print(f"  Bandwidth   : {actual_bw/1e6:.1f} MHz")

    # ── Gain ──────────────────────────────────────────────────────────────────
    usrp.set_rx_gain(gain, 0)
    actual_gain = usrp.get_rx_gain(0)
    print(f"  Gain        : {actual_gain:.1f} dB")

    # ── Antenna — RFnB is the standard RX port on X300/UBX-160 ───────────────
    usrp.set_rx_antenna("RX2", 0)
    print(f"  Antenna     : {usrp.get_rx_antenna(0)}")

    # ── Stream ────────────────────────────────────────────────────────────────
    # fc32 on CPU side: UHD converts sc16 wire format to complex float32
    st_args          = uhd.usrp.StreamArgs(STREAM_CPU_FMT, STREAM_WIRE_FMT)
    st_args.channels = [0]
    streamer         = usrp.get_rx_stream(st_args)
    metadata         = uhd.types.RXMetadata()

    # Start continuous stream
    stream_cmd              = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now   = True
    streamer.issue_stream_cmd(stream_cmd)

    # Allow PLL to lock and buffers to fill
    time.sleep(0.1)
    print(f"  ✓ Stream started  (continuous, fc32)\n")

    return usrp, streamer, metadata


def close_usrp(usrp, streamer) -> None:
    """
    Stop the RX stream and release the USRP handle cleanly.
    Always call this before exiting, even on error.
    """
    try:
        import uhd
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stop_cmd)
        time.sleep(0.05)
    except Exception:
        pass
    print("[USRP] Stream stopped and device released.")


# ─────────────────────────────────────────────────────────────────────────────
#  Single frame capture
# ─────────────────────────────────────────────────────────────────────────────

def _restart_stream(streamer) -> None:
    """
    Stop and restart the continuous RX stream.
    Called after overflow or socket-closed errors to recover cleanly.
    """
    try:
        import uhd
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stop_cmd)
        import time as _t; _t.sleep(0.05)
        start_cmd              = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        start_cmd.stream_now   = True
        streamer.issue_stream_cmd(start_cmd)
        _t.sleep(0.05)
        print("[USRP] Stream restarted after overflow")
    except Exception as exc:
        print(f"[USRP] Stream restart failed: {exc}")


def capture_frame(streamer, metadata) -> np.ndarray:
    """
    Capture exactly NUM_SAMPLES IQ samples (one 80 ms frame).

    UHD recv() is called in RECV_CHUNK chunks to avoid timeout issues.
    At 50 MHz / fc32 the wire rate is 400 MB/s — well within 10 GbE.

    Returns
    -------
    iq : complex64 ndarray, shape (NUM_SAMPLES,)
        Values in approximately [-1.0, 1.0].
        UHD fc32 delivers normalised float32 directly — no /2048 needed.

    Error handling
    --------------
    TIMEOUT     : retries the recv() call — should not occur at 50 MHz
    OVERFLOW    : logged; stream is restarted after MAX_OVERFLOWS consecutive
                  overflows to recover from socket-closed state
    n_recvd==0  : skipped silently
    """
    buf            = np.zeros((1, RECV_CHUNK), dtype=np.complex64)
    total          = 0
    frame          = np.empty(NUM_SAMPLES, dtype=np.complex64)
    overflow_count = 0

    while total < NUM_SAMPLES:
        remaining = NUM_SAMPLES - total
        chunk     = min(RECV_CHUNK, remaining)

        try:
            n_recvd = streamer.recv(buf[:, :chunk], metadata, RECV_TIMEOUT)
        except Exception as exc:
            # socket closed / IOError — restart stream and retry from top
            print(f"[USRP] recv() exception: {exc} — restarting stream")
            _restart_stream(streamer)
            total = 0       # discard partial frame, start fresh
            overflow_count = 0
            continue

        if metadata.error_code == 1:    # TIMEOUT
            print(f"[USRP] Timeout at sample {total} — retrying")
            continue

        if metadata.error_code == 4:    # OVERFLOW
            overflow_count += 1
            print(f"[USRP] Overflow ({overflow_count}/{MAX_OVERFLOWS}) "
                  f"at sample {total}")
            if overflow_count >= MAX_OVERFLOWS:
                print("[USRP] Too many overflows — restarting stream")
                _restart_stream(streamer)
                total = 0
                overflow_count = 0
            continue

        overflow_count = 0  # reset on clean recv
        if n_recvd == 0:
            continue

        frame[total : total + n_recvd] = buf[0, :n_recvd]
        total += n_recvd

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
    Start a daemon thread that continuously captures IQ frames and pushes
    them onto frame_queue.

    If frame_queue is full (inference hasn't consumed the previous frame),
    the oldest entry is dropped — the pipeline always sees the most recent
    signal, never stale buffered data.

    Parameters
    ----------
    streamer    : RxStreamer from open_usrp()
    metadata    : RXMetadata from open_usrp()
    frame_queue : queue.Queue(maxsize=N) shared with the STFT/inference side
    stop_event  : threading.Event — set to stop the thread gracefully

    Returns
    -------
    thread : started Thread (daemon=True)
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
                print(f"[USRP] Capture error on frame {frame_idx}: {exc}")
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
    p.add_argument("--addr", default="192.168.10.2", help="X300 IP address")
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
        t0 = time.perf_counter()
        iq = capture_frame(streamer, metadata)
        ms = (time.perf_counter() - t0) * 1000
        pwr = 10 * np.log10(np.mean(np.abs(iq) ** 2) + 1e-12)
        print(f"  Frame {i}: shape={iq.shape}  dtype={iq.dtype}"
              f"  power={pwr:.2f} dBFS  time={ms:.1f} ms")

    close_usrp(usrp, streamer)