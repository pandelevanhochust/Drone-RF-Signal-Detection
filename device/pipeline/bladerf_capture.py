"""
bladerf_capture.py
==================
BladeRF USB device management and continuous IQ frame capture.

Responsibilities
----------------
- Open and configure the BladeRF RX channel (frequency, sample rate,
  bandwidth, gain, sync interface)
- Capture exactly one 80 ms IQ frame per call (4,800,000 SC16Q11 samples)
- Run a background producer thread that pushes frames onto a queue

This module has NO dependency on numpy STFT, PIL, TFLite, or AI inference.
It only depends on: bladerf, numpy, threading, queue.

RF parameters
-------------
    Center frequency : 2.375 GHz
    Sample rate      : 60 MHz
    Bandwidth        : 60 MHz
    Frame duration   : 80 ms  →  4,800,000 samples per frame

SC16Q11 format
--------------
    Each sample from BladeRF is a pair of int16 (I, Q) with values in
    [-2048, 2047] (12-bit ADC data packed in a 16-bit container).
    Conversion to normalised complex float32:
        iq = (I + jQ) / 2048.0   →  approx [-1, 1]

Setup (run once on RB3 Gen 2)
------------------------------
    sudo apt update
    sudo apt install cmake python3-pip libusb-1.0-0 -y
    cd ~ && git clone --depth 1 https://github.com/Nuand/bladeRF.git
    cd bladeRF && mkdir host/build && cd host/build
    cmake ../ -DINSTALL_UDEV_RULES=ON
    make -j4 && sudo make install && sudo ldconfig

    # Allow USB access without sudo
    sudo adduser $USER plugdev
    # Log out and back in, then verify:
    bladeRF-cli -p
    bladeRF-cli -e "version"

    pip3 install bladerf numpy

Standalone test
---------------
    python3 bladerf_capture.py
    # Opens device, captures 3 frames, prints shape and IQ power range
"""

import queue
import threading
import time

import numpy as np
import bladerf

# ─────────────────────────────────────────────────────────────────────────────
#  RF constants — edit here if your collection parameters differ
# ─────────────────────────────────────────────────────────────────────────────

CENTER_FREQ_HZ = 2_375_000_000      # 2.375 GHz
SAMPLE_RATE_HZ = 60_000_000         # 60 MHz
BANDWIDTH_HZ   = 60_000_000         # 60 MHz
FRAME_DURATION = 0.080              # 80 ms per frame
NUM_SAMPLES    = int(SAMPLE_RATE_HZ * FRAME_DURATION)   # 4,800,000

# sync_config tuning — increase num_buffers / num_transfers if you see
# sample drops on a busy USB bus
_NUM_BUFFERS   = 16
_BUFFER_SIZE   = 8192
_NUM_TRANSFERS = 8
_STREAM_TIMEOUT_MS = 3500


# ─────────────────────────────────────────────────────────────────────────────
#  Device open / close
# ─────────────────────────────────────────────────────────────────────────────

def open_bladerf(gain: int = None):
    """
    Open the first available BladeRF device and configure RX channel 0.

    Parameters
    ----------
    gain : int or None
        Manual RX gain in dB.  Pass None to use Fast-Attack AGC (default).
        Typical manual range for bladeRF 2.0: 0 – 60 dB.

    Returns
    -------
    dev : bladerf.BladeRF
        Configured, enabled BladeRF handle. Pass to capture_frame() and
        close_bladerf() when done.

    Notes
    -----
    To target a specific unit by serial number:
        dev = bladerf.BladeRF("*:serial=ABCD1234")
    """
    try:
        import bladerf as _bladerf
    except ImportError:
        raise ImportError(
            "bladerf Python bindings not installed.\n"
            "  pip3 install bladerf\n"
            "  or build from source: https://github.com/Nuand/bladeRF"
        )

    print("[BladeRF] Opening device ...")
    dev  = _bladerf.BladeRF()
    info = dev.get_devinfo()
    print(f"  ✓ Found    : {info}")

    ch = _bladerf.CHANNEL_RX(0)

    dev.set_frequency(ch, CENTER_FREQ_HZ)
    print(f"  Freq       : {dev.get_frequency(ch) / 1e9:.4f} GHz")

    dev.set_sample_rate(ch, SAMPLE_RATE_HZ)
    print(f"  Sample rate: {dev.get_sample_rate(ch) / 1e6:.1f} MHz")

    dev.set_bandwidth(ch, BANDWIDTH_HZ)
    print(f"  Bandwidth  : {dev.get_bandwidth(ch) / 1e6:.1f} MHz")

    if gain is not None:
        dev.set_gain_mode(ch, bladerf._bladerf.GainMode.Manual)
        dev.set_gain(ch, gain)
        print(f"  Gain       : {gain} dB (manual)")
    else:
        dev.set_gain_mode(ch, bladerf._bladerf.GainMode.FastAttack_AGC)
        print(f"  Gain       : AGC (fast attack)")

    dev.sync_config(
        layout        = bladerf._bladerf.ChannelLayout.RX_X1,
        fmt           = bladerf._bladerf.Format.SC16_Q11,
        num_buffers   = _NUM_BUFFERS,
        buffer_size   = _BUFFER_SIZE,
        num_transfers = _NUM_TRANSFERS,
        stream_timeout= _STREAM_TIMEOUT_MS,
    )

    dev.enable_module(ch, True)
    print(f"  ✓ RX channel 0 enabled\n")
    return dev


def close_bladerf(dev) -> None:
    """
    Disable RX channel 0 and release the device handle.
    Always call this before exiting, even on error.
    """
    try:
        import bladerf as _bladerf
        dev.enable_module(_bladerf.CHANNEL_RX(0), False)
    except Exception:
        pass
    try:
        dev.close()
    except Exception:
        pass
    print("[BladeRF] Device closed.")


# ─────────────────────────────────────────────────────────────────────────────
#  Single frame capture
# ─────────────────────────────────────────────────────────────────────────────

def capture_frame(dev) -> np.ndarray:
    """
    Capture exactly NUM_SAMPLES IQ samples (one 80 ms frame).

    Returns
    -------
    iq : complex64 ndarray, shape (NUM_SAMPLES,)
        Values in approximately [-1.0, 1.0].

    Raises
    ------
    RuntimeError if sync_rx times out (device disconnected, USB issue).

    SC16Q11 → complex float32
    -------------------------
    BladeRF stores samples as interleaved int16 pairs [I0, Q0, I1, Q1, ...].
    buf[0::2] = I channel,  buf[1::2] = Q channel.
    Divide by 2048.0 to normalise the 12-bit values to [-1, 1].
    """
    buf = np.zeros(NUM_SAMPLES * 2, dtype=np.int16)
    dev.sync_rx(buf, NUM_SAMPLES, timeout_ms=5000)
    iq = (buf[0::2].astype(np.float32) +
          1j * buf[1::2].astype(np.float32)) / 2048.0
    return iq.astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
#  Background capture thread
# ─────────────────────────────────────────────────────────────────────────────

def start_capture_thread(
    dev,
    frame_queue : queue.Queue,
    stop_event  : threading.Event,
) -> threading.Thread:
    """
    Start a daemon thread that continuously captures IQ frames and pushes
    them onto frame_queue.

    If frame_queue is full (inference hasn't consumed the previous frame),
    the oldest entry is discarded so the pipeline always sees the most
    recent signal — never stale buffered data.

    Parameters
    ----------
    dev         : open BladeRF handle from open_bladerf()
    frame_queue : queue.Queue(maxsize=N) shared with the STFT/inference side
    stop_event  : threading.Event — set it to stop the thread gracefully

    Returns
    -------
    thread : the started Thread object (daemon=True, joins automatically
             when the main process exits)
    """
    def _run():
        frame_idx = 0
        while not stop_event.is_set():
            try:
                iq = capture_frame(dev)
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()   # drop oldest
                    except queue.Empty:
                        pass
                frame_queue.put_nowait(iq)
                frame_idx += 1
            except Exception as exc:
                print(f"[BladeRF] Capture error on frame {frame_idx}: {exc}")
                time.sleep(0.01)

    t = threading.Thread(target=_run, name="BladeRF-Capture", daemon=True)
    t.start()
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    try:
        import bladerf  # noqa: F401
    except ImportError:
        print("ERROR: bladerf not installed.  pip3 install bladerf")
        sys.exit(1)

    dev = open_bladerf(gain=None)
    print(f"Capturing 3 test frames ({NUM_SAMPLES:,} samples each) ...\n")

    for i in range(3):
        t0 = time.perf_counter()
        iq = capture_frame(dev)
        elapsed = (time.perf_counter() - t0) * 1000
        power_db = 10 * np.log10(np.mean(np.abs(iq) ** 2) + 1e-12)
        print(f"  Frame {i}: shape={iq.shape}  dtype={iq.dtype}"
              f"  power={power_db:.2f} dBFS  time={elapsed:.1f} ms")

    close_bladerf(dev)