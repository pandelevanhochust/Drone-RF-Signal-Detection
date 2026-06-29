"""
usrp_capture.py
===============
Thu tín hiệu IQ từ USRP X300 với chế độ quét tần số (sweep).

Thông số quét mặc định:
    Bắt đầu : 2.400 GHz
    Kết thúc : 2.475 GHz
    Bước     : 25 MHz  →  4 tần số: 2.400 / 2.425 / 2.450 / 2.475 GHz
    Số frame : 5 frame/tần số  (~575 ms/kênh)
    Chu kỳ   : ~2.3 s cho một vòng quét đầy đủ

Thông số RF (khớp với dữ liệu huấn luyện):
    Sample rate : 25 MHz  (decimation=8)
    Bandwidth   : 30 MHz
    Frame size  : 2.000.000 mẫu  (80 ms @ 25 MHz)
    Stream fmt  : fc32 CPU / sc16 wire

Luồng thread:
    Thread nền (daemon)  →  vòng lặp quét tần số, đẩy (freq_hz, iq) vào frame_queue
    Main thread          →  kéo (freq_hz, iq) từ queue, chạy STFT + inference
"""

import queue
import threading
import time

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Cấu hình sweep
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_START_HZ  = 2_400_000_000    # 2.400 GHz
SWEEP_STOP_HZ   = 2_475_000_000    # 2.475 GHz
SWEEP_STEP_HZ   =    25_000_000    # 25 MHz mỗi bước
FRAMES_PER_STEP = 5                # số frame thu trước khi đổi tần số

# Danh sách tần số quét: [2.400, 2.425, 2.450, 2.475] GHz
SWEEP_FREQS_HZ = list(range(SWEEP_START_HZ,
                             SWEEP_STOP_HZ + SWEEP_STEP_HZ,
                             SWEEP_STEP_HZ))
NUM_SWEEP_STEPS = len(SWEEP_FREQS_HZ)

# ─────────────────────────────────────────────────────────────────────────────
#  Hằng số RF
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE_HZ  = 25_000_000        # 25 MHz — khớp với tham số --fs 25e6 lúc train
BANDWIDTH_HZ    = 30_000_000
FRAME_DURATION  = 0.080             # 80 ms
NUM_SAMPLES     = int(SAMPLE_RATE_HZ * FRAME_DURATION)   # 2.000.000 mẫu

RECV_CHUNK      = 100_000           # số mẫu đọc mỗi lần recv()
RECV_TIMEOUT    = 3.0               # timeout recv() tính bằng giây
MAX_OVERFLOWS   = 3                 # số lần overflow liên tiếp trước khi restart stream
DEFAULT_GAIN_DB = 30.0

STREAM_CPU_FMT  = "fc32"            # định dạng dữ liệu phía CPU
STREAM_WIRE_FMT = "sc16"            # định dạng truyền qua dây (10GbE)

# Thời gian chờ PLL khóa sau khi đổi tần số (X300 lock ~10 ms, 50 ms an toàn)
RETUNE_SETTLE_S = 0.050


# ─────────────────────────────────────────────────────────────────────────────
#  Mở / đóng thiết bị
# ─────────────────────────────────────────────────────────────────────────────

def open_usrp(addr: str = "192.168.5.111", gain: float = DEFAULT_GAIN_DB):
    """Kết nối và cấu hình USRP X300, trả về (usrp, streamer, metadata)."""
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

    # Cài sample rate
    usrp.set_rx_rate(SAMPLE_RATE_HZ, 0)
    actual_rate = usrp.get_rx_rate(0)
    print(f"  Sample rate : {actual_rate/1e6:.3f} MHz  "
          f"(decimation={int(200e6/actual_rate)})")

    if abs(actual_rate - SAMPLE_RATE_HZ) > 1e3:
        print(f"  ⚠  Rate mismatch: requested {SAMPLE_RATE_HZ/1e6} MHz, "
              f"got {actual_rate/1e6:.3f} MHz — STFT will NOT match training.")

    # Tần số ban đầu — bắt đầu sweep tại đây
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

    # Anten
    usrp.set_rx_antenna("RX2", 0)
    print(f"  Antenna     : {usrp.get_rx_antenna(0)}")

    # Tạo stream
    st_args          = uhd.usrp.StreamArgs(STREAM_CPU_FMT, STREAM_WIRE_FMT)
    st_args.channels = [0]
    streamer         = usrp.get_rx_stream(st_args)
    metadata         = uhd.types.RXMetadata()

    # Bắt đầu stream liên tục
    stream_cmd            = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)

    time.sleep(0.2)   # chờ PLL khóa + FIFO đầy

    # In kế hoạch sweep
    print(f"\n  Sweep plan ({NUM_SWEEP_STEPS} steps × {FRAMES_PER_STEP} frames):")
    for i, f in enumerate(SWEEP_FREQS_HZ):
        print(f"    step {i+1:>2d} : {f/1e9:.4f} GHz")
    cycle_s = NUM_SWEEP_STEPS * FRAMES_PER_STEP * (FRAME_DURATION + RETUNE_SETTLE_S / FRAMES_PER_STEP)
    print(f"  Est. cycle  : ~{cycle_s:.1f} s per full sweep\n")

    return usrp, streamer, metadata


def close_usrp(usrp, streamer) -> None:
    """Dừng stream và giải phóng thiết bị."""
    try:
        import uhd
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stop_cmd)
        time.sleep(0.05)
    except Exception:
        pass
    print("[USRP] Stream stopped and device released.")


# ─────────────────────────────────────────────────────────────────────────────
#  Hàm nội bộ
# ─────────────────────────────────────────────────────────────────────────────

def _retune(usrp, streamer, freq_hz: int) -> None:
    """
    Đổi tần số và khởi động lại stream sạch.

    Các bước:
    1. Dừng stream
    2. Cài tần số mới
    3. Chờ RETUNE_SETTLE_S để PLL khóa
    4. Khởi động lại stream
    5. Flush một chunk rác từ FIFO — nếu bỏ qua, frame đầu tiên sau retune
       sẽ chứa mẫu IQ cũ từ tần số trước, gây sai lệch spectrogram.
    """
    try:
        import uhd

        # Dừng stream
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stop_cmd)
        time.sleep(0.01)

        # Đổi tần số
        tune_req = uhd.libpyuhd.types.tune_request(freq_hz)
        usrp.set_rx_freq(tune_req, 0)
        time.sleep(RETUNE_SETTLE_S)   # chờ PLL ổn định

        # Khởi động lại stream
        start_cmd            = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        start_cmd.stream_now = True
        streamer.issue_stream_cmd(start_cmd)
        time.sleep(0.01)

        # Xả mẫu cũ trong FIFO
        flush_buf = np.zeros((1, RECV_CHUNK), dtype=np.complex64)
        meta      = uhd.types.RXMetadata()
        streamer.recv(flush_buf, meta, timeout=0.5)

    except Exception as exc:
        print(f"[USRP] Retune to {freq_hz/1e9:.4f} GHz failed: {exc}")


def _restart_stream(streamer) -> None:
    """Khôi phục stream sau overflow mà không cần đổi tần số."""
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
#  Thu một frame IQ
# ─────────────────────────────────────────────────────────────────────────────

def capture_frame(streamer, metadata) -> np.ndarray:
    """Thu đúng NUM_SAMPLES mẫu IQ (một frame 80 ms). Trả về mảng complex64 shape (N,)."""
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
#  Thread thu sweep tần số
# ─────────────────────────────────────────────────────────────────────────────

def start_capture_thread(
    streamer,
    metadata,
    frame_queue : queue.Queue,
    stop_event  : threading.Event,
    usrp        = None,
) -> threading.Thread:
    """
    Chạy thread nền liên tục quét SWEEP_FREQS_HZ, đẩy tuple (freq_hz, iq) vào queue.

    - Thu FRAMES_PER_STEP frame tại mỗi tần số trước khi đổi kênh.
    - Nếu queue đầy → bỏ frame cũ nhất, ưu tiên frame mới nhất cho inference.
    - Truyền usrp=None để cố định tần số (không sweep).

    Item trong queue: (freq_hz: int, iq: np.ndarray complex64)
    """
    sweep_enabled = usrp is not None

    def _run():
        step_idx    = 0    # vị trí hiện tại trong SWEEP_FREQS_HZ
        frame_count = 0    # số frame đã thu ở bước hiện tại

        while not stop_event.is_set():
            current_freq = SWEEP_FREQS_HZ[step_idx]

            try:
                iq   = capture_frame(streamer, metadata)
                item = (current_freq, iq)

                # Bỏ frame cũ nếu queue đầy
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put_nowait(item)

                frame_count += 1

                # Chuyển sang tần số tiếp theo sau đủ FRAMES_PER_STEP
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
#  Chạy thử độc lập
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="USRP X300 sweep capture test")
    p.add_argument("--addr",   default="192.168.5.111")
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