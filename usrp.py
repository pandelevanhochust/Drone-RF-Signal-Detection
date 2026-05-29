import argparse
import numpy as np
import uhd
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="USRP X300 High-Speed IQ Recorder")
    parser.add_argument("-f", "--freq", type=float, default=2375000000, help="Frequency in Hz")
    parser.add_argument("-r", "--rate", type=float, default=25000000, help="Sample rate in Hz (50MHz)")
    parser.add_argument("-g", "--gain", type=float, default=20.0, help="RX Gain")
    parser.add_argument("-n", "--nsamps", type=int, default=100000000, help="Total samples to record (2 seconds)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file path")
    args = parser.parse_args()

    print(f"Connecting to USRP X300...")
    try:
        # Pass the explicit library path hint if needed, otherwise connect directly via IP
        usrp = uhd.usrp.MultiUSRP("addr=192.168.10.2")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    # Configure Radio Hardware
    usrp.set_rx_rate(args.rate, 0)
    usrp.set_rx_freq(uhd.types.TuneRequest(args.freq), 0)
    usrp.set_rx_gain(args.gain, 0)
    time.sleep(0.5)

    print(f"--- Hardware Settings Synced ---")
    print(f"Actual Rate: {usrp.get_rx_rate(0) / 1e6} MHz")
    print(f"Actual Freq: {usrp.get_rx_freq(0) / 1e6} MHz")

    # Set up stream configuration (fc32 = Float32 Complex)
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_rx_stream(st_args)

    buffer_size = streamer.get_max_num_samps()
    if buffer_size <= 0:
        buffer_size = 4096
    recv_buffer = np.zeros((1, buffer_size), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    print(f"\nRecording {args.nsamps} samples to: {args.output}...")

    # --- FIXED ATTRIBUTE NAMING CONVENTION FOR PYUHD ---
    # Python bindings shorten the C++ naming from 'start_continuous' to 'start_cont'
    try:
        mode_enum = uhd.types.StreamMode.start_cont
        stop_enum = uhd.types.StreamMode.stop_cont
    except AttributeError:
        # Secondary fallback if your specific wheels match full uppercase abbreviations
        mode_enum = uhd.types.StreamMode.START_CONT
        stop_enum = uhd.types.StreamMode.STOP_CONT

    # Create the command passing the correct required enum type
    stream_cmd = uhd.types.StreamCMD(mode_enum)
    stream_cmd.stream_now = True
    stream_cmd.num_samps = 0
    stream_cmd.time_spec = uhd.types.TimeSpec(0.0)

    # Send the start configuration command to the radio
    usrp.issue_stream_cmd(stream_cmd)

    # Prepare stop command for later script teardown
    stop_cmd = uhd.types.StreamCMD(stop_enum)
    stop_cmd.stream_now = True
    stop_cmd.time_spec = uhd.types.TimeSpec(0.0)

    samples_collected = 0
    overflow_count = 0

    with open(args.output, "wb") as f:
        try:
            while samples_collected < args.nsamps:
                samps_to_recv = min(buffer_size, args.nsamps - samples_collected)

                # Fetch data directly into the pre-allocated slice
                num_rx_samps = streamer.recv(recv_buffer[:, :samps_to_recv], metadata, 1.0)

                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                    if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                        overflow_count += 1
                        print("O", end="", flush=True)
                    else:
                        print(f"\nStream Error: {metadata.strerror()}")
                        # If we get a timeout but collected some data, don't crash entirely
                        if samples_collected > 0:
                            break

                if num_rx_samps > 0:
                    recv_buffer[0, :num_rx_samps].tofile(f)
                    samples_collected += num_rx_samps

        except KeyboardInterrupt:
            print("\nRecording manually interrupted.")

    print(f"\n--- Done ---")
    print(f"Successfully recorded: {samples_collected} samples.")
    print(f"Total Overflows: {overflow_count}")


if __name__ == "__main__":
    main()