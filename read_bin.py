import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# ================= CONFIGURATION =================
FILE_PATH = r'1toan.bin'
SAMPLE_RATE = 28e6
WINDOW_SIZE = 100000  # Number of samples per image
OUTPUT_DIR = 'stft_segments'


# =================================================

def extract_all_segments():
    # 1. Validation and File Info
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    file_size_bytes = os.path.getsize(FILE_PATH)
    bytes_per_sample = 4  # 2 bytes I + 2 bytes Q
    total_samples = file_size_bytes // bytes_per_sample

    # Calculate total segments
    num_segments = total_samples // WINDOW_SIZE

    # Calculate total capture duration for your records
    duration_sec = total_samples / SAMPLE_RATE

    print(f"--- File Statistics ---")
    print(f"Total Samples:  {total_samples:,}")
    print(f"Total Duration: {duration_sec:.2f} seconds")
    print(f"Segment Count:  {num_segments} images will be created")
    print(f"-----------------------")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Loop through every segment
    for i in range(num_segments):
        start_sample = i * WINDOW_SIZE
        offset = start_sample * bytes_per_sample

        with open(FILE_PATH, 'rb') as f:
            f.seek(offset)
            raw_data = np.fromfile(f, dtype=np.int16, count=WINDOW_SIZE * 2)

        # Convert to Complex IQ
        data = (raw_data[0::2] + 1j * raw_data[1::2]).astype(np.complex64) / 2048.0

        # 3. Generate STFT
        nperseg = 1024
        noverlap = nperseg // 2
        f, t, Sxx = signal.spectrogram(data, fs=SAMPLE_RATE, return_onesided=False,
                                       nperseg=nperseg, noverlap=noverlap)

        f = np.fft.fftshift(f)
        Sxx = np.fft.fftshift(Sxx, axes=0)

        # 4. Plot and Save
        fig = plt.figure(figsize=(10, 6))
        plt.pcolormesh(t * 1e6, f / 1e6, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')

        plt.title(f"Segment {i:04d} | Start Index: {start_sample}")
        plt.ylabel("Frequency [MHz]")
        plt.xlabel("Time [μs]")

        save_path = os.path.join(OUTPUT_DIR, f"segment_{i:04d}.png")
        plt.savefig(save_path, dpi=100)  # dpi=100 keeps file size small
        plt.close(fig)  # CRITICAL: Frees memory

        if i % 10 == 0:
            print(f"Progress: {i}/{num_segments} images saved...")

    print(f"\nDone! All images are in the '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    extract_all_segments()