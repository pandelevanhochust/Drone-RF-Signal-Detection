import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# ================= CONFIGURATION =================
FILE_PATH = r'non_toan.bin' # Use 'r' before path
SAMPLE_RATE = 28e6      # 28 MHz (as set in bladeRF-cli)
CENTER_FREQ = 2.445e9   # For plot labeling (optional)
START_SAMPLE = 1000000  # Jump to this sample index
WINDOW_SIZE = 100000    # Number of samples to analyze (STFT width)
# =================================================

def draw_time_frequency_stft_from_bin():
    # Validation
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    # 1. Load the slice
    # 1 sample = 4 bytes (2x int16). 
    bytes_per_sample = 4 
    offset = START_SAMPLE * bytes_per_sample

    try:
        with open(FILE_PATH, 'rb') as f:
            f.seek(offset)
            # Read I and Q pairs (count is total integers, so 2 * WINDOW_SIZE)
            raw_data = np.fromfile(f, dtype=np.int16, count=WINDOW_SIZE * 2)
    except Exception as e:
        print(f"Read Error: {e}")
        return

    # 2. Convert to Complex IQ (Normalized for 12-bit BladeRF)
    # We cast to complex64 for signal processing efficiency
    data = (raw_data[0::2] + 1j * raw_data[1::2]).astype(np.complex64) / 2048.0
    
    print(f"Processing {WINDOW_SIZE} samples from {FILE_PATH}...")

    # 3. Configure STFT
    nperseg = 1024 # Increased for better frequency resolution on wideband signals
    noverlap = nperseg // 2 

    f, t, Sxx = signal.spectrogram(
        data, 
        fs=SAMPLE_RATE, 
        return_onesided=False, 
        nperseg=nperseg, 
        noverlap=noverlap,
        detrend=False
    )

    # 4. Shift zero-frequency to center
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    # 5. Plotting
    plt.figure(figsize=(12, 7))
    
    # Frequency is shown relative to the center (e.g., -14MHz to +14MHz)
    # Convert time to microseconds (us) for very short windows
    plt.pcolormesh(t * 1e6, f / 1e6, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    
    plt.colorbar(label='Intensity [dB]')
    plt.title(f"STFT: DJI OcuSync Signal Analysis\nFile: {os.path.basename(FILE_PATH)}")
    plt.ylabel("Frequency [MHz] (Offset from Center)")
    plt.xlabel("Time [μs]")
    plt.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_time_frequency_stft_from_bin()