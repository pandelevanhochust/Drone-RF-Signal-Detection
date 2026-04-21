import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import os


def compute_spectrogram_final(file_path, sample_rate, center_freq, start_ms=0, duration_ms=200, nfft=1024):
    # Calculate how many SAMPLES to skip and how many to read
    samples_to_skip = int(sample_rate * (start_ms / 1000))
    samples_to_read = int(sample_rate * (duration_ms / 1000))

    # Each complex sample = 2 integers (I and Q)
    # We multiply by 2 to get the index in the int16 array
    start_idx = 2 * samples_to_skip
    end_idx = start_idx + (2 * samples_to_read)

    data_map = np.memmap(file_path, dtype=np.int16, mode='r')

    raw_chunk = data_map[start_idx:end_idx]

    i_ch = raw_chunk[0::2].astype(np.float32) / 32768.0
    q_ch = raw_chunk[1::2].astype(np.float32) / 32768.0
    i_ch -= np.mean(i_ch)
    q_ch -= np.mean(q_ch)
    complex_chunk = i_ch + 1j * q_ch

    f, t, Zxx = stft(complex_chunk, sample_rate, return_onesided=False,
                     window=windows.hamming(nfft), nperseg=nfft)

    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    return f, t + (start_ms / 1000), Zxx


# --- Execution ---
# FILE_PATH = r"../2toan.bin"
# FILE_PATH = r"DJI_B1_21_04_2026/dji_cao50_xa100_low.bin"
FILE_PATH = r"DJI_B1_21_04_2026/dji_hover_up.bin"


FS = 60e6
CENTER_FREQ = 2.445e9

# FILE_PATH = r"MAV_1110_04.dat"
# FS = 60e6
# CENTER_FREQ = 2.375e9

f, t, Zxx = compute_spectrogram_final(FILE_PATH, FS, CENTER_FREQ,start_ms=1000,duration_ms=200)

# Convert to dB
spec_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-12)

plt.figure(figsize=(12, 6))
extent = [t[0] * 1000, t[-1] * 1000, (f[0] + CENTER_FREQ) / 1e6, (f[-1] + CENTER_FREQ) / 1e6]

# 3. SET LIMITS (vmin and vmax) to clean up the "Green" noise
# This forces the background to be blue
plt.imshow(spec_db, aspect='auto', extent=extent, origin='lower',
           cmap='jet', vmin=-120, vmax=-40)

filename = os.path.basename(FILE_PATH)
plt.title(f"Spectrogram: {filename}")
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (MHz)")
plt.colorbar(label="Intensity (dB)")
plt.show()