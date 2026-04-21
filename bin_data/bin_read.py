import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import os

def compute_spectrogram_efficient(file_path, sample_rate, center_freq, duration_ms=10, nfft=2048):
    # 1. Calculate how many samples to read
    # Each sample is 4 bytes (2 bytes for I, 2 bytes for Q)
    num_samples = int(sample_rate * (duration_ms / 1000))

    # 2. Use memmap to "link" the file without loading it all
    # dtype is int16 because bladeRF uses SC16
    data_map = np.memmap(file_path, dtype=np.int16, mode='r')

    # 3. Pull only the slice we need (I and Q are interleaved)
    # Total integers to pull = 2 * num_samples
    raw_chunk = data_map[:2 * num_samples]

    # 4. Convert only this chunk to complex
    i_ch = raw_chunk[0::2].astype(np.float32)
    q_ch = raw_chunk[1::2].astype(np.float32)
    complex_chunk = i_ch + 1j * q_ch

    # 5. Perform STFT
    f, t, Zxx = stft(complex_chunk, sample_rate,
                     return_onesided=False,
                     window=windows.hamming(nfft),
                     nperseg=nfft)

    # 6. Shift for visualization
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    return f, t, Zxx

# --- Execution ---
FILE_PATH = r"1toan.bin"
# FILE_PATH = r"DJI_B1_21_04_2026/dji_hover_up.bin"
FS = 28e6
CENTER_FREQ = 2.445e9

# FILE_PATH = r"MAV_1110_04.dat"
# FS = 60e6
# CENTER_FREQ = 2.4375e9

# Let's just look at the first 20ms to save memory
f, t, Zxx = compute_spectrogram_efficient(FILE_PATH, FS, CENTER_FREQ, duration_ms=100)

# Convert to dB
spec_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

# Plot
plt.figure(figsize=(12, 6))
extent = [t[0] * 1000, t[-1] * 1000, (f[0] + CENTER_FREQ) / 1e6, (f[-1] + CENTER_FREQ) / 1e6]

plt.imshow(spec_db, aspect='auto', extent=extent, origin='lower', cmap='jet')
filename = os.path.basename(FILE_PATH)
plt.title(f"Spectrogram: {filename}")
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (MHz)")
plt.colorbar(label="Intensity (dB)")
plt.show()