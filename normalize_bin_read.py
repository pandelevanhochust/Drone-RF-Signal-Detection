import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, windows


def compute_spectrogram_final(file_path, sample_rate, center_freq, duration_ms=200, nfft=4096):
    num_samples = int(sample_rate * (duration_ms / 1000))
    data_map = np.memmap(file_path, dtype=np.int16, mode='r')

    # Ensure we don't read past the end of the file
    raw_chunk = data_map[:2 * num_samples]

    # 1. Convert to float and NORMALIZE (Divide by 32768 because it's 16-bit)
    i_ch = raw_chunk[0::2].astype(np.float32) / 32768.0
    q_ch = raw_chunk[1::2].astype(np.float32) / 32768.0

    # i_ch = raw_chunk[0::2].astype(np.float32) / 2048.0
    # q_ch = raw_chunk[1::2].astype(np.float32) / 2048.0

    # 2. Subtract Mean (Removes that annoying red horizontal line)
    i_ch -= np.mean(i_ch)
    q_ch -= np.mean(q_ch)

    complex_chunk = i_ch + 1j * q_ch

    f, t, Zxx = stft(complex_chunk, sample_rate,
                     return_onesided=False,
                     window=windows.hamming(nfft),
                     nperseg=nfft)

    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    return f, t, Zxx


# --- Execution ---
FILE_PATH = r"2toan.bin"
FS = 28e6
CENTER_FREQ = 2.445e9

# FILE_PATH = r"MAV_1110_04.dat"
# FS = 60e6
# CENTER_FREQ = 2.375e9

f, t, Zxx = compute_spectrogram_final(FILE_PATH, FS, CENTER_FREQ,duration_ms=2)

# Convert to dB
spec_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-12)

plt.figure(figsize=(12, 6))
extent = [t[0] * 1000, t[-1] * 1000, (f[0] + CENTER_FREQ) / 1e6, (f[-1] + CENTER_FREQ) / 1e6]

# 3. SET LIMITS (vmin and vmax) to clean up the "Green" noise
# This forces the background to be blue
plt.imshow(spec_db, aspect='auto', extent=extent, origin='lower',
           cmap='jet', vmin=-120, vmax=-40)

plt.title("Cleaned BladeRF Spectrogram")
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (MHz)")
plt.colorbar(label="Intensity (dB)")
plt.show()