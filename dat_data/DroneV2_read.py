import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, windows


def compute_spectrogram_efficient(file_path, sample_rate, center_freq,start_ms=0, duration_ms=20, nfft=1024):
    # 1. Calculate how many samples to read
    # Each sample is 4 bytes (2 bytes for I, 2 bytes for Q)
    skip_num_samples = int(sample_rate * (start_ms / 1000))
    num_samples = int(sample_rate * (duration_ms / 1000))

    # 2. Use memmap to "link" the file without loading it all
    # dtype is float32 bcs GNU Radio
    start_idx = 2 * skip_num_samples
    end_idx = start_idx + (2 * num_samples)

    data_map = np.memmap(file_path, dtype=np.float32, mode='r')

    # 3. Pull only the slice we need (I and Q are interleaved)
    # Total integers to pull = 2 * num_samples
    raw_chunk = data_map[start_idx:end_idx]

    # 4. Convert only this chunk to complex
    i_ch = raw_chunk[0::2]
    q_ch = raw_chunk[1::2]
    complex_chunk = i_ch + 1j * q_ch

    # 5. Perform STFT
    f, t, Zxx = stft(complex_chunk, sample_rate,
                     return_onesided=False,
                     window=windows.hamming(nfft),
                     nperseg=nfft)

    # 6. Shift for visualization
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    return f, t + (start_ms / 1000), Zxx


# --- Execution ---
# FILE_PATH = r"../captures/seg_0030_2440MHz.dat"
# FILE_PATH = r"droneV2_data/DIS_0010_03.dat"
FILE_PATH = r"stft_segmentation/NewCapture/25G.23.dat"

FS = 30e6
CENTER_FREQ = 2.440e9

# Let's just look at the first 20ms to save memory
f, t, Zxx = compute_spectrogram_efficient(FILE_PATH, FS, CENTER_FREQ,1, duration_ms=1)

# Convert to dB
spec_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

# Plot
plt.figure(figsize=(12, 6))
extent = [t[0] * 1000, t[-1] * 1000, (f[0] + CENTER_FREQ) / 1e6, (f[-1] + CENTER_FREQ) / 1e6]

plt.imshow(spec_db, aspect='auto', extent=extent, origin='lower', cmap='viridis')
plt.title(f"BladeRF Spectrogram (20ms Slice at {CENTER_FREQ / 1e9} GHz)")
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (MHz)")
plt.colorbar(label="Intensity (dB)")
plt.show()