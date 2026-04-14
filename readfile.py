import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def draw_time_frequency_stft(file_path, start_idx=0, num_samples=50000, fs=1.0):
    # 1. Load a very short slice of the data
    # Use mmap_mode to avoid loading the whole 15M elements
    full_data = np.load(file_path, mmap_mode='r')
    data = full_data[start_idx: start_idx + num_samples]

    print(f"Analyzing {num_samples} samples starting at index {start_idx}...")

    # 2. Configure STFT parameters
    # nperseg: Controls the resolution. 256-1024 is usually good for "short" bursts.
    nperseg = 512
    noverlap = nperseg // 2  # 50% overlap for smoothness

    f, t, Sxx = signal.spectrogram(
        data,
        fs=fs,
        return_onesided=False,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False
    )

    # 3. Center the frequencies (Standard for IQ/Complex data)
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    # 4. Plotting the Image
    plt.figure(figsize=(10, 6))

    # Use log10 to convert to Decibels (dB) for better contrast
    # Shading='gouraud' makes the image look smooth rather than "blocky"
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')

    plt.colorbar(label='Intensity [dB]')
    plt.title(f"Time-Frequency Domain (STFT)\n{num_samples} samples")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.show()


if __name__ == "__main__":
    path = input("Enter .npy file path: ").strip()
    # Analyzing a tiny 50,000 sample window of your 15,000,000 sample file
    draw_time_frequency_stft(path, start_idx=50000, num_samples=50000, fs=1.0)