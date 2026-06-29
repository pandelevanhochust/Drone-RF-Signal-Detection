import os
import numpy as np
import matplotlib.pyplot as plt

path = ("D:\CODIng\Thesis\SpectrumAnalyzer\dat_data\droneV2_data\MAV_0010_00.dat")
file_size = os.path.getsize(path)
total_floats = file_size / 4  # 4 bytes per float32

print(f"File Size: {file_size / (1024**3):.2f} GB")
print(f"Total Raw Float32 Samples: {total_floats:,.0f}")

# If they are I/Q pairs (Complex), the number of complex points is:
print(f"Total Complex (I/Q) Points: {total_floats / 2:,.0f}")

with open(path, "rb") as f:
    raw_samples = np.fromfile(f, dtype="float32", count=100000)
    plt.figure(figsize=(12, 4))
    plt.plot(raw_samples[:5000])
    plt.title("Raw Signal - Time Domain (First 5000 Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

