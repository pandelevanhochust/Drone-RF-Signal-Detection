"""
stft_preprocessor.py
====================
Chuyển đổi một IQ frame (complex64 ndarray) thành tensor spectrogram STFT
chuẩn hoá về [0.0, 1.0] sẵn sàng đưa vào mô hình TFLite EfficientNet-B0.

Pipeline xử lý (phải khớp chính xác với get_transforms() trong train):
    IQ → scipy STFT → fftshift → dB → cắt skirt → chuẩn hoá min-max → uint8
    → resize bilinear (256×512) → viridis LUT → /255.0 → NCHW float32

    KHÔNG áp dụng ImageNet mean/std.
    Script train chỉ dùng ToTensor() → [0.0, 1.0].

Thông số STFT (khớp chính xác với compute_spectrogram_efficient lúc tạo dataset):
    thư viện        : scipy.signal.stft
    nfft            : 1024
    window          : hamming(1024)
    return_onesided : False
    noverlap        : 0  (hop = nfft)
    fftshift        : có
    thang dB        : 10*log10(|Zxx|^2+1e-10)
    colormap        : viridis
    origin          : lower
    kích thước ra   : (256, 512)  resize bilinear

Standalone test
---------------
    python3 stft_preprocessor.py
"""

import os
import time

import matplotlib
import numpy as np
from PIL import Image
from scipy.signal import stft as scipy_stft
from scipy.signal.windows import hamming

# ─────────────────────────────────────────────────────────────────────────────
#  Constants — matched to training script
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE_HZ = 25_000_000
NFFT           = 1024
NOVERLAP       = 0
WINDOW         = hamming(NFFT)

IMG_H, IMG_W   = 256, 512

# Kept for save_spectrogram_png() visualisation ONLY — not used in inference
IMAGENET_MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Anti-aliasing filter skirt crop
SKIRT_CROP     = 0.12

# Viridis colormap LUT — matches training cmap='viridis'
_VIRIDIS_LUT = (
    matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255
).astype(np.uint8)   # (256, 3) uint8 RGB


# ─────────────────────────────────────────────────────────────────────────────
#  Core transform
# ─────────────────────────────────────────────────────────────────────────────

def iq_to_spectrogram(iq: np.ndarray) -> np.ndarray:
    """
    Convert one IQ frame to a [0.0, 1.0] normalised STFT spectrogram tensor.

    Pipeline (matches train_and_export.py get_transforms exactly):
        scipy STFT → fftshift → dB → crop skirt → min-max → uint8
              → resize (256×512) → viridis LUT → /255.0 → NCHW float32

    KHÔNG áp dụng ImageNet mean/std — train chỉ dùng ToTensor() (/255).

    Tham số
    -------
    iq : complex64 ndarray, shape (N,)

    Trả về
    ------
    tensor : float32 ndarray, shape (1, 3, 256, 512)  NCHW  giá trị [0.0, 1.0]
    """
    # Step 1: scipy STFT
    _, _, Zxx = scipy_stft(
        iq,
        fs              = SAMPLE_RATE_HZ,
        window          = WINDOW,
        nperseg         = NFFT,
        noverlap        = NOVERLAP,
        return_onesided = False,
    )

    # Step 2: fftshift — DC to centre row
    Zxx = np.fft.fftshift(Zxx, axes=0)

    # Step 3: dB scale
    spec_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    # Step 4: crop filter skirt + min-max normalise + flip freq axis
    skirt        = int(NFFT * SKIRT_CROP)
    spec_db_crop = spec_db[skirt : NFFT - skirt, :]
    s_min, s_max = float(spec_db_crop.min()), float(spec_db_crop.max())
    denom        = s_max - s_min if s_max > s_min else 1.0
    norm8        = ((spec_db_crop[::-1] - s_min) / denom * 255).clip(0, 255).astype(np.uint8)

    # Step 5: resize grayscale to (256, 512)
    small = np.array(
        Image.fromarray(norm8, mode="L").resize((IMG_W, IMG_H), Image.BILINEAR)
    )   # (256, 512) uint8

    # Step 6: viridis colormap LUT
    rgb = _VIRIDIS_LUT[small]   # (256, 512, 3) uint8

    # Step 7: /255.0 → NCHW float32  — matches ToTensor() in training exactly
    # DO NOT apply IMAGENET_MEAN / IMAGENET_STD here.
    arr = rgb.astype(np.float32) / 255.0              # HWC [0.0, 1.0]
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)   # (1,3,H,W)


def iq_to_spectrogram_debug(iq: np.ndarray, debug_dir: str = "debug_stft") -> np.ndarray:
    #Debug
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(debug_dir, exist_ok=True)

    # Step 1
    _, _, Zxx = scipy_stft(iq, fs=SAMPLE_RATE_HZ, window=WINDOW,
                           nperseg=NFFT, noverlap=NOVERLAP, return_onesided=False)
    Zxx     = np.fft.fftshift(Zxx, axes=0)
    spec_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(debug_dir, "step1_raw_db.png"),
                dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  [debug] step1_raw_db.png  dB=[{spec_db.min():.1f}, {spec_db.max():.1f}]")

    # Step 2
    skirt        = int(NFFT * SKIRT_CROP)
    spec_db_crop = spec_db[skirt : NFFT - skirt, :]
    s_min, s_max = float(spec_db_crop.min()), float(spec_db_crop.max())
    denom        = s_max - s_min if s_max > s_min else 1.0
    norm8        = ((spec_db_crop[::-1] - s_min) / denom * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(norm8, mode="L").save(os.path.join(debug_dir, "step2_norm8.png"))
    print(f"  [debug] step2_norm8.png   shape={norm8.shape}")

    # Step 3
    small = np.array(Image.fromarray(norm8, mode="L").resize((IMG_W, IMG_H), Image.BILINEAR))
    Image.fromarray(small, mode="L").save(os.path.join(debug_dir, "step3_small.png"))
    print(f"  [debug] step3_small.png   shape={small.shape}")

    # Step 4
    rgb = _VIRIDIS_LUT[small]
    Image.fromarray(rgb, mode="RGB").save(os.path.join(debug_dir, "step4_rgb.png"))
    print(f"  [debug] step4_rgb.png     shape={rgb.shape}")

    # Step 5: /255.0 only — no ImageNet norm
    arr    = rgb.astype(np.float32) / 255.0
    tensor = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    # Save step 5 — scale back to uint8 (should look identical to step4)
    arr_back = (tensor[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr_back).save(os.path.join(debug_dir, "step5_final.png"))
    print(f"  [debug] step5_final.png   should match step4_rgb.png exactly (no norm applied)")

    return tensor


def save_spectrogram_png(tensor: np.ndarray, path: str) -> None:
    """
    Lưu tensor (1, 3, H, W) [0,1] thành file PNG RGB có thể xem được.
    Không cần denormalise vì không áp dụng ImageNet norm.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = (tensor[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N     = int(SAMPLE_RATE_HZ * 0.080)
    t_arr = np.arange(N, dtype=np.float32) / SAMPLE_RATE_HZ

    print("=" * 62)
    print("  stft_preprocessor — [0,1] range, NO ImageNet norm")
    print("=" * 62)

    iq = (
        0.5 * np.exp(1j * 2 * np.pi * 5e6 * t_arr) *
        (0.5 + 0.5 * np.sin(2 * np.pi * 100 * t_arr)) +
        0.2 * np.exp(1j * 2 * np.pi * -15e6 * t_arr)
    ).astype(np.complex64)

    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        tensor = iq_to_spectrogram(iq)
        times.append((time.perf_counter() - t0) * 1000)

    assert tensor.shape == (1, 3, IMG_H, IMG_W)
    assert np.all(np.isfinite(tensor))
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0, \
        f"Range error: [{tensor.min():.4f}, {tensor.max():.4f}] — expected [0.0, 1.0]"

    print(f"  shape      : {tensor.shape}")
    print(f"  range      : [{tensor.min():.4f}, {tensor.max():.4f}]  (expected [0.0, 1.0])")
    print(f"  avg time   : {np.mean(times[2:]):.1f} ms")
    save_spectrogram_png(tensor, "debug_stft/inference_spec.png")
    print(f"  saved      : debug_stft/inference_spec.png")
    print()

    print("  Running debug pipeline (saves step1–step5 PNGs) ...")
    iq_to_spectrogram_debug(iq, "debug_stft")
    print("\n  All tests passed ✓")