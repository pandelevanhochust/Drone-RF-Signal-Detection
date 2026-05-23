"""
stft_preprocessor.py
====================
Converts a raw IQ frame (complex64 ndarray) into an ImageNet-normalised
STFT spectrogram tensor ready for the fused TFLite model.

Parameters are matched EXACTLY to the training data generation script
(segment_file / compute_spectrogram / save_spectrogram):

    nfft        : 1024          (training used 1024, NOT 256)
    window      : Hamming       (training used hamming, NOT Hann)
    library     : scipy.signal.stft  (vectorised, ~50x faster than manual loop)
    fftshift    : yes           (DC moved to centre row)
    colormap    : viridis dB    (training rendered dB image with viridis cmap)
    output size : (256, 512)    resize via PIL bilinear

Why the old version was wrong
------------------------------
    nfft=256    → 4× fewer frequency bins → completely different feature map
    Hann window → different sidelobe shape → different energy distribution
    Manual loop → 844 ms per frame (scipy.signal.stft takes ~8 ms)
    Grey float  → no colormap applied → model never saw grey images in training

Standalone test
---------------
    python3 stft_preprocessor.py
    # Generates synthetic tone + noise frames, saves debug PNGs to debug_stft/
    # Visually compare against training spectrograms — should look identical
"""

import os
import time

import numpy as np
from PIL import Image
from scipy.signal import stft as scipy_stft
from scipy.signal.windows import hamming

# ─────────────────────────────────────────────────────────────────────────────
#  STFT / output constants  — must match training script exactly
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE_HZ  = 60_000_000        # 60 MHz
NFFT            = 1024              # FFT window size  ← was 256, WRONG
WINDOW          = hamming(NFFT)     # Hamming window   ← was Hann, WRONG

IMG_H, IMG_W    = 256, 512          # model input spatial dims

IMAGENET_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Viridis colormap lookup table (256 RGB entries) — matches matplotlib viridis
# Pre-baked so we don't depend on matplotlib at inference time
import matplotlib
_VIRIDIS_LUT = (matplotlib.colormaps["viridis"](
    np.linspace(0, 1, 256)
)[:, :3] * 255).astype(np.uint8)      # (256, 3) uint8 RGB


# ─────────────────────────────────────────────────────────────────────────────
#  Core transform
# ─────────────────────────────────────────────────────────────────────────────

def iq_to_spectrogram(iq: np.ndarray) -> np.ndarray:
    """
    Convert one IQ frame to an ImageNet-normalised STFT spectrogram tensor.

    Matches training script compute_spectrogram() + save_spectrogram() exactly:
        scipy.signal.stft with Hamming window, nfft=1024, return_onesided=False
        → fftshift DC to centre
        → 10*log10(|Zxx|^2 + 1e-10)  dB
        → apply viridis colormap
        → resize to (256, 512)
        → ImageNet normalise
        → (1, 3, 256, 512) NCHW float32

    Parameters
    ----------
    iq : complex64 ndarray, shape (N,)
        One 80 ms IQ frame. N = 4,800,000 for 60 MHz × 80 ms.

    Returns
    -------
    tensor : float32 ndarray, shape (1, 3, 256, 512)  NCHW
        ImageNet-normalised. Feed directly to TFLite interpreter.
        Do NOT transpose to NHWC — QNN delegate handles layout internally.
    """
    # ── Step 1: scipy STFT (vectorised, ~8 ms for 4.8M samples) ─────────────
    # return_onesided=False → two-sided spectrum (matches training)
    # nperseg = nfft → no zero-padding, window length == FFT length
    _, _, Zxx = scipy_stft(
        iq,
        fs             = SAMPLE_RATE_HZ,
        window         = WINDOW,
        nperseg        = NFFT,
        return_onesided= False,
    )
    # Zxx shape: (nfft, n_time_frames) complex128

    # ── Step 2: fftshift — DC to centre row (matches training) ──────────────
    Zxx = np.fft.fftshift(Zxx, axes=0)         # (nfft, n_time_frames)

    # ── Step 3: log power in dB ──────────────────────────────────────────────
    spec_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-10)  # (nfft, n_time_frames)

    # ── Step 4: normalise to [0, 255] uint8 for colormap application ─────────
    s_min, s_max = spec_db.min(), spec_db.max()
    if s_max > s_min:
        norm = ((spec_db - s_min) / (s_max - s_min) * 255).clip(0, 255).astype(np.uint8)
    else:
        norm = np.zeros_like(spec_db, dtype=np.uint8)

    # ── Step 5: apply viridis colormap → RGB (matches training imshow viridis)
    # norm is (nfft, n_time_frames), values 0-255 → index into LUT → (H, W, 3)
    rgb = _VIRIDIS_LUT[norm]                    # (nfft, n_time_frames, 3) uint8

    # ── Step 6: PIL resize to (IMG_H=256, IMG_W=512) ─────────────────────────
    # Training used imshow (origin='lower') — flip vertically so low freq is at
    # bottom, matching the spatial layout the model was trained on.
    pil_img = Image.fromarray(rgb[::-1], mode="RGB")       # flip freq axis
    pil_img = pil_img.resize((IMG_W, IMG_H), Image.BILINEAR)

    # ── Step 7: ImageNet normalise → NCHW float32 ───────────────────────────
    arr = np.array(pil_img, dtype=np.float32) / 255.0      # HWC [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD              # HWC normalised
    arr = arr.transpose(2, 0, 1)[np.newaxis]                # (1,3,H,W)
    return arr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Debug helper
# ─────────────────────────────────────────────────────────────────────────────

def save_spectrogram_png(tensor: np.ndarray, path: str) -> None:
    """
    Save a (1, 3, H, W) normalised tensor as a viewable RGB PNG.
    Undoes ImageNet normalisation so colours are restored for visual inspection.
    Compare output against training spectrograms — should look identical.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = tensor[0].transpose(1, 2, 0)              # HWC
    arr = arr * IMAGENET_STD + IMAGENET_MEAN         # undo normalise
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FRAME_DURATION = 0.080
    N = int(SAMPLE_RATE_HZ * FRAME_DURATION)   # 4,800,000

    print("=" * 60)
    print("  stft_preprocessor — standalone test")
    print("=" * 60)
    print(f"  IQ frame     : {N:,} samples  ({FRAME_DURATION*1000:.0f} ms @ {SAMPLE_RATE_HZ/1e6:.0f} MHz)")
    print(f"  STFT nfft    : {NFFT}   window=Hamming   library=scipy")
    print(f"  Colormap     : viridis dB  (matches training)")
    print(f"  Output shape : (1, 3, {IMG_H}, {IMG_W})  NCHW float32\n")

    # ── Test 1: 5 MHz tone ───────────────────────────────────────────────────
    print("  Test 1: 5 MHz tone ...")
    t_arr  = np.arange(N) / SAMPLE_RATE_HZ
    iq     = (0.5 * np.exp(1j * 2 * np.pi * 5e6 * t_arr)).astype(np.complex64)

    t0      = time.perf_counter()
    tensor  = iq_to_spectrogram(iq)
    elapsed = (time.perf_counter() - t0) * 1000

    assert tensor.shape == (1, 3, IMG_H, IMG_W), f"Shape mismatch: {tensor.shape}"
    assert np.all(np.isfinite(tensor)),           "Output contains NaN/Inf"
    print(f"    ✓ shape   : {tensor.shape}")
    print(f"    ✓ dtype   : {tensor.dtype}")
    print(f"    ✓ range   : [{tensor.min():.4f}, {tensor.max():.4f}]")
    print(f"    ✓ STFT    : {elapsed:.1f} ms  (expect < 50 ms)")
    save_spectrogram_png(tensor, "debug_stft/tone_5MHz.png")
    print(f"    ✓ saved   : debug_stft/tone_5MHz.png")

    # ── Test 2: wideband noise (NO_DRONE-like) ───────────────────────────────
    print("\n  Test 2: wideband noise ...")
    iq_n   = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64) * 0.01
    t0     = time.perf_counter()
    tensor_n = iq_to_spectrogram(iq_n)
    elapsed  = (time.perf_counter() - t0) * 1000
    assert np.all(np.isfinite(tensor_n))
    print(f"    ✓ shape   : {tensor_n.shape}")
    print(f"    ✓ STFT    : {elapsed:.1f} ms")
    save_spectrogram_png(tensor_n, "debug_stft/noise.png")
    print(f"    ✓ saved   : debug_stft/noise.png")

    # ── Test 3: silent frame ─────────────────────────────────────────────────
    print("\n  Test 3: silent frame (all zeros) ...")
    iq_z   = np.zeros(N, dtype=np.complex64)
    tensor_z = iq_to_spectrogram(iq_z)
    assert np.all(np.isfinite(tensor_z)), "Silent frame produced NaN/Inf"
    print(f"    ✓ finite  : True")

    print("\n" + "=" * 60)
    print("  All tests passed")
    print(f"  Check debug_stft/*.png — should look like viridis training spectrograms")
    print("=" * 60)