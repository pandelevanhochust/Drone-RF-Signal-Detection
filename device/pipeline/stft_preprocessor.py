"""
stft_preprocessor.py
====================
Converts a raw IQ frame (complex64 ndarray) into an ImageNet-normalised
STFT spectrogram tensor ready for the fused TFLite model.

Responsibilities
----------------
- Compute STFT (Hann window, nfft=256, hop=128, fftshift DC to centre)
- Convert to log-power dB scale
- Normalise to [0, 1], resize to (256, 512) via PIL bilinear
- Expand to 3-channel RGB, apply ImageNet mean/std normalisation
- Return (1, 3, 256, 512) NCHW float32 tensor — model input format

This module has NO dependency on BladeRF, TFLite, or AI inference.
It only depends on: numpy, Pillow.

STFT parameters (must match training data collection)
-----------------------------------------------------
    Window size (nfft) : 256 samples
    Hop size           : 128 samples  (50% overlap)
    Window function    : Hann
    Frequency axis     : fftshifted (DC at centre row)
    Power              : 10 * log10(|X|^2 + 1e-12)  [dB]
    Output size        : (256, 512) after bilinear resize

ImageNet normalisation (must match drone_dataloader.get_transforms)
-------------------------------------------------------------------
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

Standalone test
---------------
    python3 stft_preprocessor.py
    # Generates a synthetic IQ frame, runs the full transform,
    # prints output shape/range, and saves a debug PNG.
"""

import os

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  STFT / output constants
# ─────────────────────────────────────────────────────────────────────────────

NFFT          = 256                              # FFT window size
HOP           = 128                              # hop → 50% overlap
WINDOW        = np.hanning(NFFT).astype(np.float32)

IMG_H, IMG_W  = 256, 512                         # model input spatial dims

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Core transform
# ─────────────────────────────────────────────────────────────────────────────

def iq_to_spectrogram(iq: np.ndarray) -> np.ndarray:
    """
    Convert one IQ frame to an ImageNet-normalised STFT spectrogram tensor.

    Parameters
    ----------
    iq : complex64 ndarray, shape (N,)
        One 80 ms IQ frame from BladeRF, values in approx [-1, 1].
        N = 4,800,000 for 60 MHz sample rate × 80 ms.

    Returns
    -------
    tensor : float32 ndarray, shape (1, 3, 256, 512)  NCHW
        ImageNet-normalised. Feed directly to TFLite interpreter as-is.
        Do NOT transpose to NHWC — the QNN delegate handles layout internally.

    Processing steps
    ----------------
    1. Segment IQ into overlapping windows of NFFT=256, hop=128.
       For N=4,800,000: n_frames = (4,800,000 − 256) // 128 + 1 = 37,497
    2. Apply Hann window element-wise to each segment.
    3. FFT each segment along the frequency axis.
    4. fftshift: rotate so DC component lands at the centre frequency row.
       This matches the training spectrogram format.
    5. Power spectrum: 10 * log10(|X|^2 + ε)  [dB]
       ε = 1e-12 avoids log(0) for silent / noise-floor frames.
    6. Transpose to (freq_bins=256, time_steps=37,497) — freq on Y axis.
    7. Min-max normalise to [0, 1] within this frame.
       Silent frames (p_max == p_min) become all-zero (NO_DRONE path).
    8. Convert to uint8, resize to (IMG_W=512, IMG_H=256) via PIL bilinear.
       PIL.resize takes (W, H) order.
    9. Convert grayscale L → RGB (3 identical channels).
   10. float32 / 255 → subtract IMAGENET_MEAN → divide IMAGENET_STD (HWC).
   11. Transpose HWC → CHW, add batch dim → (1, 3, 256, 512).
    """
    n        = len(iq)
    n_frames = (n - NFFT) // HOP + 1

    # Step 1-2: windowed segments — shape (n_frames, NFFT)
    idx      = np.arange(NFFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames   = iq[idx] * WINDOW[None, :]

    # Step 3-4: FFT + fftshift — shape (n_frames, NFFT)
    spectrum = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)

    # Step 5-6: log power, transpose → (freq_bins, time_steps)
    power    = 10.0 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    power    = power.T                                      # (NFFT, n_frames)

    # Step 7: min-max normalise to [0, 1]
    p_min, p_max = float(power.min()), float(power.max())
    if p_max > p_min:
        power = (power - p_min) / (p_max - p_min)
    else:
        power = np.zeros_like(power)

    # Step 8: PIL bilinear resize to (IMG_H, IMG_W)
    uint8    = (power * 255).clip(0, 255).astype(np.uint8)
    pil_img  = Image.fromarray(uint8, mode="L")             # grayscale (H, W)
    pil_img  = pil_img.resize((IMG_W, IMG_H), Image.BILINEAR)

    # Step 9-11: L → RGB → normalise → NCHW
    arr      = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    arr      = (arr - IMAGENET_MEAN) / IMAGENET_STD          # HWC normalised
    arr      = arr.transpose(2, 0, 1)[np.newaxis]            # (1, 3, H, W)
    return arr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Debug helper — save a human-readable PNG (undoes normalisation)
# ─────────────────────────────────────────────────────────────────────────────

def save_spectrogram_png(tensor: np.ndarray, path: str) -> None:
    """
    Save a normalised (1, 3, H, W) tensor as a viewable RGB PNG.

    Undoes ImageNet normalisation so the pixel values are back in [0, 255].
    Useful for visually comparing live spectrograms against training data.

    Parameters
    ----------
    tensor : float32 ndarray (1, 3, H, W) — output of iq_to_spectrogram()
    path   : destination file path (e.g. "debug/frame_000001.png")
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = tensor[0].transpose(1, 2, 0)                  # (H, W, 3)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN             # undo normalise
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    SAMPLE_RATE_HZ = 60_000_000
    FRAME_DURATION = 0.080
    N              = int(SAMPLE_RATE_HZ * FRAME_DURATION)    # 4,800,000

    print("=" * 55)
    print("  stft_preprocessor — standalone test")
    print("=" * 55)
    print(f"  IQ frame     : {N:,} samples  ({FRAME_DURATION*1000:.0f} ms @ {SAMPLE_RATE_HZ/1e6:.0f} MHz)")
    print(f"  STFT nfft    : {NFFT}   hop={HOP}   window=Hann")
    print(f"  Output shape : (1, 3, {IMG_H}, {IMG_W})  NCHW float32\n")

    # ── Test 1: synthetic tone at 5 MHz offset ────────────────────────────────
    print("  Test 1: 5 MHz tone ...")
    t   = np.arange(N) / SAMPLE_RATE_HZ
    iq  = (0.5 * np.exp(1j * 2 * np.pi * 5e6 * t)).astype(np.complex64)

    t0      = time.perf_counter()
    tensor  = iq_to_spectrogram(iq)
    elapsed = (time.perf_counter() - t0) * 1000

    assert tensor.shape == (1, 3, IMG_H, IMG_W), \
        f"Expected (1,3,{IMG_H},{IMG_W}), got {tensor.shape}"
    assert np.all(np.isfinite(tensor)), "Output contains NaN or Inf"
    print(f"    ✓ shape  : {tensor.shape}")
    print(f"    ✓ dtype  : {tensor.dtype}")
    print(f"    ✓ range  : [{tensor.min():.4f}, {tensor.max():.4f}]")
    print(f"    ✓ time   : {elapsed:.1f} ms")
    save_spectrogram_png(tensor, "debug_stft/tone_5MHz.png")
    print(f"    ✓ saved  : debug_stft/tone_5MHz.png")

    # ── Test 2: pure noise (NO_DRONE-like) ────────────────────────────────────
    print("\n  Test 2: noise only (NO_DRONE-like) ...")
    iq_noise = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64) * 0.01
    tensor_n = iq_to_spectrogram(iq_noise)
    assert tensor_n.shape == (1, 3, IMG_H, IMG_W)
    assert np.all(np.isfinite(tensor_n))
    print(f"    ✓ shape  : {tensor_n.shape}")
    print(f"    ✓ range  : [{tensor_n.min():.4f}, {tensor_n.max():.4f}]")
    save_spectrogram_png(tensor_n, "debug_stft/noise.png")
    print(f"    ✓ saved  : debug_stft/noise.png")

    # ── Test 3: silent frame (all zeros) ─────────────────────────────────────
    print("\n  Test 3: silent frame (all zeros) ...")
    iq_zero  = np.zeros(N, dtype=np.complex64)
    tensor_z = iq_to_spectrogram(iq_zero)
    assert tensor_z.shape == (1, 3, IMG_H, IMG_W)
    assert np.all(np.isfinite(tensor_z)), "Silent frame produced NaN/Inf"
    print(f"    ✓ shape  : {tensor_z.shape}")
    print(f"    ✓ finite : True  (silent frame handled correctly)")

    print("\n" + "=" * 55)
    print("  All tests passed")
    print("=" * 55)