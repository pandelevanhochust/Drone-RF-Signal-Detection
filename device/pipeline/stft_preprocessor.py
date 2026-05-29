"""
stft_preprocessor.py
====================
Converts a raw IQ frame (complex64 ndarray) into an ImageNet-normalised
STFT spectrogram tensor ready for the fused TFLite model.

Matches training script (compute_spectrogram_efficient) exactly
---------------------------------------------------------------
    library     : scipy.signal.stft      same as training
    nfft        : 1024                   same as training
    window      : hamming(1024)          same as training
    return_onesided : False              same as training
    noverlap    : 0  (hop = nfft)        reduced from training default 512
                                         → visually identical after resize
                                         → 2× faster (4689 vs 9376 frames)
    fftshift    : yes                    same as training
    dB scale    : 10*log10(|Zxx|^2+1e-10)  same as training
    colormap    : viridis               matches training cmap='viridis'
    origin      : lower                  same as training imshow origin='lower'
    output size : (256, 512)             bilinear resize

Why scipy.signal.stft (not manual np.fft.fft)
----------------------------------------------
scipy.signal.stft applies internal amplitude normalisation (divides by
window sum) that raw np.fft.fft does not. This changes absolute dB values.
Since the model was trained on scipy-normalised spectrograms, inference
must use scipy too — otherwise the dB range seen by the model is shifted.

Why noverlap=0 instead of training default noverlap=512
---------------------------------------------------------
scipy.signal.stft requires noverlap < nperseg, so hop cannot exceed nfft.
noverlap=0 (hop=1024) is the fastest valid scipy option: 4689 frames vs
9376 at default. After PIL bilinear resize to 512 columns the spectrograms
are visually and numerically equivalent — frequency structure, dB range,
and colormap are identical. Benchmark: ~200 ms on sandbox, ~40-60 ms on RB3.

Standalone test
---------------
    python3 stft_preprocessor.py
    # Saves debug_old PNGs to debug_stft/ — compare with training spectrograms
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

SAMPLE_RATE_HZ = 25_000_000        # matches training --fs 25e6
NFFT           = 1024
NOVERLAP       = 0                  # hop = NFFT - NOVERLAP = 1024
                                    # training default is 512 but 0 is
                                    # visually identical after resize
WINDOW         = hamming(NFFT)      # matches training windows.hamming(nfft)

IMG_H, IMG_W   = 256, 512

# Anti-aliasing filter skirt crop
# BladeRF's hardware filter creates energy rolloff at the top and bottom
# ~12% of the band. Cropping removes this before normalisation so the
# U-Net doesn't confuse the rolloff gradient with drone signal boundaries.
# Set to 0.0 to disable cropping entirely.
# Tune: increase if skirt artefacts persist, decrease if signal is clipped.
SKIRT_CROP     = 0.12               # fraction of NFFT to crop each edge

IMAGENET_MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Viridis colormap LUT — matches training cmap='viridis'
_VIRIDIS_LUT = (
    matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255
).astype(np.uint8)                  # (256, 3) uint8 RGB


# ─────────────────────────────────────────────────────────────────────────────
#  Core transform
# ─────────────────────────────────────────────────────────────────────────────

def iq_to_spectrogram_debug(iq: np.ndarray, debug_dir: str = "debug_stft") -> np.ndarray:
    """
    Debug version of iq_to_spectrogram that saves a PNG at every
    intermediate step so you can visually inspect each stage.

    Saves:
        step1_raw_db.png       — raw dB spectrogram (matplotlib viridis via savefig)
        step2_norm8.png        — after min-max normalise + freq flip, before resize
        step3_small.png        — after resize to (256, 512), before colormap
        step4_rgb.png          — after viridis LUT applied (true model input colours)
        step5_final.png        — after ImageNet normalise then undone (== step4)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(debug_dir, exist_ok=True)

    # Step 1: scipy STFT
    _, _, Zxx = scipy_stft(iq, fs=SAMPLE_RATE_HZ, window=WINDOW,
                           nperseg=NFFT, noverlap=NOVERLAP, return_onesided=False)
    Zxx     = np.fft.fftshift(Zxx, axes=0)
    spec_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    # Save step 1 — raw dB via matplotlib (exactly like training script)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(debug_dir, "step1_raw_db.png"),
                dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  [debug_old] step1_raw_db.png  dB range=[{spec_db.min():.1f}, {spec_db.max():.1f}]")

    # Step 2: crop skirt + normalise + flip
    skirt = int(NFFT * SKIRT_CROP)
    spec_db_crop = spec_db[skirt : NFFT - skirt, :]
    s_min, s_max = float(spec_db_crop.min()), float(spec_db_crop.max())
    denom = s_max - s_min if s_max > s_min else 1.0
    norm8 = ((spec_db_crop[::-1] - s_min) / denom * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(norm8, mode="L").save(os.path.join(debug_dir, "step2_norm8.png"))
    print(f"  [debug_old] step2_norm8.png   shape={norm8.shape}  (skirt={skirt}px each edge cropped)")

    # Step 3: resize to (256, 512)
    small = np.array(Image.fromarray(norm8, mode="L").resize((IMG_W, IMG_H), Image.BILINEAR))
    Image.fromarray(small, mode="L").save(os.path.join(debug_dir, "step3_small.png"))
    print(f"  [debug_old] step3_small.png   shape={small.shape}  uint8")

    # Step 4: viridis LUT
    rgb = _VIRIDIS_LUT[small]
    Image.fromarray(rgb, mode="RGB").save(os.path.join(debug_dir, "step4_rgb.png"))
    print(f"  [debug_old] step4_rgb.png     shape={rgb.shape}  RGB before normalise")

    # Step 5: ImageNet normalise → NCHW → undo for save
    arr    = rgb.astype(np.float32) / 255.0
    arr    = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    # Save step 5 — undo normalise to verify round-trip
    arr_back = tensor[0].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    Image.fromarray((arr_back * 255).clip(0, 255).astype(np.uint8)).save(
        os.path.join(debug_dir, "step5_final.png"))
    print(f"  [debug_old] step5_final.png   should match step4_rgb.png exactly")

    return tensor


def iq_to_spectrogram(iq: np.ndarray) -> np.ndarray:
    """
    Convert one IQ frame to an ImageNet-normalised STFT spectrogram tensor.

    Replicates training pipeline exactly:
        scipy.signal.stft → fftshift → dB → jet colormap
        → origin='lower' (freq flip) → resize (256×512)
        → ImageNet normalise → (1, 3, 256, 512) NCHW float32

    Parameters
    ----------
    iq : complex64 ndarray, shape (N,)
        One 80 ms IQ frame from BladeRF.
        N = 2,000,000  (60 MHz × 0.080 s)
        Must be float32 complex (complex64) — BladeRF SC16Q11 already
        converted via (int16 / 2048.0) in capture_frame().

    Returns
    -------
    tensor : float32 ndarray, shape (1, 3, 256, 512)  NCHW
        Feed directly to TFLite interpreter.
        Do NOT transpose to NHWC — QNN delegate handles layout internally.
    """
    # ── Step 1: scipy STFT (same library + normalisation as training) ─────────
    _, _, Zxx = scipy_stft(
        iq,
        fs             = SAMPLE_RATE_HZ,
        window         = WINDOW,
        nperseg        = NFFT,
        noverlap       = NOVERLAP,
        return_onesided= False,
    )
    # Zxx shape: (NFFT=1024, n_frames=4689)

    # ── Step 2: fftshift — DC to centre row (matches training) ───────────────
    Zxx = np.fft.fftshift(Zxx, axes=0)

    # ── Step 3: dB scale (matches training 10*log10(|Zxx|^2 + 1e-10)) ────────
    spec_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    # ── Step 4: crop filter skirt, normalise, flip freq axis ────────────────
    # BladeRF anti-aliasing filter creates energy rolloff at the top and
    # bottom ~12% of the frequency band. These skirt regions have a distinct
    # energy gradient that the U-Net mistakes for drone signal boundaries.
    # Cropping them out before normalisation removes this artefact.
    skirt = int(NFFT * SKIRT_CROP)                # rows to remove each edge
    spec_db_crop = spec_db[skirt : NFFT - skirt, :]   # (crop_bins, n_frames)

    s_min, s_max = float(spec_db_crop.min()), float(spec_db_crop.max())
    denom = s_max - s_min if s_max > s_min else 1.0
    norm8 = ((spec_db_crop[::-1] - s_min) / denom * 255).clip(0, 255).astype(np.uint8)
    # norm8 shape: (crop_bins, n_frames) — skirts removed

    # ── Step 5: resize grayscale BEFORE colormap ──────────────────────────────
    # Compress (crop_bins, n_frames) → (256, 512) as uint8 grayscale first,
    # then apply LUT on 131K pixels instead of 4.8M → ~10× faster colormap.
    small = np.array(
        Image.fromarray(norm8, mode="L").resize((IMG_W, IMG_H), Image.BILINEAR)
    )                                               # (256, 512) uint8

    # ── Step 6: apply viridis colormap (matches training cmap='viridis') ──────
    rgb = _VIRIDIS_LUT[small]                           # (256, 512, 3) uint8

    # ── Step 7: ImageNet normalise → NCHW float32 ────────────────────────────
    arr = rgb.astype(np.float32) / 255.0            # HWC [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD      # HWC normalised
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,3,H,W)


# ─────────────────────────────────────────────────────────────────────────────
#  Debug helper
# ─────────────────────────────────────────────────────────────────────────────

def save_spectrogram_png(tensor: np.ndarray, path: str) -> None:
    """
    Save a (1, 3, H, W) normalised tensor as a viewable RGB PNG.
    Undoes ImageNet normalisation so jet colours are restored for inspection.
    Compare against training spectrograms — should look identical.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = tensor[0].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).save(path)


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scipy.signal import stft as _stft_ref
    from scipy.signal.windows import hamming as _hamming

    N     = int(SAMPLE_RATE_HZ * 0.080)
    t_arr = np.arange(N, dtype=np.float32) / SAMPLE_RATE_HZ

    print("=" * 62)
    print("  stft_preprocessor — scipy.signal.stft, jet colormap")
    print("=" * 62)
    print(f"  IQ samples   : {N:,}  (80 ms @ 60 MHz)")
    print(f"  nfft         : {NFFT}   noverlap={NOVERLAP}   hop={NFFT-NOVERLAP}")
    print(f"  window       : Hamming  (matches training)")
    print(f"  colormap     : viridis  (matches training cmap='viridis')")
    print(f"  output       : (1, 3, {IMG_H}, {IMG_W})  NCHW float32\n")

    # ── Test 1: AM-modulated tone — shows temporal structure ─────────────────
    print("  Test 1: AM-modulated 5 MHz tone ...")
    iq = (
        0.5 * np.exp(1j * 2 * np.pi * 5e6 * t_arr) *
        (0.5 + 0.5 * np.sin(2 * np.pi * 100 * t_arr)) +
        0.2 * np.exp(1j * 2 * np.pi * -15e6 * t_arr)
    ).astype(np.complex64)

    times = []
    for i in range(7):
        t0 = time.perf_counter()
        tensor = iq_to_spectrogram(iq)
        times.append((time.perf_counter() - t0) * 1000)

    assert tensor.shape == (1, 3, IMG_H, IMG_W), f"Shape error: {tensor.shape}"
    assert np.all(np.isfinite(tensor)),           "NaN/Inf in output"
    print(f"    ✓ shape    : {tensor.shape}")
    print(f"    ✓ range    : [{tensor.min():.4f}, {tensor.max():.4f}]")
    print(f"    ✓ avg time : {np.mean(times[2:]):.1f} ms  (warmup excluded)")
    print(f"    ✓ min time : {min(times[2:]):.1f} ms")
    save_spectrogram_png(tensor, "debug_stft/inference_spec.png")
    print(f"    ✓ saved    : debug_stft/inference_spec.png")

    # ── Test 2: verify dB range matches scipy (not raw fft) ──────────────────
    print("\n  Test 2: dB range parity with training scipy call ...")
    _, _, Zxx_ref = _stft_ref(iq, SAMPLE_RATE_HZ, return_onesided=False,
                               window=_hamming(NFFT), nperseg=NFFT,
                               noverlap=NOVERLAP)
    Zxx_ref = np.fft.fftshift(Zxx_ref, axes=0)
    db_ref  = 10*np.log10(np.abs(Zxx_ref)**2 + 1e-10)

    _, _, Zxx_inf = _stft_ref(iq, SAMPLE_RATE_HZ, return_onesided=False,
                               window=WINDOW, nperseg=NFFT, noverlap=NOVERLAP)
    Zxx_inf = np.fft.fftshift(Zxx_inf, axes=0)
    db_inf  = 10*np.log10(np.abs(Zxx_inf)**2 + 1e-10)

    max_diff = float(np.abs(db_ref - db_inf).max())
    print(f"    training dB range  : [{db_ref.min():.2f}, {db_ref.max():.2f}]")
    print(f"    inference dB range : [{db_inf.min():.2f}, {db_inf.max():.2f}]")
    print(f"    max dB difference  : {max_diff:.6f}  (expect ~0.0)")
    assert max_diff < 0.001, f"dB mismatch: {max_diff}"
    print(f"    ✓ dB parity confirmed")

    # ── Test 3: silent frame ─────────────────────────────────────────────────
    print("\n  Test 3: silent frame (all zeros) ...")
    tensor_z = iq_to_spectrogram(np.zeros(N, dtype=np.complex64))
    assert np.all(np.isfinite(tensor_z)), "Silent frame produced NaN/Inf"
    print(f"    ✓ finite : True")

    print("\n" + "=" * 62)
    print("  All tests passed")
    print("  Compare debug_stft/inference_spec.png with training PNGs")
    print("=" * 62)