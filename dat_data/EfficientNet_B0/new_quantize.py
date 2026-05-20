"""
qai_hub_submit.py
=================
Submits a pre-exported fused ONNX to Qualcomm AI Hub for:
  1. INT8 quantisation  (weights + activations, real calibration data)
  2. TFLite compilation (--target_runtime tflite --compute_unit npu)
  3. On-device profiling (latency + peak memory on RB3 Gen 2 NPU)
  4. Download of the compiled .tflite to disk

This file is intentionally decoupled from fused_pipeline.py.
Run fused_pipeline.py first to produce drone_pipeline_fused.onnx,
then run this script to submit it to Hub.

Calibration preprocessing
--------------------------
Images are preprocessed to exactly match drone_dataloader.get_transforms('val'):

    PIL RGB → resize (img_h, img_w) → / 255
           → subtract IMAGENET_MEAN → divide IMAGENET_STD

The calibration dict key 'spectrogram' must match the ONNX input name.

On-device inference (TFLite runtime)
--------------------------------------
    import numpy as np
    import tflite_runtime.interpreter as tflite

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    interp = tflite.Interpreter("drone_pipeline_fused_quantized.tflite")
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    img  = load_spectrogram_rgb_hwc()          # (256, 512, 3) uint8
    arr  = img.astype(np.float32) / 255.0
    arr  = (arr - MEAN) / STD
    arr  = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 256, 512) NCHW

    interp.set_tensor(inp_idx, arr)
    interp.invoke()
    logits = interp.get_tensor(out_idx)        # (1, num_classes)

    probs = np.exp(logits - logits.max()); probs /= probs.sum()
    with open("class_names.txt") as f:
        names = [l.strip() for l in f]
    print(names[probs.argmax()], f"{probs.max():.1%}")

Note: feed NCHW tensors. QNN/TFLite on RB3 handles the layout swap
at the first op — do NOT transpose to NHWC before feeding.

Usage
-----
  python qai_hub_submit.py \\
      --onnx  exports/drone_pipeline_fused.onnx \\
      --cal_dir /path/to/raw_spectrograms

  # Dry run — build calibration data and stop before Hub submission
  python qai_hub_submit.py \\
      --onnx exports/drone_pipeline_fused.onnx \\
      --cal_dir /path/to/raw_spectrograms \\
      --dry_run

Requirements
------------
  pip install qai-hub numpy pillow
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ImageNet normalisation — must match drone_dataloader.get_transforms()
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration data
# ─────────────────────────────────────────────────────────────────────────────

def build_calibration_data(
    cal_dir    : str,
    img_h      : int = 256,
    img_w      : int = 512,
    max_images : int = 100,
) -> dict:
    """
    Scan cal_dir recursively for PNG/JPG images and preprocess each one
    with the same transform chain used during training:

        PIL RGB → resize (img_h, img_w) → / 255
               → subtract IMAGENET_MEAN → divide IMAGENET_STD

    The calibration dict key 'spectrogram' matches the ONNX input name
    defined in fused_pipeline.export_fused_onnx().

    Why preprocessing must match training exactly
    ---------------------------------------------
    The quantiser uses calibration data to compute the min/max activation
    range for each INT8 quantisation bin. If the preprocessing differs
    from training (e.g. only / 255, no mean/std normalisation), the
    inferred ranges are shifted ~0.45 from the model's true operational
    range. Under INT8 this directly maps to fewer usable quantisation
    levels, degrading accuracy — especially on NO_DRONE frames whose
    energy sits near zero before normalisation.

    Parameters
    ----------
    cal_dir    : Root directory to scan (recursive). Accepts mixed subsets.
    img_h/w    : Must match --img_h/--img_w used during training.
    max_images : Cap to avoid excessive Hub calibration cost (default 100).

    Returns
    -------
    {"spectrogram": [ndarray(1,3,H,W), ...]}
    """
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(cal_dir, "**", ext), recursive=True))

    if not files:
        raise FileNotFoundError(
            f"No images found under '{cal_dir}'.\n"
            "Pass --cal_dir pointing at your raw spectrogram folder."
        )

    files   = sorted(files)[:max_images]
    print(f"[Calibration] {len(files)} images found (cap={max_images})")

    arrays, skipped = [], 0
    for path in files:
        try:
            # PIL.resize takes (W, H); spectrogram is (H=256, W=512)
            img = Image.open(path).convert("RGB").resize(
                (img_w, img_h), Image.BILINEAR
            )
            arr = np.array(img, dtype=np.float32) / 255.0     # HWC [0,1]
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD         # HWC normalised
            arr = arr.transpose(2, 0, 1)                       # CHW
            arr = arr[np.newaxis]                              # (1,3,H,W)
            arrays.append(arr)
        except Exception as exc:
            print(f"  [Calibration] Skipping {Path(path).name}: {exc}")
            skipped += 1

    if not arrays:
        raise ValueError(
            "All calibration images failed to load. "
            "Check --cal_dir and image integrity."
        )

    print(f"  ✓ {len(arrays)} frames loaded  "
          f"({skipped} skipped)  shape={arrays[0].shape}"
          f"  range=[{arrays[0].min():.3f}, {arrays[0].max():.3f}]")
    return {"spectrogram": arrays}


# ─────────────────────────────────────────────────────────────────────────────
#  Qualcomm AI Hub pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_qai_hub_pipeline(
    onnx_path  : str,
    cal_data   : dict,
    out_dir    : str = "exports",
    img_h      : int = 256,
    img_w      : int = 512,
    device_name: str = "Dragonwing RB3 Gen 2 Vision Kit",
) -> tuple:
    """
    Full Hub flow:  upload → quantise → compile → profile → download.

    Step 1 — Upload
        The ONNX and its optional .data sidecar are copied to a clean
        staging directory before upload. This prevents Hub from seeing
        stale files from a previous export in the same folder.

    Step 2 — Quantise (INT8)
        FP32 weights and activations are quantised to INT8 using the
        provided calibration data. The entire fused graph is quantised
        in one job: U-Net, mask binarisation, ROI multiply+resize, and
        EfficientNet all calibrated in their correct operational context.

    Step 3 — Compile (TFLite / NPU)
        The quantised ONNX is compiled to TFLite targeting the Hexagon
        NPU on the RB3 Gen 2 with --compute_unit npu.

    Step 4 — Download
        The compiled model is saved to out_dir as
        drone_pipeline_fused_quantized.tflite.

    Step 5 — Profile
        An on-device profile job estimates NPU latency and peak memory.

    Parameters
    ----------
    onnx_path   : Path to the fused ONNX from fused_pipeline.export_fused_onnx().
    cal_data    : Dict from build_calibration_data().
    out_dir     : Local directory to save the downloaded .tflite.
    img_h/w     : Must match the ONNX input shape.
    device_name : Qualcomm Hub device string.

    Returns
    -------
    (tflite_path, latency_ms, peak_memory_mb)
    """
    try:
        import qai_hub as hub
    except ImportError:
        raise ImportError(
            "qai-hub not installed.\n"
            "  pip install qai-hub\n"
            "  qai-hub configure   # enter your API token"
        )

    os.makedirs(out_dir, exist_ok=True)
    device      = hub.Device(device_name)
    input_shape = (1, 3, img_h, img_w)
    sep         = "─" * 55

    # ── Step 1: Upload ────────────────────────────────────────────────────────
    onnx_name  = Path(onnx_path).name
    data_name  = onnx_name + ".data"
    upload_dir = os.path.join(out_dir, "_hub_upload")

    shutil.rmtree(upload_dir, ignore_errors=True)
    os.makedirs(upload_dir)
    shutil.copy2(onnx_path, os.path.join(upload_dir, onnx_name))

    data_src = str(Path(onnx_path).parent / data_name)
    has_sidecar = os.path.exists(data_src)
    if has_sidecar:
        shutil.copy2(data_src, os.path.join(upload_dir, data_name))
        print(f"\n[Hub  1/5] Uploading ONNX + .data sidecar ...")
    else:
        print(f"\n[Hub  1/5] Uploading ONNX (no .data sidecar) ...")

    source_model = hub.upload_model(
        model = upload_dir,
        name  = "DroneDetectionFusedPipeline",
    )
    print(f"  ✓ Uploaded → {source_model}")

    # ── Step 2: Quantise ──────────────────────────────────────────────────────
    n_cal = len(cal_data["spectrogram"])
    print(f"\n[Hub  2/5] Quantising FP32 → INT8  ({n_cal} calibration frames) ...")
    quantise_job = hub.submit_quantize_job(
        model             = source_model,
        calibration_data  = cal_data,
        weights_dtype     = hub.QuantizeDtype.INT8,
        activations_dtype = hub.QuantizeDtype.INT8,
    )
    quantised_model = quantise_job.get_target_model()
    print(f"  ✓ Quantised model → {quantised_model}")

    # ── Step 3: Compile ───────────────────────────────────────────────────────
    print(f"\n[Hub  3/5] Compiling → TFLite / NPU / {device_name} ...")
    compile_job = hub.submit_compile_job(
        model       = quantised_model,
        device      = device,
        input_specs = {"spectrogram": input_shape},
        options     = "--target_runtime tflite --compute_unit npu",
    )
    compiled_model = compile_job.get_target_model()
    print(f"  ✓ Compiled model → {compiled_model}")

    # ── Step 4: Download ──────────────────────────────────────────────────────
    tflite_path = os.path.join(out_dir, "drone_pipeline_fused_quantized.tflite")
    print(f"\n[Hub  4/5] Downloading compiled TFLite ...")
    compiled_model.download(tflite_path)
    tflite_mb = Path(tflite_path).stat().st_size / 1e6
    print(f"  ✓ Saved → {tflite_path}  ({tflite_mb:.1f} MB)")

    # ── Step 5: Profile ───────────────────────────────────────────────────────
    print(f"\n[Hub  5/5] Profiling on {device_name} NPU ...")
    profile_job = hub.submit_profile_job(
        model   = compiled_model,
        device  = device,
        options = "--compute_unit npu",
    )
    profile    = profile_job.download_profile()
    summary    = profile["execution_summary"]
    latency_ms = summary["estimated_inference_time"]
    peak_mb    = summary["estimated_inference_peak_memory"] / (1024 * 1024)

    print(f"\n{sep}")
    print(f"  Performance — {device_name}  (fused pipeline, NPU INT8)")
    print(f"  Input        : spectrogram  {input_shape}")
    print(f"  Output       : class_logits (1, num_classes)")
    print(f"  Avg latency  : {latency_ms:.2f} ms")
    print(f"  Peak memory  : {peak_mb:.2f} MB")
    print(sep)

    return str(Path(tflite_path).resolve()), latency_ms, peak_mb


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Submit fused ONNX to Qualcomm AI Hub: quantise → compile → profile"
    )
    p.add_argument("--onnx",       required=True,
                   help="Path to drone_pipeline_fused.onnx from fused_pipeline.py")
    p.add_argument("--cal_dir",    required=True,
                   help="Directory of calibration spectrograms (recursive scan)")
    p.add_argument("--out_dir",    default="exports",
                   help="Output directory for the downloaded .tflite")
    p.add_argument("--img_h",      type=int, default=256,
                   help="Spectrogram height used at export time")
    p.add_argument("--img_w",      type=int, default=512,
                   help="Spectrogram width  used at export time")
    p.add_argument("--cal_images", type=int, default=100,
                   help="Max calibration images (default 100)")
    p.add_argument("--device",     default="Dragonwing RB3 Gen 2 Vision Kit",
                   help="Qualcomm Hub device name")
    p.add_argument("--dry_run",    action="store_true",
                   help="Build calibration data and print summary, skip Hub submission")
    return p.parse_args()


def main():
    args = get_args()

    sep = "=" * 55
    print(f"\n{sep}")
    print("  Qualcomm AI Hub — Fused Pipeline Submission")
    print(sep)
    print(f"  ONNX       : {args.onnx}")
    print(f"  Cal dir    : {args.cal_dir}")
    print(f"  Output dir : {args.out_dir}")
    print(f"  Device     : {args.device}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"{sep}\n")

    cal_data = build_calibration_data(
        cal_dir    = args.cal_dir,
        img_h      = args.img_h,
        img_w      = args.img_w,
        max_images = args.cal_images,
    )

    if args.dry_run:
        print(f"\n[Dry run] Calibration data ready — skipping Hub submission.")
        print(f"  Frames   : {len(cal_data['spectrogram'])}")
        print(f"  Shape    : {cal_data['spectrogram'][0].shape}")
        print(f"  Range    : [{cal_data['spectrogram'][0].min():.3f}, "
              f"{cal_data['spectrogram'][0].max():.3f}]")
        return

    tflite_path, latency, peak_mb = run_qai_hub_pipeline(
        onnx_path   = args.onnx,
        cal_data    = cal_data,
        out_dir     = args.out_dir,
        img_h       = args.img_h,
        img_w       = args.img_w,
        device_name = args.device,
    )

    print(f"\n{sep}")
    print("  Done")
    print(f"  TFLite  : {tflite_path}")
    print(f"  Latency : {latency:.2f} ms")
    print(f"  Memory  : {peak_mb:.2f} MB")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()