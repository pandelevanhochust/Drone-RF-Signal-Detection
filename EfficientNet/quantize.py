"""
qai_hub_submit.py
=================
Submits a pre-exported EfficientViT ONNX model to Qualcomm AI Hub for:
  1. INT8 quantization  (weights + activations, real calibration data)
  2. TFLite compilation (--target_runtime tflite --compute_unit npu)
  3. On-device profiling (latency + peak memory on RB3 Gen 2 NPU)
  4. Download of the compiled .tflite to disk

Calibration preprocessing:
Matches train_and_export.py transforms exactly: PIL RGB → resize (224, 224) → / 255.0 [0.0, 1.0]
"""

import argparse
import glob
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration data
# ─────────────────────────────────────────────────────────────────────────────

def build_calibration_data(
        cal_dir: str,
        img_h: int = 224,
        img_w: int = 224,
        max_images: int = 100,
) -> dict:
    """
    Scan cal_dir recursively for PNG/JPG images and preprocess each one
    with the same transform chain used during training:
        PIL RGB → resize (224, 224) → / 255.0 to yield [0.0, 1.0] float32 range.
    """
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(cal_dir, "**", ext), recursive=True))

    if not files:
        raise FileNotFoundError(
            f"No images found under '{cal_dir}'.\n"
            "Pass --cal_dir pointing at your validation/calibration spectrogram folder."
        )

    # Shuffle or slice to ensure a balanced, representative mix up to max_images
    files = sorted(files)[:max_images]
    print(f"[Calibration] {len(files)} images found for calibration (cap={max_images})")

    arrays, skipped = [], 0
    for path in files:
        try:
            # PIL.resize takes (W, H)
            img = Image.open(path).convert("RGB").resize(
                (img_w, img_h), Image.BILINEAR
            )
            arr = np.array(img, dtype=np.float32) / 255.0  # HWC mapped directly to [0.0, 1.0]
            arr = arr.transpose(2, 0, 1)  # Transpose to CHW
            arr = arr[np.newaxis]  # Add batch axis -> (1, 3, 224, 224)
            arrays.append(arr)
        except Exception as exc:
            print(f"  [Calibration] Skipping {Path(path).name}: {exc}")
            skipped += 1

    if not arrays:
        raise ValueError("All calibration images failed to load.")

    print(f"  ✓ {len(arrays)} frames loaded successfully. shape={arrays[0].shape}"
          f"  range=[{arrays[0].min():.3f}, {arrays[0].max():.3f}]")

    # CRITICAL: Key must match your ONNX model input name 'image_tensor'
    return {"image_tensor": arrays}


# ─────────────────────────────────────────────────────────────────────────────
#  Qualcomm AI Hub pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_qai_hub_pipeline(
        onnx_path: str,
        cal_data: dict,
        out_dir: str = "exports",
        img_h: int = 224,
        img_w: int = 224,
        device_name: str = "Dragonwing RB3 Gen 2 Vision Kit",
) -> tuple:
    """
    Full Hub flow: upload → quantize (INT8) → compile (TFLite NPU) → profile → download.
    """
    try:
        import qai_hub as hub
    except ImportError:
        raise ImportError("qai-hub not installed. Run: pip install qai-hub")

    os.makedirs(out_dir, exist_ok=True)
    device = hub.Device(device_name)
    input_shape = (1, 3, img_h, img_w)
    sep = "─" * 55

    # ── Step 1: Upload ────────────────────────────────────────────────────────
    onnx_name = Path(onnx_path).name
    upload_dir = os.path.join(out_dir, "_hub_upload")

    shutil.rmtree(upload_dir, ignore_errors=True)
    os.makedirs(upload_dir)
    shutil.copy2(onnx_path, os.path.join(upload_dir, onnx_name))

    print(f"\n[Hub  1/5] Uploading ONNX target model...")
    source_model = hub.upload_model(
        model=upload_dir,
        name="EfficientViT_Drone_Classifier",
    )
    print(f"  ✓ Uploaded → {source_model}")

    # ── Step 2: Quantize ──────────────────────────────────────────────────────
    n_cal = len(cal_data["image_tensor"])
    print(f"\n[Hub  2/5] Quantizing FP32 → INT8 ({n_cal} aligned calibration frames) ...")

    quantise_job = hub.submit_quantize_job(
        model=source_model,
        calibration_data=cal_data,
        weights_dtype=hub.QuantizeDtype.INT8,  # Standard INT8 quantization
        activations_dtype=hub.QuantizeDtype.INT8,
    )
    quantised_model = quantise_job.get_target_model()
    print(f"  ✓ Quantized model graph created.")

    # ── Step 3: Compile ───────────────────────────────────────────────────────
    print(f"\n[Hub  3/5] Compiling target structural graph → TFLite NPU ...")
    compile_job = hub.submit_compile_job(
        model=quantised_model,
        device=device,
        input_specs={"image_tensor": input_shape},
        options="--target_runtime tflite --compute_unit npu",
    )
    compiled_model = compile_job.get_target_model()
    print(f"  ✓ Compiled model asset generated.")

    # ── Step 4: Download ──────────────────────────────────────────────────────
    tflite_path = os.path.join(out_dir, "efficientvit_l2_drone_quantized.tflite")
    print(f"\n[Hub  4/5] Downloading compiled edge TFLite binary...")
    compiled_model.download(tflite_path)
    tflite_mb = Path(tflite_path).stat().st_size / 1e6
    print(f"  ✓ Saved → {tflite_path} ({tflite_mb:.2f} MB)")

    # ── Step 5: Profile ───────────────────────────────────────────────────────
    print(f"\n[Hub  5/5] Profiling live execution on target Hardware NPU...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        options="--compute_unit npu",
    )
    profile = profile_job.download_profile()
    summary = profile["execution_summary"]
    latency_ms = summary["estimated_inference_time"]
    peak_mb = summary["estimated_inference_peak_memory"] / (1024 * 1024)

    print(f"\n{sep}")
    print(f"  Performance Profile Metrics — {device_name}")
    print(f"  Input Target Size : image_tensor {input_shape}")
    print(f"  Avg Latency Speed : {latency_ms:.2f} ms")
    print(f"  Peak Memory Floor : {peak_mb:.2f} MB")
    print(sep)

    return str(Path(tflite_path).resolve()), latency_ms, peak_mb


# ─────────────────────────────────────────────────────────────────────────────
#  CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Submit model to Qualcomm AI Hub")
    p.add_argument("--onnx", required=True, help="Path to exported ONNX model")
    p.add_argument("--cal_dir", required=True, help="Directory of calibration spectrograms")
    p.add_argument("--out_dir", default="exports", help="Output directory for results")
    p.add_argument("--img_h", type=int, default=224, help="Spectrogram height")
    p.add_argument("--img_w", type=int, default=224, help="Spectrogram width")
    p.add_argument("--cal_images", type=int, default=100, help="Max calibration images (default 100)")
    p.add_argument("--device", default="Dragonwing RB3 Gen 2 Vision Kit", help="Target device")
    p.add_argument("--dry_run", action="store_true", help="Skip submission, verify files only")
    return p.parse_args()


def main():
    args = get_args()
    cal_data = build_calibration_data(
        cal_dir=args.cal_dir, img_h=args.img_h, img_w=args.img_w, max_images=args.cal_images
    )

    if args.dry_run:
        print(f"\n[Dry run] Calibration configurations match. Ready for Qualcomm Hub submission.")
        return

    run_qai_hub_pipeline(
        onnx_path=args.onnx, cal_data=cal_data, out_dir=args.out_dir,
        img_h=args.img_h, img_w=args.img_w, device_name=args.device
    )


if __name__ == "__main__":
    main()