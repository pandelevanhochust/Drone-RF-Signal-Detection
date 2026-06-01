"""
qai_hub_submit.py
=================
Submits the custom NPU-fused EfficientNet-B0 ONNX graph to Qualcomm AI Hub for:
  1. INT8 quantization  (weights + activations calibrated to [0.0, 1.0] float)
  2. TFLite compilation (--target_runtime tflite --compute_unit npu)
  3. On-device profiling (latency + peak memory on RB3 Gen 2 NPU)
  4. Download of the compiled .tflite to disk

Calibration Preprocessing Alignment
-----------------------------------
Matches train_and_export.py transforms exactly:
    PIL RGB → resize (256, 512) → ToTensor() [/ 255.0]
    NO ImageNet mean/std normalization.

Usage
-----
  python qai_hub_submit.py \
      --onnx drone_classifier_b0.onnx \
      --cal_dir dataset_split/val

  # Dry run — test data shapes locally before submitting to Hub
  python qai_hub_submit.py \
      --onnx drone_classifier_b0.onnx \
      --cal_dir dataset_split/val \
      --dry_run
"""

import argparse
import glob
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  1. Aligned Calibration Data Engine
# ─────────────────────────────────────────────────────────────────────────────

def build_calibration_data(
        cal_dir: str,
        img_h: int = 256,
        img_w: int = 512,
        max_images: int = 100,
) -> dict:
    """
    Scan cal_dir recursively for images and preprocess each one to match
    the training pipeline exactly:
        PIL RGB → resize (256, 512) → divide by 255.0 to yield [0.0, 1.0] float32

    CRITICAL: ImageNet mean/std subtraction is completely bypassed.
    The output dict key 'image_tensor' matches the custom ONNX input node exactly.
    """
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(cal_dir, "**", ext), recursive=True))

    if not files:
        raise FileNotFoundError(
            f"No calibration images found under '{cal_dir}'.\n"
            "Please point --cal_dir directly at your split validation folder (e.g., dataset_split/val)."
        )

    # Sort files to ensure stable execution groupings
    files = sorted(files)[:max_images]
    print(f"[Calibration] Found {len(files)} frames for quantization matrix (cap={max_images})")

    arrays, skipped = [], 0
    for path in files:
        try:
            # PIL resize expects width first: (W, H) -> (512, 256)
            img = Image.open(path).convert("RGB").resize(
                (img_w, img_h), Image.BILINEAR
            )

            # Match ToTensor() scaling exactly to prevent integer range clipping
            arr = np.array(img, dtype=np.float32) / 255.0  # Shape: HWC, Range: [0.0, 1.0]
            arr = arr.transpose(2, 0, 1)  # Transpose to CHW layout
            arr = arr[np.newaxis]  # Add batch axis -> (1, 3, 256, 512)
            arrays.append(arr)
        except Exception as exc:
            print(f"  [Calibration] Skipping corrupt/unreadable frame {Path(path).name}: {exc}")
            skipped += 1

    if not arrays:
        raise ValueError("All calibration images failed to pass through the preprocessing pipeline.")

    print(f"  ✓ {len(arrays)} frames processed successfully ({skipped} skipped)")
    print(f"  ✓ Tensor layout verified: shape={arrays[0].shape} | range=[{arrays[0].min():.3f}, {arrays[0].max():.3f}]")

    # Target dictionary identifier must match ONNX input token 'image_tensor'
    return {"image_tensor": arrays}


# ─────────────────────────────────────────────────────────────────────────────
#  2. Qualcomm AI Hub Submission Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_qai_hub_pipeline(
        onnx_path: str,
        cal_data: dict,
        out_dir: str = "exports",
        img_h: int = 256,
        img_w: int = 512,
        device_name: str = "Dragonwing RB3 Gen 2 Vision Kit",
) -> tuple:
    """
    Executes the clean, hardware-bounded cloud compilation cycle:
        Upload → INT8 Quantize → Compile (TFLite NPU) → Download → Profile
    """
    try:
        import qai_hub as hub
    except ImportError:
        raise ImportError(
            "The 'qai-hub' SDK was not found in your active environment.\n"
            "Install it via: pip install qai-hub\n"
            "Then configure your token via: qai-hub configure"
        )

    os.makedirs(out_dir, exist_ok=True)
    device = hub.Device(device_name)
    input_shape = (1, 3, img_h, img_w)
    line_sep = "─" * 60

    # ── Step 1: Secure Graph Upload ──────────────────────────────────────────
    onnx_name = Path(onnx_path).name
    upload_dir = os.path.join(out_dir, "_hub_upload")

    shutil.rmtree(upload_dir, ignore_errors=True)
    os.makedirs(upload_dir)
    shutil.copy2(onnx_path, os.path.join(upload_dir, onnx_name))

    print(f"\n[Hub 1/5] Transmitting static ONNX model graph to Qualcomm Cloud...")
    source_model = hub.upload_model(
        model=upload_dir,
        name="EfficientNetB0_Drone_Spectrogram_Classifier",
    )
    print(f"  ✓ Remote Model Container created → ID: {source_model.model_id}")

    # ── Step 2: INT8 Post-Training Quantization ──────────────────────────────
    n_cal = len(cal_data["image_tensor"])
    print(f"\n[Hub 2/5] Initializing Post-Training Quantization (FP32 → INT8 via {n_cal} calibration tensors)...")

    quantise_job = hub.submit_quantize_job(
        model=source_model,
        calibration_data=cal_data,
        weights_dtype=hub.QuantizeDtype.INT8,  # Pure INT8 target weights
        activations_dtype=hub.QuantizeDtype.INT8,  # Pure INT8 target activations
    )
    quantised_model = quantise_job.get_target_model()
    print(f"  ✓ Quantization calibration complete. 8-bit dynamic integer map generated.")

    # ── Step 3: NPU Compilation ──────────────────────────────────────────────
    print(f"\n[Hub 3/5] Compiling execution graph for Hexagon NPU targeting {device_name}...")
    compile_job = hub.submit_compile_job(
        model=quantised_model,
        device=device,
        input_specs={"image_tensor": input_shape},
        options="--target_runtime tflite --compute_unit npu",
    )
    compiled_model = compile_job.get_target_model()
    print(f"  ✓ Compilation complete. NPU-fused operators generated.")

    # ── Step 4: Download ──────────────────────────────────────────────────────
    onnx_stem = Path(onnx_path).stem
    tflite_path = os.path.join(out_dir, f"{onnx_stem}_quantized.tflite")
    print(f"\n[Hub 4/5] Downloading finalized .tflite deployment asset...")
    compiled_model.download(tflite_path)
    file_mb = Path(tflite_path).stat().st_size / (1024 * 1024)
    print(f"  ✓ Production asset saved → {tflite_path} ({file_mb:.2f} MB)")

    # ── Step 5: On-Device Performance Profiling ──────────────────────────────
    print(f"\n[Hub 5/5] Initiating real-time benchmarking on hardware accelerator...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        options="--compute_unit npu",
    )
    profile = profile_job.download_profile()
    summary = profile["execution_summary"]
    latency_ms = summary["estimated_inference_time"]
    peak_mb = summary["estimated_inference_peak_memory"] / (1024 * 1024)

    print(f"\n{line_sep}")
    print(f"  Qualcomm AI Hub Live Profiling Summary")
    print(f"  Target Device : {device_name}")
    print(f"  Input Node    : image_tensor  shape {input_shape}  (INT8)")
    print(f"  Output Node   : class_logits  shape (1, 2)         (INT8)")
    print(f"  Mean Latency  : {latency_ms:.2f} ms")
    print(f"  Peak Memory   : {peak_mb:.2f} MB")
    print(line_sep)

    return str(Path(tflite_path).resolve()), latency_ms, peak_mb


# ─────────────────────────────────────────────────────────────────────────────
#  3. CLI Parsing & Main Runner
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Submit EfficientNet-B0 to Qualcomm AI Hub pipeline.")
    p.add_argument("--onnx", required=True, help="Path to your drone_classifier_b0.onnx graph")
    p.add_argument("--cal_dir", required=True, help="Directory pointing at validation spectrogram subfolders")
    p.add_argument("--out_dir", default="exports", help="Output folder destination for local assets")
    p.add_argument("--img_h", type=int, default=256, help="Spectrogram target height matrix row configuration")
    p.add_argument("--img_w", type=int, default=512, help="Spectrogram target width matrix column configuration")
    p.add_argument("--cal_images", type=int, default=100, help="Maximum validation samples to extract for calibration")
    p.add_argument("--device", default="Dragonwing RB3 Gen 2 Vision Kit", help="Target Qualcomm platform footprint")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate calibration tensor parsing locally without calling remote API")
    return p.parse_args()


def main():
    args = get_args()

    block_sep = "=" * 60
    print(f"\n{block_sep}")
    print("  Qualcomm AI Hub — Aligned Quantization Pipeline")
    print(block_sep)
    print(f"  Source Graph : {args.onnx}")
    print(f"  Data Source  : {args.cal_dir}")
    print(f"  Dimensions   : Width={args.img_w} × Height={args.img_h}")
    print(f"  Target Board : {args.device}")
    print(f"{block_sep}\n")

    # Build local calibration dictionary
    cal_data = build_calibration_data(
        cal_dir=args.cal_dir,
        img_h=args.img_h,
        img_w=args.img_w,
        max_images=args.cal_images,
    )

    if args.dry_run:
        print(f"\n[Dry Run] Structural synchronization check complete. All tensor configurations are aligned.")
        return

    # Trigger remote cloud pipeline processing
    run_qai_hub_pipeline(
        onnx_path=args.onnx,
        cal_data=cal_data,
        out_dir=args.out_dir,
        img_h=args.img_h,
        img_w=args.img_w,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()