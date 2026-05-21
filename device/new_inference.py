"""
drone_inference.py
==================
On-device inference for drone_pipeline_fused_quantized.tflite on the
Dragonwing RB3 Gen 2 (Rubik Pi 3) using the LiteRT / QNN NPU delegate.

Pipeline covered by the single .tflite
---------------------------------------
    spectrogram (1, 3, 256, 512)
        → DroneROIUNet  (mask)
        → ROI extraction (multiply + bilinear resize)
        → EfficientNet-B0
        → class_logits   (1, 8)

No Python preprocessing between stages — everything runs on NPU.

Setup (run once on device)
--------------------------
    git clone -b ubuntu_setup --single-branch \\
        https://github.com/rubikpi-ai/rubikpi-script.git
    cd rubikpi-script && ./install_ppa_pkgs.sh

    python3 -m venv .venv-drone --system-site-packages
    source .venv-drone/bin/activate
    pip3 install ai-edge-litert==1.3.0 Pillow numpy

Copy files to device (from your dev machine)
--------------------------------------------
    scp exports/drone_pipeline_fused_quantized.tflite ubuntu@<IP>:/home/ubuntu/
    scp exports/class_names.txt                        ubuntu@<IP>:/home/ubuntu/

Usage
-----
    # Single image, NPU
    python3 drone_inference.py --image spectrogram.png

    # Single image, CPU only (for comparison / debugging)
    python3 drone_inference.py --image spectrogram.png --cpu

    # Benchmark: run N times and report average latency
    python3 drone_inference.py --image spectrogram.png --runs 50

    # Folder of images — prints one prediction per line
    python3 drone_inference.py --folder /path/to/spectrograms/

Important: input preprocessing
-------------------------------
Must exactly match drone_dataloader.get_transforms('val'):
    PIL RGB  →  resize (256, 512)  →  / 255.0
             →  subtract IMAGENET_MEAN  →  divide IMAGENET_STD
             →  NCHW float32  (1, 3, 256, 512)

The QNN/TFLite runtime handles the NCHW → NHWC layout swap at the first
NPU op.  Feed NCHW tensors — do NOT manually transpose to NHWC.

Output dequantisation
---------------------
INT8 TFLite outputs are quantised integers.  Dequantise with:
    logits_f32 = (output_int8 - zero_point) * scale
Then apply softmax to get probabilities.  Do NOT add a second softmax
if the model already contains one (check output_details[0]['quantization']).
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter, load_delegate

# ── ImageNet normalisation — must match training exactly ─────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_H, IMG_W  = 256, 512       # spectrogram input size
DELEGATE_LIB  = "libQnnTFLiteDelegate.so"


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> np.ndarray:
    """
    Load a spectrogram PNG and produce the model input tensor.

    Steps (must mirror drone_dataloader.get_transforms('val')):
        1. Open as RGB
        2. Resize to (IMG_H=256, IMG_W=512) — PIL.resize takes (W, H)
        3. Convert to float32, divide by 255
        4. Subtract IMAGENET_MEAN, divide by IMAGENET_STD  (HWC)
        5. Transpose HWC → CHW
        6. Add batch dim → (1, 3, 256, 512)  NCHW float32

    The QNN delegate handles NCHW internally — do NOT transpose to NHWC.
    """
    img = Image.open(image_path).convert("RGB").resize(
        (IMG_W, IMG_H), Image.BILINEAR
    )
    arr = np.array(img, dtype=np.float32) / 255.0       # HWC [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD           # HWC normalised
    arr = arr.transpose(2, 0, 1)                         # CHW
    arr = arr[np.newaxis]                                # (1,3,256,512) NCHW
    return arr


# ─────────────────────────────────────────────────────────────────────────────
#  Interpreter setup
# ─────────────────────────────────────────────────────────────────────────────

def build_interpreter(model_path: str, use_npu: bool):
    """
    Load the LiteRT interpreter with or without the QNN NPU delegate.
    """
    delegates = []
    if use_npu:
        try:
            # Explicitly point to the Hexagon Tensor Processor (HTP) core and skel libraries
            delegate_options = {
                "backend_type": "htp",
                "library_path": "/usr/lib/libQnnHtp.so",
                "skel_library_dir": "/usr/lib/rfsa/adsp"
            }

            qnn = load_delegate(DELEGATE_LIB, options=delegate_options)
            delegates = [qnn]
            print(f"[Setup] QNN delegate loaded successfully (Backend: HTP / Hexagon NPU)")
        except Exception as exc:
            print(f"[Setup] WARNING: could not load QNN delegate: {exc}")
            print(f"[Setup] Falling back to CPU-only inference.")

    interp = Interpreter(
        model_path=model_path,
        experimental_delegates=delegates)
    interp.allocate_tensors()
    return interp


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    interp,
    input_tensor : np.ndarray,
    inp_idx      : int,
    out_idx      : int,
    out_quant    : tuple,
) -> np.ndarray:
    """
    Feed one preprocessed tensor, invoke, dequantise, return float32 logits.

    INT8 TFLite output dequantisation:
        logits_f32 = (output_int8.astype(float32) - zero_point) * scale

    The scale and zero_point come from output_details[0]['quantization'].
    For a model exported via Qualcomm Hub these are always set — if they are
    both 0.0 the model is FP32 and no dequantisation is needed.
    """
    interp.set_tensor(inp_idx, input_tensor)
    interp.invoke()
    out_raw = interp.get_tensor(out_idx)        # (1, 8) int8 or float32

    scale, zero_point = out_quant
    if scale != 0.0:
        # INT8 model (our case) — dequantise to float32
        logits = (out_raw.astype(np.float32) - zero_point) * scale
    else:
        # FP32 model — no dequantisation needed
        logits = out_raw.astype(np.float32)

    return logits                                # (1, 8) float32


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(np.clip(x, -500, 500))
    return e / e.sum(axis=-1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main inference routine
# ─────────────────────────────────────────────────────────────────────────────

def infer_image(
    interp,
    inp_idx   : int,
    out_idx   : int,
    out_quant : tuple,
    class_names,
    image_path: str,
    runs      : int = 1,
    verbose   : bool = True,
) -> dict:
    """
    Run inference on a single spectrogram image.

    Parameters
    ----------
    runs    : number of forward passes (use > 1 for latency benchmarking;
              first run is discarded as warmup)
    verbose : print prediction table to stdout

    Returns
    -------
    dict with keys: predicted_class, confidence, probs, latency_ms
    """
    tensor = preprocess(image_path)

    # Warmup — first invoke initialises NPU kernels; exclude from timing
    interp.set_tensor(inp_idx, tensor)
    interp.invoke()

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        logits = run_inference(interp, tensor, inp_idx, out_idx, out_quant)
        times.append((time.perf_counter() - t0) * 1000)

    probs       = softmax(logits)[0]             # (8,) float32
    pred_idx    = int(np.argmax(probs))
    pred_class  = class_names[pred_idx]
    confidence  = float(probs[pred_idx])
    latency_ms  = float(np.mean(times))

    if verbose:
        print(f"\n  Image   : {Path(image_path).name}")
        print(f"  Result  : {pred_class}  ({confidence*100:.1f}%)")
        print(f"  Latency : {latency_ms:.2f} ms  (avg over {runs} run(s))")
        print(f"\n  {'Class':<12} {'Prob':>7}")
        print(f"  {'─'*20}")
        for i, (name, p) in enumerate(zip(class_names, probs)):
            marker = " ◀" if i == pred_idx else ""
            print(f"  {name:<12} {p*100:>6.2f}%{marker}")

    return {
        "predicted_class" : pred_class,
        "confidence"      : confidence,
        "probs"           : probs,
        "latency_ms"      : latency_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Drone detection inference — fused TFLite on RB3 Gen 2 NPU"
    )
    p.add_argument("--model",   default="drone_pipeline_fused_quantized.tflite",
                   help="Path to the fused quantized .tflite model")
    p.add_argument("--labels",  default="class_names.txt",
                   help="Path to class_names.txt (one class per line)")
    p.add_argument("--image",   default=None,
                   help="Single spectrogram image to classify")
    p.add_argument("--folder",  default=None,
                   help="Folder of spectrogram images — classifies all PNG/JPG")
    p.add_argument("--runs",    type=int, default=1,
                   help="Number of inference runs per image (for benchmarking)")
    p.add_argument("--cpu",     action="store_true",
                   help="Disable NPU delegate and run on CPU only")
    return p.parse_args()


def main():
    args = get_args()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            "Copy it from your dev machine:\n"
            f"  scp exports/drone_pipeline_fused_quantized.tflite ubuntu@<IP>:/home/ubuntu/"
        )
    if not os.path.exists(args.labels):
        raise FileNotFoundError(
            f"Class names not found: {args.labels}\n"
            "Copy it alongside the model:\n"
            f"  scp exports/class_names.txt ubuntu@<IP>:/home/ubuntu/"
        )
    if args.image is None and args.folder is None:
        raise ValueError("Provide --image <path> or --folder <dir>")

    # ── Load class names ──────────────────────────────────────────────────────
    with open(args.labels) as f:
        class_names = [l.strip() for l in f if l.strip()]
    print(f"[Setup] Classes ({len(class_names)}): {class_names}")

    # ── Build interpreter ─────────────────────────────────────────────────────
    use_npu = not args.cpu
    interp  = build_interpreter(args.model, use_npu)

    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    inp_idx   = inp_details[0]["index"]
    inp_shape = inp_details[0]["shape"]         # expect (1, 3, 256, 512)
    out_idx   = out_details[0]["index"]
    out_quant = out_details[0]["quantization"]  # (scale, zero_point)

    print(f"[Setup] Input  : {inp_shape}  dtype={inp_details[0]['dtype']}")
    print(f"[Setup] Output : {out_details[0]['shape']}  "
          f"quant=(scale={out_quant[0]:.6f}, zp={out_quant[1]})")
    print(f"[Setup] Backend: {'NPU (HTP delegate)' if use_npu else 'CPU only'}\n")

    # Verify input shape matches what we produce in preprocess()
    assert list(inp_shape) == [1, 3, IMG_H, IMG_W], (
        f"Model expects input {list(inp_shape)} but script produces "
        f"(1, 3, {IMG_H}, {IMG_W}). Check IMG_H / IMG_W constants."
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    sep = "=" * 50

    if args.image:
        print(sep)
        infer_image(
            interp, inp_idx, out_idx, out_quant,
            class_names, args.image, runs=args.runs,
        )
        print(sep)

    elif args.folder:
        exts   = (".png", ".jpg", ".jpeg")
        images = sorted(
            p for p in Path(args.folder).rglob("*") if p.suffix.lower() in exts
        )
        if not images:
            raise FileNotFoundError(f"No images found in {args.folder}")

        print(f"[Batch] {len(images)} images  |  runs per image: {args.runs}\n")
        print(sep)

        results   = []
        latencies = []
        for img_path in images:
            r = infer_image(
                interp, inp_idx, out_idx, out_quant,
                class_names, str(img_path), runs=args.runs, verbose=True,
            )
            results.append(r)
            latencies.append(r["latency_ms"])

        print(sep)
        print(f"\n[Summary]  {len(images)} images")
        print(f"  Avg latency : {np.mean(latencies):.2f} ms")
        print(f"  Min latency : {np.min(latencies):.2f} ms")
        print(f"  Max latency : {np.max(latencies):.2f} ms")

        # Class distribution
        from collections import Counter
        dist = Counter(r["predicted_class"] for r in results)
        print(f"\n  Prediction distribution:")
        for cls, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {cls:<12} {cnt:>4}  ({100*cnt/len(results):.1f}%)")


if __name__ == "__main__":
    main()