"""
export_and_infer.py
===================
Export FusedDronePipeline to ONNX and run inference.

One model, one ONNX, one Qualcomm AI Hub job
---------------------------------------------
Because the full pipeline is now a single graph, you export one ONNX file
and submit one quantize + compile job to Qualcomm AI Hub. No CPU↔NPU
handoffs, no two-model orchestration on-device.

Export
------
    python export_and_infer.py export \
        --ckpt checkpoints/fused_best.pt \
        --out  exports/drone_fused.onnx

Inference — PyTorch (.pt)
    python export_and_infer.py infer \
        --backend pt \
        --ckpt    checkpoints/fused_best.pt \
        --image   spectrogram.png

Inference — ONNX (pre-deployment validation)
    python export_and_infer.py infer \
        --backend  onnx \
        --onnx     exports/drone_fused.onnx \
        --labels   exports/class_names.txt \
        --image    spectrogram.png

Qualcomm AI Hub quantization (run in notebook / colab)
-------------------------------------------------------
    See the docstring of `hub_quantize_snippet()` below.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(
    ckpt_path: str,
    out_path : str  = "exports/drone_fused.onnx",
    opset    : int  = 17,
    img_h    : int  = 256,
    img_w    : int  = 512,
):
    """
    Export FusedDronePipeline to a single ONNX graph.

    Input  : spectrogram  (1, 3, img_h, img_w)  float32
    Output : class_logits (1, num_classes)       float32

    The mask is an internal node — not an output. SNPE/AI Hub only needs
    the final logits. If you want the mask as a second output for debugging,
    call model.forward(x, return_mask=True) and export that instead.

    SNPE compatibility guarantees
    ------------------------------
    ✓  ConvTranspose2d decoder — native NPU op, no bilinear upsample
    ✓  Soft mask (sigmoid) — continuous graph, no binary branch
    ✓  dim=(2,3) tuple in ReduceMean — static ONNX attribute
    ✓  No AdaptiveAvgPool2d — x.mean(dim=(2,3)) pattern
    ✓  No Softmax output — logits only
    ✓  dynamo=False — legacy TorchScript exporter for static Slice nodes
    ✓  static batch=1, no dynamic_axes — required for SNPE DLC conversion
    """
    import torch
    from fused_model import load_pipeline

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    device = torch.device("cpu")
    model  = load_pipeline(ckpt_path, device)
    model.eval()

    dummy = torch.zeros(1, 3, img_h, img_w)

    print(f"\n[Export] {tuple(dummy.shape)} → {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            opset_version       = opset,
            input_names         = ["spectrogram"],
            output_names        = ["class_logits"],
            dynamic_axes        = {},       # fully static — SNPE requirement
            do_constant_folding = True,
            export_params       = True,
            dynamo              = False,    # legacy exporter → static Slice ops
        )

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  ✓ Saved  ({size_mb:.1f} MB)")

    # Save class names for on-device labelling
    ckpt = __import__("torch").load(ckpt_path, map_location="cpu",
                                    weights_only=False)
    labels_path = str(Path(out_path).parent / "class_names.txt")
    with open(labels_path, "w") as f:
        for n in ckpt["class_names"]:
            f.write(n + "\n")
    print(f"  Class names → {labels_path}")
    print(f"  Classes: {ckpt['class_names']}")
    return out_path


def validate_onnx(onnx_path: str, img_h: int = 256, img_w: int = 512):
    """Run one forward pass through onnxruntime to sanity-check the export."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Validate] pip install onnxruntime  — skipping.")
        return
    sess  = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, img_h, img_w).astype(np.float32)
    out   = sess.run(None, {"spectrogram": dummy})
    print(f"\n[Validate] input {dummy.shape} → output {out[0].shape}")
    print(f"  logits: {out[0][0].tolist()}")
    print(f"  ✓ ONNX validated — safe to submit to Qualcomm AI Hub.")


# ─────────────────────────────────────────────────────────────────────────────
#  Qualcomm AI Hub snippet
# ─────────────────────────────────────────────────────────────────────────────

def hub_quantize_snippet():
    """
    Copy-paste this into your Colab notebook to quantize + compile on AI Hub.

    Key fixes vs your previous notebook
    ------------------------------------
    1. Calibration uses the FULL preprocessing chain (resize + normalise).
       The model was trained with ImageNet mean/std — calibration must match.
    2. Calibration images should be SPECTROGRAMS (pipeline input), not ROI
       patches. The fused model handles ROI extraction internally.
    3. 200 images recommended (was 50). More images = tighter INT8 ranges.
    4. One model, one job — no separate U-Net + classifier jobs needed.
    """
    snippet = '''
import qai_hub as hub
import numpy as np
from PIL import Image
import glob, os

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_H, IMG_W  = 256, 512   # must match training --img_size

device = hub.Device("Dragonwing RB3 Gen 2 Vision Kit")

# 1. Upload the single fused ONNX
model_source = hub.upload_model("exports/drone_fused.onnx",
                                 name="Drone_Fused_FP32")

# 2. Build calibration dataset
#    Use spectrograms (full pipeline input), not ROI patches.
#    Must apply the same preprocessing as drone_dataloader.py.
cal_files = glob.glob("/path/to/spectrograms/**/*.png", recursive=True)[:200]

cal_images = []
for p in cal_files:
    img = Image.open(p).convert("RGB").resize((IMG_W, IMG_H))  # (W, H) for PIL
    arr = np.array(img, dtype=np.float32) / 255.0              # [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD                 # normalise ← was missing
    arr = arr.transpose(2, 0, 1)                               # HWC → CHW
    arr = arr[np.newaxis]                                      # → (1,3,H,W)
    cal_images.append(arr)

cal_data = {"spectrogram": cal_images}
print(f"Calibration: {len(cal_images)} images")

# 3. Quantize FP32 → INT8
quantize_job = hub.submit_quantize_job(
    model              = model_source,
    calibration_data   = cal_data,
    weights_dtype      = hub.QuantizeDtype.INT8,
    activations_dtype  = hub.QuantizeDtype.INT8,
)
quantized_model = quantize_job.get_target_model()

# 4. Compile for NPU
compile_job = hub.submit_compile_job(
    model       = quantized_model,
    device      = device,
    input_specs = dict(spectrogram=(1, 3, IMG_H, IMG_W)),
    options     = "--target_runtime tflite --compute_unit npu",
)
compiled_model = compile_job.get_target_model()
print("Compiled:", compiled_model)

# 5. Profile
profile_job = hub.submit_profile_job(
    model   = compiled_model,
    device  = device,
    options = "--compute_unit npu",
)
results = profile_job.download_profile()
lat = results["execution_summary"]["estimated_inference_time"]
mem = results["execution_summary"]["estimated_inference_peak_memory"] / 1024**2
print(f"Latency : {lat:.2f} ms")
print(f"Peak mem: {mem:.2f} MB")

# 6. Inference
test_img = Image.open("spectrogram.png").convert("RGB").resize((IMG_W, IMG_H))
test_arr = np.array(test_img, dtype=np.float32) / 255.0
test_arr = (test_arr - IMAGENET_MEAN) / IMAGENET_STD
test_arr = test_arr.transpose(2, 0, 1)[np.newaxis]

inf_job = hub.submit_inference_job(
    model   = compiled_model,
    device  = device,
    inputs  = {"spectrogram": [test_arr]},
    options = "--compute_unit npu",
)
output   = inf_job.download_output_data()
logits   = list(output.values())[0][0]     # (num_classes,)
probs    = np.exp(logits) / np.exp(logits).sum()

class_names = open("exports/class_names.txt").read().splitlines()
pred = class_names[np.argmax(probs)]
conf = float(probs.max())
print(f"Prediction: {pred}  ({conf:.1%})")
'''
    return snippet


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing / postprocessing  (shared by all inference backends)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_path: str, img_h: int = 256, img_w: int = 512) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((img_w, img_h))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis]   # (1, 3, H, W)


def postprocess(logits: np.ndarray, class_names: list) -> dict:
    logits = logits[0]
    probs  = np.exp(logits - logits.max())
    probs /= probs.sum()
    top3   = np.argsort(probs)[::-1][:3]
    return {
        "pred"      : class_names[top3[0]],
        "confidence": float(probs[top3[0]]),
        "top3"      : [{"class": class_names[i], "prob": float(probs[i])}
                       for i in top3],
        "all_probs" : {class_names[i]: float(probs[i])
                       for i in range(len(class_names))},
    }


def print_result(result: dict):
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Image      : {Path(result['image']).name}")
    print(f"  Prediction : {result['pred']}  ({result['confidence']:.1%})")
    print(f"  Latency    : {result['latency_ms']} ms")
    print(f"\n  Top-3:")
    for i, t in enumerate(result["top3"], 1):
        bar = "█" * int(t["prob"] * 25) + "░" * (25 - int(t["prob"] * 25))
        print(f"    {i}. {t['class']:12s} [{bar}] {t['prob']:.1%}")
    print(f"\n  All classes:")
    for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 25) + "░" * (25 - int(prob * 25))
        print(f"    {cls:12s} [{bar}] {prob:.2%}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
#  Inference backends
# ─────────────────────────────────────────────────────────────────────────────

class PTBackend:
    def __init__(self, ckpt_path: str, device: str = "auto"):
        import torch
        from fused_model import load_pipeline
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )
        self.model = load_pipeline(ckpt_path, self.device)
        self.model.eval()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.class_names = ckpt["class_names"]
        print(f"[PT] {ckpt_path}  classes: {self.class_names}")

    def run(self, spec: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            x      = torch.from_numpy(spec).to(self.device)
            logits = self.model(x)
        return logits.cpu().numpy()


class ONNXBackend:
    def __init__(self, onnx_path: str, labels_path: str, device: str = "cpu"):
        try:
            import onnxruntime as ort
        except ImportError:
            sys.exit("pip install onnxruntime")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.sess        = ort.InferenceSession(onnx_path, providers=providers)
        self.class_names = open(labels_path).read().splitlines()
        self.in_name     = self.sess.get_inputs()[0].name
        self.out_name    = self.sess.get_outputs()[0].name
        print(f"[ONNX] {onnx_path}  input='{self.in_name}'  "
              f"output='{self.out_name}'")
        print(f"[ONNX] classes: {self.class_names}")

    def run(self, spec: np.ndarray) -> np.ndarray:
        return self.sess.run([self.out_name], {self.in_name: spec})[0]


def infer_one(
    backend,
    image_path: str,
    img_h: int = 256,
    img_w: int = 512,
    save_mask: bool = False,
):
    t0   = time.perf_counter()
    spec = preprocess(image_path, img_h, img_w)
    logits = backend.run(spec)
    result = postprocess(logits, backend.class_names)
    result["image"]      = image_path
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    print_result(result)
    return result


def infer_batch(backend, image_dir: str, img_h: int = 256, img_w: int = 512,
                out_csv: str = None):
    paths = sorted(Path(image_dir).glob("**/*.png")) + \
            sorted(Path(image_dir).glob("**/*.jpg"))
    if not paths:
        print(f"No images in {image_dir}")
        return []
    results = []
    correct = total_gt = 0
    print(f"\n[Batch] {len(paths)} images in {image_dir}\n")
    for i, p in enumerate(paths, 1):
        r     = infer_one(backend, str(p), img_h, img_w)
        parent = p.parent.name
        ok     = "✓" if parent == r["pred"] else "✗"
        if parent in backend.class_names:
            correct   += int(parent == r["pred"])
            total_gt  += 1
        print(f"  [{i:4d}/{len(paths)}] {ok}  {p.name:<50}  "
              f"{r['pred']:<12}  {r['confidence']:.1%}  ({r['latency_ms']} ms)")
        results.append(r)
    if total_gt:
        print(f"\n  Accuracy: {correct/total_gt:.2%}  ({correct}/{total_gt})")
    lats = [r["latency_ms"] for r in results]
    print(f"  Avg latency: {np.mean(lats):.1f} ms")
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, ["image","pred","confidence","latency_ms"])
            w.writeheader()
            for r in results:
                w.writerow({"image": Path(r["image"]).name,
                             "pred": r["pred"],
                             "confidence": f"{r['confidence']:.4f}",
                             "latency_ms": r["latency_ms"]})
        print(f"  Results → {out_csv}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    # export
    ex = sub.add_parser("export", help="Export .pt → .onnx")
    ex.add_argument("--ckpt",    default="checkpoints/fused_best.pt")
    ex.add_argument("--out",     default="exports/drone_fused.onnx")
    ex.add_argument("--opset",   type=int, default=17)
    ex.add_argument("--img_h",   type=int, default=256)
    ex.add_argument("--img_w",   type=int, default=512)
    ex.add_argument("--validate",action="store_true")
    ex.add_argument("--hub",     action="store_true",
                    help="Print Qualcomm AI Hub quantization snippet")

    # infer
    inf = sub.add_parser("infer", help="Run inference on image(s)")
    inf.add_argument("--backend",   choices=["pt","onnx"], default="pt")
    inf.add_argument("--ckpt",      default="checkpoints/fused_best.pt")
    inf.add_argument("--onnx",      default="exports/drone_fused.onnx")
    inf.add_argument("--labels",    default="exports/class_names.txt")
    inf.add_argument("--image",     default=None)
    inf.add_argument("--image_dir", default=None)
    inf.add_argument("--img_h",     type=int, default=256)
    inf.add_argument("--img_w",     type=int, default=512)
    inf.add_argument("--out_csv",   default=None)
    inf.add_argument("--device",    default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    if args.cmd == "export":
        path = export_onnx(args.ckpt, args.out, args.opset,
                           args.img_h, args.img_w)
        if args.validate:
            validate_onnx(path, args.img_h, args.img_w)
        if args.hub:
            print("\n" + "─"*60)
            print("Qualcomm AI Hub quantization snippet:")
            print("─"*60)
            print(hub_quantize_snippet())

    elif args.cmd == "infer":
        if args.image is None and args.image_dir is None:
            sys.exit("Provide --image or --image_dir")
        backend = (PTBackend(args.ckpt, args.device)
                   if args.backend == "pt"
                   else ONNXBackend(args.onnx, args.labels, args.device))
        if args.image:
            infer_one(backend, args.image, args.img_h, args.img_w)
        if args.image_dir:
            infer_batch(backend, args.image_dir, args.img_h, args.img_w,
                        args.out_csv)
    else:
        print("Commands: export | infer")


if __name__ == "__main__":
    main()