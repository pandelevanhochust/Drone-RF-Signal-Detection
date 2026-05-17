"""
infer.py
========
Run inference on a spectrogram image through the full two-stage pipeline.

Supports three backends
-----------------------
  --backend pt    PyTorch  (.pt checkpoint)   — development / debugging
  --backend onnx  ONNX     (.onnx files)      — pre-deployment validation
  --backend dlc   SNPE DLC (.dlc files)       — on-device Qualcomm runtime

All three backends produce identical predictions. Use pt during training,
onnx to validate the export, dlc on the actual Qualcomm device.

Pipeline (same logic for every backend)
-----------------------------------------
  spectrogram.png
      │
      ▼  preprocess()          resize → tensor → ImageNet normalise
      │
      ▼  U-Net                 (B,3,H,W) → (B,1,H,W) sigmoid mask
      │
      ▼  roi_extract()         mask × spectrogram → resize 224×224
      │
      ▼  Classifier            (B,3,224,224) → (B,num_classes) logits
      │
      ▼  softmax + argmax      → class name + confidence

Usage
-----
  # PyTorch
  python infer.py --backend pt \
                  --ckpt   checkpoints/classifier_best.pt \
                  --image  my_spectrogram.png

  # ONNX
  python infer.py --backend   onnx \
                  --unet_onnx exports/drone_unet.onnx \
                  --cls_onnx  exports/drone_classifier.onnx \
                  --labels    exports/class_names.txt \
                  --image     my_spectrogram.png

  # DLC  (run on Qualcomm device with SNPE Python bindings)
  python infer.py --backend  dlc \
                  --unet_dlc exports/drone_unet.dlc \
                  --cls_dlc  exports/drone_classifier.dlc \
                  --labels   exports/class_names.txt \
                  --image    my_spectrogram.png

  # Batch: infer on every PNG in a directory
  python infer.py --backend pt --ckpt checkpoints/classifier_best.pt \
                  --image_dir data/test_spectrograms/

  # Save mask visualisation alongside prediction
  python infer.py --backend pt --ckpt checkpoints/classifier_best.pt \
                  --image my_spectrogram.png --save_mask
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  ImageNet normalisation constants  (must match drone_dataloader.py)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared pre/post processing  (numpy, backend-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_path: str, img_h: int = 256, img_w: int = 512) -> np.ndarray:
    """
    Load a spectrogram PNG and convert to a normalised float32 array.

    Returns
    -------
    np.ndarray  shape (1, 3, img_h, img_w)  dtype float32  — NCHW
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_w, img_h), Image.BILINEAR)   # PIL: (W, H)
    arr = np.array(img, dtype=np.float32) / 255.0       # HWC [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD          # normalise
    arr = arr.transpose(2, 0, 1)                        # HWC → CHW
    return arr[np.newaxis]                              # → NCHW (1,3,H,W)


def roi_extract(
    spectrogram: np.ndarray,   # (1, 3, H, W)
    mask       : np.ndarray,   # (1, 1, H, W)  values in [0, 1]
    threshold  : float = 0.5,
    out_size   : tuple = (224, 224),
) -> np.ndarray:               # (1, 3, 224, 224)
    """
    Numpy ROI extractor — mirrors ROIExtractor(strategy='multiply').
    Runs on CPU for all backends (lightweight resize op).
    """
    binary = (mask >= threshold).astype(np.float32)     # (1,1,H,W)
    roi    = spectrogram * binary                        # zero background

    # Bilinear resize via PIL per channel
    B, C, H, W = roi.shape
    out_h, out_w = out_size
    result = np.zeros((B, C, out_h, out_w), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            ch  = roi[b, c]                             # (H, W)
            pil = Image.fromarray(
                np.clip(ch * 255, 0, 255).astype(np.uint8)
            ).resize((out_w, out_h), Image.BILINEAR)
            result[b, c] = np.array(pil, dtype=np.float32) / 255.0
    return result


def postprocess(
    logits     : np.ndarray,   # (1, num_classes)
    class_names: list,
) -> dict:
    """Softmax → top-3 predictions with confidence scores."""
    logits  = logits[0]                                 # (num_classes,)
    exp     = np.exp(logits - logits.max())             # stable softmax
    probs   = exp / exp.sum()
    top3_idx = np.argsort(probs)[::-1][:3]
    return {
        "pred"      : class_names[top3_idx[0]],
        "confidence": float(probs[top3_idx[0]]),
        "top3"      : [
            {"class": class_names[i], "prob": float(probs[i])}
            for i in top3_idx
        ],
        "probs_all" : {class_names[i]: float(probs[i])
                       for i in range(len(class_names))},
    }


def save_mask_png(
    mask      : np.ndarray,   # (1, 1, H, W)
    image_path: str,
    out_dir   : str = ".",
):
    """Save the U-Net mask as a greyscale PNG next to the source image."""
    mask_uint8 = (mask[0, 0] * 255).clip(0, 255).astype(np.uint8)
    pil_mask   = Image.fromarray(mask_uint8, mode="L")
    stem       = Path(image_path).stem
    out_path   = os.path.join(out_dir, f"{stem}_mask.png")
    pil_mask.save(out_path)
    return out_path


def load_class_names(labels_path: str) -> list:
    with open(labels_path) as f:
        return [l.strip() for l in f if l.strip()]


# ─────────────────────────────────────────────────────────────────────────────
#  Backend: PyTorch  (.pt)
# ─────────────────────────────────────────────────────────────────────────────

class PTBackend:
    """
    Inference using the original PyTorch checkpoint.
    Loads both U-Net and classifier from a Stage-2 .pt file.
    Use this during development — fastest iteration, full debugging.
    """

    def __init__(self, ckpt_path: str, device: str = "auto"):
        import torch
        from EfficientNetB0_Classification import load_classifier

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )
        self.unet, self.extractor, self.cls_net, self.class_names = \
            load_classifier(ckpt_path, self.device)
        self.unet.eval()
        self.cls_net.eval()
        self.torch = torch
        print(f"[PT]  Loaded from {ckpt_path}  ({self.device})")
        print(f"[PT]  Classes: {self.class_names}")

    def run(self, spectrogram_nchw: np.ndarray) -> tuple:
        """
        Returns
        -------
        mask    : np.ndarray (1,1,H,W)
        logits  : np.ndarray (1, num_classes)
        """
        import torch
        with torch.no_grad():
            x     = torch.from_numpy(spectrogram_nchw).to(self.device)
            mask  = self.unet(x)
            roi   = self.extractor(x, mask)
            logits = self.cls_net(roi)
        return mask.cpu().numpy(), logits.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Backend: ONNX  (.onnx)
# ─────────────────────────────────────────────────────────────────────────────

class ONNXBackend:
    """
    Inference using exported ONNX files via onnxruntime.

    Use this to validate the ONNX export produces the same results as PT
    before handing off to Qualcomm's snpe-onnx-to-dlc converter.

    Requires:  pip install onnxruntime
    """

    def __init__(
        self,
        unet_onnx  : str,
        cls_onnx   : str,
        labels_path: str,
        device     : str = "cpu",   # "cpu" or "cuda"
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            sys.exit("onnxruntime not installed.  pip install onnxruntime")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.sess_unet = ort.InferenceSession(unet_onnx, providers=providers)
        self.sess_cls  = ort.InferenceSession(cls_onnx,  providers=providers)
        self.class_names = load_class_names(labels_path)

        self.unet_input  = self.sess_unet.get_inputs()[0].name
        self.cls_input   = self.sess_cls.get_inputs()[0].name
        self.unet_output = self.sess_unet.get_outputs()[0].name
        self.cls_output  = self.sess_cls.get_outputs()[0].name

        print(f"[ONNX] U-Net:      {unet_onnx}")
        print(f"[ONNX] Classifier: {cls_onnx}")
        print(f"[ONNX] Classes:    {self.class_names}")
        print(f"[ONNX] U-Net   input  '{self.unet_input}'  "
              f"output '{self.unet_output}'")
        print(f"[ONNX] Cls     input  '{self.cls_input}'  "
              f"output '{self.cls_output}'")

    def run(self, spectrogram_nchw: np.ndarray) -> tuple:
        # Step 1: U-Net → mask
        mask = self.sess_unet.run(
            [self.unet_output],
            {self.unet_input: spectrogram_nchw},
        )[0]   # (1, 1, H, W)

        # Step 2: ROI extraction (numpy, same as PT backend)
        roi = roi_extract(spectrogram_nchw, mask)   # (1, 3, 224, 224)

        # Step 3: Classifier → logits
        logits = self.sess_cls.run(
            [self.cls_output],
            {self.cls_input: roi},
        )[0]   # (1, num_classes)

        return mask, logits


# ─────────────────────────────────────────────────────────────────────────────
#  Backend: DLC  (.dlc)  — Qualcomm SNPE on-device
# ─────────────────────────────────────────────────────────────────────────────

class DLCBackend:
    """
    Inference using Qualcomm SNPE DLC files.

    Requires the Qualcomm AI Engine Direct SDK (SNPE) Python bindings.
    Must be run on the target Qualcomm device or Qualcomm-enabled host.

    Setup
    -----
        source $SNPE_ROOT/bin/envsetup.sh
        export PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH

    Runtime targets
    ---------------
    --dlc_runtime cpu   : Qualcomm Kryo CPU
    --dlc_runtime gpu   : Qualcomm Adreno GPU
    --dlc_runtime dsp   : Qualcomm Hexagon DSP (fastest, requires int8 DLC)
    --dlc_runtime aic   : Qualcomm Cloud AI (server)
    """

    def __init__(
        self,
        unet_dlc   : str,
        cls_dlc    : str,
        labels_path: str,
        runtime    : str = "cpu",
    ):
        try:
            from snpe import SNPEBuilder, SNPERuntime
            self._SNPERuntime = SNPERuntime
        except ImportError:
            sys.exit(
                "SNPE Python bindings not found.\n"
                "Source the SNPE SDK:  source $SNPE_ROOT/bin/envsetup.sh\n"
                "Then add to PYTHONPATH: $SNPE_ROOT/lib/python"
            )

        runtime_map = {
            "cpu": SNPERuntime.CPU,
            "gpu": SNPERuntime.GPU,
            "dsp": SNPERuntime.DSP,
            "aic": SNPERuntime.AIP,
        }
        rt = runtime_map.get(runtime, SNPERuntime.CPU)

        self.snpe_unet = SNPEBuilder(unet_dlc).build(rt)
        self.snpe_cls  = SNPEBuilder(cls_dlc).build(rt)
        self.class_names = load_class_names(labels_path)

        print(f"[DLC] U-Net:      {unet_dlc}")
        print(f"[DLC] Classifier: {cls_dlc}")
        print(f"[DLC] Runtime:    {runtime.upper()}")
        print(f"[DLC] Classes:    {self.class_names}")

    def run(self, spectrogram_nchw: np.ndarray) -> tuple:
        """
        SNPE expects NHWC by default. If the DLC was converted with
        --input_layout NCHW, pass NCHW directly. Otherwise transpose here.
        We transpose to NHWC to match default SNPE behaviour.
        """
        # NCHW → NHWC for SNPE default layout
        spec_nhwc = spectrogram_nchw.transpose(0, 2, 3, 1)   # (1,H,W,C)

        # Step 1: U-Net
        unet_out  = self.snpe_unet.execute({"spectrogram": spec_nhwc})
        mask_nhwc = unet_out["roi_mask"]                      # (1,H,W,1)
        mask_nchw = mask_nhwc.transpose(0, 3, 1, 2)          # → (1,1,H,W)

        # Step 2: ROI extraction (numpy)
        roi_nchw = roi_extract(spectrogram_nchw, mask_nchw)   # (1,3,224,224)
        roi_nhwc = roi_nchw.transpose(0, 2, 3, 1)             # (1,224,224,3)

        # Step 3: Classifier
        cls_out = self.snpe_cls.execute({"roi_patch": roi_nhwc})
        logits  = cls_out["class_logits"]                     # (1, num_classes)

        return mask_nchw, logits


# ─────────────────────────────────────────────────────────────────────────────
#  Single image inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_one(
    backend,
    image_path : str,
    img_h      : int   = 256,
    img_w      : int   = 512,
    save_mask  : bool  = False,
    mask_dir   : str   = ".",
    verbose    : bool  = True,
) -> dict:
    t0 = time.perf_counter()

    # Preprocess
    spec = preprocess(image_path, img_h, img_w)   # (1,3,H,W)

    # Run pipeline
    mask, logits = backend.run(spec)

    # Postprocess
    result = postprocess(logits, backend.class_names)
    result["image"]      = image_path
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    if save_mask:
        mask_path = save_mask_png(mask, image_path, mask_dir)
        result["mask_saved"] = mask_path

    if verbose:
        print_result(result)

    return result


def print_result(result: dict):
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  Image   : {Path(result['image']).name}")
    print(f"  Pred    : {result['pred']}  ({result['confidence']:.1%})")
    print(f"  Latency : {result['latency_ms']} ms")
    print(f"\n  Top-3:")
    for i, t in enumerate(result["top3"], 1):
        bar = "█" * int(t["prob"] * 20) + "░" * (20 - int(t["prob"] * 20))
        print(f"    {i}. {t['class']:12s} [{bar}] {t['prob']:.1%}")
    print(f"\n  All class probabilities:")
    for cls, prob in sorted(result["probs_all"].items(),
                            key=lambda x: -x[1]):
        bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
        print(f"    {cls:12s} [{bar}] {prob:.3%}")
    if "mask_saved" in result:
        print(f"\n  Mask saved → {result['mask_saved']}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
#  Batch inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_batch(
    backend,
    image_dir  : str,
    img_h      : int  = 256,
    img_w      : int  = 512,
    save_mask  : bool = False,
    out_csv    : str  = None,
) -> list:
    paths = sorted(
        list(Path(image_dir).glob("**/*.png")) +
        list(Path(image_dir).glob("**/*.jpg"))
    )
    if not paths:
        print(f"[Batch] No images found in {image_dir}")
        return []

    print(f"[Batch] {len(paths)} images found in {image_dir}\n")
    results = []
    correct = 0

    for i, p in enumerate(paths, 1):
        r = infer_one(backend, str(p), img_h, img_w,
                      save_mask=save_mask, verbose=False)
        results.append(r)

        # If path contains a class subfolder, compute accuracy
        parent = p.parent.name
        is_correct = (parent == r["pred"])
        if parent in backend.class_names:
            correct += int(is_correct)
        ok = "✓" if is_correct else "✗"

        print(f"  [{i:4d}/{len(paths)}]  {ok}  {p.name:<50}  "
              f"{r['pred']:<12}  {r['confidence']:.1%}  "
              f"({r['latency_ms']} ms)")

    # Summary
    n_with_gt = sum(1 for r in results
                    if Path(r["image"]).parent.name in backend.class_names)
    if n_with_gt > 0:
        acc = correct / n_with_gt
        print(f"\n  Accuracy on labelled samples: {acc:.2%}  "
              f"({correct}/{n_with_gt})")

    avg_lat = np.mean([r["latency_ms"] for r in results])
    print(f"  Average latency: {avg_lat:.1f} ms  "
          f"({1000/avg_lat:.1f} FPS equivalent)")

    # Optional CSV export
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image", "pred", "confidence", "latency_ms"]
            )
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "image"      : Path(r["image"]).name,
                    "pred"       : r["pred"],
                    "confidence" : f"{r['confidence']:.4f}",
                    "latency_ms" : r["latency_ms"],
                })
        print(f"\n  Results saved → {out_csv}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Drone pipeline inference — PT / ONNX / DLC backends",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Backend selection
    p.add_argument("--backend", required=True,
                   choices=["pt", "onnx", "dlc"],
                   help="Inference backend:\n"
                        "  pt   — PyTorch checkpoint (development)\n"
                        "  onnx — ONNX via onnxruntime  (pre-deployment check)\n"
                        "  dlc  — Qualcomm SNPE DLC     (on-device)")

    # PT backend
    p.add_argument("--ckpt",       default="checkpoints/classifier_best.pt",
                   help="[pt] Stage-2 PyTorch checkpoint")

    # ONNX / DLC backends
    p.add_argument("--unet_onnx",  default="exports/drone_unet.onnx")
    p.add_argument("--cls_onnx",   default="exports/drone_classifier.onnx")
    p.add_argument("--unet_dlc",   default="exports/drone_unet.dlc")
    p.add_argument("--cls_dlc",    default="exports/drone_classifier.dlc")
    p.add_argument("--labels",     default="exports/class_names.txt",
                   help="[onnx/dlc] Path to class_names.txt")

    # Input
    p.add_argument("--image",      default=None,
                   help="Single spectrogram image to infer")
    p.add_argument("--image_dir",  default=None,
                   help="Directory of spectrograms for batch inference")

    # Options
    p.add_argument("--img_h",      type=int, default=256)
    p.add_argument("--img_w",      type=int, default=512)
    p.add_argument("--save_mask",  action="store_true",
                   help="Save U-Net mask PNG alongside input image")
    p.add_argument("--mask_dir",   default=".",
                   help="Directory to save mask PNGs (default: current dir)")
    p.add_argument("--out_csv",    default=None,
                   help="[batch] Save predictions to this CSV file")
    p.add_argument("--device",     default="cpu",
                   help="[pt/onnx] 'cpu' or 'cuda'")
    p.add_argument("--dlc_runtime",default="cpu",
                   choices=["cpu", "gpu", "dsp", "aic"],
                   help="[dlc] SNPE runtime target")

    return p.parse_args()


def main():
    args = parse_args()

    if args.image is None and args.image_dir is None:
        sys.exit("Provide --image <path> or --image_dir <dir>")

    # ── Build backend ─────────────────────────────────────────────────────────
    if args.backend == "pt":
        backend = PTBackend(args.ckpt, device=args.device)

    elif args.backend == "onnx":
        backend = ONNXBackend(
            unet_onnx   = args.unet_onnx,
            cls_onnx    = args.cls_onnx,
            labels_path = args.labels,
            device      = args.device,
        )

    elif args.backend == "dlc":
        backend = DLCBackend(
            unet_dlc    = args.unet_dlc,
            cls_dlc     = args.cls_dlc,
            labels_path = args.labels,
            runtime     = args.dlc_runtime,
        )

    # ── Run inference ─────────────────────────────────────────────────────────
    if args.image:
        infer_one(
            backend,
            image_path = args.image,
            img_h      = args.img_h,
            img_w      = args.img_w,
            save_mask  = args.save_mask,
            mask_dir   = args.mask_dir,
        )

    if args.image_dir:
        infer_batch(
            backend,
            image_dir  = args.image_dir,
            img_h      = args.img_h,
            img_w      = args.img_w,
            save_mask  = args.save_mask,
            out_csv    = args.out_csv,
        )


if __name__ == "__main__":
    main()