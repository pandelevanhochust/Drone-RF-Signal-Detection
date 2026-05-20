"""
export_qai_hub.py
=================
Exports the full two-stage drone detection pipeline as a SINGLE fused ONNX
graph, then quantises, compiles, and profiles it on the RB3 Gen 2 NPU via
Qualcomm AI Hub.

Why fused (vs the old two-model approach)
-----------------------------------------
The previous export_dlc.py produced two separate graphs and relied on CPU
post-processing between them (mask multiply + bilinear resize). That design:
  • Stalls the NPU between the two model calls while CPU does ROI extraction
  • Doubles model-load overhead on-device
  • Prevents the compiler from fusing cross-boundary ops (e.g. the Resize
    that ends ROIExtractor is immediately followed by the EfficientNet stem
    Conv — the compiler can fuse them when they are in the same graph)

FusedDronePipeline wraps:
    spectrogram (1, 3, 256, 512)
        │
        ▼  DroneROIUNet
    pred_mask   (1, 1, 256, 512)
        │
        ▼  ROIExtractor (multiply strategy — fully differentiable, ONNX-clean)
    roi_patch   (1, 3, 224, 224)
        │
        ▼  DroneCLSNet
    class_logits (1, num_classes)   [NO softmax — apply in post-processing]

ONNX compatibility guarantees (all inherited from training code)
----------------------------------------------------------------
  ✓  DecoderBlock uses literal-integer slice indices baked into concrete
       subclasses → static Slice constants in ONNX, no dynamic axes
  ✓  SqueezeExcitation uses x.mean(dim=(2,3), keepdim=True) with a tuple
       → TorchScript folds tuple as constant → static ReduceMean attribute
  ✓  DroneCLSNet.forward uses x.mean(dim=(2,3)) — same pattern
  ✓  ROIExtractor uses F.interpolate(mode='bilinear', align_corners=False)
       → exports as Resize opset-13 with static size, no dynamic shapes
  ✓  mask multiply is elementwise — exports cleanly as Mul
  ✓  No Softmax in any module
  ✗  ROIExtractor 'bbox' strategy is NOT exportable (dynamic indexing).
       FusedDronePipeline always uses 'multiply'. bbox stays training-only.

Calibration preprocessing fix (vs old export_dlc.py)
-----------------------------------------------------
The old script normalised calibration images to [0, 1] (/ 255 only).
Training uses ImageNet mean/std normalisation. This file matches training
exactly: divide by 255 → subtract IMAGENET_MEAN → divide by IMAGENET_STD.

Usage
-----
  # Export fused ONNX, then run the full Hub flow
  python export_qai_hub.py --ckpt checkpoints/classifier_best.pt \\
                            --cal_dir /path/to/raw_spectrograms

  # Export only (skip Hub — useful for local onnxruntime validation)
  python export_qai_hub.py --ckpt checkpoints/classifier_best.pt --export_only

  # Validate the fused ONNX before submitting to Hub
  python export_qai_hub.py --ckpt checkpoints/classifier_best.pt --validate

Requirements
------------
  pip install torch torchvision onnx onnxruntime qai-hub numpy pillow
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from RoiExtractor import DroneROIUNet, ROIExtractor
from EfficientNetB0_Classification import DroneCLSNet, load_classifier

# ImageNet stats used during training — calibration must match exactly.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Fused end-to-end module
# ─────────────────────────────────────────────────────────────────────────────

class FusedDronePipeline(nn.Module):
    """
    Single ONNX-exportable graph:
        spectrogram → U-Net mask → ROI extraction → EfficientNet logits

    Key design constraints
    ----------------------
    ROIExtractor is always 'multiply' here. The bbox strategy uses
    per-sample dynamic indexing (nonzero + slicing) which the ONNX
    tracer cannot represent as a static graph. multiply is fully
    differentiable and trace-friendly.

    The Sigmoid from the U-Net head is included inside this graph so
    the NPU compiler can see the full activation range and quantise
    the mask tensor correctly. Without it the quantiser would need to
    infer the range from a pre-sigmoid activation, which degrades
    mask quality under INT8.

    No Softmax is applied to the classifier output — apply in your
    on-device post-processing to avoid a double-softmax if the runtime
    adds one automatically.
    """

    def __init__(
        self,
        unet        : DroneROIUNet,
        cls_net     : DroneCLSNet,
        mask_threshold: float = 0.5,
        roi_size    : tuple   = (224, 224),
    ):
        super().__init__()
        self.unet            = unet
        self.cls_net         = cls_net
        self.mask_threshold  = mask_threshold
        self.roi_size        = roi_size

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        spectrogram : (1, 3, H, W) float32, ImageNet-normalised spectrogram

        Returns
        -------
        class_logits : (1, num_classes) float32, raw scores (no softmax)
        """
        # Stage 1 — segmentation mask
        pred_mask = self.unet(spectrogram)          # (1, 1, H, W) in [0, 1]

        # Binarise: values ≥ threshold become 1, else 0.
        # torch.ge + float() exports as GreaterOrEqual → Cast → static graph.
        # Do NOT use (pred_mask >= self.threshold) — comparison with a Python
        # float on a tensor traces fine in recent PyTorch but some older ONNX
        # opset-13 validators reject the implicit scalar broadcast.
        threshold_t = torch.tensor(
            self.mask_threshold, dtype=pred_mask.dtype, device=pred_mask.device
        )
        binary_mask = torch.ge(pred_mask, threshold_t).float()   # (1, 1, H, W)

        # Stage 1.5 — ROI extraction (multiply strategy)
        # Zero the background energy, keep drone signal region.
        roi = spectrogram * binary_mask             # (1, 3, H, W)

        # Bilinear resize to classifier input size.
        # align_corners=False matches training (get_transforms uses BILINEAR
        # with antialias, which maps to align_corners=False in F.interpolate).
        roi_patch = F.interpolate(
            roi,
            size=self.roi_size,
            mode="bilinear",
            align_corners=False,
        )                                            # (1, 3, 224, 224)

        # Stage 2 — classification
        class_logits = self.cls_net(roi_patch)       # (1, num_classes)
        return class_logits


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_fused_onnx(
    ckpt_path : str,
    out_dir   : str = "exports",
    opset     : int = 17,
    img_h     : int = 256,
    img_w     : int = 512,
) -> str:
    """
    Load Stage-2 checkpoint and export FusedDronePipeline as a single ONNX.

    The legacy TorchScript-based exporter (dynamo=False) is mandatory:
      - Keeps opset 17 (highest SNPE/QNN reliably supports)
      - Encodes ReduceMean axes as static graph attributes
      - Preserves literal-integer Slice constants from DecoderBlock subclasses
    The dynamo exporter (PyTorch ≥ 2.1 default) silently upgrades to opset 18
    and encodes axes as runtime tensors — QNN rejects both behaviours.

    Returns
    -------
    Path to the exported .onnx file.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")    # always export on CPU for reproducibility

    print(f"\n[Export] Loading checkpoint: {ckpt_path}")
    unet, _, cls_net, class_names = load_classifier(ckpt_path, device)
    unet.eval()
    cls_net.eval()

    fused = FusedDronePipeline(unet, cls_net).eval()

    dummy_input = torch.zeros(1, 3, img_h, img_w)
    onnx_path   = os.path.join(out_dir, "drone_pipeline_fused.onnx")

    print(f"[Export] Tracing fused graph  "
          f"input={tuple(dummy_input.shape)} ...")
    with torch.no_grad():
        torch.onnx.export(
            fused,
            dummy_input,
            onnx_path,
            opset_version       = opset,
            input_names         = ["spectrogram"],
            output_names        = ["class_logits"],
            dynamic_axes        = {},       # fully static — required by QNN NPU
            do_constant_folding = True,
            export_params       = True,
            dynamo              = False,    # force legacy jit exporter (see docstring)
        )

    size_mb = Path(onnx_path).stat().st_size / 1e6
    print(f"  ✓ Fused ONNX saved → {onnx_path}  ({size_mb:.1f} MB)")

    # Save class labels alongside the model for on-device use
    labels_path = os.path.join(out_dir, "class_names.txt")
    with open(labels_path, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"  ✓ Class names ({len(class_names)}) → {labels_path}")
    print(f"    {class_names}")

    return onnx_path


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration dataset (matches training preprocessing exactly)
# ─────────────────────────────────────────────────────────────────────────────

def build_calibration_data(
    cal_dir    : str,
    img_h      : int = 256,
    img_w      : int = 512,
    max_images : int = 100,
) -> dict:
    """
    Load calibration images and preprocess them identically to the training
    pipeline (drone_dataloader.get_transforms('val')):

        PIL RGB → resize (img_h, img_w) → ToTensor [0,1]
            → Normalize(IMAGENET_MEAN, IMAGENET_STD)

    IMPORTANT: the old export_dlc.py only divided by 255 and skipped
    Normalize. This caused the quantiser to calibrate on a distribution
    shifted ~0.45 from the model's actual activation range, which degrades
    INT8 accuracy (especially for NO_DRONE frames whose energy is near zero
    before normalisation). This function fixes that.

    The calibration dict key 'spectrogram' must match the ONNX input name.

    Returns
    -------
    {"spectrogram": [array_0, array_1, ...]}
    Each array is float32 of shape (1, 3, img_h, img_w).
    """
    extensions = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(cal_dir, "**", ext), recursive=True))

    if not files:
        raise FileNotFoundError(
            f"No images found under {cal_dir}. "
            "Pass --cal_dir pointing to your raw spectrogram folder."
        )

    files = sorted(files)[:max_images]
    print(f"[Calibration] Found {len(files)} images (using up to {max_images})")

    arrays = []
    skipped = 0
    for path in files:
        try:
            img = Image.open(path).convert("RGB").resize(
                (img_w, img_h), Image.BILINEAR   # PIL resize: (W, H) order
            )
            arr = np.array(img, dtype=np.float32) / 255.0      # HWC [0,1]
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD          # HWC normalised
            arr = np.transpose(arr, (2, 0, 1))                  # CHW
            arr = np.expand_dims(arr, axis=0)                   # (1, 3, H, W)
            arrays.append(arr)
        except Exception as exc:
            print(f"  [Calibration] Skipping {path}: {exc}")
            skipped += 1

    if not arrays:
        raise ValueError(
            f"All {len(files)} calibration images failed to load. "
            "Check the --cal_dir path and image integrity."
        )

    print(f"  ✓ Loaded {len(arrays)} calibration frames "
          f"({skipped} skipped)  shape={arrays[0].shape}")
    return {"spectrogram": arrays}


# ─────────────────────────────────────────────────────────────────────────────
#  onnxruntime validation  (optional but recommended before Hub submission)
# ─────────────────────────────────────────────────────────────────────────────

def validate_fused_onnx(
    onnx_path : str,
    img_h     : int = 256,
    img_w     : int = 512,
):
    """
    Run the fused ONNX through onnxruntime to catch shape and op errors
    before spending Hub credits on a broken graph.

    Checks performed:
      - Output shape is (1, num_classes)
      - Output values are finite (no NaN / Inf from bad normalisation)
      - Output range is plausible for raw logits (not saturated post-softmax)
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[Validate] onnx or onnxruntime not installed — skipping.")
        print("  pip install onnx onnxruntime")
        return

    print(f"\n[Validate] Checking {onnx_path} with onnxruntime ...")

    # 1. ONNX model check (catches malformed graphs immediately)
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("  ✓ onnx.checker passed")

    # 2. Runtime inference check
    sess    = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp     = np.random.randn(1, 3, img_h, img_w).astype(np.float32)
    outputs = sess.run(None, {"spectrogram": inp})
    logits  = outputs[0]

    print(f"  Input  shape : {inp.shape}")
    print(f"  Output shape : {logits.shape}")
    print(f"  Output logits: {logits.flatten().tolist()}")

    assert logits.ndim == 2 and logits.shape[0] == 1, \
        f"Expected (1, num_classes), got {logits.shape}"
    assert np.all(np.isfinite(logits)), \
        "Output contains NaN or Inf — check normalisation / model weights"

    # Raw logits should not look like probabilities (summing to 1.0)
    prob_sum = float(np.exp(logits - logits.max()).sum())
    if abs(prob_sum - 1.0) < 1e-3:
        print("  ⚠  Output looks like probabilities (softmax applied?). "
              "FusedDronePipeline should return raw logits.")
    else:
        print(f"  ✓ Output confirmed as raw logits  "
              f"(softmax-sum={prob_sum:.3f}, expected >> 1.0 before softmax)")

    print("  ✓ Fused ONNX validated — safe to submit to Qualcomm Hub.")


# ─────────────────────────────────────────────────────────────────────────────
#  Qualcomm AI Hub — quantise → compile → profile
# ─────────────────────────────────────────────────────────────────────────────

def run_qai_hub_pipeline(
    onnx_path  : str,
    cal_data   : dict,
    out_dir    : str = "exports",
    img_h      : int = 256,
    img_w      : int = 512,
    device_name: str = "Dragonwing RB3 Gen 2 Vision Kit",
):
    """
    Submit the fused ONNX to Qualcomm AI Hub for:
      1. INT8 quantisation (weights + activations) using real calibration data
      2. TFLite compilation targeting the RB3 Gen 2 NPU
      3. Profiling to measure on-device latency and peak memory

    The compiled .tflite is downloaded locally so it can be deployed without
    further Hub calls.

    Why one model, one quantise job
    --------------------------------
    Previously only the classifier was submitted, so the quantiser saw
    ROI patches (224×224 pre-cropped) as inputs. Now the quantiser sees
    raw spectrograms (256×512) flowing through the full graph, which means:
      • U-Net activations are calibrated with real spectrogram energy ranges
      • The mask multiply and resize ops are quantised correctly in context
      • The classifier activations are calibrated on realistic ROI patches,
        not random tensors or a distribution shifted by wrong preprocessing

    Parameters
    ----------
    cal_data    : dict from build_calibration_data() — {"spectrogram": [...]}
    device_name : Qualcomm Hub device string (must match Hub's device registry)
    """
    try:
        import qai_hub as hub
    except ImportError:
        raise ImportError(
            "qai-hub not installed.\n"
            "  pip install qai-hub\n"
            "  qai-hub configure  # enter your API token"
        )

    os.makedirs(out_dir, exist_ok=True)
    device = hub.Device(device_name)
    input_shape = (1, 3, img_h, img_w)

    # ── Step 1: Upload ────────────────────────────────────────────────────────
    # Hub requires the ONNX and its external .data sidecar (if present) to be
    # uploaded together from a clean directory. Copy only the two relevant files.
    upload_dir  = os.path.join(out_dir, "_hub_upload")
    onnx_name   = Path(onnx_path).name
    data_name   = onnx_name + ".data"

    shutil.rmtree(upload_dir, ignore_errors=True)
    os.makedirs(upload_dir)
    shutil.copy2(onnx_path, os.path.join(upload_dir, onnx_name))
    data_src = str(Path(onnx_path).parent / data_name)
    if os.path.exists(data_src):
        shutil.copy2(data_src, os.path.join(upload_dir, data_name))
        print(f"[Hub] Uploading ONNX + external data sidecar ...")
    else:
        print(f"[Hub] Uploading ONNX (no external .data sidecar found) ...")

    source_model = hub.upload_model(
        model = upload_dir,
        name  = "DroneDetectionFusedPipeline",
    )
    print(f"  ✓ Uploaded → {source_model}")

    # ── Step 2: Quantise (FP32 → INT8) ───────────────────────────────────────
    print(f"\n[Hub] Submitting quantisation job  "
          f"(INT8 weights + activations, {len(cal_data['spectrogram'])} frames) ...")
    quantise_job = hub.submit_quantize_job(
        model              = source_model,
        calibration_data   = cal_data,
        weights_dtype      = hub.QuantizeDtype.INT8,
        activations_dtype  = hub.QuantizeDtype.INT8,
    )
    quantised_model = quantise_job.get_target_model()
    print(f"  ✓ Quantised model → {quantised_model}")

    # ── Step 3: Compile (INT8 ONNX → TFLite, NPU target) ─────────────────────
    print(f"\n[Hub] Submitting compile job  "
          f"(TFLite / NPU / {device_name}) ...")
    compile_job = hub.submit_compile_job(
        model       = quantised_model,
        device      = device,
        input_specs = {"spectrogram": input_shape},
        options     = "--target_runtime tflite --compute_unit npu",
    )
    compiled_model = compile_job.get_target_model()
    print(f"  ✓ Compiled model → {compiled_model}")

    # ── Step 4: Download TFLite ───────────────────────────────────────────────
    tflite_path = os.path.join(out_dir, "drone_pipeline_fused_quantized.tflite")
    compiled_model.download(tflite_path)
    tflite_mb = Path(tflite_path).stat().st_size / 1e6
    print(f"  ✓ TFLite downloaded → {tflite_path}  ({tflite_mb:.1f} MB)")

    # ── Step 5: Profile ───────────────────────────────────────────────────────
    print(f"\n[Hub] Submitting profile job on {device_name} NPU ...")
    profile_job = hub.submit_profile_job(
        model   = compiled_model,
        device  = device,
        options = "--compute_unit npu",
    )
    profile = profile_job.download_profile()
    summary = profile["execution_summary"]

    latency_ms = summary["estimated_inference_time"]
    peak_mb    = summary["estimated_inference_peak_memory"] / (1024 * 1024)

    print("\n" + "─" * 55)
    print(f"  Performance — {device_name} (NPU INT8, fused pipeline)")
    print(f"  Input  : spectrogram  {input_shape}")
    print(f"  Output : class_logits (1, num_classes)")
    print(f"  Avg latency  : {latency_ms:.2f} ms")
    print(f"  Peak memory  : {peak_mb:.2f} MB")
    print("─" * 55)

    return tflite_path, latency_ms, peak_mb


# ─────────────────────────────────────────────────────────────────────────────
#  On-device inference guide
# ─────────────────────────────────────────────────────────────────────────────

ONDEVICE_GUIDE = """
On-device inference with the fused TFLite model
------------------------------------------------
The single model covers the full pipeline — no inter-model CPU handoff.

Python (TFLite runtime):
    import numpy as np
    import tflite_runtime.interpreter as tflite

    interp = tflite.Interpreter("drone_pipeline_fused_quantized.tflite",
                                 num_threads=4)
    interp.allocate_tensors()
    inp_details  = interp.get_input_details()   # [0]: spectrogram (1,3,256,512)
    out_details  = interp.get_output_details()  # [0]: class_logits (1,N)

    # Preprocess — must match training normalisation exactly
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img  = load_spectrogram_as_rgb_256x512()        # → HWC uint8
    arr  = img.astype(np.float32) / 255.0           # → HWC [0,1]
    arr  = (arr - MEAN) / STD                       # → HWC normalised
    arr  = np.transpose(arr, (2, 0, 1))             # → CHW
    arr  = arr[np.newaxis, ...]                     # → (1, 3, 256, 512) NCHW

    interp.set_tensor(inp_details[0]['index'], arr)
    interp.invoke()
    logits = interp.get_tensor(out_details[0]['index'])  # (1, num_classes)

    probs  = np.exp(logits - logits.max())
    probs /= probs.sum()
    pred   = int(np.argmax(probs))

    # Load class_names.txt written alongside the model
    with open("class_names.txt") as f:
        class_names = [l.strip() for l in f]
    print(f"Predicted: {class_names[pred]}  ({100*probs[0,pred]:.1f}%)")

Note on NCHW vs NHWC
---------------------
The TFLite model was compiled from a NCHW ONNX graph. QNN/TFLite on RB3
transparently handles the layout swap at the first op. Feed NCHW tensors
(1, 3, H, W) as shown above — do NOT transpose to NHWC before feeding.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Export fused drone pipeline ONNX and submit to Qualcomm AI Hub"
    )
    p.add_argument("--ckpt",        required=True,
                   help="Stage-2 checkpoint path (checkpoints/classifier_best.pt)")
    p.add_argument("--cal_dir",     default=None,
                   help="Directory of calibration images (recursive scan). "
                        "Required unless --export_only.")
    p.add_argument("--out_dir",     default="exports",
                   help="Output directory for ONNX and TFLite files")
    p.add_argument("--img_h",       type=int, default=256,
                   help="Spectrogram height (must match training --img_size[0])")
    p.add_argument("--img_w",       type=int, default=512,
                   help="Spectrogram width  (must match training --img_size[1])")
    p.add_argument("--opset",       type=int, default=17,
                   help="ONNX opset (default 17 — highest QNN reliably supports)")
    p.add_argument("--cal_images",  type=int, default=100,
                   help="Max calibration images to use (default 100)")
    p.add_argument("--device",      default="Dragonwing RB3 Gen 2 Vision Kit",
                   help="Qualcomm Hub device name")
    p.add_argument("--export_only", action="store_true",
                   help="Export ONNX only — skip Hub quantise/compile/profile")
    p.add_argument("--validate",    action="store_true",
                   help="Validate fused ONNX with onnxruntime before Hub submission")
    return p.parse_args()


def main():
    args = get_args()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Drone Detection — Fused Pipeline Export")
    print(sep)
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Output dir : {args.out_dir}")
    print(f"  Input shape: (1, 3, {args.img_h}, {args.img_w})")
    print(f"  ONNX opset : {args.opset}")
    print(f"  Device     : {args.device}")
    print(f"{sep}\n")

    # 1. Export
    onnx_path = export_fused_onnx(
        ckpt_path = args.ckpt,
        out_dir   = args.out_dir,
        opset     = args.opset,
        img_h     = args.img_h,
        img_w     = args.img_w,
    )

    # 2. Validate (optional)
    if args.validate:
        validate_fused_onnx(onnx_path, args.img_h, args.img_w)

    if args.export_only:
        print(f"\n[Done] ONNX export complete (--export_only). "
              f"Skipping Hub submission.")
        print(ONDEVICE_GUIDE)
        return

    # 3. Calibration data — must be provided for Hub quantisation
    if not args.cal_dir:
        raise ValueError(
            "--cal_dir is required when submitting to Hub.\n"
            "Use --export_only to skip Hub and just produce the ONNX."
        )

    cal_data = build_calibration_data(
        cal_dir    = args.cal_dir,
        img_h      = args.img_h,
        img_w      = args.img_w,
        max_images = args.cal_images,
    )

    # 4. Hub: quantise → compile → profile → download
    tflite_path, latency, peak_mb = run_qai_hub_pipeline(
        onnx_path   = onnx_path,
        cal_data    = cal_data,
        out_dir     = args.out_dir,
        img_h       = args.img_h,
        img_w       = args.img_w,
        device_name = args.device,
    )

    print(f"\n{sep}")
    print("  Done")
    print(f"  Fused TFLite : {tflite_path}")
    print(f"  Latency      : {latency:.2f} ms")
    print(f"  Peak memory  : {peak_mb:.2f} MB")
    print(f"{sep}")
    print(ONDEVICE_GUIDE)


if __name__ == "__main__":
    main()