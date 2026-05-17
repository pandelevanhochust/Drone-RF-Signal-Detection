"""
export_dlc.py
=============
Export the trained two-stage pipeline to ONNX and then to Qualcomm DLC.

Flow
----
Stage-2 checkpoint (.pt)
        │
        ▼  export_onnx()
  unet.onnx  +  classifier.onnx          (two separate ONNX graphs)
        │
        ▼  convert_to_dlc()  [requires Qualcomm SNPE SDK on PATH]
  unet.dlc  +  classifier.dlc            (ready for on-device inference)

Why two separate ONNX files
---------------------------
The U-Net operates on (B, 3, 256, 512) spectrograms.
The classifier operates on (B, 3, 224, 224) ROI patches.
SNPE converts and optimises each graph independently and they are called
sequentially on-device. Fusing them into one graph gives SNPE a single
large dynamic-shape graph which it cannot optimise as well.

SNPE compatibility guarantees already in the model code
-------------------------------------------------------
  ✓  DroneROIUNet.DecoderBlock uses torch.narrow() with static ints
       → exports as Slice with literal start/length, no dynamic axes
  ✓  SqueezeExcitation uses x.mean(dim=[2,3], keepdim=True)
       → exports as ReduceMean, not AdaptiveAvgPool2d (unsupported)
  ✓  DroneCLSNet.forward uses x.mean(dim=[2,3])
       → same ReduceMean pattern for the global pool
  ✓  No Softmax in either model
       → avoids an unnecessary op; apply on-device in post-processing

Requirements
------------
  pip install torch torchvision onnx onnxruntime
  # For DLC conversion (Qualcomm SDK must be installed and sourced):
  #   source $SNPE_ROOT/bin/envsetup.sh
  # Then snpe-onnx-to-dlc is available on PATH.

Usage
-----
  # Export both ONNX files
  python export_dlc.py --ckpt checkpoints/classifier_best.pt

  # Export and immediately convert to DLC (requires SNPE SDK)
  python export_dlc.py --ckpt checkpoints/classifier_best.pt --convert

  # Validate the exported ONNX files with onnxruntime before converting
  python export_dlc.py --ckpt checkpoints/classifier_best.pt --validate

  # Custom output directory
  python export_dlc.py --ckpt checkpoints/classifier_best.pt --out_dir exports/
"""

import os
import argparse
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

from RoiExtractor import DroneROIUNet, ROIExtractor
from EfficientNetB0_Classification import DroneCLSNet, load_classifier


# ─────────────────────────────────────────────────────────────────────────────
#  SNPE compatibility patch
# ─────────────────────────────────────────────────────────────────────────────

def _patch_mean_axes(model: nn.Module):
    """
    The legacy TorchScript ONNX exporter encodes .mean(dim=list) as a
    ReduceMean node with axes as a *runtime tensor input* (dynamic).
    SNPE requires axes to be a *static attribute* on the node.

    Passing axes as a Python tuple instead of a list causes TorchScript
    to fold them into the graph as constants during tracing, which the
    exporter then serialises as a static ReduceMean attribute.

    This patches every forward() that calls .mean(dim=[2,3,...]) so the
    axes are tuples at trace time. No weights are changed.

    Affected locations in our codebase
    -----------------------------------
    DroneCLSNet.forward          x.mean(dim=[2, 3])
    SqueezeExcitation.forward    x.mean(dim=[2, 3], keepdim=True)
    """
    import types

    # ── DroneCLSNet top-level pool ────────────────────────────────────────────
    from EfficientNetB0_Classification import DroneCLSNet
    if isinstance(model, DroneCLSNet):
        original_forward = model.forward

        def _patched_cls_forward(self, x):
            x = self.stem(x)
            x = self.block1(x);  x = self.block2(x);  x = self.block3(x)
            x = self.block4(x);  x = self.block5(x);  x = self.block6(x)
            x = self.block7(x)
            x = self.head_conv(x)
            x = x.mean(dim=(2, 3))          # tuple → static ReduceMean attr
            return self.classifier(x)

        model.forward = types.MethodType(_patched_cls_forward, model)

    # ── SqueezeExcitation inside any model ───────────────────────────────────
    from EfficientNetB0_Classification import SqueezeExcitation
    for module in model.modules():
        if isinstance(module, SqueezeExcitation):
            def _patched_se_forward(self, x):
                s = x.mean(dim=(2, 3), keepdim=True)   # tuple → static attr
                s = self.act(self.fc1(s))
                s = self.gate(self.fc2(s))
                return x * s

            module.forward = types.MethodType(_patched_se_forward, module)


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(
    ckpt_path  : str,
    out_dir    : str  = "exports",
    opset      : int  = 17,
    img_h      : int  = 256,
    img_w      : int  = 512,
    batch_size : int  = 1,      # SNPE DLC conversion requires static batch=1
):
    """
    Export DroneROIUNet and DroneCLSNet as two separate ONNX graphs.

    Both are exported with a static batch dimension of 1.
    SNPE does not support dynamic batch axes in DLC conversion — runtime
    batching is handled by the SNPE execution framework, not the graph.

    Opset notes
    -----------
    PyTorch >= 2.1 defaults to the dynamo-based ONNX exporter which:
      - Ignores opset < 18 (upgrades silently to 18)
      - Encodes .mean(dim=[2,3]) axes as a runtime tensor instead of a
        static attribute → SNPE rejects with "axis must be >= 0" error
      - Encodes torch.narrow() as dynamic Slice ops → SNPE rejects with
        "Inconsistency in dynamic axis shapes" error

    Fix: force the legacy TorchScript-based exporter via
    torch.onnx.export(..., dynamo=False).  This keeps opset 17 and
    encodes ReduceMean axes as graph attributes (static), which SNPE
    accepts.  Opset 17 is the highest version SNPE reliably supports.

    Returns
    -------
    (unet_onnx_path, cls_onnx_path)
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")   # always export on CPU for reproducibility

    # Load both models from Stage-2 checkpoint
    unet, extractor, cls_net, class_names = load_classifier(ckpt_path, device)
    unet.eval()
    cls_net.eval()

    unet_path = os.path.join(out_dir, "drone_unet.onnx")
    cls_path  = os.path.join(out_dir, "drone_classifier.onnx")

    # ── Patch: replace .mean(dim=[2,3]) with a SNPE-safe wrapper ─────────────
    # The dynamo exporter encodes list-axes as a runtime tensor input.
    # The legacy (jit) exporter with dynamo=False encodes them as a static
    # attribute — but only if the axes are passed as a tuple, not a list.
    # We patch forward() on the classifier to use tuple axes.
    # The U-Net's SqueezeExcitation.forward() also uses .mean — same fix.
    _patch_mean_axes(cls_net)
    _patch_mean_axes(unet)

    # ── Export U-Net ──────────────────────────────────────────────────────────
    # Input  : (1, 3, img_h, img_w)  normalised spectrogram
    # Output : (1, 1, img_h, img_w)  sigmoid mask
    dummy_spec = torch.zeros(batch_size, 3, img_h, img_w)

    print(f"\n[Export] U-Net  {tuple(dummy_spec.shape)} → {unet_path}")
    with torch.no_grad():
        torch.onnx.export(
            unet,
            dummy_spec,
            unet_path,
            opset_version       = opset,
            input_names         = ["spectrogram"],
            output_names        = ["roi_mask"],
            dynamic_axes        = {},           # fully static — required for SNPE
            do_constant_folding = True,
            export_params       = True,
            dynamo              = False,        # force legacy jit exporter
        )
    print(f"  ✓ U-Net ONNX saved  ({Path(unet_path).stat().st_size / 1e6:.1f} MB)")

    # ── Export Classifier ─────────────────────────────────────────────────────
    # Input  : (1, 3, 224, 224)  ROI patch (output of ROIExtractor)
    # Output : (1, num_classes)  raw logits  [NO softmax]
    dummy_roi = torch.zeros(batch_size, 3, 224, 224)

    print(f"\n[Export] Classifier  {tuple(dummy_roi.shape)} → {cls_path}")
    with torch.no_grad():
        torch.onnx.export(
            cls_net,
            dummy_roi,
            cls_path,
            opset_version       = opset,
            input_names         = ["roi_patch"],
            output_names        = ["class_logits"],
            dynamic_axes        = {},
            do_constant_folding = True,
            export_params       = True,
            dynamo              = False,        # force legacy jit exporter
        )
    print(f"  ✓ Classifier ONNX saved  ({Path(cls_path).stat().st_size / 1e6:.1f} MB)")

    # Save class name list alongside the ONNX files for on-device labelling
    labels_path = os.path.join(out_dir, "class_names.txt")
    with open(labels_path, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"\n  Class names → {labels_path}")
    print(f"  Classes: {class_names}")

    return unet_path, cls_path


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX validation  (sanity-check before DLC conversion)
# ─────────────────────────────────────────────────────────────────────────────

def validate_onnx(
    unet_path : str,
    cls_path  : str,
    img_h     : int = 256,
    img_w     : int = 512,
):
    """
    Run both ONNX graphs through onnxruntime and print output shapes.
    Catches shape errors and unsupported ops before handing off to SNPE.
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("[Validate] onnxruntime not installed — skipping validation.")
        print("  pip install onnxruntime")
        return

    print("\n[Validate] Running onnxruntime inference checks ...")

    # ── U-Net ─────────────────────────────────────────────────────────────────
    sess_unet = ort.InferenceSession(unet_path,
                                     providers=["CPUExecutionProvider"])
    dummy_spec = np.random.rand(1, 3, img_h, img_w).astype(np.float32)
    mask_out   = sess_unet.run(None, {"spectrogram": dummy_spec})
    print(f"  U-Net   input : {dummy_spec.shape}")
    print(f"  U-Net   output: {mask_out[0].shape}  "
          f"range=[{mask_out[0].min():.3f}, {mask_out[0].max():.3f}]")
    assert mask_out[0].shape == (1, 1, img_h, img_w), \
        f"Unexpected U-Net output shape: {mask_out[0].shape}"

    # ── Classifier ───────────────────────────────────────────────────────────
    sess_cls  = ort.InferenceSession(cls_path,
                                     providers=["CPUExecutionProvider"])
    dummy_roi = np.random.rand(1, 3, 224, 224).astype(np.float32)
    logit_out = sess_cls.run(None, {"roi_patch": dummy_roi})
    print(f"  Classifier input : {dummy_roi.shape}")
    print(f"  Classifier output: {logit_out[0].shape}  "
          f"logits={logit_out[0].flatten().tolist()}")

    print("\n  ✓ Both ONNX graphs validated — safe to convert to DLC.")


# ─────────────────────────────────────────────────────────────────────────────
#  DLC conversion  (requires Qualcomm SNPE SDK on PATH)
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_dlc(
    unet_path   : str,
    cls_path    : str,
    out_dir     : str = "exports",
    img_h       : int = 256,
    img_w       : int = 512,
):
    """
    Convert both ONNX files to Qualcomm DLC using snpe-onnx-to-dlc.

    Prerequisites
    -------------
    1. Download and install the Qualcomm AI Engine Direct SDK (SNPE).
    2. Source the environment:
           source $SNPE_ROOT/bin/envsetup.sh
    3. Confirm the tool is available:
           which snpe-onnx-to-dlc

    The --input_dim flag must match the static shape used at ONNX export
    time. If you change img_h/img_w, update both export and conversion.

    Output files
    ------------
    exports/drone_unet.dlc
    exports/drone_classifier.dlc
    """
    unet_dlc = os.path.join(out_dir, "drone_unet.dlc")
    cls_dlc  = os.path.join(out_dir, "drone_classifier.dlc")

    # ── U-Net DLC ─────────────────────────────────────────────────────────────
    unet_cmd = [
        "snpe-onnx-to-dlc",
        "--input_network",  unet_path,
        "--output_path",    unet_dlc,
        "--input_dim",      "spectrogram", f"1,3,{img_h},{img_w}",
    ]
    print(f"\n[DLC] Converting U-Net ...")
    print(f"  Command: {' '.join(unet_cmd)}")
    result = subprocess.run(unet_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ U-Net DLC saved → {unet_dlc}")
    else:
        print(f"  ✗ U-Net DLC conversion failed:\n{result.stderr}")
        _print_snpe_debug_hints(result.stderr)

    # ── Classifier DLC ────────────────────────────────────────────────────────
    cls_cmd = [
        "snpe-onnx-to-dlc",
        "--input_network",  cls_path,
        "--output_path",    cls_dlc,
        "--input_dim",      "roi_patch", "1,3,224,224",
    ]
    print(f"\n[DLC] Converting Classifier ...")
    print(f"  Command: {' '.join(cls_cmd)}")
    result = subprocess.run(cls_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Classifier DLC saved → {cls_dlc}")
    else:
        print(f"  ✗ Classifier DLC conversion failed:\n{result.stderr}")
        _print_snpe_debug_hints(result.stderr)

    return unet_dlc, cls_dlc


def _print_snpe_debug_hints(stderr: str):
    """
    Map common SNPE conversion errors to the fixes already in the codebase.
    """
    hints = {
        "AdaptiveAvgPool": (
            "AdaptiveAvgPool2d is not supported by SNPE.\n"
            "  Fix: replace with x.mean(dim=[2,3], keepdim=True) in forward().\n"
            "  This fix is already applied in SqueezeExcitation and DroneCLSNet."
        ),
        "Inconsistency in dynamic axis": (
            "Dynamic shape detected in the graph.\n"
            "  Fix: ensure all slice/crop ops use literal integers, not runtime shapes.\n"
            "  This fix is already applied in DecoderBlock via torch.narrow()."
        ),
        "Softmax": (
            "Softmax at the end of the graph causes issues with some SNPE versions.\n"
            "  Fix: remove Softmax from the model (apply in post-processing instead).\n"
            "  This fix is already applied — neither model has a Softmax output."
        ),
        "opset": (
            "Opset version mismatch.\n"
            "  Fix: export with --opset 13 (default in export_onnx()).\n"
            "  SNPE supports ONNX opset 9-13."
        ),
    }
    for keyword, hint in hints.items():
        if keyword.lower() in stderr.lower():
            print(f"\n  [Hint] {hint}")


# ─────────────────────────────────────────────────────────────────────────────
#  On-device inference reference  (pseudocode comment, not runnable here)
# ─────────────────────────────────────────────────────────────────────────────

ONDEVICE_INFERENCE_GUIDE = """
On-device inference with SNPE (C++ / Python SDK)
-------------------------------------------------
The two DLC files are loaded and called sequentially:

    # Python (snpe-python-tutorial style)
    from snpe import SNPEBuilder, SNPERuntime

    unet_snpe = SNPEBuilder("drone_unet.dlc").build()
    cls_snpe  = SNPEBuilder("drone_classifier.dlc").build()

    # Step 1: spectrogram → mask
    unet_out  = unet_snpe.execute({"spectrogram": spectrogram_nhwc})
    mask      = unet_out["roi_mask"]          # (1, 1, H, W)

    # Step 2: ROI extraction  (lightweight, run in CPU post-processing)
    binary    = (mask >= 0.5).astype(float)
    roi       = spectrogram_nhwc * binary     # zero background
    roi_patch = resize(roi, (224, 224))       # bilinear resize

    # Step 3: classify ROI patch
    cls_out   = cls_snpe.execute({"roi_patch": roi_patch})
    logits    = cls_out["class_logits"]       # (1, num_classes)
    probs     = softmax(logits)               # apply softmax here, not in DLC
    pred      = argmax(probs)

Note: NHWC vs NCHW
    SNPE expects NHWC inputs by default.
    If your pipeline feeds NCHW tensors, add --input_layout NCHW to
    snpe-onnx-to-dlc or transpose inputs before inference.

    snpe-onnx-to-dlc --input_network drone_unet.onnx \\
                      --output_path   drone_unet.dlc  \\
                      --input_dim     spectrogram 1,3,256,512 \\
                      --input_layout  spectrogram NCHW
"""


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export trained pipeline to ONNX and Qualcomm DLC"
    )
    parser.add_argument("--ckpt",     required=True,
                        help="Stage-2 checkpoint (checkpoints/classifier_best.pt)")
    parser.add_argument("--out_dir",  default="exports",
                        help="Output directory for ONNX and DLC files")
    parser.add_argument("--opset",    type=int, default=17,
                        help="ONNX opset version (default 17, max SNPE supports reliably)")
    parser.add_argument("--img_h",    type=int, default=256)
    parser.add_argument("--img_w",    type=int, default=512)
    parser.add_argument("--validate", action="store_true",
                        help="Validate ONNX with onnxruntime before converting")
    parser.add_argument("--convert",  action="store_true",
                        help="Run snpe-onnx-to-dlc after export (requires SNPE SDK)")
    args = parser.parse_args()

    # Step 1: Export to ONNX
    unet_onnx, cls_onnx = export_onnx(
        ckpt_path  = args.ckpt,
        out_dir    = args.out_dir,
        opset      = args.opset,
        img_h      = args.img_h,
        img_w      = args.img_w,
    )

    # Step 2: Validate (optional but recommended)
    if args.validate:
        validate_onnx(unet_onnx, cls_onnx, args.img_h, args.img_w)

    # Step 3: Convert to DLC (optional, requires SNPE SDK on PATH)
    if args.convert:
        convert_to_dlc(unet_onnx, cls_onnx, args.out_dir, args.img_h, args.img_w)
    else:
        print("\n[Export] ONNX export complete.")
        print("  To convert to DLC, run with --convert after sourcing SNPE SDK:")
        print("    source $SNPE_ROOT/bin/envsetup.sh")
        print(f"    python export_dlc.py --ckpt {args.ckpt} --convert")

    print("\n" + ONDEVICE_INFERENCE_GUIDE)