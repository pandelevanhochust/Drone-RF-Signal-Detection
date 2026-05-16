"""
export_onnx.py
===============
Convert a trained DronePipeline checkpoint (.pth) to ONNX format.

The exported ONNX graph covers the full pipeline:
    Raw spectrogram (B, 3, H, W)
        -> DroneROIUNet       -> mask  (B, 1, H, W)
        -> ROIExtractor       -> roi   (B, 3, 224, 224)
        -> DroneCLSNet        -> logits (B, num_classes)

Outputs:
    <out_dir>/drone_pipeline.onnx        -- full pipeline
    <out_dir>/drone_pipeline_sim.onnx    -- simplified (if onnxsim installed)

Usage:
    python export_onnx.py --checkpoint checkpoints/best_model.pth

    python export_onnx.py --checkpoint checkpoints/best_model.pth \
                          --out_dir ./onnx_exports \
                          --img_size 256 512 \
                          --batch_size 1 \
                          --simplify

Requirements:
    pip install torch torchvision onnx
    pip install onnxsim          # optional but recommended for simplification
    pip install onnxruntime      # optional for verification
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from roi import DronePipeline


# =============================================================================
# Wrapper: pipeline that returns only logits (ONNX-friendly single output)
# =============================================================================

class DronePipelineONNX(nn.Module):
    """
    Thin wrapper around DronePipeline that returns ONLY logits.
    ONNX export works best with a single tensor output.
    For inspection, mask export is available via --export_mask flag.
    """

    def __init__(self, pipeline: DronePipeline, export_mask: bool = False):
        super().__init__()
        self.pipeline    = pipeline
        self.export_mask = export_mask

    def forward(self, x: torch.Tensor):
        if self.export_mask:
            logits, mask = self.pipeline(x, return_mask=True)
            return logits, mask
        else:
            logits = self.pipeline(x, return_mask=False)
            return logits


# =============================================================================
# Checkpoint key remapping
# =============================================================================

def remap_state_dict(raw_state: dict) -> dict:
    """
    Remap checkpoint keys from the old layer structure to the new SNPE-compatible
    structure after SqueezeExcitation and DroneCLSNet.classifier were refactored.

    Changes:
    --------
    SqueezeExcitation refactor:
        OLD: self.se = nn.Sequential(AdaptiveAvgPool2d, Conv2d, SiLU, Conv2d, Sigmoid)
             keys: ...se.1.weight / se.1.bias / se.3.weight / se.3.bias
        NEW: self.fc1, self.fc2  (separate Conv2d members, GAP via .mean() in forward)
             keys: ...fc1.weight / fc1.bias / fc2.weight / fc2.bias

    DroneCLSNet.classifier refactor:
        OLD: nn.Sequential(AdaptiveAvgPool2d[0], Flatten[1], Dropout[2], Linear[3])
             keys: classifier.classifier.3.weight / .bias
        NEW: nn.Sequential(Dropout[0], Linear[1])  -- GAP via .mean() in forward
             keys: classifier.classifier.1.weight / .bias
    """
    remapped = {}
    for k, v in raw_state.items():
        new_k = k
        # SqueezeExcitation: se.1 -> fc1,  se.3 -> fc2
        new_k = new_k.replace(".se.1.weight", ".fc1.weight")
        new_k = new_k.replace(".se.1.bias",   ".fc1.bias")
        new_k = new_k.replace(".se.3.weight", ".fc2.weight")
        new_k = new_k.replace(".se.3.bias",   ".fc2.bias")
        # Classifier Linear: index 3 -> index 1
        new_k = new_k.replace(
            "classifier.classifier.3.weight",
            "classifier.classifier.1.weight",
        )
        new_k = new_k.replace(
            "classifier.classifier.3.bias",
            "classifier.classifier.1.bias",
        )
        remapped[new_k] = v
    return remapped


# =============================================================================
# Loader
# =============================================================================

def load_pipeline(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load DronePipeline from checkpoint with automatic key remapping.

    Returns:
        model      : DronePipeline in eval mode
        meta       : metadata dict
        train_args : original training args dict
    """
    print(f"  Loading checkpoint : {checkpoint_path}")
    ckpt       = torch.load(checkpoint_path, map_location=device)
    meta       = ckpt.get("meta", {})
    train_args = ckpt.get("args", {})

    num_classes       = meta.get("num_classes", train_args.get("num_classes", 8))
    unet_base_filters = train_args.get("unet_base_filters", 32)
    img_size          = meta.get("img_size", (256, 512))

    print(f"  Epoch             : {ckpt.get('epoch', '?')}")
    print(f"  Best val accuracy : {ckpt.get('best_val_acc', 0.0):.2f}%")
    print(f"  Classes           : {num_classes}  ->  {meta.get('class_names', [])}")
    print(f"  Image size        : {img_size}")
    print(f"  UNet base filters : {unet_base_filters}")

    model = DronePipeline(
        num_classes       = num_classes,
        in_channels       = 3,
        unet_base_filters = unet_base_filters,
        roi_output_size   = (224, 224),
        mask_threshold    = train_args.get("mask_threshold", 0.7),
        roi_strategy      = train_args.get("roi_strategy", "multiply"),
    ).to(device)

    # remap old key names to new SNPE-compatible structure
    remapped = remap_state_dict(ckpt["model_state"])
    result   = model.load_state_dict(remapped, strict=False)

    if result.missing_keys:
        print(f"  [WARN] Missing keys  ({len(result.missing_keys)}): "
              f"{result.missing_keys[:3]} ...")
    if result.unexpected_keys:
        print(f"  [WARN] Unexpected keys ({len(result.unexpected_keys)}): "
              f"{result.unexpected_keys[:3]} ...")
    if not result.missing_keys and not result.unexpected_keys:
        print("  [OK]  All checkpoint keys loaded successfully")

    model.eval()
    return model, meta, train_args


# =============================================================================
# Export
# =============================================================================

def export_onnx(
    checkpoint_path : str,
    out_dir         : str   = ".",
    img_size        : tuple = (256, 512),
    batch_size      : int   = 1,
    export_mask     : bool  = False,
    simplify        : bool  = False,
    opset           : int   = 13,
) -> str:
    """
    Export DronePipeline to ONNX.

    Args:
        checkpoint_path : path to best_model.pth
        out_dir         : output directory for .onnx file
        img_size        : (H, W) input spectrogram size
        batch_size      : static batch size for export (use 1 for deployment)
        export_mask     : if True, export logits + mask as two outputs
        simplify        : run onnx-simplifier after export
        opset           : ONNX opset version (default 13 for SNPE compat)

    Returns:
        path to exported .onnx file
    """
    import onnx

    device = torch.device("cpu")   # always export on CPU for portability
    H, W   = img_size

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load model
    print(f"\n{'='*60}")
    print("  Loading model ...")
    print('='*60)
    pipeline, meta, train_args = load_pipeline(checkpoint_path, device)

    export_model = DronePipelineONNX(pipeline, export_mask=export_mask)
    export_model.eval()

    dummy_input  = torch.randn(batch_size, 3, H, W, device=device)
    output_names = ["logits"]
    if export_mask:
        output_names.append("mask")

    dynamic_axes = {"input": {0: "batch_size"}, "logits": {0: "batch_size"}}
    if export_mask:
        dynamic_axes["mask"] = {0: "batch_size"}

    suffix    = "_with_mask" if export_mask else ""
    onnx_path = out_dir / f"drone_pipeline{suffix}.onnx"

    print(f"\n{'='*60}")
    print("  Exporting to ONNX ...")
    print('='*60)
    print(f"  Input shape  : ({batch_size}, 3, {H}, {W})")
    print(f"  Output names : {output_names}")
    print(f"  Opset        : {opset}")
    print(f"  Output path  : {onnx_path}")

    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            str(onnx_path),
            export_params       = True,
            opset_version       = opset,
            do_constant_folding = True,
            input_names         = ["input"],
            output_names        = output_names,
            dynamic_axes        = dynamic_axes,
        )

    print(f"  [OK] Exported -> {onnx_path}")

    # validate
    print(f"\n{'='*60}")
    print("  Validating ONNX model ...")
    print('='*60)
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  [OK] ONNX model is valid")

    print(f"\n  Inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name:<20} shape={shape}")
    print(f"  Outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name:<20} shape={shape}")

    file_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"\n  File size : {file_mb:.1f} MB")

    # simplify
    sim_path = None
    if simplify:
        print(f"\n{'='*60}")
        print("  Running onnx-simplifier ...")
        print('='*60)
        try:
            from onnxsim import simplify as onnx_simplify
            sim_model, check = onnx_simplify(onnx_model)
            if check:
                sim_path = out_dir / f"drone_pipeline{suffix}_sim.onnx"
                onnx.save(sim_model, str(sim_path))
                sim_mb = sim_path.stat().st_size / 1024 / 1024
                print(f"  [OK] Simplified -> {sim_path}  ({sim_mb:.1f} MB)")
            else:
                print("  [WARN] Simplification check failed -- using original")
        except ImportError:
            print("  [SKIP] onnxsim not installed: pip install onnxsim")

    # verify with onnxruntime
    print(f"\n{'='*60}")
    print("  Running OnnxRuntime inference check ...")
    print('='*60)
    try:
        import onnxruntime as ort
        import numpy as np

        target   = str(sim_path) if sim_path else str(onnx_path)
        sess     = ort.InferenceSession(target, providers=["CPUExecutionProvider"])
        dummy_np = dummy_input.numpy()
        ort_out  = sess.run(None, {"input": dummy_np})

        print(f"  [OK] OnnxRuntime forward pass succeeded")
        print(f"  logits shape : {ort_out[0].shape}")
        if export_mask:
            print(f"  mask shape   : {ort_out[1].shape}")

        with torch.no_grad():
            pt_out = export_model(dummy_input)
        pt_logits = pt_out[0].numpy() if export_mask else pt_out.numpy()
        max_diff  = float(np.abs(ort_out[0] - pt_logits).max())
        status    = "[OK]" if max_diff < 1e-4 else "[WARN] diff > 1e-4"
        print(f"  Max logit diff (PT vs ORT) : {max_diff:.6f}  {status}")

    except ImportError:
        print("  [SKIP] onnxruntime not installed: pip install onnxruntime")

    # summary
    print(f"\n{'='*60}")
    print("  Export complete")
    print('='*60)
    print(f"  ONNX model  : {onnx_path}")
    if sim_path:
        print(f"  Simplified  : {sim_path}")
    print(f"  Class names : {meta.get('class_names', [])}")
    print(f"\n  Next step -- convert to DLC:")
    print(f"  snpe-onnx-to-dlc \\")
    if sim_path:
        print(f"      -i {sim_path} \\")
    else:
        print(f"      -i {onnx_path} \\")
    print(f"      -o checkpoints/drone.dlc \\")
    print(f"      --overwrite_input_shapes input:1,3,{H},{W}")
    print('='*60)

    return str(onnx_path)


# =============================================================================
# CLI
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="Export DronePipeline checkpoint to ONNX."
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to best_model.pth saved by main.py.",
    )
    p.add_argument(
        "--out_dir", default="onnx_exports",
        help="Output directory for .onnx files. Default: ./onnx_exports",
    )
    p.add_argument(
        "--img_size", nargs=2, type=int, default=[256, 512],
        metavar=("H", "W"),
        help="Input spectrogram size. Default: 256 512",
    )
    p.add_argument(
        "--batch_size", type=int, default=1,
        help="Static batch size for export. Default: 1",
    )
    p.add_argument(
        "--export_mask", action="store_true",
        help="Export both logits and U-Net mask as two ONNX outputs.",
    )
    p.add_argument(
        "--simplify", action="store_true",
        help="Run onnx-simplifier (pip install onnxsim).",
    )
    p.add_argument(
        "--opset", type=int, default=13,
        help="ONNX opset version. Default: 13 (best SNPE compatibility).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    export_onnx(
        checkpoint_path = args.checkpoint,
        out_dir         = args.out_dir,
        img_size        = tuple(args.img_size),
        batch_size      = args.batch_size,
        export_mask     = args.export_mask,
        simplify        = args.simplify,
        opset           = args.opset,
    )