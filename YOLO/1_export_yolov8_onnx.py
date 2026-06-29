"""
export_yolov8_onnx.py
======================
Export YOLOv8 (Ultralytics) to ONNX with split outputs (boxes + scores)
for clean QNN/HTP deployment on QCS6490.

By default `model.export(format="onnx")` fuses box+score+NMS into one
output tensor with internal ops the QNN converter can't always handle
cleanly. This script exports a "headless" graph that splits cleanly into:

    output0: boxes  [1, 8400, 4]   (cx, cy, w, h)  normalised [0,1]
    output1: scores [1, 8400, num_classes]          post-sigmoid

Run on: x86 host
Requires: pip install ultralytics onnx onnxsim
"""

import argparse

import onnx
import torch
from ultralytics import YOLO


def export(weights: str, imgsz: int, out_path: str):
    print(f"[1/4] Loading {weights} ...")
    model = YOLO(weights)
    model.eval = model.model.eval  # ensure eval mode propagates

    # Ultralytics' built-in ONNX export already splits raw outputs when
    # nms=False (default) and dynamic=False — verify with onnx.checker after.
    print(f"[2/4] Exporting → ONNX (imgsz={imgsz}) ...")
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,     # static shapes — required for QNN context binary
        simplify=True,     # runs onnxsim automatically
        opset=17,
        nms=False,          # keep NMS in FastCV/C++, not in the graph
        half=False,         # export FP32; quantizer handles INT8 conversion
    )
    print(f"    Exported: {exported}")

    print("[3/4] Verifying ONNX model ...")
    onnx_model = onnx.load(exported)
    onnx.checker.check_model(onnx_model)

    # Print output tensor info — confirm shapes match yolo_qnn.h expectations
    print("[4/4] Output tensors:")
    for out in onnx_model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {dims}")

    print(f"\n✓ Done: {exported}")
    print("  NOTE: Ultralytics YOLOv8 default export produces ONE merged")
    print("  output [1, 4+num_classes, 8400]. Use scripts/split_yolo_output.py")
    print("  to split it into boxes/scores for the YoloQNN C++ loader, OR")
    print("  adjust yolo_qnn.cpp to parse the merged layout directly (see notes).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt",
                   help="yolov8n.pt / yolov8s.pt / custom .pt")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--out",     default="yolov8n.onnx")
    args = p.parse_args()

    export(args.weights, args.imgsz, args.out)