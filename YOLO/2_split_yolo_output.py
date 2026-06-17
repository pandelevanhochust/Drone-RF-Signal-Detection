"""
split_yolo_output.py
=====================
Ultralytics YOLOv8 ONNX export produces ONE merged output tensor:
    output: [1, 4 + num_classes, 8400]   (transposed: channels-first)

The C++ YoloQNN class (yolo_qnn.cpp) expects TWO separate outputs:
    boxes:  [1, 8400, 4]
    scores: [1, 8400, num_classes]

This script inserts Split + Transpose nodes into the ONNX graph so the
QNN converter emits a context binary with two clean output tensors,
matching yolo_qnn.h's `setup_tensors()`.

Run on: x86 host
Requires: pip install onnx onnx-graphsurgeon
"""

import argparse

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def split_yolo_output(input_path: str, output_path: str, num_classes: int = 80):
    print(f"[1/3] Loading {input_path} ...")
    graph = gs.import_onnx(onnx.load(input_path))

    # Find the single merged output (Ultralytics names it "output0")
    merged = graph.outputs[0]
    print(f"    Merged output: {merged.name}  shape={merged.shape}")
    # Expected shape: [1, 4+num_classes, num_anchors]  e.g. [1, 84, 8400]

    num_anchors = merged.shape[-1]
    print(f"    num_anchors={num_anchors}  num_classes={num_classes}")

    # ── Transpose [1, 84, 8400] → [1, 8400, 84] ──────────────────────────────
    transposed = gs.Variable(
        name="output_transposed",
        dtype=np.float32,
        shape=(1, num_anchors, 4 + num_classes),
    )
    transpose_node = gs.Node(
        op="Transpose",
        name="transpose_output",
        inputs=[merged],
        outputs=[transposed],
        attrs={"perm": [0, 2, 1]},
    )

    # ── Split → boxes [1,8400,4]  +  scores [1,8400,num_classes] ────────────
    boxes = gs.Variable(name="boxes", dtype=np.float32, shape=(1, num_anchors, 4))
    scores_raw = gs.Variable(name="scores_raw", dtype=np.float32, shape=(1, num_anchors, num_classes))

    # 1. Define the split sizes as a Constant tensor (required for Opset 13+)
    split_sizes = gs.Constant(
        name="split_sizes",
        values=np.array([4, num_classes], dtype=np.int64)
    )

    # 2. Update the Split node
    split_node = gs.Node(
        op="Split",
        name="split_boxes_scores",
        # Pass split_sizes as the SECOND input here
        inputs=[transposed, split_sizes],
        outputs=[boxes, scores_raw],
        # Remove the "split" attribute, keep only the "axis"
        attrs={"axis": 2},
    )

    # ── Sigmoid on scores (Ultralytics outputs raw logits pre-sigmoid) ───────
    scores = gs.Variable(name="scores", dtype=np.float32,
                         shape=(1, num_anchors, num_classes))
    sigmoid_node = gs.Node(
        op="Sigmoid",
        name="sigmoid_scores",
        inputs=[scores_raw],
        outputs=[scores],
    )

    graph.nodes.extend([transpose_node, split_node, sigmoid_node])
    graph.outputs = [boxes, scores]

    print("[2/3] Cleaning up graph ...")
    graph.cleanup().toposort()

    print(f"[3/3] Saving → {output_path}")
    onnx.save(gs.export_onnx(graph), output_path)

    # Verify
    onnx.checker.check_model(onnx.load(output_path))
    print(f"\n✓ Done. Outputs: boxes{tuple(boxes.shape)}  scores{tuple(scores.shape)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",   default="yolov8n.onnx")
    p.add_argument("--output",  default="yolov8n_split.onnx")
    p.add_argument("--classes", type=int, default=80)
    args = p.parse_args()

    split_yolo_output(args.input, args.output, args.classes)