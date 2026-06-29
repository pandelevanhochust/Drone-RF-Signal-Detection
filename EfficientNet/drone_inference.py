"""
drone_inference.py
==================
TFLite inference for the hardware-optimized EfficientNet-B0 binary drone classifier
optimized for execution on the Qualcomm Hexagon NPU (Dragonwing RB3 Gen 2).

Model details
-------------
    Architecture : EfficientNet-B0 (Custom, 100% NPU-compliant)
    Task         : Binary classification — DRONE (0) vs NO_DRONE (1)
    Input name   : image_tensor   shape (1, 3, 256, 512)  float32  [0.0, 1.0]
    Output name  : class_logits   shape (1, 2)             float32  raw logits
    Preprocessing: PIL RGB → resize (256, 512) → ToTensor() [/ 255.0]

Usage
-----
    # 1. Run standalone hardware benchmark test using random tensors:
    python3 drone_inference.py --model exports/efficientvit_l2_drone_quantized.tflite

    # 2. Run inference against a real saved spectrogram image file:
    python3 drone_inference.py --model exports/efficientvit_l2_drone_quantized.tflite --input capture_01.png
"""

import argparse
import os
import time
import numpy as np
from PIL import Image

# Default configuration parameters matching the EfficientNet-B0 structural grid
DEFAULT_IMG_H = 256
DEFAULT_IMG_W = 512
DELEGATE_LIB = "libQnnTFLiteDelegate.so"
CLASS_NAMES_DEFAULT = ["DRONE", "NO_DRONE"]
CONFIDENCE_THRESHOLD = 0.70


class DroneInferencer:
    """
    Wraps the optimized INT8 EfficientNet-B0 TFLite model graph
    for low-latency single-frame evaluation.
    """

    def __init__(
            self,
            model_path: str,
            labels_path: str = "class_names.txt",
            use_npu: bool = True,
            confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Compiled TFLite model asset not found at path: {model_path}\n"
                "Please verify file placement or transfer it from your build system."
            )

        # Parse class names from file or load static target labels
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        else:
            print(f"[Initialization] WARNING: {labels_path} not found — defaulting to standard categories.")
            self.class_names = CLASS_NAMES_DEFAULT

        self.confidence_threshold = confidence_threshold

        # Initialize the underlying acceleration hardware interpreter
        self.interp = self._build_interpreter(model_path, use_npu)

        inp = self.interp.get_input_details()[0]
        out = self.interp.get_output_details()[0]

        self._inp_idx = inp["index"]
        self._out_idx = out["index"]
        self._out_scale = out["quantization"][0]
        self._out_zp = out["quantization"][1]
        self._inp_shape = tuple(int(dim) for dim in inp["shape"])

        print(f"[Initialization] Hardware input map shape : {inp['shape']} | Type: {inp['dtype']}")
        print(
            f"[Initialization] Hardware output map shape: {out['shape']} | Scale: {self._out_scale:.6f}, Zero-Point: {self._out_zp}")
        print(f"[Initialization] Categorization labels    : {self.class_names}")
        print(f"[Initialization] Suppression threshold    : {confidence_threshold}\n")

        # Core Hardware Engine Warmup
        # Forces driver allocation, pointer caching, and kernel initialization before real loops execute.
        dummy_warmup = np.zeros(self._inp_shape, dtype=np.float32)
        self.interp.set_tensor(self._inp_idx, dummy_warmup)
        self.interp.invoke()
        print("[Initialization] Hardware compilation warmup executed successfully.\n")

    def run(self, tensor: np.ndarray) -> dict:
        """
        Pushes a preprocessed float32 image matrix through the hardware runtime.

        Parameters
        ----------
        tensor : float32 ndarray structured as (1, 3, H, W) normalized to [0.0, 1.0]

        Returns
        -------
        dict containing latency metrics, normalized confidences, and target classifications.
        """
        self.interp.set_tensor(self._inp_idx, tensor)

        # Precise timing isolation for target computation tracking
        t0 = time.perf_counter()
        self.interp.invoke()
        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract output tensor parameters
        raw_output = self.interp.get_tensor(self._out_idx)  # Quantized array: (1, 2)
        logits = self._dequantise(raw_output)  # Rescaled float logits
        probs = self._softmax(logits)[0]  # Categorized probabilities

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        raw_class = self.class_names[pred_idx]

        # Edge Safety Guardrail: Force suppression on weak signature matches
        if confidence < self.confidence_threshold:
            pred_class = "NO_DRONE"
        else:
            pred_class = raw_class

        return {
            "class": pred_class,
            "raw_class": raw_class,
            "confidence": confidence,
            "is_drone": (pred_class == "DRONE"),
            "probs": probs,
            "latency_ms": latency_ms,
            "suppressed": (pred_class != raw_class),
        }

    @staticmethod
    def _build_interpreter(model_path: str, use_npu: bool):
        """Compiles the runtime engine path targeting either the HTP NPU delegate or native CPU."""
        from ai_edge_litert.interpreter import Interpreter, load_delegate
        delegates = []

        if use_npu:
            try:
                # Binds execution graph directly to the Hexagon NPU backend context driver
                delegates = [load_delegate(DELEGATE_LIB, {"backend_type": "htp"})]
                print(f"[Runtime] Qualcomm QNN Delegate bound successfully. Target: Hexagon Hardware NPU.")
            except Exception as exc:
                print(f"[Runtime] WARNING: QNN Delegate initialization failed: {exc}")
                print(f"[Runtime] Reverting execution context path back to standard CPU cores.")
        else:
            print(f"[Runtime] Manual override active: Running in CPU-only mode.")

        interp = Interpreter(model_path=model_path, experimental_delegates=delegates)
        interp.allocate_tensors()
        return interp

    def _dequantise(self, raw: np.ndarray) -> np.ndarray:
        """Converts raw quantized integer arrays back to meaningful mathematical float logits."""
        if self._out_scale != 0.0:
            return (raw.astype(np.float32) - self._out_zp) * self._out_scale
        return raw.astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Computes numerically stable softmax activations across rows."""
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(np.clip(x, -500, 500))
        return e / e.sum(axis=-1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Image Preprocessing Engine (Matches Training Transforms Exactly)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_spectrogram(image_path: str, img_h: int, img_w: int) -> np.ndarray:
    """
    Transforms a real spectrogram image file from disk into a compliant tensor format.
    Matches PyTorch 'transforms.ToTensor()' with zero ImageNet shifts.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target spectrogram file could not be found at path: {image_path}")

    # 1. Force image context to explicit RGB 3-channel layout
    img = Image.open(image_path).convert("RGB")

    # 2. Resize to exact target dimension matrix. Note: PIL uses (Width, Height) layout order.
    img = img.resize((img_w, img_h), Image.BILINEAR)

    # 3. Convert image structure to float matrix and normalize integers [0, 255] -> [0.0, 1.0]
    arr = np.array(img, dtype=np.float32) / 255.0

    # 4. Transpose layout axes from HWC to NCHW for structural hardware compliance
    arr = arr.transpose(2, 0, 1)  # Shift channels to front -> (3, Height, Width)
    arr = arr[np.newaxis]  # Expand dimensions for batch tracking -> (1, 3, Height, Width)

    return arr


# ─────────────────────────────────────────────────────────────────────────────
#  Command Line Configuration Interface
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="EfficientNet-B0 Edge Inference Runner")
    p.add_argument("--model", default="exports/efficientvit_l2_drone_quantized.tflite",
                   help="Path to production .tflite file")
    p.add_argument("--input", default=None, help="Optional: Path to real spectrogram image file to evaluate")
    p.add_argument("--labels", default="class_names.txt", help="Path to classification definitions text file")
    p.add_argument("--cpu", action="store_true", help="Force execution fallback to the ARM system CPU")
    p.add_argument("--runs", type=int, default=10, help="Total execution benchmark iterations (Standalone mode only)")
    p.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Drone classification execution gate")
    return p.parse_args()


def main():
    args = get_args()

    line_sep = "=" * 60
    print(f"\n{line_sep}")
    print("  Qualcomm Edge Deployment — Real-Time Signal Profiler")
    print(line_sep)

    # Initialize the inference engine wrapper object
    inferencer = DroneInferencer(
        model_path=args.model,
        labels_path=args.labels,
        use_npu=not args.cpu,
        confidence_threshold=args.threshold,
    )

    # Determine execution vector path based on input tracking state
    if args.input is not None:
        # Vector A: Live file compilation and evaluation
        print(f"[Execution] Sourcing real target signature data from: {args.input}")

        # Dynamically sample the input dimensions expected by the current runtime graph
        target_h, target_w = inferencer._inp_shape[2], inferencer._inp_shape[3]

        input_tensor = load_and_preprocess_spectrogram(args.input, target_h, target_w)
        result = inferencer.run(input_tensor)

        suppress_flag = " [SUPPRESSED VIA THRESHOLD GUARDRAIL]" if result["suppressed"] else ""
        print(f"\n{'-' * 40}")
        print(f"  Classification Output : {result['class']}{suppress_flag}")
        print(f"  Target Prediction Conf: {result['confidence'] * 100:.2f}%")
        print(f"  Hardware Latency Speed: {result['latency_ms']:.2f} ms")
        print(f"{'-' * 40}\n")

    else:
        # Vector B: Standalone hardware benchmarking optimization loop
        print(f"[Execution] No target file passed via '--input'. Initializing standalone benchmark...")
        print(f"[Execution] Synthesizing {args.runs} random float arrays over layout shape: {inferencer._inp_shape}\n")

        simulated_tensor = np.random.uniform(0.0, 1.0, inferencer._inp_shape).astype(np.float32)
        latencies = []

        for idx in range(args.runs):
            res = inferencer.run(simulated_tensor)
            latencies.append(res["latency_ms"])
            suppress_text = " [Suppressed]" if res["suppressed"] else ""
            print(
                f"  Iteration {idx + 1:>2d}: Target Classification -> {res['class']:10s} | Latency: {res['latency_ms']:6.2f} ms{suppress_text}")

        print(f"\n{'─' * 50}")
        print(f"  Hardware Accelerator Performance Analytics")
        print(f"  Mean Frame Processing Time : {np.mean(latencies):.2f} ms")
        print(f"  Minimum Turnaround Time    : {np.min(latencies):.2f} ms")
        print(f"  Maximum Turnaround Time    : {np.max(latencies):.2f} ms")
        print(f"{'─' * 50}\n")


if __name__ == "__main__":
    main()