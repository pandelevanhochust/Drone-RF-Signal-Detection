"""
drone_inference.py
==================
ONNX Runtime GPU (CUDA) inference for EfficientNet-B0 3-class drone classifier.
Optimized for workstation testing with NVIDIA GeForce RTX 2080.

Model details
-------------
    Input name    : image_tensor   shape (1, 3, 256, 512)  float32
    Output name   : class_logits   shape (1, 3)             float32
    Classes (alphabetical):
        index 0 → DRONE          strong vertical stripe blocks
        index 1 → DRONE_SIGNAL   sparse energy bursts + weak lower-band floor
        index 2 → NO_DRONE       diffuse background noise
"""

import argparse
import os
import time
import numpy as np

IMG_H                = 256
IMG_W                = 512
CLASS_NAMES_DEFAULT  = ["DRONE", "DRONE_SIGNAL", "NO_DRONE"]
NUM_CLASSES          = 3
CONFIDENCE_THRESHOLD = 0.70


class DroneInferencer:
    """
    Wraps EfficientNet-B0 ONNX model for single-call 3-class inference using CUDA.
    """

    def __init__(
        self,
        model_path           : str,
        labels_path          : str  = "class_names.txt",
        use_npu              : bool = False, # Parameter kept for run_pipeline compatibility
        confidence_threshold : float = CONFIDENCE_THRESHOLD,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # ── Class names ───────────────────────────────────────────────────────
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                self.class_names = [l.strip() for l in f if l.strip()]
        else:
            print(f"[Inference] WARNING: {labels_path} not found — using defaults")
            self.class_names = CLASS_NAMES_DEFAULT

        assert len(self.class_names) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes."
        self.confidence_threshold = confidence_threshold

        # ── Build ONNX Runtime Session with CUDA Provider ─────────────────────
        import onnxruntime as ort

        print(f"[Inference] Initializing ONNX Runtime Session...")
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # Limit to 2GB VRAM
            }),
            'CPUExecutionProvider'
        ]

        self.session = ort.InferenceSession(model_path, providers=providers)

        # Check active provider to verify if GPU is actually being used
        active_providers = self.session.get_providers()
        print(f"[Inference] Active Hardware Providers: {self.session.get_provider_options().keys()}")
        if 'CUDAExecutionProvider' in self.session.get_provider_options().keys():
            print("[Inference]  ✓ SUCCESS: Accelerating inference on NVIDIA GPU (CUDA)")
        else:
            print("[Inference]  ⚠ WARNING: CUDA failed, falling back to CPU execution.")

        self.input_name = self.session.get_inputs()[0].name
        self._inp_shape = tuple(self.session.get_inputs()[0].shape)

        print(f"[Inference] Model       : {os.path.basename(model_path)}")
        print(f"[Inference] Input Node  : {self.input_name} {self._inp_shape}")
        print(f"[Inference] Classes     : {self.class_names}")
        print(f"[Inference] Threshold   : {confidence_threshold}")

        # ── Warmup ────────────────────────────────────────────────────────────
        dummy = np.zeros((1, 3, IMG_H, IMG_W), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})
        print("[Inference] GPU Warmup done\n")

    def run(self, tensor: np.ndarray) -> dict:
        """
        Run one inference pass on the GPU.
        tensor: float32 numpy array (1, 3, 256, 512)
        """
        t0 = time.perf_counter()
        # Run ONNX session forward pass
        onnx_outputs = self.session.run(None, {self.input_name: tensor})
        latency_ms = (time.perf_counter() - t0) * 1000

        logits = onnx_outputs[0] # Shape (1, 3)
        probs = self._softmax(logits)[0] # Shape (3,)

        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        raw_class  = self.class_names[pred_idx]

        # Apply confidence filtering
        suppressed = confidence < self.confidence_threshold
        pred_class = "NO_DRONE" if suppressed else raw_class

        return {
            "class"           : pred_class,
            "raw_class"       : raw_class,
            "confidence"      : confidence,
            "is_drone"        : pred_class in ("DRONE", "DRONE_SIGNAL"),
            "is_drone_full"   : pred_class == "DRONE",
            "is_drone_signal" : pred_class == "DRONE_SIGNAL",
            "probs"           : probs,
            "latency_ms"      : latency_ms,
            "suppressed"      : suppressed,
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(np.clip(x, -500, 500))
        return e / e.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DroneInferencer ONNX GPU Standalone test")
    p.add_argument("--model",     default="drone_classifier_b0.onnx")
    p.add_argument("--labels",    default="class_names.txt")
    p.add_argument("--runs",      type=int, default=10)
    p.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    args = p.parse_args()

    inferencer = DroneInferencer(
        model_path=args.model, labels_path=args.labels, confidence_threshold=args.threshold
    )
    dummy = np.random.uniform(0.0, 1.0, (1, 3, IMG_H, IMG_W)).astype(np.float32)

    print(f"Running {args.runs} passes on GPU...")
    latencies = []
    for i in range(args.runs):
        r = inferencer.run(dummy)
        latencies.append(r["latency_ms"])
        print(f"  Run {i+1:>2d}: {r['class']:<14s} latency={r['latency_ms']:.2f} ms")

    print(f"\nAvg latency : {np.mean(latencies):.2f} ms")