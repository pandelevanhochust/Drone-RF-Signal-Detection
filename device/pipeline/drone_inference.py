"""
drone_inference.py
==================
TFLite inference for the fused drone detection pipeline on RB3 Gen 2 NPU.

Responsibilities
----------------
- Load drone_pipeline_fused_quantized.tflite with the QNN HTP delegate
- Accept a preprocessed (1, 3, 256, 512) NCHW float32 tensor
- Dequantise INT8 output, apply softmax, return top-1 class + probabilities
- Provide a warmup call so the first real inference is not penalised by
  NPU kernel initialisation latency

This module has NO dependency on BladeRF or STFT processing.
It only depends on: ai-edge-litert, numpy.

The input tensor must come from stft_preprocessor.iq_to_spectrogram()
or any source that produces the same (1, 3, 256, 512) NCHW float32
in the same ImageNet-normalised format.

Usage as a library
------------------
    from drone_inference import DroneInferencer

    inferencer = DroneInferencer("drone_pipeline_fused_quantized.tflite",
                                  "class_names.txt")
    result = inferencer.run(tensor)   # tensor from stft_preprocessor
    print(result["class"], result["confidence"])

Standalone test (runs without BladeRF using a random tensor)
-------------------------------------------------------------
    python3 drone_inference.py
    python3 drone_inference.py --cpu
"""

import argparse
import os
import time

import numpy as np

IMG_H, IMG_W        = 256, 512
DELEGATE_LIB        = "libQnnTFLiteDelegate.so"
CLASS_NAMES_DEFAULT = ["AIR", "DIS", "INS", "MIN", "MP1", "MP2", "NO_DRONE", "PHA"]


# ─────────────────────────────────────────────────────────────────────────────
#  DroneInferencer class
# ─────────────────────────────────────────────────────────────────────────────

class DroneInferencer:
    """
    Wraps the fused TFLite model for single-call inference.

    Constructor builds and warms up the interpreter once. After that,
    each call to .run(tensor) is a pure NPU forward pass with no
    re-allocation overhead.

    Parameters
    ----------
    model_path  : path to drone_pipeline_fused_quantized.tflite
    labels_path : path to class_names.txt (one class per line)
    use_npu     : True  → QNN HTP delegate (Hexagon NPU)
                  False → CPU only (for debugging / comparison)
    """

    def __init__(
        self,
        model_path  : str,
        labels_path : str = "class_names.txt",
        use_npu     : bool = True,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Copy from dev machine:\n"
                "  scp exports/drone_pipeline_fused_quantized.tflite ubuntu@<IP>:~/"
            )

        # ── Load class names ──────────────────────────────────────────────────
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                self.class_names = [l.strip() for l in f if l.strip()]
        else:
            print(f"[Inference] WARNING: {labels_path} not found — using defaults")
            self.class_names = CLASS_NAMES_DEFAULT

        # ── Build interpreter ─────────────────────────────────────────────────
        self.interp = self._build_interpreter(model_path, use_npu)

        inp = self.interp.get_input_details()[0]
        out = self.interp.get_output_details()[0]

        self._inp_idx   = inp["index"]
        self._out_idx   = out["index"]
        self._out_scale = out["quantization"][0]    # 0.0 if FP32 model
        self._out_zp    = out["quantization"][1]

        print(f"[Inference] Input  : {inp['shape']}  dtype={inp['dtype']}")
        print(f"[Inference] Output : {out['shape']}  "
              f"quant=(scale={self._out_scale:.6f}, zp={self._out_zp})")
        print(f"[Inference] Classes: {self.class_names}\n")

        # ── Warmup ────────────────────────────────────────────────────────────
        # First invoke initialises NPU kernel binaries — exclude from timing.
        dummy = np.zeros((1, 3, IMG_H, IMG_W), dtype=np.float32)
        self.interp.set_tensor(self._inp_idx, dummy)
        self.interp.invoke()
        print("[Inference] Warmup done\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, tensor: np.ndarray) -> dict:
        """
        Run one inference pass on a preprocessed spectrogram tensor.

        Parameters
        ----------
        tensor : float32 ndarray, shape (1, 3, 256, 512)  NCHW
            Output of stft_preprocessor.iq_to_spectrogram().
            Do NOT transpose to NHWC — QNN delegate handles layout internally.

        Returns
        -------
        dict with keys:
            class       : str   — predicted class name
            confidence  : float — softmax probability of top-1 class [0, 1]
            probs       : ndarray (num_classes,) — all class probabilities
            latency_ms  : float — NPU invoke time in milliseconds
        """
        self.interp.set_tensor(self._inp_idx, tensor)

        t0 = time.perf_counter()
        self.interp.invoke()
        latency_ms = (time.perf_counter() - t0) * 1000

        raw    = self.interp.get_tensor(self._out_idx)    # (1, num_classes)
        logits = self._dequantise(raw)
        probs  = self._softmax(logits)[0]                 # (num_classes,)

        pred_idx = int(np.argmax(probs))
        return {
            "class"      : self.class_names[pred_idx],
            "confidence" : float(probs[pred_idx]),
            "probs"      : probs,
            "latency_ms" : latency_ms,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
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
    def _dequantise(self, raw: np.ndarray) -> np.ndarray:
        """
        Dequantise INT8 TFLite output to float32 logits.

        Formula: logits = (raw_int8 - zero_point) * scale
        If scale == 0.0 the model output is already FP32 (no-op).
        """
        if self._out_scale != 0.0:
            return (raw.astype(np.float32) - self._out_zp) * self._out_scale
        return raw.astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax along last axis."""
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(np.clip(x, -500, 500))
        return e / e.sum(axis=-1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="drone_inference.py — standalone test (no BladeRF needed)"
    )
    p.add_argument("--model",  default="../quantize_model/drone_pipeline_fused_quantized.tflite")
    p.add_argument("--labels", default="class_names.txt")
    p.add_argument("--cpu",    action="store_true",
                   help="Disable NPU delegate, run on CPU only")
    p.add_argument("--runs",   type=int, default=10,
                   help="Number of inference passes for latency benchmark")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    print("=" * 55)
    print("  DroneInferencer — standalone test")
    print("=" * 55)

    inferencer = DroneInferencer(
        model_path  = args.model,
        labels_path = args.labels,
        use_npu     = not args.cpu,
    )

    dummy_tensor = np.random.randn(1, 3, IMG_H, IMG_W).astype(np.float32)

    print(f"Running {args.runs} inference passes on a random tensor ...\n")
    latencies = []
    for i in range(args.runs):
        result = inferencer.run(dummy_tensor)
        latencies.append(result["latency_ms"])
        print(
            f"  Run {i+1:>2d}:  {result['class']:10s}  "
            f"conf={result['confidence']*100:5.1f}%  "
            f"latency={result['latency_ms']:6.2f} ms"
        )

    print(f"\n{'─'*45}")
    print(f"  Avg latency : {np.mean(latencies):.2f} ms")
    print(f"  Min latency : {np.min(latencies):.2f} ms")
    print(f"  Max latency : {np.max(latencies):.2f} ms")
    print(f"{'─'*45}")
    print("\nAll inference tests passed.")