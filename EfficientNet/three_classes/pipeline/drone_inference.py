"""
drone_inference.py
==================
TFLite inference for EfficientViT-L2 3-class drone classifier on RB3 Gen 2.

Model details
-------------
    Input name    : image_tensor   shape (1, 3, 256, 512)  float32
    Output name   : class_logits   shape (1, 3)             float32
    Quantisation  : FP32 (scale=0.0, zp=0) — no dequantisation needed
    Classes (alphabetical, matches ImageFolder sort order):
        index 0 → DRONE          strong vertical stripe blocks
        index 1 → DRONE_SIGNAL   sparse energy bursts + weak lower-band floor
        index 2 → NO_DRONE       diffuse background noise

Preprocessing (must match train_and_export.py get_transforms exactly)
-----------------------------------------------------------------------
    STFT spectrogram → RGB → resize (256, 512) → / 255.0  [0.0, 1.0]
    NO ImageNet mean/std normalisation

Standalone test
---------------
    python3 drone_inference.py --model <path>.tflite
    python3 drone_inference.py --model <path>.tflite --cpu
"""

import argparse
import os
import time

import numpy as np

IMG_H                = 256
IMG_W                = 512
DELEGATE_LIB         = "libQnnTFLiteDelegate.so"
CLASS_NAMES_DEFAULT  = ["DRONE", "DRONE_SIGNAL", "NO_DRONE"]
NUM_CLASSES          = 3
CONFIDENCE_THRESHOLD = 0.70


class DroneInferencer:
    """
    Wraps EfficientViT-L2 TFLite for single-call 3-class inference.

    Parameters
    ----------
    model_path           : path to .tflite model file
    labels_path          : path to class_names.txt (one class per line)
    use_npu              : True  → QNN HTP delegate (Hexagon NPU)
                           False → CPU (XNNPACK)
    confidence_threshold : Minimum softmax confidence to trust top-1 class.
                           Below threshold → result forced to NO_DRONE.
    """

    def __init__(
        self,
        model_path           : str,
        labels_path          : str   = "class_names.txt",
        use_npu              : bool  = True,
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

        assert len(self.class_names) == NUM_CLASSES, (
            f"Expected {NUM_CLASSES} classes, got {len(self.class_names)}: "
            f"{self.class_names}. Update class_names.txt to include "
            f"DRONE, DRONE_SIGNAL, NO_DRONE (one per line, alphabetical order)."
        )

        self.confidence_threshold = confidence_threshold

        # ── Build interpreter ─────────────────────────────────────────────────
        self.interp = self._build_interpreter(model_path, use_npu)

        inp = self.interp.get_input_details()[0]
        out = self.interp.get_output_details()[0]

        self._inp_idx   = inp["index"]
        self._out_idx   = out["index"]
        self._inp_shape = tuple(int(x) for x in inp["shape"])
        self._out_scale = out["quantization"][0]
        self._out_zp    = out["quantization"][1]

        print(f"[Inference] Model       : {os.path.basename(model_path)}")
        print(f"[Inference] Input       : {inp['shape']}  dtype={inp['dtype'].__name__}")
        print(f"[Inference] Output      : {out['shape']}  "
              f"quant=(scale={self._out_scale}, zp={self._out_zp})")
        print(f"[Inference] Classes     : {self.class_names}")
        print(f"[Inference] Threshold   : {confidence_threshold}")

        # ── Shape guard ───────────────────────────────────────────────────────
        assert self._inp_shape == (1, 3, IMG_H, IMG_W), (
            f"Model expects {self._inp_shape} but pipeline produces "
            f"(1, 3, {IMG_H}, {IMG_W}). Check IMG_H/IMG_W in stft_preprocessor.py"
        )
        assert out["shape"][-1] == NUM_CLASSES, (
            f"Model output has {out['shape'][-1]} classes but expected {NUM_CLASSES}. "
            f"Re-export ONNX/TFLite from the updated train_and_export.py."
        )

        # ── Warmup ────────────────────────────────────────────────────────────
        dummy = np.zeros(self._inp_shape, dtype=np.float32)
        self.interp.set_tensor(self._inp_idx, dummy)
        self.interp.invoke()
        print("[Inference] Warmup done\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, tensor: np.ndarray) -> dict:
        """
        Run one inference pass on a preprocessed spectrogram tensor.

        Parameters
        ----------
        tensor : float32 ndarray (1, 3, 256, 512) from stft_preprocessor.
                 Values in [0.0, 1.0] — /255 only, no ImageNet norm.

        Returns
        -------
        dict:
            class           : "DRONE" | "DRONE_SIGNAL" | "NO_DRONE"
                              (confidence threshold applied — falls back to
                               NO_DRONE when top-1 confidence < threshold)
            raw_class       : top-1 class before threshold check
            confidence      : float — softmax probability of top-1 [0.0, 1.0]
            is_drone        : bool  — True for DRONE or DRONE_SIGNAL
            is_drone_full   : bool  — True only for DRONE (strong signature)
            is_drone_signal : bool  — True only for DRONE_SIGNAL (partial sig)
            probs           : ndarray (3,) — [p_DRONE, p_DRONE_SIGNAL, p_NO_DRONE]
            latency_ms      : float — invoke time in milliseconds
            suppressed      : bool  — True when threshold forced class to NO_DRONE
        """
        self.interp.set_tensor(self._inp_idx, tensor)

        t0 = time.perf_counter()
        self.interp.invoke()
        latency_ms = (time.perf_counter() - t0) * 1000

        raw    = self.interp.get_tensor(self._out_idx)   # (1, 3) float32
        logits = self._dequantise(raw)                   # no-op for FP32 model
        probs  = self._softmax(logits)[0]                # (3,)

        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        raw_class  = self.class_names[pred_idx]

        # Confidence threshold — suppress low-confidence detections to NO_DRONE
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_interpreter(model_path: str, use_npu: bool):
        from ai_edge_litert.interpreter import Interpreter, load_delegate
        delegates = []
        if use_npu:
            try:
                delegates = [load_delegate(DELEGATE_LIB, {"backend_type": "htp"})]
                print("[Inference] QNN delegate loaded  (HTP / Hexagon NPU)")
            except Exception as exc:
                print(f"[Inference] WARNING: QNN delegate failed: {exc}")
                print("[Inference] Falling back to CPU (XNNPACK).")
        else:
            print("[Inference] CPU-only mode (XNNPACK)")
        interp = Interpreter(model_path=model_path,
                             experimental_delegates=delegates)
        interp.allocate_tensors()
        return interp

    def _dequantise(self, raw: np.ndarray) -> np.ndarray:
        """Dequantise INT8 output. No-op for FP32 model (scale == 0.0)."""
        if self._out_scale != 0.0:
            return (raw.astype(np.float32) - self._out_zp) * self._out_scale
        return raw.astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(np.clip(x, -500, 500))
        return e / e.sum(axis=-1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="DroneInferencer 3-class standalone test")
    p.add_argument("--model",     default="../new_three_classes.tflite")
    p.add_argument("--labels",    default="class_names.txt")
    p.add_argument("--cpu",       action="store_true", help="Disable NPU delegate")
    p.add_argument("--runs",      type=int, default=10)
    p.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    print("=" * 62)
    print("  DroneInferencer — EfficientViT-L2  3-class  (standalone test)")
    print("=" * 62)

    inferencer = DroneInferencer(
        model_path           = args.model,
        labels_path          = args.labels,
        use_npu              = not args.cpu,
        confidence_threshold = args.threshold,
    )

    dummy = np.random.uniform(0.0, 1.0, (1, 3, IMG_H, IMG_W)).astype(np.float32)

    print(f"Running {args.runs} passes on random [0,1] tensor ...\n")
    latencies = []
    for i in range(args.runs):
        r = inferencer.run(dummy)
        latencies.append(r["latency_ms"])
        p = r["probs"]
        flag = ""
        if r["is_drone_full"]:
            flag = "  ⚠ DRONE"
        elif r["is_drone_signal"]:
            flag = "  ~ DRONE_SIGNAL"
        supp = "  [suppressed]" if r["suppressed"] else ""
        print(
            f"  Run {i+1:>2d}:  {r['class']:<14s}"
            f"  DRONE={p[0]*100:5.1f}%"
            f"  DRONE_SIGNAL={p[1]*100:5.1f}%"
            f"  NO_DRONE={p[2]*100:5.1f}%"
            f"  {r['latency_ms']:6.2f} ms"
            f"{flag}{supp}"
        )

    print(f"\n{'─'*55}")
    print(f"  Avg latency : {np.mean(latencies):.2f} ms")
    print(f"  Min latency : {np.min(latencies):.2f} ms")
    print(f"  Max latency : {np.max(latencies):.2f} ms")
    print(f"{'─'*55}")