import onnxruntime as ort

sess = ort.InferenceSession(
    "checkpoints/drone_pipeline.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    # OnnxRuntime tries CUDA first, falls back to CPU if unavailable
)