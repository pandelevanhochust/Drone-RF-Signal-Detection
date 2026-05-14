import onnxruntime as ort

sess = ort.InferenceSession(
    "helper/drone_pipeline.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    # OnnxRuntime tries CUDA first, falls back to CPU if unavailable
)