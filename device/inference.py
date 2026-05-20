import os
import time
import numpy as np
from PIL import Image

# Use the appropriate runtime import present on your Qualcomm Linux environment
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tflite

# =====================================================================
# CONFIGURATION & PATH SETUP
# =====================================================================
MODEL_PATH = "drone_classifier_quantized.tflite"
TEST_IMAGE_PATH = "sample_spectrogram.png"

# Verify hardware asset existence
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Quantized TFLite model file not found at: {MODEL_PATH}")

# =====================================================================
# STEP 1: INITIALIZE INTERPRETER WITH NPU DELEGATE ACCELERATION
# =====================================================================
print("Initializing LiteRT Interpreter...")

# To target the QCS6490 Hexagon NPU hardware core natively,
# we load the interpreter and link it against Qualcomm's QNN TFLite Delegate library.
try:
    # Path to the Qualcomm QNN delegate binary on the RB3 image filesystem
    qnn_delegate_path = "/usr/lib/libQnnTfLiteDelegate.so"

    if os.path.exists(qnn_delegate_path):
        experimental_delegates = [tflite.load_delegate(qnn_delegate_path)]
        print("✓ Successfully loaded Qualcomm QNN Hardware Delegate for NPU acceleration.")
    else:
        experimental_delegates = None
        print("⚠️ Warning: QNN Delegate library not found at standard path. Falling back to default routing.")

    interpreter = tflite.Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=experimental_delegates
    )
except Exception as e:
    print(f"Failed to initialize hardware delegate ({e}). Falling back to standard runtime.")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)

# Allocate internal tensor buffers for the execution graph
interpreter.allocate_tensors()

# Retrieve input/output tensor layout details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Model Input Details: {input_details[0]['name']} -> Shape: {input_details[0]['shape']}")
print(f"Model Output Details: {output_details[0]['name']} -> Shape: {output_details[0]['shape']}\n")


# =====================================================================
# STEP 2: PIPELINE - RAW DATA CAPTURE AND TRANSFORM (CPU EXECUTION)
# =====================================================================
def preprocess_raw_image(image_path, target_size=(224, 224)):
    """
    Transforms raw data into an appropriately shaped array.
    Executes entirely on the CPU using optimized NumPy functions.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No source file found at {image_path}")

    # 1. Load raw data format via Pillow
    img = Image.open(image_path).convert("RGB")

    # 2. Resize to target dimension tracking expectations
    img = img.resize(target_size)

    # 3. Transpose from HWC to CHW and normalize values to [0.0, 1.0] range
    img_array = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))

    # 4. Inject matching Batch Size Dimension -> results in shape (1, 3, 224, 224)
    img_array_4d = np.expand_dims(img_array, axis=0)

    return img_array_4d


# Execute the CPU-bound data pipeline transformation stage
print(f"Processing raw data source: {TEST_IMAGE_PATH} on CPU...")
input_tensor = preprocess_raw_image(TEST_IMAGE_PATH)

# =====================================================================
# STEP 3: EXECUTE INFERENCE ON TARGET NPU ACCELERATOR
# =====================================================================
print("Feeding input tensor down to NPU engine...")

# Bind the processed CPU buffer to your model's expected layer index ('roi_patch')
interpreter.set_tensor(input_details[0]['index'], input_tensor)

# Warm-up run to initialize hardware pipelines cleanly
interpreter.invoke()

# Measure true hardware execution latency
start_time = time.perf_counter()
interpreter.invoke()  # Triggers actual NPU computation
inference_time = (time.perf_counter() - start_time) * 1000.0

# Retrieve output array maps
output_data = interpreter.get_tensor(output_details[0]['index'])

# =====================================================================
# STEP 4: OUTPUT PIPELINE & METRICS DISPLAY
# =====================================================================
print("-" * 50)
print("ON-DEVICE EXECUTION SUCCESSFUL")
print(f"NPU Inference Latency: {inference_time:.2f} ms")
print(f"Raw Output Tensor Shape: {output_data.shape}")
print("Predicted Output Array Content:")
print(output_data)
print("-" * 50)