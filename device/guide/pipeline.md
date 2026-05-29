Good — full file read. Now I'll write three clean files.Four clean files, each with a single responsibility.

---

## Module breakdown

```
bladerf_capture.py          stft_preprocessor.py        drone_inference.py
──────────────────          ────────────────────        ──────────────────
open_bladerf()              iq_to_spectrogram()         DroneInferencer
close_bladerf()               → Hann window               __init__()
capture_frame()               → FFT + fftshift              build interpreter
start_capture_thread()        → log power dB               warmup invoke
                              → PIL bilinear resize       .run(tensor)
Depends on:                   → ImageNet normalise          dequantise INT8
  bladerf, numpy            → (1,3,256,512) NCHW            softmax
                                                          Depends on:
                            save_spectrogram_png()          ai-edge-litert
                            Depends on:                     numpy
                              numpy, Pillow

                    run_pipeline.py
                    ───────────────
                    Imports all three modules
                    BladeRF capture thread → queue → STFT → NPU
                    CLI flags, Ctrl+C shutdown
```

## Test each module independently before running the full pipeline

```bash
# 1. STFT only — no BladeRF, no model needed
python3 stft_preprocessor.py
# Generates synthetic tone + noise frames, saves to debug_stft/

# 2. Inference only — no BladeRF needed, random tensor
python3 drone_inference.py --runs 10

# 3. BladeRF capture only — no inference
python3 bladerf_capture.py
# Captures 3 frames, prints power dBFS

# 4. Full pipeline
python3 run_pipeline.py
python3 run_pipeline.py --save_dir debug_specs/ --no_infer  # STFT debug_old
```