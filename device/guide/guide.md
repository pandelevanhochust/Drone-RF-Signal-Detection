Good — I have the full doc. Now I'll write the inference script tailored exactly to your fused pipeline.Here's the full guide to get it running on the RB3 Gen 2.

---

## Step 1 — One-time device setup

SSH into the board and run:

```bash
# Install PPA packages (QNN delegate + system libs)
git clone -b ubuntu_setup --single-branch \
    https://github.com/rubikpi-ai/rubikpi-script.git
cd rubikpi-script && ./install_ppa_pkgs.sh

# Create a venv (--system-site-packages gives access to system gi/gtk)
python3 -m venv .venv-drone --system-site-packages
source .venv-drone/bin/activate
pip3 install ai-edge-litert==1.3.0 Pillow numpy
```

## Step 2 — Copy files to device

From your dev machine:

```bash
scp exports/drone_pipeline_fused_quantized.tflite ubuntu@<IP>:/home/ubuntu/
scp exports/class_names.txt                        ubuntu@<IP>:/home/ubuntu/
scp drone_inference.py                             ubuntu@<IP>:/home/ubuntu/
```

## Step 3 — Run

```bash
source .venv-drone/bin/activate

# Single image on NPU
python3 drone_inference.py --image spectrogram.png

# Single image, CPU only (useful for accuracy comparison vs NPU)
python3 drone_inference.py --image spectrogram.png --cpu

# Benchmark latency — 50 runs, reports average
python3 drone_inference.py --image spectrogram.png --runs 50

# Whole folder
python3 drone_inference.py --folder /path/to/spectrograms/
```

## What the output looks like

```
[Setup] Classes (8): ['AIR', 'DIS', 'INS', 'MIN', 'MP1', 'MP2', 'NO_DRONE', 'PHA']
[Setup] QNN delegate loaded  (backend: HTP / Hexagon NPU)
[Setup] Input  : (1, 3, 256, 512)  dtype=float32
[Setup] Output : (1, 8)  quant=(scale=0.054321, zp=-12)
[Setup] Backend: NPU (HTP delegate)

  Image   : dji_cao50_seg0010.png
  Result  : MP1  (94.3%)
  Latency : 38.7 ms  (avg over 50 run(s))

  Class        Prob
  ────────────────────
  AIR           0.21%
  DIS           0.08%
  INS           1.14%
  MIN           0.43%
  MP1          94.31% ◀
  MP2           2.67%
  NO_DRONE      0.73%
  PHA           0.43%
```

## Three things to watch for

**NPU delegation log** — when the delegate loads you'll see a line like `1382 nodes delegated out of 419 nodes with N partitions`. For the fused model all 419 ops should delegate to NPU since it was compiled specifically for the RB3 with `--compute_unit npu`. If you see a low delegation count it means the delegate version on the board doesn't match the one used at compile time — update the board's PPA packages.

**Dequantisation** — the script reads `scale` and `zero_point` from `output_details[0]['quantization']` automatically, so it works regardless of what the Hub quantiser chose. You don't need to hardcode those values.

**NCHW vs NHWC** — the script feeds `(1, 3, 256, 512)` NCHW as produced by the training pipeline. The QNN delegate handles the layout transpose at the first op internally. Do not manually permute to NHWC before feeding.