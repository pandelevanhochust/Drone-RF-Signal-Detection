"""
prepare_yolo_calibration_data.py
==================================
Prepares calibration .raw files for YOLOv8 quantization.

IMPORTANT: YOLOv8 preprocessing differs from classification models —
NO ImageNet mean/std normalisation, just /255.0 to [0,1] range.
This must match exactly what fastcv_pipeline.cpp::preprocess() does.

Run on: x86 host
Usage:  python3 prepare_yolo_calibration_data.py \
            --img_dir  ./images/coco_calib/ \
            --out_dir  ./data/calib_yolo/ \
            --num_imgs 200
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def letterbox_resize(img: Image.Image, size: int = 640) -> np.ndarray:
    """
    Letterbox resize matching Ultralytics' default preprocessing:
    pad to square with grey (114,114,114), preserve aspect ratio.
    NOTE: fastcv_pipeline.cpp uses plain bilinear resize (no letterbox) —
    keep this consistent with your C++ implementation, or update
    fastcv_pipeline.cpp to letterbox too for best accuracy match.
    """
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    return np.array(canvas, dtype=np.float32)


def simple_resize(img: Image.Image, size: int = 640) -> np.ndarray:
    """Plain resize — matches fastcv_pipeline.cpp's fcvScaleBilinearu8 path."""
    img_resized = img.resize((size, size), Image.BILINEAR)
    return np.array(img_resized, dtype=np.float32)


def preprocess(img_path: str, size: int = 640, use_letterbox: bool = False) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    arr = letterbox_resize(img, size) if use_letterbox else simple_resize(img, size)

    arr = arr / 255.0                       # [0,1] — NO mean/std subtraction
    arr = arr.transpose(2, 0, 1)            # HWC → CHW
    arr = arr[np.newaxis, ...]              # → NCHW (1,3,640,640)
    return np.ascontiguousarray(arr, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir",  required=True)
    p.add_argument("--out_dir",  required=True)
    p.add_argument("--num_imgs", type=int, default=200)
    p.add_argument("--size",     type=int, default=640)
    p.add_argument("--letterbox", action="store_true",
                   help="Use letterbox resize (must match C++ pipeline if set)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted(p for p in Path(args.img_dir).iterdir()
                  if p.suffix.lower() in exts)[: args.num_imgs]

    if not imgs:
        raise FileNotFoundError(f"No images in {args.img_dir}")

    print(f"Processing {len(imgs)} images "
          f"({'letterbox' if args.letterbox else 'simple resize'}) → {out_dir}")

    for i, img_path in enumerate(imgs):
        tensor = preprocess(str(img_path), args.size, args.letterbox)
        raw_path = out_dir / (img_path.stem + ".raw")
        tensor.tofile(str(raw_path))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(imgs)}")

    print(f"\n✓ Saved {len(imgs)} raw files to {out_dir}")
    print(f"  Bytes per file: {1*3*args.size*args.size*4} (float32 NCHW)")


if __name__ == "__main__":
    main()

# scripts/prepare_yolo_calibration_data.py --img_dir images/coco_calib/ --out_dir data/calib_yolo/ --num_imgs 200