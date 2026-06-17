from ultralytics.utils.downloads import download
from pathlib import Path

# This downloads COCO128 (128 sample images used by Ultralytics)
download('https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip', dir=Path('.'))