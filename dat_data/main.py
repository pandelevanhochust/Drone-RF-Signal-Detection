from drone_dataloader import build_dataloaders
from roi import DronePipeline
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1 — Build loaders
train_loader, val_loader, test_loader, meta = build_dataloaders(
    root       = "output_spectrograms/",   # segment_dataset.py output
    subsets    = ["BOTH"],                 # or ["BOTH", "CLEAN"]
    img_size   = (256, 512),               # H×W, both div-by-16
    batch_size = 16,
)

# Step 2 — Build model
pipeline = DronePipeline(num_classes=meta["num_classes"]).to(device)

# Step 3 — Train loop
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    probs = pipeline(images)              # (B, num_classes)