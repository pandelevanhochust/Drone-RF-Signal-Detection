"""
stage1_unet.py
==============
Stage 1 — U-Net ROI Segmentation + ROI Extraction

What this stage does
--------------------
Takes a raw STFT spectrogram → produces a binary segmentation mask that
highlights the drone RF signal region among noise → extracts and resizes
that region to 224×224 for the classifier.

There are NO ground-truth mask files. Supervision comes from a
NO_DRONE-aware energy proxy mask (Strategy 4): the energy distribution of
the spectrogram itself is thresholded to produce a weak binary mask, and
samples labelled NO_DRONE always receive an all-zero mask target.

Separation from Stage 2
------------------------
This module is self-contained. It exposes:
  - DroneROIUNet     : the segmentation model
  - ROIExtractor     : mask × spectrogram → 224×224 patch
  - build_proxy_mask : weak-supervision mask target (no .png files needed)
  - train_unet()     : standalone U-Net training loop using only BCE on the
                       proxy mask, with classifier frozen / absent
  - load_unet()      : reload a saved Stage-1 checkpoint

Stage 2 imports DroneROIUNet + ROIExtractor from here and can either:
  (a) load a pretrained Stage-1 checkpoint and fine-tune end-to-end, or
  (b) freeze Stage-1 weights entirely and only train the classifier.
"""

import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


# ─────────────────────────────────────────────────────────────────────────────
#  Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Conv3×3 → BN → ReLU → [Dropout2d] → Conv3×3 → BN → ReLU.

    Dropout2d (Strategy 2) drops entire feature-map channels rather than
    individual scalars, which is more effective for spatially structured
    spectrogram features. Default p=0.0 leaves encoder blocks unchanged.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv → (skip_for_decoder, downsampled_for_next_level)."""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    """
    Base decoder block — do not instantiate directly.
    Use make_decoder_block() which returns a concrete subclass with
    literal-integer slice indices baked into the class body.

    Why subclasses instead of a single class with self.skip_h
    ----------------------------------------------------------
    TorchScript's ONNX exporter serialises `x[:, :, :self.skip_h, :]` as a
    Slice node where `end` is loaded from the module's attribute dict at
    runtime — making it a dynamic input to the Slice op.  SNPE's shape
    inference cannot resolve dynamic Slice ends and errors with:
        "invalid stride 1 for begin 0 and end 0 at axis N"

    The only way to produce a *static* Slice constant in the exported ONNX
    is to write the integer literally in the Python source so TorchScript
    folds it as a compile-time constant.  Each concrete subclass below does
    exactly that — the integers are literals in the class body, not loaded
    from self.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use make_decoder_block()")


# ── Concrete subclasses with literal-integer slices ───────────────────────────
# One class per decoder level for each supported (img_h, img_w).
# The slice indices are Python integer literals → TorchScript folds them
# as ONNX Slice constants → SNPE resolves them statically.

class _Dec256x512_L3(DecoderBlock):   # skip = img_h//8, img_w//8 = 32, 64
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = x[:, :, :32, :64]
        return self.conv(torch.cat([skip, x], dim=1))

class _Dec256x512_L2(DecoderBlock):   # skip = 64, 128
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = x[:, :, :64, :128]
        return self.conv(torch.cat([skip, x], dim=1))

class _Dec256x512_L1(DecoderBlock):   # skip = 128, 256
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = x[:, :, :128, :256]
        return self.conv(torch.cat([skip, x], dim=1))

class _Dec256x512_L0(DecoderBlock):   # skip = 256, 512
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = x[:, :, :256, :512]
        return self.conv(torch.cat([skip, x], dim=1))


# 128×256 variants (half resolution, for smaller GPU budgets)
class _Dec128x256_L3(DecoderBlock):   # 16, 32
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = x[:, :, :16, :32]
        return self.conv(torch.cat([skip, x], dim=1))

class _Dec128x256_L2(DecoderBlock):   # 32, 64
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = x[:, :, :32, :64]
        return self.conv(torch.cat([skip, x], dim=1))

class _Dec128x256_L1(DecoderBlock):   # 64, 128
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = x[:, :, :64, :128]
        return self.conv(torch.cat([skip, x], dim=1))

class _Dec128x256_L0(DecoderBlock):   # 128, 256
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = x[:, :, :128, :256]
        return self.conv(torch.cat([skip, x], dim=1))


# Lookup table: (img_h, img_w) → [L3_cls, L2_cls, L1_cls, L0_cls]
_DECODER_CLASSES = {
    (256, 512): [_Dec256x512_L3, _Dec256x512_L2,
                 _Dec256x512_L1, _Dec256x512_L0],
    (128, 256): [_Dec128x256_L3, _Dec128x256_L2,
                 _Dec128x256_L1, _Dec128x256_L0],
}


def make_decoder_block(
    in_ch    : int,
    out_ch   : int,
    level    : int,     # 3 = deepest (smallest spatial), 0 = full resolution
    img_h    : int,
    img_w    : int,
    dropout_p: float = 0.0,
) -> DecoderBlock:
    """
    Return a concrete DecoderBlock subclass whose forward() uses literal
    integer slice indices matching (img_h, img_w) at the given U-Net level.

    level 3 → skip size = (img_h//8,  img_w//8)
    level 2 → skip size = (img_h//4,  img_w//4)
    level 1 → skip size = (img_h//2,  img_w//2)
    level 0 → skip size = (img_h,     img_w)
    """
    key = (img_h, img_w)
    if key not in _DECODER_CLASSES:
        supported = list(_DECODER_CLASSES.keys())
        raise ValueError(
            f"No static decoder class for img_size=({img_h},{img_w}).\n"
            f"Supported: {supported}\n"
            f"Add a concrete subclass to stage1_unet.py for your resolution."
        )
    cls = _DECODER_CLASSES[key][3 - level]   # level 3→index 0, level 0→index 3
    block = cls(in_ch, out_ch, dropout_p=dropout_p)
    return block



# ─────────────────────────────────────────────────────────────────────────────
#  U-Net
# ─────────────────────────────────────────────────────────────────────────────

class DroneROIUNet(nn.Module):
    """
    U-Net segmentation model for drone RF signal localisation.

    Input  : raw STFT spectrogram  (B, C, img_h, img_w)
    Output : binary-ish mask       (B, 1, img_h, img_w)  values in [0, 1]

    Dropout2d rates (Strategy 2)
    ----------------------------
    Encoder   : 0.0  — preserve spatial features
    Bottleneck: 0.3  — heaviest regularisation at the narrowest point
    Decoder   : 0.1  — light regularisation on the way back up

    base_filters (Strategy 3)
    --------------------------
    Default 32  → ~8 M parameters  (was 64 → ~31 M)
    """

    def __init__(
        self,
        in_channels : int = 3,
        base_filters: int = 32,
        img_h       : int = 256,
        img_w       : int = 512,
    ):
        super().__init__()
        f = base_filters

        # Encoder (no dropout)
        self.enc0 = EncoderBlock(in_channels, f,     dropout_p=0.0)
        self.enc1 = EncoderBlock(f,           f * 2, dropout_p=0.0)
        self.enc2 = EncoderBlock(f * 2,       f * 4, dropout_p=0.0)
        self.enc3 = EncoderBlock(f * 4,       f * 8, dropout_p=0.0)

        # Bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16, dropout_p=0.3)

        # Decoder — concrete subclasses with literal-integer slice indices.
        # make_decoder_block() selects the right subclass for (img_h, img_w)
        # so TorchScript sees integer *literals* in forward(), not attribute
        # loads, producing static ONNX Slice constants that SNPE accepts.
        self.dec3 = make_decoder_block(f * 16 + f * 8, f * 8,  3, img_h, img_w, dropout_p=0.1)
        self.dec2 = make_decoder_block(f * 8  + f * 4, f * 4,  2, img_h, img_w, dropout_p=0.1)
        self.dec1 = make_decoder_block(f * 4  + f * 2, f * 2,  1, img_h, img_w, dropout_p=0.1)
        self.dec0 = make_decoder_block(f * 2  + f,      f,     0, img_h, img_w, dropout_p=0.1)

        # Output head: 1-channel sigmoid mask
        self.head = nn.Sequential(
            nn.Conv2d(f, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0, x = self.enc0(x)
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        x = self.dec0(x, skip0)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
#  ROI Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ROIExtractor(nn.Module):
    """
    Applies the U-Net binary mask to the raw spectrogram and resizes to
    the classifier's expected input size (224×224).

    This runs on every training batch — no patches are written to disk.

    Strategies
    ----------
    'multiply' (default)
        roi = mask × spectrogram  → background energy zeroed out, then resize.
        Fully differentiable w.r.t. the mask. SNPE-compatible.

    'bbox'
        Crop to the tight bounding box of the mask, then resize.
        Provides a larger effective zoom on the drone signal region, but
        is not differentiable w.r.t. mask boundaries (indexing op).
    """

    def __init__(
        self,
        output_size: tuple = (224, 224),
        threshold  : float = 0.5,
        strategy   : str   = "multiply",
    ):
        super().__init__()
        self.output_size = output_size
        self.threshold   = threshold
        self.strategy    = strategy

    @torch.no_grad()
    def forward(
        self,
        spectrogram: torch.Tensor,   # (B, C, H, W)  raw normalised spectrogram
        mask       : torch.Tensor,   # (B, 1, H, W)  U-Net sigmoid output
    ) -> torch.Tensor:               # (B, C, 224, 224)

        binary = (mask >= self.threshold).float()

        if self.strategy == "multiply":
            roi = spectrogram * binary
            return F.interpolate(roi, size=self.output_size,
                                 mode="bilinear", align_corners=False)

        elif self.strategy == "bbox":
            B, C, H, W = spectrogram.shape
            out = []
            for b in range(B):
                m    = binary[b, 0]
                rows = m.any(dim=1).nonzero(as_tuple=False)
                cols = m.any(dim=0).nonzero(as_tuple=False)
                if rows.numel() == 0 or cols.numel() == 0:
                    patch = spectrogram[b]   # fallback: full frame
                else:
                    r0, r1 = int(rows[0]),  int(rows[-1])  + 1
                    c0, c1 = int(cols[0]),  int(cols[-1])  + 1
                    patch  = spectrogram[b, :, r0:r1, c0:c1]
                patch = F.interpolate(
                    patch.unsqueeze(0), size=self.output_size,
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
                out.append(patch)
            return torch.stack(out, dim=0)

        else:
            raise ValueError(f"Unknown ROI strategy: {self.strategy!r}")


# ─────────────────────────────────────────────────────────────────────────────
#  Proxy mask  (weak supervision — no GT mask files required)
# ─────────────────────────────────────────────────────────────────────────────

def build_proxy_mask(
    images       : torch.Tensor,   # (B, C, H, W) normalised spectrogram
    labels       : torch.Tensor,   # (B,) integer class labels
    no_drone_idx : int,            # label index for NO_DRONE class
    threshold    : float = 0.5,
) -> torch.Tensor:                 # (B, 1, H, W) binary float32
    """
    Constructs the U-Net supervision target on-the-fly from the spectrogram
    energy — no ground-truth mask PNG files are needed.

    Drone samples
        Channel-mean energy is min-max normalised per sample, then
        binarised at `threshold`. High-energy bins (drone signal) → 1,
        low-energy bins (noise floor) → 0.

    NO_DRONE samples (Strategy 4)
        All-zero mask. Telling the U-Net there is nothing to localise for
        background-only frames prevents it from hallucinating signal regions.
    """
    # Per-sample energy: collapse channels → (B, 1, H, W)
    energy = images.mean(dim=1, keepdim=True)

    # Min-max normalise each sample independently
    e_flat = energy.flatten(1)
    e_min  = e_flat.min(1)[0].view(-1, 1, 1, 1)
    e_max  = e_flat.max(1)[0].view(-1, 1, 1, 1)
    normed = (energy - e_min) / (e_max - e_min + 1e-8)

    gt_mask = (normed > threshold).float()

    # Zero the mask for any NO_DRONE sample in the batch
    no_drone = (labels == no_drone_idx)
    if no_drone.any():
        gt_mask[no_drone] = 0.0

    return gt_mask   # (B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone U-Net training
# ─────────────────────────────────────────────────────────────────────────────

def train_unet(
    train_loader,
    val_loader,
    no_drone_idx : int,
    ckpt_dir     : str   = "checkpoints",
    base_filters : int   = 32,
    in_channels  : int   = 3,
    img_h        : int   = 256,
    img_w        : int   = 512,
    epochs       : int   = 30,
    lr           : float = 1e-3,
    weight_decay : float = 1e-4,
    proxy_threshold: float = 0.5,
    device_str   : str   = "auto",
    log_interval : int   = 50,
):
    """
    Train DroneROIUNet independently using only BCE loss on the proxy mask.
    No classifier is involved — this focuses purely on mask quality.

    The resulting checkpoint is loaded by Stage 2, which can then either
    freeze these weights or continue fine-tuning them end-to-end.

    Args
    ----
    train_loader / val_loader
        DataLoaders yielding (spectrogram_tensor, class_label) batches —
        the same loaders produced by drone_dataloader.build_dataloaders().
        Labels are only used to zero NO_DRONE masks; no classification loss.
    no_drone_idx
        Integer label index of the NO_DRONE class from meta["class_to_idx"].
    """
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto" else torch.device(device_str)
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    model = DroneROIUNet(in_channels, base_filters, img_h, img_w).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    best_val_loss = float("inf")
    best_ckpt     = os.path.join(ckpt_dir, "unet_best.pt")

    print(f"\n[Stage 1] Training U-Net on {device}")
    print(f"  Params  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs  : {epochs}   LR: {lr}   Base filters: {base_filters}\n")

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                gt_mask = build_proxy_mask(
                    images, labels, no_drone_idx, proxy_threshold
                )

            optimizer.zero_grad(set_to_none=True)
            pred_mask = model(images)
            loss      = criterion(pred_mask, gt_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % log_interval == 0:
                avg = train_loss / ((batch_idx + 1) * train_loader.batch_size)
                print(f"  Ep {epoch:03d} | batch {batch_idx+1:4d} | "
                      f"train_bce={avg:.4f}")

        train_loss /= n_train

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                gt_mask   = build_proxy_mask(
                    images, labels, no_drone_idx, proxy_threshold
                )
                pred_mask = model(images)
                val_loss += criterion(pred_mask, gt_mask).item() * images.size(0)
        val_loss /= n_val
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"train_bce={train_loss:.4f}  val_bce={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_loss"    : val_loss,
                "config": dict(
                    in_channels=in_channels, base_filters=base_filters,
                    img_h=img_h, img_w=img_w,
                ),
            }, best_ckpt)
            print(f"  ✓ Saved best U-Net checkpoint  (val_bce={val_loss:.4f})")

    print(f"\n[Stage 1] Done. Best checkpoint: {best_ckpt}")
    return best_ckpt


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint loader
# ─────────────────────────────────────────────────────────────────────────────

def load_unet(ckpt_path: str, device=None) -> DroneROIUNet:
    """
    Reconstruct DroneROIUNet from a Stage-1 checkpoint.
    Called by Stage 2 when loading pretrained U-Net weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt  = torch.load(ckpt_path, map_location=device)
    cfg   = ckpt["config"]
    model = DroneROIUNet(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[Stage 1] Loaded U-Net from {ckpt_path}  "
          f"(epoch {ckpt['epoch']}, val_bce={ckpt['val_loss']:.4f})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  CLI  (optional standalone entry point)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: Train U-Net segmentation on STFT spectrograms"
    )
    parser.add_argument("--root",          default="output_spectrograms/")
    parser.add_argument("--subsets",       nargs="+", default=["BLUE", "BOTH", "CLEAN", "WIFI"])
    parser.add_argument("--img_size",      nargs=2, type=int, default=[256, 512])
    parser.add_argument("--base_filters",  type=int,   default=32)
    parser.add_argument("--in_channels",   type=int,   default=3)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--workers",       type=int,   default=4)
    parser.add_argument("--ckpt_dir",      default="checkpoints/")
    parser.add_argument("--log_interval",  type=int,   default=50)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    # Import dataloader only when running standalone
    from drone_dataloader import build_dataloaders

    train_loader, val_loader, _, meta = build_dataloaders(
        root        = args.root,
        subsets     = args.subsets,
        img_size    = tuple(args.img_size),
        batch_size  = args.batch_size,
        num_workers = args.workers,
        seed        = args.seed,
    )
    no_drone_idx = meta["class_to_idx"].get("NO_DRONE", -1)

    train_unet(
        train_loader   = train_loader,
        val_loader     = val_loader,
        no_drone_idx   = no_drone_idx,
        ckpt_dir       = args.ckpt_dir,
        base_filters   = args.base_filters,
        in_channels    = args.in_channels,
        img_h          = args.img_size[0],
        img_w          = args.img_size[1],
        epochs         = args.epochs,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        log_interval   = args.log_interval,
    )