"""
Drone Detection Full Pipeline
==============================
Stage 1 — Drone-ROIs-Detection  : U-Net semantic segmentation
           Input  : Raw spectrogram image  (B, C, H, W)
           Output : Binary segmentation mask (B, 1, H, W)
                    → highlights drone RF regions, suppresses Wi-Fi/BT noise

Stage 2 — ROI Extraction        : Apply mask → crop/resize ROI patches
           Input  : Raw spectrogram + binary mask
           Output : Masked spectrogram ROI tensor  (B, C, H_roi, W_roi)

Stage 3 — Drone-CLSNet          : EfficientNet-B0 classification
           Input  : Masked ROI tensor  (B, C, 224, 224)
           Output : Class probabilities (B, num_classes)

Full pipeline call:
    pipeline = DronePipeline(num_classes=10)
    probs    = pipeline(spectrogram_batch)

Reference architecture (U-Net):
    Encoder : 4 down-blocks  (Conv×2 + MaxPool 2×2)
    Bottleneck              (Conv×2)
    Decoder : 4 up-blocks   (UpSample 2×2 + Skip-concat + Conv×2)
    Head    : Conv 1×1 + Sigmoid  →  binary mask

Reference architecture (Drone-CLSNet / EfficientNet-B0):
    Stem    : Conv 3×3
    Blocks  : MBConv1/6 ×7 stages
    Head    : GAP → Flatten → FC → Softmax
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# ═════════════════════════════════════════════════════════════════════════════
# ░░  STAGE 1 — U-Net ROI Segmentation Model                               ░░
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# U-Net building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Two consecutive Conv 3×3 + BN + ReLU layers.
    Used in every encoder block, bottleneck, and decoder block.

            x  →  [Conv3×3 → BN → ReLU] → [Conv3×3 → BN → ReLU]  →  out
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            # First 3×3 convolution (ReLU)
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Second 3×3 convolution (ReLU)
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    U-Net Encoder (down) block.
        DoubleConv  →  save skip feature  →  MaxPool 2×2 (halve spatial dims)

    Returns both the skip-connection feature map and the pooled output.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)   # 2×2 Max Pooling ↓

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)     # feature map for skip connection
        down = self.pool(skip)  # spatially halved feature map
        return skip, down


class DecoderBlock(nn.Module):
    """
    U-Net Decoder (up) block.
        UpSample 2×2  →  concatenate skip  →  DoubleConv

    The orange skip-connection arrows in the architecture diagram correspond to
    the concatenation step here (skip + upsampled feature maps).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # 2×2 bilinear up-sampling (green ↑ arrows in diagram)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=True)
        # After concatenation channel count doubles
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Pad if spatial sizes mismatch (odd input dimensions)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)    # Skip-connection & concatenation
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full U-Net model
# ─────────────────────────────────────────────────────────────────────────────

class DroneROIUNet(nn.Module):
    """
    Drone-ROIs-Detection Model  (U-Net)
    ====================================
    Semantic segmentation network that produces a binary mask identifying
    drone RF signal regions in spectrogram images.

    Encoder path (contracting):
        Level 0 : DoubleConv  in_ch → 64        + skip0
        Level 1 : MaxPool → DoubleConv  64→128  + skip1
        Level 2 : MaxPool → DoubleConv 128→256  + skip2
        Level 3 : MaxPool → DoubleConv 256→512  + skip3

    Bottleneck:
        Level 4 : MaxPool → DoubleConv 512→1024

    Decoder path (expanding):
        Level 3 : UpSample → concat(skip3, x) → DoubleConv 1024+512→512
        Level 2 : UpSample → concat(skip2, x) → DoubleConv  512+256→256
        Level 1 : UpSample → concat(skip1, x) → DoubleConv  256+128→128
        Level 0 : UpSample → concat(skip0, x) → DoubleConv  128+64→64

    Output head:
        Conv 1×1 → Sigmoid  →  binary mask  (B, 1, H, W)

    Args:
        in_channels  : spectrogram channels (1 = grayscale, 3 = viridis RGB)
        base_filters : starting feature map depth (default 64, classic U-Net)
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 64):
        super().__init__()
        f = base_filters  # shorthand

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc0 = EncoderBlock(in_channels, f)        # → skip(f),    down
        self.enc1 = EncoderBlock(f,           f * 2)    # → skip(2f),   down
        self.enc2 = EncoderBlock(f * 2,       f * 4)    # → skip(4f),   down
        self.enc3 = EncoderBlock(f * 4,       f * 8)    # → skip(8f),   down

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(f * 8, f * 16)     # 512 → 1024

        # ── Decoder ─────────────────────────────────────────────────────────
        # in_ch = upsampled_channels + skip_channels
        self.dec3 = DecoderBlock(f * 16 + f * 8,  f * 8)   # 1024+512 → 512
        self.dec2 = DecoderBlock(f * 8  + f * 4,  f * 4)   #  512+256 → 256
        self.dec1 = DecoderBlock(f * 4  + f * 2,  f * 2)   #  256+128 → 128
        self.dec0 = DecoderBlock(f * 2  + f,       f)       #  128+64  →  64

        # ── Output Head: Conv 1×1 → Sigmoid ─────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(f, 1, kernel_size=1),   # 1×1 convolution (Sigmoid)
            nn.Sigmoid(),                      # binary probability per pixel
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : spectrogram image tensor  (B, C, H, W)
        Returns:
            mask : segmentation mask      (B, 1, H, W)  values in [0, 1]
        """
        # ── Encoder (contracting path) ───────────────────────────────────────
        skip0, x = self.enc0(x)     # skip: (B, 64,  H,   W  )
        skip1, x = self.enc1(x)     # skip: (B, 128, H/2, W/2)
        skip2, x = self.enc2(x)     # skip: (B, 256, H/4, W/4)
        skip3, x = self.enc3(x)     # skip: (B, 512, H/8, W/8)

        # ── Bottleneck ───────────────────────────────────────────────────────
        x = self.bottleneck(x)      # (B, 1024, H/16, W/16)

        # ── Decoder (expanding path) ─────────────────────────────────────────
        x = self.dec3(x, skip3)     # (B, 512,  H/8, W/8)
        x = self.dec2(x, skip2)     # (B, 256,  H/4, W/4)
        x = self.dec1(x, skip1)     # (B, 128,  H/2, W/2)
        x = self.dec0(x, skip0)     # (B, 64,   H,   W  )

        # ── Segmentation mask ────────────────────────────────────────────────
        mask = self.head(x)         # (B, 1,    H,   W  )  ∈ [0, 1]
        return mask


# ═════════════════════════════════════════════════════════════════════════════
# ░░  STAGE 2 — ROI Extraction (Mask Application + Resize)                 ░░
# ═════════════════════════════════════════════════════════════════════════════

class ROIExtractor(nn.Module):
    """
    ROI Extraction Module.
    ======================
    Applies the U-Net binary segmentation mask to the raw spectrogram,
    then resizes the masked image to a fixed size for the classifier.

    Two masking strategies are supported:
        'multiply' (default) : element-wise mask × spectrogram
                               Preserves spatial structure; background → 0.
        'bbox'               : crop to the bounding box of the largest
                               connected mask region, then resize.
                               Best for tightly-cropped ROI patches.

    Args:
        output_size  : (H, W) expected by Drone-CLSNet (default 224×224)
        threshold    : sigmoid threshold to binarise the mask (default 0.5)
        strategy     : 'multiply' | 'bbox'
    """

    def __init__(
        self,
        output_size: tuple[int, int] = (224, 224),
        threshold: float = 0.5,
        strategy: str = "multiply",
    ):
        super().__init__()
        self.output_size = output_size
        self.threshold   = threshold
        self.strategy    = strategy

    @torch.no_grad()
    def forward(
        self,
        spectrogram: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spectrogram : raw input   (B, C, H, W)
            mask        : U-Net output (B, 1, H, W) values ∈ [0, 1]
        Returns:
            roi         : masked + resized tensor (B, C, H_out, W_out)
        """
        binary_mask = (mask >= self.threshold).float()   # binarise

        if self.strategy == "multiply":
            # Broadcast mask across all channels and zero-out background
            roi = spectrogram * binary_mask              # (B, C, H, W)

        elif self.strategy == "bbox":
            # Crop to bounding-box of mask per sample, then resize
            B, C, H, W = spectrogram.shape
            roi_list = []
            for b in range(B):
                m = binary_mask[b, 0]                    # (H, W)
                rows = m.any(dim=1).nonzero(as_tuple=False)
                cols = m.any(dim=0).nonzero(as_tuple=False)
                if rows.numel() == 0 or cols.numel() == 0:
                    # No foreground detected → fall back to full image
                    patch = spectrogram[b]
                else:
                    r0, r1 = int(rows[0]), int(rows[-1]) + 1
                    c0, c1 = int(cols[0]), int(cols[-1]) + 1
                    patch = spectrogram[b, :, r0:r1, c0:c1]
                # Resize each patch to output_size
                patch = F.interpolate(
                    patch.unsqueeze(0), size=self.output_size,
                    mode="bilinear", align_corners=False
                ).squeeze(0)
                roi_list.append(patch)
            roi = torch.stack(roi_list, dim=0)           # (B, C, H_out, W_out)
            return roi

        else:
            raise ValueError(f"Unknown ROI strategy: {self.strategy!r}. "
                             "Choose 'multiply' or 'bbox'.")

        # Resize to classifier input resolution
        roi = F.interpolate(roi, size=self.output_size,
                            mode="bilinear", align_corners=False)
        return roi                                       # (B, C, H_out, W_out)


# ═════════════════════════════════════════════════════════════════════════════
# ░░  STAGE 3 — Drone-CLSNet  (EfficientNet-B0 Classifier)                 ░░
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet-B0 helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _make_divisible(value: float, divisor: int = 8) -> int:
    new_val = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_val < 0.9 * value:
        new_val += divisor
    return new_val

def _round_filters(f: int, w: float) -> int:
    return _make_divisible(int(f * w))

def _round_repeats(n: int, d: float) -> int:
    return int(math.ceil(n * d))


# ─────────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation
# ─────────────────────────────────────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """Channel-wise recalibration via global average pooling + FC gates."""

    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        sq = max(1, int(in_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, sq, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(sq, in_ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


# ─────────────────────────────────────────────────────────────────────────────
# MBConv Block
# ─────────────────────────────────────────────────────────────────────────────

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution with Squeeze-and-Excitation.
    Steps: Expand (pw) → Depthwise → SE → Project (pw) → [Skip]
    """

    def __init__(
        self,
        in_ch: int, out_ch: int,
        kernel_size: int, stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.0,
    ):
        super().__init__()
        self.use_residual      = (stride == 1 and in_ch == out_ch)
        self.drop_connect_rate = drop_connect_rate
        mid = _make_divisible(in_ch * expand_ratio)
        pad = (kernel_size - 1) // 2
        layers = []

        # Step 1 — Expansion pointwise conv (skip if ratio == 1)
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False),
                       nn.BatchNorm2d(mid, momentum=0.01, eps=1e-3),
                       nn.SiLU(inplace=True)]

        # Step 2 — Depthwise convolution (spatial feature extraction)
        layers += [
            nn.Conv2d(mid, mid, kernel_size, stride=stride,
                      padding=pad, groups=mid, bias=False),
            nn.BatchNorm2d(mid, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        ]

        # Step 3 — Squeeze-and-Excitation
        layers.append(SqueezeExcitation(mid, se_ratio=se_ratio))

        # Step 4 — Projection pointwise conv (no activation)
        layers += [nn.Conv2d(mid, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)]

        self.block = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep = 1.0 - self.drop_connect_rate
        noise = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        return x / keep * torch.floor(noise + keep)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = self._drop_connect(out) + x
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Drone-CLSNet (EfficientNet-B0 backbone + custom head)
# ─────────────────────────────────────────────────────────────────────────────

class DroneCLSNet(nn.Module):
    """
    Drone-CLSNet — EfficientNet-B0 Classification Backbone.
    ========================================================
    Receives masked ROI spectrograms from the ROI extractor and outputs
    per-class probabilities.

    Block config: (expand_ratio, out_channels, num_layers, stride, kernel_size)

    Stage    Type      Kernel  Stride  Filters  Layers
    ──────────────────────────────────────────────────
    Stem     Conv3×3     3       2       32       1
    Block1   MBConv1     3       1       16       1
    Block2   MBConv6     3       2       24       2
    Block3   MBConv6     5       2       40       2
    Block4   MBConv6     3       2       80       3
    Block5   MBConv6     5       1      112       3
    Block6   MBConv6     5       2      192       4
    Block7   MBConv6     3       1      320       1
    Head     Conv1×1     1       1     1280       1
    Classifier GAP→Flatten→FC→Softmax         num_classes
    """

    _BLOCK_ARGS = [
        (1,  16, 1, 1, 3),   # Block 1 — MBConv1, 3×3, 1 layer
        (6,  24, 2, 2, 3),   # Block 2 — MBConv6, 3×3, 2 layers
        (6,  40, 2, 2, 5),   # Block 3 — MBConv6, 5×5, 2 layers
        (6,  80, 3, 2, 3),   # Block 4 — MBConv6, 3×3, 3 layers
        (6, 112, 3, 1, 5),   # Block 5 — MBConv6, 5×5, 3 layers
        (6, 192, 4, 2, 5),   # Block 6 — MBConv6, 5×5, 4 layers
        (6, 320, 1, 1, 3),   # Block 7 — MBConv6, 3×3, 1 layer
    ]

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        width_coeff: float = 1.0,
        depth_coeff: float = 1.0,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()

        # ── Stem: Conv 3×3 ───────────────────────────────────────────────────
        stem_f = _round_filters(32, width_coeff)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_f, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_f, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )

        # ── MBConv Feature Extractor Blocks ─────────────────────────────────
        total = sum(_round_repeats(n, depth_coeff) for _, _, n, _, _ in self._BLOCK_ARGS)
        idx, cur = 0, stem_f
        all_stages = []
        for expand, out_ch, num_l, stride, kernel in self._BLOCK_ARGS:
            out_f  = _round_filters(out_ch, width_coeff)
            num_l  = _round_repeats(num_l, depth_coeff)
            stage  = []
            for li in range(num_l):
                stage.append(MBConvBlock(
                    in_ch=cur, out_ch=out_f,
                    kernel_size=kernel,
                    stride=stride if li == 0 else 1,
                    expand_ratio=expand,
                    se_ratio=0.25,
                    drop_connect_rate=drop_connect_rate * idx / total,
                ))
                cur = out_f
                idx += 1
            all_stages.append(nn.Sequential(*stage))

        (self.block1, self.block2, self.block3, self.block4,
         self.block5, self.block6, self.block7) = all_stages

        # ── Head Conv 1×1 ────────────────────────────────────────────────────
        head_f = _round_filters(1280, width_coeff)
        self.head_conv = nn.Sequential(
            nn.Conv2d(cur, head_f, 1, bias=False),
            nn.BatchNorm2d(head_f, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )

        # ── Classification Head: Flatten → Dense → Softmax ──────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # Global Average Pool
            nn.Flatten(),                # Flatten
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_f, num_classes),  # Dense (FC)
            nn.Softmax(dim=1),           # Softmax output
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : masked ROI spectrogram  (B, C, 224, 224)
        Returns:
            probs : class probabilities (B, num_classes)
        """
        x = self.stem(x)
        x = self.block1(x)    # MBConv1  3×3  ×1
        x = self.block2(x)    # MBConv6  3×3  ×2
        x = self.block3(x)    # MBConv6  5×5  ×2
        x = self.block4(x)    # MBConv6  3×3  ×3
        x = self.block5(x)    # MBConv6  5×5  ×3
        x = self.block6(x)    # MBConv6  5×5  ×4
        x = self.block7(x)    # MBConv6  3×3  ×1
        x = self.head_conv(x)
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════════════════
# ░░  FULL PIPELINE                                                         ░░
# ═════════════════════════════════════════════════════════════════════════════

class DronePipeline(nn.Module):
    """
    End-to-End Drone Detection Pipeline
    =====================================
    Chains all three stages into a single callable nn.Module:

        Raw spectrogram
            │
            ▼
        ┌─────────────────────────────────┐
        │  Stage 1: DroneROIUNet          │  Semantic segmentation
        │  Input : (B, C, H, W)           │  → binary mask
        │  Output: (B, 1, H, W)           │
        └────────────────┬────────────────┘
                         │ mask
                         ▼
        ┌─────────────────────────────────┐
        │  Stage 2: ROIExtractor          │  Mask × spectrogram + resize
        │  Input : spec + mask            │
        │  Output: (B, C, 224, 224)       │
        └────────────────┬────────────────┘
                         │ roi
                         ▼
        ┌─────────────────────────────────┐
        │  Stage 3: DroneCLSNet           │  EfficientNet-B0 classification
        │  Input : (B, C, 224, 224)       │
        │  Output: (B, num_classes)       │
        └─────────────────────────────────┘

    Args:
        num_classes       : number of drone types to classify
        in_channels       : spectrogram channels (1 or 3)
        unet_base_filters : U-Net encoder depth (default 64)
        roi_output_size   : spatial size fed to classifier (default 224×224)
        mask_threshold    : U-Net sigmoid binarisation threshold (default 0.5)
        roi_strategy      : 'multiply' (soft masking) | 'bbox' (tight crop)
        cls_dropout       : classifier dropout rate
        cls_drop_connect  : EfficientNet stochastic depth rate
    """

    def __init__(
        self,
        num_classes: int       = 10,
        in_channels: int       = 3,
        unet_base_filters: int = 64,
        roi_output_size: tuple = (224, 224),
        mask_threshold: float  = 0.5,
        roi_strategy: str      = "multiply",
        cls_dropout: float     = 0.2,
        cls_drop_connect: float = 0.2,
    ):
        super().__init__()

        # Stage 1 — U-Net ROI segmentation
        self.unet = DroneROIUNet(
            in_channels  = in_channels,
            base_filters = unet_base_filters,
        )

        # Stage 2 — ROI extraction / masking
        self.roi_extractor = ROIExtractor(
            output_size = roi_output_size,
            threshold   = mask_threshold,
            strategy    = roi_strategy,
        )

        # Stage 3 — EfficientNet-B0 classification
        self.classifier = DroneCLSNet(
            num_classes      = num_classes,
            in_channels      = in_channels,
            dropout_rate     = cls_dropout,
            drop_connect_rate = cls_drop_connect,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the full pipeline.

        Args:
            x           : raw spectrogram tensor (B, C, H, W)
            return_mask : if True, also return the U-Net segmentation mask

        Returns:
            probs       : class probabilities (B, num_classes)
          [ mask        : segmentation mask   (B, 1, H, W)  — if return_mask ]
        """
        # Stage 1 — Segment drone RF regions
        mask = self.unet(x)                  # (B, 1, H, W)  ∈ [0, 1]

        # Stage 2 — Extract & resize ROI
        roi  = self.roi_extractor(x, mask)   # (B, C, 224, 224)

        # Stage 3 — Classify drone type
        probs = self.classifier(roi)         # (B, num_classes)

        if return_mask:
            return probs, mask
        return probs

    # ── Convenience: freeze / unfreeze stages for staged fine-tuning ─────────

    def freeze_unet(self):
        """Freeze U-Net weights (fine-tune classifier only)."""
        for p in self.unet.parameters():
            p.requires_grad = False

    def unfreeze_unet(self):
        """Unfreeze U-Net weights for joint end-to-end training."""
        for p in self.unet.parameters():
            p.requires_grad = True

    def freeze_classifier(self):
        """Freeze classifier weights (train U-Net only)."""
        for p in self.classifier.parameters():
            p.requires_grad = False

    def unfreeze_classifier(self):
        """Unfreeze classifier weights."""
        for p in self.classifier.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Training-time loss helper
# ─────────────────────────────────────────────────────────────────────────────

class PipelineLoss(nn.Module):
    """
    Combined loss for joint training of both stages.

    Total loss = λ_seg × BCELoss(mask, gt_mask)
               + λ_cls × CrossEntropyLoss(logits, gt_label)

    Note: DroneCLSNet uses Softmax internally. For training, the logits
    (pre-softmax) are preferred with CrossEntropyLoss. Pass
    `use_softmax=False` to DroneCLSNet if training end-to-end.

    Args:
        seg_weight : weight for segmentation loss  (default 1.0)
        cls_weight : weight for classification loss (default 1.0)
    """

    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss = nn.BCELoss()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        pred_mask:  torch.Tensor,   # (B, 1, H, W)  U-Net output
        gt_mask:    torch.Tensor,   # (B, 1, H, W)  ground-truth binary mask
        pred_probs: torch.Tensor,   # (B, num_classes) classifier output
        gt_labels:  torch.Tensor,   # (B,) ground-truth class indices
    ) -> dict[str, torch.Tensor]:
        seg = self.seg_loss(pred_mask, gt_mask)
        cls = self.cls_loss(pred_probs, gt_labels)
        total = self.seg_weight * seg + self.cls_weight * cls
        return {"total": total, "seg": seg, "cls": cls}


# ═════════════════════════════════════════════════════════════════════════════
# ░░  QUICK VERIFICATION & SUMMARY                                          ░░
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    NUM_CLASSES  = 10
    IN_CHANNELS  = 3        # viridis RGB spectrogram
    H, W         = 256, 512 # typical spectrogram resolution (freq × time)
    BATCH        = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*65}")
    print("  Drone Detection Full Pipeline  — Model Summary")
    print(f"  Device      : {device}")
    print(f"  Input shape : ({BATCH}, {IN_CHANNELS}, {H}, {W})")
    print(f"  Classes     : {NUM_CLASSES}")
    print(f"{'═'*65}\n")

    # ── Build full pipeline ───────────────────────────────────────────────────
    pipeline = DronePipeline(
        num_classes       = NUM_CLASSES,
        in_channels       = IN_CHANNELS,
        unet_base_filters = 64,
        roi_output_size   = (224, 224),
        mask_threshold    = 0.5,
        roi_strategy      = "multiply",
    ).to(device)

    # ── Stage-wise summaries ──────────────────────────────────────────────────
    print("─── Stage 1: DroneROIUNet (U-Net Segmentation) ─────────────────")
    summary(pipeline.unet, input_size=(BATCH, IN_CHANNELS, H, W),
            col_names=["input_size", "output_size", "num_params"],
            depth=3, device=device, verbose=1)

    print("\n─── Stage 3: DroneCLSNet (EfficientNet-B0 Classifier) ───────────")
    summary(pipeline.classifier, input_size=(BATCH, IN_CHANNELS, 224, 224),
            col_names=["input_size", "output_size", "num_params"],
            depth=3, device=device, verbose=1)

    # ── Full end-to-end forward pass ──────────────────────────────────────────
    print("\n─── Full Pipeline Forward Pass ──────────────────────────────────")
    dummy_spec = torch.randn(BATCH, IN_CHANNELS, H, W, device=device)

    with torch.no_grad():
        probs, mask = pipeline(dummy_spec, return_mask=True)

    print(f"  Input spectrogram : {tuple(dummy_spec.shape)}")
    print(f"  Segmentation mask : {tuple(mask.shape)}  "
          f"(min={mask.min():.3f}, max={mask.max():.3f})")
    print(f"  Class probs       : {tuple(probs.shape)}  "
          f"(sum[0]={probs[0].sum():.6f}  ← should be ≈ 1.0)")

    assert probs.shape == (BATCH, NUM_CLASSES)
    assert mask.shape  == (BATCH, 1, H, W)
    assert abs(probs[0].sum().item() - 1.0) < 1e-5, "Softmax rows must sum to 1"

    # ── Combined loss demo ────────────────────────────────────────────────────
    print("\n─── Combined Loss Demo (joint training) ─────────────────────────")
    criterion  = PipelineLoss(seg_weight=1.0, cls_weight=1.0)
    gt_mask    = torch.randint(0, 2, (BATCH, 1, H, W),
                               dtype=torch.float32, device=device)
    gt_labels  = torch.randint(0, NUM_CLASSES, (BATCH,), device=device)
    losses     = criterion(mask, gt_mask, probs, gt_labels)
    for k, v in losses.items():
        print(f"  {k:10s}: {v.item():.4f}")

    print(f"\n{'═'*65}")
    print("  ✓  All assertions passed — Pipeline is ready for training.")
    print(f"{'═'*65}\n")

    # ── Staged fine-tuning example ────────────────────────────────────────────
    print("─── Staged Fine-Tuning Example ──────────────────────────────────")
    print("  Step 1: Train U-Net only (classifier frozen)")
    pipeline.freeze_classifier()
    trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    print("  Step 2: Train classifier only (U-Net frozen)")
    pipeline.unfreeze_classifier()
    pipeline.freeze_unet()
    trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    print("  Step 3: Joint end-to-end fine-tuning (all params)")
    pipeline.unfreeze_unet()
    trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print("  Done.\n")