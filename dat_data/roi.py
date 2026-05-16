"""
roi.py  (refactored)
=====================
Changes from previous version
------------------------------
Strategy 2 — Dropout2d added to U-Net bottleneck and decoder blocks.
Strategy 3 — unet_base_filters default reduced from 64 → 32,
             cutting U-Net parameter count from ~31 M to ~8 M.
Strategy 4 (partial) — Softmax removed from DroneCLSNet classifier head.
             CrossEntropyLoss in PipelineLoss now receives raw logits,
             which is the correct PyTorch usage and fixes silent gradient
             corruption that occurred with double-softmax.

Architecture overview
---------------------
Stage 1  DroneROIUNet      U-Net semantic segmentation  → binary mask
Stage 2  ROIExtractor      mask × spectrogram + resize  → ROI patch
Stage 3  DroneCLSNet       EfficientNet-B0 backbone     → class logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# ░░  STAGE 1 — U-Net ROI Segmentation Model                               ░░
# ═════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """
    Two consecutive Conv 3×3 + BN + ReLU layers.

    Strategy 2: optional Dropout2d inserted between the two conv layers
    so the U-Net cannot memorise the training set.  Default p=0.0 keeps
    encoder blocks unchanged; set p>0 for bottleneck / decoder blocks.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            # Dropout2d zeros entire feature-map channels (spatially coherent)
            layers.append(nn.Dropout2d(p=dropout_p))
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv → skip + MaxPool 2×2."""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """
    Upsample 2×2 → pad to match skip → cat(skip) → DoubleConv.

    SNPE/QNN compatibility fix:
        The original code used F.interpolate(x, size=skip.shape[2:]) to
        handle spatial size mismatches. This produces a dynamic shape op
        that SNPE's shape inference cannot resolve statically, causing the
        'Inconsistency in dynamic axis shapes' error during DLC conversion.

        Replaced with F.pad() using statically-computable diff values.
        F.pad with constant offsets is fully supported by SNPE opset 13.
        With img_size=(256,512) both divisible by 16, the mismatch is at
        most 1 pixel per side, so padding cost is negligible.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Pad x to match skip spatial dims if there is a 1-pixel mismatch.
        # Using F.pad with integer offsets produces a static shape op
        # that SNPE can trace correctly -- unlike F.interpolate(size=skip.shape).
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,   # left, right
                diff_h // 2, diff_h - diff_h // 2,   # top,  bottom
            ])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class DroneROIUNet(nn.Module):
    """
    Drone-ROIs-Detection Model  (U-Net)

    Strategy 2 — Dropout2d rates:
        Encoder blocks  : 0.0   (no dropout — preserve spatial features)
        Bottleneck      : 0.3   (highest dropout — most abstract features)
        Decoder blocks  : 0.1   (light dropout — reconstruction path)

    Strategy 3 — base_filters default is now 32 (was 64).
        32  filters → ~8 M  parameters   ← new default
        64  filters → ~31 M parameters   ← old default
    Set base_filters=64 to restore the original capacity if needed.

    Args:
        in_channels  : 1 (grayscale) or 3 (viridis RGB)
        base_filters : starting feature depth (default 32)
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        f = base_filters

        # ── Encoder (no dropout — preserve low-level spatial features) ────────
        self.enc0 = EncoderBlock(in_channels, f,      dropout_p=0.0)
        self.enc1 = EncoderBlock(f,           f * 2,  dropout_p=0.0)
        self.enc2 = EncoderBlock(f * 2,       f * 4,  dropout_p=0.0)
        self.enc3 = EncoderBlock(f * 4,       f * 8,  dropout_p=0.0)

        # ── Bottleneck (strongest dropout — regularise abstract features) ──────
        # Strategy 2: Dropout2d(0.3) inside DoubleConv
        self.bottleneck = DoubleConv(f * 8, f * 16, dropout_p=0.3)

        # ── Decoder (light dropout — regularise reconstruction path) ──────────
        # Strategy 2: Dropout2d(0.1) inside each DecoderBlock's DoubleConv
        self.dec3 = DecoderBlock(f * 16 + f * 8, f * 8,  dropout_p=0.1)
        self.dec2 = DecoderBlock(f * 8  + f * 4, f * 4,  dropout_p=0.1)
        self.dec1 = DecoderBlock(f * 4  + f * 2, f * 2,  dropout_p=0.1)
        self.dec0 = DecoderBlock(f * 2  + f,      f,      dropout_p=0.1)

        # ── Output head: 1×1 conv → Sigmoid ──────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(f, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, C, H, W)
        Returns:
            mask : (B, 1, H, W)  values ∈ [0, 1]
        """
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


# ═════════════════════════════════════════════════════════════════════════════
# ░░  STAGE 2 — ROI Extraction  (unchanged)                                ░░
# ═════════════════════════════════════════════════════════════════════════════

class ROIExtractor(nn.Module):
    """
    Applies the U-Net binary mask to the raw spectrogram, then resizes
    to the classifier's expected input resolution.

    Strategies:
        'multiply' (default) : mask × spectrogram, background → 0
        'bbox'               : crop to bounding box of mask, then resize
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
        binary_mask = (mask >= self.threshold).float()

        if self.strategy == "multiply":
            roi = spectrogram * binary_mask

        elif self.strategy == "bbox":
            B, C, H, W = spectrogram.shape
            roi_list = []
            for b in range(B):
                m    = binary_mask[b, 0]
                rows = m.any(dim=1).nonzero(as_tuple=False)
                cols = m.any(dim=0).nonzero(as_tuple=False)
                if rows.numel() == 0 or cols.numel() == 0:
                    patch = spectrogram[b]
                else:
                    r0, r1 = int(rows[0]),  int(rows[-1]) + 1
                    c0, c1 = int(cols[0]),  int(cols[-1]) + 1
                    patch  = spectrogram[b, :, r0:r1, c0:c1]
                patch = F.interpolate(
                    patch.unsqueeze(0), size=self.output_size,
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
                roi_list.append(patch)
            return torch.stack(roi_list, dim=0)

        else:
            raise ValueError(f"Unknown ROI strategy: {self.strategy!r}")

        roi = F.interpolate(roi, size=self.output_size,
                            mode="bilinear", align_corners=False)
        return roi


# ═════════════════════════════════════════════════════════════════════════════
# ░░  STAGE 3 — DroneCLSNet  (EfficientNet-B0, Softmax REMOVED)            ░░
# ═════════════════════════════════════════════════════════════════════════════

def _make_divisible(value: float, divisor: int = 8) -> int:
    new_val = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_val < 0.9 * value:
        new_val += divisor
    return new_val

def _round_filters(f: int, w: float) -> int:
    return _make_divisible(int(f * w))

def _round_repeats(n: int, d: float) -> int:
    return int(math.ceil(n * d))


class SqueezeExcitation(nn.Module):
    """
    Channel-wise SE recalibration.

    SNPE compatibility fix:
        nn.AdaptiveAvgPool2d(1) is not supported by SNPE/QNN converters.
        Replaced with x.mean(dim=[2,3], keepdim=True) which exports as
        a ReduceMean ONNX op -- fully supported by SNPE opset 13.
        Weight layout is unchanged so existing checkpoints load correctly.
    """

    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        sq = max(1, int(in_ch * se_ratio))
        # No AdaptiveAvgPool2d -- global avg done in forward() via .mean()
        self.fc1  = nn.Conv2d(in_ch, sq, 1, bias=True)
        self.act  = nn.SiLU(inplace=True)
        self.fc2  = nn.Conv2d(sq, in_ch, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReduceMean over H,W -- static shape, SNPE compatible
        s = x.mean(dim=[2, 3], keepdim=True)   # (B, C, 1, 1)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


class MBConvBlock(nn.Module):
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

        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False),
                       nn.BatchNorm2d(mid, momentum=0.01, eps=1e-3),
                       nn.SiLU(inplace=True)]

        layers += [
            nn.Conv2d(mid, mid, kernel_size, stride=stride,
                      padding=pad, groups=mid, bias=False),
            nn.BatchNorm2d(mid, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        ]
        layers.append(SqueezeExcitation(mid, se_ratio=se_ratio))
        layers += [nn.Conv2d(mid, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)]

        self.block = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep  = 1.0 - self.drop_connect_rate
        noise = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        return x / keep * torch.floor(noise + keep)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = self._drop_connect(out) + x
        return out


class DroneCLSNet(nn.Module):
    """
    EfficientNet-B0 classification backbone.

    Strategy 4 — Softmax REMOVED from the classifier head.
    The model now returns raw logits (pre-softmax).
    CrossEntropyLoss in PipelineLoss applies log-softmax internally,
    which is numerically stable and the correct PyTorch pattern.

    At inference time, apply torch.softmax(logits, dim=1) manually
    if you need probability scores, or use torch.argmax(logits, dim=1)
    directly for the predicted class.
    """

    _BLOCK_ARGS = [
        (1,  16, 1, 1, 3),
        (6,  24, 2, 2, 3),
        (6,  40, 2, 2, 5),
        (6,  80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    ]

    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 3,
        width_coeff: float = 1.0,
        depth_coeff: float = 1.0,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()

        stem_f = _round_filters(32, width_coeff)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_f, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_f, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )

        total = sum(_round_repeats(n, depth_coeff)
                    for _, _, n, _, _ in self._BLOCK_ARGS)
        idx, cur = 0, stem_f
        all_stages = []
        for expand, out_ch, num_l, stride, kernel in self._BLOCK_ARGS:
            out_f = _round_filters(out_ch, width_coeff)
            num_l = _round_repeats(num_l, depth_coeff)
            stage = []
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

        head_f = _round_filters(1280, width_coeff)
        self.head_conv = nn.Sequential(
            nn.Conv2d(cur, head_f, 1, bias=False),
            nn.BatchNorm2d(head_f, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )

        # ── Strategy 4: Softmax REMOVED -- returns raw logits ────────────────
        # SNPE compatibility fix:
        #   AdaptiveAvgPool2d(1) removed from Sequential -- not supported by SNPE.
        #   Flatten removed from Sequential -- done implicitly after .mean() in forward().
        #   Global average pooling now done as x.mean(dim=[2,3]) in forward(),
        #   which exports as ReduceMean and is fully supported by SNPE opset 13.
        #   Checkpoint weights are unaffected -- Linear layer weights unchanged.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_f, num_classes),
            # nn.Softmax(dim=1)  <- REMOVED: CrossEntropyLoss expects logits
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
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, 224, 224)
        Returns:
            logits : (B, num_classes)  raw scores, NO softmax applied
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.head_conv(x)
        # Global average pool as ReduceMean -- SNPE compatible
        # replaces AdaptiveAvgPool2d(1) + Flatten from classifier Sequential
        x = x.mean(dim=[2, 3])    # (B, head_f)
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════════════════
# ░░  FULL PIPELINE                                                         ░░
# ═════════════════════════════════════════════════════════════════════════════

class DronePipeline(nn.Module):
    """
    End-to-end Drone Detection Pipeline.

    forward() returns:
        logits : (B, num_classes)  raw class scores  [NO softmax]
        mask   : (B, 1, H, W)     U-Net segmentation mask  [optional]

    At inference, convert logits to probabilities with:
        probs = torch.softmax(logits, dim=1)
    Or get the predicted class directly with:
        pred  = logits.argmax(dim=1)
    """

    def __init__(
        self,
        num_classes: int        = 8,
        in_channels: int        = 3,
        unet_base_filters: int  = 32,       # Strategy 3: default 32 (was 64)
        roi_output_size: tuple  = (224, 224),
        mask_threshold: float   = 0.5,
        roi_strategy: str       = "multiply",
        cls_dropout: float      = 0.2,
        cls_drop_connect: float = 0.2,
    ):
        super().__init__()

        self.unet = DroneROIUNet(
            in_channels  = in_channels,
            base_filters = unet_base_filters,   # Strategy 3
        )

        self.roi_extractor = ROIExtractor(
            output_size = roi_output_size,
            threshold   = mask_threshold,
            strategy    = roi_strategy,
        )

        self.classifier = DroneCLSNet(
            num_classes       = num_classes,
            in_channels       = in_channels,
            dropout_rate      = cls_dropout,
            drop_connect_rate = cls_drop_connect,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mask   = self.unet(x)                  # (B, 1, H, W)
        roi    = self.roi_extractor(x, mask)   # (B, C, 224, 224)
        logits = self.classifier(roi)          # (B, num_classes)  raw logits

        if return_mask:
            return logits, mask
        return logits

    def freeze_unet(self):
        for p in self.unet.parameters():
            p.requires_grad = False

    def unfreeze_unet(self):
        for p in self.unet.parameters():
            p.requires_grad = True

    def freeze_classifier(self):
        for p in self.classifier.parameters():
            p.requires_grad = False

    def unfreeze_classifier(self):
        for p in self.classifier.parameters():
            p.requires_grad = True


# ═════════════════════════════════════════════════════════════════════════════
# ░░  Loss                                                                  ░░
# ═════════════════════════════════════════════════════════════════════════════

class PipelineLoss(nn.Module):
    """
    Combined segmentation + classification loss.

    total = λ_seg × BCELoss(mask, gt_mask)
          + λ_cls × CrossEntropyLoss(logits, gt_labels)

    CrossEntropyLoss receives RAW LOGITS (no softmax applied beforehand).
    This is numerically stable and the correct PyTorch usage.
    """

    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss   = nn.BCELoss()
        self.cls_loss   = nn.CrossEntropyLoss()

    def forward(
        self,
        pred_mask:  torch.Tensor,   # (B, 1, H, W)  U-Net sigmoid output
        gt_mask:    torch.Tensor,   # (B, 1, H, W)  binary target
        pred_logits: torch.Tensor,  # (B, num_classes)  raw logits (no softmax)
        gt_labels:  torch.Tensor,   # (B,)  class indices
    ) -> dict[str, torch.Tensor]:
        seg   = self.seg_loss(pred_mask, gt_mask)
        cls   = self.cls_loss(pred_logits, gt_labels)
        total = self.seg_weight * seg + self.cls_weight * cls
        return {"total": total, "seg": seg, "cls": cls}