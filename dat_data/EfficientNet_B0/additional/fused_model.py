"""
fused_model.py
==============
Single-graph drone detection pipeline: spectrogram → class logits.

Architecture
------------
    spectrogram (B, 3, H, W)
         │
    ┌────▼────────────────────┐
    │  U-Net Encoder          │  4× EncoderBlock (Conv3×3, BN, ReLU, MaxPool)
    │  U-Net Bottleneck       │  DoubleConv with Dropout2d
    │  U-Net Decoder          │  4× DecoderBlock (ConvTranspose2d, cat skip)
    │  Mask head              │  Conv1×1 → raw logit (NO Sigmoid)
    └────┬────────────────────┘
         │  soft_mask = sigmoid(mask_logit)   ← applied in forward, no branch
         │
    ┌────▼────────────────────┐
    │  Soft ROI Extractor     │  spectrogram × soft_mask → resize 224×224
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │  EfficientNet-B0        │  stem → 7 MBConv stages → head → logits
    └────┬────────────────────┘
         │
    (B, num_classes)  raw logits  [NO Softmax]

Key design decisions for NPU quantizability
--------------------------------------------
1.  ConvTranspose2d replaces nn.Upsample(bilinear)
        Bilinear upsample falls back to CPU on Qualcomm NPU.
        ConvTranspose2d (stride-2 deconv) is a native NPU op.

2.  No Sigmoid in the U-Net graph
        Sigmoid output wastes INT8 range (only [0,1] used out of [-128,127]).
        We apply sigmoid in forward() as a floating-point soft mask — the
        Sigmoid op itself stays in the graph but after the mask head Conv,
        before the ROI multiply. SNPE handles Sigmoid fine; the issue was
        the hard-threshold branch (mask >= 0.5) which created a data-
        dependent graph break. Soft mask eliminates that break.

3.  Soft mask — no binary threshold branch
        Old: binary = (mask >= 0.5).float()   ← data-dependent branch
        New: soft   = torch.sigmoid(mask_logit)
        The multiply spectrogram × soft_mask is fully continuous and
        differentiable — one static graph from input to output.

4.  F.interpolate with static size tuple
        (224, 224) passed as a literal tuple so SNPE traces it as a
        static Resize op, not a dynamic shape input.

5.  ReduceMean via dim=(2,3) tuple not list
        TorchScript folds tuple integer constants as ONNX Slice/Reduce
        attributes. List args become dynamic tensor inputs (SNPE rejects).

6.  No AdaptiveAvgPool2d anywhere
        All global pooling done as x.mean(dim=(2,3)).

7.  BCEWithLogitsLoss instead of BCELoss
        Numerically stable. Sigmoid is applied inside the loss during
        training, not in the model graph.

8.  CrossEntropyLoss on raw logits (no Softmax in model)
        Standard PyTorch pattern — avoids double-softmax gradient bug.

Single checkpoint
-----------------
Training produces one file: checkpoints/fused_best.pt
It contains both U-Net and classifier weights.
Export produces one ONNX: exports/drone_fused.onnx
Quantize that one ONNX on Qualcomm AI Hub → one INT8 TFLite/DLC.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
#  Shared building blocks
# ═════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """Conv3×3 → BN → ReLU → [Dropout2d] → Conv3×3 → BN → ReLU."""

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


# ═════════════════════════════════════════════════════════════════════════════
#  U-Net encoder
# ═════════════════════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    """DoubleConv → (skip, MaxPool↓2)."""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return skip, self.pool(skip)


# ═════════════════════════════════════════════════════════════════════════════
#  U-Net decoder  — ConvTranspose2d replaces nn.Upsample(bilinear)
# ═════════════════════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    """
    ConvTranspose2d ×2 → cat(skip) → DoubleConv.

    Why ConvTranspose2d instead of nn.Upsample(bilinear)
    -----------------------------------------------------
    Bilinear upsample is not supported on Qualcomm Hexagon NPU — the SNPE
    compiler falls back to CPU for that op, breaking the single-graph NPU
    execution.  ConvTranspose2d (learnable deconvolution) is a native NPU
    op on Snapdragon, runs in INT8, and is slightly better at reconstruction
    than fixed bilinear interpolation.

    The ConvTranspose2d here is depthwise (groups=in_ch) — cheap, same
    parameter count as a pointwise conv, and SNPE supports depthwise deconv.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        # Depthwise transposed conv for upsampling — cheap and NPU-native
        self.upsample = nn.ConvTranspose2d(
            in_ch, in_ch,
            kernel_size=2, stride=2,
            groups=in_ch, bias=False,
        )
        self.conv = DoubleConv(in_ch + out_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)                        # (B, in_ch, H*2, W*2)
        # For fixed input sizes (256×512) all dims are even — upsample
        # always matches skip exactly. If you use odd sizes, add crop here.
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ═════════════════════════════════════════════════════════════════════════════
#  U-Net  (mask head outputs raw logit — NO Sigmoid in the model)
# ═════════════════════════════════════════════════════════════════════════════

class UNet(nn.Module):
    """
    U-Net segmentation backbone.

    Output: raw mask logit (B, 1, H, W) — Sigmoid applied externally.
    Training loss: BCEWithLogitsLoss (Sigmoid fused into loss, numerically stable).
    Inference / ROI: sigmoid(output) used as soft attention mask.
    """

    def __init__(
        self,
        in_channels : int = 3,
        base_filters: int = 32,
    ):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc0 = EncoderBlock(in_channels, f,     dropout_p=0.0)
        self.enc1 = EncoderBlock(f,           f * 2, dropout_p=0.0)
        self.enc2 = EncoderBlock(f * 2,       f * 4, dropout_p=0.0)
        self.enc3 = EncoderBlock(f * 4,       f * 8, dropout_p=0.0)

        # Bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16, dropout_p=0.3)

        # Decoder — in_ch must match bottleneck/decoder output channels
        self.dec3 = DecoderBlock(f * 16, f * 8,  dropout_p=0.1)
        self.dec2 = DecoderBlock(f * 8,  f * 4,  dropout_p=0.1)
        self.dec1 = DecoderBlock(f * 4,  f * 2,  dropout_p=0.1)
        self.dec0 = DecoderBlock(f * 2,  f,       dropout_p=0.1)

        # Raw mask logit — NO Sigmoid here
        self.mask_head = nn.Conv2d(f, 1, kernel_size=1)

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
        return self.mask_head(x)          # (B, 1, H, W)  raw logit


# ═════════════════════════════════════════════════════════════════════════════
#  EfficientNet-B0 building blocks
# ═════════════════════════════════════════════════════════════════════════════

def _make_divisible(v: float, d: int = 8) -> int:
    new_v = max(d, int(v + d / 2) // d * d)
    if new_v < 0.9 * v:
        new_v += d
    return new_v

def _round_filters(f: int, w: float) -> int:
    return _make_divisible(int(f * w))

def _round_repeats(n: int, d: float) -> int:
    return int(math.ceil(n * d))


class SqueezeExcitation(nn.Module):
    """
    SE recalibration.
    dim=(2,3) tuple → TorchScript constant → static ReduceMean ONNX attr.
    No AdaptiveAvgPool2d (unsupported by SNPE).
    """

    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        sq = max(1, int(in_ch * se_ratio))
        self.fc1  = nn.Conv2d(in_ch, sq, 1, bias=True)
        self.act  = nn.SiLU(inplace=True)
        self.fc2  = nn.Conv2d(sq, in_ch, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)   # tuple → static attr
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
            SqueezeExcitation(mid, se_ratio=se_ratio),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
        ]
        self.block = nn.Sequential(*layers)

    def _drop_connect(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep  = 1.0 - self.drop_connect_rate
        noise = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        return x / keep * torch.floor(noise + keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = self._drop_connect(out) + x
        return out


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 classifier.
    Input : (B, 3, 224, 224)
    Output: (B, num_classes) raw logits — NO Softmax.
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
        num_classes      : int   = 8,
        in_channels      : int   = 3,
        dropout_rate     : float = 0.3,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()
        stem_f = _round_filters(32, 1.0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_f, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_f, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )
        total = sum(n for _, _, n, _, _ in self._BLOCK_ARGS)
        idx, cur = 0, stem_f
        stages = []
        for expand, out_ch, num_l, stride, kernel in self._BLOCK_ARGS:
            out_f = _round_filters(out_ch, 1.0)
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
                cur = out_f; idx += 1
            stages.append(nn.Sequential(*stage))
        (self.b1, self.b2, self.b3, self.b4,
         self.b5, self.b6, self.b7) = stages

        head_f = _round_filters(1280, 1.0)
        self.head_conv = nn.Sequential(
            nn.Conv2d(cur, head_f, 1, bias=False),
            nn.BatchNorm2d(head_f, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_f, num_classes),
            # NO Softmax — CrossEntropyLoss applies log_softmax internally
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.b4(x); x = self.b5(x); x = self.b6(x); x = self.b7(x)
        x = self.head_conv(x)
        x = x.mean(dim=(2, 3))        # global avg pool — tuple → static attr
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════════════════
#  Fused pipeline  — single graph, one ONNX, one NPU call
# ═════════════════════════════════════════════════════════════════════════════

class FusedDronePipeline(nn.Module):
    """
    End-to-end fused pipeline: spectrogram → class logits.

    One forward pass, one graph, one SNPE/NPU call at inference.

    forward() outputs
    -----------------
    logits    : (B, num_classes)  always returned
    mask_logit: (B, 1, H, W)     returned when return_mask=True (training only)

    Loss during training
    --------------------
    Use FusedPipelineLoss which combines:
        BCEWithLogitsLoss(mask_logit, proxy_mask)   ← seg supervision
        CrossEntropyLoss(logits, labels)             ← cls supervision
    Both losses flow gradients back through the entire fused graph — the
    classifier loss directly trains the U-Net mask quality.
    """

    def __init__(
        self,
        num_classes      : int   = 8,
        in_channels      : int   = 3,
        base_filters     : int   = 32,
        cls_dropout      : float = 0.3,
        cls_drop_connect : float = 0.2,
        roi_size         : int   = 224,   # output size for classifier input
    ):
        super().__init__()
        self.roi_size = roi_size

        self.unet       = UNet(in_channels, base_filters)
        self.classifier = EfficientNetB0(
            num_classes       = num_classes,
            in_channels       = in_channels,
            dropout_rate      = cls_dropout,
            drop_connect_rate = cls_drop_connect,
        )

    def forward(
        self,
        x          : torch.Tensor,
        return_mask: bool = False,
    ):
        # ── Stage 1: U-Net → soft mask ────────────────────────────────────────
        mask_logit  = self.unet(x)                          # (B,1,H,W) raw logit
        soft_mask   = torch.sigmoid(mask_logit)             # (B,1,H,W) in [0,1]

        # ── ROI extraction: soft attention, no branch ─────────────────────────
        # Expand mask to match spectrogram channels, zero background softly
        roi = x * soft_mask.expand_as(x)                   # (B,C,H,W)

        # Resize to classifier input — static tuple size, no dynamic shapes
        roi = F.interpolate(
            roi,
            size=(self.roi_size, self.roi_size),
            mode="bilinear",
            align_corners=False,
        )                                                   # (B,C,224,224)

        # ── Stage 2: EfficientNet-B0 classifier ──────────────────────────────
        logits = self.classifier(roi)                       # (B,num_classes)

        if return_mask:
            return logits, mask_logit
        return logits

    # ── Freeze helpers ────────────────────────────────────────────────────────

    def freeze_unet(self):
        for p in self.unet.parameters():
            p.requires_grad = False

    def unfreeze_unet(self):
        for p in self.unet.parameters():
            p.requires_grad = True

    def freeze_classifier(self):
        for p in self.classifier.parameters():
            p.requires_grad = False

    def freeze_cls_backbone(self):
        """Freeze EfficientNet feature extractor, keep linear head trainable."""
        for m in [self.classifier.stem,
                  self.classifier.b1, self.classifier.b2, self.classifier.b3,
                  self.classifier.b4, self.classifier.b5, self.classifier.b6,
                  self.classifier.b7, self.classifier.head_conv]:
            for p in m.parameters():
                p.requires_grad = False
        for p in self.classifier.classifier.parameters():
            p.requires_grad = True

    def unfreeze_cls_from_block(self, block: int = 5):
        """Gradually unfreeze EfficientNet from block N onward (1-indexed)."""
        all_blocks = [self.classifier.b1, self.classifier.b2,
                      self.classifier.b3, self.classifier.b4,
                      self.classifier.b5, self.classifier.b6,
                      self.classifier.b7]
        for i, blk in enumerate(all_blocks, 1):
            for p in blk.parameters():
                p.requires_grad = (i >= block)
        for p in self.classifier.head_conv.parameters():
            p.requires_grad = True
        for p in self.classifier.classifier.parameters():
            p.requires_grad = True

    def param_groups(self, unet_lr: float, cls_lr: float) -> list:
        """
        Return optimizer param groups with per-component learning rates.
        Call this to build the Phase-2 optimizer.
        """
        return [
            {"params": self.unet.parameters(),
             "lr": unet_lr,       "name": "unet"},
            {"params": self.classifier.stem.parameters(),
             "lr": cls_lr * 0.1,  "name": "cls_stem"},
            {"params": list(self.classifier.b1.parameters()) +
                       list(self.classifier.b2.parameters()) +
                       list(self.classifier.b3.parameters()) +
                       list(self.classifier.b4.parameters()),
             "lr": cls_lr * 0.1,  "name": "cls_frozen_blocks"},
            {"params": list(self.classifier.b5.parameters()) +
                       list(self.classifier.b6.parameters()) +
                       list(self.classifier.b7.parameters()) +
                       list(self.classifier.head_conv.parameters()),
             "lr": cls_lr * 0.5,  "name": "cls_upper_blocks"},
            {"params": self.classifier.classifier.parameters(),
             "lr": cls_lr,        "name": "cls_head"},
        ]


# ═════════════════════════════════════════════════════════════════════════════
#  Loss
# ═════════════════════════════════════════════════════════════════════════════

class FusedPipelineLoss(nn.Module):
    """
    Combined loss for the fused pipeline.

        total = λ_seg × BCEWithLogitsLoss(mask_logit, proxy_mask)
              + λ_cls × CrossEntropyLoss(logits, labels)

    BCEWithLogitsLoss vs BCELoss
    ----------------------------
    BCEWithLogitsLoss applies Sigmoid internally in a numerically stable way
    (log-sum-exp trick). The model outputs a raw logit so no Sigmoid is needed
    in the forward pass — this is the correct PyTorch pattern.

    CrossEntropyLoss applies log_softmax internally — no Softmax in the model.
    """

    def __init__(
        self,
        seg_weight    : float = 0.3,
        cls_weight    : float = 1.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss   = nn.BCEWithLogitsLoss()
        self.cls_loss   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        mask_logit : torch.Tensor,   # (B, 1, H, W)  raw U-Net output
        proxy_mask : torch.Tensor,   # (B, 1, H, W)  energy-threshold target
        logits     : torch.Tensor,   # (B, num_classes)
        labels     : torch.Tensor,   # (B,)
    ) -> dict:
        seg   = self.seg_loss(mask_logit, proxy_mask)
        cls   = self.cls_loss(logits, labels)
        total = self.seg_weight * seg + self.cls_weight * cls
        return {"total": total, "seg": seg, "cls": cls}


# ═════════════════════════════════════════════════════════════════════════════
#  Proxy mask  (unchanged from old pipeline — no GT masks needed)
# ═════════════════════════════════════════════════════════════════════════════

def build_proxy_mask(
    images       : torch.Tensor,
    labels       : torch.Tensor,
    no_drone_idx : int,
    threshold    : float = 0.5,
) -> torch.Tensor:
    """
    Energy-threshold proxy mask for U-Net supervision.
    NO_DRONE samples → all-zero mask (nothing to segment).
    Drone samples    → high-energy bins → 1, noise floor → 0.
    """
    energy = images.mean(dim=1, keepdim=True)
    e_flat = energy.flatten(1)
    e_min  = e_flat.min(1)[0].view(-1, 1, 1, 1)
    e_max  = e_flat.max(1)[0].view(-1, 1, 1, 1)
    normed = (energy - e_min) / (e_max - e_min + 1e-8)
    proxy  = (normed > threshold).float()
    no_drone = (labels == no_drone_idx)
    if no_drone.any():
        proxy[no_drone] = 0.0
    return proxy


# ═════════════════════════════════════════════════════════════════════════════
#  Checkpoint helpers
# ═════════════════════════════════════════════════════════════════════════════

def save_checkpoint(state: dict, path: str):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_pipeline(ckpt_path: str, device=None) -> FusedDronePipeline:
    """Load a FusedDronePipeline from a training checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]
    model = FusedDronePipeline(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[Pipeline] Loaded '{ckpt_path}'  "
          f"epoch={ckpt['epoch']}  val_acc={ckpt.get('val_acc', 0):.2f}%")
    return model