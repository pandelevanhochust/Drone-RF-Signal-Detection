"""
stage2_classifier.py
====================
Stage 2 — EfficientNet-B0 Fine-tuning for Drone Classification

What this stage does
--------------------
Receives the 224×224 ROI patch produced on-the-fly by Stage 1 (U-Net mask
× raw spectrogram → resize) and classifies it into one of 8 drone classes.

The input to this stage is NEVER a saved image file — it is always a live
tensor produced by ROIExtractor inside the same training batch.

Separation from Stage 1
------------------------
Stage 2 imports DroneROIUNet, ROIExtractor, and build_proxy_mask from
stage1_unet.py. It never calls the U-Net training loop itself. The two
stages connect only through ROIExtractor.forward().

Training modes
--------------
Mode A — Frozen U-Net (default first phase)
    U-Net weights are loaded from a Stage-1 checkpoint and frozen.
    Only the EfficientNet classifier is trained. Fast and safe — the
    pretrained mask quality is preserved while the head adapts.

Mode B — Full end-to-end fine-tune
    U-Net unfrozen at a very low LR (10× smaller than classifier LR).
    The segmentation and classification objectives improve jointly.
    Use after Mode A has converged.

Loss
----
Classification only: CrossEntropyLoss on raw logits (no Softmax).
The combined seg+cls loss from the original PipelineLoss is also available
via DronePipelineLoss for full joint training.
"""

import os
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from RoiExtractor import (
    DroneROIUNet,
    ROIExtractor,
    build_proxy_mask,
    load_unet,
)


# ─────────────────────────────────────────────────────────────────────────────
#  EfficientNet-B0 building blocks  (from-scratch, SNPE-compatible)
# ─────────────────────────────────────────────────────────────────────────────

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
    Channel-wise SE recalibration.

    SNPE compatibility: nn.AdaptiveAvgPool2d(1) is rejected by SNPE/QNN.
    Replaced with x.mean(dim=[2,3], keepdim=True) which exports as a
    ReduceMean ONNX op — fully supported in opset 13.
    """

    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        sq = max(1, int(in_ch * se_ratio))
        self.fc1  = nn.Conv2d(in_ch, sq, 1, bias=True)
        self.act  = nn.SiLU(inplace=True)
        self.fc2  = nn.Conv2d(sq, in_ch, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=[2, 3], keepdim=True)   # ReduceMean — SNPE-safe
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


class MBConvBlock(nn.Module):
    """Mobile inverted bottleneck conv with optional drop-connect."""

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


# ─────────────────────────────────────────────────────────────────────────────
#  DroneCLSNet — EfficientNet-B0 classifier
# ─────────────────────────────────────────────────────────────────────────────

class DroneCLSNet(nn.Module):
    """
    EfficientNet-B0 backbone classifier for drone type recognition.

    Input  : (B, C, 224, 224) ROI patch from ROIExtractor
    Output : (B, num_classes) raw logits  — NO Softmax applied

    Design decisions
    ----------------
    * No Softmax in forward() — CrossEntropyLoss applies log_softmax
      internally. Adding Softmax beforehand silently corrupts gradients
      (double-softmax bug).
    * AdaptiveAvgPool2d replaced by .mean(dim=[2,3]) in forward() so
      the model exports as ReduceMean ONNX op (SNPE opset-13 compatible).
    * EfficientNet-B0 configuration (_BLOCK_ARGS) matches the Qualcomm
      AI Hub checkpoint exactly so pretrained weights can be loaded.
    """

    _BLOCK_ARGS = [
        # (expand_ratio, out_ch, num_layers, stride, kernel_size)
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
        width_coeff      : float = 1.0,
        depth_coeff      : float = 1.0,
        dropout_rate     : float = 0.2,
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

        # No Softmax — raw logits for CrossEntropyLoss
        # No AdaptiveAvgPool2d — replaced by .mean() in forward()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_f, num_classes),
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
        Args:  x      : (B, C, 224, 224) ROI patch
        Returns: logits: (B, num_classes) raw scores, NO softmax
        """
        x = self.stem(x)
        x = self.block1(x);  x = self.block2(x);  x = self.block3(x)
        x = self.block4(x);  x = self.block5(x);  x = self.block6(x)
        x = self.block7(x)
        x = self.head_conv(x)
        x = x.mean(dim=[2, 3])   # (B, 1280) — ReduceMean, SNPE-safe
        return self.classifier(x)

    # ── Granular freeze helpers ───────────────────────────────────────────────

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze_backbone(self):
        """Freeze stem + all MBConv stages, leave classifier head trainable."""
        for m in [self.stem, self.block1, self.block2, self.block3,
                  self.block4, self.block5, self.block6, self.block7,
                  self.head_conv]:
            for p in m.parameters():
                p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_from_block(self, block_idx: int = 5):
        """
        Unfreeze MBConv blocks ≥ block_idx (1-indexed) for gradual unfreezing.
        Blocks 1-4 learn low-level spectrogram features that transfer well;
        blocks 5-7 learn task-specific patterns and benefit from fine-tuning.
        """
        all_blocks = [self.block1, self.block2, self.block3, self.block4,
                      self.block5, self.block6, self.block7]
        for i, blk in enumerate(all_blocks, start=1):
            requires = (i >= block_idx)
            for p in blk.parameters():
                p.requires_grad = requires
        for p in self.head_conv.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
#  Full pipeline loss  (seg + cls jointly)
# ─────────────────────────────────────────────────────────────────────────────

class DronePipelineLoss(nn.Module):
    """
    Combined loss for joint Stage-1 + Stage-2 end-to-end training.

        total = λ_seg × BCE(pred_mask, proxy_mask)
              + λ_cls × CrossEntropy(logits, labels)

    CrossEntropyLoss receives RAW LOGITS — no softmax applied beforehand.
    """

    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss   = nn.BCELoss()
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(
        self,
        pred_mask  : torch.Tensor,   # (B, 1, H, W)      sigmoid output
        gt_mask    : torch.Tensor,   # (B, 1, H, W)      proxy target
        pred_logits: torch.Tensor,   # (B, num_classes)  raw logits
        gt_labels  : torch.Tensor,   # (B,)              class indices
    ) -> dict[str, torch.Tensor]:
        seg   = self.seg_loss(pred_mask, gt_mask)
        cls   = self.cls_loss(pred_logits, gt_labels)
        total = self.seg_weight * seg + self.cls_weight * cls
        return {"total": total, "seg": seg, "cls": cls}


# ─────────────────────────────────────────────────────────────────────────────
#  Stage-2 training
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(
    train_loader,
    val_loader,
    no_drone_idx    : int,
    class_names     : list,
    unet_ckpt       : str   = None,   # path to Stage-1 checkpoint; None = random init
    ckpt_dir        : str   = "checkpoints",
    num_classes     : int   = 8,
    in_channels     : int   = 3,
    unet_base_filters: int  = 32,
    img_h           : int   = 256,
    img_w           : int   = 512,
    roi_strategy    : str   = "multiply",
    mask_threshold  : float = 0.5,
    cls_dropout     : float = 0.2,
    cls_drop_connect: float = 0.2,
    seg_weight      : float = 1.0,
    cls_weight      : float = 1.0,
    # Phase 1 — frozen U-Net, head only
    phase1_epochs   : int   = 15,
    phase1_lr       : float = 1e-3,
    # Phase 2 — full end-to-end
    phase2_epochs   : int   = 40,
    phase2_lr       : float = 1e-4,
    unfreeze_block  : int   = 5,    # unfreeze EfficientNet from this MBConv block
    unet_lr_scale   : float = 0.1,  # U-Net LR = phase2_lr × this factor
    weight_decay    : float = 1e-4,
    proxy_threshold : float = 0.5,
    device_str      : str   = "auto",
    log_interval    : int   = 50,
):
    """
    Two-phase fine-tuning of DroneCLSNet on live ROI patches.

    Phase 1 — U-Net frozen
    ----------------------
    Loads Stage-1 U-Net weights (or uses random init if unet_ckpt is None).
    Freezes all U-Net parameters. Trains only the EfficientNet classifier
    head using classification CrossEntropyLoss only.

    The ROI patches are still produced on-the-fly by ROIExtractor every
    batch — the U-Net runs in eval() mode with no_grad, so its weights
    are not updated but the mask quality is fully utilised.

    Phase 2 — End-to-end
    ---------------------
    Unfreezes EfficientNet from `unfreeze_block` onward and the U-Net at
    a very low LR (unet_lr_scale × phase2_lr). Uses DronePipelineLoss
    (seg + cls) so both components improve jointly.

    The U-Net LR being 10× smaller than the classifier LR prevents
    catastrophic forgetting of the segmentation ability while allowing
    the mask quality to adapt to classification feedback.
    """
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto" else torch.device(device_str)
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Build models ─────────────────────────────────────────────────────────
    if unet_ckpt and os.path.exists(unet_ckpt):
        unet = load_unet(unet_ckpt, device=device)
    else:
        print("[Stage 2] No U-Net checkpoint — using random init")
        unet = DroneROIUNet(in_channels, unet_base_filters, img_h, img_w).to(device)

    extractor = ROIExtractor(
        output_size=(224, 224),
        threshold=mask_threshold,
        strategy=roi_strategy,
    )

    cls_net = DroneCLSNet(
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_rate=cls_dropout,
        drop_connect_rate=cls_drop_connect,
    ).to(device)

    pipeline_loss = DronePipelineLoss(seg_weight, cls_weight)
    cls_only_loss = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_ckpt    = os.path.join(ckpt_dir, "classifier_best.pt")

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)

    def _forward_batch(images, labels, unet_grad: bool):
        """Shared forward for train and val."""
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Proxy mask target (always no_grad — it's a heuristic, not a model)
        with torch.no_grad():
            gt_mask = build_proxy_mask(images, labels, no_drone_idx,
                                       proxy_threshold)

        if unet_grad:
            pred_mask = unet(images)            # U-Net with grad
        else:
            with torch.no_grad():
                pred_mask = unet(images)        # U-Net frozen

        roi    = extractor(images, pred_mask)   # (B, C, 224, 224)
        logits = cls_net(roi)                   # (B, num_classes)
        return pred_mask, gt_mask, logits, labels

    def _run_val():
        unet.eval(); cls_net.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                pred_mask, gt_mask, logits, labels = _forward_batch(
                    images, labels, unet_grad=False
                )
                losses = pipeline_loss(pred_mask, gt_mask, logits, labels)
                total_loss += losses["total"].item() * images.size(0)
                correct    += (logits.argmax(1) == labels).sum().item()
                total      += images.size(0)
        return total_loss / total, 100.0 * correct / total

    def _save_best(epoch, val_acc, phase):
        nonlocal best_val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch"        : epoch,
                "phase"        : phase,
                "unet_state"   : unet.state_dict(),
                "cls_state"    : cls_net.state_dict(),
                "val_acc"      : val_acc,
                "class_names"  : class_names,
                "num_classes"  : num_classes,
                "unet_config"  : dict(
                    in_channels=in_channels,
                    base_filters=unet_base_filters,
                    img_h=img_h, img_w=img_w,
                ),
            }, best_ckpt)
            print(f"  ✓ Saved best checkpoint  (val_acc={val_acc:.2f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 1 — U-Net frozen, classifier head only
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Stage 2 / Phase 1] U-Net FROZEN — training classifier head only")
    print(f"  Epochs: {phase1_epochs}   LR: {phase1_lr}\n")

    for p in unet.parameters():
        p.requires_grad = False
    cls_net.freeze_backbone()     # only linear head is active

    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, cls_net.parameters()),
        lr=phase1_lr, weight_decay=weight_decay,
    )
    sch1 = CosineAnnealingLR(opt1, T_max=phase1_epochs, eta_min=phase1_lr * 0.01)

    for epoch in range(1, phase1_epochs + 1):
        unet.eval(); cls_net.train()
        train_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            pred_mask, gt_mask, logits, labels = _forward_batch(
                images, labels, unet_grad=False
            )
            loss = cls_only_loss(logits, labels)   # cls loss only in Phase 1
            opt1.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(cls_net.parameters(), max_norm=1.0)
            opt1.step()

            B = images.size(0)
            train_loss += loss.item() * B
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += B

            if (batch_idx + 1) % log_interval == 0:
                print(f"  P1 Ep {epoch:03d} | batch {batch_idx+1:4d} | "
                      f"loss={train_loss/total:.4f}  "
                      f"acc={100.*correct/total:.1f}%")

        sch1.step()
        val_loss, val_acc = _run_val()
        elapsed = time.time() - t0
        print(f"P1 Epoch {epoch:03d}/{phase1_epochs} | "
              f"train_loss={train_loss/n_train:.4f}  train_acc={100.*correct/n_train:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
              f"time={elapsed:.0f}s")
        _save_best(epoch, val_acc, phase=1)

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 2 — End-to-end fine-tune  (seg + cls jointly)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Stage 2 / Phase 2] End-to-end fine-tune  "
          f"(EfficientNet from block {unfreeze_block}, U-Net at {unet_lr_scale}× LR)")
    print(f"  Epochs: {phase2_epochs}   Classifier LR: {phase2_lr}   "
          f"U-Net LR: {phase2_lr * unet_lr_scale}\n")

    # Unfreeze EfficientNet from unfreeze_block onward
    cls_net.unfreeze_from_block(unfreeze_block)

    # Unfreeze U-Net at a lower LR
    for p in unet.parameters():
        p.requires_grad = True

    opt2 = torch.optim.Adam([
        {"params": unet.parameters(),          "lr": phase2_lr * unet_lr_scale},
        {"params": cls_net.stem.parameters(),  "lr": phase2_lr * 0.1},
        # Frozen blocks have requires_grad=False so their params are excluded
        {"params": filter(lambda p: p.requires_grad,
                          list(cls_net.block5.parameters()) +
                          list(cls_net.block6.parameters()) +
                          list(cls_net.block7.parameters()) +
                          list(cls_net.head_conv.parameters())),
                                               "lr": phase2_lr * 0.5},
        {"params": cls_net.classifier.parameters(), "lr": phase2_lr},
    ], weight_decay=weight_decay)
    sch2 = CosineAnnealingLR(opt2, T_max=phase2_epochs, eta_min=phase2_lr * 0.001)

    for epoch in range(1, phase2_epochs + 1):
        unet.train(); cls_net.train()
        train_loss, train_seg, train_cls = 0.0, 0.0, 0.0
        correct, total = 0, 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            pred_mask, gt_mask, logits, labels = _forward_batch(
                images, labels, unet_grad=True   # U-Net gets gradients now
            )
            losses = pipeline_loss(pred_mask, gt_mask, logits, labels)
            opt2.zero_grad(set_to_none=True)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(
                list(unet.parameters()) + list(cls_net.parameters()),
                max_norm=1.0,
            )
            opt2.step()

            B = images.size(0)
            train_loss += losses["total"].item() * B
            train_seg  += losses["seg"].item()   * B
            train_cls  += losses["cls"].item()   * B
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += B

            if (batch_idx + 1) % log_interval == 0:
                print(f"  P2 Ep {epoch:03d} | batch {batch_idx+1:4d} | "
                      f"total={train_loss/total:.4f}  "
                      f"seg={train_seg/total:.4f}  "
                      f"cls={train_cls/total:.4f}  "
                      f"acc={100.*correct/total:.1f}%")

        sch2.step()
        val_loss, val_acc = _run_val()
        elapsed = time.time() - t0
        print(f"P2 Epoch {epoch:03d}/{phase2_epochs} | "
              f"train_total={train_loss/n_train:.4f}  "
              f"seg={train_seg/n_train:.4f}  cls={train_cls/n_train:.4f}  "
              f"train_acc={100.*correct/n_train:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
              f"time={elapsed:.0f}s")
        _save_best(epoch, val_acc, phase=2)

    print(f"\n[Stage 2] Done. Best val_acc={best_val_acc:.2f}%")
    print(f"          Best checkpoint: {best_ckpt}")
    return best_ckpt


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint loader
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(ckpt_path: str, device=None):
    """
    Reconstruct both U-Net and DroneCLSNet from a Stage-2 checkpoint.
    Returns (unet, extractor, cls_net, class_names).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    ucfg    = ckpt["unet_config"]
    unet    = DroneROIUNet(**ucfg).to(device)
    unet.load_state_dict(ckpt["unet_state"])

    cls_net = DroneCLSNet(num_classes=ckpt["num_classes"]).to(device)
    cls_net.load_state_dict(ckpt["cls_state"])

    extractor = ROIExtractor(output_size=(224, 224), strategy="multiply")

    print(f"[Stage 2] Loaded checkpoint from {ckpt_path}  "
          f"(epoch {ckpt['epoch']}, phase {ckpt['phase']}, "
          f"val_acc={ckpt['val_acc']:.2f}%)")
    return unet, extractor, cls_net, ckpt["class_names"]


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: EfficientNet-B0 fine-tuning on live ROI patches"
    )
    parser.add_argument("--root",             default="output_spectrograms/")
    parser.add_argument("--subsets",          nargs="+", default=["BLUE", "BOTH", "CLEAN", "WIFI"])
    parser.add_argument("--img_size",         nargs=2, type=int, default=[256, 512])
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--workers",          type=int,   default=4)
    parser.add_argument("--unet_ckpt",        default="checkpoints/unet_best.pt",
                        help="Stage-1 checkpoint. Omit to train from scratch.")
    parser.add_argument("--ckpt_dir",         default="checkpoints/")
    parser.add_argument("--num_classes",      type=int,   default=8)
    parser.add_argument("--unet_base_filters",type=int,   default=32)
    parser.add_argument("--roi_strategy",     default="multiply",
                        choices=["multiply", "bbox"])
    parser.add_argument("--seg_weight",       type=float, default=1.0)
    parser.add_argument("--cls_weight",       type=float, default=1.0)
    parser.add_argument("--phase1_epochs",    type=int,   default=15)
    parser.add_argument("--phase1_lr",        type=float, default=1e-3)
    parser.add_argument("--phase2_epochs",    type=int,   default=40)
    parser.add_argument("--phase2_lr",        type=float, default=1e-4)
    parser.add_argument("--unfreeze_block",   type=int,   default=5)
    parser.add_argument("--unet_lr_scale",    type=float, default=0.1)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument("--log_interval",     type=int,   default=50)
    parser.add_argument("--seed",             type=int,   default=42)
    args = parser.parse_args()

    from drone_dataloader import build_dataloaders
    train_loader, val_loader, _, meta = build_dataloaders(
        root        = args.root,
        subsets     = args.subsets,
        img_size    = tuple(args.img_size),
        batch_size  = args.batch_size,
        num_workers = args.workers,
        seed        = args.seed,
    )

    train_classifier(
        train_loader     = train_loader,
        val_loader       = val_loader,
        no_drone_idx     = meta["class_to_idx"].get("NO_DRONE", -1),
        class_names      = meta["class_names"],
        unet_ckpt        = args.unet_ckpt,
        ckpt_dir         = args.ckpt_dir,
        num_classes      = args.num_classes,
        in_channels      = 3,
        unet_base_filters= args.unet_base_filters,
        img_h            = args.img_size[0],
        img_w            = args.img_size[1],
        roi_strategy     = args.roi_strategy,
        seg_weight       = args.seg_weight,
        cls_weight       = args.cls_weight,
        phase1_epochs    = args.phase1_epochs,
        phase1_lr        = args.phase1_lr,
        phase2_epochs    = args.phase2_epochs,
        phase2_lr        = args.phase2_lr,
        unfreeze_block   = args.unfreeze_block,
        unet_lr_scale    = args.unet_lr_scale,
        weight_decay     = args.weight_decay,
        log_interval     = args.log_interval,
    )