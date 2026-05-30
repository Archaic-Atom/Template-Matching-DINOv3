# -*- coding: utf-8 -*-
"""StandardMatchingHead — the composite matching head used by TM_Model_A / B / D.

Wraps the stride-4 + stride-8 coarse heads, a small stride-1 fine head on
raw images, and the coarse argmax + crop logic in a single nn.Module. Its
attribute layout (`proj_sar_4`, `head_4`, `fine_sar`, …) matches TM_Model_A
so that state_dict keys under `trainable_tail.matching.*` are byte-identical.

forward signature: standardized for ComposedTM.
    forward(sar_img, rgb_img, sar_feat, rgb_feat, fusion_logits) → dict
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register
from .coarse_matching import ProjMatchSAR, ProjMatchRGB, ProjMatchJBU, MatchingHead


def _gn(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    g = num_groups if num_channels >= num_groups else 1
    return nn.GroupNorm(g, num_channels)


class FineFeatureHead(nn.Module):
    """Small stride-1 CNN mapping raw image (3ch) to pixel-level features."""

    def __init__(self, out_dim: int = 32, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
            _gn(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            _gn(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_dim, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


def crop_search_at(rgb: torch.Tensor,
                   center_xy_pix: torch.Tensor,
                   sar_size: int,
                   margin: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop per-sample search region from RGB at given pixel anchors.

    Returns (rgb_crop, anchor_xy) where crop_size = sar_size + 2*margin and
    each sample's anchor is clipped to keep the crop inside the RGB image.
    """
    if center_xy_pix.dim() != 2 or center_xy_pix.shape[1] != 2:
        raise ValueError(f"center_xy_pix must be (B, 2); got {tuple(center_xy_pix.shape)}")

    B, _, H, W = rgb.shape
    crop = sar_size + 2 * margin
    if H < crop or W < crop:
        raise ValueError(f"RGB ({H}x{W}) is smaller than required crop {crop}x{crop}")

    cx = center_xy_pix[:, 0].long()
    cy = center_xy_pix[:, 1].long()
    x0 = (cx - margin).clamp(min=0, max=W - crop)
    y0 = (cy - margin).clamp(min=0, max=H - crop)

    crops = rgb.new_empty((B, rgb.shape[1], crop, crop))
    for b in range(B):
        xs, ys = int(x0[b].item()), int(y0[b].item())
        crops[b] = rgb[b, :, ys:ys + crop, xs:xs + crop]

    anchor = torch.stack([x0, y0], dim=1)  # (B, 2)
    return crops, anchor


@register("StandardMatchingHead")
class StandardMatchingHead(nn.Module):
    """A/B/D's matching head: coarse stride-4 + stride-8 + simple stride-1 fine."""

    def __init__(self,
                 sar_ch_c2: int = 64,
                 sar_ch_c3: int = 128,
                 dino_dim: int = 1024,
                 matching_dim: int = 128,
                 fine_dim: int = 32,
                 fine_margin: int = 4,
                 sar_size: int = 64) -> None:
        super().__init__()
        self.sar_size = int(sar_size)
        self.fine_margin = int(fine_margin)

        # Coarse stride-4
        self.proj_sar_4 = ProjMatchSAR(in_ch=sar_ch_c2, out_dim=matching_dim)
        self.proj_rgb_4 = ProjMatchRGB(dino_dim=dino_dim, out_dim=matching_dim, target_stride=4)
        self.head_4 = MatchingHead(refine=True)

        # Coarse stride-8
        self.proj_sar_8 = ProjMatchSAR(in_ch=sar_ch_c3, out_dim=matching_dim)
        self.proj_rgb_8 = ProjMatchRGB(dino_dim=dino_dim, out_dim=matching_dim, target_stride=8)
        self.head_8 = MatchingHead(refine=True)

        # Fine stride-1
        self.fine_sar = FineFeatureHead(out_dim=fine_dim)
        self.fine_rgb = FineFeatureHead(out_dim=fine_dim)
        self.head_fine = MatchingHead(refine=True)

    @staticmethod
    def _coarse_argmax_pixel(H4: torch.Tensor, H8: torch.Tensor,
                             fusion_logits: torch.Tensor) -> torch.Tensor:
        alphas = F.softmax(fusion_logits, dim=0)
        H8_up = F.interpolate(H8, size=H4.shape[-2:],
                              mode='bilinear', align_corners=False)
        H_fused = alphas[0] * H4 + alphas[1] * H8_up
        B = H_fused.shape[0]
        h, w = H_fused.shape[-2:]
        idx = H_fused.view(B, -1).argmax(dim=1)
        v = idx // w
        u = idx % w
        return torch.stack([u, v], dim=1) * 4  # stride-4 → pixel anchor

    def forward(self,
                sar_img: torch.Tensor,
                rgb_img: torch.Tensor,
                sar_feat: dict,
                rgb_feat: list,
                fusion_logits: torch.Tensor) -> dict:
        sar_c2 = sar_feat['C2']
        sar_c3 = sar_feat['C3']
        rgb_16 = rgb_feat[-1]

        # Coarse
        sar_4 = self.proj_sar_4(sar_c2)
        sar_8 = self.proj_sar_8(sar_c3)
        rgb_4 = self.proj_rgb_4(rgb_16)
        rgb_8 = self.proj_rgb_8(rgb_16)
        H_4 = self.head_4(template=sar_4, search=rgb_4)
        H_8 = self.head_8(template=sar_8, search=rgb_8)
        alphas = F.softmax(fusion_logits, dim=0)

        # Fine
        with torch.no_grad():
            center_xy = self._coarse_argmax_pixel(H_4, H_8, fusion_logits)
        rgb_crop, fine_anchor = crop_search_at(
            rgb_img, center_xy, sar_size=self.sar_size, margin=self.fine_margin)
        sar_f = self.fine_sar(sar_img)
        rgb_f = self.fine_rgb(rgb_crop)
        H_fine = self.head_fine(template=sar_f, search=rgb_f)

        return {
            'heatmap_per_stride': {4: H_4, 8: H_8},
            'heatmap_fine':       H_fine,
            'fine_anchor_pix':    fine_anchor,
            'fusion_weights':     alphas,
        }


@register("StandardMatchingHeadJBU")
class StandardMatchingHeadJBU(nn.Module):
    """A/B/D variants' matching head with FeatUp JBU upsamplers.

    Same overall pipeline as ``StandardMatchingHead`` (stride-4 + stride-8
    coarse fusion + stride-1 fine on raw pixels), but BOTH modalities go
    through learned JBU upsamplers from their stride-16 1024-d features
    plus their raw image as guidance, instead of:
      - RGB: bilinear interpolation + 1×1 conv (old ``ProjMatchRGB``)
      - SAR: 1×1 conv on the encoder's native stride-4/8 features
              (``C2`` / ``C3`` via old ``ProjMatchSAR``)

    与 ``StandardMatchingHead`` 同样的"stride-4 + stride-8 粗匹配融合 +
    stride-1 细匹配"流程，但**双边**都改成：stride-16 (1024-d) + 原图
    → JBU → stride-4 (256-d) 的可学习路径。对称设计，参数不共享——若
    SAR 侧 JBU 不带来 gain，网络可以学到把 SAR JBU 退化为接近恒等。

    State_dict layout differs from ``StandardMatchingHead`` (different
    submodule names + new JBU params); these are NEW weights trained
    from scratch in Stage 2.
    state_dict 不与旧 head 兼容，必须新训。

    Args:
        dino_dim: channel count of stride-16 features (DINOv3 ViT-L/16 →
            1024; SAR encoder's ``sar16`` after ``ProjHead`` → 1024).
            stride-16 特征的通道数。
        matching_dim: shared output channel count of all coarse projection
            heads; the cross-correlation runs in this dim.
            匹配子空间维度（256 默认；比旧 head 的 128 翻倍）。
        fine_dim: stride-1 fine head's output channels.
            stride-1 细匹配头的输出通道数。
        fine_margin: crop margin around the coarse argmax for the fine
            stride-1 search region.
            细匹配的 crop margin。
        sar_size: SAR template pixel size (used by fine crop logic).
            SAR 模板像素尺寸。
    """

    def __init__(self,
                 dino_dim: int = 1024,
                 matching_dim: int = 256,
                 fine_dim: int = 32,
                 fine_margin: int = 4,
                 sar_size: int = 64) -> None:
        super().__init__()
        self.sar_size = int(sar_size)
        self.fine_margin = int(fine_margin)

        # Coarse stride-4: JBU with 2 levels (×4 upsample, stride-16 → 4)
        # Stride-4 粗匹配：2 级 JBU，×4 上采。
        self.proj_rgb_4 = ProjMatchJBU(
            feat_dim=dino_dim, out_dim=matching_dim, levels=2)
        self.proj_sar_4 = ProjMatchJBU(
            feat_dim=dino_dim, out_dim=matching_dim, levels=2)
        self.head_4 = MatchingHead(refine=True)

        # Coarse stride-8: JBU with 1 level (×2 upsample, stride-16 → 8)
        # Stride-8 粗匹配：1 级 JBU，×2 上采。
        self.proj_rgb_8 = ProjMatchJBU(
            feat_dim=dino_dim, out_dim=matching_dim, levels=1)
        self.proj_sar_8 = ProjMatchJBU(
            feat_dim=dino_dim, out_dim=matching_dim, levels=1)
        self.head_8 = MatchingHead(refine=True)

        # Fine stride-1: identical to StandardMatchingHead, operates on raw pixels.
        # stride-1 细匹配：与 StandardMatchingHead 一致，吃原图。
        self.fine_sar = FineFeatureHead(out_dim=fine_dim)
        self.fine_rgb = FineFeatureHead(out_dim=fine_dim)
        self.head_fine = MatchingHead(refine=True)

    @staticmethod
    def _coarse_argmax_pixel(H4: torch.Tensor, H8: torch.Tensor,
                             fusion_logits: torch.Tensor) -> torch.Tensor:
        """Fuse H4/H8 heatmaps + take argmax → pixel anchor (B, 2).

        与 StandardMatchingHead._coarse_argmax_pixel 完全一致，用于 fine
        阶段的 crop 中心定位。
        """
        alphas = F.softmax(fusion_logits, dim=0)
        H8_up = F.interpolate(H8, size=H4.shape[-2:],
                              mode='bilinear', align_corners=False)
        H_fused = alphas[0] * H4 + alphas[1] * H8_up
        B = H_fused.shape[0]
        h, w = H_fused.shape[-2:]
        idx = H_fused.view(B, -1).argmax(dim=1)
        v = idx // w
        u = idx % w
        return torch.stack([u, v], dim=1) * 4  # stride-4 → pixel anchor

    def forward(self,
                sar_img: torch.Tensor,
                rgb_img: torch.Tensor,
                sar_feat: dict,
                rgb_feat: list,
                fusion_logits: torch.Tensor) -> dict:
        # Both modalities consume their stride-16 1024-d features.
        # 双边都吃 stride-16 1024-d 特征。
        rgb_16 = rgb_feat[-1]            # (B, 1024, 8, 8)
        sar_16 = sar_feat['sar16']        # (B, 1024, 4, 4) L2-normed from ProjHead

        # Coarse stride-4 via JBU (image-guided ×4 upsample)
        # stride-4 粗匹配：图像引导的 ×4 上采。
        rgb_4 = self.proj_rgb_4(rgb_16, rgb_img)
        sar_4 = self.proj_sar_4(sar_16, sar_img)
        H_4 = self.head_4(template=sar_4, search=rgb_4)

        # Coarse stride-8 via JBU (×2 upsample)
        # stride-8 粗匹配：×2 上采。
        rgb_8 = self.proj_rgb_8(rgb_16, rgb_img)
        sar_8 = self.proj_sar_8(sar_16, sar_img)
        H_8 = self.head_8(template=sar_8, search=rgb_8)

        alphas = F.softmax(fusion_logits, dim=0)

        # Fine stride-1 — identical to StandardMatchingHead.
        # stride-1 细匹配，逻辑与 StandardMatchingHead 一致。
        with torch.no_grad():
            center_xy = self._coarse_argmax_pixel(H_4, H_8, fusion_logits)
        rgb_crop, fine_anchor = crop_search_at(
            rgb_img, center_xy, sar_size=self.sar_size,
            margin=self.fine_margin)
        sar_f = self.fine_sar(sar_img)
        rgb_f = self.fine_rgb(rgb_crop)
        H_fine = self.head_fine(template=sar_f, search=rgb_f)

        return {
            'heatmap_per_stride': {4: H_4, 8: H_8},
            'heatmap_fine':       H_fine,
            'fine_anchor_pix':    fine_anchor,
            'fusion_weights':     alphas,
        }
