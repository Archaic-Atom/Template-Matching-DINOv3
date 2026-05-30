# -*- coding: utf-8 -*-
"""Coarse matching building blocks (shared by all variants).

Provides the projection / cross-correlation primitives used by every
matching head. Not @register'd individually — they're used as internal
parts of composite heads (StandardMatchingHead, CascadeHead).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    g = num_groups if num_channels >= num_groups else 1
    return nn.GroupNorm(g, num_channels)


class ProjMatchSAR(nn.Module):
    """Project SAR encoder feature to matching dim, L2-normalize per pixel."""

    def __init__(self, in_ch: int, out_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_dim, 1, bias=False),
            _gn(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=1)


class ProjMatchRGB(nn.Module):
    """Upsample DINO stride-16 to target_stride (4 or 8), project + L2-normalize."""

    def __init__(self, dino_dim: int = 1024, out_dim: int = 128,
                 target_stride: int = 4) -> None:
        super().__init__()
        if target_stride <= 0 or 16 % target_stride != 0:
            raise ValueError(f"target_stride must divide 16; got {target_stride}")
        self.scale_factor = 16 // target_stride
        self.target_stride = target_stride
        self.proj = nn.Sequential(
            nn.Conv2d(dino_dim, out_dim, 1, bias=False),
            _gn(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
        )

    def forward(self, x_16: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x_16, scale_factor=self.scale_factor,
                          mode='bilinear', align_corners=False)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)


class ProjMatchJBU(nn.Module):
    """FeatUp-style JBU projection: stride-16 features + raw image guide
    → upsampled features at out_dim, L2-normalized.

    用 FeatUp JBU 把 stride-16 特征上采 + 原图导引，再 1×1 投影 + L2-norm。
    替代 ProjMatchRGB 的 bilinear 上采 + 1×1 投影；同样可用于 SAR 侧（对称
    设计：双边都吃自己的 stride-16 1024-d 特征，用各自的原图 guide）。

    Use cases:
        RGB stride-4: feat_dim=1024 (DINOv3), out_dim=256, levels=2
            dino_16 (B, 1024, 8, 8)  + rgb_img (B, 3, 128, 128)
            → (B, 256, 32, 32)
        SAR stride-4: feat_dim=1024 (sar16), out_dim=256, levels=2
            sar_16  (B, 1024, 4, 4)  + sar_img (B, 3, 64, 64)
            → (B, 256, 16, 16)
        Stride-8 (either modality): levels=1, halves the upsample.
    """

    def __init__(self, feat_dim: int = 1024, out_dim: int = 256,
                 levels: int = 2, guidance_dim: int = 3) -> None:
        super().__init__()
        # Lazy import keeps optional FeatUp dependency contained.
        # 延迟 import 让 FeatUp 这一可选依赖只在用到此类时报错。
        from .jbu_upsampler import JBUStack
        self.jbu = JBUStack(feat_dim=feat_dim, out_dim=out_dim,
                            levels=levels, guidance_dim=guidance_dim)

    def forward(self, source: torch.Tensor,
                guidance: torch.Tensor) -> torch.Tensor:
        """Upsample ``source`` guided by ``guidance``, return L2-normalized.

        Args:
            source: ``(B, feat_dim, h, w)`` low-res feature (stride-16).
            guidance: ``(B, guidance_dim, H, W)`` raw image.
        """
        x = self.jbu(source, guidance)
        return F.normalize(x, p=2, dim=1)


class MatchingHead(nn.Module):
    """Cross-correlation (grouped conv) between L2-normalized template and search,
    optionally followed by a 3x3 conv refinement on the resulting heatmap.
    """

    def __init__(self, refine: bool = True, refine_hidden: int = 16) -> None:
        super().__init__()
        self.refine = (
            nn.Sequential(
                nn.Conv2d(1, refine_hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(refine_hidden, 1, kernel_size=1),
            ) if refine else None
        )

    def forward(self, template: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        H = self._cross_correlate(template, search)
        if self.refine is not None:
            H = self.refine(H)
        return H

    @staticmethod
    def _cross_correlate(template: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        B, C, h_t, w_t = template.shape
        Bs, Cs, h_s, w_s = search.shape
        if B != Bs or C != Cs:
            raise ValueError(
                f"shape mismatch: template={tuple(template.shape)}, "
                f"search={tuple(search.shape)}"
            )
        if h_s < h_t or w_s < w_t:
            raise ValueError(
                f"search must be >= template in HxW; "
                f"search={(h_s, w_s)}, template={(h_t, w_t)}"
            )
        # groups=B trick: each sample's template is its own conv kernel.
        weight = template
        inp = search.reshape(1, B * C, h_s, w_s)
        out = F.conv2d(inp, weight, groups=B)
        return out.transpose(0, 1).contiguous()  # (B, 1, h_o, w_o)
