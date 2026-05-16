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
