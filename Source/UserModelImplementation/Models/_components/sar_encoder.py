# -*- coding: utf-8 -*-
"""SAR encoder: ResNet-like CNN with GroupNorm, projects to DINO dim at stride 16.

Mirrors TM_Model_A/Nets/_create_sar_encoder.py exactly so the saved
state_dict keys (sar_encoder.stem.*, sar_encoder.layer{1,2,3}.*,
sar_encoder.proj16.*) are byte-identical across variants. Stage 1 ckpts
load into any variant that registers this same class.

The class also exposes `.channels` so the builder can auto-wire downstream
matching heads with the correct stride-4 / stride-8 channel widths.
"""
from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register


def _gn(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    g = num_groups if num_channels >= num_groups else 1
    return nn.GroupNorm(g, num_channels)


class BasicBlockGN(nn.Module):
    """ResNet BasicBlock with GroupNorm (no BatchNorm)."""
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2 = _gn(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                _gn(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = F.relu(self.gn1(out), inplace=True)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class ProjHead(nn.Module):
    """Project feature map to DINO dim + L2 normalize per pixel."""

    def __init__(self, in_ch: int, out_dim: int = 1024) -> None:
        super().__init__()
        mid = max(in_ch, out_dim)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            _gn(mid),
            nn.GELU(),
            nn.Conv2d(mid, out_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=1)


@register("SarResNet16GN")
class SarResNet16GN(nn.Module):
    """ResNet-like CNN (GN) up to stride-16, for SAR↔DINO alignment.

    forward(x) returns {'sar16', 'raw16', 'C2', 'C3'}:
        sar16: L2-normalized, projected to dino_dim, stride 16
        raw16: pre-projection stride-16 feature
        C2:    stride-4 feature (used by stride-4 matching head)
        C3:    stride-8 feature (used by stride-8 matching head)
    """

    def __init__(
        self,
        in_chans: int = 3,
        dino_dim: int = 1024,
        layers: Tuple[int, int, int] = (3, 4, 6),
        channels: Tuple[int, int, int] = (64, 128, 256),
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        n1, n2, n3 = layers
        # Exposed for builder auto-wiring.
        self.channels = (c1, c2, c3)

        # Stem: stride-2 conv + stride-2 maxpool → total stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, c1, kernel_size=7, stride=2, padding=3, bias=False),
            _gn(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(c1, c1, n1, stride=1)   # stride 4
        self.layer2 = self._make_layer(c1, c2, n2, stride=2)   # stride 8
        self.layer3 = self._make_layer(c2, c3, n3, stride=2)   # stride 16
        self.proj16 = ProjHead(in_ch=c3, out_dim=dino_dim)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlockGN(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlockGN(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x);  c2 = x
        x = self.layer2(x);  c3 = x
        x = self.layer3(x);  c4 = x
        sar16 = self.proj16(c4)
        return {"sar16": sar16, "raw16": c4, "C2": c2, "C3": c3}
