from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    g = num_groups if num_channels >= num_groups else 1
    return nn.GroupNorm(g, num_channels)


class BasicBlockGN(nn.Module):
    """ResNet BasicBlock with GroupNorm (no BatchNorm)."""
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2 = gn(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                gn(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = F.relu(self.gn1(out), inplace=True)
        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(out + identity, inplace=True)
        return out


class ProjHead(nn.Module):
    """Project feature map to DINO dim + L2 normalize."""

    def __init__(self, in_ch: int, out_dim: int = 1024) -> None:
        super().__init__()
        mid = max(in_ch, out_dim)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            gn(mid),
            nn.GELU(),
            nn.Conv2d(mid, out_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)


class SarResNet16GN(nn.Module):
    """ResNet-like CNN (GN) up to stride-16, for SAR->DINO alignment."""

    def __init__(
        self,
        in_chans: int = 3,
        dino_dim: int = 1024,
        layers: Tuple[int, int, int] = (3, 4, 6),  # like ResNet-34 up to layer3
        channels: Tuple[int, int, int] = (64, 128, 256),
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        n1, n2, n3 = layers

        # Stem: stride=2 conv + stride=2 maxpool => total stride=4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, c1, kernel_size=7, stride=2, padding=3, bias=False),
            gn(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(c1, c1, n1, stride=1)  # stride=4
        self.layer2 = self._make_layer(c1, c2, n2, stride=2)  # stride=8
        self.layer3 = self._make_layer(c2, c3, n3, stride=2)  # stride=16

        self.proj16 = ProjHead(in_ch=c3, out_dim=dino_dim)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlockGN(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlockGN(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)           # stride=4
        x = self.layer1(x)
        c2 = x                     # 1/4

        x = self.layer2(x)
        c3 = x                     # 1/8

        x = self.layer3(x)
        c4 = x                     # 1/16

        sar16 = self.proj16(c4)    # B x 1024 x 4 x 4 (if input 64x64)
        return {"sar16": sar16, "raw16": c4, "C2": c2, "C3": c3}


if __name__ == "__main__":
    m = SarResNet16GN().cuda()
    x = torch.randn(2, 3, 64, 64, device="cuda")
    out = m(x)
    print(out["sar16"].shape, out["C2"].shape, out["C3"].shape, out["raw16"].shape)
