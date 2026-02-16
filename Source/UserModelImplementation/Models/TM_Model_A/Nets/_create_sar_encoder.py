# -*- coding: utf-8 -*-
"""SAR ConvNeXt encoder for 1/16 alignment with DINOv3 tokens.

This module builds a ConvNeXt backbone (via timm) and projects the stride-16
feature map to the DINO token embedding dimension (e.g., 1024 for ViT-L/16).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:
    raise ImportError("Please install timm: python -m pip install timm") from exc


class ProjHead16(nn.Module):
    """Projection head that maps stride-16 features to DINO token dimension.

    The head uses a lightweight 1x1 -> GN -> GELU -> 1x1 design, followed by
    L2 normalization along the channel dimension to stabilize token similarity.
    """

    def __init__(self, in_ch: int, out_dim: int = 1024) -> None:
        """Initializes the projection head.

        Args:
            in_ch: Number of input channels in the stride-16 feature map.
            out_dim: Target embedding dimension (DINO token dim).
        """
        super().__init__()
        mid = max(out_dim, in_ch)
        self._proj = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32 if mid >= 32 else 1, num_channels=mid),
            nn.GELU(),
            nn.Conv2d(mid, out_dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature map of shape (B, in_ch, H/16, W/16).

        Returns:
            Projected feature map of shape (B, out_dim, H/16, W/16), L2-normalized
            along the channel dimension.
        """
        x = self._proj(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class SarConvNeXt16(nn.Module):
    """SAR encoder (ConvNeXt) for 1/16 alignment.

    This encoder returns a projected stride-16 feature map ("sar16") whose
    channel dimension matches the DINO token embedding dimension.
    """

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        dino_dim: int = 1024,
        in_chans: int = 3,
    ) -> None:
        """Initializes the SAR encoder.

        Args:
            model_name: ConvNeXt variant name supported by timm (e.g., "convnext_tiny").
            pretrained: Whether to load ImageNet-pretrained ConvNeXt weights.
            dino_dim: DINO token embedding dimension (ViT-L/16 commonly uses 1024).
            in_chans: Number of input channels (pseudo-color SAR uses 3).
        """
        super().__init__()

        # Only keep feature maps up to stride-16: (1/4, 1/8, 1/16).
        self._backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2),
        )

        feat_channels = self._backbone.feature_info.channels()
        c4 = feat_channels[2]  # Stride-16 stage (1/16).
        self._proj16 = ProjHead16(in_ch=c4, out_dim=dino_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input SAR image tensor of shape (B, in_chans, H, W).

        Returns:
            A dict containing:
              - "sar16": Projected stride-16 feature map (B, dino_dim, H/16, W/16).
              - "raw16": Raw stride-16 ConvNeXt feature map before projection.
              - "C2": Stride-4 feature map (B, C2, H/4, W/4).
              - "C3": Stride-8 feature map (B, C3, H/8, W/8).
        """
        feats = self._backbone(x)  # [C2(1/4), C3(1/8), C4(1/16)]
        c2, c3, c4 = feats

        sar16 = self._proj16(c4)
        return {
            "sar16": sar16,
            "raw16": c4,
            "C2": c2,
            "C3": c3,
        }


def _demo() -> None:
    """Runs a small shape check demo."""
    model = SarConvNeXt16(
        model_name="convnext_tiny",
        pretrained=False,
        dino_dim=1024,
        in_chans=3,
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    print("sar16:", out["sar16"].shape)  # Expected: (2, 1024, 4, 4)
    print("raw16:", out["raw16"].shape)
    print("C2:", out["C2"].shape)
    print("C3:", out["C3"].shape)


if __name__ == "__main__":
    _demo()
