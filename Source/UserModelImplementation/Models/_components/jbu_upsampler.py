# -*- coding: utf-8 -*-
"""FeatUp JBU upsampler wrappers for TM-DINOv3 Stage-2 matching head.

封装 FeatUp 的 JBU 上采，用作 Stage 2 中 stride-16 → stride-4/8 的可学习
上采。原图（RGB 或预处理后复制成 3 通道的 SAR）作为 guidance，让上采
后特征的边界对齐原图局部结构——比纯 bilinear 上采保留更多空间细节。

The JBUStack here is a slimmed version of FeatUp's official
`featup.upsamplers.JBUStack` (4 levels = ×16): we expose `levels` so
the same class can produce ×4 (stride-16 → stride-4) or ×2 (→ stride-8)
upsamples. A small fixup conv with 0.1 residual scale and a trainable
1×1 projection to the target dim complete the stack.

Training: parameters initialized at random (no FeatUp-DINOv2 pretrained
weights used — DINOv3 differs in patch size and the SAR-RGB matching
objective differs from FeatUp's reconstruction objective). Joint-trained
end-to-end with the matching head during Stage 2.

训练：随机初始化，与 matching head 在 Stage 2 端到端联合训练。不使用
FeatUp 在 DINOv2 上的预训练权重（patch size / 任务目标都不一致）。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from featup.upsamplers import JBULearnedRange


class JBUStack(nn.Module):
    """Slim FeatUp-style JBU stack with configurable level count.

    可配置层数的 FeatUp 风格 JBU 栈。

    Each level upsamples the source by 2× using image-guided JBU
    (`featup.upsamplers.JBULearnedRange`); after all levels we apply a
    Dropout2d + 1×1 conv as a "fixup" residual block (small scale 0.1)
    and a final 1×1 conv that projects the feature channel count from
    ``feat_dim`` to ``out_dim``.

    每级使用图像导引的 JBU 把 source 上采 2×；所有级之后接 Dropout +
    1×1 conv 的 fixup 残差块（残差比例 0.1），最后用 1×1 conv 把通道数
    从 ``feat_dim`` 投影到 ``out_dim``。

    Args:
        feat_dim: input feature channels (e.g. 1024 for DINOv3 ViT-L/16,
            or for SAR encoder's ``sar16`` projection output).
            输入特征通道数（DINOv3 ViT-L/16 = 1024；SAR encoder 的
            ``sar16`` 经 ProjHead 后也是 1024）。
        out_dim: output channels after the final 1×1 projection.
            最终 1×1 投影后的输出通道数。
        levels: number of 2× upsample steps. ``2`` = ×4 total (stride-16
            → stride-4); ``1`` = ×2 total (stride-16 → stride-8).
            2× 上采的级数。
        guidance_dim: channels in the guide image. Default 3 (RGB; SAR is
            pre-duplicated to 3 channels in the preprocessing pipeline).
            引导图通道数（默认 3；SAR 在预处理时已复制到 3 通道）。
        key_dim: JBULearnedRange's internal key dim. Default 32.
            JBULearnedRange 内部 key 维度。
        radius: JBULearnedRange kernel radius (diameter = 2*r+1).
            JBU 核半径。
    """

    def __init__(self, feat_dim: int, out_dim: int, levels: int = 2,
                 guidance_dim: int = 3, key_dim: int = 32,
                 radius: int = 3) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1; got {levels}")
        self.levels = int(levels)
        self.feat_dim = int(feat_dim)
        self.out_dim = int(out_dim)
        self.guidance_dim = int(guidance_dim)

        # One JBULearnedRange per level — independent params, not shared.
        # 每级 JBU 参数独立，不共享。
        self.ups = nn.ModuleList([
            JBULearnedRange(guidance_dim, feat_dim, key_dim, radius=radius)
            for _ in range(self.levels)
        ])
        # Fixup residual block (matches user's earlier draft).
        # 与用户原稿一致的 fixup 残差块（Dropout2d + 1×1 conv，×0.1 残差）。
        self.fixup_proj = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
        )
        # Final 1×1 projection feat_dim → out_dim (no L2-norm — caller decides).
        # 末端 1×1 投影；是否 L2-normalize 由调用方处理。
        self.proj = nn.Conv2d(feat_dim, out_dim, kernel_size=1)

    def _upsample_once(self, source: torch.Tensor, guidance: torch.Tensor,
                       up: nn.Module) -> torch.Tensor:
        """One JBU level: pool guide to 2× source size and apply JBU.

        单级 JBU 调用：把 guide 下采到 source 尺寸的 2×，再调用 JBU。

        Args:
            source: ``(B, feat_dim, h, w)`` low-resolution feature.
            guidance: ``(B, guidance_dim, H, W)`` high-resolution image.
                Pooled to ``(2h, 2w)`` for this level.
            up: a ``JBULearnedRange`` module from ``self.ups``.

        Returns:
            ``(B, feat_dim, 2h, 2w)`` upsampled feature.
        """
        _, _, h, w = source.shape
        small_g = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        return up(source, small_g)

    def forward(self, source: torch.Tensor,
                guidance: torch.Tensor) -> torch.Tensor:
        """Run the full stack.

        Args:
            source: ``(B, feat_dim, h, w)`` low-resolution feature.
            guidance: ``(B, guidance_dim, ≥h·2^levels, ≥w·2^levels)``
                high-resolution image; each level pools it to its own
                2× target size internally.

        Returns:
            ``(B, out_dim, h·2^levels, w·2^levels)`` upsampled +
            projected feature. NOT L2-normalized.
        """
        x = source
        for up in self.ups:
            x = self._upsample_once(x, guidance, up)
        # 0.1-scaled residual fixup matches user's earlier draft.
        # 0.1 比例的残差 fixup，与用户原稿一致。
        x = self.fixup_proj(x) * 0.1 + x
        return self.proj(x)
