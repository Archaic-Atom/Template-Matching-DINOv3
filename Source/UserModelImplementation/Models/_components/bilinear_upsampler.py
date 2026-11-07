# -*- coding: utf-8 -*-
"""Unguided bilinear counterpart of ``jbu_upsampler.JBUStack``.

This module exists for ONE reason: the guided-upsampling ablation asked for
by Reviewer 1 (TGRS-2026-06939, comment 7). Sec. III claims the joint
bilateral upsampler preserves boundaries that plain bilinear interpolation
would blur; ``BilinearStack`` is the control that lets that claim be measured
instead of asserted.

本模块只为一件事存在：审稿人 1 第 7 条要求的引导上采样消融。论文第三节断言
JBU 能保住 bilinear 会糊掉的边界；``BilinearStack`` 就是把这个断言变成可测量
对照的那一半。

**The control is clean by construction.** ``BilinearStack`` is a byte-for-byte
copy of ``JBUStack`` with the ``JBULearnedRange`` levels — and only those —
replaced by ``F.interpolate(mode='bilinear')``. Same input contract (stride-16
features + a guidance image), same per-level ×2 schedule, same 0.1-scaled
fixup residual, same final 1×1 projection to ``out_dim``, same output stride
and channel count. The guidance image is accepted and ignored, so the caller
needs no branching.

**对照是构造上干净的**：``BilinearStack`` 与 ``JBUStack`` 逐行一致，只把
``JBULearnedRange`` 那几级换成 ``F.interpolate(mode='bilinear')``，其余
（输入契约、逐级 ×2、0.1 残差 fixup、末端 1×1 投影、输出 stride 与通道数）
全部不动。guidance 照收不用，调用方无需分支。

**Not parameter-matched, deliberately.** Dropping ``JBULearnedRange`` drops
its parameters (6,233 per level: the range projection, the kernel fixup, and
the two scalar temperatures). Padding the control back up with filler
parameters would answer a different question — "does capacity help?" instead
of "does guidance help?" — so the counts are reported honestly instead. Run
``Source/Tools/count_upsampler_params.py`` for the exact ledger.

**刻意不做参数对齐**：去掉 ``JBULearnedRange`` 就少了它的参数（每级 6,233）。
硬塞填充参数补齐反而在回答另一个问题（“容量有没有用”而不是“引导有没有用”），
所以我们如实报告参数差。精确账目见 ``Source/Tools/count_upsampler_params.py``。

Unlike ``jbu_upsampler``, this module has **no FeatUp dependency** — it is
importable (and testable) anywhere torch is, which also makes it usable as the
"plain bilinear ×4" control inside the offline capacity probes.

与 ``jbu_upsampler`` 不同，本模块**不依赖 FeatUp**，只要有 torch 就能 import
和测试，因此也可直接用作离线容量探针里的 “bilinear ×4” 对照。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BilinearStack(nn.Module):
    """Unguided ×2^levels upsample + fixup residual + 1×1 projection.

    无引导的 ×2^levels 上采 + fixup 残差 + 1×1 投影。

    Structural mirror of ``jbu_upsampler.JBUStack``: each level upsamples the
    source by 2× (here with plain bilinear interpolation rather than an
    image-guided JBU kernel); after all levels a Dropout2d + 1×1 conv "fixup"
    residual block is applied at scale 0.1, followed by a final 1×1 conv that
    projects ``feat_dim`` → ``out_dim``.

    结构上与 ``JBUStack`` 一一对应：每级把 source 上采 2×（这里用纯双线性，
    而非图像引导的 JBU 核）；所有级之后接 Dropout + 1×1 conv 的 fixup 残差块
    （残差比例 0.1），最后用 1×1 conv 把通道数从 ``feat_dim`` 投到 ``out_dim``。

    ``key_dim`` and ``radius`` are accepted and ignored so that a config or a
    probe can hand the same kwargs to either stack.
    ``key_dim`` / ``radius`` 照收不用，好让同一份 kwargs 能喂给两种 stack。

    Args:
        feat_dim: input feature channels (1024 for DINOv3 ViT-L/16 and for the
            SAR encoder's ``sar16`` projection output).
            输入特征通道数（DINOv3 ViT-L/16 与 SAR encoder 的 ``sar16`` 都是 1024）。
        out_dim: output channels after the final 1×1 projection.
            最终 1×1 投影后的输出通道数。
        levels: number of 2× upsample steps. ``2`` = ×4 total (stride-16 →
            stride-4); ``1`` = ×2 total (stride-16 → stride-8).
            2× 上采的级数。
        guidance_dim: kept for signature parity with ``JBUStack``; the guidance
            image is not consumed by this stack.
            仅为与 ``JBUStack`` 保持签名一致；本 stack 不使用引导图。
        key_dim: ignored (JBU-only knob). 忽略（仅 JBU 使用）。
        radius: ignored (JBU-only knob). 忽略（仅 JBU 使用）。
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

        # No per-level parameters here — that absence IS the ablation.
        # 这里没有逐级参数，这个“缺失”本身就是被消融掉的东西。
        # Fixup residual block — identical to JBUStack's.
        # fixup 残差块，与 JBUStack 完全一致。
        self.fixup_proj = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
        )
        # Final 1×1 projection feat_dim → out_dim (no L2-norm — caller decides).
        # 末端 1×1 投影；是否 L2-normalize 由调用方处理。
        self.proj = nn.Conv2d(feat_dim, out_dim, kernel_size=1)

    @staticmethod
    def _upsample_once(source: torch.Tensor) -> torch.Tensor:
        """One unguided level: plain bilinear ×2.

        单级无引导上采：纯双线性 ×2。

        Stepping level by level (rather than one ×2^levels call) keeps the
        resampling schedule identical to ``JBUStack``'s, so the two stacks
        differ in the kernel alone.
        逐级上采（而不是一次 ×2^levels）是为了让重采样节奏与 ``JBUStack`` 一致，
        两者的差别只剩核本身。

        Args:
            source: ``(B, feat_dim, h, w)`` low-resolution feature.

        Returns:
            ``(B, feat_dim, 2h, 2w)`` upsampled feature.
        """
        return F.interpolate(source, scale_factor=2, mode='bilinear',
                             align_corners=False)

    def forward(self, source: torch.Tensor,
                guidance: torch.Tensor | None = None) -> torch.Tensor:
        """Run the full stack.

        Args:
            source: ``(B, feat_dim, h, w)`` low-resolution feature.
            guidance: accepted for signature parity with ``JBUStack`` and
                ignored. 为与 ``JBUStack`` 签名一致而保留，不使用。

        Returns:
            ``(B, out_dim, h·2^levels, w·2^levels)`` upsampled + projected
            feature. NOT L2-normalized.
        """
        del guidance  # unguided by design / 本对照就是不用引导
        x = source
        for _ in range(self.levels):
            x = self._upsample_once(x)
        # 0.1-scaled residual fixup — same as JBUStack.
        # 0.1 比例的残差 fixup，与 JBUStack 一致。
        x = self.fixup_proj(x) * 0.1 + x
        return self.proj(x)
