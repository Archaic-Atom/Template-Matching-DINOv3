# -*- coding: utf-8 -*-
"""Learned-inner-product alignment head for Stage 1 (W_s / W_d projectors).

Stage-1-only scaffolding: linearly projects BOTH modalities into a shared
``proj_dim``-dim space where PatchNCE aligns them. The matcher never
consumes these projections (Stage 2 eats the trunk's C2/C3), so at Stage-2
load they are dropped as unexpected keys by ``strict=False`` — SimCLR-style
"the projector trains the trunk, then is thrown away".

仅 Stage-1 使用的"可学内积"对齐头:把两个模态各自 1×1 线性投影到共享的
``proj_dim`` 维空间,在该空间里做 PatchNCE 对齐。Stage-2 的匹配头只吃 trunk
的 C2/C3,不消费这两个投影 —— 加载时作为 unexpected key 被 strict=False
丢弃,即 SimCLR 式"projector 训练主干,用后即弃"。

Why W_d is learnable (2026-06-11 decision): the weak Stage-1 baseline
(peak@GT 0.59) is exactly the W_d=identity form; both probes that reached
0.80–0.86 projected the DINO side too (CCA-k22 / PCA-64). Since neither
projector needs to transfer to Stage 2, the DINO side is free to learn.
W_d 设为可学(2026-06-11 决定):弱 Stage-1 基线(peak@GT 0.59)正是
W_d=identity 的形式;两个到 0.80–0.86 的探针都投影了 DINO 侧。既然两个
投影头都不需要迁移到 Stage-2,DINO 侧没有理由不学。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register


@register("AlignmentIPHead")
class AlignmentIPHead(nn.Module):
    """Symmetric 1×1 linear projectors W_s / W_d into a shared space.

    Args:
        sar_dim: Channel dim of the SAR-side input (sar16, post-proj16).
            SAR 侧输入通道数(proj16 之后的 sar16)。
        dino_dim: Channel dim of the DINO-side input (block-23 tokens).
            DINO 侧输入通道数(block-23 token)。
        proj_dim: Shared projection dim. Keep ≥ 2× the rank target (64)
            so the VICReg variance term has room to spread mass.
            共享投影维度;至少取秩目标(64)的 2 倍,VICReg 才有铺开方差的余地。
    """

    def __init__(self, sar_dim: int = 1024, dino_dim: int = 1024,
                 proj_dim: int = 256) -> None:
        super().__init__()
        self.w_s = nn.Conv2d(sar_dim, proj_dim, 1, bias=False)
        self.w_d = nn.Conv2d(dino_dim, proj_dim, 1, bias=False)

    def forward(self, sar16: torch.Tensor,
                rgb16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return PRE-norm projections of the two feature maps.

        NCE-side normalization happens inside the loss; the VICReg rank
        constraint MUST see pre-norm features (L2-norm conceals rank
        collapse — the probe lesson).
        返回 pre-norm 投影。NCE 需要的归一化由 loss 内部做;VICReg 的秩
        约束必须作用在 pre-norm 特征上(L2 归一化会掩盖秩塌缩 —— 探针教训)。
        """
        # Symmetrize input scale (review 2026-06-11): sar16 arrives unit-norm
        # (ProjHead normalizes) but raw DINO block-23 tokens carry large,
        # position-varying norms; the quartic VICReg covariance term would let
        # the d-side dominate the whole objective at init. Token magnitude is
        # a nuisance (the normalize_logits lesson) — direction carries content.
        # 对称化输入尺度:sar16 已单位化,而 raw DINO token 范数大且不均,
        # VICReg 协方差项对输入尺度是四次方敏感,会让 d 侧在初始压过整个目标。
        # 范数是噪声(normalize_logits 的教训),方向才携带内容。
        rgb16 = F.normalize(rgb16, p=2, dim=1)
        return self.w_s(sar16), self.w_d(rgb16)
