# -*- coding: utf-8 -*-
"""Stage-1 alignment loss: PatchNCE on local window + global cosine.

Copied verbatim from TM_Model_A/Loss/alignment_loss_func.py so behavior
matches the in-progress baseline run exactly. The public `alignment_loss`
is registered for config-driven dispatch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import register


def _grid_index(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(h, device=device)
    xs = torch.arange(w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)


def soft_window_patchnce_loss(
    sar16: torch.Tensor,
    rgb16: torch.Tensor,
    tau: float = 0.07,
    radius: int = 1,
) -> torch.Tensor:
    """Soft-window PatchNCE with batch negatives. See A's docstring."""
    if sar16.shape != rgb16.shape:
        raise ValueError(f"Shape mismatch: sar16={sar16.shape}, rgb16={rgb16.shape}")

    b, d, h, w = sar16.shape
    n = h * w
    device = sar16.device

    s = sar16.flatten(2).transpose(1, 2).contiguous()
    r = rgb16.flatten(2).transpose(1, 2).contiguous()

    s_all = s.view(b * n, d)
    r_all = r.view(b * n, d)
    logits = (s_all @ r_all.t()) / tau

    coords = _grid_index(h, w, device)
    yy = coords[:, 0]; xx = coords[:, 1]
    dy = (yy[:, None] - yy[None, :]).abs()
    dx = (xx[:, None] - xx[None, :]).abs()
    within = (dy <= radius) & (dx <= radius)

    pos_mask = torch.zeros((b * n, b * n), device=device, dtype=torch.bool)
    for bi in range(b):
        a = bi * n
        pos_mask[a:a + n, a:a + n] = within

    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    lse_pos = torch.logsumexp(pos_logits, dim=1)
    lse_all = torch.logsumexp(logits, dim=1)
    return (-(lse_pos - lse_all)).mean()


def global_cosine_loss(sar16: torch.Tensor, rgb16: torch.Tensor) -> torch.Tensor:
    g_sar = F.normalize(sar16.mean(dim=(2, 3)), p=2, dim=1)
    g_rgb = F.normalize(rgb16.mean(dim=(2, 3)), p=2, dim=1)
    return (1.0 - (g_sar * g_rgb).sum(dim=1)).mean()


@register("alignment_loss")
def alignment_loss(
    sar16: torch.Tensor,
    rgb16: torch.Tensor,
    tau: float = 0.07,
    radius: int = 1,
    lambda_global: float = 0.1,
) -> torch.Tensor:
    """Combined Stage-1 alignment loss: local PatchNCE + global cosine."""
    l_local = soft_window_patchnce_loss(sar16, rgb16, tau=tau, radius=radius)
    l_global = global_cosine_loss(sar16, rgb16)
    return l_local + lambda_global * l_global
