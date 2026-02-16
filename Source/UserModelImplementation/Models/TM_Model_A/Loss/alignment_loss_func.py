# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def _grid_index(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Builds (N, 2) grid coordinates (y, x) for an h x w token grid."""
    ys = torch.arange(h, device=device)
    xs = torch.arange(w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # (N, 2)
    return coords


def soft_window_patchnce_loss(
    sar16: torch.Tensor,
    rgb16: torch.Tensor,
    tau: float = 0.07,
    radius: int = 1,
) -> torch.Tensor:
    """Computes soft-window PatchNCE loss with batch negatives.

    For each SAR token, positives are RGB tokens within a (2*radius+1)^2 window
    around the same (y, x) position (within the SAME sample). Negatives include
    all other RGB tokens across the whole batch.

    Args:
        sar16: SAR features of shape (B, D, h, w). Recommended to be L2-normalized.
        rgb16: RGB(DINO) features of shape (B, D, h, w). Recommended to be L2-normalized.
        tau: Temperature for InfoNCE.
        radius: Positive window radius. radius=1 corresponds to 3x3 window.

    Returns:
        A scalar loss tensor.
    """
    if sar16.shape != rgb16.shape:
        raise ValueError(f"Shape mismatch: sar16={sar16.shape}, rgb16={rgb16.shape}")

    b, d, h, w = sar16.shape
    n = h * w
    device = sar16.device

    # Flatten tokens: (B, D, h, w) -> (B, N, D)
    s = sar16.flatten(2).transpose(1, 2).contiguous()
    r = rgb16.flatten(2).transpose(1, 2).contiguous()

    # Batch negatives: compare each SAR token to all RGB tokens in the batch.
    # logits: (B*N, B*N)
    s_all = s.view(b * n, d)
    r_all = r.view(b * n, d)
    logits = (s_all @ r_all.t()) / tau

    # Build within-window mask for tokens in the same sample.
    coords = _grid_index(h, w, device)  # (N, 2)
    yy = coords[:, 0]
    xx = coords[:, 1]
    dy = (yy[:, None] - yy[None, :]).abs()
    dx = (xx[:, None] - xx[None, :]).abs()
    within = (dy <= radius) & (dx <= radius)  # (N, N)

    # Block-diagonal positive mask across batch: (B*N, B*N)
    pos_mask = torch.zeros((b * n, b * n), device=device, dtype=torch.bool)
    for bi in range(b):
        a = bi * n
        pos_mask[a:a + n, a:a + n] = within

    # Soft positives: logsumexp over positives vs logsumexp over all.
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    lse_pos = torch.logsumexp(pos_logits, dim=1)
    lse_all = torch.logsumexp(logits, dim=1)

    loss = (-(lse_pos - lse_all)).mean()
    return loss


def global_cosine_loss(sar16: torch.Tensor, rgb16: torch.Tensor) -> torch.Tensor:
    """Computes global cosine loss using mean pooled tokens."""
    g_sar = F.normalize(sar16.mean(dim=(2, 3)), p=2, dim=1)
    g_rgb = F.normalize(rgb16.mean(dim=(2, 3)), p=2, dim=1)
    return (1.0 - (g_sar * g_rgb).sum(dim=1)).mean()


def alignment_loss(
    sar16: torch.Tensor,
    rgb16: torch.Tensor,
    tau: float = 0.07,
    radius: int = 1,
    lambda_global: float = 0.1,
) -> torch.Tensor:
    """Computes combined alignment loss: local PatchNCE + global cosine.

    Args:
        sar16: (B, D, h, w) SAR features (L2-normalized recommended).
        rgb16: (B, D, h, w) RGB(DINO) features (L2-normalized recommended).
        tau: Temperature for local PatchNCE.
        radius: Positive window radius for local PatchNCE.
        lambda_global: Weight for global cosine loss.

    Returns:
        A scalar loss tensor.
    """
    l_local = soft_window_patchnce_loss(sar16, rgb16, tau=tau, radius=radius)
    l_global = global_cosine_loss(sar16, rgb16)
    return l_local + lambda_global * l_global
