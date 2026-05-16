# -*- coding: utf-8 -*-
"""Sub-pixel peak refinement utilities.

Given a heatmap and its integer argmax, fit a 2D quadratic surface over
the 3x3 neighborhood around each peak and return sub-pixel offsets in
[-0.5, 0.5]. Used by TM_Model_D's accuracy() to recover sub-pixel precision
from a discrete argmax over a coarse heatmap.

The quadratic fit follows the standard Taylor-expansion form:
    f(u, v) ≈ a + b*u + c*v + d*u² + e*v² + f*u*v
Differentiating and setting ∂f/∂u = ∂f/∂v = 0 gives offsets:
    [u_off, v_off] = -0.5 * H^{-1} g
where H is the local 2x2 Hessian and g the local gradient at the peak.

This is the same trick used in SIFT/Hessian detectors and modern stereo
methods (e.g., PSMNet's regression head replaces it with soft argmax, but
quadratic fit is the cheap drop-in version).
"""
from __future__ import annotations

import torch

from .registry import register


@register("quadratic_subpixel_fit")
def quadratic_subpixel_fit(heatmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard argmax + 2D quadratic peak refinement.

    Args:
        heatmap: (B, 1, H, W) logits (or similarities; just needs a peak).
                 Shape (B, H, W) also accepted.

    Returns:
        peak_int_xy:  (B, 2) int64 integer-pixel argmax (u, v) where
                      u indexes the W dim and v indexes the H dim.
        sub_offset_xy: (B, 2) float sub-pixel offset in [-0.5, 0.5]
                       (u_off, v_off). Final sub-pixel peak is
                       peak_int_xy.float() + sub_offset_xy.
                       Returns zeros at borders (no neighborhood available).
    """
    if heatmap.dim() == 4:
        if heatmap.shape[1] != 1:
            raise ValueError(
                f"Expected (B,1,H,W); got channel dim {heatmap.shape[1]}"
            )
        H_map = heatmap.squeeze(1)
    elif heatmap.dim() == 3:
        H_map = heatmap
    else:
        raise ValueError(f"Heatmap must be (B,H,W) or (B,1,H,W); got {tuple(heatmap.shape)}")

    B, H, W = H_map.shape
    flat = H_map.view(B, -1)
    idx = flat.argmax(dim=1)
    v = (idx // W).long()
    u = (idx % W).long()
    peak_int = torch.stack([u, v], dim=1)

    # Sub-pixel via 1D quadratic fit independently in each axis.
    # f(u-1), f(u), f(u+1)  →  u_off = 0.5 * (f(u-1) - f(u+1)) / (f(u-1) - 2*f(u) + f(u+1))
    sub_off = torch.zeros((B, 2), dtype=torch.float32, device=heatmap.device)
    safe_u = (u > 0) & (u < W - 1)
    safe_v = (v > 0) & (v < H - 1)

    if safe_u.any():
        bi = torch.arange(B, device=heatmap.device)[safe_u]
        ui = u[safe_u]
        vi = v[safe_u]
        f_minus = H_map[bi, vi, ui - 1].float()
        f_zero  = H_map[bi, vi, ui].float()
        f_plus  = H_map[bi, vi, ui + 1].float()
        denom = (f_minus - 2 * f_zero + f_plus).clamp_max(-1e-12)  # negative at a max
        offs = 0.5 * (f_minus - f_plus) / denom
        sub_off[safe_u, 0] = offs.clamp(-0.5, 0.5)

    if safe_v.any():
        bi = torch.arange(B, device=heatmap.device)[safe_v]
        ui = u[safe_v]
        vi = v[safe_v]
        f_minus = H_map[bi, vi - 1, ui].float()
        f_zero  = H_map[bi, vi, ui].float()
        f_plus  = H_map[bi, vi + 1, ui].float()
        denom = (f_minus - 2 * f_zero + f_plus).clamp_max(-1e-12)
        offs = 0.5 * (f_minus - f_plus) / denom
        sub_off[safe_v, 1] = offs.clamp(-0.5, 0.5)

    return peak_int, sub_off
