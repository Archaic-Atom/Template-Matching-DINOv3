# -*- coding: utf-8 -*-
"""Multi-level supervision for the cascade head (TM_Model_C Stage 2).

Per-level BCE-with-logits against a Gaussian target centered at the GT
displacement (in the cell coordinates of that level). High-resolution levels
are gated by per-level warmup ramps so the coarse heads converge first.

Output dict consumed (from CascadeHead.forward):
    heatmap_per_stride: {4: H_4}              or  {4: H_4, 8: H_8}
    heatmap_cascade:    {2: H_2, 1: H_1}
    cascade_anchor_2_cells: (B, 2) int       — stride-2 cell anchor of H_2
    fine_anchor_pix:        (B, 2) int       — stride-1 pixel anchor of H_1

Loss = sum over levels:
        Level 4: BCE(H_4, gaussian(disp_xy / 4, sigma_4)) * w_4
        Level 8: BCE(H_8, gaussian(disp_xy / 8, sigma_8)) * w_8        (if H_8)
        Level 2: BCE(H_2, gaussian(disp_xy / 2 - anchor_2, sigma_2)) * w_2 * warmup(e, e_2)
        Level 1: BCE(H_1, gaussian(disp_xy     - anchor_1, sigma_1)) * w_1 * warmup(e, e_1)

Samples whose GT falls outside the windowed levels are masked out of those
levels' loss (they're already covered by the coarse levels).
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F

from .registry import register
from .matching_loss import gaussian_target, matching_loss_single


def _linear_warmup(current_epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, current_epoch / float(warmup_epochs))


def _windowed_bce(heatmap_logits: torch.Tensor,
                  target_xy_cells: torch.Tensor,
                  sigma_cells: float,
                  pos_weight: Optional[float] = None) -> torch.Tensor:
    """BCE-with-logits in a (B, 1, H, W) cropped window. target_xy in cell coords
    of the heatmap (NOT image space). Samples whose target is outside the heatmap
    contribute 0 (already supervised by coarser levels).
    """
    B, _, H, W = heatmap_logits.shape
    inside = ((target_xy_cells[:, 0] >= 0) & (target_xy_cells[:, 0] <= W - 1)
              & (target_xy_cells[:, 1] >= 0) & (target_xy_cells[:, 1] <= H - 1))
    if not inside.any():
        return heatmap_logits.sum() * 0.0
    H_in = heatmap_logits[inside]
    t_in = target_xy_cells[inside]
    target = gaussian_target(t_in, H, W, sigma=sigma_cells, stride=1)
    pw = (torch.tensor(pos_weight, device=H_in.device, dtype=H_in.dtype)
          if pos_weight is not None else None)
    return F.binary_cross_entropy_with_logits(H_in, target, pos_weight=pw)


@register("cascade_loss")
def cascade_loss(output: dict,
                 displacement_xy: torch.Tensor,
                 *,
                 config: dict,
                 current_epoch: int):
    """Combined cascade loss across stride 4 / 8 / 2 / 1 with warmup on fine levels.

    Returns:
        Tuple ``(total, components)``. ``components`` keys (always present):
        ``coarse_s4``, ``coarse_s8`` (zero if stride-8 disabled), ``cascade_s2``,
        ``cascade_s1`` — raw per-level losses (before weight + warmup), useful
        for diagnosing whether each level is learning.
    """
    # --- Coarse levels (full-image, no windowing) ---
    heatmaps_full = output['heatmap_per_stride']  # {4: H_4} or {4, 8}
    pos_weight_coarse = config.get('match_pos_weight', None)

    sigma_4 = float(config.get('match_sigma4', 1.0))
    w_4 = float(config.get('match_w4', 1.0))
    L_4 = matching_loss_single(heatmaps_full[4], displacement_xy,
                               stride=4, sigma=sigma_4,
                               pos_weight=pos_weight_coarse)

    L = w_4 * L_4
    L_8 = None
    if 8 in heatmaps_full:
        sigma_8 = float(config.get('match_sigma8', 1.0))
        w_8 = float(config.get('match_w8', 1.0))
        L_8 = matching_loss_single(heatmaps_full[8], displacement_xy,
                                   stride=8, sigma=sigma_8,
                                   pos_weight=pos_weight_coarse)
        L = L + w_8 * L_8

    # --- Cascade levels (windowed) ---
    cascade = output['heatmap_cascade']
    pos_weight_fine = config.get('match_fine_pos_weight', None)

    # Level 1/2: heatmap covers stride-2 cells starting at anchor_2.
    H_2 = cascade[2]
    anchor_2 = output['cascade_anchor_2_cells'].float()      # cell coords
    target_2 = (displacement_xy.float() / 2.0) - anchor_2     # in stride-2 cells
    sigma_2 = float(config.get('match_sigma_2', 1.0))
    w_2 = float(config.get('match_w_2', 0.5))
    warmup_2 = int(config.get('warmup_epoch_2', 0))
    L_2 = _windowed_bce(H_2, target_2, sigma_cells=sigma_2, pos_weight=pos_weight_fine)
    L = L + w_2 * _linear_warmup(current_epoch, warmup_2) * L_2

    # Level 1/1: heatmap covers stride-1 pixels starting at anchor_1.
    H_1 = cascade[1]
    anchor_1 = output['fine_anchor_pix'].float()              # pixel coords
    target_1 = displacement_xy.float() - anchor_1             # in pixels (stride-1 cells)
    sigma_1 = float(config.get('match_sigma_1', 1.0))
    w_1 = float(config.get('match_w_1', 1.0))
    warmup_1 = int(config.get('warmup_epoch_1', 0))
    L_1 = _windowed_bce(H_1, target_1, sigma_cells=sigma_1, pos_weight=pos_weight_fine)
    L = L + w_1 * _linear_warmup(current_epoch, warmup_1) * L_1

    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    components = {
        'coarse_s4': L_4.detach(),
        'coarse_s8': L_8.detach() if L_8 is not None else zero,
        'cascade_s2': L_2.detach(),
        'cascade_s1': L_1.detach(),
    }
    return L, components
