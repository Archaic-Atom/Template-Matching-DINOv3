# -*- coding: utf-8 -*-
"""Stage-2 matching loss + metrics.

Core functions (matching_loss, fine_matching_loss, two_stage_matching_loss,
peak_displacement, fuse_heatmaps, matching_accuracy, two_stage_matching_accuracy)
are copied from TM_Model_A/Loss/matching_loss_func.py so behavior matches the
baseline exactly.

The config-driven adapter registered for builder dispatch is
`two_stage_matching_loss_config` — it takes the variant's loss config block
plus current_epoch and computes the correct fine_weight (linear warmup),
then calls two_stage_matching_loss.
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .registry import register


# ---------------------------------------------------------------------------
# Core loss building blocks (verbatim from TM_Model_A)
# ---------------------------------------------------------------------------

def gaussian_target(displacement_xy: torch.Tensor,
                    h_out: int, w_out: int,
                    sigma: float, stride: int) -> torch.Tensor:
    if displacement_xy.dim() != 2 or displacement_xy.shape[1] != 2:
        raise ValueError(f"displacement_xy must be (B, 2); got {tuple(displacement_xy.shape)}")
    device = displacement_xy.device
    dx = displacement_xy[:, 0].float() / float(stride)
    dy = displacement_xy[:, 1].float() / float(stride)
    ys = torch.arange(h_out, device=device, dtype=torch.float32)
    xs = torch.arange(w_out, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    yy = yy.unsqueeze(0); xx = xx.unsqueeze(0)
    dx = dx.view(-1, 1, 1); dy = dy.view(-1, 1, 1)
    dist2 = (xx - dx) ** 2 + (yy - dy) ** 2
    target = torch.exp(-dist2 / (2.0 * sigma ** 2))
    return target.unsqueeze(1)


def matching_loss_single(heatmap_logits: torch.Tensor,
                         displacement_xy: torch.Tensor,
                         stride: int, sigma: float,
                         pos_weight: Optional[float] = None) -> torch.Tensor:
    _, _, h, w = heatmap_logits.shape
    target = gaussian_target(displacement_xy, h, w, sigma, stride)
    pw = (torch.tensor(pos_weight, device=heatmap_logits.device, dtype=heatmap_logits.dtype)
          if pos_weight is not None else None)
    return F.binary_cross_entropy_with_logits(heatmap_logits, target, pos_weight=pw)


def matching_loss(heatmap_logits_per_stride: Dict[int, torch.Tensor],
                  displacement_xy: torch.Tensor,
                  scale_weights: Optional[Dict[int, float]] = None,
                  sigma_per_stride: Optional[Dict[int, float]] = None,
                  pos_weight: Optional[float] = None) -> torch.Tensor:
    if not heatmap_logits_per_stride:
        raise ValueError("heatmap_logits_per_stride is empty")
    if scale_weights is None:
        scale_weights = {s: 1.0 for s in heatmap_logits_per_stride}
    if sigma_per_stride is None:
        sigma_per_stride = {s: 1.0 for s in heatmap_logits_per_stride}
    total = None
    for s, H in heatmap_logits_per_stride.items():
        L_s = matching_loss_single(H, displacement_xy, stride=s,
                                   sigma=sigma_per_stride[s], pos_weight=pos_weight)
        w_s = scale_weights[s]
        total = (w_s * L_s) if total is None else (total + w_s * L_s)
    return total


def fine_matching_loss(heatmap_fine_logits: torch.Tensor,
                       displacement_xy: torch.Tensor,
                       fine_anchor_pix: torch.Tensor,
                       sigma: float = 1.0,
                       pos_weight: Optional[float] = None) -> torch.Tensor:
    B, _, h_f, w_f = heatmap_fine_logits.shape
    rel = displacement_xy.float() - fine_anchor_pix.float()
    inside = (rel[:, 0] >= 0) & (rel[:, 0] <= w_f - 1) & \
             (rel[:, 1] >= 0) & (rel[:, 1] <= h_f - 1)
    if not inside.any():
        return heatmap_fine_logits.sum() * 0.0
    H_in = heatmap_fine_logits[inside]
    rel_in = rel[inside]
    target = gaussian_target(rel_in, h_f, w_f, sigma, stride=1)
    pw = (torch.tensor(pos_weight, device=H_in.device, dtype=H_in.dtype)
          if pos_weight is not None else None)
    return F.binary_cross_entropy_with_logits(H_in, target, pos_weight=pw)


def two_stage_matching_loss(output: dict,
                            displacement_xy: torch.Tensor,
                            coarse_kwargs: Optional[dict] = None,
                            fine_sigma: float = 1.0,
                            fine_weight: float = 1.0,
                            fine_pos_weight: Optional[float] = None) -> torch.Tensor:
    coarse_kwargs = coarse_kwargs or {}
    L_coarse = matching_loss(output['heatmap_per_stride'], displacement_xy, **coarse_kwargs)
    L_fine = fine_matching_loss(
        output['heatmap_fine'], displacement_xy, output['fine_anchor_pix'],
        sigma=fine_sigma, pos_weight=fine_pos_weight)
    return L_coarse + fine_weight * L_fine


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def peak_displacement(heatmap_logits: torch.Tensor, stride: int) -> torch.Tensor:
    B = heatmap_logits.shape[0]
    h, w = heatmap_logits.shape[-2:]
    flat = heatmap_logits.view(B, -1)
    idx = flat.argmax(dim=1)
    v = (idx // w).float() * float(stride)
    u = (idx % w).float() * float(stride)
    return torch.stack([u, v], dim=1)


def fuse_heatmaps(heatmap_logits_per_stride: Dict[int, torch.Tensor],
                  primary_stride: int,
                  weights: Optional[Dict[int, float]] = None) -> torch.Tensor:
    if primary_stride not in heatmap_logits_per_stride:
        raise ValueError(f"primary_stride={primary_stride} not in inputs")
    if weights is None:
        weights = {s: 1.0 for s in heatmap_logits_per_stride}
    target_hw = heatmap_logits_per_stride[primary_stride].shape[-2:]
    fused = None
    total_w = 0.0
    for s, H in heatmap_logits_per_stride.items():
        if H.shape[-2:] != target_hw:
            H = F.interpolate(H, size=target_hw, mode='bilinear', align_corners=False)
        w_s = float(weights[s])
        total_w += w_s
        fused = (w_s * H) if fused is None else (fused + w_s * H)
    return fused / total_w if total_w > 0 else fused


def matching_accuracy(heatmap_logits_per_stride: Dict[int, torch.Tensor],
                      displacement_xy: torch.Tensor,
                      primary_stride: int = 4,
                      scale_weights: Optional[Dict[int, float]] = None) -> torch.Tensor:
    if len(heatmap_logits_per_stride) == 1:
        s, H = next(iter(heatmap_logits_per_stride.items()))
        pred = peak_displacement(H, stride=s)
    else:
        fused = fuse_heatmaps(heatmap_logits_per_stride, primary_stride, scale_weights)
        pred = peak_displacement(fused, stride=primary_stride)
    err = ((pred - displacement_xy.float()) ** 2).sum(dim=1).sqrt()
    return err.mean()


def two_stage_matching_accuracy(output: dict,
                                displacement_xy: torch.Tensor,
                                primary_stride: int = 4,
                                scale_weights: Optional[Dict[int, float]] = None
                                ) -> torch.Tensor:
    fine_logits = output['heatmap_fine']
    anchor = output['fine_anchor_pix'].float()
    B = fine_logits.shape[0]
    h_f, w_f = fine_logits.shape[-2:]
    flat = fine_logits.view(B, -1)
    idx = flat.argmax(dim=1)
    vy = (idx // w_f).float()
    ux = (idx % w_f).float()
    pred = anchor + torch.stack([ux, vy], dim=1)
    err = ((pred - displacement_xy.float()) ** 2).sum(dim=1).sqrt()
    return err.mean()


# ---------------------------------------------------------------------------
# Config-driven adapter (the one builder dispatches to via registry)
# ---------------------------------------------------------------------------

@register("two_stage_matching_loss")
def two_stage_matching_loss_config(output: dict,
                                   displacement_xy: torch.Tensor,
                                   *,
                                   config: dict,
                                   current_epoch: int) -> torch.Tensor:
    """Config-driven Stage-2 loss wrapper.

    Reads the yaml loss block, computes fine_weight from current_epoch +
    fine_warmup_epochs (linear ramp from 0), and dispatches to
    two_stage_matching_loss.

    Expected config keys (all optional, defaults match TM_Model_A):
        match_w4, match_w8                    coarse scale weights
        match_sigma4, match_sigma8            coarse Gaussian sigmas
        match_pos_weight                      coarse BCE pos_weight (or None)
        match_sigma_fine                      fine Gaussian sigma (pixels)
        match_w_fine                          target fine weight (post-warmup)
        match_fine_pos_weight                 fine BCE pos_weight (or None)
        fine_warmup_epochs                    linear ramp over this many epochs
    """
    coarse_kwargs = dict(
        scale_weights={4: float(config.get('match_w4', 1.0)),
                       8: float(config.get('match_w8', 1.0))},
        sigma_per_stride={4: float(config.get('match_sigma4', 1.0)),
                          8: float(config.get('match_sigma8', 1.0))},
        pos_weight=config.get('match_pos_weight', None),
    )
    target_fine_w = float(config.get('match_w_fine', 1.0))
    warmup = int(config.get('fine_warmup_epochs', 3))
    if warmup <= 0:
        fine_weight = target_fine_w
    else:
        fine_weight = target_fine_w * min(1.0, current_epoch / float(warmup))

    return two_stage_matching_loss(
        output, displacement_xy,
        coarse_kwargs=coarse_kwargs,
        fine_sigma=float(config.get('match_sigma_fine', 1.0)),
        fine_weight=fine_weight,
        fine_pos_weight=config.get('match_fine_pos_weight', None),
    )
