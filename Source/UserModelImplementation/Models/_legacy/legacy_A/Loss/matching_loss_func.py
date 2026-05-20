# -*- coding: utf-8 -*-
"""Stage 2 matching loss and metrics.

Pipeline:
    predicted heatmap logits (per stride)  ──┐
                                             ├─►  BCE-with-logits
    Gaussian target at (Δx/s, Δy/s)        ──┘

Final displacement is decoded by argmax over the (optionally fused) heatmap
and rescaled by the stride.
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def gaussian_target(displacement_xy: torch.Tensor,
                    h_out: int, w_out: int,
                    sigma: float, stride: int) -> torch.Tensor:
    """Generate a 2D Gaussian heatmap target centered at (Δx/s, Δy/s).

    Args:
        displacement_xy: (B, 2) pixel displacement (Δx, Δy) in the original image.
        h_out, w_out:    target heatmap size at stride-s.
        sigma:           Gaussian standard deviation in stride-s units (e.g. 1.0).
        stride:          spatial stride of the heatmap.

    Returns:
        (B, 1, h_out, w_out) target in [0, 1] with peak == 1.0 at (Δx/s, Δy/s).
    """
    if displacement_xy.dim() != 2 or displacement_xy.shape[1] != 2:
        raise ValueError(f"displacement_xy must be (B, 2); got {tuple(displacement_xy.shape)}")

    device = displacement_xy.device
    dx = displacement_xy[:, 0].float() / float(stride)   # (B,)
    dy = displacement_xy[:, 1].float() / float(stride)

    ys = torch.arange(h_out, device=device, dtype=torch.float32)
    xs = torch.arange(w_out, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')        # (h, w)
    yy = yy.unsqueeze(0)                                  # (1, h, w)
    xx = xx.unsqueeze(0)

    dx = dx.view(-1, 1, 1)
    dy = dy.view(-1, 1, 1)
    dist2 = (xx - dx) ** 2 + (yy - dy) ** 2               # (B, h, w)
    target = torch.exp(-dist2 / (2.0 * sigma ** 2))
    return target.unsqueeze(1)                            # (B, 1, h, w)


def matching_loss_single(heatmap_logits: torch.Tensor,
                         displacement_xy: torch.Tensor,
                         stride: int,
                         sigma: float,
                         pos_weight: Optional[float] = None) -> torch.Tensor:
    """BCE-with-logits at a single scale (numerically stable; sigmoid implicit)."""
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
    """Multi-scale (coarse) matching loss (Paper Eq. 26): weighted sum of per-scale BCE losses.

    Args:
        heatmap_logits_per_stride: {stride -> (B, 1, h_s, w_s)} logits.
        displacement_xy:           (B, 2) pixel displacement.
        scale_weights:             {stride -> β_s}; defaults to uniform 1.0.
        sigma_per_stride:          {stride -> σ_s in stride units}; defaults to 1.0 each.
        pos_weight:                positive class weight to address peak/non-peak imbalance.

    Returns:
        Scalar weighted sum of per-scale BCE losses.
    """
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
    """Fine-stage BCE loss in the (2m+1)x(2m+1) cropped window (stride-1).

    The fine heatmap covers RGB pixels [anchor_x, anchor_x+win) × [anchor_y, anchor_y+win)
    where win = 2m+1. The target Gaussian is centered at the ground-truth pixel
    position **relative to the anchor**:
        Δx_in_window = gt_x - anchor_x
        Δy_in_window = gt_y - anchor_y

    Samples whose GT lies outside this window contribute a flat (zero) target
    everywhere — they're already penalized by the coarse loss and don't help
    train the fine head. We mask them out from the fine loss.
    """
    B, _, h_f, w_f = heatmap_fine_logits.shape
    rel = displacement_xy.float() - fine_anchor_pix.float()   # (B, 2)
    inside = (rel[:, 0] >= 0) & (rel[:, 0] <= w_f - 1) & \
             (rel[:, 1] >= 0) & (rel[:, 1] <= h_f - 1)
    if not inside.any():
        # Avoid empty loss: return a zero that still has a grad path.
        return heatmap_fine_logits.sum() * 0.0

    H_in = heatmap_fine_logits[inside]                        # (B', 1, h_f, w_f)
    rel_in = rel[inside]                                      # (B', 2)
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
    """Combined coarse-to-fine matching loss.

    Args:
        output:           dict from TMModelANet._matching_proc, containing
                          'heatmap_per_stride', 'heatmap_fine', 'fine_anchor_pix'.
        displacement_xy:  (B, 2) ground-truth pixel displacement.
        coarse_kwargs:    kwargs forwarded to matching_loss for the coarse stage
                          (scale_weights, sigma_per_stride, pos_weight).
        fine_sigma:       Gaussian σ for fine target (in pixels, since stride=1).
        fine_weight:      multiplier on the fine BCE loss (set 0 during warmup).
        fine_pos_weight:  optional pos_weight for fine BCE.
    """
    coarse_kwargs = coarse_kwargs or {}
    L_coarse = matching_loss(output['heatmap_per_stride'], displacement_xy, **coarse_kwargs)
    L_fine = fine_matching_loss(
        output['heatmap_fine'], displacement_xy, output['fine_anchor_pix'],
        sigma=fine_sigma, pos_weight=fine_pos_weight)
    return L_coarse + fine_weight * L_fine


def peak_displacement(heatmap_logits: torch.Tensor, stride: int) -> torch.Tensor:
    """Decode predicted displacement (Δx, Δy) in pixels by argmax + stride scaling.

    Args:
        heatmap_logits: (B, 1, h, w) raw scores (sigmoid not applied; argmax is invariant).
        stride: spatial stride of the heatmap.

    Returns:
        (B, 2) tensor with predicted (Δx, Δy) in pixels.
    """
    B = heatmap_logits.shape[0]
    h, w = heatmap_logits.shape[-2:]
    flat = heatmap_logits.view(B, -1)
    idx = flat.argmax(dim=1)                              # (B,)
    v = (idx // w).float() * float(stride)                # y
    u = (idx %  w).float() * float(stride)                # x
    return torch.stack([u, v], dim=1)                     # (B, 2)


def fuse_heatmaps(heatmap_logits_per_stride: Dict[int, torch.Tensor],
                  primary_stride: int,
                  weights: Optional[Dict[int, float]] = None) -> torch.Tensor:
    """Upsample non-primary heatmaps to the primary stride's spatial size and
    average them with optional weights (Paper Eq. 25)."""
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
    """Mean L2 error in pixels between coarse-only argmax-decoded displacement and ground truth.

    For multi-scale inputs the heatmaps are fused at the primary stride before argmax.
    Precision is bounded by the coarse stride (e.g. 4 px) — combine with the fine
    stage via two_stage_matching_accuracy for pixel-level precision.
    """
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
    """Mean L2 error using coarse + fine: final prediction = fine_anchor + fine_argmax.

    The fine head's argmax within its (2m+1)x(2m+1) window gives a per-pixel offset
    from the crop anchor; adding the anchor yields the pixel-level prediction.
    """
    fine_logits = output['heatmap_fine']                    # (B, 1, h_f, w_f)
    anchor = output['fine_anchor_pix'].float()              # (B, 2)
    B = fine_logits.shape[0]
    h_f, w_f = fine_logits.shape[-2:]
    flat = fine_logits.view(B, -1)
    idx = flat.argmax(dim=1)                                # (B,)
    vy = (idx // w_f).float()
    ux = (idx %  w_f).float()
    pred = anchor + torch.stack([ux, vy], dim=1)            # (B, 2) pixel-level
    err = ((pred - displacement_xy.float()) ** 2).sum(dim=1).sqrt()
    return err.mean()
