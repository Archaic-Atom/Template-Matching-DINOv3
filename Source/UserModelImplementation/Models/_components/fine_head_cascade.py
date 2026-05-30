# -*- coding: utf-8 -*-
"""CascadeHead — 3-level coarse-to-fine cascade with learned upsamplers.

Design (TM_Model_C):

    coarse (stride-4 + stride-8, same modules as A)
        |
        v  fused argmax (detached) gives anchor at stride-4
        |
    stride-2 level
        - LearnedUpsampler ×2 on sar_4 / rgb_4 (bilinear + 2 conv refine)
        - Windowed cross-correlation (search cropped to ± window cells around
          2 × coarse argmax) → heatmap H_2
        - argmax(H_2) (detached) gives anchor at stride-2
        |
        v
    stride-1 level
        - LearnedUpsampler ×2 on stride-2 features
        - RawImageStem on raw SAR / raw RGB (3-conv stride-1 stem) → high-freq features
        - Concat(upsampled, raw_stem) → 1x1 conv + L2-norm → matching features
        - Windowed cross-correlation around 2 × stride-2 argmax → heatmap H_1

Each level produces a Gaussian-targeted heatmap, supervised independently with
BCE. cascade_loss combines per-level losses with optional warmup ramps on the
finer levels.

state_dict prefixes (relative to ComposedTM.trainable_tail.matching.):
    proj_sar_4, proj_rgb_4, head_4              ← reused from A
    proj_sar_8, proj_rgb_8, head_8              ← reused from A (if use_stride_8_coarse)
    sar_upsample_4to2, rgb_upsample_4to2        ← cascade-specific
    head_2                                       ← cascade-specific
    sar_upsample_2to1, rgb_upsample_2to1        ← cascade-specific
    sar_raw_stem, rgb_raw_stem                  ← cascade-specific
    proj_sar_1, proj_rgb_1                      ← cascade-specific
    head_1                                       ← cascade-specific
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register
from .coarse_matching import ProjMatchSAR, ProjMatchRGB, ProjMatchJBU, MatchingHead


def _gn(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    g = num_groups if num_channels >= num_groups else 1
    return nn.GroupNorm(g, num_channels)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LearnedUpsampler(nn.Module):
    """Bilinear upsample + 2-conv refinement at the target resolution.

    Pure bilinear interpolation cannot generate new high-frequency signal;
    the refinement convs are where pixel-level information is *learned*
    (supervised end-to-end by the per-level matching loss). This is the
    standard "stereo cascade" upsample-then-refine pattern.
    """

    def __init__(self, channels: int, scale: int = 2) -> None:
        super().__init__()
        self.scale = int(scale)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _gn(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale,
                          mode='bilinear', align_corners=False)
        return self.refine(x)


class RawImageStem(nn.Module):
    """Tiny stride-1 CNN on a raw image (3 channels) → low-dim pixel features.

    Used only at the highest cascade level to provide a high-frequency residual
    signal (sharp edges / corners) that semantic features from DINO+SAR encoders
    don't carry through bilinear upsampling.
    """

    def __init__(self, out_dim: int = 16, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
            _gn(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_dim, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _windowed_search_crop(search: torch.Tensor,
                          center_xy_cells: torch.Tensor,
                          template_size_cells: int,
                          margin_cells: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop a per-sample search window centered on (cx, cy) in cell coords.

    The crop has size `template_size_cells + 2*margin_cells` so the resulting
    cross-correlation heatmap is exactly `(2*margin_cells + 1) ^ 2` cells.

    Uses torch.stack on per-sample slices so the autograd graph through
    `search` is preserved (the cascade upsamplers and 1×1 projections that
    produce `search` need gradients from the windowed cross-correlation loss).

    Args:
        search:             (B, C, H, W) search feature map at some stride.
        center_xy_cells:    (B, 2) integer cell coordinates of where to center.
        template_size_cells: side length of the template feature in cells.
        margin_cells:        ± window radius in cells.

    Returns:
        crops:     (B, C, crop, crop) where crop = template_size + 2*margin
        anchor_xy: (B, 2) integer top-left cell coords after clipping
    """
    if center_xy_cells.dim() != 2 or center_xy_cells.shape[1] != 2:
        raise ValueError(f"center_xy_cells must be (B, 2); got {tuple(center_xy_cells.shape)}")

    B, C, H, W = search.shape
    crop = template_size_cells + 2 * margin_cells
    if H < crop or W < crop:
        raise ValueError(
            f"search ({H}x{W}) smaller than required crop {crop}x{crop}; "
            f"reduce margin_cells or use a larger search region."
        )

    cx = center_xy_cells[:, 0].long()
    cy = center_xy_cells[:, 1].long()
    x0 = (cx - margin_cells).clamp(min=0, max=W - crop)
    y0 = (cy - margin_cells).clamp(min=0, max=H - crop)

    # Per-sample slice + stack — preserves grad_fn through `search`.
    crops_list = []
    for b in range(B):
        xs, ys = int(x0[b].item()), int(y0[b].item())
        crops_list.append(search[b, :, ys:ys + crop, xs:xs + crop])
    crops = torch.stack(crops_list, dim=0)
    anchor = torch.stack([x0, y0], dim=1)
    return crops, anchor


def _argmax_xy(heatmap: torch.Tensor) -> torch.Tensor:
    """Hard argmax of a (B, 1, H, W) heatmap → (B, 2) cell coords (x, y)."""
    B = heatmap.shape[0]
    h = heatmap.shape[-2]
    w = heatmap.shape[-1]
    idx = heatmap.view(B, -1).argmax(dim=1)
    return torch.stack([idx % w, idx // w], dim=1)


# ---------------------------------------------------------------------------
# CascadeHead
# ---------------------------------------------------------------------------

@register("CascadeHead")
class CascadeHead(nn.Module):
    """3-level cascade fine head: stride-4 + 8 (coarse) → stride-2 → stride-1.

    forward signature matches StandardMatchingHead so ComposedTM doesn't care
    which fine implementation is wired in:

        forward(sar_img, rgb_img, sar_feat, rgb_feat, fusion_logits) → dict

    Output dict (compatible with ComponentBasedInterface.accuracy):
        heatmap_per_stride: {4: H_4, 8: H_8}    coarse, BCE-supervised
        heatmap_cascade:    {2: H_2, 1: H_1}    cascade refinements
        heatmap_fine:       H_1                  alias for legacy accuracy path
        fine_anchor_pix:    anchor_1 in pixels   alias for legacy accuracy path
        cascade_anchor_2_cells: stride-2 cell anchor of H_2 (for cascade_loss)
        fusion_weights:     (2,) coarse fusion αs
    """

    def __init__(self,
                 sar_ch_c2: int = 64,
                 sar_ch_c3: int = 128,
                 dino_dim: int = 1024,
                 matching_dim: int = 128,
                 sar_size: int = 64,
                 raw_stem_dim: int = 16,
                 use_stride_8_coarse: bool = True,
                 cascade_window_2: int = 4,
                 cascade_window_1: int = 4) -> None:
        super().__init__()
        self.sar_size = int(sar_size)
        self.use_stride_8_coarse = bool(use_stride_8_coarse)
        self.cascade_window_2 = int(cascade_window_2)
        self.cascade_window_1 = int(cascade_window_1)

        # === Coarse stride-4 (verbatim from A's StandardMatchingHead) ===
        self.proj_sar_4 = ProjMatchSAR(in_ch=sar_ch_c2, out_dim=matching_dim)
        self.proj_rgb_4 = ProjMatchRGB(dino_dim=dino_dim, out_dim=matching_dim, target_stride=4)
        self.head_4 = MatchingHead(refine=True)

        # === Coarse stride-8 (optional, default on for parity with A) ===
        if self.use_stride_8_coarse:
            self.proj_sar_8 = ProjMatchSAR(in_ch=sar_ch_c3, out_dim=matching_dim)
            self.proj_rgb_8 = ProjMatchRGB(dino_dim=dino_dim, out_dim=matching_dim, target_stride=8)
            self.head_8 = MatchingHead(refine=True)

        # === Cascade level 1/2: learned ×2 upsample on stride-4 features ===
        self.sar_upsample_4to2 = LearnedUpsampler(channels=matching_dim, scale=2)
        self.rgb_upsample_4to2 = LearnedUpsampler(channels=matching_dim, scale=2)
        self.head_2 = MatchingHead(refine=True)

        # === Cascade level 1/1: learned ×2 upsample + raw-image stem concat ===
        self.sar_upsample_2to1 = LearnedUpsampler(channels=matching_dim, scale=2)
        self.rgb_upsample_2to1 = LearnedUpsampler(channels=matching_dim, scale=2)
        self.sar_raw_stem = RawImageStem(out_dim=raw_stem_dim)
        self.rgb_raw_stem = RawImageStem(out_dim=raw_stem_dim)
        # Concat(upsampled, raw_stem) → matching_dim, L2-normalized.
        self.proj_sar_1 = ProjMatchSAR(in_ch=matching_dim + raw_stem_dim, out_dim=matching_dim)
        self.proj_rgb_1 = ProjMatchSAR(in_ch=matching_dim + raw_stem_dim, out_dim=matching_dim)
        self.head_1 = MatchingHead(refine=True)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _coarse_argmax_cells_stride4(self,
                                     H_4: torch.Tensor,
                                     H_8: torch.Tensor | None,
                                     fusion_logits: torch.Tensor) -> torch.Tensor:
        """Fused argmax on the coarse heatmaps, returned in stride-4 cell coords."""
        if H_8 is None:
            H_fused = H_4
        else:
            alphas = F.softmax(fusion_logits, dim=0)
            H_8_up = F.interpolate(H_8, size=H_4.shape[-2:],
                                   mode='bilinear', align_corners=False)
            H_fused = alphas[0] * H_4 + alphas[1] * H_8_up
        return _argmax_xy(H_fused)  # (B, 2) cell coords (u, v) at stride-4

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self,
                sar_img: torch.Tensor,
                rgb_img: torch.Tensor,
                sar_feat: dict,
                rgb_feat: list,
                fusion_logits: torch.Tensor) -> dict:
        sar_c2 = sar_feat['C2']        # (B, 64,  16, 16)
        sar_c3 = sar_feat['C3']        # (B, 128, 8,  8)
        rgb_16 = rgb_feat[-1]          # (B, 1024, 8, 8)

        # ---- Level 1/4 ----
        sar_4 = self.proj_sar_4(sar_c2)            # (B, D, 16, 16) L2-norm
        rgb_4 = self.proj_rgb_4(rgb_16)            # (B, D, 32, 32) L2-norm
        H_4 = self.head_4(template=sar_4, search=rgb_4)   # (B, 1, 17, 17)

        # ---- Level 1/8 (optional, full-image coarse) ----
        if self.use_stride_8_coarse:
            sar_8 = self.proj_sar_8(sar_c3)            # (B, D, 8, 8)
            rgb_8 = self.proj_rgb_8(rgb_16)            # (B, D, 16, 16)
            H_8 = self.head_8(template=sar_8, search=rgb_8)   # (B, 1, 9, 9)
        else:
            H_8 = None

        # ---- Coarse argmax → stride-4 cell coords (detached) ----
        with torch.no_grad():
            uv_4 = self._coarse_argmax_cells_stride4(H_4, H_8, fusion_logits)

        # ---- Level 1/2 ----
        # Learned ×2 upsample on the projected stride-4 features, re-L2-normalize.
        sar_2_full = F.normalize(self.sar_upsample_4to2(sar_4), p=2, dim=1)  # (B, D, 32, 32)
        rgb_2_full = F.normalize(self.rgb_upsample_4to2(rgb_4), p=2, dim=1)  # (B, D, 64, 64)
        # Windowed crop on RGB around 2 × stride-4 argmax (stride-2 cell coords).
        sar_size_s2 = self.sar_size // 2  # 32 cells (template size at stride-2)
        center_s2 = (uv_4 * 2).long()
        rgb_2_crop, anchor_2 = _windowed_search_crop(
            rgb_2_full, center_s2, template_size_cells=sar_size_s2,
            margin_cells=self.cascade_window_2)
        H_2 = self.head_2(template=sar_2_full, search=rgb_2_crop)  # (B, 1, 2W+1, 2W+1)

        with torch.no_grad():
            uv_2_rel = _argmax_xy(H_2)               # cell coords within H_2
            uv_2 = anchor_2 + uv_2_rel               # absolute stride-2 cell coords

        # ---- Level 1/1 ----
        # Learned ×2 upsample to stride-1; concat raw-image stem; project; L2-norm.
        sar_1_feat = self.sar_upsample_2to1(sar_2_full)         # (B, D, 64, 64)
        rgb_1_feat = self.rgb_upsample_2to1(rgb_2_full)         # (B, D, 128, 128)
        sar_1_raw  = self.sar_raw_stem(sar_img)                 # (B, raw_d, 64, 64)
        rgb_1_raw  = self.rgb_raw_stem(rgb_img)                 # (B, raw_d, 128, 128)
        sar_1 = self.proj_sar_1(torch.cat([sar_1_feat, sar_1_raw], dim=1))  # (B, D, 64, 64)
        rgb_1 = self.proj_rgb_1(torch.cat([rgb_1_feat, rgb_1_raw], dim=1))  # (B, D, 128,128)
        # Windowed crop on RGB at stride-1 (= pixel coords) around 2 × stride-2 argmax.
        center_s1 = (uv_2 * 2).long()
        rgb_1_crop, anchor_1 = _windowed_search_crop(
            rgb_1, center_s1, template_size_cells=self.sar_size,
            margin_cells=self.cascade_window_1)
        H_1 = self.head_1(template=sar_1, search=rgb_1_crop)  # (B, 1, 2W+1, 2W+1)

        out: dict = {
            'heatmap_per_stride':       {4: H_4} if H_8 is None else {4: H_4, 8: H_8},
            'heatmap_cascade':          {2: H_2, 1: H_1},
            # Aliases so ComponentBasedInterface.accuracy works without override:
            'heatmap_fine':             H_1,
            'fine_anchor_pix':          anchor_1,
            'cascade_anchor_2_cells':   anchor_2,
            'fusion_weights':           (F.softmax(fusion_logits, dim=0)
                                         if H_8 is not None else None),
        }
        return out


# ---------------------------------------------------------------------------
# CascadeHeadJBU — C 变体的 FeatUp 版本
# ---------------------------------------------------------------------------

@register("CascadeHeadJBU")
class CascadeHeadJBU(nn.Module):
    """CascadeHead variant with FeatUp JBU upsamplers at the COARSE stage.

    Same 3-level cascade structure as ``CascadeHead`` (coarse stride-4 +
    stride-8 fusion → cascade stride-2 → cascade stride-1 with raw-image
    stem), but the coarse paths use ``ProjMatchJBU`` (FeatUp JBU + raw
    image guidance) instead of bilinear + 1×1 conv (RGB) / 1×1 conv on
    ResNet C2/C3 (SAR).

    与 ``CascadeHead`` 的 3 级结构一致，但 coarse stride-4/8 改为
    ``ProjMatchJBU``——双边都吃 stride-16 1024-d 特征 + 自己的原图作
    JBU 引导。Cascade 阶段（stride-4 → 2 → 1）的 ``LearnedUpsampler``
    保留不动：那里的输入已经是投到 matching_dim 的特征（不是 ViT 原
    raw features），JBU 的 image-guided range kernel 在这种紧凑特征
    上意义不大；而 stride-1 早已通过 ``RawImageStem`` 引入原图高频
    残差（功能上接近 image guidance）。

    State_dict layout differs from ``CascadeHead`` in the coarse heads
    (proj_*_4 / proj_*_8 now contain JBU stacks); cascade-specific
    modules keep the same names + shapes so loss code and accuracy code
    don't need changes.

    Args:
        dino_dim: channel count of stride-16 features (DINOv3 ViT-L/16 →
            1024; SAR encoder's ``sar16`` after ``ProjHead`` → 1024).
        matching_dim: shared output channel count of all matching heads.
            ``256`` recommended (vs A's old 128).
        sar_size: SAR template pixel size (64 px standard).
        raw_stem_dim: RawImageStem output channels (16 default).
        use_stride_8_coarse: enable the stride-8 coarse path; default true
            for parity with A's multi-scale coarse fusion.
        cascade_window_2: ± cells at stride-2 around 2 × coarse argmax.
        cascade_window_1: ± pixels at stride-1 around 2 × stride-2 argmax.
    """

    def __init__(self,
                 dino_dim: int = 1024,
                 matching_dim: int = 256,
                 sar_size: int = 64,
                 raw_stem_dim: int = 16,
                 use_stride_8_coarse: bool = True,
                 cascade_window_2: int = 4,
                 cascade_window_1: int = 4) -> None:
        super().__init__()
        self.sar_size = int(sar_size)
        self.use_stride_8_coarse = bool(use_stride_8_coarse)
        self.cascade_window_2 = int(cascade_window_2)
        self.cascade_window_1 = int(cascade_window_1)

        # === Coarse stride-4: JBU 2 levels (×4 upsample, stride-16 → 4) ===
        # 与 StandardMatchingHeadJBU 一致：双边都用 stride-16 1024-d 特征。
        self.proj_sar_4 = ProjMatchJBU(
            feat_dim=dino_dim, out_dim=matching_dim, levels=2)
        self.proj_rgb_4 = ProjMatchJBU(
            feat_dim=dino_dim, out_dim=matching_dim, levels=2)
        self.head_4 = MatchingHead(refine=True)

        # === Coarse stride-8: JBU 1 level (×2 upsample, stride-16 → 8) ===
        if self.use_stride_8_coarse:
            self.proj_sar_8 = ProjMatchJBU(
                feat_dim=dino_dim, out_dim=matching_dim, levels=1)
            self.proj_rgb_8 = ProjMatchJBU(
                feat_dim=dino_dim, out_dim=matching_dim, levels=1)
            self.head_8 = MatchingHead(refine=True)

        # === Cascade level 1/2: bilinear+conv on PROJECTED matching feats ===
        # Inputs here are already in matching_dim (256), not raw ViT features,
        # so JBU's image-guided range kernel offers limited benefit. Keep
        # LearnedUpsampler for consistency with CascadeHead.
        # 这里的输入是已经投到 matching_dim 的特征，不是 ViT 原始特征，
        # JBU 的 image-guidance 收益不大，保留 LearnedUpsampler。
        self.sar_upsample_4to2 = LearnedUpsampler(channels=matching_dim, scale=2)
        self.rgb_upsample_4to2 = LearnedUpsampler(channels=matching_dim, scale=2)
        self.head_2 = MatchingHead(refine=True)

        # === Cascade level 1/1: bilinear+conv + raw-image stem concat ===
        # RawImageStem already supplies sharp pixel-level signal directly
        # from the raw image; that's functionally similar to image guidance.
        # RawImageStem 已经从原图引入高频残差，相当于在 stride-1 加入
        # 图像引导信号，所以这里不需要额外的 JBU。
        self.sar_upsample_2to1 = LearnedUpsampler(channels=matching_dim, scale=2)
        self.rgb_upsample_2to1 = LearnedUpsampler(channels=matching_dim, scale=2)
        self.sar_raw_stem = RawImageStem(out_dim=raw_stem_dim)
        self.rgb_raw_stem = RawImageStem(out_dim=raw_stem_dim)
        # Concat(upsampled, raw_stem) → matching_dim, L2-normalized.
        self.proj_sar_1 = ProjMatchSAR(
            in_ch=matching_dim + raw_stem_dim, out_dim=matching_dim)
        self.proj_rgb_1 = ProjMatchSAR(
            in_ch=matching_dim + raw_stem_dim, out_dim=matching_dim)
        self.head_1 = MatchingHead(refine=True)

    # -----------------------------------------------------------------
    # Helpers (same as CascadeHead)
    # -----------------------------------------------------------------

    def _coarse_argmax_cells_stride4(self,
                                     H_4: torch.Tensor,
                                     H_8: torch.Tensor | None,
                                     fusion_logits: torch.Tensor) -> torch.Tensor:
        """Fused argmax on the coarse heatmaps, in stride-4 cell coords."""
        if H_8 is None:
            H_fused = H_4
        else:
            alphas = F.softmax(fusion_logits, dim=0)
            H_8_up = F.interpolate(H_8, size=H_4.shape[-2:],
                                   mode='bilinear', align_corners=False)
            H_fused = alphas[0] * H_4 + alphas[1] * H_8_up
        return _argmax_xy(H_fused)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self,
                sar_img: torch.Tensor,
                rgb_img: torch.Tensor,
                sar_feat: dict,
                rgb_feat: list,
                fusion_logits: torch.Tensor) -> dict:
        # Both coarse modalities consume their stride-16 1024-d features.
        # 粗匹配双边都吃 stride-16 1024-d 特征。
        rgb_16 = rgb_feat[-1]            # (B, 1024, 8, 8)
        sar_16 = sar_feat['sar16']        # (B, 1024, 4, 4) L2-normed

        # ---- Level 1/4 (JBU 2 levels) ----
        sar_4 = self.proj_sar_4(sar_16, sar_img)   # (B, D, 16, 16)
        rgb_4 = self.proj_rgb_4(rgb_16, rgb_img)   # (B, D, 32, 32)
        H_4 = self.head_4(template=sar_4, search=rgb_4)   # (B, 1, 17, 17)

        # ---- Level 1/8 (JBU 1 level) ----
        if self.use_stride_8_coarse:
            sar_8 = self.proj_sar_8(sar_16, sar_img)   # (B, D, 8, 8)
            rgb_8 = self.proj_rgb_8(rgb_16, rgb_img)   # (B, D, 16, 16)
            H_8 = self.head_8(template=sar_8, search=rgb_8)   # (B, 1, 9, 9)
        else:
            H_8 = None

        # ---- Coarse argmax → stride-4 cell coords (detached) ----
        with torch.no_grad():
            uv_4 = self._coarse_argmax_cells_stride4(H_4, H_8, fusion_logits)

        # ---- Level 1/2 (LearnedUpsampler on projected features) ----
        sar_2_full = F.normalize(self.sar_upsample_4to2(sar_4), p=2, dim=1)
        rgb_2_full = F.normalize(self.rgb_upsample_4to2(rgb_4), p=2, dim=1)
        sar_size_s2 = self.sar_size // 2
        center_s2 = (uv_4 * 2).long()
        rgb_2_crop, anchor_2 = _windowed_search_crop(
            rgb_2_full, center_s2, template_size_cells=sar_size_s2,
            margin_cells=self.cascade_window_2)
        H_2 = self.head_2(template=sar_2_full, search=rgb_2_crop)

        with torch.no_grad():
            uv_2_rel = _argmax_xy(H_2)
            uv_2 = anchor_2 + uv_2_rel

        # ---- Level 1/1 (LearnedUpsampler + RawImageStem concat) ----
        sar_1_feat = self.sar_upsample_2to1(sar_2_full)
        rgb_1_feat = self.rgb_upsample_2to1(rgb_2_full)
        sar_1_raw  = self.sar_raw_stem(sar_img)
        rgb_1_raw  = self.rgb_raw_stem(rgb_img)
        sar_1 = self.proj_sar_1(torch.cat([sar_1_feat, sar_1_raw], dim=1))
        rgb_1 = self.proj_rgb_1(torch.cat([rgb_1_feat, rgb_1_raw], dim=1))
        center_s1 = (uv_2 * 2).long()
        rgb_1_crop, anchor_1 = _windowed_search_crop(
            rgb_1, center_s1, template_size_cells=self.sar_size,
            margin_cells=self.cascade_window_1)
        H_1 = self.head_1(template=sar_1, search=rgb_1_crop)

        out: dict = {
            'heatmap_per_stride':       {4: H_4} if H_8 is None else {4: H_4, 8: H_8},
            'heatmap_cascade':          {2: H_2, 1: H_1},
            'heatmap_fine':             H_1,
            'fine_anchor_pix':          anchor_1,
            'cascade_anchor_2_cells':   anchor_2,
            'fusion_weights':           (F.softmax(fusion_logits, dim=0)
                                         if H_8 is not None else None),
        }
        return out
