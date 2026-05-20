# -*- coding: utf-8 -*-
"""Stage 2 matching components.

  ProjMatchSAR  : project a SAR encoder stride-s feature map to matching dim + L2 norm
  ProjMatchRGB  : upsample a DINOv3 stride-16 token grid to stride-s + project + L2 norm
  MatchingHead  : batched cross-correlation (grouped conv) + optional 3x3 conv refine

All matching is template-search style: the SAR feature is the template (smaller),
the optical feature is the search region (larger). Output heatmap spatial size is
search_size - template_size + 1 along each axis (valid correlation positions).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    g = num_groups if num_channels >= num_groups else 1
    return nn.GroupNorm(g, num_channels)


class ProjMatchSAR(nn.Module):
    """Project a SAR encoder feature map to matching dim, then L2-normalize per pixel.

    Input:  (B, in_ch, h, w) feature map at some stride.
    Output: (B, out_dim, h, w) L2-normalized along channels.
    """

    def __init__(self, in_ch: int, out_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_dim, kernel_size=1, bias=False),
            _gn(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=1)


class ProjMatchRGB(nn.Module):
    """Upsample a DINOv3 stride-16 token grid to a target finer stride, project, L2-normalize.

    Input:  (B, dino_dim, h_16, w_16)         at stride 16
    Output: (B, out_dim,  h_t,  w_t)          at target_stride, L2-normalized per pixel

    target_stride must divide 16 (so 4 and 8 are valid).
    """

    def __init__(self, dino_dim: int = 1024, out_dim: int = 128,
                 target_stride: int = 4) -> None:
        super().__init__()
        if target_stride <= 0 or 16 % target_stride != 0:
            raise ValueError(f"target_stride must divide 16; got {target_stride}")
        self.scale_factor = 16 // target_stride
        self.target_stride = target_stride

        self.proj = nn.Sequential(
            nn.Conv2d(dino_dim, out_dim, kernel_size=1, bias=False),
            _gn(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
        )

    def forward(self, x_16: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x_16, scale_factor=self.scale_factor,
                          mode='bilinear', align_corners=False)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)


class MatchingHead(nn.Module):
    """Cross-correlation between a SAR template and an optical search feature,
    followed by an optional 3x3 conv refinement.

    Both inputs are expected to be L2-normalized per pixel (so the correlation
    becomes a cosine similarity at each shift, in [-1, 1] before refinement).

    forward(template, search):
        template: (B, C, h_t, w_t)
        search:   (B, C, h_s, w_s)   with h_s >= h_t and w_s >= w_t
    returns:
        (B, 1, h_s - h_t + 1, w_s - w_t + 1) heatmap logits
    """

    def __init__(self, refine: bool = True, refine_hidden: int = 16) -> None:
        super().__init__()
        if refine:
            self.refine = nn.Sequential(
                nn.Conv2d(1, refine_hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(refine_hidden, 1, kernel_size=1),
            )
        else:
            self.refine = None

    def forward(self, template: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        H = self._cross_correlate(template, search)
        if self.refine is not None:
            H = self.refine(H)
        return H

    @staticmethod
    def _cross_correlate(template: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        """Batched cross-correlation via grouped 2D convolution.

        Each sample's template tensor acts as its own conv kernel applied to
        its search feature. Implemented as a single F.conv2d with groups=B.
        """
        B, C, h_t, w_t = template.shape
        Bs, Cs, h_s, w_s = search.shape
        if B != Bs or C != Cs:
            raise ValueError(
                f"shape mismatch: template={tuple(template.shape)}, "
                f"search={tuple(search.shape)}"
            )
        if h_s < h_t or w_s < w_t:
            raise ValueError(
                f"search must be >= template in HxW; "
                f"search={(h_s, w_s)}, template={(h_t, w_t)}"
            )
        # groups=B trick:
        #   weight : (B,   C, h_t, w_t)  ->  each sample's template is a separate filter
        #   input  : (1, B*C, h_s, w_s)  ->  search features stacked along channels
        #   output : (1,   B, h_o, w_o)  ->  one channel per sample
        weight = template
        inp = search.reshape(1, B * C, h_s, w_s)
        out = F.conv2d(inp, weight, groups=B)
        return out.transpose(0, 1).contiguous()  # (B, 1, h_o, w_o)


# ---------------------------------------------------------------------------
# Fine (stride-1) matching components for two-stage coarse-to-fine refinement.
# ---------------------------------------------------------------------------

class FineFeatureHead(nn.Module):
    """Small stride-1 CNN that maps a raw image (3 channels) to pixel-level
    matching features. Used independently for SAR (template) and RGB (search
    crop) in the fine stage.

    All layers keep spatial size (stride=1, padding=1), so the output has the
    same H, W as the input.
    """

    def __init__(self, out_dim: int = 32, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
            _gn(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            _gn(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_dim, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)


def crop_search_at(rgb: torch.Tensor,
                   center_xy_pix: torch.Tensor,
                   sar_size: int,
                   margin: int) -> torch.Tensor:
    """Crop a per-sample search region from RGB at given pixel anchors.

    The anchor (x, y) is the top-left corner where the SAR template would be
    placed. The crop spans (x - margin, y - margin) to (x + sar_size + margin,
    y + sar_size + margin) — i.e., size = sar_size + 2*margin. Each sample's
    anchor is clipped to keep the crop inside the RGB image.

    Args:
        rgb:           (B, 3, H, W) full RGB search image.
        center_xy_pix: (B, 2) integer anchors (x, y) in pixel space.
        sar_size:      side length of the SAR template (e.g., 64).
        margin:        extra pixels on each side around the SAR footprint.

    Returns:
        rgb_crop:  (B, 3, crop_size, crop_size) where crop_size = sar_size + 2*margin.
        anchor_xy: (B, 2) actual top-left pixel of the crop after clipping.
    """
    if center_xy_pix.dim() != 2 or center_xy_pix.shape[1] != 2:
        raise ValueError(f"center_xy_pix must be (B, 2); got {tuple(center_xy_pix.shape)}")

    B, _, H, W = rgb.shape
    crop = sar_size + 2 * margin
    if H < crop or W < crop:
        raise ValueError(f"RGB ({H}x{W}) is smaller than required crop {crop}x{crop}")

    cx = center_xy_pix[:, 0].long()
    cy = center_xy_pix[:, 1].long()
    # Top-left of crop (so the SAR anchor lies at margin offset inside it).
    x0 = (cx - margin).clamp(min=0, max=W - crop)
    y0 = (cy - margin).clamp(min=0, max=H - crop)

    crops = rgb.new_empty((B, rgb.shape[1], crop, crop))
    for b in range(B):
        xs, ys = int(x0[b].item()), int(y0[b].item())
        crops[b] = rgb[b, :, ys:ys + crop, xs:xs + crop]

    anchor = torch.stack([x0, y0], dim=1)  # (B, 2)
    return crops, anchor


if __name__ == "__main__":
    # Shape sanity check
    proj_sar = ProjMatchSAR(in_ch=64, out_dim=128)
    proj_rgb = ProjMatchRGB(dino_dim=1024, out_dim=128, target_stride=4)
    head = MatchingHead(refine=True)
    fine_sar = FineFeatureHead(out_dim=32)
    fine_rgb = FineFeatureHead(out_dim=32)

    sar_c2 = torch.randn(2, 64, 16, 16)        # 64x64 SAR -> stride-4 -> 16x16
    rgb_16 = torch.randn(2, 1024, 8, 8)        # 128x128 RGB -> stride-16 -> 8x8

    sar_t = proj_sar(sar_c2)                   # (2, 128, 16, 16)
    rgb_s = proj_rgb(rgb_16)                   # (2, 128, 32, 32)
    H = head(sar_t, rgb_s)                     # (2, 1, 17, 17)
    print(f"coarse: sar_t={tuple(sar_t.shape)}, rgb_s={tuple(rgb_s.shape)}, H={tuple(H.shape)}")

    sar_img = torch.randn(2, 3, 64, 64)
    rgb_img = torch.randn(2, 3, 128, 128)
    anchors = torch.tensor([[32, 32], [10, 50]])
    rgb_crop, real_anchor = crop_search_at(rgb_img, anchors, sar_size=64, margin=4)
    sar_fine = fine_sar(sar_img)               # (2, 32, 64, 64)
    rgb_fine = fine_rgb(rgb_crop)              # (2, 32, 72, 72)
    Hf = head(sar_fine, rgb_fine)              # (2, 1, 9, 9)
    print(f"fine: sar_fine={tuple(sar_fine.shape)}, rgb_fine={tuple(rgb_fine.shape)}, "
          f"Hf={tuple(Hf.shape)}, anchors={real_anchor.tolist()}")
