# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ._create_backbone import create_backbone, get_dino_layers_id
    from ._create_sar_encoder import SarResNet16GN
    from ._create_matching_head import (
        ProjMatchSAR, ProjMatchRGB, MatchingHead,
        FineFeatureHead, crop_search_at,
    )
except ImportError:
    from _create_backbone import create_backbone, get_dino_layers_id
    from _create_sar_encoder import SarResNet16GN
    from _create_matching_head import (
        ProjMatchSAR, ProjMatchRGB, MatchingHead,
        FineFeatureHead, crop_search_at,
    )


# SAR encoder channel widths at each stride (must match SarResNet16GN.channels).
_SAR_CH_C2 = 64      # stride 4
_SAR_CH_C3 = 128     # stride 8
_DINO_DIM = 1024     # dinov3_vitl16


class TMModelANet(nn.Module):
    def __init__(self, args: object) -> None:
        super().__init__()
        self._args = args
        self.is_stage_2 = bool(getattr(args, 'is_stage_2', False))

        self.rgb_encoder = create_backbone(
            'dinov3', 'dinov3_vitl16',
            os.path.join(args.root_path, args.dinov3_path),
            os.path.join(args.root_path, args.pre_trained_weights_path))
        self.sar_encoder = SarResNet16GN(in_chans=3, dino_dim=_DINO_DIM)

        if self.is_stage_2:
            match_dim = int(getattr(args, 'matching_dim', 128))
            fine_dim = int(getattr(args, 'fine_dim', 32))
            self.fine_margin = int(getattr(args, 'fine_margin', 4))  # ±4 px around anchor
            self.sar_size = int(getattr(args, 'sar_size', 64))

            # --- Coarse (stride-4 + stride-8) ---
            self.proj_sar_4 = ProjMatchSAR(in_ch=_SAR_CH_C2, out_dim=match_dim)
            self.proj_sar_8 = ProjMatchSAR(in_ch=_SAR_CH_C3, out_dim=match_dim)
            self.proj_rgb_4 = ProjMatchRGB(dino_dim=_DINO_DIM, out_dim=match_dim, target_stride=4)
            self.proj_rgb_8 = ProjMatchRGB(dino_dim=_DINO_DIM, out_dim=match_dim, target_stride=8)
            self.head_4 = MatchingHead(refine=True)
            self.head_8 = MatchingHead(refine=True)
            # Learnable inference-time fusion weights (softmax -> α_4, α_8).
            self.fusion_logits = nn.Parameter(torch.zeros(2))

            # --- Fine (stride-1, refined in a window around the coarse peak) ---
            self.fine_sar = FineFeatureHead(out_dim=fine_dim)
            self.fine_rgb = FineFeatureHead(out_dim=fine_dim)
            self.head_fine = MatchingHead(refine=True)

            matching_components = nn.ModuleDict(dict(
                proj_sar_4=self.proj_sar_4,
                proj_sar_8=self.proj_sar_8,
                proj_rgb_4=self.proj_rgb_4,
                proj_rgb_8=self.proj_rgb_8,
                head_4=self.head_4,
                head_8=self.head_8,
                fine_sar=self.fine_sar,
                fine_rgb=self.fine_rgb,
                head_fine=self.head_fine,
            ))
            # Keep sar_encoder in trainable_tail so its Stage-1 weights are
            # loaded/saved via the existing checkpoint helpers. load_model uses
            # strict=False, so matching_components init from scratch on first run.
            self.trainable_tail = nn.ModuleDict(dict(
                sar_encoder=self.sar_encoder,
                matching=matching_components,
                fusion=nn.ParameterDict({'logits': self.fusion_logits}),
            ))
        else:
            self.trainable_tail = nn.ModuleDict(dict(sar_encoder=self.sar_encoder))

    # ---- Stage 1: cross-modal feature alignment -----------------------
    def _feature_align_proc(
            self, sar_img: torch.Tensor, rgb_img: torch.Tensor) -> dict[str, torch.Tensor]:
        rbg_feat = self.rgb_encoder.get_intermediate_layers(
            rgb_img, n=get_dino_layers_id('dinov3_vitl16'), reshape=True,
            return_class_token=False, norm=False)
        sar_feat = self.sar_encoder(sar_img)
        return {'sar_feat': sar_feat, 'rbg_feat': rbg_feat}

    # ---- Stage 2: template matching via multi-scale cross-correlation -
    def _coarse_argmax_pixel(self, H4: torch.Tensor, H8: torch.Tensor) -> torch.Tensor:
        """Argmax over the fused coarse heatmap, returned as pixel anchors.

        Used at inference time to choose where the fine stage crops RGB.
        Non-differentiable; returns int64 (B, 2).
        """
        alphas = F.softmax(self.fusion_logits, dim=0)
        # Bring H8 up to H4's spatial size before fusion.
        H8_up = F.interpolate(H8, size=H4.shape[-2:],
                              mode='bilinear', align_corners=False)
        H_fused = alphas[0] * H4 + alphas[1] * H8_up      # (B, 1, h, w)
        B = H_fused.shape[0]
        h, w = H_fused.shape[-2:]
        idx = H_fused.view(B, -1).argmax(dim=1)
        v = idx // w                                       # (B,) int
        u = idx %  w
        # Each cell of the stride-4 heatmap == 4 pixels.
        return torch.stack([u, v], dim=1) * 4              # (B, 2) pixel anchor

    def _matching_proc(self, sar_img: torch.Tensor, rgb_img: torch.Tensor) -> dict:
        """Stage 2 forward (coarse-to-fine).

        Coarse stage runs stride-4 and stride-8 cross-correlation over the full
        search space. Fine stage crops RGB at the coarse argmax and runs a
        stride-1 cross-correlation in a (2m+1)x(2m+1) window for pixel-level
        precision. Argmax is detached so its non-differentiability doesn't
        affect training — the fine head learns from whatever local context the
        coarse stage selects; once coarse converges, fine signal stabilizes.

        Args:
            sar_img: (B, 3, sar_size, sar_size) SAR template patch.
            rgb_img: (B, 3, H_r, W_r) RGB search region.

        Returns dict with:
            heatmap_per_stride: {4: H4, 8: H8} coarse heatmaps (logits).
            heatmap_fine:       (B, 1, 2m+1, 2m+1) fine heatmap (logits).
            fine_anchor_pix:    (B, 2) top-left pixel of the fine crop after
                                clipping; subtract from GT to obtain the
                                in-crop ground-truth position used by the loss.
            fusion_weights:     (2,) α_4, α_8 after softmax (for inference fusion).
        """
        # --- Frozen backbones (no autograd) ---
        with torch.no_grad():
            sar_feat = self.sar_encoder(sar_img)
            rgb_layers = self.rgb_encoder.get_intermediate_layers(
                rgb_img, n=get_dino_layers_id('dinov3_vitl16'), reshape=True,
                return_class_token=False, norm=False)
            sar_c2 = sar_feat['C2'].detach()
            sar_c3 = sar_feat['C3'].detach()
            rgb_16 = rgb_layers[-1].detach()

        # --- Coarse stride-4 + stride-8 ---
        sar_4 = self.proj_sar_4(sar_c2)
        sar_8 = self.proj_sar_8(sar_c3)
        rgb_4 = self.proj_rgb_4(rgb_16)
        rgb_8 = self.proj_rgb_8(rgb_16)
        H_4 = self.head_4(template=sar_4, search=rgb_4)
        H_8 = self.head_8(template=sar_8, search=rgb_8)
        alphas = F.softmax(self.fusion_logits, dim=0)

        # --- Fine stage: crop RGB at coarse argmax (detached) ---
        with torch.no_grad():
            center_xy = self._coarse_argmax_pixel(H_4, H_8)
        rgb_crop, fine_anchor = crop_search_at(
            rgb_img, center_xy, sar_size=self.sar_size, margin=self.fine_margin)

        sar_f = self.fine_sar(sar_img)                     # (B, D_f, sar, sar)
        rgb_f = self.fine_rgb(rgb_crop)                    # (B, D_f, crop, crop)
        H_fine = self.head_fine(template=sar_f, search=rgb_f)  # (B, 1, 2m+1, 2m+1)

        return {
            'heatmap_per_stride': {4: H_4, 8: H_8},
            'heatmap_fine':       H_fine,
            'fine_anchor_pix':    fine_anchor,             # (B, 2) int top-left
            'fusion_weights':     alphas,
        }

    def forward(self, sar_img: torch.Tensor, rgb_img: torch.Tensor) -> dict:
        # Single dispatch: --is_stage_2 selects Stage 2 (matching); otherwise
        # default to Stage 1 (feature alignment). The legacy --pre_trained flag
        # is kept in args for backward compatibility but no longer gates forward.
        if self.is_stage_2:
            return self._matching_proc(sar_img, rgb_img)
        return self._feature_align_proc(sar_img, rgb_img)


def _demo() -> None:
    import argparse
    """Runs a small shape check demo."""
    ROOT_PATH = '/data/raozhibo/Documents/Programs/Template-Matching-DINOv3/'
    DINOv3_PATH = 'Source/Libs/dinov3/'
    WEIGHTS_PATH = 'Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'

    parser = argparse.ArgumentParser(description="TM_Model_A training arguments")
    parser.add_argument('--root_path', type=str, default=ROOT_PATH, help='root path')
    parser.add_argument('--dinov3_path', type=str, default=DINOv3_PATH, help='dinov3 path')
    parser.add_argument('--pre_trained_weights_path', type=str,
                        default=WEIGHTS_PATH, help='dinov3 weight path')
    parser.add_argument('--pre_trained', type=bool, default=True,
                        help='enable feature alignment forward path')
    parser.add_argument('--is_stage_2', type=bool, default=False,
                        help='enable Stage 2 (matching) forward path')
    parser.add_argument('--matching_dim', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TMModelANet(args).to(device)

    sar_img = torch.rand(1, 3, 64, 64, device=device)
    rgb_img = (torch.rand(1, 3, 128, 128, device=device)
               if args.is_stage_2 else torch.rand(1, 3, 64, 64, device=device))

    out = model(sar_img, rgb_img)
    print(out)


if __name__ == "__main__":
    _demo()
