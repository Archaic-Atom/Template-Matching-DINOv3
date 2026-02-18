# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn

try:
    from ._create_backbone import create_backbone, get_dino_layers_id
    from ._create_sar_encoder import SarResNet16GN
except ImportError:
    from _create_backbone import create_backbone, get_dino_layers_id
    from _create_sar_encoder import SarResNet16GN


class TMModelANet(nn.Module):
    def __init__(self, args: object) -> None:
        super().__init__()
        self._args = args
        self.rgb_encoder = create_backbone(
            'dinov3', 'dinov3_vitl16', os.path.join(args.root_path, args.dinov3_path),
            os.path.join(args.root_path, args.pre_trained_weights_path))
        self.sar_encoder = SarResNet16GN(in_chans=3, dino_dim=1024)

        self.trainable_tail = nn.ModuleDict(dict(
            sar_encoder=self.sar_encoder
        ))

    def _feature_align_proc(
            self, sar_img: torch.Tensor, rgb_img: torch.Tensor) -> dict[str, torch.Tensor]:
        rbg_feat = self.rgb_encoder.get_intermediate_layers(
            rgb_img, n = get_dino_layers_id('dinov3_vitl16'), reshape = True,
            return_class_token = False, norm = False)
        sar_feat = self.sar_encoder(sar_img)
        return {'sar_feat': sar_feat, 'rbg_feat': rbg_feat}

    def forward(self, sar_img: torch.Tensor, rgb_img: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._args.pre_trained:
            return self._feature_align_proc(sar_img, rgb_img)
        return {'sar_feat': sar_img, 'rbg_feat': rgb_img}


def _demo() -> None:
    import argparse
    """Runs a small shape check demo."""
    ROOT_PATH = '/data/raozhibo/Documents/Programs/Template-Matching-DINOv3/'
    DINOv3_PATH = 'Source/Libs/dinov3/'
    WEIGHTS_PATH = 'Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'

    parser = argparse.ArgumentParser(description="TM_Model_A training arguments")
    parser.add_argument('--root_path', type=str, default=ROOT_PATH,
                        help='root path')
    parser.add_argument('--dinov3_path', type=str, default=DINOv3_PATH,
                        help='dinov3 path')
    parser.add_argument('--pre_trained_weights_path', type=str,
                        default=WEIGHTS_PATH, help='dinov3 weight path')
    parser.add_argument('--pre_trained', type=bool,
                        default=True,
                        help='use or not use lr scheduler')
    args = parser.parse_args()

    model = TMModelANet(args).cuda()

    rgb_img = torch.rand(1, 3, 64, 64).cuda() \
        if torch.cuda.is_available() else torch.rand(1, 3, 64, 64)
    sar_img = torch.rand(1, 3, 64, 64).cuda() \
        if torch.cuda.is_available() else torch.rand(1, 3, 64, 64)

    out = model(sar_img, rgb_img)
    print(out)


if __name__ == "__main__":
    _demo()
