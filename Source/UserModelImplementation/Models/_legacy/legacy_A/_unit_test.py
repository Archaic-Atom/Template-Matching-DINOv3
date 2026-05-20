import os
import torch

try:
    from .Nets import SarConvNeXt16
    from .Nets import create_backbone, get_dino_layers_id
    from .Loss import alignment_loss
except ImportError:
    from Nets import SarConvNeXt16
    from Nets import create_backbone, get_dino_layers_id
    from Loss import alignment_loss


ROOT_PATH = '/data/raozhibo/Documents/Programs/Template-Matching-DINOv3/'
DINOv3_PATH = 'Source/Libs/dinov3/'
DINOv2_PATH = 'Source/Libs/dinov2/'
WEIGHTS_PATH = 'Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
WEIGHTS_DINOV2_PATH = 'Weights/depth_anything/model.pth'


if __name__ == '__main__':
    rgb_encoder = create_backbone(
        'dinov3', 'dinov3_vitl16', os.path.join(ROOT_PATH, DINOv3_PATH),
        os.path.join(ROOT_PATH, WEIGHTS_PATH))

    sar_encoder = SarConvNeXt16(
        model_name="convnext_tiny",
        pretrained=False,
        dino_dim=1024,
        in_chans=3,
    )

    rgb_encoder = rgb_encoder.cuda() if torch.cuda.is_available() else rgb_encoder
    sar_encoder = sar_encoder.cuda() if torch.cuda.is_available() else sar_encoder

    rgb_img = torch.rand(1, 3, 64, 64).cuda() \
        if torch.cuda.is_available() else torch.rand(1, 3, 64, 64)
    sar_img = torch.rand(1, 3, 64, 64).cuda() \
        if torch.cuda.is_available() else torch.rand(1, 3, 64, 64)

    print(rgb_encoder)
    rgb_output_features = rgb_encoder.get_intermediate_layers(
        rgb_img, n = get_dino_layers_id('dinov3_vitl16'), reshape = True,
        return_class_token = False, norm = False)
    print(f"Shape of the upsampled output features: {rgb_output_features[3].shape}")

    sar_output_features = sar_encoder(sar_img)
    print("sar16:", sar_output_features["sar16"].shape)  # Expected: (1, 1024, 4, 4)

    loss = alignment_loss(sar_output_features["sar16"], rgb_output_features[3])
    print(loss)
