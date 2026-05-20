# -*- coding: utf-8 -*-
import argparse
import JackFramework as jf
import UserModelImplementation.user_define as user_def

# model and dataloader
from UserModelImplementation import Models
from UserModelImplementation import Dataloaders


class UserInterface(jf.UserTemplate.NetWorkInferenceTemplate):
    """docstring for UserInterface"""

    def __init__(self) -> object:
        super().__init__()

    def inference(self, args: object) -> object:
        dataloader = Dataloaders.dataloaders_zoo(args, args.dataset)
        model = Models.model_zoo(args, args.modelName)
        return model, dataloader

    def user_parser(self, parser: object) -> object:
        parser.add_argument('--root_path', type=str, default=user_def.ROOT_PATH,
                            help='root path')
        parser.add_argument('--dinov3_path', type=str, default=user_def.DINOv3_PATH,
                            help='dinov3 path')
        parser.add_argument('--pre_trained_weights_path', type=str,
                            default=user_def.WEIGHTS_PATH, help='dinov3 weight path')
        parser.add_argument('--pre_trained', type=UserInterface.__str2bool,
                            default=user_def.PRE_TRAINED,
                            help='[deprecated, kept for backward compat] '
                                 'forward path is now selected by --is_stage_2 alone')

        # Stage 2 (template matching) options
        parser.add_argument('--is_stage_2', type=UserInterface.__str2bool, default=False,
                            help='enable Stage 2 (template matching) — uses frozen SAR '
                                 'encoder + DINOv3 with new ProjMatch/MatchingHead modules')
        parser.add_argument('--matching_dim', type=int, default=128,
                            help='Stage 2 matching-space feature dimension')
        parser.add_argument('--match_w4', type=float, default=1.0,
                            help='Stage 2 loss weight beta_4 for stride-4 head')
        parser.add_argument('--match_w8', type=float, default=0.5,
                            help='Stage 2 loss weight beta_8 for stride-8 head')
        parser.add_argument('--match_sigma4', type=float, default=1.0,
                            help='Stage 2 Gaussian target sigma at stride 4 (in stride units)')
        parser.add_argument('--match_sigma8', type=float, default=1.0,
                            help='Stage 2 Gaussian target sigma at stride 8 (in stride units)')
        parser.add_argument('--match_primary_stride', type=int, default=4,
                            help='primary stride used for coarse displacement decoding at eval')
        parser.add_argument('--match_pos_weight', type=float, default=None,
                            help='optional BCE pos_weight to address peak/non-peak imbalance')

        # Fine (stride-1) refinement options
        parser.add_argument('--fine_dim', type=int, default=32,
                            help='Stage 2 fine-feature dimension')
        parser.add_argument('--fine_margin', type=int, default=4,
                            help='Stage 2 fine search radius in pixels around the coarse peak')
        parser.add_argument('--sar_size', type=int, default=64,
                            help='SAR template size in pixels')
        parser.add_argument('--match_w_fine', type=float, default=1.0,
                            help='Stage 2 fine BCE loss weight (after warmup)')
        parser.add_argument('--match_sigma_fine', type=float, default=1.0,
                            help='Stage 2 fine Gaussian sigma in pixels')
        parser.add_argument('--match_fine_pos_weight', type=float, default=None,
                            help='optional BCE pos_weight for the fine stage')
        parser.add_argument('--fine_warmup_epochs', type=int, default=3,
                            help='ramp the fine-stage loss weight linearly from 0 over '
                                 'these many epochs (lets coarse converge first)')
        parser.add_argument('--shuffle_buffer', type=int, default=100000,
                            help='WebDataset sample shuffle buffer size (samples). '
                                 'Higher = better intra-shard mixing for contrastive '
                                 'training, but eats num_workers * shuffle_buffer * '
                                 '~35KB raw bytes of RAM. Default 100000 ≈ 3.5 GB/worker, '
                                 '~112 GB total under DDP-4 num_workers=8.')
        return parser

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')
