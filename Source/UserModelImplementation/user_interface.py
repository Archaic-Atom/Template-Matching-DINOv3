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
                            help='use or not use lr scheduler')
        return parser

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')
