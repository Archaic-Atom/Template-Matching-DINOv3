# -*- coding: utf-8 -*-
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

import JackFramework as jf
from .Nets import TMModelANet
from .Loss import alignment_loss
# import UserModelImplementation.user_define as user_def


class TMModelAInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""
    ID_MODEL = 0
    ID_SAR, ID_RGB, ID_LABEL = 0, 1, 0
    ID_PRED = 0

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    def get_model(self) -> list:
        model = TMModelANet(self.__args)

        # lock the dinov3
        for name, param in model.named_parameters():
            if "rgb_encoder" in name:
                param.requires_grad = False
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        opt = torch.optim.SGD(
            [{'params': model[self.ID_MODEL].parameters()}],
            lr=args.lr, momentum=0.9, weight_decay=0.0001,)
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.8, patience=5
        )
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        return sch.step(ave_loss)

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        output = model(input_data[self.ID_SAR], input_data[self.ID_RGB])
        return [output]

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        args = self.__args
        acc = None
        if args.pre_trained:
            output = output_data[self.ID_PRED]
            acc = alignment_loss(output['sar_feat']["sar16"], output['rbg_feat'][3])
        return [acc]

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        args = self.__args

        loss = None
        if args.pre_trained:
            output = output_data[self.ID_PRED]
            loss = alignment_loss(output['sar_feat']["sar16"], output['rbg_feat'][3])
        return [loss]

    # Optional
    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass

    # Optional
    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass

    # Optional
    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        assert model_id == self.ID_MODEL
        model = self._unwrap_ddp(model)
        model.trainable_tail.load_state_dict(checkpoint['model_0'], strict=False)
        jf.log.info("load the old model")
        return True

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return True

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        assert len(model_list) == len(opt_list)
        model_dict = {'epoch': epoch}
        for i, model in enumerate(model_list):
            model_name = f'model_{i}'
            opt_name = f'opt_{i}'
            model = self._unwrap_ddp(model)
            model_dict[model_name] = model.trainable_tail.state_dict()
            model_dict[opt_name] = opt_list[i].state_dict()
        return model_dict

    @staticmethod
    def _unwrap_ddp(model):
        return model.module if hasattr(model, "module") else model
