# -*- coding: utf-8 -*-
import torch
import torch.optim as optim

import JackFramework as jf
from .Nets import TMModelANet
from .Loss import (
    alignment_loss,
    matching_loss,
    matching_accuracy,
    two_stage_matching_loss,
    two_stage_matching_accuracy,
)


def _is_stage_2(args) -> bool:
    return bool(getattr(args, 'is_stage_2', False))


class TMModelAInterface(jf.UserTemplate.ModelHandlerTemplate):
    """Stage 1 (feature alignment) and Stage 2 (template matching) share one model."""
    ID_MODEL = 0
    ID_SAR, ID_RGB, ID_LABEL = 0, 1, 0
    ID_PRED = 0

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        # Updated by pretreatment() each epoch; used to ramp the fine-stage loss
        # weight after the coarse heads have had a chance to converge.
        self.__current_epoch = 0

    def get_model(self) -> list:
        model = TMModelANet(self.__args)
        is_stage_2 = _is_stage_2(self.__args)

        # Always freeze DINOv3.
        # In Stage 2 also freeze the SAR encoder so its Stage-1 alignment is preserved.
        for name, param in model.named_parameters():
            if 'rgb_encoder' in name:
                param.requires_grad = False
            elif is_stage_2 and 'sar_encoder' in name:
                param.requires_grad = False
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        params = [p for p in model[self.ID_MODEL].parameters() if p.requires_grad]
        if _is_stage_2(args):
            # AdamW is friendlier for the small matching head than SGD.
            opt = torch.optim.AdamW(params, lr=args.lr,
                                    betas=(0.9, 0.999), weight_decay=0.01)
        else:
            opt = torch.optim.SGD(params, lr=args.lr,
                                  momentum=0.9, weight_decay=0.0001)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8, patience=5)
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        return sch.step(ave_loss)

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        output = model(input_data[self.ID_SAR], input_data[self.ID_RGB])
        return [output]

    # -----------------------------------------------------------------
    # Loss / accuracy: dispatch per stage.
    # -----------------------------------------------------------------
    def _coarse_kwargs(self) -> dict:
        a = self.__args
        return dict(
            scale_weights={4: float(getattr(a, 'match_w4', 1.0)),
                           8: float(getattr(a, 'match_w8', 1.0))},
            sigma_per_stride={4: float(getattr(a, 'match_sigma4', 1.0)),
                              8: float(getattr(a, 'match_sigma8', 1.0))},
            pos_weight=getattr(a, 'match_pos_weight', None),
        )

    def _fine_weight(self) -> float:
        """Ramp fine loss from 0 to args.match_w_fine over args.fine_warmup_epochs."""
        a = self.__args
        target = float(getattr(a, 'match_w_fine', 1.0))
        warmup = int(getattr(a, 'fine_warmup_epochs', 3))
        if warmup <= 0:
            return target
        return target * min(1.0, self.__current_epoch / float(warmup))

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        args = self.__args
        output = output_data[self.ID_PRED]
        if _is_stage_2(args):
            disp_xy = label_data[self.ID_LABEL]                # (B, 2)
            return [two_stage_matching_loss(
                output, disp_xy,
                coarse_kwargs=self._coarse_kwargs(),
                fine_sigma=float(getattr(args, 'match_sigma_fine', 1.0)),
                fine_weight=self._fine_weight(),
                fine_pos_weight=getattr(args, 'match_fine_pos_weight', None),
            )]
        # Default: Stage 1 alignment loss.
        return [alignment_loss(output['sar_feat']['sar16'], output['rbg_feat'][3])]

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        args = self.__args
        output = output_data[self.ID_PRED]
        if _is_stage_2(args):
            disp_xy = label_data[self.ID_LABEL]
            with torch.no_grad():
                # Report pixel-level error after the fine stage.
                err_px = two_stage_matching_accuracy(
                    output, disp_xy,
                    primary_stride=int(getattr(args, 'match_primary_stride', 4)),
                    scale_weights={4: float(getattr(args, 'match_w4', 1.0)),
                                   8: float(getattr(args, 'match_w8', 1.0))})
            return [err_px]
        return [alignment_loss(output['sar_feat']['sar16'], output['rbg_feat'][3])]

    # Optional
    def pretreatment(self, epoch: int, rank: object) -> None:
        self.__current_epoch = int(epoch)

    # Optional
    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        pass

    # Optional
    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        assert model_id == self.ID_MODEL
        model = self._unwrap_ddp(model)
        # strict=False so that loading a Stage-1 checkpoint into a Stage-2 model
        # silently leaves matching components at their fresh-init values.
        missing, unexpected = model.trainable_tail.load_state_dict(
            checkpoint['model_0'], strict=False)
        if _is_stage_2(self.__args):
            jf.log.info(f"[stage-2] loaded checkpoint into trainable_tail "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            jf.log.info("load the old model")
        return True

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # When transitioning Stage 1 -> Stage 2 we do NOT want to inherit the
        # Stage 1 SGD state: Stage 2 uses AdamW and the new params are random.
        return not _is_stage_2(self.__args)

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
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
