# -*- coding: utf-8 -*-
"""Shared training/inference interface, driven by YAML config.

ComponentBasedInterface implements the JackFramework ModelHandlerTemplate
generically: it reads a config dict for architecture / loss / optimizer /
inference choices, dispatches to registered components, and only requires
per-variant Python overrides when something genuinely can't be expressed
in YAML.

Conventions enforced here so all yaml-driven variants behave consistently:

- args.is_stage_2 selects between Stage 1 (alignment) and Stage 2 (matching)
  forward paths.
- rgb_encoder is always frozen.
- In Stage 2, sar_encoder is frozen unless config.training.stage_2.freeze_sar_encoder
  is set to False.
- trainable_tail layout matches TM_Model_A so checkpoints transfer.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.optim as optim

import JackFramework as jf

from .builder import build_model_from_yaml, load_config
from ._components.registry import get_component


class ComponentBasedInterface(jf.UserTemplate.ModelHandlerTemplate):
    """Generic yaml-driven interface. Subclass only when config can't express the variant."""

    ID_MODEL = 0
    ID_SAR, ID_RGB, ID_LABEL = 0, 1, 0
    ID_PRED = 0

    # Variants set this to the path of their config.yaml (relative to repo root
    # or absolute). Resolution happens in __init__.
    CONFIG_PATH: str | None = None

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self._args = args
        self._current_epoch = 0

        if self.CONFIG_PATH is None:
            raise RuntimeError(
                f"{type(self).__name__} did not set CONFIG_PATH. "
                f"Override the class attribute with the variant's config.yaml path."
            )
        self._config = load_config(self.CONFIG_PATH)
        self._is_stage_2 = bool(getattr(args, 'is_stage_2', False))

    # -----------------------------------------------------------------
    # Model construction & parameter freezing
    # -----------------------------------------------------------------

    def get_model(self) -> list:
        model = build_model_from_yaml(self.CONFIG_PATH, self._args)

        rgb_freeze = self._config['architecture']['rgb_encoder'].get('freeze', True)
        sar_freeze_stage2 = (
            self._config.get('training', {})
            .get('stage_2', {})
            .get('freeze_sar_encoder', True)
        )
        for name, param in model.named_parameters():
            if 'rgb_encoder' in name and rgb_freeze:
                param.requires_grad = False
            elif self._is_stage_2 and 'sar_encoder' in name and sar_freeze_stage2:
                param.requires_grad = False

        return [model]

    # -----------------------------------------------------------------
    # Optimizer & scheduler
    # -----------------------------------------------------------------

    def optimizer(self, model: list, lr: float) -> list:
        stage_key = 'stage_2' if self._is_stage_2 else 'stage_1'
        opt_cfg = self._config.get('training', {}).get(stage_key, {}).get('optimizer', {})
        opt_type = opt_cfg.get('type', 'adamw' if self._is_stage_2 else 'sgd')

        params = [p for p in model[self.ID_MODEL].parameters() if p.requires_grad]

        if opt_type.lower() == 'adamw':
            opt = torch.optim.AdamW(
                params, lr=self._args.lr,
                betas=tuple(opt_cfg.get('betas', (0.9, 0.999))),
                weight_decay=float(opt_cfg.get('weight_decay', 0.01)),
            )
        elif opt_type.lower() == 'sgd':
            opt = torch.optim.SGD(
                params, lr=self._args.lr,
                momentum=float(opt_cfg.get('momentum', 0.9)),
                weight_decay=float(opt_cfg.get('weight_decay', 1e-4)),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min',
            factor=float(opt_cfg.get('lr_factor', 0.8)),
            patience=int(opt_cfg.get('lr_patience', 5)),
        )
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        return sch.step(ave_loss)

    # -----------------------------------------------------------------
    # Forward / loss / accuracy — dispatched via config
    # -----------------------------------------------------------------

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        output = model(input_data[self.ID_SAR], input_data[self.ID_RGB])
        return [output]

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        output = output_data[self.ID_PRED]
        stage_key = 'stage_2' if self._is_stage_2 else 'stage_1'
        loss_cfg = self._config['training'][stage_key]['loss']
        loss_fn = get_component(loss_cfg['type'])

        if self._is_stage_2:
            disp_xy = label_data[self.ID_LABEL]
            return [loss_fn(
                output, disp_xy,
                config=loss_cfg,
                current_epoch=self._current_epoch,
            )]
        # Stage 1
        return [loss_fn(
            output['sar_feat']['sar16'],
            output['rbg_feat'][3],
            **{k: v for k, v in loss_cfg.items() if k != 'type'},
        )]

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        """Report mean pixel error (Stage 2) or feature cosine similarity (Stage 1).

        For Stage 2, the decoding mode is driven by config.inference.argmax_type:
          - 'hard'           : argmax over fine heatmap, anchor + integer offset
          - 'soft_subpixel'  : argmax + 2D quadratic peak fit for sub-pixel offset
        """
        if not self._is_stage_2:
            from ._components.alignment_loss import alignment_cos_accuracy
            output = output_data[self.ID_PRED]
            with torch.no_grad():
                return [alignment_cos_accuracy(
                    output['sar_feat']['sar16'],
                    output['rbg_feat'][3],
                )]

        from ._components.matching_loss import two_stage_matching_accuracy
        from ._components.sub_pixel import quadratic_subpixel_fit

        output = output_data[self.ID_PRED]
        disp_xy = label_data[self.ID_LABEL]
        argmax_type = self._config.get('inference', {}).get('argmax_type', 'hard')
        primary_stride = int(self._config.get('inference', {}).get('primary_stride', 4))

        with torch.no_grad():
            if argmax_type == 'hard':
                # Default: integer-pixel pred from fine head
                err_px = two_stage_matching_accuracy(
                    output, disp_xy, primary_stride=primary_stride,
                    scale_weights={
                        4: float(self._config['training']['stage_2']['loss'].get('match_w4', 1.0)),
                        8: float(self._config['training']['stage_2']['loss'].get('match_w8', 1.0)),
                    },
                )
                return [err_px]
            elif argmax_type == 'soft_subpixel':
                # Apply quadratic sub-pixel fit to the fine heatmap
                fine_logits = output['heatmap_fine']
                anchor = output['fine_anchor_pix'].float()
                peak_int, sub_off = quadratic_subpixel_fit(fine_logits)
                pred = anchor + peak_int.float() + sub_off
                err = ((pred - disp_xy.float()) ** 2).sum(dim=1).sqrt()
                return [err.mean()]
            else:
                raise ValueError(f"Unknown inference.argmax_type: {argmax_type!r}")

    # -----------------------------------------------------------------
    # Per-epoch hooks
    # -----------------------------------------------------------------

    def pretreatment(self, epoch: int, rank: object) -> None:
        self._current_epoch = int(epoch)

    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        pass

    # -----------------------------------------------------------------
    # Checkpoint I/O
    # -----------------------------------------------------------------

    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        """Load ckpt into model.trainable_tail with strict=False.

        Respects config.training.stage_1.load_pretrained_sar: if False, any
        sar_encoder.* keys in the ckpt are filtered out before loading. This
        is B's ablation: even if a checkpoint happens to be present in
        modelDir, the SAR encoder will stay at fresh init.
        """
        assert model_id == self.ID_MODEL
        load_pretrained_sar = bool(
            self._config.get('training', {})
            .get('stage_1', {})
            .get('load_pretrained_sar', True)
        )
        model = self._unwrap_ddp(model)
        state = dict(checkpoint['model_0'])

        if not load_pretrained_sar:
            sar_keys = [k for k in state if k.startswith('sar_encoder.')]
            for k in sar_keys:
                del state[k]
            jf.log.info(
                f"[{type(self).__name__}] load_pretrained_sar=False — "
                f"dropped {len(sar_keys)} sar_encoder.* keys from ckpt; "
                f"SAR encoder will train from fresh init."
            )

        missing, unexpected = model.trainable_tail.load_state_dict(state, strict=False)
        jf.log.info(
            f"[{type(self).__name__}] loaded trainable_tail "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return True

    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # Stage-1 ckpt's optimizer state should not carry into Stage-2 (different optimizer family).
        return not self._is_stage_2

    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        assert len(model_list) == len(opt_list)
        model_dict = {'epoch': epoch}
        for i, model in enumerate(model_list):
            unwrapped = self._unwrap_ddp(model)
            model_dict[f'model_{i}'] = unwrapped.trainable_tail.state_dict()
            model_dict[f'opt_{i}'] = opt_list[i].state_dict()
        return model_dict

    @staticmethod
    def _unwrap_ddp(model):
        return model.module if hasattr(model, "module") else model


def find_variant_config(variant_dir: str) -> str:
    """Helper for variant __init__: locate config.yaml inside the variant folder."""
    path = os.path.join(os.path.dirname(variant_dir), 'config.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No config.yaml found at {path}")
    return path
