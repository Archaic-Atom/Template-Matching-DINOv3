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

        # LR scheduler — config-driven, default to CosineAnnealingLR over
        # maxEpochs because our typical budget is 6-10 epochs and the previous
        # ReduceLROnPlateau (patience=5) never triggered.
        # 学习率策略：默认走 maxEpochs 周期的余弦退火；以前的 ReduceLROnPlateau
        # 在 6-10 epoch 预算下几乎不会触发。
        sch_cfg = opt_cfg.get('scheduler', {})
        sch_type = str(sch_cfg.get('type', 'cosine')).lower()
        max_epochs = int(getattr(self._args, 'maxEpochs', 10))
        if sch_type == 'cosine':
            sch = optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=max(1, int(sch_cfg.get('T_max', max_epochs))),
                eta_min=float(sch_cfg.get('eta_min', 1e-5)),
            )
        elif sch_type == 'multistep':
            ms_default = [int(max_epochs * 0.6), int(max_epochs * 0.85)]
            sch = optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=list(sch_cfg.get('milestones', ms_default)),
                gamma=float(sch_cfg.get('gamma', 0.5)),
            )
        elif sch_type in ('reduce_on_plateau', 'plateau'):
            sch = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min',
                factor=float(sch_cfg.get('lr_factor', 0.8)),
                patience=int(sch_cfg.get('lr_patience', 5)),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sch_type}")
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # ReduceLROnPlateau is the only built-in scheduler that takes a loss
        # value as step() argument; all the others (Cosine / MultiStep / ...)
        # take no args.
        # 只有 ReduceLROnPlateau 的 step() 需要 loss 参数；其它 scheduler
        # 都是无参 step。
        if isinstance(sch, optim.lr_scheduler.ReduceLROnPlateau):
            return sch.step(float(ave_loss[sch_id][0]))
        return sch.step()

    # -----------------------------------------------------------------
    # Forward / loss / accuracy — dispatched via config
    # -----------------------------------------------------------------

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        output = model(input_data[self.ID_SAR], input_data[self.ID_RGB])
        # In test mode, attach a decoded primary-head prediction so the
        # dataloader's ``save_result`` can write a CSV row without needing
        # the yaml config. Uses the same decoder as ``accuracy()``'s a0.
        # test 模式下顺手把主 head 解码结果挂到 output 上，dataloader 的
        # save_result 直接读即可，省去再传 yaml config。
        if (self._is_stage_2 and
                getattr(self._args, 'mode', None) == 'test'):
            from ._components.decode import decode_primary
            argmax_type = self._config.get(
                'inference', {}).get('argmax_type', 'hard')
            with torch.no_grad():
                pred_xy, score = decode_primary(
                    output, argmax_type=argmax_type)
                output['_test_pred_xy'] = pred_xy
                output['_test_score'] = score
        return [output]

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        """Return ``[total_loss, *components]`` so framework displays l0..ln.

        Convention (matches JackFramework's ``loss[OPT_LOSS_ID].backward()``):
          - ``loss[0]`` is the total optimized loss (autograd-traced)
          - ``loss[1..n]`` are detached component scalars (logging only)

        损失返回结构：第 0 个是反传用的总损失，之后是各组成项的 detach 标量，
        框架会显示成 l1、l2、…。
        """
        output = output_data[self.ID_PRED]
        stage_key = 'stage_2' if self._is_stage_2 else 'stage_1'
        loss_cfg = self._config['training'][stage_key]['loss']
        loss_fn = get_component(loss_cfg['type'])

        if self._is_stage_2:
            disp_xy = label_data[self.ID_LABEL]
            total, components = loss_fn(
                output, disp_xy,
                config=loss_cfg,
                current_epoch=self._current_epoch,
            )
            return [total, *components.values()]

        # Stage 1. Learned-IP variants surface projected features under
        # align_s_pre/align_d_pre — the loss then operates on that pair.
        # Stage 1:可学内积变体把投影特征挂在 align_s_pre/align_d_pre 下,
        # 检测到即用投影对,否则走旧的 sar16/DINO 原始对。
        loss_kwargs = {k: v for k, v in loss_cfg.items() if k != 'type'}
        # Guard the silent-dispatch trap: the arch block and the loss type
        # must be paired, else a stale yaml trains the wrong objective quietly.
        # 防静默分发陷阱:alignment_ip 架构块与 loss type 必须配对出现。
        is_ip_output = 'align_s_pre' in output
        is_ip_loss = loss_cfg['type'] == 'alignment_ip_loss'
        if is_ip_output != is_ip_loss:
            raise RuntimeError(
                f"Stage-1 config mismatch: architecture.alignment_ip is "
                f"{'present' if is_ip_output else 'absent'} but "
                f"stage_1.loss.type={loss_cfg['type']!r}. The alignment_ip "
                f"block must be paired with type: alignment_ip_loss."
            )
        if is_ip_output:
            total, components = loss_fn(
                output['align_s_pre'], output['align_d_pre'], **loss_kwargs)
        else:
            total, components = loss_fn(
                output['sar_feat']['sar16'], output['rbg_feat'][3],
                **loss_kwargs)
        return [total, *components.values()]

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        """Report one accuracy metric per available output head.

        模型有几个输出头就报几个精度指标。框架显示成 a0、a1、…，
        a0 是 "headline"（与最终推理用的 argmax_type 一致），后续是各 level
        的 component 误差，便于诊断"哪一级 head 在收敛"。

        Stage 1 (alignment):
          a0 = per-position cosine similarity at GT positions

        Stage 2 with StandardMatchingHead (A/B/D):
          a0 = pixel err at fine head (sub-pix if argmax_type=soft_subpixel)
          a1 = pixel err at coarse fused (stride-4 + stride-8)
          a2 = pixel err at stride-4 alone

        Stage 2 with CascadeHead (C/E):
          a0 = pixel err at cascade stride-1 (sub-pix if soft_subpixel)
          a1 = pixel err at cascade stride-2
          a2 = pixel err at coarse fused
          a3 = pixel err at stride-4 alone
        """
        if not self._is_stage_2:
            from ._components.alignment_loss import alignment_cos_accuracy
            output = output_data[self.ID_PRED]
            with torch.no_grad():
                if 'align_s_pre' in output:
                    # Learned-IP variant:
                    #   a0 = cosine in the learned projected space;
                    #   a1 = cosine in the RAW trunk space (sar16 vs DINO) —
                    #        the trunk is the ONLY thing that transfers to
                    #        Stage-2 (projectors are discarded), so a healthy
                    #        a0 with a flat a1 means the heads absorbed the
                    #        alignment and the run is wasted. Comparable to
                    #        the weak baseline's raw-cosine trajectory.
                    # 可学内积变体:a0=投影空间余弦;a1=trunk 原始空间余弦 ——
                    # 迁移到 Stage-2 的只有 trunk,a0 高而 a1 平 = 对齐被投影头
                    # 吸走,这一炉就白训了。a1 可与弱基线的轨迹直接对比。
                    return [
                        alignment_cos_accuracy(
                            output['align_s_pre'], output['align_d_pre']),
                        alignment_cos_accuracy(
                            output['sar_feat']['sar16'], output['rbg_feat'][3]),
                    ]
                return [alignment_cos_accuracy(
                    output['sar_feat']['sar16'],
                    output['rbg_feat'][3],
                )]

        from ._components.matching_loss import (
            two_stage_matching_accuracy,
            matching_accuracy,
            peak_displacement,
        )
        from ._components.sub_pixel import quadratic_subpixel_fit

        output = output_data[self.ID_PRED]
        disp_xy = label_data[self.ID_LABEL]
        argmax_type = self._config.get('inference', {}).get('argmax_type', 'hard')
        primary_stride = int(self._config.get('inference', {}).get('primary_stride', 4))
        loss_cfg = self._config['training']['stage_2']['loss']
        scale_weights = {
            4: float(loss_cfg.get('match_w4', 1.0)),
            8: float(loss_cfg.get('match_w8', 1.0)),
        }

        with torch.no_grad():
            metrics = []

            has_cascade = 'heatmap_cascade' in output
            has_fine = 'heatmap_fine' in output

            # ----- a0: primary (matches inference) -----
            # 主指标：跟推理用的 argmax 一致。
            if has_cascade:
                # CascadeHead's stride-1 cascade output.
                primary_logits = output['heatmap_cascade'][1]
                primary_anchor = output['fine_anchor_pix'].float()
            elif has_fine:
                primary_logits = output['heatmap_fine']
                primary_anchor = output['fine_anchor_pix'].float()
            else:
                primary_logits = None

            if primary_logits is not None:
                if argmax_type == 'soft_subpixel':
                    peak_int, sub_off = quadratic_subpixel_fit(primary_logits)
                    pred = primary_anchor + peak_int.float() + sub_off
                else:
                    B = primary_logits.shape[0]
                    h_f, w_f = primary_logits.shape[-2:]
                    flat = primary_logits.view(B, -1)
                    idx = flat.argmax(dim=1)
                    vy = (idx // w_f).float()
                    ux = (idx % w_f).float()
                    pred = primary_anchor + torch.stack([ux, vy], dim=1)
                err_primary = ((pred - disp_xy.float()) ** 2).sum(dim=1).sqrt().mean()
                metrics.append(err_primary)

            # ----- a1: cascade stride-2 (C/E only) -----
            if has_cascade:
                H_2 = output['heatmap_cascade'][2]
                anchor_2 = output['cascade_anchor_2_cells'].float() * 2.0  # cells -> pixels
                B = H_2.shape[0]
                h2, w2 = H_2.shape[-2:]
                flat = H_2.view(B, -1)
                idx = flat.argmax(dim=1)
                vy = (idx // w2).float() * 2.0
                ux = (idx % w2).float() * 2.0
                pred_s2 = anchor_2 + torch.stack([ux, vy], dim=1)
                err_s2 = ((pred_s2 - disp_xy.float()) ** 2).sum(dim=1).sqrt().mean()
                metrics.append(err_s2)

            # ----- a1 (A/B/D) or a2 (C/E): coarse fused -----
            if 'heatmap_per_stride' in output:
                err_fused = matching_accuracy(
                    output['heatmap_per_stride'], disp_xy,
                    primary_stride=primary_stride, scale_weights=scale_weights,
                )
                metrics.append(err_fused)

                # Last: err at stride-4 alone (always the strongest coarse signal)
                hps = output['heatmap_per_stride']
                if 4 in hps:
                    pred_s4 = peak_displacement(hps[4], stride=4)
                    err_s4 = ((pred_s4 - disp_xy.float()) ** 2).sum(dim=1).sqrt().mean()
                    metrics.append(err_s4)

            return metrics

    # -----------------------------------------------------------------
    # Per-epoch hooks
    # -----------------------------------------------------------------

    def pretreatment(self, epoch: int, rank: object) -> None:
        self._current_epoch = int(epoch)

    def post_process(self, epoch: int, rank: object,
                     ave_tower_loss: list, ave_tower_acc: list) -> None:
        # In ``--mode test`` the framework calls user_post_process(0) once
        # after the test loop finishes; piggy-back on it to trigger the
        # dataloader's per-rank merge + final-CSV emission.
        # test 模式下框架在 loop 结束后调用一次 user_post_process(0)，
        # 借此触发 dataloader 的 per-rank 合并 + 最终 CSV 输出。
        if getattr(self._args, 'mode', None) == 'test':
            try:
                from UserModelImplementation.Dataloaders.tm_dataloader import (
                    finalize_active_test,
                )
                finalize_active_test()
            except ImportError:
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
            # Only strip the SAR encoder when this is a STAGE-1 checkpoint (it
            # contains ONLY sar_encoder.* keys): that is B's ablation — start
            # Stage 2 without inheriting a Stage-1-pretrained SAR encoder. When
            # RESTORING a STAGE-2 checkpoint (resume / test) the ckpt also holds
            # the matching head, and its sar_encoder.* are B's OWN trained
            # weights, which MUST be kept — stripping them here was the bug that
            # made B's checkpoints evaluate/resume on a random SAR encoder.
            # 仅当这是 STAGE-1 ckpt（只含 sar_encoder.* key）时才剥离 SAR encoder——
            # 那是 B 的消融：Stage 2 不继承 Stage-1 预训练的 SAR。恢复 STAGE-2 ckpt
            # （续训/测试）时 ckpt 含 matching 头，其 sar_encoder.* 是 B 自己训练好的
            # 权重，必须保留（之前误删导致 B 的 ckpt 在随机 SAR encoder 上评测）。
            is_stage2_ckpt = any(k.startswith('matching.') for k in state)
            if is_stage2_ckpt:
                jf.log.info(
                    f"[{type(self).__name__}] load_pretrained_sar=False but ckpt "
                    f"is a Stage-2 checkpoint (has matching head) — keeping its "
                    f"trained sar_encoder.* (restore/test path)."
                )
            else:
                sar_keys = [k for k in state if k.startswith('sar_encoder.')]
                for k in sar_keys:
                    del state[k]
                jf.log.info(
                    f"[{type(self).__name__}] load_pretrained_sar=False — dropped "
                    f"{len(sar_keys)} sar_encoder.* keys from Stage-1 ckpt; SAR "
                    f"encoder starts from fresh init."
                )

        missing, unexpected = model.trainable_tail.load_state_dict(state, strict=False)
        jf.log.info(
            f"[{type(self).__name__}] loaded trainable_tail "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return True

    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        """Restore optimizer state if compatible; gracefully skip otherwise.

        优化器状态恢复：兼容则加载，不兼容则跳过（冷启动）。

        Three cases this method covers:
          1. Same-stage resume (Stage 1 → Stage 1, Stage 2 → Stage 2): opt
             types match, load_state_dict succeeds.
          2. Stage 1 → Stage 2 bootstrap: Stage 1's SGD state is incompatible
             with Stage 2's AdamW. load_state_dict raises ValueError; we log
             and start optimizer fresh.
          3. Bootstrap from a strip_stage1_opt.py-cleaned ckpt: opt_0 key is
             missing entirely; we log and start optimizer fresh.

        Framework contract: returning True means "user handled this; do NOT
        fall back to ModelSaver.load_opt". We always return True so the
        framework's strict load (which raises KeyError on missing opt_0) is
        never reached.
        """
        key = f'opt_{model_id}'
        if key not in checkpoint:
            jf.log.info(
                f"[{type(self).__name__}] no '{key}' in ckpt — "
                f"starting optimizer from fresh init"
            )
            return True
        try:
            opt.load_state_dict(checkpoint[key])
            jf.log.info(
                f"[{type(self).__name__}] restored '{key}' from ckpt"
            )
        except (ValueError, KeyError) as e:
            jf.log.warning(
                f"[{type(self).__name__}] failed to load '{key}' "
                f"(likely Stage 1 SGD → Stage 2 AdamW mismatch): {e}. "
                f"Starting optimizer from fresh init."
            )
        return True

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
