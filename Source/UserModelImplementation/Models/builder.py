# -*- coding: utf-8 -*-
"""YAML-driven model composition.

Parses a config dict (loaded from TM_Model_X/config.yaml) and instantiates
component modules via the registry, then wires them into a ComposedTM nn.Module.

The wired model's state_dict structure mirrors TM_Model_A's `trainable_tail`
layout exactly, so checkpoints from any variant load cleanly (strict=False)
into any other variant that shares the relevant sub-components:

    stage 1: trainable_tail = {sar_encoder}
    stage 2: trainable_tail = {sar_encoder, matching, fusion}
                              ^ matching's internal layout (proj_*, head_*, etc.)
                                is owned by the matching component; different
                                matching components (StandardMatchingHead vs
                                CascadeHead) will produce different sub-keys.

This way `sar_encoder.*` keys (the 8M-sample Stage-1 trained weights) are
guaranteed to transfer across all variants regardless of matching choice.
"""
from __future__ import annotations

import os
import re
from typing import Any

import torch
import torch.nn as nn

from ._components.registry import get_component


_VAR_PATTERN = re.compile(r'\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}')


# ---------------------------------------------------------------------------
# Config loading & arg resolution
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load a YAML config from disk. Lazy-imports yaml so unused builds skip the dep."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required for YAML-driven model configs. "
            "Install with `pip install pyyaml`."
        ) from exc
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _substitute_vars(s: str, framework_args: object) -> str:
    """Replace every ${var} in s with str(getattr(framework_args, var))."""
    def _sub(match: re.Match) -> str:
        attr = match.group(1)
        if not hasattr(framework_args, attr):
            raise KeyError(
                f"Config references ${{{attr}}} but framework_args has no '{attr}'. "
                f"Add the matching argparse arg in user_interface.py."
            )
        return str(getattr(framework_args, attr))
    return _VAR_PATTERN.sub(_sub, s)


def _resolve_args(args_cfg: dict | None, framework_args: object) -> dict:
    """Substitute ${attr} placeholders (anywhere within string values) from framework_args."""
    if not args_cfg:
        return {}
    out: dict[str, Any] = {}
    for k, v in args_cfg.items():
        if isinstance(v, str) and _VAR_PATTERN.search(v):
            # If the whole string is one ${var}, preserve the original attribute's type
            # (e.g., booleans, ints). Otherwise do string interpolation.
            m = _VAR_PATTERN.fullmatch(v)
            if m is not None:
                attr = m.group(1)
                if not hasattr(framework_args, attr):
                    raise KeyError(
                        f"Config references ${{{attr}}} but framework_args has no '{attr}'."
                    )
                out[k] = getattr(framework_args, attr)
            else:
                out[k] = _substitute_vars(v, framework_args)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Composed model
# ---------------------------------------------------------------------------

class ComposedTM(nn.Module):
    """Top-level TM model assembled from registered components.

    Forward dispatch:
        stage 1: returns {'sar_feat': ..., 'rbg_feat': ...} for alignment loss
        stage 2: delegates to self.matching, returning matching head's dict output

    The fusion_logits Parameter is owned at the top level (not inside the
    matching component) to keep state_dict layout aligned with TM_Model_A.
    """

    def __init__(self, components: dict, is_stage_2: bool,
                 freeze_sar_in_stage2: bool = True,
                 freeze_rgb: bool = True) -> None:
        super().__init__()
        self.is_stage_2 = bool(is_stage_2)
        # In stage 2, controls whether the SAR encoder runs under no_grad and
        # produces detached features. When False (B's ablation: no Stage 1),
        # gradients flow through sar_encoder and it trains jointly.
        self.freeze_sar_in_stage2 = bool(freeze_sar_in_stage2)
        # rgb_encoder (DINOv3) is virtually always frozen; exposed for symmetry.
        self.freeze_rgb = bool(freeze_rgb)

        self.rgb_encoder = components['rgb_encoder']
        self.sar_encoder = components['sar_encoder']

        if self.is_stage_2:
            self.matching = components['matching']
            self.fusion_logits = nn.Parameter(torch.zeros(2))
            # Stage-2 never builds the alignment projectors: the matcher eats
            # the trunk's C2/C3, and a parameter-bearing-but-unused module
            # would trip DDP's unused-parameter check.
            # Stage-2 不构建对齐投影头:匹配头只吃 trunk 的 C2/C3,挂一个有参数
            # 却不前向的模块还会踩 DDP unused-parameter。
            self.align_ip = None

            # Mirror TM_Model_A's trainable_tail layout for ckpt compatibility.
            self.trainable_tail = nn.ModuleDict({
                'sar_encoder': self.sar_encoder,
                'matching': self.matching,
                'fusion': nn.ParameterDict({'logits': self.fusion_logits}),
            })
        else:
            # Optional learned-IP projectors (W_s/W_d); inside trainable_tail
            # so they checkpoint/resume. At Stage-2 load their keys surface as
            # unexpected and are dropped by strict=False — projector discarded,
            # trunk transfers.
            # 可选的可学内积投影头(W_s/W_d);挂进 trainable_tail 才会存档/续训。
            # Stage-2 加载时这些 key 是 unexpected,被 strict=False 丢弃 ——
            # projector 用后即弃,迁移的是 trunk。
            self.align_ip = components.get('align_ip')
            tail = {'sar_encoder': self.sar_encoder}
            if self.align_ip is not None:
                tail['align_ip'] = self.align_ip
            self.trainable_tail = nn.ModuleDict(tail)

    def forward(self, sar_img: torch.Tensor, rgb_img: torch.Tensor) -> dict:
        if not self.is_stage_2:
            return self._stage1_forward(sar_img, rgb_img)
        return self._stage2_forward(sar_img, rgb_img)

    # -- Stage 1: cross-modal feature alignment -----------------------------

    def _stage1_forward(self, sar_img: torch.Tensor, rgb_img: torch.Tensor) -> dict:
        # rgb_encoder typically frozen even in stage 1; let requires_grad
        # control via interface freeze settings rather than no_grad here.
        rgb_feat = self.rgb_encoder(rgb_img)
        sar_feat = self.sar_encoder(sar_img)
        out = {'sar_feat': sar_feat, 'rbg_feat': rgb_feat}
        if self.align_ip is not None:
            # Learned-IP variant: project both sides; loss/accuracy switch to
            # this pair when the keys are present.
            # 可学内积变体:两侧各自投影;loss/accuracy 检测到这两个 key 即改用。
            s_pre, d_pre = self.align_ip(sar_feat['sar16'], rgb_feat[3])
            out['align_s_pre'] = s_pre
            out['align_d_pre'] = d_pre
        return out

    # -- Stage 2: matching --------------------------------------------------

    def _stage2_forward(self, sar_img: torch.Tensor, rgb_img: torch.Tensor) -> dict:
        # RGB encoder: always under no_grad in stage 2 (always frozen).
        if self.freeze_rgb:
            with torch.no_grad():
                rgb_feat = self.rgb_encoder(rgb_img)
                rgb_feat = [t.detach() for t in rgb_feat]
        else:
            rgb_feat = self.rgb_encoder(rgb_img)

        # SAR encoder: no_grad only when frozen. B's ablation needs grads through here.
        if self.freeze_sar_in_stage2:
            with torch.no_grad():
                sar_feat = self.sar_encoder(sar_img)
                sar_feat = {k: v.detach() for k, v in sar_feat.items()}
        else:
            sar_feat = self.sar_encoder(sar_img)

        return self.matching(
            sar_img=sar_img,
            rgb_img=rgb_img,
            sar_feat=sar_feat,
            rgb_feat=rgb_feat,
            fusion_logits=self.fusion_logits,
        )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_model_from_config(config: dict, framework_args: object) -> ComposedTM:
    """Construct a ComposedTM from a parsed config dict.

    Args:
        config: parsed YAML, with top-level keys `architecture`, `training`,
            `inference`. See TM_Model_A/config.yaml for the canonical schema.
        framework_args: argparse Namespace; used to resolve ${...} placeholders
            and to read is_stage_2 (forward path selector).

    Returns:
        ComposedTM instance, ready to wrap in DDP / load ckpt / train.
    """
    arch = config['architecture']
    is_stage_2 = bool(getattr(framework_args, 'is_stage_2', False))

    components: dict[str, nn.Module] = {}

    # --- rgb encoder ---
    rgb_cfg = arch['rgb_encoder']
    components['rgb_encoder'] = get_component(rgb_cfg['type'])(
        **_resolve_args(rgb_cfg.get('args'), framework_args)
    )

    # --- sar encoder ---
    sar_cfg = arch['sar_encoder']
    components['sar_encoder'] = get_component(sar_cfg['type'])(
        **_resolve_args(sar_cfg.get('args'), framework_args)
    )

    # --- learned-IP alignment head (stage 1 only, optional) ---
    if not is_stage_2 and 'alignment_ip' in arch:
        ip_cfg = arch['alignment_ip']
        components['align_ip'] = get_component(ip_cfg['type'])(
            **_resolve_args(ip_cfg.get('args'), framework_args)
        )

    # --- matching (stage 2 only) ---
    if is_stage_2:
        if 'matching' not in arch:
            raise KeyError(
                f"Config has is_stage_2=True but no 'architecture.matching' "
                f"section. Add it or set is_stage_2=False."
            )
        match_cfg = arch['matching']
        match_args = _resolve_args(match_cfg.get('args'), framework_args)

        # Auto-wire SAR channel widths from sar_encoder if the matching head
        # advertises sar_ch_c2 / sar_ch_c3 in its signature and the user
        # didn't already specify them.
        # 仅当 head 签名里有 sar_ch_c2 / sar_ch_c3 时才注入，避免 head
        # （如 StandardMatchingHeadJBU 不吃 SAR encoder 的 C2/C3）误收
        # 多余参数。
        sar_enc = components['sar_encoder']
        if hasattr(sar_enc, 'channels'):
            import inspect
            head_cls = get_component(match_cfg['type'])
            try:
                head_sig = inspect.signature(head_cls.__init__).parameters
            except (TypeError, ValueError):
                head_sig = {}
            chs = sar_enc.channels
            if 'sar_ch_c2' in head_sig:
                match_args.setdefault('sar_ch_c2', chs[0])
            if 'sar_ch_c3' in head_sig:
                match_args.setdefault('sar_ch_c3', chs[1])

        components['matching'] = get_component(match_cfg['type'])(**match_args)

    # Read freeze flags from config so ComposedTM can route gradients correctly.
    rgb_freeze = bool(arch['rgb_encoder'].get('freeze', True))
    sar_freeze_s2 = bool(
        config.get('training', {}).get('stage_2', {}).get('freeze_sar_encoder', True)
    )

    return ComposedTM(
        components,
        is_stage_2=is_stage_2,
        freeze_sar_in_stage2=sar_freeze_s2,
        freeze_rgb=rgb_freeze,
    )


def build_model_from_yaml(config_path: str, framework_args: object) -> ComposedTM:
    """Convenience: load YAML from disk then build."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    config = load_config(config_path)
    return build_model_from_config(config, framework_args)
