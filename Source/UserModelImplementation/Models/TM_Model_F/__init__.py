# -*- coding: utf-8 -*-
"""TM_Model_F variant entry point.

Learned-IP Stage-1 (W_s/W_d projectors + VICReg rank floor) + frozen-SAR
Stage-2 — the "strong Stage-1, protected" counterfactual for the F0 claim.
Architecture matches A except the Stage-1 objective and the Stage-2 freeze;
all logic inherited from ComponentBasedInterface.

可学内积 Stage-1 + Stage-2 冻结 encoder 的 F0 反事实变体。除 Stage-1 目标与
Stage-2 冻结外与 A 完全一致;无 Python 覆写。
"""
import os

from .._interface_base import ComponentBasedInterface


class TMModelFInterface(ComponentBasedInterface):
    """Learned-IP Stage-1 + frozen-SAR Stage-2 variant. Config-only diff from A."""
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
