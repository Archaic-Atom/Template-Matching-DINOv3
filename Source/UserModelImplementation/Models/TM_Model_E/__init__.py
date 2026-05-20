# -*- coding: utf-8 -*-
"""TM_Model_E variant entry point.

Full method: cascade fine head (same as TM_Model_C) + sub-pixel quadratic
refinement at inference (same trick as TM_Model_D). State_dict is byte-
compatible with TM_Model_C so E reuses C's trained Stage-2 checkpoint at
eval time — no separate training required.

TM_Model_E 入口：cascade 头 + 推理时亚像素精修。与 TM_Model_C 共用权重，
仅在评估阶段切换 argmax 模式，无需单独训练。
"""
import os

from .._interface_base import ComponentBasedInterface


class TMModelEInterface(ComponentBasedInterface):
    """Cascade + sub-pixel refinement. Eval-only on TM_Model_C's ckpt."""
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
