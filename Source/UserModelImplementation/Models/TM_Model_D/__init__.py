# -*- coding: utf-8 -*-
"""TM_Model_D variant entry point.

Ablation: soft argmax + sub-pixel quadratic fit on the fine heatmap. All
training logic identical to A; difference is in inference postprocess,
which is config-driven (no Python overrides needed).
"""
import os

from .._interface_base import ComponentBasedInterface


class TMModelDInterface(ComponentBasedInterface):
    """Sub-pixel ablation variant. Driven entirely by config.yaml."""
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
