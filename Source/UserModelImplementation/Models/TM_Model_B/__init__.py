# -*- coding: utf-8 -*-
"""TM_Model_B variant entry point.

Ablation: no Stage-1 alignment. Architecture identical to A; differs only
in the load_pretrained_sar flag in config.yaml. No Python overrides needed.
"""
import os

from .._interface_base import ComponentBasedInterface


class TMModelBInterface(ComponentBasedInterface):
    """No-Stage-1 ablation variant. All logic inherited from ComponentBasedInterface."""
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
