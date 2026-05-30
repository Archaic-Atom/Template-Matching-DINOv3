# -*- coding: utf-8 -*-
"""TM_Model_C variant entry point.

Main proposed method: SAR encoder + frozen DINOv3 + multi-scale coarse heads +
3-level cascade fine head (CascadeHeadJBU). CascadeHead/CascadeHeadJBU.forward
and cascade_loss are fully implemented in ../_components/.
"""
import os

from .._interface_base import ComponentBasedInterface


class TMModelCInterface(ComponentBasedInterface):
    """Cascade-based main method. Fully implemented (cascade head + cascade_loss);
    all training/inference logic driven by config.yaml."""
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
