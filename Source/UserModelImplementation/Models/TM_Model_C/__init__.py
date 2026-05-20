# -*- coding: utf-8 -*-
"""TM_Model_C variant entry point.

Main proposed method: SAR encoder + frozen DINOv3 + multi-scale coarse heads +
3-level cascade fine head (CascadeHead). Phase 1 has the wiring in place;
Phase 2 will implement CascadeHead.forward and cascade_loss.
"""
import os

from .._interface_base import ComponentBasedInterface


class TMModelCInterface(ComponentBasedInterface):
    """Cascade-based main method. Phase 1: registration + plumbing.
    Phase 2: CascadeHead.forward + cascade_loss implementation."""
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
