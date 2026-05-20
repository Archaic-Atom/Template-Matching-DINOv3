# -*- coding: utf-8 -*-
"""TM_Model_A — baseline (yaml-driven).

The original A implementation lived in Nets/_create_*.py, Loss/*.py, inference.py
in this folder. Those files have been moved to ../_legacy/legacy_A/ for
reference; A is now driven by config.yaml + ../_components/ + ../builder.py
+ ../_interface_base.py, identical pattern to B / C / D.

The state_dict layout is byte-identical to the legacy A's (verified by
Scripts/verify_state_dict_compat.py), so checkpoints saved by the legacy
training run (e.g., model_epoch_0.pth from PID 54182) load cleanly via
strict=False (and strict=True after the matching head is also trained).

Additional tarball backups on remote:
    /data/raozhibo/backups_claude/tm_model_a_legacy_backup_*.tar.gz
"""
import os

from .._interface_base import ComponentBasedInterface


class TMModelAInterface(ComponentBasedInterface):
    """Baseline variant. All training logic driven by config.yaml."""
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
