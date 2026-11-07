# -*- coding: utf-8 -*-
"""TM_Model_CW variant entry point.

TM_Model_C with the WEB-pretrained (LVD-1689M) DINOv3 provider in place of the
satellite-pretrained (SAT-493M) one — the provider-corpus ablation control for
Reviewer 1's comment 5. Config-only diff from C; no Python override.

把 TM_Model_C 的光学 provider 从卫星预训练(SAT-493M)换成 web 预训练
(LVD-1689M)的对照变体,回应审稿人 1 第 5 条。仅 config 差异,无 Python 覆写。
"""
import os

from .._interface_base import ComponentBasedInterface


class TMModelCWInterface(ComponentBasedInterface):
    """Web-pretrained-provider control for TM_Model_C. Config-only diff."""
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
