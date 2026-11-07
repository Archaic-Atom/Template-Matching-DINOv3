# -*- coding: utf-8 -*-
"""TM_Model_G variant entry point.

TM_Model_F with the Stage-2 SAR encoder TRAINABLE. G exists for one reason
only: to turn ``training.stage_2.freeze_sar_encoder`` into a single-line
toggle between two otherwise byte-identical configs, so that the revised 2x2
is a strict factorial. F and G together ARE the factorial control — one model
family, one architecture, one Stage-1 checkpoint, two binary factors:

    factor 1  initialization : shared Stage-1 seed  vs. empty modelDir
    factor 2  trainability   : TM_Model_G (trainable) vs. TM_Model_F (frozen)

The published 2x2 could not do this: its trainable row used TM_Model_A/B
(seeded from the ASYMMETRIC PatchNCE Stage-1, or random) while its frozen row
used TM_Model_F (seeded from the SYMMETRIC learned-inner-product Stage-1, or
not seeded), so "initialization" was crossed with "which Stage-1 recipe" and
the two rows did not even share a model definition (Reviewer 2, comment 5a).

G 的唯一存在理由:把 ``freeze_sar_encoder`` 变成一行可切换的开关,让修订后的
2x2 成为严格析因设计。F 与 G 合起来就是那个析因对照——同一模型族、同一架构、
同一份 Stage-1 权重,只有两个二值因子(初始化 / 是否冻结)。原论文的 2x2 做不到
这点:可训行用 A/B(非对称 PatchNCE Stage-1 或随机),冻结行用 F(对称可学内积
Stage-1 或不加载),初始化因子与 Stage-1 配方混淆(审稿人 2 意见 5a)。

All logic is inherited from ComponentBasedInterface; no Python override.
所有逻辑继承自 ComponentBasedInterface,无 Python 覆写。
"""
import os

from .._interface_base import ComponentBasedInterface


class TMModelGInterface(ComponentBasedInterface):
    """Learned-IP Stage-1 + TRAINABLE-SAR Stage-2. One-line config diff from F."""
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
