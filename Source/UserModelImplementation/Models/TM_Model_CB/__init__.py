# -*- coding: utf-8 -*-
"""TM_Model_CB variant entry point — guided-upsampling ablation control.

TM_Model_C with the coarse joint-bilateral guided upsampling replaced by plain
bilinear interpolation (CascadeHeadBilinear). Answers Reviewer 1's comment 7
(TGRS-2026-06939): Sec. III asserts bilinear "would blur the boundaries that
matching relies on"; this arm measures it.

TM_Model_C 的对照臂：coarse 阶段的联合双边引导上采换成纯双线性
（CascadeHeadBilinear），用来回答审稿人 1 第 7 条。论文第三节断言双线性会
“糊掉匹配所依赖的边界”，这一臂把断言变成测量。

The feature source is unchanged (stride-16, 1024-d, both modalities), so the
upsampling kernel is the only difference — see TM_Model_CB/config.yaml for the
one-line diff against TM_Model_C, and the CascadeHeadBilinear docstring for
what is deliberately NOT ablated.

特征来源不变（双边都是 stride-16 1024-d），因此唯一的差别就是上采核。与
TM_Model_C 的一行 diff 见 TM_Model_CB/config.yaml；哪些部分刻意未被消融见
CascadeHeadBilinear 的 docstring。
"""
import os

from .._interface_base import ComponentBasedInterface


class TMModelCBInterface(ComponentBasedInterface):
    """Bilinear-upsampling control for the cascade. Same interface as C —
    only ``architecture.matching.type`` differs in the config."""
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
