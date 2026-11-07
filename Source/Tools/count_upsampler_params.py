# -*- coding: utf-8 -*-
"""Parameter ledger for the guided-upsampling ablation (JBU vs bilinear).

The response letter to Reviewer 1 (comment 7, TGRS-2026-06939) must state
whether the bilinear control is parameter-matched with the deployed JBU
cascade, and if not, by how much. This script produces that number instead of
letting anyone re-derive it by hand.

给审稿人 1 第 7 条的回复里必须写清：双线性对照臂是否与部署的 JBU 级联参数对齐，
若不对齐差多少。本脚本直接算出这个数，免得每次手推。

It counts three things, from the actual modules, at the configured geometry:
    1. one coarse projector (ProjMatchJBU vs ProjMatchBilinear), per level count
    2. the whole matching head (CascadeHeadJBU vs CascadeHeadBilinear)
    3. the delta, in absolute parameters and as a share of the head

按实际模块、按配置的几何，统计三件事：单个 coarse 投影器、整个 matching head、
以及两者的差值（绝对值与占比）。

FeatUp is an optional dependency of the JBU side only. Without it installed
(e.g. on the Mac) the script still reports the bilinear side and falls back to
FeatUp's published ``JBULearnedRange`` parameter formula for the JBU side,
printing which mode it used. On the server (``conda activate stereoA``) both
sides are counted from the real modules.

FeatUp 只有 JBU 那一侧需要。没装（比如在 Mac 上）时脚本仍会统计双线性侧，JBU 侧
退回按 FeatUp 官方 ``JBULearnedRange`` 的参数公式解析计算，并打印当前用的是哪种
模式。服务器上（``conda activate stereoA``）两侧都按真实模块统计。

Usage / 用法:
    python Source/Tools/count_upsampler_params.py
    python Source/Tools/count_upsampler_params.py --matching-dim 256 --sar-size 192
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_DIR = os.path.dirname(_THIS_DIR)
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from UserModelImplementation.Models._components.coarse_matching import (  # noqa: E402
    ProjMatchBilinear,
)
from UserModelImplementation.Models._components.fine_head_cascade import (  # noqa: E402
    CascadeHeadBilinear,
)


def count_params(module: nn.Module) -> int:
    """Total number of parameters in a module (trainable or not).

    统计模块的参数总数（不区分是否可训练）。

    Args:
        module: any ``nn.Module``.

    Returns:
        Sum of ``p.numel()`` over ``module.parameters()``.
    """
    return sum(p.numel() for p in module.parameters())


def jbu_kernel_params_analytic(guidance_dim: int = 3, key_dim: int = 32,
                               radius: int = 3) -> int:
    """Parameter count of one ``featup.upsamplers.JBULearnedRange`` level.

    单个 ``JBULearnedRange`` 层的参数量（解析公式）。

    Mirrors FeatUp's published module exactly:
        range_temp                                        1
        range_proj  Conv2d(g, k, 1) + Conv2d(k, k, 1)      g·k + k + k² + k
        fixup_proj  Conv2d(g + d², d², 1) + Conv2d(d², d², 1)
                                                          (g + d²)·d² + d² + d⁴ + d²
        sigma_spatial                                     1
    with ``d = 2·radius + 1``. The feature dim does NOT appear — the JBU kernel
    is predicted from the guidance image alone, which is why the ablation
    removes so few parameters.

    与 FeatUp 官方模块逐项对应，``d = 2·radius + 1``。注意特征维度不出现在公式里
    ——JBU 核只由引导图预测，这正是本消融只去掉很少参数的原因。

    Args:
        guidance_dim: guidance image channels (3).
        key_dim: JBU internal key dim (32).
        radius: JBU kernel radius (3 → 7×7 kernel).

    Returns:
        Parameter count of one level.
    """
    diameter = 2 * radius + 1
    d_sq = diameter * diameter
    range_proj = (guidance_dim * key_dim + key_dim) + (key_dim * key_dim + key_dim)
    fixup_proj = ((guidance_dim + d_sq) * d_sq + d_sq) + (d_sq * d_sq + d_sq)
    return 1 + range_proj + fixup_proj + 1


def try_build_jbu(feat_dim: int, out_dim: int, levels: int) -> Any:
    """Build a ``ProjMatchJBU`` if FeatUp is importable, else ``None``.

    能 import FeatUp 就构造 ``ProjMatchJBU``，否则返回 ``None``。

    Args:
        feat_dim: stride-16 feature channels.
        out_dim: matching-space channels.
        levels: number of ×2 steps.

    Returns:
        The module, or ``None`` when FeatUp is unavailable.
    """
    try:
        from UserModelImplementation.Models._components.coarse_matching import (
            ProjMatchJBU,
        )
        return ProjMatchJBU(feat_dim=feat_dim, out_dim=out_dim, levels=levels)
    except Exception as exc:  # ImportError, or a FeatUp CUDA-ext failure
        print(f"  (FeatUp unavailable: {type(exc).__name__}: {exc})")
        return None


def _fmt(n: int) -> str:
    """Thousands-separated integer. 千位分隔的整数。"""
    return f"{n:,}"


def main() -> None:
    """CLI: print the JBU-vs-bilinear parameter ledger."""
    parser = argparse.ArgumentParser(
        description="Parameter ledger for the JBU vs bilinear upsampling ablation.")
    parser.add_argument('--dino-dim', type=int, default=1024,
                        help="stride-16 feature channels (default 1024)")
    parser.add_argument('--matching-dim', type=int, default=256,
                        help="matching-space channels (default 256)")
    parser.add_argument('--sar-size', type=int, default=192,
                        help="SAR template pixel size (default 192, 256/192 geometry)")
    args = parser.parse_args()

    feat_dim, out_dim = args.dino_dim, args.matching_dim

    # --- 1. one coarse projector, per level count ---
    print("=" * 74)
    print("1. ONE COARSE PROJECTOR  (stride-16 -> stride-4 uses levels=2,")
    print("                          stride-16 -> stride-8 uses levels=1)")
    print("=" * 74)
    per_level_analytic = jbu_kernel_params_analytic()
    print(f"  JBULearnedRange, analytic per level : {_fmt(per_level_analytic)}")

    jbu_counts: dict[int, int] = {}
    bil_counts: dict[int, int] = {}
    measured = True
    for levels in (2, 1):
        bil = ProjMatchBilinear(feat_dim=feat_dim, out_dim=out_dim, levels=levels)
        bil_counts[levels] = count_params(bil)
        jbu = try_build_jbu(feat_dim, out_dim, levels)
        if jbu is None:
            measured = False
            jbu_counts[levels] = bil_counts[levels] + levels * per_level_analytic
        else:
            jbu_counts[levels] = count_params(jbu)
        print(f"  levels={levels}:  JBU {_fmt(jbu_counts[levels]):>12}"
              f"   bilinear {_fmt(bil_counts[levels]):>12}"
              f"   delta {_fmt(jbu_counts[levels] - bil_counts[levels]):>8}")
    print(f"  mode: {'measured from real modules' if measured else 'JBU side ANALYTIC (FeatUp not installed)'}")

    # --- 2. whole matching head ---
    print()
    print("=" * 74)
    print(f"2. WHOLE MATCHING HEAD  (dino_dim={feat_dim}, matching_dim={out_dim}, "
          f"sar_size={args.sar_size})")
    print("=" * 74)
    head_kwargs = dict(dino_dim=feat_dim, matching_dim=out_dim,
                       sar_size=args.sar_size)
    head_bil = CascadeHeadBilinear(**head_kwargs)
    n_bil = count_params(head_bil)
    # Four coarse projectors: sar_4 / rgb_4 at levels=2, sar_8 / rgb_8 at levels=1.
    # 四个 coarse 投影器：sar_4/rgb_4 用 levels=2，sar_8/rgb_8 用 levels=1。
    head_delta = 2 * (jbu_counts[2] - bil_counts[2]) + \
        2 * (jbu_counts[1] - bil_counts[1])
    n_jbu = n_bil + head_delta
    if measured:
        from UserModelImplementation.Models._components.fine_head_cascade import (
            CascadeHeadJBU,
        )
        n_jbu_measured = count_params(CascadeHeadJBU(**head_kwargs))
        if n_jbu_measured != n_jbu:
            raise AssertionError(
                f"head delta inconsistent: measured JBU head {n_jbu_measured} "
                f"!= bilinear {n_bil} + projector delta {head_delta}. The two "
                f"heads differ somewhere other than the coarse projectors — "
                f"the ablation is NOT single-factor. Investigate before running."
            )
        n_jbu = n_jbu_measured
    print(f"  CascadeHeadJBU       (TM_Model_C)  : {_fmt(n_jbu)}")
    print(f"  CascadeHeadBilinear  (TM_Model_CB) : {_fmt(n_bil)}")

    # --- 3. the delta ---
    print()
    print("=" * 74)
    print("3. DELTA  (the sentence the response letter needs)")
    print("=" * 74)
    pct = 100.0 * head_delta / n_jbu
    print(f"  The bilinear control is NOT parameter-matched: it is")
    print(f"  {_fmt(head_delta)} parameters smaller ({pct:.3f} % of the head),")
    print(f"  all of it the four dropped JBULearnedRange stacks")
    print(f"  (2 x levels=2 + 2 x levels=1, {_fmt(per_level_analytic)} per level).")
    print()
    print("  Padding the control with filler parameters would answer a different")
    print("  question (capacity, not guidance), so the counts are reported as-is.")
    print("  用填充参数把对照臂补齐会变成在回答另一个问题（容量而非引导），因此如实报告。")


if __name__ == '__main__':
    main()
