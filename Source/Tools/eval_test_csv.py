# -*- coding: utf-8 -*-
"""Evaluate a test-mode CSV emitted by ``TMDataloader.save_result``.

Computes end-to-end pixel error statistics and PCK-style hit rates at
1/2/3/4/5 pixel thresholds, optionally on a random subsample.

读取 test 模式输出的 CSV，计算端到端像素误差与 1/2/3/4/5 像素命中率
（PCK），可选随机子集采样。

Usage:
    # 全量评估
    python Source/Tools/eval_test_csv.py \\
        --csv DebugResult/test_<dataset>_<timestamp>.csv

    # 不放回随机抽 1000 样本（看 N=1000 估计和全量的差距，多跑几次看方差）
    python Source/Tools/eval_test_csv.py \\
        --csv DebugResult/test_<dataset>_<timestamp>.csv \\
        --n 1000

    # 复现某次具体抽样
    python Source/Tools/eval_test_csv.py \\
        --csv DebugResult/test_<dataset>_<timestamp>.csv \\
        --n 1000 --seed 42

Input CSV columns (header required):
    index, sar_name, rgb_name, gt_x, gt_y, pred_x, pred_y, score
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import Dict, Sequence

import pandas as pd

PCK_THRESHOLDS_PX = (1, 2, 3, 4, 5)


def _parse_bool(value: str) -> bool:
    """Argparse helper: parse human-friendly bool strings.

    Argparse 助手：解析 true/false/yes/no/1/0 等字符串。
    """
    low = str(value).strip().lower()
    if low in ('true', 't', 'yes', 'y', '1'):
        return True
    if low in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean string, got {value!r}")


def load_csv(path: str) -> pd.DataFrame:
    """Load and minimally validate the test CSV.

    加载并最小程度校验测试 CSV。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    required = {'gt_x', 'gt_y', 'pred_x', 'pred_y'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV {path} missing required columns: {sorted(missing)}")
    # Coerce numeric — catches malformed rows early.
    for col in ('gt_x', 'gt_y', 'pred_x', 'pred_y'):
        df[col] = pd.to_numeric(df[col], errors='raise')
    return df


def compute_metrics(df: pd.DataFrame,
                    thresholds: Sequence[int] = PCK_THRESHOLDS_PX,
                    ) -> Dict[str, float]:
    """Compute L2 error stats + PCK hit rates.

    Args:
        df: dataframe with gt_x/gt_y/pred_x/pred_y columns.
        thresholds: pixel thresholds for PCK reporting.

    Returns:
        dict with keys ``count``, ``mean_l2``, ``median_l2``, ``rmse``,
        ``max_l2``, and one ``pck_at_{k}px`` per threshold (in 0..1).

    计算 L2 误差统计 + PCK 命中率，返回字典。
    """
    n = len(df)
    if n == 0:
        return {'count': 0}

    dx = df['pred_x'] - df['gt_x']
    dy = df['pred_y'] - df['gt_y']
    dist = (dx ** 2 + dy ** 2) ** 0.5

    metrics: Dict[str, float] = {
        'count': int(n),
        'mean_l2': float(dist.mean()),
        'median_l2': float(dist.median()),
        'rmse': float(math.sqrt(float((dist ** 2).mean()))),
        'max_l2': float(dist.max()),
    }
    for k in thresholds:
        # PCK@k: fraction of samples with L2 < k px. We use strict < so
        # "1px error" means sub-pixel — adjust to <= if you want inclusive.
        # PCK@k: L2 严格小于 k 像素的占比（亚像素口径）。
        metrics[f'pck_at_{k}px'] = float((dist < float(k)).mean())
    return metrics


def subsample(df: pd.DataFrame, n: int, replace: bool,
              seed: int | None) -> pd.DataFrame:
    """Return ``n`` rows; ``n=0`` returns the full df unchanged.

    Args:
        df: full dataframe.
        n: target sample count; ``0`` = no sampling (return df as-is).
        replace: with-replacement vs without-replacement sampling.
        seed: optional RNG seed for reproducibility; ``None`` means truly
            random (different result each run, useful for variance probing).

    返回 n 行的子集；n=0 表示不抽样，原样返回。
    """
    if n <= 0:
        return df
    total = len(df)
    if not replace and n > total:
        raise ValueError(
            f"--n={n} > rows={total} but --replace=false; "
            "either use --replace true or lower --n.")
    rng = random.Random(seed) if seed is not None else random.Random()
    if replace:
        # Bootstrap-style with replacement.
        idx = [rng.randrange(total) for _ in range(n)]
    else:
        # Subsampling without replacement.
        idx = rng.sample(range(total), n)
    return df.iloc[idx].reset_index(drop=True)


def format_metrics(metrics: Dict[str, float]) -> str:
    """One-line + multi-line human readable summary.

    把指标格式化为可读的多行字符串。
    """
    if metrics.get('count', 0) == 0:
        return 'count=0 (empty input)'
    lines = [
        f"count       : {metrics['count']}",
        f"mean L2     : {metrics['mean_l2']:.3f} px",
        f"median L2   : {metrics['median_l2']:.3f} px",
        f"RMSE        : {metrics['rmse']:.3f} px",
        f"max L2      : {metrics['max_l2']:.3f} px",
    ]
    for k in PCK_THRESHOLDS_PX:
        key = f'pck_at_{k}px'
        if key in metrics:
            lines.append(f"PCK@{k}px     : {metrics[key] * 100:.2f}%")
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', type=str, required=True,
                        help='path to the test CSV emitted by '
                             'TMDataloader.save_result')
    parser.add_argument('--n', type=int, default=0,
                        help='subsample size; 0 = full evaluation (default)')
    parser.add_argument('--replace', type=_parse_bool, default=False,
                        help='with-replacement sampling? (default: false, '
                             'i.e. plain subsampling without replacement)')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed for reproducibility. Omit to use '
                             'true randomness (different result per run, '
                             'useful for variance probing).')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df_full = load_csv(args.csv)
    df = subsample(df_full, args.n, args.replace, args.seed)

    metrics = compute_metrics(df)
    print(f"CSV         : {args.csv}")
    print(f"total rows  : {len(df_full)}")
    if args.n > 0:
        mode = 'with replacement' if args.replace else 'without replacement'
        seed_str = 'random' if args.seed is None else f'seed={args.seed}'
        print(f"sampling    : n={args.n} ({mode}, {seed_str})")
    else:
        print("sampling    : full (no subsampling)")
    print('-' * 40)
    print(format_metrics(metrics))
    return 0


if __name__ == '__main__':
    sys.exit(main())
