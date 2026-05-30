# -*- coding: utf-8 -*-
"""Aggregate Stage-1 separation metrics from per-sample CSV.

聚合 stage1_separation_2k.py 输出的 per-sample CSV，给出 Stage-1 在自身
PatchNCE 训练目标上的可分离性指标。

Reports:
- cos_gt vs cos_max_far / cos_mean_far distributions
  (per-sample positive vs hardest / average far-negative cos)
- Per-sample d-prime: (cos_gt - cos_mean_far) / cos_std_far
- ROC AUC across samples: cos_gt vs cos_max_far / cos_mean_far
- rank_gt histogram: how often GT cell sits among top-K of the heatmap

报告内容：cos_gt vs cos_max_far / cos_mean_far 的分布；per-sample
d-prime；跨样本 ROC AUC；rank_gt 直方图。

Usage:
    python Source/Tools/analyze_stage1_separation.py \\
        --csv DebugResult/stage1_2k_sep_epoch3.csv
"""
import argparse

import numpy as np
import pandas as pd


def auc_mann_whitney(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC AUC via Mann-Whitney U on pos vs neg sample scores.

    用 Mann-Whitney U 直接算 ROC AUC。

    Args:
        pos: positive-class scores.
            正类得分。
        neg: negative-class scores.
            负类得分。

    Returns:
        AUC in [0, 1]; 0.5 is random, 1.0 is perfect.
        AUC ∈ [0, 1]；0.5 随机，1.0 完美。
    """
    # P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)
    # 用 rank-based 直接算，n_pos * n_neg 分母。
    all_scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), dtype=bool),
                             np.zeros(len(neg), dtype=bool)])
    order = np.argsort(all_scores, kind='stable')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    # Average ranks across ties to match Mann-Whitney convention.
    # 处理 ties，给并列分数赋平均 rank。
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    n_pos = len(pos)
    n_neg = len(neg)
    sum_pos_ranks = ranks[labels].sum()
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def main() -> None:
    """CLI entry — print aggregated Stage-1 separation report.

    命令行入口：打印 Stage-1 分离度聚合报告。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--csv', required=True,
                        help='per-sample CSV from stage1_separation_2k.py')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f'Loaded {len(df)} rows from {args.csv}')

    in_bound = df['gt_in_bound'] == 1
    n_oob = int((~in_bound).sum())
    df = df[in_bound].copy()
    n = len(df)
    print(f'Samples with GT in bound: {n}  ({n_oob} out-of-bound dropped)')
    print(f'Feat grid: H mean={df.feat_h.mean():.1f}, '
          f'W mean={df.feat_w.mean():.1f}; '
          f'n_far_cells mean={df.n_far_cells.mean():.0f}')

    cos_gt = df['cos_gt'].astype(float).values
    cos_max = df['cos_max'].astype(float).values
    cos_max_far = df['cos_max_far'].astype(float).values
    cos_mean_far = df['cos_mean_far'].astype(float).values
    cos_std_far = df['cos_std_far'].astype(float).values

    def _stats(name: str, x: np.ndarray) -> None:
        print(f'  {name:<14} mean={x.mean():+.4f}  median={np.median(x):+.4f}'
              f'  std={x.std():.4f}  min={x.min():+.4f}  '
              f'max={x.max():+.4f}')

    print('\n========= cos distributions (across samples) =========')
    _stats('cos_gt',       cos_gt)
    _stats('cos_max',      cos_max)
    _stats('cos_max_far',  cos_max_far)
    _stats('cos_mean_far', cos_mean_far)
    _stats('cos_std_far',  cos_std_far)

    print('\n========= sample-level separation =========')
    # Per-sample d-prime: positive's cos vs the far-pool's mean / std.
    # 单样本 d-prime：cos_gt 相对远阴池的 mean / std 的标准化差。
    valid_std = cos_std_far > 1e-6
    dprime = (cos_gt[valid_std] - cos_mean_far[valid_std]) / \
        cos_std_far[valid_std]
    m_dp = float(dprime.mean())
    md_dp = float(np.median(dprime))
    print(f'per-sample d-prime = (cos_gt - mean_far) / std_far  '
          f'(n={len(dprime)})')
    print(f'  mean={m_dp:+.3f}  median={md_dp:+.3f}  '
          f'std={dprime.std():.3f}')
    print(f'  fraction with d-prime > 0:  {100 * (dprime > 0).mean():5.1f}%  '
          '(GT above the far-pool mean)')
    print(f'  fraction with d-prime > 1:  {100 * (dprime > 1).mean():5.1f}%')
    print(f'  fraction with d-prime > 2:  {100 * (dprime > 2).mean():5.1f}%')
    print(f'  fraction with d-prime > 3:  {100 * (dprime > 3).mean():5.1f}%')

    # Margin against the hardest far negative (per sample).
    # 单样本 margin = cos_gt - cos_max_far（GT 相对最强远阴的差）。
    margin_hard = cos_gt - cos_max_far
    print(f'\nper-sample margin = cos_gt - cos_max_far')
    print(f'  mean={margin_hard.mean():+.4f}  '
          f'median={np.median(margin_hard):+.4f}  '
          f'std={margin_hard.std():.4f}')
    print(f'  fraction > 0 (cos_gt beats hardest far neg): '
          f'{100 * (margin_hard > 0).mean():.1f}%')

    print('\n========= ROC AUC across samples =========')
    print('Treat cos_gt (per sample) as positive scores,')
    print('treat far-negative cos (per sample) as negative scores,')
    print('then compute AUC over the 2 × n score pairs.\n')
    auc_hard = auc_mann_whitney(cos_gt, cos_max_far)
    auc_mean = auc_mann_whitney(cos_gt, cos_mean_far)
    print(f'  cos_gt vs cos_max_far : AUC = {auc_hard:.4f}  '
          '(against per-sample hardest far negative)')
    print(f'  cos_gt vs cos_mean_far: AUC = {auc_mean:.4f}  '
          '(against average far cell)')
    print('  random baseline = 0.500; perfect = 1.000')

    print('\n========= rank_gt distribution =========')
    print('(rank = 1 means GT cell is the absolute argmax of the heatmap)')
    ranks = df['rank_gt'].astype(int).values
    for k in (1, 3, 5, 10, 25, 50):
        hit = int((ranks <= k).sum())
        print(f'  rank ≤ {k:3d}: {hit:4d}  ({100 * hit / n:5.1f}%)')
    print(f'  median rank: {int(np.median(ranks))}  '
          f'mean rank: {ranks.mean():6.1f}  '
          f'worst rank: {ranks.max()}')

    # Percentile rank — rank as a fraction of heatmap cells.
    # rank 百分位：rank / n_cells，越小越好（0 = 完美）。
    n_cells = (df['feat_h'] * df['feat_w']).astype(float).values
    pct = ranks / n_cells
    print(f'\nrank percentile (rank / n_cells):  '
          f'median {100 * np.median(pct):.1f}%  '
          f'mean {100 * pct.mean():.1f}%')
    print(f'  random baseline: 50%; perfect: ~{100 / n_cells.mean():.2f}%')


if __name__ == '__main__':
    main()
