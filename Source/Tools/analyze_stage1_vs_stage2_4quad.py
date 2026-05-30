# -*- coding: utf-8 -*-
"""4-quadrant breakdown of Stage-1 vs Stage-2 per-sample outcomes.

Stage-1 vs Stage-2 per-sample 4 象限分析。

Inputs:
    --s1 CSV from stage1_eval_2k.py (cols: index, ..., pred_x, pred_y,
         score, gt_score, margin, err_sub, cell_dist)
    --s2 CSV from JackFramework test mode (cols: index, sar_name,
         rgb_name, gt_x, gt_y, pred_x, pred_y, score)

Output: text report with quadrant counts, mean errs, and Stage-1
margin / score correlations with Stage-2 error.

输入两个 per-sample CSV（Stage 1 来自 stage1_eval_2k.py，Stage 2 来自
JackFramework test mode），按 index merge，做 4 象限统计并报告
Stage-1 margin / score 与 Stage-2 err 的相关。

Usage:
    python Source/Tools/analyze_stage1_vs_stage2_4quad.py \\
        --s1 DebugResult/stage1_2k_epoch3.csv \\
        --s2 DebugResult/test_2k_epoch3.csv \\
        --threshold 16
"""
import argparse

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--s1', required=True, help='Stage-1 per-sample CSV.')
    parser.add_argument('--s2', required=True, help='Stage-2 per-sample CSV.')
    parser.add_argument('--threshold', type=float, default=16.0,
                        help='Pixel err threshold for OK/fail split. '
                             '16 px = 1 stride-16 cell = quantization floor.')
    args = parser.parse_args()

    s1 = pd.read_csv(args.s1)
    s2 = pd.read_csv(args.s2)
    print(f'Stage 1 rows: {len(s1)}; Stage 2 rows: {len(s2)}')

    # Recompute err for both to be uniform.
    # 重新计算两侧的像素 L2 err，保持口径一致。
    s1['err_s1'] = np.hypot(s1['pred_x'] - s1['gt_x'],
                            s1['pred_y'] - s1['gt_y'])
    s2['err_s2'] = np.hypot(s2['pred_x'] - s2['gt_x'],
                            s2['pred_y'] - s2['gt_y'])

    df = s1[['index', 'err_s1', 'cell_dist', 'score', 'gt_score',
             'margin']].merge(
        s2[['index', 'err_s2']], on='index', how='inner',
        suffixes=('', '_s2'))
    print(f'Merged rows: {len(df)}')
    n = len(df)

    thr = args.threshold
    df['s1_ok'] = df['err_s1'] < thr
    df['s2_ok'] = df['err_s2'] < thr

    print(f'\nThreshold for OK: err < {thr} px (= 1 stride-16 cell)')
    print(f'Stage 1 OK: {df.s1_ok.sum()} / {n} '
          f'({100 * df.s1_ok.mean():.1f}%)')
    print(f'Stage 2 OK: {df.s2_ok.sum()} / {n} '
          f'({100 * df.s2_ok.mean():.1f}%)')

    print('\n========= 4-quadrant table (Stage 1 vs Stage 2) =========')
    quadrants = [
        ('Q1 both OK',         df.s1_ok & df.s2_ok,
         'signal traveled cleanly'),
        ('Q2 S1 OK, S2 fail',  df.s1_ok & ~df.s2_ok,
         'matching head DEGRADES the signal'),
        ('Q3 S1 fail, S2 OK',  ~df.s1_ok & df.s2_ok,
         'matching head fixes upstream'),
        ('Q4 both fail',       ~df.s1_ok & ~df.s2_ok,
         'Stage-1 alignment CEILING — modality-aware loss could help'),
    ]
    print(f'{"Quadrant":<22}{"count":>7}{"pct":>8}  '
          f'{"err_s1 μ":>10}{"err_s2 μ":>10}  meaning')
    for name, mask, meaning in quadrants:
        sub = df[mask]
        if len(sub):
            e1 = sub['err_s1'].mean()
            e2 = sub['err_s2'].mean()
            print(f'{name:<22}{len(sub):>7}{100*len(sub)/n:>7.1f}%  '
                  f'{e1:>10.2f}{e2:>10.2f}  {meaning}')
        else:
            print(f'{name:<22}{0:>7}{0:>7.1f}%       —         —    {meaning}')

    print('\n========= Stage 1 cell_dist (stride-fair) =========')
    le0 = (df.cell_dist == 0).sum()
    le1 = (df.cell_dist <= 1).sum()
    print(f'cell_dist == 0: {le0} ({100 * le0 / n:.1f}%)')
    print(f'cell_dist <= 1: {le1} ({100 * le1 / n:.1f}%)  '
          f'<- stride-fair top-1')

    print('\n========= Stage 1 signals vs Stage 2 err =========')
    # Pearson + Spearman of margin and score with err_s2.
    # 计算 Stage-1 margin / max_cos / gt_cos 与 Stage-2 err 的 Pearson 与
    # Spearman 相关性。绝对值越大相关越强。
    from scipy.stats import pearsonr, spearmanr
    for col in ('margin', 'score', 'gt_score'):
        x = df[col].astype(float)
        y = df['err_s2'].astype(float)
        mask = x.notna() & y.notna()
        pr, _ = pearsonr(x[mask], y[mask])
        sr, _ = spearmanr(x[mask], y[mask])
        print(f'  {col:<10} vs err_s2:  pearson={pr:+.3f}  '
              f'spearman={sr:+.3f}')

    print('\n========= Q4 (both fail) deeper look =========')
    q4 = df[~df.s1_ok & ~df.s2_ok]
    if len(q4):
        print(f'n={len(q4)}, mean err_s1={q4.err_s1.mean():.1f}, '
              f'err_s2={q4.err_s2.mean():.1f}')
        print(f'  err_s1 quartiles: '
              f'{q4.err_s1.quantile([.25, .5, .75]).round(1).tolist()}')
        print(f'  err_s2 quartiles: '
              f'{q4.err_s2.quantile([.25, .5, .75]).round(1).tolist()}')
        # Correlation within Q4 — within failure cases, do Stage 1 and
        # Stage 2 fail on the SAME samples (correlated err) or are they
        # independently wrong?
        # Q4 内部 Stage 1 与 Stage 2 的 err 是同源失败还是各自失败？
        corr_q4 = q4[['err_s1', 'err_s2']].corr().iloc[0, 1]
        print(f'  pearson(err_s1, err_s2) within Q4: {corr_q4:+.3f}  '
              '(high → fail on same samples; low → independent failures)')

    print('\n========= Q2 (S1 OK, S2 fail) regression cases =========')
    q2 = df[df.s1_ok & ~df.s2_ok]
    if len(q2):
        print(f'n={len(q2)} = {100*len(q2)/n:.1f}% — matching head '
              'introduced these errors')
        print(f'  mean err_s1={q2.err_s1.mean():.1f}  '
              f'err_s2={q2.err_s2.mean():.1f}')
        print(f'  err_s2 quartiles: '
              f'{q2.err_s2.quantile([.25, .5, .75]).round(1).tolist()}')


if __name__ == '__main__':
    main()
