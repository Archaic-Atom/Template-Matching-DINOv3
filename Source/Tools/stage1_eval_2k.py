# -*- coding: utf-8 -*-
"""Per-sample Stage-1 cos-sim matching on a fixed CSV, all rows in order.

按 CSV 顺序逐行跑 Stage-1 cos-sim 匹配并写 per-sample CSV。

Output schema matches the first 8 columns of Stage-2's test CSV
(``TMDataloader.save_result`` — ``index, sar_name, rgb_name, gt_x, gt_y,
pred_x, pred_y, score``) plus 4 Stage-1-only diagnostics
(``gt_score, margin, err_sub, cell_dist``) so the resulting file joins
trivially with ``DebugResult/test_2k_epoch*.csv`` on ``index`` for the
4-quadrant Stage-1-vs-Stage-2 analysis.

输出列前 8 列对齐 Stage-2 的 test CSV（``index, sar_name, rgb_name, gt_x,
gt_y, pred_x, pred_y, score``），后面再附 4 列 Stage-1 专属诊断
（``gt_score, margin, err_sub, cell_dist``），便于按 ``index`` 与
``DebugResult/test_2k_epoch*.csv`` 行 join，做 4 象限对比。

Usage:
    cd <project_root>
    CUDA_VISIBLE_DEVICES=0 python Source/Tools/stage1_eval_2k.py \\
        --ckpt Checkpoint_v3/model_epoch_3.pth \\
        --csv Datasets/image_test_2k.csv \\
        --out DebugResult/stage1_2k_epoch3.csv
"""
import argparse
import csv
import json
import os
import sys
import time

import pandas as pd
import torch

# Source on PYTHONPATH so eval_stage1_matching's UserModelImplementation
# imports also resolve when this script lives anywhere on disk.
# 把 Source 加进 PYTHONPATH，使 eval_stage1_matching 内部的 import 可用。
sys.path.insert(0, 'Source')

from Tools.eval_stage1_matching import evaluate_sample, load_model


def main() -> None:
    """CLI entry — write per-sample Stage-1 metrics CSV.

    命令行入口：逐行评估并写 per-sample CSV。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--ckpt', default='Checkpoint_v3/model_epoch_3.pth',
                        help='Stage-1 checkpoint .pth path.')
    parser.add_argument('--csv', default='Datasets/image_test_2k.csv',
                        help='Input CSV with columns SAR_Image, RGB_Image, '
                             'Label (JSON [x, y]).')
    parser.add_argument('--out', default='DebugResult/stage1_2k_epoch3.csv',
                        help='Per-sample output CSV.')
    parser.add_argument('--config',
                        default='Source/UserModelImplementation/'
                                'Models/TM_Model_A/config.yaml',
                        help='Variant yaml; Stage-1 uses A.')
    parser.add_argument('--stride', type=int, default=16,
                        help='Encoder output stride.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Index passed to CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--n', type=int, default=-1,
                        help='Limit rows for quick smoke test; -1 = all.')
    parser.add_argument('--report_every', type=int, default=100,
                        help='Print + flush every N rows.')
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = torch.device('cuda:0')

    print(f'Building model + loading ckpt={args.ckpt}', flush=True)
    model = load_model(args.ckpt, args.config, device)

    print(f'Reading CSV: {args.csv}', flush=True)
    df = pd.read_csv(args.csv, usecols=[0, 1, 2])
    n_total = len(df) if args.n < 0 else min(args.n, len(df))
    print(f'Total rows to evaluate: {n_total}', flush=True)

    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)

    cols = ['index', 'sar_name', 'rgb_name', 'gt_x', 'gt_y',
            'pred_x', 'pred_y', 'score',
            'gt_score', 'margin', 'err_sub', 'cell_dist']

    t0 = time.time()
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(n_total):
            row = df.iloc[i]
            sar_path = row.iloc[0]
            rgb_path = row.iloc[1]
            gt = json.loads(row.iloc[2])
            gt_x, gt_y = int(gt[0]), int(gt[1])

            r = evaluate_sample(
                model, sar_path, rgb_path, (gt_x, gt_y),
                args.stride, device,
            )
            pred_x_sub, pred_y_sub = r['pred_pixel_sub']

            gt_score = '' if r['gt_cos'] is None else f'{r["gt_cos"]:.6f}'
            margin = '' if r['margin'] is None else f'{r["margin"]:.6f}'

            writer.writerow([
                i,
                os.path.basename(sar_path),
                os.path.basename(rgb_path),
                gt_x, gt_y,
                f'{pred_x_sub:.3f}', f'{pred_y_sub:.3f}',
                f'{r["max_cos"]:.6f}',
                gt_score, margin,
                f'{r["err_sub"]:.3f}',
                r['cell_dist'],
            ])

            if (i + 1) % args.report_every == 0 or i == n_total - 1:
                dt = time.time() - t0
                rate = (i + 1) / dt
                eta = (n_total - i - 1) / rate if rate > 0 else 0.0
                print(f'  {i + 1:4d}/{n_total}  {rate:5.1f} samp/s  '
                      f'elapsed {dt / 60:5.2f} min  '
                      f'eta {eta / 60:5.2f} min', flush=True)
                f.flush()

    print(f'Wrote {n_total} rows to {args.out}', flush=True)


if __name__ == '__main__':
    main()
