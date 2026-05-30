# -*- coding: utf-8 -*-
"""Per-sample Stage-1 separation statistics on a fixed CSV.

按 CSV 顺序逐行跑 Stage-1 cos-sim heatmap 并提取每样本的判别力统计。

This evaluates Stage 1 on ITS OWN training objective (PatchNCE relative
ranking), not on sub-pixel pixel error. Per sample we record:

- ``cos_gt``    cos sim at the GT cell (the positive)
- ``cos_max``   cos sim at the absolute argmax (=Stage-1's predicted cell)
- ``cos_max_far`` max cos among cells with cell_dist > far_radius from GT
  (hardest random negative outside the GT neighborhood)
- ``cos_mean_far`` / ``cos_std_far``  mean / std over the same far pool
- ``rank_gt``   rank of cos_gt across all cells of this heatmap (1 = top)
- ``n_far_cells`` size of the far-negative pool
- ``cell_dist_argmax`` L∞ cell distance between argmax and GT
- ``gt_in_bound`` 1 iff GT cell falls inside the valid heatmap region

测的是 Stage 1 的 PatchNCE 训练目标（相对排序），不是亚像素误差。

The downstream analysis (``analyze_stage1_separation.py``) reads this
CSV and emits d-prime distributions, ROC AUC (cos_gt vs cos_max_far),
and rank_gt histograms.

Usage:
    cd <project_root>
    CUDA_VISIBLE_DEVICES=4 python Source/Tools/stage1_separation_2k.py \\
        --ckpt Checkpoint_v3/model_epoch_3.pth \\
        --csv Datasets/image_test_2k.csv \\
        --out DebugResult/stage1_2k_sep_epoch3.csv
"""
import argparse
import csv
import json
import os
import sys
import time

import pandas as pd
import torch
from torchvision.io import ImageReadMode, read_image

# Source on PYTHONPATH so eval_stage1_matching internals resolve.
# 把 Source 加进 PYTHONPATH，让 eval_stage1_matching 的内部 import 生效。
sys.path.insert(0, 'Source')

from Tools.eval_stage1_matching import load_model, preprocess_image, score_match  # noqa: E402


def evaluate_separation(
    model: torch.nn.Module, sar_path: str, rgb_path: str,
    gt_xy: tuple, stride: int, device: torch.device, far_radius: int,
) -> dict:
    """Compute per-sample cos statistics from one Stage-1 heatmap.

    单样本：跑 Stage-1 heatmap，提取 GT cell 的 cos、远阴性 cells 的 cos
    统计、以及 GT 的 rank。

    Args:
        model: Loaded Stage-1 composed model.
            加载好的 Stage-1 模型。
        sar_path / rgb_path: PNG file paths.
            SAR / RGB 图像路径。
        gt_xy: Ground-truth (x, y) top-left of template inside RGB.
            SAR 在 RGB 内的 GT 左上角像素坐标。
        stride: Encoder output stride.
            编码器步幅。
        device: Inference device.
        far_radius: Cells with cell_dist > this from GT are "far
            negatives". Default 2 → excludes the 5×5 neighborhood.
            cell_dist > far_radius 的 cells 算"远阴性"。默认 2 表示
            排除 GT 的 5×5 邻域。

    Returns:
        Dict with the per-sample fields listed in the module docstring.
        返回上面 module docstring 列出的每样本字段字典。
    """
    sar = read_image(sar_path, ImageReadMode.GRAY)
    rgb = read_image(rgb_path, ImageReadMode.RGB)
    sar_t = preprocess_image(sar, duplicate_channels=True).unsqueeze(0)
    rgb_t = preprocess_image(rgb).unsqueeze(0)

    score = score_match(model, sar_t, rgb_t, device)
    fh, fw = score.shape

    gt_x, gt_y = gt_xy
    gt_xf, gt_yf = gt_x // stride, gt_y // stride
    in_bound = (0 <= gt_xf < fw) and (0 <= gt_yf < fh)

    flat_idx = score.argmax()
    pred_yf = (flat_idx // fw).item()
    pred_xf = (flat_idx % fw).item()
    cos_max = score.max().item()
    cell_dist_argmax = max(abs(pred_xf - gt_xf), abs(pred_yf - gt_yf))

    if not in_bound:
        return {
            'feat_h': fh, 'feat_w': fw,
            'cos_gt': None, 'cos_max': cos_max,
            'cos_max_far': None, 'cos_mean_far': None, 'cos_std_far': None,
            'rank_gt': None, 'n_far_cells': 0,
            'cell_dist_argmax': cell_dist_argmax, 'gt_in_bound': 0,
        }

    cos_gt = score[gt_yf, gt_xf].item()
    # Rank: number of cells strictly above cos_gt, plus 1.
    # rank: 比 cos_gt 严格大的 cell 数 + 1（rank=1 表示 GT 就是 argmax）。
    rank_gt = int((score.flatten() > cos_gt).sum().item()) + 1

    # Build cell-distance grid relative to GT, then far mask.
    # 构造每个 cell 到 GT 的 L∞ 距离，得到远阴性掩码。
    ys = torch.arange(fh, device=score.device)
    xs = torch.arange(fw, device=score.device)
    dy = (ys[:, None] - gt_yf).abs()
    dx = (xs[None, :] - gt_xf).abs()
    cell_dist = torch.maximum(dy, dx)
    far_mask = cell_dist > far_radius
    far_scores = score[far_mask]
    n_far = int(far_mask.sum().item())

    if n_far > 0:
        cos_max_far = far_scores.max().item()
        cos_mean_far = far_scores.mean().item()
        cos_std_far = far_scores.std().item() if n_far > 1 else 0.0
    else:
        cos_max_far = cos_mean_far = cos_std_far = None

    return {
        'feat_h': fh, 'feat_w': fw,
        'cos_gt': cos_gt, 'cos_max': cos_max,
        'cos_max_far': cos_max_far,
        'cos_mean_far': cos_mean_far,
        'cos_std_far': cos_std_far,
        'rank_gt': rank_gt, 'n_far_cells': n_far,
        'cell_dist_argmax': cell_dist_argmax, 'gt_in_bound': 1,
    }


def _f(v: float | None) -> str:
    """Format float or empty string if None.

    数值格式化；None 返回空串以便 CSV 留空。
    """
    return '' if v is None else f'{v:.6f}'


def main() -> None:
    """CLI entry — emit per-sample separation CSV.

    命令行入口：逐样本评估并写 per-sample CSV。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--ckpt', default='Checkpoint_v3/model_epoch_3.pth')
    parser.add_argument('--csv', default='Datasets/image_test_2k.csv')
    parser.add_argument('--out',
                        default='DebugResult/stage1_2k_sep_epoch3.csv')
    parser.add_argument('--config',
                        default='Source/UserModelImplementation/'
                                'Models/TM_Model_A/config.yaml')
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--far_radius', type=int, default=2,
                        help='cells with cell_dist > this from GT count '
                             'as far negatives; default 2 → excludes '
                             '5×5 GT neighborhood.')
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--n', type=int, default=-1,
                        help='Row cap for quick smoke; -1 = all.')
    parser.add_argument('--report_every', type=int, default=100)
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = torch.device('cuda:0')

    print(f'Building model + loading ckpt={args.ckpt}', flush=True)
    model = load_model(args.ckpt, args.config, device)

    df = pd.read_csv(args.csv, usecols=[0, 1, 2])
    n_total = len(df) if args.n < 0 else min(args.n, len(df))
    print(f'Total rows: {n_total}, far_radius={args.far_radius}', flush=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cols = ['index', 'sar_name', 'rgb_name', 'feat_h', 'feat_w',
            'cos_gt', 'cos_max', 'cos_max_far',
            'cos_mean_far', 'cos_std_far',
            'rank_gt', 'n_far_cells',
            'cell_dist_argmax', 'gt_in_bound']

    t0 = time.time()
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(n_total):
            row = df.iloc[i]
            gt = json.loads(row.iloc[2])
            r = evaluate_separation(
                model, row.iloc[0], row.iloc[1],
                (int(gt[0]), int(gt[1])), args.stride, device,
                args.far_radius,
            )
            writer.writerow([
                i,
                os.path.basename(row.iloc[0]),
                os.path.basename(row.iloc[1]),
                r['feat_h'], r['feat_w'],
                _f(r['cos_gt']), _f(r['cos_max']), _f(r['cos_max_far']),
                _f(r['cos_mean_far']), _f(r['cos_std_far']),
                r['rank_gt'] if r['rank_gt'] is not None else '',
                r['n_far_cells'],
                r['cell_dist_argmax'],
                r['gt_in_bound'],
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
