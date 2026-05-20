# -*- coding: utf-8 -*-
"""Stage-1 checkpoint matching evaluation on test data.

Stage 1 检查点在测试集上的匹配评估。

Computes fairer metrics than raw cos sim because GT pixel labels are not
aligned to the encoder's stride-16 feature grid:
    - cell distance (L∞ in feature units): how far argmax cell sits from
      the GT cell, after floor-dividing GT pixel coords by stride.
    - top-K hit rates: GT cell within the top K predicted positions.
    - sub-pixel refined pixel error: quadratic peak fit around argmax.

由于 GT 像素坐标并非对齐到 encoder 的 stride-16 特征网格，单看 cos 相似度
不公平，本脚本提供更公平的度量：cell 距离（L∞）、top-K 命中率、
sub-pixel 精修后的像素误差。

Usage:
    cd <project_root>
    CUDA_VISIBLE_DEVICES=4 python Source/Tools/eval_stage1_matching.py \\
        --n 1000 --seed 42 --ckpt Checkpoint_v3/model_epoch_0.pth
"""
import argparse
import json
import os
import random
import sys
from typing import Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image


def preprocess_image(img: torch.Tensor, duplicate_channels: bool = False) -> torch.Tensor:
    """Convert a uint8 image tensor to float, normalize per-image.

    将 uint8 图像张量转为 float 并按图归一化（减均值除标准差）。

    Args:
        img: ``[C, H, W]`` uint8 tensor as loaded by ``read_image``.
            ``read_image`` 加载得到的 uint8 张量。
        duplicate_channels: If True and the image is single-channel,
            repeat the channel to produce a 3-channel RGB-like tensor.
            若为 True 且单通道，则复制通道至 3 通道（SAR 复制成"伪 RGB"用）。

    Returns:
        ``[C', H, W]`` float32 tensor with zero mean and unit std per
        sample (C' = 3 if ``duplicate_channels`` triggers, else C).
    """
    if img.dtype != torch.float32:
        img = img.float()
    img = img.div(255.0)
    if img.shape[0] == 1 and duplicate_channels:
        img = img.repeat(3, 1, 1)
    mean = img.mean(dim=(1, 2), keepdim=True)
    std = img.std(dim=(1, 2), keepdim=True).clamp_min(1e-7)
    return (img - mean) / std


def quadratic_subpixel(
    heatmap_2d: torch.Tensor, peak_y: int, peak_x: int,
) -> Tuple[float, float]:
    """Refine an integer peak position to sub-pixel via 1D quadratic fit.

    用一维二次拟合把整数峰值位置精修到 sub-pixel 精度。

    Fits a parabola through ``(peak-1, peak, peak+1)`` along each axis
    independently. Stable for sharp peaks; for flat / noisy peaks it can
    pull the offset toward neighboring bins. Border peaks return (0, 0).

    Args:
        heatmap_2d: ``[H, W]`` score map.
            分数热图。
        peak_y: Row index of the integer-pixel peak.
            整数峰值的行索引。
        peak_x: Column index of the integer-pixel peak.
            整数峰值的列索引。

    Returns:
        ``(dy, dx)`` offset in feature-grid units, clamped to [-0.5, 0.5].
        多 stride 倍数后可得像素偏移。
    """
    H, W = heatmap_2d.shape
    if peak_y <= 0 or peak_y >= H - 1 or peak_x <= 0 or peak_x >= W - 1:
        return 0.0, 0.0
    yp1 = heatmap_2d[peak_y + 1, peak_x].item()
    ym1 = heatmap_2d[peak_y - 1, peak_x].item()
    y0 = heatmap_2d[peak_y, peak_x].item()
    denom_y = (2 * y0 - yp1 - ym1)
    dy = 0.5 * (ym1 - yp1) / denom_y if abs(denom_y) > 1e-9 else 0.0
    xp1 = heatmap_2d[peak_y, peak_x + 1].item()
    xm1 = heatmap_2d[peak_y, peak_x - 1].item()
    denom_x = (2 * y0 - xp1 - xm1)
    dx = 0.5 * (xm1 - xp1) / denom_x if abs(denom_x) > 1e-9 else 0.0
    dy = max(-0.5, min(0.5, dy))
    dx = max(-0.5, min(0.5, dx))
    return dy, dx


class _FrameworkArgs:
    """Minimal stand-in for argparse Namespace required by builder.

    builder 期望传入的 argparse-like 对象。这里只填它在 Stage-1 路径下
    会用到的字段，其它字段 yaml 没引用就不传。
    """

    is_stage_2 = False
    pre_trained = True
    dist = False
    root_path = '.'
    modelDir = '.'
    gpu = 1
    lr = 0.0002


def load_model(ckpt_path: str, config_path: str, device: torch.device) -> torch.nn.Module:
    """Build the composed model from yaml and load a Stage-1 checkpoint.

    通过 yaml 配置构建模型并加载 Stage-1 权重。

    Args:
        ckpt_path: Path to a JackFramework-saved ``.pth`` file with a
            ``model_0`` key holding ``trainable_tail.state_dict()``.
            JackFramework 保存的 .pth，外层 dict 含 ``model_0`` 键。
        config_path: Path to ``TM_Model_X/config.yaml``.
            模型的 yaml 配置文件路径。
        device: Target device.
            目标设备（一般是 ``torch.device('cuda:0')``）。

    Returns:
        Composed model in eval mode on ``device``.
    """
    from UserModelImplementation.Models.builder import build_model_from_yaml

    model = build_model_from_yaml(config_path, _FrameworkArgs())
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_0', ckpt)
    missing, unexpected = model.trainable_tail.load_state_dict(state, strict=False)
    print(f'  loaded ckpt: missing={len(missing)}, unexpected={len(unexpected)}')
    return model.to(device).eval()


def score_match(
    model: torch.nn.Module, sar: torch.Tensor, rgb: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run encoder pair, return cos-sim score map for matching positions.

    跑双 encoder，输出每个候选位置的余弦相似度（平均到 SAR 模板大小）。

    Args:
        model: Composed model with ``sar_encoder`` + ``rgb_encoder``.
            含 SAR 与 RGB 双 encoder 的模型。
        sar: ``[1, C, sH, sW]`` preprocessed SAR template tensor.
            预处理后的 SAR 模板张量。
        rgb: ``[1, 3, rH, rW]`` preprocessed RGB search-region tensor.
            预处理后的 RGB 搜索区张量。
        device: Device for inference.

    Returns:
        ``[oh, ow]`` cosine score map (mean over template footprint).
        余弦分数热图，每个位置代表 SAR 模板放在该位置时的平均 cos 相似度。
    """
    with torch.no_grad():
        output = model(sar.to(device), rgb.to(device))
    sar_feat = output['sar_feat']['sar16']
    rgb_feat = output['rbg_feat'][3]

    # L2-normalize each token along channel dim so conv gives cos sim.
    # 沿通道维做 L2 归一化，使后续卷积结果就是逐位置的 cos 相似度之和。
    sar_n = F.normalize(sar_feat, dim=1)
    rgb_n = F.normalize(rgb_feat, dim=1)

    _, channels, sar_h, sar_w = sar_n.shape
    weight = sar_n.view(1, channels, sar_h, sar_w)
    score = F.conv2d(rgb_n, weight) / (sar_h * sar_w)
    return score.squeeze(0).squeeze(0)


def evaluate_sample(
    model: torch.nn.Module, sar_path: str, rgb_path: str,
    gt_xy: Tuple[int, int], stride: int, device: torch.device,
) -> dict:
    """Evaluate one (SAR, RGB, GT) triple and return per-sample metrics.

    评估单组样本，返回 per-sample 指标字典。

    Args:
        model: Loaded model.
            加载好的模型。
        sar_path: Filesystem path to the SAR PNG.
            SAR PNG 路径。
        rgb_path: Filesystem path to the RGB PNG.
            RGB PNG 路径。
        gt_xy: Ground-truth (x, y) top-left pixel position of the SAR
            template inside the RGB image.
            SAR 模板在 RGB 中匹配的真值左上角像素坐标 (x, y)。
        stride: Encoder output stride (16 for stride-16 head).
            编码器输出步幅。
        device: Inference device.

    Returns:
        Dict with keys ``cell_dist``, ``rank``, ``err_raw``,
        ``err_sub``, ``max_cos``, ``gt_cos``, ``feat_hw``, etc.
        含 cell 距离、rank、原始/精修像素误差、max/gt cos 等的字典。
    """
    sar = read_image(sar_path, ImageReadMode.GRAY)
    rgb = read_image(rgb_path, ImageReadMode.RGB)
    sar_t = preprocess_image(sar, duplicate_channels=True).unsqueeze(0)
    rgb_t = preprocess_image(rgb).unsqueeze(0)

    score = score_match(model, sar_t, rgb_t, device)
    feat_h, feat_w = score.shape

    flat_idx = score.argmax()
    pred_y_feat = (flat_idx // feat_w).item()
    pred_x_feat = (flat_idx % feat_w).item()
    max_score = score.max().item()

    dy, dx = quadratic_subpixel(score, pred_y_feat, pred_x_feat)
    pred_x_raw = pred_x_feat * stride
    pred_y_raw = pred_y_feat * stride
    pred_x_sub = (pred_x_feat + dx) * stride
    pred_y_sub = (pred_y_feat + dy) * stride

    gt_x, gt_y = gt_xy
    err_raw = ((pred_x_raw - gt_x) ** 2 + (pred_y_raw - gt_y) ** 2) ** 0.5
    err_sub = ((pred_x_sub - gt_x) ** 2 + (pred_y_sub - gt_y) ** 2) ** 0.5

    gt_x_feat = gt_x // stride
    gt_y_feat = gt_y // stride
    cell_dist = max(abs(pred_x_feat - gt_x_feat), abs(pred_y_feat - gt_y_feat))

    if 0 <= gt_x_feat < feat_w and 0 <= gt_y_feat < feat_h:
        gt_score = score[gt_y_feat, gt_x_feat].item()
        rank = int((score.flatten() > gt_score).sum().item()) + 1
        margin = max_score - gt_score
    else:
        # GT outside score-map valid region (rare, RGB smaller than SAR + stride).
        # GT 落在评分图有效范围之外（很少见，仅当 RGB 比 SAR 还小时）。
        gt_score = None
        rank = None
        margin = None

    return {
        'feat_hw': (feat_h, feat_w),
        'gt_pixel': (gt_x, gt_y),
        'gt_feat': (gt_x_feat, gt_y_feat),
        'pred_feat': (pred_x_feat, pred_y_feat),
        'pred_pixel_raw': (pred_x_raw, pred_y_raw),
        'pred_pixel_sub': (pred_x_sub, pred_y_sub),
        'cell_dist': cell_dist,
        'rank': rank,
        'err_raw': err_raw,
        'err_sub': err_sub,
        'max_cos': max_score,
        'gt_cos': gt_score,
        'margin': margin,
    }


def main() -> None:
    """CLI entry: random-sample evaluation and aggregate summary.

    命令行入口：随机抽样评估并输出汇总。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n', type=int, default=1000,
                        help='Number of random samples to evaluate.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt', default='Checkpoint_v3/model_epoch_0.pth',
                        help='Checkpoint file path (Stage-1 .pth).')
    parser.add_argument('--csv', default='Datasets/image_test.csv',
                        help='Test CSV with columns sar_path, rgb_path, label.')
    parser.add_argument('--config', default='Source/UserModelImplementation/'
                        'Models/TM_Model_A/config.yaml')
    parser.add_argument('--stride', type=int, default=16,
                        help='Encoder output stride; default 16 for stride-16 head.')
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-sample lines (not just summary).')
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    # Source is on PYTHONPATH so UserModelImplementation imports work.
    # 把 Source 加进 PYTHONPATH，让 UserModelImplementation 能正常 import。
    sys.path.insert(0, 'Source')

    device = torch.device('cuda:0')
    print(f'Building model + loading ckpt={args.ckpt}')
    model = load_model(args.ckpt, args.config, device)

    print(f'Reading CSV: {args.csv}')
    df = pd.read_csv(args.csv, usecols=[0, 1, 2])
    random.seed(args.seed)
    indices = random.sample(range(len(df)), args.n)

    # Aggregates.
    # 累加器。
    cell_dists, errs_raw, errs_sub, margins = [], [], [], []
    max_coses, gt_coses = [], []
    top1 = top3 = top5 = 0
    feat_hw_last = None

    import time
    t0 = time.time()
    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        gt = json.loads(row.iloc[2])
        result = evaluate_sample(
            model, row.iloc[0], row.iloc[1],
            (int(gt[0]), int(gt[1])), args.stride, device,
        )
        feat_hw_last = result['feat_hw']

        # Periodic progress for long runs.
        # 长跑下定期打印进度，便于监控。
        if (i + 1) % 5000 == 0 or i == args.n - 1:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (args.n - i - 1) / rate
            print(f'  progress {i + 1:6d}/{args.n}  '
                  f'{rate:5.1f} samp/s  elapsed {dt/60:.1f} min  '
                  f'eta {eta/60:.1f} min', flush=True)

        cell_dists.append(result['cell_dist'])
        errs_raw.append(result['err_raw'])
        errs_sub.append(result['err_sub'])
        max_coses.append(result['max_cos'])
        if result['margin'] is not None:
            margins.append(result['margin'])
            gt_coses.append(result['gt_cos'])
            rank = result['rank']
            if rank <= 1:
                top1 += 1
            if rank <= 3:
                top3 += 1
            if rank <= 5:
                top5 += 1

        if args.verbose:
            r = result
            print(f"\nsample {i:4d} (row {idx})  feat {r['feat_hw'][0]}x{r['feat_hw'][1]}")
            print(f"  gt {r['gt_pixel']}  feat {r['gt_feat']}")
            print(f"  pred raw {r['pred_pixel_raw']}  sub {r['pred_pixel_sub']}  feat {r['pred_feat']}")
            print(f"  cell_dist={r['cell_dist']}  rank={r['rank']}  "
                  f"err_raw={r['err_raw']:.1f}  err_sub={r['err_sub']:.1f}  "
                  f"max_cos={r['max_cos']:.4f}  gt_cos={r['gt_cos']}")

    n = args.n
    h_out, w_out = feat_hw_last
    print(f'\n{"=" * 12} SUMMARY (n={n}) {"=" * 12}')
    # Stride quantization makes ±1-cell matches genuinely correct: a
    # template that isn't grid-aligned overlaps a neighbor cell more
    # than its own floor(gt/stride) cell. Report fair hit prominently.
    # 由于 stride 量化，模板不对齐网格时邻居 cell 才是真正最优匹配，
    # 所以 cell_dist≤1 才是公平 top-1，单独突出展示。
    hit_le1 = cell_dists.count(0) + cell_dists.count(1)
    hit_le2 = hit_le1 + cell_dists.count(2)
    rand_le1 = 9 / (h_out * w_out) * 100  # ~9 cells (1 + 8 neighbors)
    rand_le2 = 25 / (h_out * w_out) * 100  # 5x5 window
    print(f'cell_dist L∞:    mean={sum(cell_dists)/n:.2f}  '
          f'hist 0:{cell_dists.count(0)} 1:{cell_dists.count(1)} '
          f'2:{cell_dists.count(2)} 3+:{sum(1 for d in cell_dists if d >= 3)}')
    print(f'>>> stride-fair hit (cell_dist<=1): {hit_le1}/{n} '
          f'({100*hit_le1/n:.1f}%)  [random ~{rand_le1:.1f}%]')
    print(f'    relaxed hit  (cell_dist<=2):    {hit_le2}/{n} '
          f'({100*hit_le2/n:.1f}%)  [random ~{rand_le2:.1f}%]')
    print(f'    strict       (cell_dist==0):    {cell_dists.count(0)}/{n} '
          f'({100*cell_dists.count(0)/n:.1f}%)')
    print(f'top-1 hit: {top1}/{n} ({100*top1/n:.1f}%)  '
          f'[random {100/(h_out*w_out):.1f}%]')
    print(f'top-3 hit: {top3}/{n} ({100*top3/n:.1f}%)  '
          f'[random {300/(h_out*w_out):.1f}%]')
    print(f'top-5 hit: {top5}/{n} ({100*top5/n:.1f}%)  '
          f'[random {500/(h_out*w_out):.1f}%]')
    print(f'pixel err raw      mean={sum(errs_raw)/n:5.1f}px  '
          f'median={sorted(errs_raw)[n // 2]:5.1f}px')
    print(f'pixel err sub-pix  mean={sum(errs_sub)/n:5.1f}px  '
          f'median={sorted(errs_sub)[n // 2]:5.1f}px')
    # Cosine similarity stats.
    # 余弦相似度统计：argmax 位置 cos、GT 位置 cos、二者差(margin)。
    max_cos_sorted = sorted(max_coses)
    print(f'cos@pred (argmax)  mean={sum(max_coses)/n:.4f}  '
          f'median={max_cos_sorted[n // 2]:.4f}  '
          f'min={max_cos_sorted[0]:.4f}  max={max_cos_sorted[-1]:.4f}')
    if gt_coses:
        m = len(gt_coses)
        gt_cos_sorted = sorted(gt_coses)
        print(f'cos@gt             mean={sum(gt_coses)/m:.4f}  '
              f'median={gt_cos_sorted[m // 2]:.4f}  '
              f'min={gt_cos_sorted[0]:.4f}  max={gt_cos_sorted[-1]:.4f}')
    if margins:
        print(f'margin (max-gt cos): mean={sum(margins)/len(margins):.4f}')


if __name__ == '__main__':
    main()
