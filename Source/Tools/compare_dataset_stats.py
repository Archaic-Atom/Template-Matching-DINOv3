# -*- coding: utf-8 -*-
"""Compare patch-level SAR/RGB statistics across multiple test CSVs.

跨数据集 patch 级别 SAR/RGB 统计对比。

For each input CSV (expected columns: SAR_Image, RGB_Image, Label) we
sample N patches and compute:

- ``sar_std`` / ``rgb_std``  pixel-intensity std (single-channel for SAR,
  mean over channels for RGB)
- ``sar_edge`` / ``rgb_edge`` mean Sobel magnitude (proxy for edge / texture
  density)
- ``sar_hf_ratio`` / ``rgb_hf_ratio`` fraction of FFT magnitude energy in
  frequencies outside the lowest 25% radius band (proxy for high-freq /
  fine-texture content)
- ``sar_mean`` / ``rgb_mean``  mean intensity (cleanly flat patches show up
  here)

Then aggregates mean / median / std across the N patches per dataset and
prints a side-by-side comparison table.

针对每个输入 CSV 采 N 个 patch，算 SAR/RGB 的 std、Sobel 边缘强度、FFT 高频
能量占比、均值；最后按数据集聚合 mean / median / std 并并排打印。

Usage:
    python Source/Tools/compare_dataset_stats.py \\
        --csvs SEN1-2=Datasets/image_test_2k.csv \\
               WHU=/data5/datasets/cross_eval/whu_opt_sar/cross_eval.csv \\
               QXS=/data5/datasets/cross_eval/qxs_saropt/cross_eval.csv \\
        --n 500 --seed 42
"""
import argparse
import random

import numpy as np
import pandas as pd
from PIL import Image


def _sobel(arr: np.ndarray) -> np.ndarray:
    """Sobel edge magnitude on a 2D float array.

    单通道二维数组的 Sobel 边缘幅值。

    Args:
        arr: ``[H, W]`` float array.

    Returns:
        ``[H-2, W-2]`` edge magnitude.
    """
    gx = (arr[1:-1, 2:] - arr[1:-1, :-2]) * 2 + \
        (arr[:-2, 2:] - arr[:-2, :-2]) + (arr[2:, 2:] - arr[2:, :-2])
    gy = (arr[2:, 1:-1] - arr[:-2, 1:-1]) * 2 + \
        (arr[2:, :-2] - arr[:-2, :-2]) + (arr[2:, 2:] - arr[:-2, 2:])
    return np.hypot(gx, gy)


def _hf_ratio(arr: np.ndarray, lo_frac: float = 0.25) -> float:
    """Fraction of FFT magnitude energy outside the central ``lo_frac`` radius.

    FFT 幅度谱中半径大于 ``lo_frac × min(H,W)/2`` 的能量占比。

    Args:
        arr: ``[H, W]`` real array.
        lo_frac: cutoff radius as fraction of half-size; energy at radii
            ``> lo_frac × half_min_dim`` counts as "high frequency".
            截断半径占短边的一半的比例；半径更大的频点算"高频"。

    Returns:
        Energy ratio in [0, 1].
    """
    h, w = arr.shape
    f = np.fft.fft2(arr - arr.mean())
    mag = np.abs(np.fft.fftshift(f))
    yy = np.arange(h) - h // 2
    xx = np.arange(w) - w // 2
    r = np.hypot(yy[:, None], xx[None, :])
    cutoff = lo_frac * min(h, w) / 2
    total = mag.sum() + 1e-9
    return float(mag[r > cutoff].sum() / total)


def patch_metrics(sar: np.ndarray, rgb: np.ndarray) -> dict:
    """Compute the per-patch metrics for one SAR/RGB pair.

    单对 SAR/RGB 的 patch 级别统计。

    Args:
        sar: ``[H, W]`` uint8 SAR template.
        rgb: ``[H, W, 3]`` uint8 RGB search image.

    Returns:
        Flat metric dict (all values are Python floats).
    """
    s = sar.astype(np.float32)
    r = rgb.astype(np.float32).mean(axis=-1)
    return {
        'sar_mean': float(s.mean()),
        'sar_std': float(s.std()),
        'sar_edge': float(_sobel(s).mean()),
        'sar_hf_ratio': _hf_ratio(s),
        'rgb_mean': float(r.mean()),
        'rgb_std': float(r.std()),
        'rgb_edge': float(_sobel(r).mean()),
        'rgb_hf_ratio': _hf_ratio(r),
    }


def collect(csv_path: str, n: int, seed: int) -> pd.DataFrame:
    """Sample ``n`` rows from ``csv_path``, load patches, return metric DF.

    从 csv_path 抽 n 行，加载 patch，返回每行一组指标的 DataFrame。
    """
    df = pd.read_csv(csv_path, usecols=[0, 1, 2])
    rng = random.Random(seed)
    idx = rng.sample(range(len(df)), min(n, len(df)))
    rows = []
    for i, k in enumerate(idx):
        row = df.iloc[k]
        sar = np.asarray(Image.open(row.iloc[0]).convert('L'))
        rgb = np.asarray(Image.open(row.iloc[1]).convert('RGB'))
        rows.append(patch_metrics(sar, rgb))
        if (i + 1) % 100 == 0:
            print(f'  loaded {i + 1}/{len(idx)}', flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry — load patches per dataset, aggregate, print side-by-side.

    命令行入口：每数据集加载 patch，聚合，打印对照表。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--csvs', nargs='+', required=True,
                        help='space-separated ``LABEL=CSV_PATH`` pairs.')
    parser.add_argument('--n', type=int, default=500,
                        help='patches sampled per dataset.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    datasets = {}
    for spec in args.csvs:
        if '=' not in spec:
            raise ValueError(f'expected LABEL=PATH, got: {spec}')
        label, path = spec.split('=', 1)
        print(f'\n[{label}] collecting {args.n} patches from {path} ...',
              flush=True)
        datasets[label] = collect(path, args.n, args.seed)

    metrics = ['sar_mean', 'sar_std', 'sar_edge', 'sar_hf_ratio',
               'rgb_mean', 'rgb_std', 'rgb_edge', 'rgb_hf_ratio']
    labels = list(datasets.keys())

    print('\n========= Per-metric comparison (mean ± std across patches) '
          '=========')
    header = f"{'metric':<14}"
    for lab in labels:
        header += f'{lab:>16}'
    print(header)
    for m in metrics:
        row = f'{m:<14}'
        for lab in labels:
            x = datasets[lab][m]
            row += f'  {x.mean():7.3f}±{x.std():6.3f}'
        print(row)

    print('\n========= Per-metric medians =========')
    header = f"{'metric':<14}"
    for lab in labels:
        header += f'{lab:>11}'
    print(header)
    for m in metrics:
        row = f'{m:<14}'
        for lab in labels:
            row += f'  {datasets[lab][m].median():9.3f}'
        print(row)

    # Pairwise mean ratios so you can read e.g. "SEN1-2 / WHU = 0.7×".
    # 两两均值比，方便看"SEN1-2 比 WHU 低多少"。
    if len(labels) >= 2:
        print('\n========= Ratio of means (rows ÷ columns) =========')
        for m in metrics:
            print(f'\n  {m}:')
            for li in labels:
                row = f'    {li:<10}'
                for lj in labels:
                    mi = datasets[li][m].mean()
                    mj = datasets[lj][m].mean()
                    ratio = mi / mj if abs(mj) > 1e-9 else float('nan')
                    row += f'  {ratio:6.2f}'
                print(row)


if __name__ == '__main__':
    main()
