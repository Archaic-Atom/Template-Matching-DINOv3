#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired bootstrap CIs for PCK@k / mean-error gaps between TM test CSVs.

对多份 TM test CSV 做配对 bootstrap,给出 PCK@k 与平均误差差值的点估计、
95% CI 以及 P(A 优于 B)。所有 CSV 按模板 ``index`` 对齐(同一测试集、同序),
每次重采样对所有模型用同一组下标(配对),消除模板难度的共同方差,
得到差值更紧的置信区间。

Usage:
    python paired_bootstrap_pck.py LABEL:CSV [LABEL:CSV ...] [--B 10000] [--seed 2024]
"""

import argparse
import csv
import math
from typing import Dict, List

import numpy as np

KS = (1, 2, 3, 5, 10)


def load_errors(path: str) -> Dict[int, float]:
    """Load template-index -> L2 pixel error from a TM test CSV.

    读 test CSV,返回 {index: L2 误差(px)}。

    Args:
        path: test_*.csv 路径,列为
            index,sar_name,rgb_name,gt_x,gt_y,pred_x,pred_y,score。

    Returns:
        Dict mapping template index to its prediction L2 error in pixels.
        模板 index 到预测 L2 误差(px)的字典。
    """
    out: Dict[int, float] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            idx = int(row["index"])
            out[idx] = math.hypot(
                float(row["pred_x"]) - float(row["gt_x"]),
                float(row["pred_y"]) - float(row["gt_y"]),
            )
    return out


def main() -> None:
    """Parse args, align CSVs by index, run paired bootstrap, print report.

    解析参数、按 index 对齐、跑配对 bootstrap、打印报告。
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("specs", nargs="+", help="LABEL:CSV pairs / 标签:CSV 路径")
    ap.add_argument("--B", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=2024, help="rng seed")
    args = ap.parse_args()

    labels: List[str] = []
    errs: List[Dict[int, float]] = []
    for spec in args.specs:
        label, path = spec.split(":", 1)
        labels.append(label)
        errs.append(load_errors(path))

    # Common, sorted index set so every model is evaluated on the same templates.
    # 取公共 index 并排序,保证每个模型评的是同一批模板。
    common = sorted(set.intersection(*[set(e.keys()) for e in errs]))
    n = len(common)
    M = len(labels)
    E = np.array([[e[i] for i in common] for e in errs])  # shape (M, n)
    print(f"aligned n={n} templates across {M} models: {', '.join(labels)}")

    def pck(row: np.ndarray, k: int) -> float:
        return 100.0 * float(np.mean(row <= k))

    print("\n== point estimates ==")
    for m in range(M):
        cells = "  ".join(f"PCK@{k}={pck(E[m], k):.2f}" for k in KS)
        print(f"  {labels[m]:>5}: {cells}   mean={E[m].mean():.3f}px")

    # Paired bootstrap: one shared index draw per resample applied to all models.
    # 配对 bootstrap:每次重采样取一组共享下标,套用到所有模型。
    rng = np.random.default_rng(args.seed)
    boot_idx = rng.integers(0, n, size=(args.B, n))  # (B, n)
    pck_boot = {k: np.empty((M, args.B)) for k in KS}
    mean_boot = np.empty((M, args.B))
    for m in range(M):
        em = E[m][boot_idx]  # (B, n)
        mean_boot[m] = em.mean(axis=1)
        for k in KS:
            pck_boot[k][m] = 100.0 * (em <= k).mean(axis=1)

    print(f"\n== paired bootstrap (B={args.B}, seed={args.seed}) ==")
    for a in range(M):
        for b in range(a + 1, M):
            print(f"\n--- {labels[a]} vs {labels[b]}  (Δ = {labels[a]} − {labels[b]}) ---")
            for k in KS:
                d = pck_boot[k][a] - pck_boot[k][b]
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = float(np.mean(d > 0))  # P(a has higher PCK@k)
                sig = "  *" if (lo > 0 or hi < 0) else ""
                print(f"  ΔPCK@{k:<2} = {d.mean():+.2f}  95%CI[{lo:+.2f},{hi:+.2f}]"
                      f"  P({labels[a]}>{labels[b]})={p:.3f}{sig}")
            dm = mean_boot[a] - mean_boot[b]
            lo, hi = np.percentile(dm, [2.5, 97.5])
            p = float(np.mean(dm < 0))  # lower mean error is better for a
            sig = "  *" if (lo > 0 or hi < 0) else ""
            print(f"  Δmean  = {dm.mean():+.3f}px 95%CI[{lo:+.3f},{hi:+.3f}]"
                  f"  P({labels[a]} lower)={p:.3f}{sig}")


if __name__ == "__main__":
    main()
