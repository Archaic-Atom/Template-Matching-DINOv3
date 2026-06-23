#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize a TM test CSV: PCK/CMR@k, median, high/low-confidence split.

读 test_*.csv（index,sar_name,rgb_name,gt_x,gt_y,pred_x,pred_y,score），
算 L2 误差的 PCK@1/2/3/5、均值/中位数，并按 score 高低半分。
"""
import csv
import sys
import math


def pck(ds, k):
    return 100.0 * sum(1 for d in ds if d <= k) / len(ds)


def main():
    path = sys.argv[1]
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            gx, gy = float(row["gt_x"]), float(row["gt_y"])
            px, py = float(row["pred_x"]), float(row["pred_y"])
            d = math.hypot(px - gx, py - gy)
            rows.append((d, float(row["score"])))
    n = len(rows)
    ds = [d for d, _ in rows]
    ds_sorted = sorted(ds)
    median = ds_sorted[n // 2]
    mean = sum(ds) / n
    print(f"file: {path}")
    print(f"n={n}  mean={mean:.2f}px  median={median:.2f}px")
    for k in (1, 2, 3, 5, 10):
        print(f"  PCK@{k}px = {pck(ds, k):.2f}%")
    # high vs low confidence half (by score)
    rows_by_score = sorted(rows, key=lambda x: x[1], reverse=True)
    half = n // 2
    hi = [d for d, _ in rows_by_score[:half]]
    lo = [d for d, _ in rows_by_score[half:]]
    print(f"  HIGH-conf half: mean={sum(hi)/len(hi):.2f}px  median={sorted(hi)[len(hi)//2]:.2f}px  PCK@3={pck(hi,3):.2f}%")
    print(f"  LOW-conf  half: mean={sum(lo)/len(lo):.2f}px  median={sorted(lo)[len(lo)//2]:.2f}px  PCK@3={pck(lo,3):.2f}%")
    # percentiles + gross tail / 分位数与粗差尾部
    def pct(q):
        return ds_sorted[min(n - 1, int(q * n))]
    print(f"  P75={pct(0.75):.2f}px  P90={pct(0.90):.2f}px  P95={pct(0.95):.2f}px")
    gross = 100.0 * sum(1 for d in ds if d >= 20) / n
    near = 100.0 * sum(1 for d in ds if d <= 32) / n  # within the +-32 window
    print(f"  gross (>=20px) = {gross:.2f}%   within-window (<=32px) = {near:.2f}%")
    # mean if the gross tail were perfectly refined to <=3px (headroom proxy)
    capped = [min(d, 3.0) for d in ds]
    print(f"  mean if all coarse-hits refined to <=3px = {sum(capped)/n:.2f}px (proxy)")


if __name__ == "__main__":
    main()
