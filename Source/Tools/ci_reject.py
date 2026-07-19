# -*- coding: utf-8 -*-
"""Bootstrap 95% CIs + reject-option curve from a TM result CSV.

从 TM 结果 CSV 计算 bootstrap 95% 置信区间(Table I/CI)与 reject-option
(选择性预测)曲线(reject-option 表)。用已有的每样本 CSV,无需重跑模型。

CSV cols: index,sar_name,rgb_name,gt_x,gt_y,pred_x,pred_y,score
- Bootstrap CI of a proportion = binomial resampling (exact, fast, tiny memory):
  CMR(k) 的 bootstrap = 二项重采样(精确、快、省内存)。
- Reject-option: rank by score desc, abstain on the lowest fraction, report retained CMR(3).
  按 score 降序,弃掉最低置信的一部分,报保留集的 CMR(3)。

Usage: python ci_reject.py <label> <result.csv>
"""
import csv
import math
import sys

import numpy as np

K = [1, 3, 5]                      # CMR 阈值(像素)
ABSTAIN = [0.0, 0.10, 0.25, 0.50]  # 弃答率 / abstention rates
B = 10000                          # bootstrap 次数
SEED = 2024


def load(path):
    """Return (l2 errors, scores) as numpy arrays. 返回 L2 误差与 score。"""
    l2, sc = [], []
    for r in csv.DictReader(open(path)):
        dx = float(r["pred_x"]) - float(r["gt_x"])
        dy = float(r["pred_y"]) - float(r["gt_y"])
        l2.append(math.hypot(dx, dy)); sc.append(float(r["score"]))
    return np.array(l2), np.array(sc)


def cmr(l2, k):
    """CMR(k) in %. 命中率(%)。"""
    return 100.0 * np.mean(l2 <= k)


def boot_ci(l2, k, rng):
    """95% bootstrap CI for CMR(k) via binomial resampling. 二项重采样 95%CI。"""
    n = len(l2)
    phat = np.mean(l2 <= k)
    draws = rng.binomial(n, phat, size=B) / n * 100.0
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def main():
    label, path = sys.argv[1], sys.argv[2]
    l2, sc = load(path)
    n = len(l2)
    rng = np.random.RandomState(SEED)

    print(f"=== {label}  (n={n}) ===")
    # --- bootstrap CIs ---
    for k in K:
        p = cmr(l2, k); lo, hi = boot_ci(l2, k, rng)
        print(f"  CMR({k}) = {p:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
    # --- reject-option: rank by score desc, abstain lowest fraction ---
    order = np.argsort(-sc)          # high score first / 高 score 在前
    l2_sorted = l2[order]
    print("  reject-option (retained CMR(3) at abstention):")
    for a in ABSTAIN:
        keep = int(round(n * (1.0 - a)))
        val = 100.0 * np.mean(l2_sorted[:keep] <= 3)
        print(f"    abstain {int(a*100):2d}%  ->  CMR(3) = {val:.2f}  (n_keep={keep})")


if __name__ == "__main__":
    main()
