# -*- coding: utf-8 -*-
"""Plot the 2x2 frozen-arms (双臂) result for the experiments section.

绘制论文实验部分的"双臂"(冻结编码器)对照实验曲线图。

Two panels, from the 20,000-template unbiased subsample (seed 2024) of the
SEN1-2 256/192 test set and its paired bootstrap (B=10^4, seed 2024):
  (a) CMR(k) vs. threshold k for the three encoders that share architecture,
      loss, optimizer, and schedule and differ only in encoder
      initialization x trainability:
        R/Tr  (B)   = random init, trainable encoder
        S1/Fr (S1F) = Stage-1 pre-aligned, frozen encoder
        R/Fr  (RF)  = random init, frozen encoder (control)
  (b) Paired-bootstrap ΔCMR(k) with 95% CIs for the three contrasts, showing
      S1F≈B (CI spans 0 at every k) >> RF (CIs exclude 0).

Data are hard-coded from results_archive/frozen_arms_e4_2026-06-14/
(paired_bootstrap_e4_20k.txt) so the figure is reproducible without the CSVs.

(a) 三臂的 CMR(k)-阈值曲线;(b) 配对 bootstrap 的 ΔCMR 及 95% 置信区间。
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- output -----------------------------------------------------------------
PAPER_IMG = (
    "/Users/rhc/Documents/001_Scholar/001_Ph.D./002_MyPaper/026_TMMatching/"
    "000_workspace/000_tgrs2026/000_paper/img/experiments"
)
OUT_PDF = os.path.join(PAPER_IMG, "fig_frozen_arms.pdf")

# --- data: 20k unbiased subsample, paired bootstrap (verified) --------------
# 阈值 k(像素)。
K = np.array([1, 2, 3, 5, 10], dtype=float)

# CMR(k) point estimates (%).  键=臂名。
CMR = {
    "R/Tr (B, trainable)":        [68.28, 81.88, 86.35, 89.41, 91.33],
    "S1/Fr (aligned, frozen)":    [68.63, 81.73, 86.30, 89.38, 91.34],
    "R/Fr (random, frozen)":      [66.94, 80.38, 85.17, 88.28, 90.41],
}

# Paired-bootstrap ΔCMR(k): mean, ci_lo, ci_hi  (percentage points).
DELTA = {
    "S1/Fr - R/Fr": {
        "mean":  [1.69, 1.36, 1.13, 1.10, 0.94],
        "lo":    [1.09, 0.93, 0.78, 0.78, 0.63],
        "hi":    [2.28, 1.78, 1.51, 1.43, 1.26],
    },
    "S1/Fr - R/Tr": {
        "mean":  [0.35, -0.15, -0.05, -0.03, 0.02],
        "lo":    [-0.25, -0.56, -0.40, -0.33, -0.27],
        "hi":    [0.94, 0.26, 0.30, 0.28, 0.33],
    },
    "R/Fr - R/Tr": {
        "mean":  [-1.34, -1.51, -1.18, -1.13, -0.92],
        "lo":    [-1.94, -1.94, -1.55, -1.45, -1.23],
        "hi":    [-0.73, -1.08, -0.82, -0.80, -0.59],
    },
}

# --- IEEE-ish style ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,  # editable text in the PDF
    "font.family": "serif",
})

C_B = "#1b6ca8"    # blue  - trainable
C_S1F = "#d1495b"  # red   - aligned frozen
C_RF = "#8a8d91"   # grey  - random frozen control

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.7))

# ---- Panel (a): CMR(k) vs k ----
styles = {
    "R/Tr (B, trainable)":     dict(color=C_B, marker="o", ls="-"),
    "S1/Fr (aligned, frozen)": dict(color=C_S1F, marker="s", ls="--"),
    "R/Fr (random, frozen)":   dict(color=C_RF, marker="^", ls=":"),
}
for name, vals in CMR.items():
    axa.plot(K, vals, label=name, **styles[name])
axa.set_xscale("log")
axa.set_xticks(K)
axa.set_xticklabels([int(k) for k in K])
axa.minorticks_off()
axa.set_xlabel(r"threshold $k$ (px)")
axa.set_ylabel(r"CMR($k$) (%)")
axa.set_title("(a) Correct-match rate vs. threshold")
axa.legend(loc="lower right", framealpha=0.9)
axa.set_ylim(64, 93)

# ---- Panel (b): bootstrap ΔCMR with 95% CI ----
dcolors = {"S1/Fr - R/Fr": C_S1F, "S1/Fr - R/Tr": "#5a5a5a", "R/Fr - R/Tr": C_RF}
dmark = {"S1/Fr - R/Fr": "s", "S1/Fr - R/Tr": "D", "R/Fr - R/Tr": "^"}
offsets = {"S1/Fr - R/Fr": -0.08, "S1/Fr - R/Tr": 0.0, "R/Fr - R/Tr": 0.08}
xpos = np.arange(len(K))
for name, d in DELTA.items():
    mean = np.array(d["mean"]); lo = np.array(d["lo"]); hi = np.array(d["hi"])
    yerr = np.vstack([mean - lo, hi - mean])
    axb.errorbar(xpos + offsets[name], mean, yerr=yerr, fmt=dmark[name],
                 color=dcolors[name], capsize=2.5, elinewidth=1.0, label=name)
axb.axhline(0.0, color="k", lw=0.8, ls="-")
axb.set_xticks(xpos)
axb.set_xticklabels([int(k) for k in K])
axb.set_xlabel(r"threshold $k$ (px)")
axb.set_ylabel(r"$\Delta$CMR (points)")
axb.set_title(r"(b) Paired-bootstrap $\Delta$CMR (95% CI)")
axb.legend(loc="upper right", framealpha=0.9)

fig.tight_layout(pad=0.6)
os.makedirs(PAPER_IMG, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
print("wrote", OUT_PDF)
