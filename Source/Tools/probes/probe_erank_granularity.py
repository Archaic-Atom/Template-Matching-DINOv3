# -*- coding: utf-8 -*-
"""Effective rank across granularity on ONE ruler.

把 22(粗对齐) / 48 / 64(细匹配) 放到同一个估计量 = effective rank 上:测匹配表征
的 erank 是否随粒度上升(stride-8 vs stride-4),并复现对齐侧的 ~22 作 sanity。

若 match-side erank ≈ 48/64 → 22/48/64 三点同尺单调,供需框架都省;
否则(erank 明显偏离 recovery rank) → erank≠recovery,回落"供给/需求"叙事。

同一份 B@e4 特征(与 48/64 的 recovery 数同源)。报告两种 erank 定义
(Roy2007 熵指数 + participation ratio),由 align_sar 的 sanity 决定哪种对齐论文的 22。
"""
import sys

import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image, ImageReadMode

sys.path.insert(0, 'Source')
from Tools.eval_stage1_matching import load_model, preprocess_image  # noqa: E402
from UserModelImplementation.Models.builder import build_model_from_yaml  # noqa: E402

DEV = torch.device('cuda')
N = 200
CSV = 'Datasets/test_256_2k.csv'
CFG_B = 'Source/UserModelImplementation/Models/TM_Model_B/config.yaml'
CKPT_B = 'Checkpoint_B_256/model_epoch_4.pth'
CFG_A = 'Source/UserModelImplementation/Models/TM_Model_A/config.yaml'
CKPT_A = 'Checkpoint_A_stage1_256/model_epoch_5.pth'
SEED = 2024


class _Stage2Args:
    is_stage_2 = True
    pre_trained = True
    dist = False
    root_path = '.'
    modelDir = '.'
    gpu = 1


class Cov:
    """Incremental mean + covariance accumulator over feature vectors."""

    def __init__(self, d):
        self.n = 0
        self.s = np.zeros(d, np.float64)
        self.ss = np.zeros((d, d), np.float64)

    def add(self, X):  # X: (m, d)
        self.n += X.shape[0]
        self.s += X.sum(0)
        self.ss += X.T @ X

    def erank(self):
        mu = self.s / self.n
        cov = self.ss / self.n - np.outer(mu, mu)
        w = np.linalg.eigvalsh(cov)
        w = w[w > 1e-10]
        sig = np.sqrt(w)
        p = sig / sig.sum()
        er_entropy = float(np.exp(-(p * np.log(p)).sum()))       # Roy2007
        pr = float(w.sum() ** 2 / (w ** 2).sum())                # participation ratio
        return er_entropy, pr, len(w)


def feats_np(t):
    """(1,C,h,w) -> (h*w, C) float64."""
    _, c, h, w = t.shape
    return t.reshape(c, h * w).t().cpu().numpy().astype(np.float64)


def main():
    torch.manual_seed(SEED)
    df = pd.read_csv(CSV)
    n = min(N, len(df))

    print('loading Stage-1 aligned model (A) ...', flush=True)
    mA = load_model(CKPT_A, CFG_A, DEV)
    print('loading matcher B ...', flush=True)
    mB = build_model_from_yaml(CFG_B, _Stage2Args())
    ckB = torch.load(CKPT_B, map_location='cpu', weights_only=False)
    miss, unexp = mB.trainable_tail.load_state_dict(ckB.get('model_0', ckB), strict=False)
    print(f'  B loaded: missing={len(miss)} unexpected={len(unexp)}', flush=True)
    mB = mB.to(DEV).eval()
    mh = mB.matching

    accs = {}

    def acc(name, X):
        if name not in accs:
            accs[name] = Cov(X.shape[1])
        accs[name].add(X)

    for i in range(n):
        r = df.iloc[i]
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            outA = mA(sar, rgb)                              # Stage-1 aligned
            acc('align_sar', feats_np(outA['sar_feat']['sar16']))
            f_rgb = mB.rgb_encoder(rgb)[-1]                  # matcher B stride-16 feats
            f_sar = mB.sar_encoder(sar)['sar16']
            acc('match_sar_str8', feats_np(mh.proj_sar_8(f_sar, sar)))
            acc('match_sar_str4', feats_np(mh.proj_sar_4(f_sar, sar)))
            acc('match_rgb_str8', feats_np(mh.proj_rgb_8(f_rgb, rgb)))
            acc('match_rgb_str4', feats_np(mh.proj_rgb_4(f_rgb, rgb)))
        if i == 0:
            print('  feat dims: '
                  f"align_sar={accs['align_sar'].s.shape[0]} "
                  f"match_sar_str4={accs['match_sar_str4'].s.shape[0]}", flush=True)
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{n}', flush=True)

    print('\n==== effective rank by granularity (ONE ruler) ====', flush=True)
    print(f'  {"representation":18s}  {"erank(Roy2007)":>14s}  {"partic.ratio":>12s}  dim/n', flush=True)
    for name in ['align_sar', 'match_sar_str8', 'match_sar_str4',
                 'match_rgb_str8', 'match_rgb_str4']:
        er, pr, d = accs[name].erank()
        print(f'  {name:18s}  {er:14.2f}  {pr:12.2f}   d={d} n={accs[name].n}', flush=True)
    print('\n  sanity: align_sar should reproduce the paper\'s ~22 for whichever column matches.', flush=True)
    print('  read: does match_*_str8 < match_*_str4, and how do they compare to 48 / 64?', flush=True)
    print('\nERANK_DONE', flush=True)


if __name__ == '__main__':
    main()
