# -*- coding: utf-8 -*-
"""erank (participation ratio) of the SAR FEATURE EXTRACTOR across training states.

全部在特征提取阶段:量各 checkpoint 的 SAR encoder stride-16 输出的参与比 erank,
回答"锁定(对齐冻结) vs 不锁定(微调/从头匹配/随机)"的 erank 各是多少。

States (all measured on the RAW sar_encoder stride-16 output = 特征提取输出):
  init_random     : matcher arch, 不加载训练 ckpt (SAR encoder 初始态)
  aligned_rawC4   : Stage-1 对齐模型的 sar_encoder 原始 C4 (对齐投影之前)
  aligned_output  : Stage-1 对齐后的 sar_feat['sar16'] (对齐投影之后 = 论文的 22 参照)
  aft_finetuned   : 对齐后再微调 (Checkpoint_A_ft_256) 的 sar_encoder 原始 C4
  matcherB        : 从头训练匹配 (Checkpoint_B_256) 的 sar_encoder 原始 C4

对照:若 aligned_rawC4 也≈22 → 22 长在编码器里;若 rawC4 高、只有 aligned_output≈22
→ 22 是对齐投影压出来的(不是特征提取固有)。参与比列由 aligned_output≈22 校准。
"""
import glob
import os
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
CFG_A = 'Source/UserModelImplementation/Models/TM_Model_A/config.yaml'
CKPT_A_S1 = 'Checkpoint_A_stage1_256/model_epoch_5.pth'
DIR_AFT = 'Checkpoint_A_ft_256'
CKPT_B = 'Checkpoint_B_256/model_epoch_4.pth'
SEED = 2024


class _Stage2Args:
    is_stage_2 = True
    pre_trained = True
    dist = False
    root_path = '.'
    modelDir = '.'
    gpu = 1


class Cov:
    """Incremental mean + covariance -> effective rank (participation ratio)."""

    def __init__(self, d):
        self.n = 0
        self.s = np.zeros(d, np.float64)
        self.ss = np.zeros((d, d), np.float64)

    def add(self, X):
        self.n += X.shape[0]
        self.s += X.sum(0)
        self.ss += X.T @ X

    def erank(self):
        mu = self.s / self.n
        cov = self.ss / self.n - np.outer(mu, mu)
        w = np.linalg.eigvalsh(cov)
        w = w[w > 1e-10]
        pr = float(w.sum() ** 2 / (w ** 2).sum())            # participation ratio
        sig = np.sqrt(w)
        p = sig / sig.sum()
        roy = float(np.exp(-(p * np.log(p)).sum()))          # Roy2007 entropy
        return pr, roy, len(w)


def feats_np(t):
    _, c, h, w = t.shape
    return t.reshape(c, h * w).t().cpu().numpy().astype(np.float64)


def latest_ckpt(d):
    fs = glob.glob(os.path.join(d, 'model_epoch_*.pth'))
    if not fs:
        return None
    return sorted(fs, key=lambda p: int(p.split('_')[-1].split('.')[0]))[-1]


def build_matcher(cfg, ckpt):
    """Build a Stage-2 matcher; load ckpt if given (else random/init state)."""
    m = build_model_from_yaml(cfg, _Stage2Args())
    if ckpt is not None:
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        miss, unexp = m.trainable_tail.load_state_dict(ck.get('model_0', ck), strict=False)
        print(f'    loaded {ckpt}: missing={len(miss)} unexpected={len(unexp)}', flush=True)
    return m.to(DEV).eval()


def measure(name, build_fn, extract_fns, df, n):
    """Load one model, accumulate erank for each named extractor, then free."""
    try:
        model = build_fn()
    except Exception as e:  # noqa: BLE001
        print(f'  [SKIP] {name}: build failed: {e}', flush=True)
        return {}
    accs = {}
    for i in range(n):
        r = df.iloc[i]
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            for key, fn in extract_fns.items():
                try:
                    X = feats_np(fn(model, sar, rgb))
                except Exception as e:  # noqa: BLE001
                    if i == 0:
                        print(f'  [SKIP] {name}:{key}: extract failed: {e}', flush=True)
                    continue
                if key not in accs:
                    accs[key] = Cov(X.shape[1])
                accs[key].add(X)
    del model
    torch.cuda.empty_cache()
    out = {}
    for key, cov in accs.items():
        out[key] = cov.erank()
    return out


def raw_c4(model, sar, rgb):
    return model.sar_encoder(sar)['sar16']


def aligned_out(model, sar, rgb):
    return model(sar, rgb)['sar_feat']['sar16']


def main():
    torch.manual_seed(SEED)
    df = pd.read_csv(CSV)
    n = min(N, len(df))
    results = {}

    print('== init_random (matcher arch, no trained ckpt) ==', flush=True)
    results.update({f'init_random.{k}': v for k, v in
                    measure('init_random', lambda: build_matcher(CFG_B, None),
                            {'rawC4': raw_c4}, df, n).items()})

    print('== aligned (Stage-1, locked target) ==', flush=True)
    results.update({f'aligned.{k}': v for k, v in
                    measure('aligned', lambda: load_model(CKPT_A_S1, CFG_A, DEV),
                            {'rawC4': raw_c4, 'output': aligned_out}, df, n).items()})

    print('== aft_finetuned (aligned then finetuned) ==', flush=True)
    aft = latest_ckpt(DIR_AFT)
    print(f'   A-ft ckpt = {aft}', flush=True)
    results.update({f'aft.{k}': v for k, v in
                    measure('aft', lambda: build_matcher(CFG_A, aft),
                            {'rawC4': raw_c4}, df, n).items()})

    print('== matcherB (matching from scratch) ==', flush=True)
    results.update({f'matcherB.{k}': v for k, v in
                    measure('matcherB', lambda: build_matcher(CFG_B, CKPT_B),
                            {'rawC4': raw_c4}, df, n).items()})

    print('\n==== erank of SAR feature extractor (participation ratio) ====', flush=True)
    print(f'  {"state":26s}  {"partic.ratio":>12s}  {"Roy2007":>8s}  dim', flush=True)
    order = ['init_random.rawC4', 'aligned.rawC4', 'aligned.output',
             'aft.rawC4', 'matcherB.rawC4']
    for k in order:
        if k in results:
            pr, roy, d = results[k]
            print(f'  {k:26s}  {pr:12.2f}  {roy:8.2f}  d={d}', flush=True)
        else:
            print(f'  {k:26s}  (missing)', flush=True)
    print('\n  LOCKED = aligned.* ; NOT-LOCKED = aft / matcherB / init_random', flush=True)
    print('  sanity: aligned.output partic.ratio should be ~22.', flush=True)
    print('\nERANK_EXTRACTORS_DONE', flush=True)


if __name__ == '__main__':
    main()
