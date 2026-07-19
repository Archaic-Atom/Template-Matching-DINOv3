# -*- coding: utf-8 -*-
"""erank across the cascade granularity axis (stride-8 → 4 → 2 → 1), ONE ruler.

在 cascade C (CascadeHeadJBU) 上,量匹配特征在四个粒度 stride-8/4/2/1 的 erank,
看是否随粒度变细而单调上升。所有匹配特征都是 matching_dim=256、L2-归一化,
**同一环境维**,所以跨 stride 的 erank 干净可比(修正了之前拿 stride-16 编码器
冒充粒度轴的错误)。

Hooks the 8 proj/upsample modules to capture each stride's matching features:
  stride-8: proj_sar_8, proj_rgb_8
  stride-4: proj_sar_4, proj_rgb_4
  stride-2: sar_upsample_4to2, rgb_upsample_4to2   (再 L2-norm,和 forward 一致)
  stride-1: proj_sar_1, proj_rgb_1
Reports both erank defs (participation ratio + Roy2007). RGB side is the clean
same-dim (256) comparison across all four strides.
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode

sys.path.insert(0, 'Source')
from Tools.eval_stage1_matching import preprocess_image  # noqa: E402
from UserModelImplementation.Models.builder import build_model_from_yaml  # noqa: E402

DEV = torch.device('cuda')
N = 150
CSV = '/data4/datasets/eval2k_opt256_sar192/image_test_2k_256.csv'
CFG_C = 'Source/UserModelImplementation/Models/TM_Model_C/config.yaml'
CKPT_C = 'Checkpoint_C_direct_sub40_256/model_epoch_2.pth'
SEED = 2024


class _Stage2Args:
    is_stage_2 = True
    pre_trained = True
    dist = False
    root_path = '.'
    modelDir = '.'
    gpu = 1


class Cov:
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
        pr = float(w.sum() ** 2 / (w ** 2).sum())
        sig = np.sqrt(w)
        p = sig / sig.sum()
        roy = float(np.exp(-(p * np.log(p)).sum()))
        return pr, roy, len(w)


def feats_np(t):
    _, c, h, w = t.shape
    return t.reshape(c, h * w).t().cpu().numpy().astype(np.float64)


def main():
    torch.manual_seed(SEED)
    df = pd.read_csv(CSV)
    n = min(N, len(df))

    print(f'building cascade C, loading {CKPT_C} ...', flush=True)
    model = build_model_from_yaml(CFG_C, _Stage2Args())
    ck = torch.load(CKPT_C, map_location='cpu', weights_only=False)
    miss, unexp = model.trainable_tail.load_state_dict(ck.get('model_0', ck), strict=False)
    print(f'  loaded: missing={len(miss)} unexpected={len(unexp)}', flush=True)
    model = model.to(DEV).eval()
    mh = model.matching

    # stride -> (name, module). str2 needs post-hoc L2-norm (upsampler output).
    plan = [
        ('str8_sar', 'proj_sar_8'), ('str8_rgb', 'proj_rgb_8'),
        ('str4_sar', 'proj_sar_4'), ('str4_rgb', 'proj_rgb_4'),
        ('str2_sar', 'sar_upsample_4to2'), ('str2_rgb', 'rgb_upsample_4to2'),
        ('str1_sar', 'proj_sar_1'), ('str1_rgb', 'proj_rgb_1'),
    ]
    captured = {}

    def mk_hook(name):
        def hook(mod, inp, out):
            captured[name] = out.detach()
        return hook

    hooked = []
    for name, attr in plan:
        mod = getattr(mh, attr, None)
        if mod is None:
            print(f'  [SKIP] {name}: module {attr} not found', flush=True)
            continue
        mod.register_forward_hook(mk_hook(name))
        hooked.append(name)

    accs = {}
    for i in range(n):
        r = df.iloc[i]
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)).unsqueeze(0).to(DEV)
        captured.clear()
        with torch.no_grad():
            model(sar, rgb)
        for name in hooked:
            if name not in captured:
                continue
            X = F.normalize(captured[name], p=2, dim=1)     # unit-sphere, consistent
            Xn = feats_np(X)
            if name not in accs:
                accs[name] = Cov(Xn.shape[1])
            accs[name].add(Xn)
        if i == 0:
            dims = {k: captured[k].shape for k in hooked if k in captured}
            print(f'  captured shapes: {dims}', flush=True)
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{n}', flush=True)

    print('\n==== cascade matching-feature erank by stride (all 256-d, L2-normed) ====', flush=True)
    print(f'  {"stride/side":14s}  {"partic.ratio":>12s}  {"Roy2007":>8s}  dim/n', flush=True)
    order = ['str8_sar', 'str4_sar', 'str2_sar', 'str1_sar',
             'str8_rgb', 'str4_rgb', 'str2_rgb', 'str1_rgb']
    for k in order:
        if k in accs:
            pr, roy, d = accs[k].erank()
            print(f'  {k:14s}  {pr:12.2f}  {roy:8.2f}  d={d} n={accs[k].n}', flush=True)
        else:
            print(f'  {k:14s}  (missing)', flush=True)
    print('\n  READ: does erank rise str8 -> str4 -> str2 -> str1 (finer -> higher)?', flush=True)
    print('  RGB side is the cleanest (search maps, all d=256).', flush=True)
    print('\nERANK_CASCADE_DONE', flush=True)


if __name__ == '__main__':
    main()
