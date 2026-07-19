# -*- coding: utf-8 -*-
"""Re-score ep1 SAR features: pure-cosine vs magnitude-weighted heatmap.

对已训好的 ep1(normalize_logits=True 训出来的)SAR 特征,eval 时用两种打分:
  COS  : conv(normalize(rgb), normalize(sar))           —— 当前 separation 用的(d'~0.03)
  MAG  : conv(rgb_RAW,        normalize(sar))            —— 未归一化 logit = cos*||r||,保留 DINO 模长
看 ||r|| 显著性线索能否在 eval 时把判别找回来。若 MAG 的 d'/win 明显更高 → 训练时丢掉 ||r|| 是元凶。
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode

sys.path.insert(0, 'Source')
from Tools.eval_stage1_matching import load_model, preprocess_image  # noqa: E402

DEV = torch.device('cuda')
STRIDE, FAR, N = 16, 2, 300
CKPT = 'Checkpoint_A_stage1_256/model_epoch_1.pth'
CFG = 'Source/UserModelImplementation/Models/TM_Model_A/config.yaml'
CSV = 'Datasets/test_256_2k.csv'


def rec(rows, score, gxf, gyf):
    fh, fw = score.shape
    flat = int(score.flatten().argmax().item())
    pxf, pyf = flat % fw, flat // fw
    cd = max(abs(pxf - gxf), abs(pyf - gyf))
    if not (0 <= gyf < fh and 0 <= gxf < fw):
        return
    cos_gt = score[gyf, gxf].item()
    ys = torch.arange(fh).view(-1, 1)
    xs = torch.arange(fw).view(1, -1)
    far = torch.maximum((ys - gyf).abs(), (xs - gxf).abs()) > FAR
    cos_far = score[far].max().item() if far.any() else float('nan')
    if not np.isnan(cos_far):
        rows.append({'cd': cd, 'gt': cos_gt, 'far': cos_far})


def report(rows, tag):
    r = pd.DataFrame(rows)
    g, f, cd = r['gt'], r['far'], r['cd']
    dp = (g.mean() - f.mean()) / np.sqrt(0.5 * (g.var() + f.var()) + 1e-9)
    print(f"[{tag}] n={len(r)} top1(<=1cell)={(cd<=1).mean():.4f} top1(exact)={(cd==0).mean():.4f} "
          f"gt>far={(g>f).mean():.3f} d'={dp:.3f}", flush=True)


def main():
    model = load_model(CKPT, CFG, DEV)
    df = pd.read_csv(CSV, usecols=[0, 1, 2])
    rc, rm = [], []
    for i in range(min(N, len(df))):
        row = df.iloc[i]
        sar = preprocess_image(read_image(row.iloc[0], ImageReadMode.GRAY), duplicate_channels=True).unsqueeze(0)
        rgb = preprocess_image(read_image(row.iloc[1], ImageReadMode.RGB)).unsqueeze(0)
        gt = json.loads(row.iloc[2])
        gxf, gyf = int(gt[0]) // STRIDE, int(gt[1]) // STRIDE
        with torch.no_grad():
            out = model(sar.to(DEV), rgb.to(DEV))
        sar_f = out['sar_feat']['sar16']
        rgb_f = out['rbg_feat'][3]
        sar_n = F.normalize(sar_f, dim=1)
        th, tw = sar_n.shape[-2:]
        cos = (F.conv2d(F.normalize(rgb_f, dim=1), sar_n) / (th * tw)).squeeze(0).squeeze(0)
        mag = (F.conv2d(rgb_f, sar_n) / (th * tw)).squeeze(0).squeeze(0)
        rec(rc, cos, gxf, gyf)
        rec(rm, mag, gxf, gyf)
    print('=== RE-SCORE ep1 (cosine vs magnitude-weighted) ===', flush=True)
    report(rc, 'COS  pure cosine (current)')
    report(rm, 'MAG  cos*||r|| (un-normalized, keeps DINO magnitude)')
    print('RESCORE_DONE', flush=True)


if __name__ == '__main__':
    main()
