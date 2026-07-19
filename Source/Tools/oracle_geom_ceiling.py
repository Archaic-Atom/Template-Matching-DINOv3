# -*- coding: utf-8 -*-
"""Geometry-ceiling probe: optical-optical (DINO) self-matching heatmap at 256/192.

几何天花板探针:用同一个冻结 DINO 对"光学 GT 裁块 vs 光学搜索图"做同模态匹配,
量出在 192/256 stride-16 下,理论上限(完美 SAR=DINO)能不能判别对。
两种 template:
  (A) separate-encode: 把 192 GT 裁块单独过 DINO(和 SAR 模板一样单独编码)→ 公平 oracle
  (B) in-context: 直接从 256 特征图切 GT 12x12 子窗(cos_gt 恒=1,只看远阴性有多平)
对比基准:SAR ep1 separation = top1(<=1cell) 0.597, d' 0.03。
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

DEVICE = torch.device('cuda')
STRIDE = 16
TSIZE = 192
FAR = 2
N = 300
CKPT = 'Checkpoint_A_stage1_256/model_epoch_1.pth'
CFG = 'Source/UserModelImplementation/Models/TM_Model_A/config.yaml'
CSV = 'Datasets/test_256_2k.csv'


def dino_feat(model, img_chw):
    """DINO feature map for a [3,H,W] uint8 image via model.rgb_encoder path."""
    rgb = preprocess_image(img_chw).unsqueeze(0).to(DEVICE)
    # model(sar, rgb) needs a sar input; we only use the rgb branch output.
    dummy_sar = torch.zeros(1, 3, TSIZE, TSIZE, device=DEVICE)
    with torch.no_grad():
        out = model(dummy_sar, rgb)
    return out['rbg_feat'][3]  # [1, D, h, w]


def heat(search_n, tmpl_n):
    th, tw = tmpl_n.shape[-2:]
    return (F.conv2d(search_n, tmpl_n) / (th * tw)).squeeze(0).squeeze(0)


def stats(rows, tag):
    r = pd.DataFrame(rows)
    cd = r['cell_dist']
    print(f'[{tag}] n={len(r)} top1(<=1cell)={(cd <= 1).mean():.4f} '
          f'top1(exact)={(cd == 0).mean():.4f} '
          f'cos_gt={r.cos_gt.mean():.4f} cos_far={r.cos_far.mean():.4f} '
          f"gt>far={ (r.cos_gt > r.cos_far).mean():.3f} "
          f"d'={(r.cos_gt.mean()-r.cos_far.mean())/np.sqrt(0.5*(r.cos_gt.var()+r.cos_far.var())+1e-9):.3f}",
          flush=True)


def main():
    print(f'Loading model {CKPT}', flush=True)
    model = load_model(CKPT, CFG, DEVICE)
    df = pd.read_csv(CSV, usecols=[0, 1, 2])
    rows_sep, rows_ctx = [], []
    n = min(N, len(df))
    for i in range(n):
        row = df.iloc[i]
        rgb_path = row.iloc[1]
        gt = json.loads(row.iloc[2])
        gx, gy = int(gt[0]), int(gt[1])
        gxf, gyf = gx // STRIDE, gy // STRIDE
        rgb = read_image(rgb_path, ImageReadMode.RGB)           # [3,256,256]
        search = F.normalize(dino_feat(model, rgb), dim=1)      # [1,D,16,16]
        fh, fw = search.shape[-2:]

        # (B) in-context GT window (cos_gt trivially ~1)
        tw_ = TSIZE // STRIDE
        ctx_tmpl = search[:, :, gyf:gyf + tw_, gxf:gxf + tw_]
        if ctx_tmpl.shape[-1] == tw_ and ctx_tmpl.shape[-2] == tw_:
            s = heat(search, ctx_tmpl)
            _rec(rows_ctx, s, gxf, gyf, FAR)

        # (A) separate-encode optical 192 crop (fair oracle)
        try:
            crop = rgb[:, gy:gy + TSIZE, gx:gx + TSIZE]          # [3,192,192]
            if crop.shape[-1] == TSIZE and crop.shape[-2] == TSIZE:
                tmpl = F.normalize(dino_feat(model, crop), dim=1)
                s = heat(search, tmpl)
                _rec(rows_sep, s, gxf, gyf, FAR)
        except Exception as e:  # noqa: BLE001
            if i == 0:
                print(f'separate-encode path failed ({e}); using in-context only', flush=True)

    print('=== GEOMETRY-CEILING ORACLE (256/192 stride16, frozen DINO) ===', flush=True)
    if rows_sep:
        stats(rows_sep, 'A optical-optical separate-encode (FAIR oracle)')
    if rows_ctx:
        stats(rows_ctx, 'B in-context GT-window (cos_gt~1, see cos_far flatness)')
    print('compare: SAR ep1 -> top1(<=1cell)=0.597 top1(exact)=0.165 d`=0.031', flush=True)
    print('ORACLE_DONE', flush=True)


def _rec(rows, score, gxf, gyf, far):
    fh, fw = score.shape
    flat = int(score.flatten().argmax().item())
    pxf, pyf = flat % fw, flat // fw
    cell_dist = max(abs(pxf - gxf), abs(pyf - gyf))
    cos_gt = score[gyf, gxf].item() if (0 <= gyf < fh and 0 <= gxf < fw) else float('nan')
    ys = torch.arange(fh).view(-1, 1)
    xs = torch.arange(fw).view(1, -1)
    dist = torch.maximum((ys - gyf).abs(), (xs - gxf).abs())
    far_mask = dist > far
    cos_far = score[far_mask].max().item() if far_mask.any() else float('nan')
    if not np.isnan(cos_gt) and not np.isnan(cos_far):
        rows.append({'cell_dist': cell_dist, 'cos_gt': cos_gt, 'cos_far': cos_far})


if __name__ == '__main__':
    main()
