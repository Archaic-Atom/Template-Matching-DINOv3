# -*- coding: utf-8 -*-
"""Coarse-only matching eval for the frozen arms: S1F-e0 vs RF-e0 vs B-e0/e4.

Bypasses the (warmup/untrained) fine head: pulls the stage-2 coarse heatmaps
output['heatmap_per_stride'][4]/[8], fuses via model.fusion_logits, argmax -> ×4
= coarse predicted (x,y) top-left. Compares to GT. Coarse is stride-4 quantized
so PCK@1 is meaningless; report a threshold sweep + top1 cell metrics.

冻结双臂的 coarse-only 评测：绕过没训练的 fine head，只比 coarse 融合热图。
四个 ckpt 一次跑完：S1F-e0 / RF-e0（冻结双臂）、B-e0（同预算可训练参照）、
B-e4（收敛 direct-training 上界）。阈值扫描用于检验"颗粒度签名"。

Usage (from PROJ root):
    CUDA_VISIBLE_DEVICES=1 python Source/Tools/coarse_eval_frozen_e0.py
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode

sys.path.insert(0, 'Source')
from Tools.eval_stage1_matching import preprocess_image  # noqa: E402
from UserModelImplementation.Models.builder import build_model_from_yaml  # noqa: E402


class _Stage2Args:
    """Build args forcing is_stage_2=True so the matching head is constructed."""
    is_stage_2 = True
    pre_trained = True
    dist = False
    root_path = '.'
    modelDir = '.'
    gpu = 1


def load_stage2_model(config_path, ckpt_path, device):
    model = build_model_from_yaml(config_path, _Stage2Args())
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_0', ckpt)
    missing, unexpected = model.trainable_tail.load_state_dict(state, strict=False)
    print(f'  loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    return model.to(device).eval()


DEV = torch.device('cuda')
STRIDE = 4
N = 2000
CSV = 'Datasets/test_256_2k.csv'


def fusion_logits_of(model):
    for attr in ('fusion_logits',):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, 'matching') and hasattr(model.matching, 'fusion_logits'):
        return model.matching.fusion_logits
    return None


def coarse_pred(model, fl, sar_path, rgb_path):
    sar = preprocess_image(read_image(sar_path, ImageReadMode.GRAY), duplicate_channels=True).unsqueeze(0)
    rgb = preprocess_image(read_image(rgb_path, ImageReadMode.RGB)).unsqueeze(0)
    with torch.no_grad():
        out = model(sar.to(DEV), rgb.to(DEV))
    H4 = out['heatmap_per_stride'][4]
    H8 = out['heatmap_per_stride'][8]
    if fl is not None:
        a = torch.softmax(fl, dim=0)
        H8u = F.interpolate(H8, size=H4.shape[-2:], mode='bilinear', align_corners=False)
        Hf = a[0] * H4 + a[1] * H8u
    else:
        Hf = H4
    h, w = Hf.shape[-2:]
    idx = int(Hf.view(-1).argmax().item())
    v, u = idx // w, idx % w
    return u * STRIDE, v * STRIDE, (u, v), (w, h)


def eval_ckpt(config, ckpt, tag):
    print(f"== loading {tag}: {ckpt} (config {config}) ==", flush=True)
    model = load_stage2_model(config, ckpt, DEV)
    fl = fusion_logits_of(model)
    df = pd.read_csv(CSV)
    rows = []
    for i in range(min(N, len(df))):
        r = df.iloc[i]
        gt = json.loads(r['Label'])
        gx, gy = int(gt[0]), int(gt[1])
        px, py, cell, hw = coarse_pred(model, fl, r['SAR_Image'], r['RGB_Image'])
        if i == 0:
            print(f"  [debug {tag}] H_fused hw={hw} fusion_logits={None if fl is None else fl.detach().cpu().numpy()} "
                  f"sample0 gt=({gx},{gy}) coarse_pred=({px},{py}) cell={cell}", flush=True)
        err = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
        gcx, gcy = gx // STRIDE, gy // STRIDE
        cell_dist = max(abs(cell[0] - gcx), abs(cell[1] - gcy))
        rows.append((err, cell_dist))
    e = np.array([x[0] for x in rows]); cd = np.array([x[1] for x in rows])
    # Threshold sweep for the granularity signature / 颗粒度签名的阈值扫描
    sweep = "  ".join(f"PCK@{k}={100*(e<k).mean():.2f}%" for k in (2, 4, 5, 8, 12, 16))
    print(f"[{tag}] n={len(e)} COARSE mean={e.mean():.2f}px median={np.median(e):.2f}px\n"
          f"  {sweep}\n"
          f"  top1(<=1cell)={(cd<=1).mean():.4f} top1(exact)={(cd==0).mean():.4f} "
          f"within32={100*(e<=32).mean():.1f}%", flush=True)
    del model
    torch.cuda.empty_cache()


# Frozen-arm epoch is parametrizable so the same script tracks the e0/e1/...
# trajectory; B references stay fixed (e0 same-budget, e4 converged upper bound).
# 冻结臂的 epoch 可由命令行参数指定,用同一脚本追踪 e0/e1/... 轨迹;
# B 参照固定(e0 同预算,e4 收敛上界)。
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 0

EVALS = [
    ('Source/UserModelImplementation/Models/TM_Model_F/config.yaml',
     f'Checkpoint_F_s2_frozen_256/model_epoch_{EP}.pth', f'S1F-e{EP}'),
    ('Source/UserModelImplementation/Models/TM_Model_F/config.yaml',
     f'Checkpoint_RF_s2_frozen_256/model_epoch_{EP}.pth', f'RF-e{EP}'),
    ('Source/UserModelImplementation/Models/TM_Model_B/config.yaml',
     'Checkpoint_B_256/model_epoch_0.pth', 'B-e0'),
    ('Source/UserModelImplementation/Models/TM_Model_B/config.yaml',
     'Checkpoint_B_256/model_epoch_4.pth', 'B-e4'),
]

for cfg, ck, tag in EVALS:
    eval_ckpt(cfg, ck, tag)
print('COARSE_FROZEN_DONE', flush=True)
