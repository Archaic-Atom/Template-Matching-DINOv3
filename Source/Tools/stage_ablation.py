# -*- coding: utf-8 -*-
"""Table VII: cascade stage-wise readout (stride-4 -> coarse -> stride-2 -> final).

级联逐级消融:同一 cascade(ep2)在各级的预测精度(填 Table VII)。各级坐标重建
镜像 ``_interface_base.accuracy()`` 的 a3/a2/a1/a0。final 用 decode_primary(与
eval CSV 一致 → 自校验:final MEE 应 ~1.1px @2k)。纯推理,无训练。
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
from UserModelImplementation.Models._components.decode import decode_primary  # noqa: E402


class _Args:
    is_stage_2 = True
    pre_trained = True
    dist = False
    root_path = '.'
    modelDir = '.'
    gpu = 1
    mode = 'test'


DEV = torch.device('cuda')
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_C/config.yaml'
CKPT = 'Checkpoint_C_phase2_8gpu/model_epoch_2.pth'
CSV = 'Datasets/test_256_2k.csv'
N = 2000


def argmax_xy(hm):
    """Hard argmax of (B,1,H,W) -> (B,2) (x,y) cells. 硬 argmax → 单元坐标。"""
    b = hm.shape[0]; w = hm.shape[-1]
    idx = hm.reshape(b, -1).argmax(1)
    return torch.stack([idx % w, idx // w], 1)


def coarse_s4(out):
    """Fused coarse argmax in stride-4 cells (mirrors _coarse_argmax_cells_stride4)."""
    hps = out['heatmap_per_stride']; h4 = hps[4]; h8 = hps.get(8)
    if h8 is None:
        hf = h4
    else:
        fw = out['fusion_weights']  # already softmaxed in out
        h8u = F.interpolate(h8, size=h4.shape[-2:], mode='bilinear', align_corners=False)
        hf = fw[0] * h4 + fw[1] * h8u
    return argmax_xy(hf)


def main():
    model = build_model_from_yaml(CONFIG, _Args())
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model.trainable_tail.load_state_dict(ck.get('model_0', ck), strict=False)
    model = model.to(DEV).eval()

    df = pd.read_csv(CSV); n = min(N, len(df))
    stages = {'stride-4': [], 'coarse': [], 'stride-2': [], 'final': []}
    for i in range(n):
        r = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(r['Label']))
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            out = model(sar, rgb)                                    # cascade out dict
            pf, _ = decode_primary(out, argmax_type='hard')          # final pixel (a0)
            xy4 = argmax_xy(out['heatmap_per_stride'][4]).float() * 4.0   # stride-4 (a3)
            uvc = coarse_s4(out).float() * 4.0                        # coarse fused (a2)
            a2 = out['cascade_anchor_2_cells'].float() * 2.0
            xy2 = argmax_xy(out['heatmap_cascade'][2]).float() * 2.0
            ps2 = a2 + xy2                                            # stride-2 (a1)
        gt = torch.tensor([[gx, gy]], dtype=torch.float32, device=DEV)
        for name, p in [('stride-4', xy4), ('coarse', uvc),
                        ('stride-2', ps2), ('final', pf.float())]:
            stages[name].append(float(((p - gt) ** 2).sum(1).sqrt().item()))
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{n}', flush=True)

    print("\nstage      CMR(1)  CMR(3)   MEE   MdEE")
    for name in ['stride-4', 'coarse', 'stride-2', 'final']:
        e = np.array(stages[name])
        print(f"{name:9s}  {100*np.mean(e <= 1):6.2f} {100*np.mean(e <= 3):6.2f} "
              f"{e.mean():6.2f} {np.median(e):6.2f}")
    print("(final should match eval CSV: MEE ~1.1px, CMR(3) ~95.7 @2k — self-check)")


if __name__ == '__main__':
    main()
