# -*- coding: utf-8 -*-
"""Profile the cascade (ep2) for the efficiency table (Table V).

对级联模型(ep2)做效率剖析,填 Table V:各组件参数量(冻结 vs 可训练)、
峰值显存、吞吐(全前向 vs 摊销后每模板)。纯推理,无训练。

Components / 组件:
  rgb_encoder  = frozen DINOv3 ViT-L/16 provider(冻结,每参考图只算一次)
  sar_encoder  = trainable SAR encoder(可训练)
  matching     = trainable matching head(coarse + cascade fine stages,可训练)
"""
import os
import sys
import time

import torch
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode

sys.path.insert(0, 'Source')
from Tools.eval_stage1_matching import preprocess_image  # noqa: E402
from UserModelImplementation.Models.builder import build_model_from_yaml  # noqa: E402


class _Args:
    is_stage_2 = True
    pre_trained = True
    dist = False
    root_path = '.'
    modelDir = '.'
    gpu = 1


DEV = torch.device('cuda')
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_C/config.yaml'
# ep2 cascade checkpoint lives in different dirs across machines / 不同机器路径不同:
#   machine 2: Checkpoint_C_phase2_8gpu/  ；machine 5 bundle: eval_c_fullset/
_CKPT_CANDIDATES = ['Checkpoint_C_phase2_8gpu/model_epoch_2.pth',
                    'eval_c_fullset/model_epoch_2.pth']
CKPT = next((c for c in _CKPT_CANDIDATES if os.path.exists(c)), _CKPT_CANDIDATES[0])
CSV = 'Datasets/test_256_2k.csv'
N_TIME = 50   # 计时迭代数 / timing iters


def fmt(n):
    """Human M/K. 参数量转 M/K。"""
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def main():
    model = build_model_from_yaml(CONFIG, _Args())
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model.trainable_tail.load_state_dict(ck.get('model_0', ck), strict=False)
    model = model.to(DEV).eval()

    # --- params per top-level component (frozen vs trainable) ---
    print("=== PARAMS (per component) ===")
    total = 0
    for name, mod in model.named_children():
        tot = sum(p.numel() for p in mod.parameters())
        tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        if tot == 0:
            continue
        total += tot
        kind = "trainable" if tr > 0 else "FROZEN"
        print(f"  {name:16s} {fmt(tot):>9s}  ({kind}, trainable={fmt(tr)})")
    tr_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {'TOTAL':16s} {fmt(total):>9s}  (trainable={fmt(tr_all)})")

    # --- one real pair for timing ---
    import pandas as pd
    r = pd.read_csv(CSV).iloc[0]
    sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                           duplicate_channels=True).unsqueeze(0).to(DEV)
    rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)).unsqueeze(0).to(DEV)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        # warmup
        for _ in range(5):
            f_rgb = model.rgb_encoder(rgb)
            f_sar = model.sar_encoder(sar)
        torch.cuda.synchronize()

        # (a) DINOv3 optical provider only (amortized once per reference)
        t = time.time()
        for _ in range(N_TIME):
            _ = model.rgb_encoder(rgb)
        torch.cuda.synchronize()
        t_dino = (time.time() - t) / N_TIME * 1000

        # (b) SAR encoder only (per-template)
        t = time.time()
        for _ in range(N_TIME):
            _ = model.sar_encoder(sar)
        torch.cuda.synchronize()
        t_sar = (time.time() - t) / N_TIME * 1000

        # (c) full per-template forward (encoders + JBU + coarse-corr + cascade).
        # 全前向:两编码器 + 引导上采样 + 粗相关 + 级联精修。matching-head
        # 时延 = 全前向 - DINOv3 - SAR(纯 GPU 计时,不含 dataloader 开销)。
        out = model(sar, rgb)
        print(f"\n  [forward output type={type(out).__name__} "
              f"keys={list(out.keys()) if isinstance(out, dict) else 'n/a'}]")
        t = time.time()
        for _ in range(N_TIME):
            _ = model(sar, rgb)
        torch.cuda.synchronize()
        t_full = (time.time() - t) / N_TIME * 1000

    t_match = t_full - t_dino - t_sar
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print("\n=== TIMING (bs=1, RTX 4080) ===")
    print(f"  DINOv3 optical provider (amortized/reference): {t_dino:.1f} ms")
    print(f"  SAR encoder (per template):                    {t_sar:.1f} ms")
    print(f"  matching head (JBU+coarse+cascade, derived):   {t_match:.1f} ms")
    print(f"  FULL forward per template (un-amortized):      {t_full:.1f} ms")
    print(f"  marginal per template (SAR + matching, cached DINOv3): "
          f"{t_sar + t_match:.1f} ms")
    print(f"  peak GPU memory (fwd):                         {peak:.0f} MiB")


if __name__ == '__main__':
    main()
