# -*- coding: utf-8 -*-
"""EMPIRICAL collapse test for a learned-inner-product Stage-1 objective.
实测：<norm(W_s·sar), norm(W_d·dino)> + InfoNCE 会不会坍缩，可学 W_d 是否安全。

Freeze the SAR encoder + DINO (use B's stride-16 features); train ONLY two linear
projection heads W_s, W_d with patch-InfoNCE (positive = GT-aligned DINO cell,
negatives = other in-image DINO cells). Two conditions:
  (A) W_d LEARNABLE       — the risky case (no fixed anchor).
  (B) W_d FIXED (DINO PCA-64, frozen) — asymmetric anchor à la stop-gradient.
Monitor over training: effective rank of the projected features (collapse meter),
held-out stride-16 localization peak@GT, and the loss. Collapse = erank drifts down
toward 1 / localization degrades.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode

sys.path.insert(0, 'Source')
from Tools.eval_stage1_matching import preprocess_image  # noqa: E402
from UserModelImplementation.Models.builder import build_model_from_yaml  # noqa: E402


class _Stage2Args:
    is_stage_2 = True
    pre_trained = True
    dist = False
    root_path = '.'
    modelDir = '.'
    gpu = 1


DEV = torch.device('cuda')
N = 300
D = 64
TAU = 0.07
STEPS = 600
LR = 1e-3
IMG_PER_STEP = 8
CSV = 'Datasets/test_256_2k.csv'
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_B/config.yaml'
CKPT = 'Checkpoint_B_256/model_epoch_4.pth'
SEED = 2024


def load_model():
    model = build_model_from_yaml(CONFIG, _Stage2Args())
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model.trainable_tail.load_state_dict(ck.get('model_0', ck), strict=False)
    return model.to(DEV).eval()


def erank(Fm):
    """Effective rank = exp(entropy of normalized squared singular values)."""
    Fc = Fm - Fm.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Fc)
    p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    model = load_model()
    df = pd.read_csv(CSV); n = min(N, len(df))

    sar_list, rgb_list, meta = [], [], []
    for i in range(n):
        r = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(r['Label']))
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            f_rgb = model.rgb_encoder(rgb)[-1][0]      # (1024,16,16)
            f_sar = model.sar_encoder(sar)['sar16'][0]  # (1024,12,12)
        oh, ow = f_rgb.shape[-2:]; sh, sw = f_sar.shape[-2:]
        gr = min(max(int(round(gy / 16)), 0), oh - sh)
        gc = min(max(int(round(gx / 16)), 0), ow - sw)
        rgb_list.append(f_rgb.reshape(1024, -1).t().contiguous())   # (256,1024)
        sar_list.append(f_sar.reshape(1024, -1).t().contiguous())   # (144,1024)
        # positive DINO flat-index for each of the 144 SAR cells (GT alignment)
        pos = [(gr + a) * ow + (gc + b) for a in range(sh) for b in range(sw)]
        meta.append((torch.tensor(pos, device=DEV), gr, gc, oh, ow, sh, sw))
        if (i + 1) % 100 == 0:
            print(f'  extracted {i + 1}/{n}', flush=True)

    idx = np.arange(n); rng.shuffle(idx)
    tr, te = idx[:n // 2].tolist(), idx[n // 2:].tolist()

    # fixed W_d = frozen DINO PCA-64 (anchor for condition B)
    alld = torch.cat([rgb_list[i] for i in tr], 0)            # (*,1024)
    dc = alld - alld.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(dc, full_matrices=False)
    Wd_fixed = Vt[:D].t().contiguous()                       # (1024,64) frozen PCA basis

    def heldout_peak(Ws, Wd):
        hits = []
        with torch.no_grad():
            for i in te:
                R = F.normalize(rgb_list[i] @ Wd, dim=1)     # (256,64)
                S = F.normalize(sar_list[i] @ Ws, dim=1)     # (144,64)
                _, gr, gc, oh, ow, sh, sw = meta[i]
                R = R.reshape(oh, ow, D); S = S.reshape(sh, sw, D)
                no, nw = oh - sh + 1, ow - sw + 1
                smap = torch.empty(no, nw, device=DEV)
                for a in range(no):
                    for b in range(nw):
                        smap[a, b] = (R[a:a + sh, b:b + sw, :] * S).sum()
                p = int(smap.view(-1).argmax().item())
                hits.append(int(max(abs(p // nw - gr), abs(p % nw - gc)) <= 1))
        return float(np.mean(hits))

    def run(learn_wd, tag):
        torch.manual_seed(SEED)
        Ws = nn.Linear(1024, D, bias=False).to(DEV)
        if learn_wd:
            Wd = nn.Linear(1024, D, bias=False).to(DEV)
            with torch.no_grad():
                Wd.weight.copy_(Wd_fixed.t())                # same init as fixed case
            params = list(Ws.parameters()) + list(Wd.parameters())
            wd_mat = lambda: Wd.weight.t()
        else:
            params = list(Ws.parameters())
            wd_mat = lambda: Wd_fixed
        opt = torch.optim.Adam(params, lr=LR)
        print(f'\n-- condition {tag} (W_d {"learnable" if learn_wd else "fixed PCA"}) --', flush=True)
        for step in range(STEPS + 1):
            batch = rng.choice(tr, size=IMG_PER_STEP, replace=False)
            loss = 0.0
            for i in batch:
                S = F.normalize(Ws(sar_list[i]), dim=1)       # (144,64)
                R = F.normalize(rgb_list[i] @ wd_mat(), dim=1)  # (256,64)
                sim = S @ R.t() / TAU                          # (144,256)
                loss = loss + F.cross_entropy(sim, meta[i][0])
            loss = loss / len(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 150 == 0:
                with torch.no_grad():
                    sproj = F.normalize(Ws(torch.cat([sar_list[i] for i in te[:60]], 0)), dim=1)
                    dproj = F.normalize(torch.cat([rgb_list[i] for i in te[:60]], 0) @ wd_mat(), dim=1)
                er_s, er_d = erank(sproj), erank(dproj)
                peak = heldout_peak(Ws.weight.t().detach(), wd_mat().detach())
                print(f'  step {step:4d}  loss={loss.item():.3f}  '
                      f'erank(W_s·sar)={er_s:.1f}/{D}  erank(W_d·dino)={er_d:.1f}/{D}  '
                      f'heldout peak@GT={peak:.3f}', flush=True)

    print(f'\n==== empirical collapse test (D={D}, tau={TAU}, {STEPS} steps) ====', flush=True)
    print(f'  erank close to D = healthy; drifting toward 1 = dimensional collapse', flush=True)
    run(False, 'B-fixed')
    run(True, 'A-learnable')
    print('\nCOLLAPSE_DONE', flush=True)


if __name__ == '__main__':
    main()
