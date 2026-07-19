# -*- coding: utf-8 -*-
"""Directly MEASURE the mechanism of F0 (why Stage-1 warm-start gives no gain).
直接测 F0 的机制：Stage-1 暖启为什么没用。

Compare three SAR encoders on the SAME images, against the frozen DINO:
  S1  = Checkpoint_A_stage1_256/model_epoch_5.pth   (Stage-1 A, BEFORE Stage-2)
  AFT = Checkpoint_A_ft_256/model_epoch_4.pth        (A-ft, AFTER Stage-2 fine-tune)
  B   = Checkpoint_B_256/model_epoch_4.pth           (random-init, AFTER Stage-2)

Metrics per encoder:
  raw-cosine alignment to DINO at GT (matched vs background per-cell cosine, d',
  stride-16 peak@GT) — did Stage-1 align? did Stage-2 keep or erase it?
Cross-encoder linear CKA on sar16 GT-cells:
  CKA(S1, AFT)  — how far Stage-2 moved A's encoder from its Stage-1 state.
  CKA(AFT, B)   — did A-ft and B converge to the same encoder? (overwrite/convergence)
  CKA(S1, B)    — baseline.
Verdict:
  S1 aligned & AFT~B~0 & CKA(AFT,B) high  → Stage-2 OVERWROTE + CONVERGED  (overwrite mechanism)
  S1 already ~0                            → Stage-1 never aligned          (failed-Stage-1 mechanism)
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
    is_stage_2 = True
    pre_trained = True
    dist = False
    root_path = '.'
    modelDir = '.'
    gpu = 1


DEV = torch.device('cuda')
N = 300
CSV = 'Datasets/test_256_2k.csv'
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_B/config.yaml'
SEED = 2024
CKPTS = {
    'S1':  'Checkpoint_A_stage1_256/model_epoch_5.pth',
    'AFT': 'Checkpoint_A_ft_256/model_epoch_4.pth',
    'B':   'Checkpoint_B_256/model_epoch_4.pth',
}


def build():
    return build_model_from_yaml(CONFIG, _Stage2Args()).to(DEV).eval()


def load_sar(model, ckpt):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = ck.get('model_0', ck)
    miss, unexp = model.trainable_tail.load_state_dict(state, strict=False)
    nsar = sum(1 for k in state if 'sar_encoder' in k)
    print(f'  {ckpt}: sar_encoder keys in ckpt={nsar}  missing={len(miss)} unexp={len(unexp)}', flush=True)


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(Y.T @ X, 'fro') ** 2
    xx = np.linalg.norm(X.T @ X, 'fro'); yy = np.linalg.norm(Y.T @ Y, 'fro')
    return float(xy / (xx * yy + 1e-12))


def main():
    rng = np.random.RandomState(SEED)
    model = build()
    df = pd.read_csv(CSV); n = min(N, len(df))

    # cache DINO GT-window cells + sar image tensors + GT (DINO is frozen, encoder-independent)
    dino_cells, sar_imgs, gts = [], [], []
    for i in range(n):
        r = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(r['Label']))
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            f_rgb = model.rgb_encoder(rgb)[-1][0].cpu()      # (1024,16,16)
        oh, ow = f_rgb.shape[-2:]; sh = sw = 12
        gr = min(max(int(round(gy / 16)), 0), oh - sh)
        gc = min(max(int(round(gx / 16)), 0), ow - sw)
        dino_cells.append(F.normalize(f_rgb[:, gr:gr + sh, gc:gc + sw].reshape(1024, -1).t(), dim=1))
        sar_imgs.append(sar); gts.append((gr, gc, oh, ow, sh, sw, f_rgb))
        if (i + 1) % 100 == 0:
            print(f'  cached DINO {i + 1}/{n}', flush=True)
    DcatN = torch.cat(dino_cells, 0).numpy()                 # (M,1024) normed DINO GT cells

    align, sarfeat = {}, {}
    for tag, ckpt in CKPTS.items():
        load_sar(model, ckpt)
        mcos, bcos, peaks, cells = [], [], [], []
        for i in range(n):
            with torch.no_grad():
                s16 = model.sar_encoder(sar_imgs[i].to(DEV))['sar16'][0].cpu()  # (1024,12,12) L2-normed
            gr, gc, oh, ow, sh, sw, f_rgb = gts[i]
            rgbn = F.normalize(f_rgb, dim=0)                  # per-cell normalize DINO (1024,16,16)
            sarc = s16.reshape(1024, -1).t()                 # (144,1024) already normed
            # matched: per-cell cos at GT window
            dwin = rgbn[:, gr:gr + sh, gc:gc + sw].reshape(1024, -1).t()
            mcos.append(float((sarc * dwin).sum(1).mean()))
            # background: random wrong offset
            br = rng.randint(0, oh - sh + 1); bc = rng.randint(0, ow - sw + 1)
            while max(abs(br - gr), abs(bc - gc)) < 2:
                br = rng.randint(0, oh - sh + 1); bc = rng.randint(0, ow - sw + 1)
            dbg = rgbn[:, br:br + sh, bc:bc + sw].reshape(1024, -1).t()
            bcos.append(float((sarc * dbg).sum(1).mean()))
            # stride-16 localization peak@GT (cross-corr argmax of summed cosine)
            no, nw = oh - sh + 1, ow - sw + 1
            R = rgbn.reshape(1024, oh, ow); S = s16.reshape(1024, sh, sw)
            smap = torch.empty(no, nw)
            for a in range(no):
                for b in range(nw):
                    smap[a, b] = (R[:, a:a + sh, b:b + sw] * S).sum()
            p = int(smap.view(-1).argmax().item())
            peaks.append(int(max(abs(p // nw - gr), abs(p % nw - gc)) <= 1))
            cells.append(sarc.numpy())
        m, b = np.array(mcos), np.array(bcos)
        d = (m.mean() - b.mean()) / (np.sqrt(0.5 * (m.std() ** 2 + b.std() ** 2)) + 1e-8)
        align[tag] = (m.mean(), b.mean(), d, float(np.mean(peaks)))
        sarfeat[tag] = np.concatenate(cells, 0)
        print(f'  [{tag}] matched_cos={m.mean():+.4f} bg_cos={b.mean():+.4f} d\'={d:.2f} '
              f'stride16 peak@GT={np.mean(peaks):.3f}', flush=True)

    print('\n==== F0 mechanism: SAR encoder alignment to frozen DINO ====', flush=True)
    print(f'  (stride-16 peak@GT chance ≈ {9/25:.2f})', flush=True)
    for tag in CKPTS:
        m, b, d, pk = align[tag]
        print(f'  {tag:4s}  matched_cos={m:+.4f}  bg_cos={b:+.4f}  d\'={d:.2f}  peak@GT={pk:.3f}', flush=True)
    print('\n==== cross-encoder CKA on sar16 (did Stage-2 converge A-ft and B?) ====', flush=True)
    print(f'  CKA(S1 , AFT) = {linear_cka(sarfeat["S1"], sarfeat["AFT"]):.3f}   (Stage-2 moved A how far from Stage-1)', flush=True)
    print(f'  CKA(AFT, B  ) = {linear_cka(sarfeat["AFT"], sarfeat["B"]):.3f}   (did A-ft & B converge?)', flush=True)
    print(f'  CKA(S1 , B  ) = {linear_cka(sarfeat["S1"], sarfeat["B"]):.3f}   (baseline)', flush=True)
    print(f'  CKA(DINO-GTcells vs themselves sanity) skipped', flush=True)
    print('\nF0MECH_DONE', flush=True)


if __name__ == '__main__':
    main()
