# -*- coding: utf-8 -*-
"""Would a LEARNED inner product beat raw cosine as the Stage-1 alignment metric?
Stage-1 若把"原始余弦"换成"学出来的内积",对齐任务会好多少？

Simulates a linear-projection Stage-1 by its optimal linear form (CCA), fit on
TRAIN images and evaluated on HELD-OUT images (honest, non-circular). Stride-16
template localization (slide the 12x12 SAR over the 16x16 optical = 5x5 offsets):
score(offset) = sum over the window of per-cell similarity. Compare:
  (a) RAW COSINE        — current Stage-1 metric (normalize raw 1024-d cells).
  (b) CCA inner product — project cells onto k learned canonical dirs, normalize,
      dot (= cosine in a learned subspace = the user's "learned inner product").
peak@GT (<=1 cell) and matched-vs-background separation d' for each.
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
N = 400
D_PCA = 64
KS = [8, 22, 64]
CSV = 'Datasets/test_256_2k.csv'
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_B/config.yaml'
CKPT = 'Checkpoint_B_256/model_epoch_4.pth'
SEED = 2024


def load_model():
    model = build_model_from_yaml(CONFIG, _Stage2Args())
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = ck.get('model_0', ck)
    model.trainable_tail.load_state_dict(state, strict=False)
    return model.to(DEV).eval()


def inv_sqrt(C, eps=1e-4):
    w, V = np.linalg.eigh(C); w = np.clip(w, eps, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def pca_basis(Xc, d):
    C = Xc.T @ Xc / Xc.shape[0]; w, V = np.linalg.eigh(C)
    return V[:, np.argsort(w)[::-1][:d]]


def fit_cca(Xtr, Ytr, d):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    Px, Py = pca_basis(Xc, d), pca_basis(Yc, d)
    Xr, Yr = Xc @ Px, Yc @ Py
    Cxx, Cyy = Xr.T @ Xr / len(Xr), Yr.T @ Yr / len(Yr)
    Cxy = Xr.T @ Yr / len(Xr)
    Kx, Ky = inv_sqrt(Cxx), inv_sqrt(Cyy)
    U, s, Vt = np.linalg.svd(Kx @ Cxy @ Ky)
    return Px @ Kx @ U, Py @ Ky @ Vt.T, mx, my   # Wx, Wy (1024,d)


def localize(score_map, gr, gc):
    """argmax of a (H,W) score map within <=1 cell of (gr,gc)."""
    h, w = score_map.shape
    idx = int(np.argmax(score_map))
    return int(max(abs(idx // w - gr), abs(idx % w - gc)) <= 1)


def main():
    rng = np.random.RandomState(SEED)
    model = load_model()
    df = pd.read_csv(CSV); n = min(N, len(df))

    feats, gts = [], []
    for i in range(n):
        r = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(r['Label']))
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            f_rgb = model.rgb_encoder(rgb)[-1][0].cpu().numpy()   # (1024,16,16)
            f_sar = model.sar_encoder(sar)['sar16'][0].cpu().numpy()  # (1024,12,12)
        oh, ow = f_rgb.shape[-2:]; sh, sw = f_sar.shape[-2:]
        gr = min(max(int(round(gy / 16)), 0), oh - sh)
        gc = min(max(int(round(gx / 16)), 0), ow - sw)
        feats.append((f_rgb, f_sar)); gts.append((gr, gc, oh, ow, sh, sw))
        if (i + 1) % 100 == 0:
            print(f'  extracted {i + 1}/{n}', flush=True)

    # image-holdout CCA fit on TRAIN corresponding cells
    idx = np.arange(n); rng.shuffle(idx)
    tr, te = idx[:n // 2], idx[n // 2:]
    Xtr, Ytr = [], []
    for i in tr:
        f_rgb, f_sar = feats[i]; gr, gc, oh, ow, sh, sw = gts[i]
        d = f_rgb[:, gr:gr + sh, gc:gc + sw].reshape(1024, -1).T
        s = f_sar.reshape(1024, -1).T
        Xtr.append(d); Ytr.append(s)
    Xtr = np.concatenate(Xtr).astype(np.float64); Ytr = np.concatenate(Ytr).astype(np.float64)
    Wx, Wy, mx, my = fit_cca(Xtr, Ytr, D_PCA)

    def proj_norm(cells, W, m, k):
        v = (cells - m) @ W[:, :k]                      # (Ncells, k)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    def l2norm(cells):
        return cells / (np.linalg.norm(cells, axis=1, keepdims=True) + 1e-8)

    methods = ['cosine'] + [f'cca_k{k}' for k in KS]
    hit = {m: [] for m in methods}
    sep = {m: {'g': [], 'b': []} for m in methods}     # matched vs background window-score
    for i in te:
        f_rgb, f_sar = feats[i]; gr, gc, oh, ow, sh, sw = gts[i]
        no = oh - sh + 1; nw = ow - sw + 1               # 5x5 offset grid
        sar_cells = f_sar.reshape(1024, -1).T            # (144,1024)
        rgb_cells = f_rgb.reshape(1024, -1).T            # (256,1024)
        reps = {'cosine': (l2norm(rgb_cells), l2norm(sar_cells))}
        for k in KS:
            reps[f'cca_k{k}'] = (proj_norm(rgb_cells, Wx, mx, k), proj_norm(sar_cells, Wy, my, k))
        for mth in methods:
            R, S = reps[mth]
            R = R.reshape(oh, ow, -1); S = S.reshape(sh, sw, -1)
            smap = np.zeros((no, nw))
            for a in range(no):
                for b in range(nw):
                    win = R[a:a + sh, b:b + sw, :]
                    smap[a, b] = float((win * S).sum())
            hit[mth].append(localize(smap, gr, gc))
            sep[mth]['g'].append(smap[gr, gc])
            bg = [smap[a, b] for a in range(no) for b in range(nw)
                  if max(abs(a - gr), abs(b - gc)) >= 2]
            sep[mth]['b'].append(np.mean(bg) if bg else smap.mean())

    no = gts[te[0]][2] - gts[te[0]][4] + 1
    chance = 9.0 / (no * no)
    print(f'\n==== Stage-1 metric: raw cosine vs LEARNED inner product (CCA) ====', flush=True)
    print(f'  held-out images, stride-16 {no}x{no}-offset localization; peak@GT chance≈{chance:.3f}\n', flush=True)
    for mth in methods:
        g = np.array(sep[mth]['g']); b = np.array(sep[mth]['b'])
        d = (g.mean() - b.mean()) / (np.sqrt(0.5 * (g.std() ** 2 + b.std() ** 2)) + 1e-8)
        print(f'  {mth:10s}  peak@GT(<=1cell)={np.mean(hit[mth]):.3f}   matched-vs-bg d\'={d:.2f}', flush=True)
    print('\nSTAGE1SIM_DONE', flush=True)


if __name__ == '__main__':
    main()
