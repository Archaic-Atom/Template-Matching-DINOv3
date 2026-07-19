# -*- coding: utf-8 -*-
"""Matching-dimension output probe (experiment A, NO TRAINING).

问题:matcher 的 matching_dim=256 是不是冗余?localization 信号实际活在多少维?
Restrict the trained matcher's 256-d matching OUTPUT (post-projection rgb_4/sar_4) to its
top-k PCA components, re-normalize, re-run the raw cosine cross-correlation, and measure
valley@GT recovery vs the full 256-d. If recovery saturates well below 256 (e.g. ~64),
matching_dim=256 is over-provisioned and ~that-many dimensions suffice — a no-training,
post-hoc read of the capacity the deployed matcher actually uses in its OUTPUT space
(complements the CCA-input rank probe, which restricts the 1024-d stride-16 INPUT).

No backprop; forward pass + PCA + SVD-free restriction only.
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
RANKS = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
CSV = 'Datasets/test_256_2k.csv'
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_B/config.yaml'
CKPT = 'Checkpoint_B_256/model_epoch_4.pth'
SEED = 2024
PCA_CELLS_PER_IMG = 80   # subsample cells per image for PCA fitting


def load_model():
    model = build_model_from_yaml(CONFIG, _Stage2Args())
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = ck.get('model_0', ck)
    miss, unexp = model.trainable_tail.load_state_dict(state, strict=False)
    print(f'  loaded {CKPT}: missing={len(miss)} unexpected={len(unexp)}', flush=True)
    return model.to(DEV).eval()


def pca_basis(Xc, d):
    """Top-d PCA directions (256,d) of centered rows Xc (n,256)."""
    C = Xc.T @ Xc / Xc.shape[0]
    w, V = np.linalg.eigh(C)
    return V[:, np.argsort(w)[::-1][:d]]


def both_hits(M, gr, gc):
    h, w = M.shape[-2:]
    flat = M.view(-1)
    imn = int(flat.argmin().item()); imx = int(flat.argmax().item())
    mn = int(max(abs(imn // w - gr), abs(imn % w - gc)) <= 1)
    mx = int(max(abs(imx // w - gr), abs(imx % w - gc)) <= 1)
    return mn, mx


def saturation_rank(curve, full, floor, frac):
    denom = max(full - floor, 1e-9)
    rec = [(v - floor) / denom for v in curve]
    for j, k in enumerate(RANKS):
        if all(c >= frac for c in rec[j:]):
            return k
    return None


def _fmt_rank(k):
    return f'{k}' if k is not None else f'>{RANKS[-1]} (no saturation)'


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    model = load_model()
    mh = model.matching
    df = pd.read_csv(CSV)
    n = min(N, len(df))

    # ---- pass 1: compute the 256-d stride-4 matching OUTPUT features + GT ----
    rgb4s, sar4s, gts4, pca_pool = [], [], [], []
    for i in range(n):
        r = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(r['Label']))
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)
                               ).unsqueeze(0).to(DEV)
        with torch.no_grad():
            f_rgb = model.rgb_encoder(rgb)[-1]
            f_sar = model.sar_encoder(sar)['sar16']
            rgb4 = mh.proj_rgb_4(f_rgb, rgb)   # (1,256,64,64) L2-normed
            sar4 = mh.proj_sar_4(f_sar, sar)   # (1,256,48,48) L2-normed
        oh, ow = f_rgb.shape[-2:]
        _, _, sh, sw = f_sar.shape
        g4r = min(max(int(round(gy / 4)), 0), (oh - sh) * 4)
        g4c = min(max(int(round(gx / 4)), 0), (ow - sw) * 4)
        rgb4s.append(rgb4[0].cpu()); sar4s.append(sar4[0].cpu()); gts4.append((g4r, g4c))
        # subsample cells for PCA pool (both modalities)
        for fmap in (rgb4[0], sar4[0]):
            c, h, w = fmap.shape
            flat = fmap.reshape(c, h * w).t().cpu().numpy()   # (HW,256)
            idx = rng.choice(flat.shape[0], min(PCA_CELLS_PER_IMG, flat.shape[0]), replace=False)
            pca_pool.append(flat[idx])
        if (i + 1) % 100 == 0:
            print(f'  extracted {i + 1}/{n}', flush=True)

    # ---- image split: fit PCA on train-half cells, evaluate on held-out imgs ----
    imgs = np.arange(n); rng.shuffle(imgs)
    tr_imgs = set(imgs[:n // 2].tolist())
    te_imgs = [i for i in range(n) if i not in tr_imgs]

    # PCA pool entries are 2 per image (rgb then sar), in image order
    pool_tr = []
    for i in range(n):
        if i in tr_imgs:
            pool_tr.append(pca_pool[2 * i]); pool_tr.append(pca_pool[2 * i + 1])
    V = np.concatenate(pool_tr).astype(np.float64)      # (M,256)
    mu = V.mean(0)
    P = pca_basis(V - mu, 256)                           # (256,256) ordered top->low
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEV)
    Pt = torch.tensor(P, dtype=torch.float32, device=DEV)
    print(f'\n  PCA fit on {V.shape[0]} matching-output cells (256-d).', flush=True)

    def restrict(fmap, k):
        """Restrict (1,256,H,W) matching output to top-k PCA dirs, re-L2-normalize."""
        b, c, h, w = fmap.shape
        x = fmap.reshape(c, h * w).t()          # (HW,256)
        xc = x - mu_t
        Qk = Pt[:, :k]
        xr = xc @ Qk @ Qk.t() + mu_t            # back-projected
        xr = F.normalize(xr, p=2, dim=1)        # cosine matching expects unit norm
        return xr.t().reshape(1, c, h, w)

    def valley(rgb4, sar4, gr, gc):
        with torch.no_grad():
            M = F.conv2d(rgb4, sar4)
        return both_hits(M, gr, gc)

    # full (k=256, i.e. no restriction) + chance
    acc = {k: [] for k in RANKS}
    acc_full = []
    for i in te_imgs:
        rgb4 = rgb4s[i].unsqueeze(0).to(DEV)
        sar4 = sar4s[i].unsqueeze(0).to(DEV)
        g4r, g4c = gts4[i]
        acc_full.append(valley(rgb4, sar4, g4r, g4c))
        for k in RANKS:
            acc[k].append(valley(restrict(rgb4, k), restrict(sar4, k), g4r, g4c))

    # map size / chance from one full conv
    i0 = te_imgs[0]
    with torch.no_grad():
        M0 = F.conv2d(rgb4s[i0].unsqueeze(0).to(DEV), sar4s[i0].unsqueeze(0).to(DEV))
    h0, w0 = M0.shape[-2:]
    chance = 9.0 / (h0 * w0)

    arr_full = np.array(acc_full)
    fmn, fmx = arr_full.mean(0)
    col = 0 if fmn >= fmx else 1
    sign = 'MIN(valley)' if col == 0 else 'MAX(peak)'
    full = float(arr_full.mean(0)[col])
    curve = [float(np.array(acc[k]).mean(0)[col]) for k in RANKS]
    denom = max(full - chance, 1e-9)
    rec = [(v - chance) / denom for v in curve]
    r80 = saturation_rank(curve, full, chance, 0.80)
    r90 = saturation_rank(curve, full, chance, 0.90)

    print('\n==== matching_dim OUTPUT-space rank restriction (B@e4, stride-4) ====', flush=True)
    print(f'  map={h0}x{w0} chance={chance:.3f}  sign={sign}  FULL(256-d) hit@GT={full:.3f}', flush=True)
    print('  k = retained matching-output PCA dims  [hit@GT | chance-norm recovery]:', flush=True)
    for k, v, rv in zip(RANKS, curve, rec):
        bar = '#' * int(round(40 * max(rv, 0.0)))
        print(f'    k={k:3d}  hit={v:.3f}  rec={rv:+.2f}  {bar}', flush=True)
    print(f'\n  saturation: 0.80 recovery @ k={_fmt_rank(r80)}   0.90 @ k={_fmt_rank(r90)}', flush=True)
    print(f'  => of the 256 matching dims, localization saturates by ~k above; '
          f'256 is over-provisioned if k<<256.', flush=True)
    print('\nMATCHDIM_DONE', flush=True)


if __name__ == '__main__':
    main()
