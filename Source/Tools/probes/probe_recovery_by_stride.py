# -*- coding: utf-8 -*-
"""Recovery rank across the FULL pipeline: stride-16 -> 8 -> 4 -> 2 -> 1.

同一把 recovery 尺,量整条流水线每个粒度定位所需的共享维度:
  stride-16  = 特征提取端(编码器原始互相关,不过 proj)
  stride-8/4 = 粗匹配头(复现论文的 48/64,full-map)
  stride-2/1 = cascade 细化级(本次新测;GT-居中 ±4 窗口 = cascade 实际做法)

Method: fit ONE de-leaked CCA shared subspace on the stride-16 GT cells; restrict
the stride-16 INPUT to top-k directions; run the REAL cascade chain on the
restricted input; measure localization hit@GT recovery at every stride. stride-2/1
use a GT-centered +-4 window (the fine-stage conv otherwise blows up via im2col on
the 192x192 template; the cascade itself only searches +-4 there). Recovery rank =
smallest k with sustained >=0.90 chance-normalized recovery. Per-stride localizing
extremum (argmin/argmax) chosen from the full-feature hit.
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

DEV = torch.device('cuda')
N = 160
D_PCA = 64
RANKS = [4, 8, 12, 16, 20, 22, 24, 28, 32, 40, 48, 56, 64]
CSV = '/data4/datasets/eval2k_opt256_sar192/image_test_2k_256.csv'
CFG_C = 'Source/UserModelImplementation/Models/TM_Model_C/config.yaml'
CKPT_C = 'Checkpoint_C_direct_sub40_256/model_epoch_2.pth'
SEED = 2024
STRIDES = [16, 8, 4, 2, 1]
W_FINE = 4               # GT-centered window (cells) for stride-2/1


class _Stage2Args:
    is_stage_2 = True
    pre_trained = True
    dist = False
    root_path = '.'
    modelDir = '.'
    gpu = 1


def inv_sqrt(C, eps=1e-4):
    w, V = np.linalg.eigh(C)
    return V @ np.diag(1.0 / np.sqrt(np.clip(w, eps, None))) @ V.T


def pca_basis(Xc, d):
    C = Xc.T @ Xc / Xc.shape[0]
    w, V = np.linalg.eigh(C)
    return V[:, np.argsort(w)[::-1][:d]]


def fit_cca(Xtr, Ytr, d):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    Px, Py = pca_basis(Xc, d), pca_basis(Yc, d)
    Xr, Yr = Xc @ Px, Yc @ Py
    Kx, Ky = inv_sqrt(Xr.T @ Xr / len(Xr)), inv_sqrt(Yr.T @ Yr / len(Yr))
    U, s, Vt = np.linalg.svd(Kx @ (Xr.T @ Yr / len(Xr)) @ Ky)
    return Px @ Kx @ U, Py @ Ky @ Vt.T, mx, my


def orthobasis(W, k):
    Q, _ = np.linalg.qr(W[:, :k])
    return Q[:, :k]


def both_hits(M, gr, gc):
    h, w = M.shape[-2:]
    flat = M.reshape(-1)
    imn, imx = int(flat.argmin()), int(flat.argmax())
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


def _fmt(k):
    return f'{k}' if k is not None else f'>{RANKS[-1]}'


def project_feat(fmap, Q, mu):
    b, c, h, w = fmap.shape
    x = fmap.reshape(c, h * w).t()
    xr = (x - mu) @ Q @ Q.t() + mu
    return xr.t().reshape(1, c, h, w)


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    df = pd.read_csv(CSV)
    n = min(N, len(df))

    print(f'building cascade C, loading {CKPT_C} ...', flush=True)
    model = build_model_from_yaml(CFG_C, _Stage2Args())
    ck = torch.load(CKPT_C, map_location='cpu', weights_only=False)
    miss, unexp = model.trainable_tail.load_state_dict(ck.get('model_0', ck), strict=False)
    print(f'  loaded: missing={len(miss)} unexpected={len(unexp)}', flush=True)
    model = model.to(DEV).eval()
    mh = model.matching

    def norm(x):
        return F.normalize(x, p=2, dim=1)

    def full_corr(rgb_s, sar_s, gy, gx, stride):
        M = F.conv2d(rgb_s, sar_s)
        h, w = M.shape[-2:]
        gr = min(max(int(round(gy / stride)), 0), h - 1)
        gc = min(max(int(round(gx / stride)), 0), w - 1)
        return M, gr, gc

    def win_corr(rgb_s, sar_s, gy, gx, stride):
        _, C_, Hs, Ws = sar_s.shape
        _, _, Hr, Wr = rgb_s.shape
        gr_tl = min(max(int(round(gy / stride)), 0), Hr - Hs)
        gc_tl = min(max(int(round(gx / stride)), 0), Wr - Ws)
        cy, cx = Hs + 2 * W_FINE, Ws + 2 * W_FINE
        y0 = min(max(gr_tl - W_FINE, 0), Hr - cy)
        x0 = min(max(gc_tl - W_FINE, 0), Wr - cx)
        M = F.conv2d(rgb_s[:, :, y0:y0 + cy, x0:x0 + cx], sar_s)
        return M, gr_tl - y0, gc_tl - x0

    def stride_maps(rgb16, sar16, rgb_img, sar_img, gy, gx):
        out = {16: full_corr(norm(rgb16), norm(sar16), gy, gx, 16)}
        sar8 = mh.proj_sar_8(sar16, sar_img); rgb8 = mh.proj_rgb_8(rgb16, rgb_img)
        out[8] = full_corr(rgb8, sar8, gy, gx, 8)
        sar4 = mh.proj_sar_4(sar16, sar_img); rgb4 = mh.proj_rgb_4(rgb16, rgb_img)
        out[4] = full_corr(rgb4, sar4, gy, gx, 4)
        sar2 = norm(mh.sar_upsample_4to2(sar4)); rgb2 = norm(mh.rgb_upsample_4to2(rgb4))
        out[2] = win_corr(rgb2, sar2, gy, gx, 2)
        sar1f = mh.sar_upsample_2to1(sar2); rgb1f = mh.rgb_upsample_2to1(rgb2)
        sar1 = mh.proj_sar_1(torch.cat([sar1f, mh.sar_raw_stem(sar_img)], dim=1))
        rgb1 = mh.proj_rgb_1(torch.cat([rgb1f, mh.rgb_raw_stem(rgb_img)], dim=1))
        out[1] = win_corr(rgb1, sar1, gy, gx, 1)
        return out

    # ---- pass 1: stride-16 feats + images + GT ----
    rgb16s, sar16s, imgs, gts = [], [], [], []
    for i in range(n):
        r = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(r['Label']))
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            f_rgb = model.rgb_encoder(rgb)[-1]
            f_sar = model.sar_encoder(sar)['sar16']
        rgb16s.append(f_rgb); sar16s.append(f_sar); imgs.append((sar, rgb)); gts.append((gy, gx))
        if (i + 1) % 40 == 0:
            print(f'  extracted {i + 1}/{n}', flush=True)

    _, C, sh, sw = sar16s[0].shape
    oh, ow = rgb16s[0].shape[-2:]
    X, Y, iid = [], [], []
    for i in range(n):
        gy, gx = gts[i]
        gr = min(max(int(round(gy / 16)), 0), oh - sh)
        gc = min(max(int(round(gx / 16)), 0), ow - sw)
        d = rgb16s[i][0][:, gr:gr + sh, gc:gc + sw].reshape(C, -1).t().cpu().numpy()
        s = sar16s[i][0].reshape(C, -1).t().cpu().numpy()
        X.append(d); Y.append(s); iid.append(np.full(d.shape[0], i))
    X = np.concatenate(X).astype(np.float64); Y = np.concatenate(Y).astype(np.float64)
    iid = np.concatenate(iid)
    order = np.arange(n); rng.shuffle(order)
    tr = set(order[:n // 2].tolist())
    trm = np.array([k for k in range(len(X)) if iid[k] in tr])
    Wx, Wy, mx, my = fit_cca(X[trm], Y[trm], D_PCA)
    te_imgs = [k for k in range(n) if k not in tr]
    mx_t = torch.tensor(mx, dtype=torch.float32, device=DEV)
    my_t = torch.tensor(my, dtype=torch.float32, device=DEV)
    Qd = {k: torch.tensor(orthobasis(Wx, k), dtype=torch.float32, device=DEV) for k in RANKS}
    Qs = {k: torch.tensor(orthobasis(Wy, k), dtype=torch.float32, device=DEV) for k in RANKS}
    print(f'  CCA on {len(trm)} GT cells; {len(te_imgs)} held-out images', flush=True)

    keys = ['full'] + [f'r{k}' for k in RANKS]
    acc = {s: {kk: [] for kk in keys} for s in STRIDES}
    mapsz = {}
    for cnt, i in enumerate(te_imgs):
        sar, rgb = imgs[i]; gy, gx = gts[i]
        with torch.no_grad():
            fm = stride_maps(rgb16s[i], sar16s[i], rgb, sar, gy, gx)
            if cnt == 0:
                for s in STRIDES:
                    M, gr, gc = fm[s]
                    mapsz[s] = tuple(M.shape[-2:])
                    print(f'  stride-{s} map={mapsz[s]} gt=({gr},{gc}) '
                          f'chance={9.0/(mapsz[s][0]*mapsz[s][1]):.3f}', flush=True)
            for s in STRIDES:
                M, gr, gc = fm[s]
                acc[s]['full'].append(both_hits(M, gr, gc))
            for k in RANKS:
                r16 = project_feat(rgb16s[i], Qd[k], mx_t)
                s16 = project_feat(sar16s[i], Qs[k], my_t)
                mps = stride_maps(r16, s16, rgb, sar, gy, gx)
                for s in STRIDES:
                    M, gr, gc = mps[s]
                    acc[s][f'r{k}'].append(both_hits(M, gr, gc))
        if (cnt + 1) % 20 == 0:
            print(f'  eval {cnt + 1}/{len(te_imgs)}', flush=True)

    print('\n==== recovery rank by stride (restrict stride-16 input, one ruler) ====', flush=True)
    print('  (16/8/4 = full-map; 2/1 = GT-centered +-4 window)', flush=True)
    print(f'  {"stride":7s} {"map":9s} {"chance":>7s} {"full":>6s} {"sign":>5s} '
          f'{"r0.90":>6s}  sweep(hit@GT over ranks 4..64)', flush=True)
    for s in STRIDES:
        arr = np.array(acc[s]['full'])
        fmin, fmax = arr.mean(0)
        col = 0 if fmin >= fmax else 1
        sign = 'MIN' if col == 0 else 'MAX'
        full = float(arr.mean(0)[col])
        curve = [float(np.array(acc[s][f'r{k}']).mean(0)[col]) for k in RANKS]
        h, w = mapsz[s]
        chance = 9.0 / (h * w)
        r90 = saturation_rank(curve, full, chance, 0.90)
        sweep = ' '.join(f'{v:.2f}' for v in curve)
        print(f'  s={s:<5d} {h}x{w:<6d} {chance:7.3f} {full:6.3f} {sign:>5s} '
              f'{_fmt(r90):>6s}  {sweep}', flush=True)

    print('\n  READ: does recovery rank (r0.90) rise 16->8->4->2->1 (finer -> more dims)?', flush=True)
    print('  stride-16 = feature-extraction side: if full~chance it does not localize (gap).', flush=True)
    print('\nRECOVERY_BY_STRIDE_DONE', flush=True)


if __name__ == '__main__':
    main()
