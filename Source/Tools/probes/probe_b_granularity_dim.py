# -*- coding: utf-8 -*-
"""Granularity vs matching-dimensionality (C2 strengthening, experiment ①).

在**同一个估计器、同一个 ckpt、同一个跨模态共享子空间**下,只改变匹配 granularity
(stride-4 vs stride-8),测量 matching 信号所需的维度,从而把 C2「对齐容量~22 vs
匹配需要~64」从两点(且跨不同估计器)扩成「同一估计器内、granularity 单变量」的趋势。

Builds directly on ``probe_b_keystone.py``. Identical E3 de-leaked CCA produces ONE
shared rank-D_PCA subspace (orthonormal Q_dino, Q_sar in 1024-d). E2 keystone is then
run TWICE — feeding the SAME subspace-restricted stride-16 features through the REAL
``proj_*_4`` (×4 JBU) AND ``proj_*_8`` (×2 JBU) of B@e4 — with a fine rank sweep, so the
ONLY variable between the two curves is the matching granularity.

Sign robustness: B-ep4's stride-4 raw (pre-refine) cosine cross-correlation localizes at
GT via its MINIMUM (sign-inverted; the refine conv flips it). We do NOT assume the same
sign holds at stride-8 — for each granularity we pick the localizing extremum (argmin vs
argmax) from the FULL-feature hit rate, then apply that sign consistently across the rank
sweep. This prevents a sign mismatch from faking a low matching dimensionality.

PASS narrative: if stride-8 matching saturates at a LOWER rank than stride-4 (which needs
~64), granularity monotonically sets matching dimensionality → C2 trend confirmed within a
single estimator. The alignment point (~22 at stride-16) completes a 3-point curve.
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
K_KEY = 22
# 更细的 rank 扫描,用来定位每个 granularity 的饱和拐点(matching dimensionality)。
RANKS = [4, 8, 12, 16, 20, 22, 24, 28, 32, 40, 48, 56, 64]
CSV = 'Datasets/test_256_2k.csv'
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_B/config.yaml'
CKPT = 'Checkpoint_B_256/model_epoch_4.pth'
SEED = 2024


def load_model():
    model = build_model_from_yaml(CONFIG, _Stage2Args())
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = ck.get('model_0', ck)
    miss, unexp = model.trainable_tail.load_state_dict(state, strict=False)
    print(f'  loaded {CKPT}: missing={len(miss)} unexpected={len(unexp)}', flush=True)
    return model.to(DEV).eval()


def inv_sqrt(C, eps=1e-4):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def pca_basis(Xc, d):
    C = Xc.T @ Xc / Xc.shape[0]
    w, V = np.linalg.eigh(C)
    return V[:, np.argsort(w)[::-1][:d]]


def fit_cca(Xtr, Ytr, d):
    """Return feature-space canonical loadings Wx,Wy (1024,d) + train means."""
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    Px, Py = pca_basis(Xc, d), pca_basis(Yc, d)
    Xr, Yr = Xc @ Px, Yc @ Py
    Cxx, Cyy = Xr.T @ Xr / len(Xr), Yr.T @ Yr / len(Yr)
    Cxy = Xr.T @ Yr / len(Xr)
    Kx, Ky = inv_sqrt(Cxx), inv_sqrt(Cyy)
    U, s, Vt = np.linalg.svd(Kx @ Cxy @ Ky)
    Wx = Px @ Kx @ U          # (1024,d) maps centered feat -> canonical variate
    Wy = Py @ Ky @ Vt.T
    return Wx, Wy, mx, my, s


def canon_corr(X, Y, Wx, Wy, mx, my, shuffle_rows=None):
    ax, ay = (X - mx) @ Wx, (Y - my) @ Wy
    if shuffle_rows is not None:
        ay = ay[shuffle_rows]
    return np.array([np.corrcoef(ax[:, k], ay[:, k])[0, 1] for k in range(ax.shape[1])])


def orthobasis(W, k):
    """Orthonormal basis (1024,k) of the top-k canonical loading directions."""
    Q, _ = np.linalg.qr(W[:, :k])
    return Q[:, :k]


def both_hits(M, gr, gc):
    """Return (min_hit, max_hit): is argmin / argmax within 1 cell of GT (gr,gc)."""
    h, w = M.shape[-2:]
    flat = M.view(-1)
    imn = int(flat.argmin().item())
    imx = int(flat.argmax().item())
    mn = int(max(abs(imn // w - gr), abs(imn % w - gc)) <= 1)
    mx = int(max(abs(imx // w - gr), abs(imx % w - gc)) <= 1)
    return mn, mx


def saturation_rank(curve, full, floor, frac):
    """First rank whose CHANCE-NORMALIZED recovery (sustainedly) reaches frac.

    recovery(k) = (hit(k) - floor) / (full - floor). Floor-normalization removes the
    cross-stride confound that the stride-8 9×9 map has a ~3.6× higher chance floor
    than the stride-4 17×17 map. Requires a SUSTAINED crossing (all subsequent ranks
    also above frac) so a non-monotone blip can't report a spuriously low rank.
    """
    denom = max(full - floor, 1e-9)
    rec = [(v - floor) / denom for v in curve]
    for j, k in enumerate(RANKS):
        if all(c >= frac for c in rec[j:]):
            return k
    return None


def _fmt_rank(k):
    """Render a saturation rank, distinguishing 'late' from 'never saturates'."""
    return f'{k}' if k is not None else f'>{RANKS[-1]} (no saturation)'


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    model = load_model()
    mh = model.matching
    df = pd.read_csv(CSV)
    n = min(N, len(df))

    # ---- pass 1: extract raw stride-16 feats + GT (stride-4 AND stride-8 offsets) ----
    rgb16, sar16, gts, gts4, gts8, paths = [], [], [], [], [], []
    for i in range(n):
        r = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(r['Label']))
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)
                               ).unsqueeze(0).to(DEV)
        with torch.no_grad():
            f_rgb = model.rgb_encoder(rgb)[-1]       # (1,1024,16,16)
            f_sar = model.sar_encoder(sar)['sar16']   # (1,1024,12,12)
        _, C, sh, sw = f_sar.shape
        oh, ow = f_rgb.shape[-2:]
        gr = min(max(int(round(gy / 16)), 0), oh - sh)   # stride-16 offset (CCA cells)
        gc = min(max(int(round(gx / 16)), 0), ow - sw)
        g4r = min(max(int(round(gy / 4)), 0), (oh - sh) * 4)  # stride-4 offset
        g4c = min(max(int(round(gx / 4)), 0), (ow - sw) * 4)
        g8r = min(max(int(round(gy / 8)), 0), (oh - sh) * 2)  # stride-8 offset
        g8c = min(max(int(round(gx / 8)), 0), (ow - sw) * 2)
        rgb16.append(f_rgb[0].cpu()); sar16.append(f_sar[0].cpu())
        gts.append((gr, gc)); gts4.append((g4r, g4c)); gts8.append((g8r, g8c))
        paths.append((r['SAR_Image'], r['RGB_Image']))
        if (i + 1) % 100 == 0:
            print(f'  extracted {i + 1}/{n}', flush=True)
    sh, sw = sar16[0].shape[-2:]

    # GT-corresponding cells, tagged by image id
    Xc, Yc, img_id = [], [], []
    for i in range(n):
        gr, gc = gts[i]
        d = rgb16[i][:, gr:gr + sh, gc:gc + sw].reshape(1024, -1).t().numpy()  # (144,1024)
        s = sar16[i].reshape(1024, -1).t().numpy()
        Xc.append(d); Yc.append(s); img_id.append(np.full(d.shape[0], i))
    X = np.concatenate(Xc).astype(np.float64)
    Y = np.concatenate(Yc).astype(np.float64)
    img_id = np.concatenate(img_id)

    # ---- E3: image-holdout CCA + pair-permutation null (identical to keystone) ----
    imgs = np.arange(n); rng.shuffle(imgs)
    tr_imgs = set(imgs[:n // 2].tolist())
    tr = np.array([k for k in range(len(X)) if img_id[k] in tr_imgs])
    te = np.array([k for k in range(len(X)) if img_id[k] not in tr_imgs])
    Wx, Wy, mx, my, s_tr = fit_cca(X[tr], Y[tr], D_PCA)
    rho_img = canon_corr(X[te], Y[te], Wx, Wy, mx, my)

    te_imgs = [k for k in range(n) if k not in tr_imgs]
    perm_map = {im: te_imgs[(j + 1) % len(te_imgs)] for j, im in enumerate(te_imgs)}
    Yperm = Y.copy()
    for im in te_imgs:
        src = perm_map[im]
        Yperm[img_id == im] = Y[img_id == src][:np.sum(img_id == im)]
    rho_null = canon_corr(X[te], Yperm[te], Wx, Wy, mx, my)

    print('\n==== E3: de-leaked CCA (shared cross-modal subspace) ====', flush=True)
    print(f'  image-holdout canonical r (top8): ' + '  '.join(f'{v:+.2f}' for v in rho_img[:8]), flush=True)
    print(f'  pair-permutation null  (top8): ' + '  '.join(f'{v:+.2f}' for v in rho_null[:8]), flush=True)
    eff = int(np.sum(rho_img > (np.abs(rho_null) + 0.05)))
    print(f'  → shared dims surviving de-leak (img-r > null+0.05): {eff}/{D_PCA};  top r={rho_img[0]:+.3f}', flush=True)

    # ---- E2: subspace-restricted matching through the REAL proj, per granularity ----
    mx_t = torch.tensor(mx, dtype=torch.float32, device=DEV)
    my_t = torch.tensor(my, dtype=torch.float32, device=DEV)

    def project_feat(fmap, Q, mu):
        b, c, h, w = fmap.shape
        x = fmap.reshape(c, h * w).t()
        xc = x - mu
        xr = xc @ Q @ Q.t()
        return (xr + mu).t().reshape(1, c, h, w)

    def comp_feat(fmap, Q, mu):
        b, c, h, w = fmap.shape
        x = fmap.reshape(c, h * w).t()
        xc = x - mu
        xr = xc - xc @ Q @ Q.t()
        return (xr + mu).t().reshape(1, c, h, w)

    def match_hits(f_rgb, f_sar, rgb_img, sar_img, gr_, gc_, stride):
        """Raw (pre-refine) cosine cross-corr; return (min_hit, max_hit) at given stride."""
        with torch.no_grad():
            if stride == 4:
                rgb_p = mh.proj_rgb_4(f_rgb, rgb_img)
                sar_p = mh.proj_sar_4(f_sar, sar_img)
            else:  # stride == 8
                rgb_p = mh.proj_rgb_8(f_rgb, rgb_img)
                sar_p = mh.proj_sar_8(f_sar, sar_img)
            M = F.conv2d(rgb_p, sar_p)
        return both_hits(M, gr_, gc_)

    Qd = {k: torch.tensor(orthobasis(Wx, k), dtype=torch.float32, device=DEV) for k in RANKS}
    Qs = {k: torch.tensor(orthobasis(Wy, k), dtype=torch.float32, device=DEV) for k in RANKS}

    # one-time forward to record each stride's cross-corr map size → chance floor.
    # within-1-cell acceptance = up to 3×3=9 positions; chance = 9 / (#cells).
    chance, mapsz = {}, {}
    i0 = te_imgs[0]
    sp0, rp0 = paths[i0]
    sar0 = preprocess_image(read_image(sp0, ImageReadMode.GRAY),
                            duplicate_channels=True).unsqueeze(0).to(DEV)
    rgb0 = preprocess_image(read_image(rp0, ImageReadMode.RGB)).unsqueeze(0).to(DEV)
    fr0, fs0 = rgb16[i0].unsqueeze(0).to(DEV), sar16[i0].unsqueeze(0).to(DEV)
    with torch.no_grad():
        for stride in (4, 8):
            pr_, ps_ = ((mh.proj_rgb_4, mh.proj_sar_4) if stride == 4
                        else (mh.proj_rgb_8, mh.proj_sar_8))
            M0 = F.conv2d(pr_(fr0, rgb0), ps_(fs0, sar0))
            h0, w0 = M0.shape[-2:]
            mapsz[stride] = (h0, w0)
            chance[stride] = 9.0 / (h0 * w0)

    # accumulate per-stride: each entry is a list of (min_hit, max_hit) over held-out imgs
    keys = ['full', f'comp{K_KEY}'] + [f'rank{k}' for k in RANKS]
    acc = {s: {key: [] for key in keys} for s in (4, 8)}
    for i in te_imgs:
        sp, rp = paths[i]
        sar = preprocess_image(read_image(sp, ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(rp, ImageReadMode.RGB)).unsqueeze(0).to(DEV)
        f_rgb = rgb16[i].unsqueeze(0).to(DEV)
        f_sar = sar16[i].unsqueeze(0).to(DEV)
        pr = {k: project_feat(f_rgb, Qd[k], mx_t) for k in RANKS}
        ps = {k: project_feat(f_sar, Qs[k], my_t) for k in RANKS}
        cr = comp_feat(f_rgb, Qd[K_KEY], mx_t)
        cs = comp_feat(f_sar, Qs[K_KEY], my_t)
        for stride, gts_s in ((4, gts4), (8, gts8)):
            gr_, gc_ = gts_s[i]
            acc[stride]['full'].append(match_hits(f_rgb, f_sar, rgb, sar, gr_, gc_, stride))
            acc[stride][f'comp{K_KEY}'].append(match_hits(cr, cs, rgb, sar, gr_, gc_, stride))
            for k in RANKS:
                acc[stride][f'rank{k}'].append(match_hits(pr[k], ps[k], rgb, sar, gr_, gc_, stride))

    print('\n==== E2: matching dimensionality per granularity (held-out images) ====', flush=True)
    summary = {}
    for stride in (4, 8):
        fl = chance[stride]
        arr_full = np.array(acc[stride]['full'])            # (Nte, 2): [min_hit, max_hit]
        full_min, full_max = arr_full.mean(0)
        col = 0 if full_min >= full_max else 1              # localizing sign for this stride
        sign = 'MIN(valley)' if col == 0 else 'MAX(peak)'
        full = float(arr_full.mean(0)[col])
        comp = float(np.array(acc[stride][f'comp{K_KEY}']).mean(0)[col])
        curve = [float(np.array(acc[stride][f'rank{k}']).mean(0)[col]) for k in RANKS]
        denom = max(full - fl, 1e-9)
        rec = [(v - fl) / denom for v in curve]            # chance-normalized recovery
        r80 = saturation_rank(curve, full, fl, 0.80)
        r90 = saturation_rank(curve, full, fl, 0.90)
        sub22 = curve[RANKS.index(K_KEY)]
        rec22 = rec[RANKS.index(K_KEY)]
        summary[stride] = dict(full=full, comp=comp, chance=fl, curve=curve, rec=rec,
                               r80=r80, r90=r90, sub22=sub22, rec22=rec22, sign=sign)
        h0, w0 = mapsz[stride]
        print(f'\n  --- stride-{stride}  map={h0}x{w0} chance={fl:.3f}  '
              f'(localizing extremum: {sign}; full min={full_min:.3f}/max={full_max:.3f}) ---', flush=True)
        print(f'    FULL hit@GT = {full:.3f}   complement(k={K_KEY}) = {comp:.3f}', flush=True)
        print(f'    sub@{K_KEY}: hit={sub22:.3f}  recovery={rec22:+.2f}', flush=True)
        print(f'    saturation (chance-normalized recovery): 0.80 @ k={_fmt_rank(r80)}   '
              f'0.90 @ k={_fmt_rank(r90)}', flush=True)
        print('    rank-k sweep  [hit@GT | recovery]:', flush=True)
        for k, v, rv in zip(RANKS, curve, rec):
            bar = '#' * int(round(40 * max(rv, 0.0)))
            print(f'      k={k:3d}  hit={v:.3f}  rec={rv:+.2f}  {bar}', flush=True)

    print('\n==== TREND (granularity → matching dimensionality) ====', flush=True)
    print(f'  alignment (stride-16) capacity ≈ {eff} (CCA de-leaked shared dims)', flush=True)
    print(f'  stride-8 matching: k≈{_fmt_rank(summary[8]["r90"])} for 0.90 recovery '
          f'(full={summary[8]["full"]:.3f}, chance={summary[8]["chance"]:.3f}, sign={summary[8]["sign"]})', flush=True)
    print(f'  stride-4 matching: k≈{_fmt_rank(summary[4]["r90"])} for 0.90 recovery '
          f'(full={summary[4]["full"]:.3f}, chance={summary[4]["chance"]:.3f}, sign={summary[4]["sign"]})', flush=True)
    print('\nGRANULARITY_DIM_DONE', flush=True)


if __name__ == '__main__':
    main()
