# -*- coding: utf-8 -*-
"""Alignment-recovery: put the stride-16 alignment capacity on the SAME ruler
as the stride-8/4 matching-dimensionality curves.

把 stride-16 对齐容量用与 stride-8/4 匹配维度**同一把尺子**表达——
"restrict 到 top-k 共享子空间 → 任务定位命中率恢复到 0.90 所需的 k"——
使 22/48/64 三点可比(commensurable)。

WHY a separate probe (not reuse the matcher features): the trained matcher
constructs NO similarity at stride-16 (modality gap; raw cross-modal cos≈0,
§gap_baseline), so its stride-16 map does not localize at full rank and there
is nothing to "recover". The STAGE-1 aligned encoder DOES localize at
stride-16 (hit≈0.59 > chance 0.36), so the recovery question is well posed
there. Estimator is identical to ``probe_b_granularity_dim.py`` (E3 de-leaked
CCA → one shared rank-D_PCA subspace; restrict features to its top-k
directions; watch the localization hit-rate recover), only the task differs:
alignment (stride-16, cos-peak) here vs. matching (stride-8/4, learned proj)
there.

Expectation: saturation near ~22 — the alignable cross-modal capacity — now
expressed as a recovery rank rather than a CCA significant-direction count.
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

DEV = torch.device('cuda')
N = 400
D_PCA = 64
K_KEY = 22
# 与 probe_b_granularity_dim 相同的 rank 扫描,便于同尺对照。
RANKS = [4, 8, 12, 16, 20, 22, 24, 28, 32, 40, 48, 56, 64]
CSV = 'Datasets/test_256_2k.csv'
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_A/config.yaml'
CKPT = 'Checkpoint_A_stage1_256/model_epoch_5.pth'
SEED = 2024
STRIDE = 16


# ---- CCA / recovery machinery (verbatim from probe_b_granularity_dim.py) ----
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
    Wx = Px @ Kx @ U
    Wy = Py @ Ky @ Vt.T
    return Wx, Wy, mx, my, s


def canon_corr(X, Y, Wx, Wy, mx, my, shuffle_rows=None):
    ax, ay = (X - mx) @ Wx, (Y - my) @ Wy
    if shuffle_rows is not None:
        ay = ay[shuffle_rows]
    return np.array([np.corrcoef(ax[:, k], ay[:, k])[0, 1] for k in range(ax.shape[1])])


def orthobasis(W, k):
    Q, _ = np.linalg.qr(W[:, :k])
    return Q[:, :k]


def both_hits(M, gr, gc):
    """Return (min_hit, max_hit): is argmin / argmax within 1 cell of GT (gr,gc)."""
    h, w = M.shape[-2:]
    flat = M.reshape(-1)
    imn = int(flat.argmin().item())
    imx = int(flat.argmax().item())
    mn = int(max(abs(imn // w - gr), abs(imn % w - gc)) <= 1)
    mx = int(max(abs(imx // w - gr), abs(imx % w - gc)) <= 1)
    return mn, mx


def saturation_rank(curve, full, floor, frac):
    """First rank whose SUSTAINED chance-normalized recovery reaches frac."""
    denom = max(full - floor, 1e-9)
    rec = [(v - floor) / denom for v in curve]
    for j, k in enumerate(RANKS):
        if all(c >= frac for c in rec[j:]):
            return k
    return None


def _fmt_rank(k):
    return f'{k}' if k is not None else f'>{RANKS[-1]} (no saturation)'


def align_score(f_rgb, f_sar):
    """Stride-16 cosine alignment score map (L2-norm + conv2d), as in score_match.

    对齐余弦分数热图:逐 token 通道 L2 归一化后卷积 = 逐位置 cos 相似度均值。
    """
    rgb_n = F.normalize(f_rgb, dim=1)
    sar_n = F.normalize(f_sar, dim=1)
    _, c, sh, sw = sar_n.shape
    return F.conv2d(rgb_n, sar_n) / (sh * sw)


def project_feat(fmap, Q, mu):
    """Restrict a (1,C,h,w) map to the top-k shared subspace spanned by Q."""
    b, c, h, w = fmap.shape
    x = fmap.reshape(c, h * w).t()
    xc = x - mu
    xr = xc @ Q @ Q.t()
    return (xr + mu).t().reshape(1, c, h, w)


def comp_feat(fmap, Q, mu):
    """Complement: strip the top-k shared subspace (for the K_KEY control)."""
    b, c, h, w = fmap.shape
    x = fmap.reshape(c, h * w).t()
    xc = x - mu
    xr = xc - xc @ Q @ Q.t()
    return (xr + mu).t().reshape(1, c, h, w)


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    model = load_model(CKPT, CONFIG, DEV)
    df = pd.read_csv(CSV)
    n = min(N, len(df))

    # ---- pass 1: extract Stage-1 aligned stride-16 feats + GT offsets ----
    rgb16, sar16, gts, paths = [], [], [], []
    for i in range(n):
        r = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(r['Label']))
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)
                               ).unsqueeze(0).to(DEV)
        with torch.no_grad():
            out = model(sar, rgb)
        f_rgb = out['rbg_feat'][3]          # (1,1024,16,16) DINO tokens (note key typo)
        f_sar = out['sar_feat']['sar16']    # (1,1024,12,12) Stage-1 aligned SAR
        _, C, sh, sw = f_sar.shape
        oh, ow = f_rgb.shape[-2:]
        gr = min(max(int(round(gy / STRIDE)), 0), oh - sh)
        gc = min(max(int(round(gx / STRIDE)), 0), ow - sw)
        rgb16.append(f_rgb[0].cpu()); sar16.append(f_sar[0].cpu())
        gts.append((gr, gc)); paths.append((r['SAR_Image'], r['RGB_Image']))
        if i == 0:
            print(f'  feat dims: rgb={tuple(f_rgb.shape)} sar={tuple(f_sar.shape)}', flush=True)
        if (i + 1) % 100 == 0:
            print(f'  extracted {i + 1}/{n}', flush=True)
    sh, sw = sar16[0].shape[-2:]

    # GT-corresponding cell pairs, tagged by image id (same as keystone)
    Xc, Yc, img_id = [], [], []
    for i in range(n):
        gr, gc = gts[i]
        d = rgb16[i][:, gr:gr + sh, gc:gc + sw].reshape(1024, -1).t().numpy()
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

    print('\n==== E3: de-leaked CCA (shared cross-modal subspace, Stage-1 aligned) ====', flush=True)
    print('  image-holdout canonical r (top8): ' + '  '.join(f'{v:+.2f}' for v in rho_img[:8]), flush=True)
    print('  pair-permutation null  (top8): ' + '  '.join(f'{v:+.2f}' for v in rho_null[:8]), flush=True)
    eff = int(np.sum(rho_img > (np.abs(rho_null) + 0.05)))
    print(f'  → shared dims surviving de-leak (img-r > null+0.05): {eff}/{D_PCA};  top r={rho_img[0]:+.3f}', flush=True)

    # ---- recovery: subspace-restricted stride-16 alignment localization ----
    mx_t = torch.tensor(mx, dtype=torch.float32, device=DEV)
    my_t = torch.tensor(my, dtype=torch.float32, device=DEV)
    Qd = {k: torch.tensor(orthobasis(Wx, k), dtype=torch.float32, device=DEV) for k in RANKS}
    Qs = {k: torch.tensor(orthobasis(Wy, k), dtype=torch.float32, device=DEV) for k in RANKS}

    # map size → chance floor (within-1-cell = up to 9 positions).
    fr0 = rgb16[te_imgs[0]].unsqueeze(0).to(DEV)
    fs0 = sar16[te_imgs[0]].unsqueeze(0).to(DEV)
    M0 = align_score(fr0, fs0)
    h0, w0 = M0.shape[-2:]
    chance = 9.0 / (h0 * w0)

    keys = ['full', f'comp{K_KEY}'] + [f'rank{k}' for k in RANKS]
    acc = {key: [] for key in keys}
    for i in te_imgs:
        gr, gc = gts[i]
        f_rgb = rgb16[i].unsqueeze(0).to(DEV)
        f_sar = sar16[i].unsqueeze(0).to(DEV)
        acc['full'].append(both_hits(align_score(f_rgb, f_sar), gr, gc))
        cr = comp_feat(f_rgb, Qd[K_KEY], mx_t); cs = comp_feat(f_sar, Qs[K_KEY], my_t)
        acc[f'comp{K_KEY}'].append(both_hits(align_score(cr, cs), gr, gc))
        for k in RANKS:
            pr = project_feat(f_rgb, Qd[k], mx_t); psf = project_feat(f_sar, Qs[k], my_t)
            acc[f'rank{k}'].append(both_hits(align_score(pr, psf), gr, gc))

    print('\n==== recovery: stride-16 alignment dimensionality (held-out images) ====', flush=True)
    arr_full = np.array(acc['full'])
    full_min, full_max = arr_full.mean(0)
    col = 0 if full_min >= full_max else 1                  # localizing sign (align → MAX)
    sign = 'MIN(valley)' if col == 0 else 'MAX(peak)'
    full = float(arr_full.mean(0)[col])
    comp = float(np.array(acc[f'comp{K_KEY}']).mean(0)[col])
    curve = [float(np.array(acc[f'rank{k}']).mean(0)[col]) for k in RANKS]
    denom = max(full - chance, 1e-9)
    rec = [(v - chance) / denom for v in curve]
    r80 = saturation_rank(curve, full, chance, 0.80)
    r90 = saturation_rank(curve, full, chance, 0.90)
    sub22 = curve[RANKS.index(K_KEY)]; rec22 = rec[RANKS.index(K_KEY)]
    print(f'\n  --- stride-16 alignment  map={h0}x{w0} chance={chance:.3f}  '
          f'(localizing extremum: {sign}; full min={full_min:.3f}/max={full_max:.3f}) ---', flush=True)
    print(f'    FULL hit@GT = {full:.3f}   complement(k={K_KEY}) = {comp:.3f}', flush=True)
    print(f'    sub@{K_KEY}: hit={sub22:.3f}  recovery={rec22:+.2f}', flush=True)
    print(f'    saturation (chance-normalized recovery): 0.80 @ k={_fmt_rank(r80)}   '
          f'0.90 @ k={_fmt_rank(r90)}', flush=True)
    print('    rank-k sweep  [hit@GT | recovery]:', flush=True)
    for k, v, rv in zip(RANKS, curve, rec):
        bar = '#' * int(round(40 * max(rv, 0.0)))
        print(f'      k={k:3d}  hit={v:.3f}  rec={rv:+.2f}  {bar}', flush=True)

    print('\n==== TREND (granularity → representation dimensionality, ONE ruler) ====', flush=True)
    print(f'  alignment (stride-16): k≈{_fmt_rank(r90)} for 0.90 recovery '
          f'(full={full:.3f}, chance={chance:.3f}, sign={sign})   [CCA count for ref: {eff}]', flush=True)
    print('  stride-8 matching: k≈48 / stride-4 matching: k≈64  (probe_b_granularity_dim.py)', flush=True)
    print('\nALIGN_RECOVERY_DONE', flush=True)


if __name__ == '__main__':
    main()
