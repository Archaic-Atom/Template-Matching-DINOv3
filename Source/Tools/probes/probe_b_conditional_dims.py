# -*- coding: utf-8 -*-
"""Conditional / sign-flipping shared dimensions (C2 mechanism, experiment ②).

把 §IV-E P2 的假设「matching 多出的 ~40 维 = 跨模态相关性随场景翻转符号/增益的方向」
从 hypothesis 变成 measurement。

Setup (reuses probe_b_keystone's de-leaked CCA machinery): fit ONE global CCA between the
GT-corresponding stride-16 DINO and SAR cells (train-image half), giving D_PCA canonical
directions. For each direction k we compare:
  - GLOBAL correlation rho_k  (corr over ALL test cells; this is what the global CCA / the
    "~17 aligned dims" sees — a low value can mean either no signal OR sign-cancellation).
  - PER-IMAGE correlation r_k(i) (corr over image i's GT cells), summarised by
    mean|r_k(i)| (per-scene magnitude) and the sign-flip rate vs rho_k's sign.
A direction that is genuinely "conditional / sign-flipping" has mean|r_k| >> |rho_k|
(per-scene signal that cancels globally) with a high sign-flip rate — whereas a truly
"aligned" direction has mean|r_k| ≈ |rho_k| (consistent sign across scenes).

Null: pair-permutation (re-pair DINO_i with SAR_j, j≠i). If per-scene mean|r_k| collapses
to the null, the per-scene correlation is genuine cross-modal structure, not spatial
autocorrelation. Group contrast = ALIGNED dims (global-significant) vs EXTRA dims (the
rest of the 64) confirms whether the extra matching dims are the conditional ones.

No model training; only a forward pass to extract stride-16 features + CCA/correlation.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
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
CSV = 'Datasets/test_256_2k.csv'
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_B/config.yaml'
CKPT = 'Checkpoint_B_256/model_epoch_4.pth'
SEED = 2024
MIN_CELLS = 16  # skip a direction/image with too few cells for a stable corr


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


def colwise_corr(A, B):
    """Per-column Pearson r between (m,d) A and B; nan if a column is constant."""
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    num = (Ac * Bc).sum(0)
    den = np.sqrt((Ac ** 2).sum(0) * (Bc ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = num / den
    return r  # (d,)


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    model = load_model()
    df = pd.read_csv(CSV)
    n = min(N, len(df))

    # ---- extract raw stride-16 feats + GT (identical to keystone) ----
    rgb16, sar16, gts = [], [], []
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
        _, C, sh, sw = f_sar.shape
        oh, ow = f_rgb.shape[-2:]
        gr = min(max(int(round(gy / 16)), 0), oh - sh)
        gc = min(max(int(round(gx / 16)), 0), ow - sw)
        rgb16.append(f_rgb[0].cpu()); sar16.append(f_sar[0].cpu()); gts.append((gr, gc))
        if (i + 1) % 100 == 0:
            print(f'  extracted {i + 1}/{n}', flush=True)
    sh, sw = sar16[0].shape[-2:]

    Xc, Yc, img_id = [], [], []
    for i in range(n):
        gr, gc = gts[i]
        d = rgb16[i][:, gr:gr + sh, gc:gc + sw].reshape(1024, -1).t().numpy()
        s = sar16[i].reshape(1024, -1).t().numpy()
        Xc.append(d); Yc.append(s); img_id.append(np.full(d.shape[0], i))
    X = np.concatenate(Xc).astype(np.float64)
    Y = np.concatenate(Yc).astype(np.float64)
    img_id = np.concatenate(img_id)

    # ---- global CCA on train-image half (de-leaked split) ----
    imgs = np.arange(n); rng.shuffle(imgs)
    tr_imgs = set(imgs[:n // 2].tolist())
    tr = np.array([k for k in range(len(X)) if img_id[k] in tr_imgs])
    Wx, Wy, mx, my, s_tr = fit_cca(X[tr], Y[tr], D_PCA)
    te_imgs = [k for k in range(n) if k not in tr_imgs]

    # canonical variates on ALL cells (centered with train means)
    AX = (X - mx) @ Wx   # (cells, D_PCA)
    AY = (Y - my) @ Wy

    # GLOBAL per-direction correlation over all test cells
    te_mask = np.array([img_id[k] not in tr_imgs for k in range(len(X))])
    rho_global = colwise_corr(AX[te_mask], AY[te_mask])  # (D_PCA,)

    # PER-IMAGE per-direction correlation (genuine pairing + pair-permutation null)
    perimg = []        # list of (D_PCA,) r per test image
    perimg_null = []
    order = list(te_imgs)
    for j, im in enumerate(order):
        cells = (img_id == im)
        ax = AX[cells]; ay = AY[cells]
        if ax.shape[0] < MIN_CELLS:
            continue
        perimg.append(colwise_corr(ax, ay))
        # pair-permutation: this image's DINO variate vs NEXT test image's SAR variate
        src = order[(j + 1) % len(order)]
        ay_src = AY[img_id == src]
        m = min(ax.shape[0], ay_src.shape[0])
        perimg_null.append(colwise_corr(ax[:m], ay_src[:m]))
    R = np.array(perimg)        # (Nimg, D_PCA)
    Rn = np.array(perimg_null)  # (Nimg, D_PCA)

    # per-direction summaries (ignore nan columns from constant variates)
    def nanmean0(a):
        return np.nanmean(a, axis=0)

    mean_r = nanmean0(R)                       # signed per-image mean
    mean_abs = nanmean0(np.abs(R))             # per-scene magnitude
    mean_abs_null = nanmean0(np.abs(Rn))
    # sign-flip rate vs the global sign of each direction
    gsign = np.sign(rho_global)
    gsign[gsign == 0] = 1
    flip = nanmean0((np.sign(R) != gsign[None, :]).astype(float))
    # "cancellation gap": how much per-scene magnitude exceeds the global |corr|
    cancel_gap = mean_abs - np.abs(rho_global)

    # ALIGNED vs EXTRA grouping: aligned = globally significant vs null
    aligned = np.abs(rho_global) > (mean_abs_null + 0.05)
    extra = (~aligned) & (mean_abs > (mean_abs_null + 0.05))  # per-scene signal, weak global
    n_align = int(aligned.sum()); n_extra = int(extra.sum())

    print('\n==== experiment ②: conditional / sign-flipping shared dimensions ====', flush=True)
    print(f'  test images used: {R.shape[0]};  directions: {D_PCA}', flush=True)
    print('\n  per-direction (k: |global_r|  mean|r_img|  flip%  cancel_gap  null|r|):', flush=True)
    for k in range(D_PCA):
        tag = 'ALIGN' if aligned[k] else ('EXTRA' if extra[k] else '  -  ')
        if k < 24 or extra[k]:
            print(f'    k={k:3d} [{tag}]  |g|={abs(rho_global[k]):.3f}  '
                  f'|r_img|={mean_abs[k]:.3f}  flip={100*flip[k]:4.1f}%  '
                  f'gap={cancel_gap[k]:+.3f}  null={mean_abs_null[k]:.3f}', flush=True)

    def grp(mask, name):
        if mask.sum() == 0:
            print(f'  {name:8s}: (none)', flush=True); return
        print(f'  {name:8s} (n={int(mask.sum())}): '
              f'|global_r|={np.nanmean(np.abs(rho_global[mask])):.3f}  '
              f'mean|r_img|={np.nanmean(mean_abs[mask]):.3f}  '
              f'flip={100*np.nanmean(flip[mask]):.1f}%  '
              f'cancel_gap={np.nanmean(cancel_gap[mask]):+.3f}  '
              f'null|r|={np.nanmean(mean_abs_null[mask]):.3f}', flush=True)

    print('\n  ==== GROUP CONTRAST (the measurement) ====', flush=True)
    grp(aligned, 'ALIGNED')
    grp(extra, 'EXTRA')
    print('\n  Hypothesis CONFIRMED if EXTRA has mean|r_img| >> |global_r| (large cancel_gap),', flush=True)
    print('  high flip%, and mean|r_img| >> null|r|  (per-scene signal that cancels globally).', flush=True)
    print('\nCONDITIONAL_DIMS_DONE', flush=True)


if __name__ == '__main__':
    main()
