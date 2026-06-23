# -*- coding: utf-8 -*-
"""KEYSTONE (E2) + de-leaked CCA (E3): does B's matching actually USE the shared
cross-modal subspace, and is that subspace real (not spatial-autocorrelation leakage)?
命门实验：B 的匹配是否真的用那个跨模态共享子空间，且该子空间是否真实（非空间自相关泄漏）。

E3 (de-leak): CCA between raw DINO rgb_16 and raw SAR sar16 on GT-corresponding cells,
  but split by IMAGE (not cell) + a PAIR-PERMUTATION null (re-pair DINO_i with SAR_j,
  j!=i, same GT-window positions). If the canonical correlation survives image-holdout
  but collapses under pair-permutation, it is genuine cross-modal — not same-scene layout.

E2 (keystone): take the de-leaked rank-k shared subspace (orthonormal basis Q_dino,
  Q_sar in 1024-d feature space). Project each raw stride-16 cell onto it, feed the
  RESTRICTED feature through the REAL proj_rgb_4 / proj_sar_4 (JBU+1x1) + cosine
  cross-correlation, measure valley@GT. Also the orthogonal COMPLEMENT, and a rank-k
  sweep. PASS = in-subspace valley@GT >= 0.8 * full AND complement <= ~chance, with a
  knee near k~22 → the matcher's signal lives in the shared subspace.
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
RANKS = [4, 8, 16, 22, 32, 64]
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


def valley_hit(M, gr, gc):
    h, w = M.shape[-2:]
    idx = int(M.view(-1).argmin().item())
    return int(max(abs(idx // w - gr), abs(idx % w - gc)) <= 1)


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    model = load_model()
    mh = model.matching
    df = pd.read_csv(CSV)
    n = min(N, len(df))

    # ---- pass 1: extract raw stride-16 feats + GT, keep image paths ----
    rgb16, sar16, gts, gts4, paths = [], [], [], [], []
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
        g4r = min(max(int(round(gy / 4)), 0), (oh - sh) * 4)  # stride-4 offset (matching)
        g4c = min(max(int(round(gx / 4)), 0), (ow - sw) * 4)
        rgb16.append(f_rgb[0].cpu()); sar16.append(f_sar[0].cpu())
        gts.append((gr, gc)); gts4.append((g4r, g4c))
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

    # ---- E3: image-holdout CCA + pair-permutation null ----
    imgs = np.arange(n); rng.shuffle(imgs)
    tr_imgs = set(imgs[:n // 2].tolist())
    tr = np.array([k for k in range(len(X)) if img_id[k] in tr_imgs])
    te = np.array([k for k in range(len(X)) if img_id[k] not in tr_imgs])
    Wx, Wy, mx, my, s_tr = fit_cca(X[tr], Y[tr], D_PCA)
    rho_img = canon_corr(X[te], Y[te], Wx, Wy, mx, my)

    # pair-permutation null: re-pair each TEST image's SAR window with another image's
    te_imgs = [k for k in range(n) if k not in tr_imgs]
    perm_map = {im: te_imgs[(j + 1) % len(te_imgs)] for j, im in enumerate(te_imgs)}
    Yperm = Y.copy()
    for im in te_imgs:
        src = perm_map[im]
        Yperm[img_id == im] = Y[img_id == src][:np.sum(img_id == im)]
    rho_null = canon_corr(X[te], Yperm[te], Wx, Wy, mx, my)

    print('\n==== E3: de-leaked CCA (image-holdout split + pair-permutation null) ====', flush=True)
    print(f'  raw mean cos(DINO,SAR) = {np.mean((F.normalize(torch.tensor(X),dim=1)*F.normalize(torch.tensor(Y),dim=1)).sum(1).numpy()):+.4f}', flush=True)
    print(f'  image-holdout canonical r (top8): ' + '  '.join(f'{v:+.2f}' for v in rho_img[:8]), flush=True)
    print(f'  pair-permutation null  (top8): ' + '  '.join(f'{v:+.2f}' for v in rho_null[:8]), flush=True)
    eff = int(np.sum(rho_img > (np.abs(rho_null) + 0.05)))
    print(f'  → shared dims surviving de-leak (img-r > null+0.05): {eff}/{D_PCA};  top r={rho_img[0]:+.3f} (null {rho_null[0]:+.3f})', flush=True)

    # ---- E2: subspace-restricted matching through the REAL proj + cross-corr ----
    # use TRAIN-fit canonical loadings → orthobases in 1024-d feature space
    mx_t = torch.tensor(mx, dtype=torch.float32, device=DEV)
    my_t = torch.tensor(my, dtype=torch.float32, device=DEV)

    def project_feat(fmap, Q, mu):
        """fmap (1,1024,H,W) → keep only the span(Q) component (per cell), add mean back."""
        b, c, h, w = fmap.shape
        x = fmap.reshape(c, h * w).t()                  # (HW,1024)
        xc = x - mu
        xr = xc @ Q @ Q.t()                             # project onto span(Q)
        return (xr + mu).t().reshape(1, c, h, w)

    def comp_feat(fmap, Q, mu):
        b, c, h, w = fmap.shape
        x = fmap.reshape(c, h * w).t()
        xc = x - mu
        xr = xc - xc @ Q @ Q.t()                        # orthogonal complement
        return (xr + mu).t().reshape(1, c, h, w)

    def match_valley(f_rgb, f_sar, rgb_img, sar_img, g4r, g4c):
        with torch.no_grad():
            rgb_4 = mh.proj_rgb_4(f_rgb, rgb_img)
            sar_4 = mh.proj_sar_4(f_sar, sar_img)
            M = F.conv2d(rgb_4, sar_4)
        return valley_hit(M, g4r, g4c)                  # true stride-4 GT offset

    # build orthobases for each rank from TRAIN CCA
    Qd = {k: torch.tensor(orthobasis(Wx, k), dtype=torch.float32, device=DEV) for k in set(RANKS + [K_KEY])}
    Qs = {k: torch.tensor(orthobasis(Wy, k), dtype=torch.float32, device=DEV) for k in set(RANKS + [K_KEY])}

    res = {'full': [], 'sub22': [], 'comp22': []}
    res.update({f'rank{k}': [] for k in RANKS})
    for i in te_imgs:                                    # evaluate on held-out images only
        sp, rp = paths[i]
        sar = preprocess_image(read_image(sp, ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(rp, ImageReadMode.RGB)).unsqueeze(0).to(DEV)
        g4r, g4c = gts4[i]
        f_rgb = rgb16[i].unsqueeze(0).to(DEV)
        f_sar = sar16[i].unsqueeze(0).to(DEV)
        res['full'].append(match_valley(f_rgb, f_sar, rgb, sar, g4r, g4c))
        res['sub22'].append(match_valley(project_feat(f_rgb, Qd[K_KEY], mx_t),
                                         project_feat(f_sar, Qs[K_KEY], my_t), rgb, sar, g4r, g4c))
        res['comp22'].append(match_valley(comp_feat(f_rgb, Qd[K_KEY], mx_t),
                                          comp_feat(f_sar, Qs[K_KEY], my_t), rgb, sar, g4r, g4c))
        for k in RANKS:
            res[f'rank{k}'].append(match_valley(project_feat(f_rgb, Qd[k], mx_t),
                                                project_feat(f_sar, Qs[k], my_t), rgb, sar, g4r, g4c))

    full = float(np.mean(res['full']))
    print('\n==== E2 KEYSTONE: subspace-restricted matching (held-out images) ====', flush=True)
    print(f'  FULL features            valley@GT = {full:.3f}', flush=True)
    print(f'  shared subspace (k={K_KEY})   valley@GT = {np.mean(res["sub22"]):.3f}  '
          f'({np.mean(res["sub22"]) / (full + 1e-9):.2f}× full)', flush=True)
    print(f'  orthogonal complement    valley@GT = {np.mean(res["comp22"]):.3f}', flush=True)
    print('  rank-k sweep (shared subspace only):', flush=True)
    for k in RANKS:
        print(f'    k={k:3d}  valley@GT={np.mean(res[f"rank{k}"]):.3f}', flush=True)
    print(f'\n  PASS if sub@{K_KEY} >= 0.8×full ({0.8*full:.3f}) AND complement ≈ chance', flush=True)
    print('\nKEYSTONE_DONE', flush=True)


if __name__ == '__main__':
    main()
