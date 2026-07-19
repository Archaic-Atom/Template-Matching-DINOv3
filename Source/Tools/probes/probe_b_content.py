# -*- coding: utf-8 -*-
"""WHY are raw DINO/SAR stride-16 features orthogonal yet projectable into a
matching subspace — and WHAT content do they share?
为什么 DINO/SAR 原始 stride-16 特征正交、却能投影到同一匹配子空间？共享的是什么内容？

Two measurements on B ep4 over GT-aligned corresponding cells:
  (1) CCA between RAW DINO rgb_16 and RAW SAR sar16, with a held-out split and a
      shuffled-pair NULL. Raw mean cosine ~0 averages over all directions; CCA finds
      the SPECIFIC linear combinations that DO correlate = the shared subspace. If
      held-out canonical correlations >> shuffled null, a genuine low-dim shared
      linear subspace exists (the thing the learned projection exposes).
      CCA 找原始特征里"确实相关的线性组合"= 共享子空间；留出集 + 打乱 null 排除过拟合。
  (2) STRUCTURE test: per-cell shared-subspace activation vs optical gradient
      magnitude (Sobel). If structural/edge cells carry the shared signal, the
      common content is geometric structure, not appearance.
      共享子空间激活 与 光学梯度幅值 的相关 → 共享内容是否结构性。
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
N = 300                 # pairs (×144 cells/pair ≈ 43k corresponding cell pairs)
D_PCA = 64              # PCA dim before CCA
N_SHOW = 10             # canonical correlations to report
CSV = 'Datasets/test_256_2k.csv'
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_B/config.yaml'
CKPT = 'Checkpoint_B_256/model_epoch_4.pth'
SEED = 2024


def load_model():
    model = build_model_from_yaml(CONFIG, _Stage2Args())
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_0', ckpt)
    miss, unexp = model.trainable_tail.load_state_dict(state, strict=False)
    print(f'  loaded {CKPT}: missing={len(miss)} unexpected={len(unexp)}', flush=True)
    return model.to(DEV).eval()


# Sobel gradient magnitude of a (1,3,H,W) image → (H,W)
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
_SOBEL_Y = _SOBEL_X.t().contiguous()


def grad_mag_cells(rgb, stride, gh, gw, gr, gc):
    """Mean Sobel-gradient magnitude per stride-cell over a gh×gw window at (gr,gc)."""
    gray = rgb.mean(dim=1, keepdim=True)                      # (1,1,H,W)
    kx = _SOBEL_X.view(1, 1, 3, 3).to(gray.device)
    ky = _SOBEL_Y.view(1, 1, 3, 3).to(gray.device)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy)                       # (1,1,H,W)
    cell = F.avg_pool2d(mag, kernel_size=stride, stride=stride)  # (1,1,H/s,W/s)
    win = cell[0, 0, gr:gr + gh, gc:gc + gw]                  # (gh,gw)
    return win.reshape(-1).detach().cpu().numpy()


def inv_sqrt(C, eps=1e-4):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def pca_basis(Xc, d):
    C = Xc.T @ Xc / Xc.shape[0]
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:d]
    return V[:, idx]


def cca_heldout(X, Y, d, seed):
    """Fit CCA on a train split, return held-out canonical correlations + a
    shuffled-pair null. X,Y: (M, p)/(M, q) corresponding rows."""
    rng = np.random.RandomState(seed)
    M = X.shape[0]
    perm = rng.permutation(M)
    tr, te = perm[:M // 2], perm[M // 2:]
    mx, my = X[tr].mean(0), Y[tr].mean(0)
    Xtr, Ytr = X[tr] - mx, Y[tr] - my
    Px, Py = pca_basis(Xtr, d), pca_basis(Ytr, d)
    Xr, Yr = Xtr @ Px, Ytr @ Py
    Cxx = Xr.T @ Xr / Xr.shape[0]
    Cyy = Yr.T @ Yr / Yr.shape[0]
    Cxy = Xr.T @ Yr / Xr.shape[0]
    Kx, Ky = inv_sqrt(Cxx), inv_sqrt(Cyy)
    U, s, Vt = np.linalg.svd(Kx @ Cxy @ Ky)
    Wx = Px @ Kx @ U                  # (p, d) maps centered X → canonical variates
    Wy = Py @ Ky @ Vt.T               # (q, d)

    def variates(Xte, Yte, shuffle=False):
        ax = (Xte - mx) @ Wx
        ay = (Yte - my) @ Wy
        if shuffle:
            ay = ay[rng.permutation(ay.shape[0])]
        # per-component Pearson correlation on held-out
        out = []
        for k in range(d):
            a, b = ax[:, k], ay[:, k]
            out.append(np.corrcoef(a, b)[0, 1])
        return np.array(out), ax, ay

    rho_true, ax, ay = variates(X[te], Y[te], shuffle=False)
    rho_null, _, _ = variates(X[te], Y[te], shuffle=True)
    return rho_true, rho_null, (te, ax, ay)


def main():
    torch.manual_seed(SEED)
    model = load_model()
    df = pd.read_csv(CSV)
    n = min(N, len(df))

    dino_cells, sar_cells, grad_cells = [], [], []
    raw_cos = []
    for i in range(n):
        r = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(r['Label']))
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)
                               ).unsqueeze(0).to(DEV)
        with torch.no_grad():
            rgb_16 = model.rgb_encoder(rgb)[-1]          # (1,1024,16,16) raw
            sar_16 = model.sar_encoder(sar)['sar16']      # (1,1024,12,12) L2-normed
        _, C, sh, sw = sar_16.shape
        oh, ow = rgb_16.shape[-2:]
        gr = min(max(int(round(gy / 16)), 0), oh - sh)
        gc = min(max(int(round(gx / 16)), 0), ow - sw)
        dwin = rgb_16[0, :, gr:gr + sh, gc:gc + sw]       # (1024,sh,sw) corresponding
        swin = sar_16[0]                                  # (1024,sh,sw)
        d = dwin.reshape(C, -1).t().contiguous()          # (sh*sw, 1024)
        s = swin.reshape(C, -1).t().contiguous()
        # reconfirm raw mean cosine of corresponding cells
        raw_cos.append(float((F.normalize(d, dim=1) * F.normalize(s, dim=1)).sum(1).mean()))
        dino_cells.append(d.cpu().numpy())
        sar_cells.append(s.cpu().numpy())
        grad_cells.append(grad_mag_cells(rgb, 16, sh, sw, gr, gc))
        if (i + 1) % 100 == 0:
            print(f'  collected {i + 1}/{n}', flush=True)

    X = np.concatenate(dino_cells, 0).astype(np.float64)   # (M,1024) DINO
    Y = np.concatenate(sar_cells, 0).astype(np.float64)    # (M,1024) SAR
    G = np.concatenate(grad_cells, 0).astype(np.float64)   # (M,)  optical grad mag
    M = X.shape[0]
    print(f'\n  corresponding cells M={M}  raw mean cosine(DINO,SAR)={np.mean(raw_cos):+.4f}', flush=True)

    rho_true, rho_null, (te, ax, ay) = cca_heldout(X, Y, D_PCA, SEED)
    eff = int(np.sum(rho_true > (rho_null + 0.05)))

    print('\n==== (1) CCA: shared linear subspace between RAW DINO & RAW SAR ====', flush=True)
    print(f'  held-out canonical correlations (top {N_SHOW}):', flush=True)
    print('    true :', '  '.join(f'{v:+.2f}' for v in rho_true[:N_SHOW]), flush=True)
    print('    null :', '  '.join(f'{v:+.2f}' for v in rho_null[:N_SHOW]), flush=True)
    print(f'  → shared dims (true>null+0.05): {eff} / {D_PCA};  top canonical r={rho_true[0]:+.3f} '
          f'(null {rho_null[0]:+.3f})', flush=True)
    print('    (raw mean cos≈0 averages over all dirs; CCA exposes the aligned subspace)', flush=True)

    # ---- (2) structure: shared-subspace activation vs optical gradient ----
    g_te = G[te]
    act = 0.5 * (np.abs(ax[:, 0]) + np.abs(ay[:, 0]))      # top-canonical activation per cell
    r_struct = np.corrcoef(act, g_te)[0, 1]
    lo, hi = np.percentile(g_te, 33), np.percentile(g_te, 67)
    act_lo = act[g_te <= lo].mean()
    act_hi = act[g_te >= hi].mean()
    print('\n==== (2) WHAT content: shared subspace vs optical structure (gradient) ====', flush=True)
    print(f'  corr(top-canonical activation, optical gradient magnitude) = {r_struct:+.3f}', flush=True)
    print(f'  mean activation — low-gradient cells={act_lo:.3f}  high-gradient cells={act_hi:.3f}  '
          f'(ratio {act_hi / (act_lo + 1e-9):.2f}x)', flush=True)
    print('    (positive corr / hi>lo ⇒ shared content concentrates on structural/edge cells)', flush=True)
    print('\nCONTENT_DONE', flush=True)


if __name__ == '__main__':
    main()
