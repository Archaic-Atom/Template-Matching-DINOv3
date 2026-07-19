# -*- coding: utf-8 -*-
"""Are the in-domain-redundant matching dimensions useless, or do they carry OOD robustness?

Fit ONE PCA basis on the IN-DOMAIN 256-d matching output (top-16 already recover in-domain
localization fully). Then, using that SAME in-domain basis, measure the top-k recovery curve
on OOD sets (WHU cross-sensor, QXS cross-resolution). If OOD localization needs dims BEYOND
the in-domain-sufficient ~16 (recovery at k=16 << full, keeps climbing to k=64/256), then the
"in-domain-redundant" directions carry cross-domain signal — NOT useless. If OOD also saturates
by ~16, they are genuinely redundant. NO TRAINING.
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
N_PCA = 400          # in-domain images to fit the PCA basis
N_EVAL = 300         # images per eval set
RANKS = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
IN_CSV = 'Datasets/test_256_2k.csv'
EVAL_SETS = [
    ('in-domain', 'Datasets/test_256_2k.csv'),
    ('WHU-xsensor', '/data5/datasets/cross_eval/whu_opt_sar/cross_eval.csv'),
    ('QXS-xres', '/data5/datasets/cross_eval/qxs_saropt/cross_eval.csv'),
]
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_B/config.yaml'
CKPT = 'Checkpoint_B_256/model_epoch_4.pth'
SEED = 2024
PCA_CELLS_PER_IMG = 80


def load_model():
    model = build_model_from_yaml(CONFIG, _Stage2Args())
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = ck.get('model_0', ck)
    miss, unexp = model.trainable_tail.load_state_dict(state, strict=False)
    print(f'  loaded {CKPT}: missing={len(miss)} unexpected={len(unexp)}', flush=True)
    return model.to(DEV).eval()


def pca_basis(Xc, d):
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


def matching_feats(model, mh, row):
    gx, gy = (int(v) for v in json.loads(row['Label']))
    sar = preprocess_image(read_image(row['SAR_Image'], ImageReadMode.GRAY),
                           duplicate_channels=True).unsqueeze(0).to(DEV)
    rgb = preprocess_image(read_image(row['RGB_Image'], ImageReadMode.RGB)).unsqueeze(0).to(DEV)
    with torch.no_grad():
        f_rgb = model.rgb_encoder(rgb)[-1]
        f_sar = model.sar_encoder(sar)['sar16']
        rgb4 = mh.proj_rgb_4(f_rgb, rgb)
        sar4 = mh.proj_sar_4(f_sar, sar)
    oh, ow = f_rgb.shape[-2:]
    _, _, sh, sw = f_sar.shape
    g4r = min(max(int(round(gy / 4)), 0), (oh - sh) * 4)
    g4c = min(max(int(round(gx / 4)), 0), (ow - sw) * 4)
    return rgb4[0].cpu(), sar4[0].cpu(), (g4r, g4c)


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    model = load_model()
    mh = model.matching

    # ---- fit PCA basis on IN-DOMAIN matching output ----
    df_id = pd.read_csv(IN_CSV)
    n_pca = min(N_PCA, len(df_id))
    pool = []
    for i in range(n_pca):
        rgb4, sar4, _ = matching_feats(model, mh, df_id.iloc[i])
        for fmap in (rgb4, sar4):
            c, h, w = fmap.shape
            flat = fmap.reshape(c, h * w).t().numpy()
            idx = rng.choice(flat.shape[0], min(PCA_CELLS_PER_IMG, flat.shape[0]), replace=False)
            pool.append(flat[idx])
        if (i + 1) % 100 == 0:
            print(f'  [pca] {i + 1}/{n_pca}', flush=True)
    V = np.concatenate(pool).astype(np.float64)
    mu = V.mean(0)
    P = pca_basis(V - mu, 256)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEV)
    Pt = torch.tensor(P, dtype=torch.float32, device=DEV)
    print(f'  PCA basis fit on {V.shape[0]} in-domain cells.\n', flush=True)

    def restrict(fmap, k):
        b, c, h, w = fmap.shape
        x = fmap.reshape(c, h * w).t()
        xc = x - mu_t
        Qk = Pt[:, :k]
        xr = F.normalize(xc @ Qk @ Qk.t() + mu_t, p=2, dim=1)
        return xr.t().reshape(1, c, h, w)

    def valley(rgb4, sar4, gr, gc):
        with torch.no_grad():
            M = F.conv2d(rgb4, sar4)
        return both_hits(M, gr, gc)

    # ---- evaluate the in-domain PCA basis on each set ----
    print('==== OOD test: in-domain PCA basis, top-k recovery on each set ====', flush=True)
    print('(if OOD needs k>16 while in-domain saturates ~16 -> the "redundant" dims carry OOD signal)\n', flush=True)
    summary = {}
    for name, csv in EVAL_SETS:
        df = pd.read_csv(csv)
        ne = min(N_EVAL, len(df))
        accs = {k: [] for k in RANKS}
        full = []
        chance = None
        for i in range(ne):
            rgb4c, sar4c, (g4r, g4c) = matching_feats(model, mh, df.iloc[i])
            rgb4 = rgb4c.unsqueeze(0).to(DEV); sar4 = sar4c.unsqueeze(0).to(DEV)
            if chance is None:
                with torch.no_grad():
                    h0, w0 = F.conv2d(rgb4, sar4).shape[-2:]
                chance = 9.0 / (h0 * w0)
            full.append(valley(rgb4, sar4, g4r, g4c))
            for k in RANKS:
                accs[k].append(valley(restrict(rgb4, k), restrict(sar4, k), g4r, g4c))
        arr = np.array(full); fmn, fmx = arr.mean(0)
        col = 0 if fmn >= fmx else 1
        full_v = float(arr.mean(0)[col])
        curve = [float(np.array(accs[k]).mean(0)[col]) for k in RANKS]
        denom = max(full_v - chance, 1e-9)
        rec16 = (curve[RANKS.index(16)] - chance) / denom
        rec64 = (curve[RANKS.index(64)] - chance) / denom
        summary[name] = (full_v, curve, rec16, rec64, col)
        sign = 'MIN' if col == 0 else 'MAX'
        print(f'--- {name} (n={ne}, sign={sign}, FULL(256)={full_v:.3f}) ---', flush=True)
        for k, v in zip(RANKS, curve):
            rv = (v - chance) / denom
            print(f'    k={k:3d}  hit={v:.3f}  rec={rv:+.2f}', flush=True)
        print(f'    >>> recovery@k16={rec16:+.2f}  recovery@k64={rec64:+.2f}  '
              f'(k16<<1 means OOD needs dims beyond in-domain-16)\n', flush=True)

    print('==== VERDICT ====', flush=True)
    for name in summary:
        full_v, curve, rec16, rec64, _ = summary[name]
        print(f'  {name:12s}: full={full_v:.3f}  rec@16={rec16:+.2f}  rec@64={rec64:+.2f}', flush=True)
    print('\nMATCHDIM_OOD_DONE', flush=True)


if __name__ == '__main__':
    main()
