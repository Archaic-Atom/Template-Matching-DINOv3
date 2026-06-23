# -*- coding: utf-8 -*-
"""Control battery: is B's stride-4 localization GENUINE cross-modal similarity
or a content-blind shortcut?  对照实验：B 在 stride-4 的定位是真跨模态相似度，
还是与 SAR 内容无关的捷径？

Decisive test = C4 foreign-template swap (optical_i × SAR_j, j!=i). If localization
survives with the WRONG SAR template, the score is template-content-blind.

Metrics (per condition, over N pairs):
  valley@GT  = frac. where argmin of the raw cosine cross-corr map is <=1 cell from GT
               (B's match is the MINIMUM; pre-refine).
  prominence = (mean - min)/std of the 17x17 map — how much the extremum stands out
               (confidence). High only when the template truly "locks on".
  chance valley@GT ~ 9/289 ~ 3% (<=1-cell tolerance on a 17x17 grid).

Conditions:
  OWN            own paired SAR template (baseline).
  C4-foreign-pos foreign SAR_j with GT offset matched to i (<=1 cell) — isolates CONTENT
                 from position. Shortcut: stays high. Genuine: -> chance.
  C4-foreign-rnd foreign SAR_j drawn at random.
  C2-const       constant (dataset-mean) template — zero spatial content.
  C3-shuffle     own template with its 48x48 cells spatially permuted (multiset of unit
                 vectors preserved, correspondence destroyed).
  C1-center      per-pair mean (rank-1 shared-bias) removed from both sides, re-normed.
  C7-Q2 / C7-Q1  two-sided / one-sided random orthogonal rotation of the 256-ch axis.
                 Genuine inner product: Q2 invariant, Q1 destroys. Artifact: Q1 ~invariant.
  C0             positional-prior ceiling (pure GT-offset statistics, no model).
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
K_FOREIGN = 8
# Optional argv[1] = eval CSV (SAR_Image,RGB_Image,Label) for cross-dataset reruns
# (e.g. WHU); default = in-domain SEN1-2 2k. 可选 argv[1] 指定 CSV 用于跨数据集。
CSV = sys.argv[1] if len(sys.argv) > 1 else 'Datasets/test_256_2k.csv'
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


def renorm(x):
    """L2-normalize per cell over channel dim."""
    return F.normalize(x, p=2, dim=1)


def cmap(rgb_4, sar_4):
    """Raw cosine cross-correlation map (1,1,Ho,Wo); sar_4 is the conv kernel."""
    return F.conv2d(rgb_4, sar_4)


def valley_hit(M, gr, gc):
    h, w = M.shape[-2:]
    idx = int(M.view(-1).argmin().item())
    return int(max(abs(idx // w - gr), abs(idx % w - gc)) <= 1)


def prominence(M):
    m = M.view(-1)
    return float(((m.mean() - m.min()) / (m.std() + 1e-8)).item())


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    model = load_model()
    mh = model.matching
    df = pd.read_csv(CSV)
    n = min(N, len(df))

    # ---- pre-pass: cache rgb_4, sar_4 (fp32 CPU), GT stride-4 offsets ----
    rgb_list, sar_list, gts = [], [], []
    for i in range(n):
        r = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(r['Label']))
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)
                               ).unsqueeze(0).to(DEV)
        with torch.no_grad():
            rgb_feat = model.rgb_encoder(rgb)
            sar_feat = model.sar_encoder(sar)
            rgb_4 = mh.proj_rgb_4(rgb_feat[-1], rgb)     # (1,256,64,64) L2-normed
            sar_4 = mh.proj_sar_4(sar_feat['sar16'], sar)  # (1,256,48,48) L2-normed
        rgb_list.append(rgb_4.float().cpu())
        sar_list.append(sar_4.float().cpu())
        ho = rgb_4.shape[-2] - sar_4.shape[-2]
        wo = rgb_4.shape[-1] - sar_4.shape[-1]
        gts.append((min(max(int(round(gy / 4)), 0), ho),
                    min(max(int(round(gx / 4)), 0), wo)))
        if (i + 1) % 100 == 0:
            print(f'  cached {i + 1}/{n}', flush=True)

    grid_h = rgb_list[0].shape[-2] - sar_list[0].shape[-2] + 1   # 17
    grid_w = rgb_list[0].shape[-1] - sar_list[0].shape[-1] + 1
    gts = np.array(gts)

    # ---- C0: positional-prior ceiling (pure label stats) ----
    best = 0.0
    for fr in range(grid_h):
        for fc in range(grid_w):
            hit = np.mean(np.maximum(np.abs(gts[:, 0] - fr),
                                     np.abs(gts[:, 1] - fc)) <= 1)
            best = max(best, hit)
    # GT offset entropy + central-5x5 fraction
    flat = gts[:, 0] * grid_w + gts[:, 1]
    _, cnt = np.unique(flat, return_counts=True)
    p = cnt / cnt.sum()
    ent = float(-(p * np.log2(p)).sum())
    cr, cc = grid_h // 2, grid_w // 2
    central = float(np.mean(np.maximum(np.abs(gts[:, 0] - cr),
                                       np.abs(gts[:, 1] - cc)) <= 2))

    # global constant template (C2): dataset-mean cell direction, broadcast
    all_cells = torch.cat([s.view(1, 256, -1) for s in sar_list], dim=2)  # (1,256,*)
    mean_cell = all_cells.mean(dim=2, keepdim=True)                        # (1,256,1)
    th, tw = sar_list[0].shape[-2:]
    const_sar = renorm(mean_cell.expand(1, 256, th * tw).reshape(1, 256, th, tw)).to(DEV)

    # one fixed random orthogonal Q on the 256-ch axis (C7)
    Q, _ = torch.linalg.qr(torch.randn(256, 256, generator=torch.Generator().manual_seed(SEED)))
    Q = Q.to(DEV)

    def rotate(x, q):  # x (1,256,H,W) -> apply q on channel axis, renorm
        b, c, h, w = x.shape
        y = torch.einsum('oc,bchw->bohw', q, x)
        return renorm(y)

    rec = {k: {'v': [], 'z': []} for k in
           ('own', 'C4_fpos', 'C4_frnd', 'C2_const', 'C3_shuf',
            'C1_center', 'C7_Q2', 'C7_Q1')}
    c1_meanmag = []

    # precompute offset->list of indices for matched-offset foreign sampling
    from collections import defaultdict
    by_off = defaultdict(list)
    for j, (r0, c0) in enumerate(gts):
        by_off[(int(r0), int(c0))].append(j)

    for i in range(n):
        rgb_4 = rgb_list[i].to(DEV)
        sar_4 = sar_list[i].to(DEV)
        gr, gc = int(gts[i, 0]), int(gts[i, 1])

        # OWN
        M = cmap(rgb_4, sar_4)
        rec['own']['v'].append(valley_hit(M, gr, gc)); rec['own']['z'].append(prominence(M))

        # C4 foreign, GT-offset matched (<=1 cell), up to K
        cand = [j for j in range(n)
                if j != i and max(abs(gts[j, 0] - gr), abs(gts[j, 1] - gc)) <= 1]
        rng.shuffle(cand)
        for j in cand[:K_FOREIGN]:
            Mf = cmap(rgb_4, sar_list[j].to(DEV))
            rec['C4_fpos']['v'].append(valley_hit(Mf, gr, gc))
            rec['C4_fpos']['z'].append(prominence(Mf))

        # C4 foreign, random
        for _ in range(K_FOREIGN):
            j = rng.randint(0, n)
            while j == i:
                j = rng.randint(0, n)
            Mr = cmap(rgb_4, sar_list[j].to(DEV))
            rec['C4_frnd']['v'].append(valley_hit(Mr, gr, gc))
            rec['C4_frnd']['z'].append(prominence(Mr))

        # C2 constant template
        Mc = cmap(rgb_4, const_sar)
        rec['C2_const']['v'].append(valley_hit(Mc, gr, gc)); rec['C2_const']['z'].append(prominence(Mc))

        # C3 cell-shuffle own template (preserve unit-vector multiset)
        perm = torch.randperm(th * tw, generator=torch.Generator().manual_seed(SEED + i))
        sar_sh = sar_4.reshape(1, 256, th * tw)[:, :, perm].reshape(1, 256, th, tw)
        Ms = cmap(rgb_4, sar_sh)
        rec['C3_shuf']['v'].append(valley_hit(Ms, gr, gc)); rec['C3_shuf']['z'].append(prominence(Ms))

        # C1 per-pair mean removal (shared rank-1 bias), renorm
        rgb_c = renorm(rgb_4 - rgb_4.mean(dim=(2, 3), keepdim=True))
        sar_c = renorm(sar_4 - sar_4.mean(dim=(2, 3), keepdim=True))
        Mm = cmap(rgb_c, sar_c)
        rec['C1_center']['v'].append(valley_hit(Mm, gr, gc)); rec['C1_center']['z'].append(prominence(Mm))
        c1_meanmag.append(float(sar_4.mean(dim=(2, 3)).abs().mean().item()))

        # C7 two-sided / one-sided orthogonal rotation
        M2 = cmap(rotate(rgb_4, Q), rotate(sar_4, Q))
        rec['C7_Q2']['v'].append(valley_hit(M2, gr, gc)); rec['C7_Q2']['z'].append(prominence(M2))
        M1 = cmap(rgb_4, rotate(sar_4, Q))
        rec['C7_Q1']['v'].append(valley_hit(M1, gr, gc)); rec['C7_Q1']['z'].append(prominence(M1))

        if (i + 1) % 100 == 0:
            print(f'  controls {i + 1}/{n}', flush=True)

    chance = 9.0 / (grid_h * grid_w)
    print('\n==== B ep4 — does stride-4 localization need SAR content? ====', flush=True)
    print(f'  N={n}  grid={grid_h}x{grid_w}  valley@GT chance≈{chance:.3f}\n', flush=True)
    print(f'[C0 positional ceiling] best fixed-offset valley@GT={best:.3f}  '
          f'GT-offset entropy={ent:.2f} bits  frac within central 5x5={central:.3f}', flush=True)
    print(f'    (if ceiling≈own → task degenerate / positional; if << own → needs per-pair info)\n',
          flush=True)
    order = [('own', 'OWN paired template (baseline)'),
             ('C4_fpos', 'C4 FOREIGN SAR, GT-offset matched  ← decisive content test'),
             ('C4_frnd', 'C4 FOREIGN SAR, random'),
             ('C2_const', 'C2 constant (mean) template'),
             ('C3_shuf', 'C3 own template, cells shuffled'),
             ('C1_center', 'C1 per-pair mean (shared-bias) removed'),
             ('C7_Q2', 'C7 two-sided orthogonal Q (sanity = own)'),
             ('C7_Q1', 'C7 one-sided orthogonal Q')]
    for k, label in order:
        v = float(np.mean(rec[k]['v'])); z = float(np.mean(rec[k]['z']))
        print(f'  {label:52s}  valley@GT={v:.3f}  prominence_z={z:.2f}', flush=True)
    print(f'\n  C1 per-cell shared-mean magnitude (pre-removal) ≈ {np.mean(c1_meanmag):.4f}', flush=True)
    print('\nCONTROLS_DONE', flush=True)


if __name__ == '__main__':
    main()
