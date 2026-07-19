# -*- coding: utf-8 -*-
"""Probe what cross-modal similarity B (joint-trained SAR encoder) learned.

探测 B（SAR 编码器联合训练、活跃）到底学到了什么样的跨模态相似度。

B's matching head (StandardMatchingHeadJBU) consumes two stride-16 1024-d
feature maps — frozen DINOv3 ``rgb_16`` and trainable ``sar16`` — then JBU-
upsamples + projects BOTH to a 256-d stride-4 subspace where a cosine cross-
correlation actually runs. This script measures the matched-vs-background
cosine separation at TWO levels to locate WHERE the learned similarity lives:

    level-1  raw stride-16 : cos(rgb_16, sar16)                 (the stage-1 target)
    level-2  projected str4: cos(proj_rgb_4, proj_sar_4)        (the matching subspace)

对每个测试对，在 GT 对齐位置 vs 背景位置分别量逐格余弦，报告 d' 分离度与
互相关峰值命中率。raw16 分不开 / proj4 分得开 ⇒ B 的相似度学在投影子空间，
不在原始 stride-16（解释 stage-1 对齐为何无效）。
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
    """Force is_stage_2=True so the matching head (JBU proj layers) is built."""
    is_stage_2 = True
    pre_trained = True
    dist = False
    root_path = '.'
    modelDir = '.'
    gpu = 1


DEV = torch.device('cuda')
N = 600                       # 测试样本数（DINO 前向是瓶颈，600≈2min）
K_BG = 4                      # 每样本背景偏移采样数
# Optional argv[1] = eval CSV (SAR_Image,RGB_Image,Label) so the same probe can
# run on a cross-dataset set (e.g. WHU) — default = in-domain SEN1-2 2k.
# 可选 argv[1] 指定 CSV(同格式),用于跨数据集(如 WHU)复跑;默认 in-domain。
CSV = sys.argv[1] if len(sys.argv) > 1 else 'Datasets/test_256_2k.csv'
CONFIG = 'Source/UserModelImplementation/Models/TM_Model_B/config.yaml'
CKPT = 'Checkpoint_B_256/model_epoch_4.pth'
SEED = 2024


def load_model(config_path, ckpt_path):
    """Build B and load the ep4 trainable_tail (strict=False)."""
    model = build_model_from_yaml(config_path, _Stage2Args())
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_0', ckpt)
    missing, unexpected = model.trainable_tail.load_state_dict(state, strict=False)
    print(f'  loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}',
          flush=True)
    return model.to(DEV).eval()


def cell_cosine_at(search_norm, templ_norm, off_r, off_c):
    """Mean per-cell cosine of a template grid placed at (off_r, off_c) in search.

    search_norm: (1, C, Hs, Ws) L2-normed over C.
    templ_norm:  (1, C, Ht, Wt) L2-normed over C.
    Returns scalar mean cosine over the Ht*Wt aligned cells.
    """
    _, _, ht, wt = templ_norm.shape
    win = search_norm[:, :, off_r:off_r + ht, off_c:off_c + wt]
    cos = (win * templ_norm).sum(dim=1)          # (1, Ht, Wt) per-cell cosine
    return float(cos.mean().item())


def xcorr_peak(search_norm, templ_norm):
    """Cosine cross-correlation argmax → predicted (off_r, off_c).

    Slides the L2-normed template as a conv kernel over the L2-normed search;
    each position = sum of per-cell cosine over the window. Returns the argmax
    offset (row, col) in the (Hs-Ht+1, Ws-Wt+1) score map.
    """
    score = F.conv2d(search_norm, templ_norm)    # (1, 1, Ho, Wo)
    ho, wo = score.shape[-2:]
    idx = int(score.view(-1).argmax().item())
    return idx // wo, idx % wo


def separation(matched, background):
    """d' = (mean_m - mean_b) / pooled_std, plus the two means."""
    m = np.array(matched); b = np.array(background)
    pooled = np.sqrt(0.5 * (m.std() ** 2 + b.std() ** 2)) + 1e-8
    return (m.mean() - b.mean()) / pooled, m.mean(), b.mean()


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    model = load_model(CONFIG, CKPT)
    mh = model.matching
    df = pd.read_csv(CSV)

    rec = {k: {'m': [], 'b': [], 'hit': []}
           for k in ('raw16', 'proj4')}
    # proj4-only: is the match the MINIMUM of the raw xcorr, and does the
    # learned refine (head_4, with its 3x3 conv) flip it into a peak at GT?
    # 验证"匹配=投影余弦最小值、refine 翻符号成峰值"的解释。
    rec['proj4']['valley'] = []   # argmin(raw xcorr) within <=1 cell of GT
    rec['proj4']['head'] = []     # argmax(head_4 WITH refine) within <=1 cell of GT

    n = min(N, len(df))
    for i in range(n):
        r = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(r['Label']))   # GT top-left pixel
        sar = preprocess_image(read_image(r['SAR_Image'], ImageReadMode.GRAY),
                               duplicate_channels=True).unsqueeze(0).to(DEV)
        rgb = preprocess_image(read_image(r['RGB_Image'], ImageReadMode.RGB)
                               ).unsqueeze(0).to(DEV)

        with torch.no_grad():
            rgb_feat = model.rgb_encoder(rgb)
            sar_feat = model.sar_encoder(sar)
            rgb_16 = F.normalize(rgb_feat[-1], p=2, dim=1)   # DINO not normed → norm now
            sar_16 = sar_feat['sar16']                       # already L2-normed
            # projected stride-4 subspace (the real matching space)
            rgb_4 = mh.proj_rgb_4(rgb_feat[-1], rgb)         # JBU expects raw (unnormed) feat
            sar_4 = mh.proj_sar_4(sar_feat['sar16'], sar)

        for tag, search, templ, stride in (
                ('raw16', rgb_16, sar_16, 16),
                ('proj4', rgb_4, sar_4, 4)):
            Hs, Ws = search.shape[-2:]
            ht, wt = templ.shape[-2:]
            max_r, max_c = Hs - ht, Ws - wt
            if max_r < 0 or max_c < 0:
                continue
            gr = int(round(gy / stride)); gc = int(round(gx / stride))
            gr = min(max(gr, 0), max_r); gc = min(max(gc, 0), max_c)
            # matched cosine at GT offset
            rec[tag]['m'].append(cell_cosine_at(search, templ, gr, gc))
            # background cosine at K random wrong offsets (>=2 cells from GT)
            tries = 0
            while tries < K_BG:
                br = rng.randint(0, max_r + 1); bc = rng.randint(0, max_c + 1)
                if max(abs(br - gr), abs(bc - gc)) >= 2:
                    rec[tag]['b'].append(cell_cosine_at(search, templ, br, bc))
                    tries += 1
                elif max_r + max_c <= 2:        # tiny grid: can't avoid GT, accept
                    rec[tag]['b'].append(cell_cosine_at(search, templ, br, bc))
                    tries += 1
            # cross-correlation peak-at-GT (<=1 cell)
            pr, pc = xcorr_peak(search, templ)
            rec[tag]['hit'].append(int(max(abs(pr - gr), abs(pc - gc)) <= 1))

        # ---- proj4 sign-convention check ----
        # raw xcorr argMIN @GT (is the match the minimum?) + head_4 (refine) argmax @GT.
        # raw 互相关的 argmin 是否在 GT？带 refine 的 head_4 argmax 是否在 GT？
        with torch.no_grad():
            raw_xc = F.conv2d(rgb_4, sar_4)              # (1,1,17,17) pre-refine
            H4_ref = mh.head_4(template=sar_4, search=rgb_4)  # WITH 3x3 refine
        ho, wo = raw_xc.shape[-2:]
        g4r = min(max(int(round(gy / 4)), 0), ho - 1)
        g4c = min(max(int(round(gx / 4)), 0), wo - 1)
        vi = int(raw_xc.view(-1).argmin().item())
        rec['proj4']['valley'].append(
            int(max(abs(vi // wo - g4r), abs(vi % wo - g4c)) <= 1))
        hi = int(H4_ref.view(-1).argmax().item())
        rec['proj4']['head'].append(
            int(max(abs(hi // wo - g4r), abs(hi % wo - g4c)) <= 1))

        if (i + 1) % 100 == 0:
            print(f'  ...{i + 1}/{n}', flush=True)

    print(f'\n==== B ep4 — learned cross-modal similarity, by feature level ====', flush=True)
    print(f'  CSV={CSV}', flush=True)
    print(f'  N={n}  (matched = GT-aligned window; background = {K_BG} wrong offsets/sample)\n',
          flush=True)
    for tag, label in (('raw16', 'RAW stride-16  cos(DINO rgb_16, SAR sar16)'),
                       ('proj4', 'PROJ stride-4  cos(proj_rgb_4, proj_sar_4)')):
        d, mm, mb = separation(rec[tag]['m'], rec[tag]['b'])
        hit = float(np.mean(rec[tag]['hit']))
        print(f'[{tag}] {label}', flush=True)
        print(f'    matched_cos={mm:+.4f}  background_cos={mb:+.4f}  '
              f"d'={d:.3f}  xcorr_peak@GT(<=1cell)={hit:.3f}", flush=True)
    valley = float(np.mean(rec['proj4']['valley']))
    head = float(np.mean(rec['proj4']['head']))
    print(f'\n[proj4 sign-check]  xcorr_VALLEY@GT(<=1cell)={valley:.3f}  '
          f'head_4(+refine)_peak@GT(<=1cell)={head:.3f}', flush=True)
    print('  → if VALLEY@GT high & head@GT high: match = MIN of projected cosine, '
          'refine flips sign. 匹配=投影余弦最小值，refine 翻符号。', flush=True)
    print('\nPROBE_DONE', flush=True)


if __name__ == '__main__':
    main()
