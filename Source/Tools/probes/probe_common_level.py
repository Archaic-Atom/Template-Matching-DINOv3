# -*- coding: utf-8 -*-
"""Same-level capacity measurement: supply and demand on ONE ruler, with
the estimator hygiene the reviewer asked for.

同一层级的容量测量:把"供给"(Stage-1 对齐码)与"需求"(已训练匹配器)放到
**同一个层级、同一组抽样、同一把尺子**上,并补齐估计器卫生(拟合/评估划分、
置换零分布显著性阈值、跨抽样稳定性、去掉测量天花板)。

WHAT THIS ANSWERS / 本脚本回答什么
----------------------------------
Reviewer 2, comment 4. The paper juxtaposes a "~22-dimensional supply"
(Sec. IV-C, ``sec:capacity``) with a "~64-dimensional demand" (Sec. IV-E,
``sec:distributed``, Fig. 8 ``fig:recovery_rank``) and reads a capacity
mismatch off the pair. The reviewer objects that the two numbers come from
different network levels, resolutions and statistical definitions, and asks
for (a) the comparison at ONE level with identical feature dimensions and
rank definitions, and (b) disclosure of the CCA fit/eval split, the
significance criterion, and the stability of the estimates.
审稿人 2 第 4 条:22 与 64 不同层级、不同分辨率、不同定义,不可直接相减。

THREE FACTS THIS SCRIPT IS BUILT ON (each verified in the code) / 三个既成事实
------------------------------------------------------------------------------
1. **The paper misdescribes its own probe.** Sec. IV-E and the Fig. 8 caption
   say the rank-$k$ restriction is applied to the *stride-4 matching
   features*. In ``probe_b_keystone.py`` (``project_feat`` / ``comp_feat``,
   L180-193) and ``probe_b_granularity_dim.py`` (L212-224) the restriction is
   BUILT AND APPLIED in the RAW 1024-d stride-16 feature space, and the
   restricted features are then read out through the real trained
   ``proj_*_4`` / ``proj_*_8`` stacks. Supply and demand are therefore
   ALREADY on one basis — the common level is the 1024-d stride-16 token
   space, not stride-4. This script makes that explicit rather than
   re-engineering the measurement; the paper's wording is what needs fixing.
   论文把自己的探针说错了:秩限制发生在 1024-d stride-16 原始特征空间,
   再经真实 proj_*_4/8 读出 —— 供给与需求本就同基,该改的是论文措辞。
2. **The demand estimate was capped by the estimator.** Both probes set
   ``D_PCA = 64`` with ``RANKS`` topping out at 64, so "~64" is the
   measurement ceiling, not an observed saturation. Here ``--d-pca`` defaults
   to 128 and ``--ranks`` runs past 64, so saturation can actually be
   observed — or its absence quoted honestly as a lower bound.
3. **The significance rule lived only in the code**:
   ``eff = sum(rho_holdout > |rho_null| + 0.05)`` — a fixed 0.05 margin
   against a single cyclic re-pairing, no test statistic, no
   multiple-comparison control, and no stability across draws anywhere. Here
   the default is a permutation null (SAR tokens shuffled across pairs at
   matched window positions, ``--perm`` shuffles, a stated percentile as the
   threshold); the old rule stays available via ``--sig-criterion margin``.

THE MEASUREMENT / 测量设计
--------------------------
For every arm, one estimator, applied identically:

  extract stride-16 features -> draw a subset -> split fit/eval BY IMAGE ->
  de-leaked CCA on GT-corresponding token pairs (fit half) -> held-out
  canonical correlations + permutation null -> restrict BOTH modalities'
  raw 1024-d stride-16 tokens to the top-$k$ canonical directions ->
  read the restricted tokens out through that arm's OWN trained operator ->
  localization hit@GT -> chance-normalized recovery -> saturation rank.

  supply  ``TM_Model_F`` Stage-1 (``Checkpoint_F_stage1_256``), read out as
          stride-16 cosine alignment (``align_raw``) and, when the variant
          has learned-IP projectors, through ``align_ip`` — the space the
          Stage-1 objective actually lives in.
  demand  the trained matcher (``TM_Model_B`` ep4 by default), read out
          through the real ``proj_*_4`` / ``proj_*_8`` (``match_4`` /
          ``match_8``), plus the stride-16 raw-cosine CONTROL (``raw16``).
  cross   (opt-in, ``--arms ...,cross``) the SUPPLY's restricted stride-16
          code pushed through the DEMAND model's trained ``proj_*_4/8`` —
          "is the aligned code sufficient input for the trained head".

THE REVERSE DIRECTION IS A PRINTED CONTROL, NOT A FOOTNOTE / 反向必须报
----------------------------------------------------------------------
``probe_recovery_by_stride.py`` found that the trained matcher does not
localize at stride-16 at all (full-feature hit 0.325 against a 0.36 chance
floor), so a *demand* rank is not defined there. That is exactly why the
common level has to be the stride-16 BASIS with each side read out through
its own operator, rather than "measure both at stride-4" or "measure both at
stride-16". The ``raw16`` control re-measures it with error bars so the
response letter can quote it.

OUTPUTS AND THE CLAIMS THEY FEED / 输出与其支撑的论文主张
---------------------------------------------------------
``probe_common_level_curves.csv``
    one tidy row per (arm, readout, rank): hit and chance-normalized
    recovery, mean +/- SD over draws. Feeds **Fig. 8**
    (``fig:recovery_rank``, ``img/analysis/plot_recovery_rank.py``) — the
    error bands and the same-level supply curve replace the current
    hard-coded single-draw arrays and the bare stride-16 marker.
``probe_common_level_summary.csv``
    one row per (arm, readout, metric): full-feature hit, chance floor,
    saturation ranks, CCA significant-direction counts under both criteria,
    the permutation threshold, and the fit/eval sample counts. Feeds
    **Sec. IV-C** (``sec:capacity``, the ~22 supply claim), **Sec. IV-E**
    (``sec:distributed``, the ~64 demand claim and the "measurement ceiling"
    sentence) and the new estimator paragraph promised for R2-4(b).
stdout ``RESPONSE-LETTER BLOCK``
    paste-ready numbers for R2-4(a) and R2-4(b).

REPORTING RULE / 报告纪律
-------------------------
A result that FAILS to preserve the supply < demand ordering must be
reported as-is. The response letter already commits to reporting whatever
the same-level measurement returns; the honest outcomes are (i) the ordering
survives at one level, (ii) the demand curve saturates below the new 128-d
ceiling so a number replaces a lower bound, or (iii) it does not and
"distributed, no compact matching subspace" becomes the stronger claim.
若同层级测量不再保留"供给 < 需求"的次序,如实报告 —— 回复信已作此承诺。

REPRODUCING THE PUBLISHED NUMBERS / 复现已发表数字
--------------------------------------------------
The published estimates come from the old estimator settings; they are the
documented defaults of three flags::

    python -u Source/Tools/probes/probe_common_level.py \\
        --n 400 --pool 400 --d-pca 64 --draws 1 --sig-criterion margin \\
        --ranks 4,8,12,16,20,22,24,28,32,40,48,56,64

USAGE / 用法 (run FROM the project root, as every probe here does)::

    cd <project-root>
    CUDA_VISIBLE_DEVICES=7 TM_SEED=2024 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
      nohup python -u Source/Tools/probes/probe_common_level.py \\
      > probe_common_level.log 2>&1 & disown

Cost / 开销: forward passes only, single GPU, no training. The encoder pass
is cached to an ``.npz`` keyed by (checkpoint, pool size, seed), so a second
pass with different ``--ranks`` or ``--draws`` re-runs no ViT-L forward;
per-draw results are cached as JSON too, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image

sys.path.insert(0, 'Source')
from Tools.eval_stage1_matching import preprocess_image  # noqa: E402
from UserModelImplementation.Models.builder import build_model_from_yaml  # noqa: E402

# --------------------------------------------------------------------------
# Defaults. The PUBLISHED_* values are what the paper's numbers were measured
# with; they are kept as documented CLI presets, not as defaults.
# PUBLISHED_* 是论文数字的测量设置,保留为可传参的预设而非默认值。
# --------------------------------------------------------------------------
PUBLISHED_N = 400
PUBLISHED_D_PCA = 64
PUBLISHED_RANKS = (4, 8, 12, 16, 20, 22, 24, 28, 32, 40, 48, 56, 64)

DEFAULT_N = 1500
DEFAULT_POOL = 2000
DEFAULT_D_PCA = 128
DEFAULT_RANKS = (4, 8, 12, 16, 20, 22, 24, 28, 32, 40, 48, 56, 64,
                 80, 96, 112, 128)
DEFAULT_DRAWS = 10
DEFAULT_PERM = 200
DEFAULT_PERM_PCT = 95.0
DEFAULT_MARGIN = 0.05          # the legacy fixed margin (fact 3)
DEFAULT_SEED = 2024
DEFAULT_EVAL_CAP = 400         # scored held-out images per draw; 0 = all
K_KEY = 22                     # the paper's headline rank, always reported

SUPPLY_CONFIG = 'Source/UserModelImplementation/Models/TM_Model_F/config.yaml'
SUPPLY_CKPT = 'Checkpoint_F_stage1_256/model_epoch_9.pth'
DEMAND_CONFIG = 'Source/UserModelImplementation/Models/TM_Model_B/config.yaml'
DEMAND_CKPT = 'Checkpoint_B_256/model_epoch_4.pth'
DEFAULT_IMAGE_CSV = 'Datasets/test_256_2k.csv'

# A readout whose full-feature hit sits within this of its chance floor does
# not localize, so no recovery rank is defined for it (the reverse-direction
# control). 全特征命中若与 chance 只差这么点,即"不定位",恢复秩无定义。
UNDEFINED_MARGIN = 0.02


class _ProbeArgs:
    """Minimal stand-in for JackFramework's argparse namespace.

    仅为 ``build_model_from_yaml`` 提供它读取的几个字段。

    Args:
        is_stage_2: Select the Stage-2 (matching) or Stage-1 (alignment)
            forward path and component set.
            选择 Stage-2(匹配)或 Stage-1(对齐)的前向路径与组件集合。
        root_path: Substituted into the config's ``${root_path}``.
            用于配置里 ``${root_path}`` 的替换。
    """

    def __init__(self, is_stage_2: bool, root_path: str = '.') -> None:
        self.is_stage_2 = bool(is_stage_2)
        self.pre_trained = True
        self.dist = False
        self.root_path = root_path
        self.modelDir = '.'
        self.gpu = 1


@dataclass
class EncoderSpec:
    """One (config, checkpoint, stage) triple = one feature source.

    一个特征来源:配置 + 权重 + stage。

    Attributes:
        key: Short name used for cache filenames and CSV rows.
        config: Path to the variant's ``config.yaml``.
        ckpt: Path to the JackFramework ``.pth``.
        is_stage_2: Stage-2 (matcher) or Stage-1 (aligned encoder).
    """

    key: str
    config: str
    ckpt: str
    is_stage_2: bool


@dataclass
class FeatureSet:
    """Cached stride-16 features for a pool of image pairs.

    一批图像对的 stride-16 特征缓存。

    Attributes:
        rgb16: ``(pool, 1024, 16, 16)`` optical tokens.
        sar16: ``(pool, 1024, 12, 12)`` SAR tokens.
        gy: ``(pool,)`` ground-truth pixel row offsets.
        gx: ``(pool,)`` ground-truth pixel column offsets.
        sar_paths: SAR image paths (re-read for JBU guidance).
        rgb_paths: optical image paths (re-read for JBU guidance).
    """

    rgb16: np.ndarray
    sar16: np.ndarray
    gy: np.ndarray
    gx: np.ndarray
    sar_paths: np.ndarray
    rgb_paths: np.ndarray


@dataclass
class Readout:
    """One task read-out of a (possibly restricted) stride-16 token pair.

    对(可能被秩限制的)stride-16 token 对的一种任务读出。

    Attributes:
        name: Row label in the CSVs.
        role: ``supply`` / ``demand`` / ``control`` — how the paper uses it.
        stride: Pixel stride of the score map this read-out produces.
        fn: ``(f_rgb, f_sar, rgb_img, sar_img) -> score map (1, 1, h, w)``.
        note: Free-text note carried into the summary CSV.
    """

    name: str
    role: str
    stride: int
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                 torch.Tensor]
    note: str = ''


@dataclass
class Arm:
    """A named group of read-outs sharing one feature source and one CCA fit.

    共用一个特征来源与一次 CCA 拟合的一组读出。
    """

    name: str
    fs_key: str
    readouts: List[Readout] = field(default_factory=list)


# ==========================================================================
# CCA / recovery machinery — behaviour-identical to probe_b_granularity_dim.py
# so the two estimators remain comparable, only vectorized and parameterized.
# CCA 与恢复率机制:与 probe_b_granularity_dim.py 行为一致(仅向量化+可配置),
# 保证新旧估计器可比。
# ==========================================================================
def inv_sqrt(cov: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Inverse matrix square root with an eigenvalue floor.

    带特征值下限的矩阵逆平方根(CCA 白化用)。

    Args:
        cov: ``(d, d)`` symmetric covariance.
        eps: Eigenvalue floor; the CCA regularization knob.
            特征值下限,即 CCA 的正则化旋钮。

    Returns:
        ``(d, d)`` matrix ``C^{-1/2}``.
    """
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, eps, None)
    return v @ np.diag(1.0 / np.sqrt(w)) @ v.T


def pca_basis(x_centered: np.ndarray, d: int) -> np.ndarray:
    """Top-``d`` PCA basis of centered data.

    中心化数据的前 d 个主成分基。
    """
    cov = x_centered.T @ x_centered / x_centered.shape[0]
    w, v = np.linalg.eigh(cov)
    return v[:, np.argsort(w)[::-1][:d]]


def fit_cca(x_tr: np.ndarray, y_tr: np.ndarray, d: int,
            eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
    """PCA-then-CCA: return feature-space canonical loadings and train means.

    先 PCA 降到 d 维再做 CCA,返回特征空间的典型载荷与训练均值。

    Args:
        x_tr: ``(n, 1024)`` optical tokens of the fit half.
        y_tr: ``(n, 1024)`` SAR tokens of the fit half.
        d: Retained PCA dimension (the measurement ceiling — fact 2).
        eps: Whitening eigenvalue floor.

    Returns:
        ``(w_x, w_y, m_x, m_y)`` with ``w_*`` of shape ``(1024, d)`` mapping
        centered features to canonical variates.
    """
    m_x, m_y = x_tr.mean(0), y_tr.mean(0)
    x_c, y_c = x_tr - m_x, y_tr - m_y
    p_x, p_y = pca_basis(x_c, d), pca_basis(y_c, d)
    x_r, y_r = x_c @ p_x, y_c @ p_y
    k_x = inv_sqrt(x_r.T @ x_r / len(x_r), eps)
    k_y = inv_sqrt(y_r.T @ y_r / len(y_r), eps)
    u, _, vt = np.linalg.svd(k_x @ (x_r.T @ y_r / len(x_r)) @ k_y)
    return p_x @ k_x @ u, p_y @ k_y @ vt.T, m_x, m_y


def standardize(a: np.ndarray) -> np.ndarray:
    """Column-wise zero-mean unit-variance, so a dot product is a correlation.

    逐列标准化,使内积即相关系数。
    """
    return (a - a.mean(0)) / (a.std(0) + 1e-12)


def orthobasis(w: np.ndarray, k: int) -> np.ndarray:
    """Orthonormal basis ``(1024, k)`` of the top-``k`` canonical directions.

    前 k 个典型方向张成子空间的正交基。
    """
    q, _ = np.linalg.qr(w[:, :k])
    return q[:, :k]


def derangement(n: int, rng: np.random.RandomState) -> np.ndarray:
    """A permutation of ``range(n)`` with no fixed point.

    无不动点的置换 —— 置换零分布必须真正拆散配对。
    """
    perm = rng.permutation(n)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def permutation_null(ax_std: np.ndarray, ay_std: np.ndarray, n_img: int,
                     n_tok: int, n_perm: int,
                     rng: np.random.RandomState) -> np.ndarray:
    """Null canonical correlations from cross-pair token shuffles.

    置换零分布:把 SAR token 在**相同窗口位置**上跨图像对打乱,重算每个
    典型方向的相关系数。位置对齐保证零分布只拆"配对身份",不拆"位置结构"。

    Args:
        ax_std: ``(n_img * n_tok, d)`` standardized optical canonical variates.
        ay_std: ``(n_img * n_tok, d)`` standardized SAR canonical variates.
        n_img: Number of held-out images contributing tokens.
        n_tok: Tokens per image (SAR grid size, matched window positions).
        n_perm: Number of shuffles.
        rng: Seeded generator.

    Returns:
        ``(n_perm, d)`` null canonical correlations.
    """
    dim = ax_std.shape[1]
    ay3 = ay_std.reshape(n_img, n_tok, dim)
    out = np.empty((n_perm, dim), dtype=np.float64)
    for s in range(n_perm):
        perm = derangement(n_img, rng)
        shuffled = ay3[perm].reshape(-1, dim)
        # Row permutation leaves column means/SDs untouched, so the already
        # standardized variates stay standardized — a plain mean of products
        # is the correlation. 行置换不改变列均值/方差,内积均值即相关系数。
        out[s] = (ax_std * shuffled).mean(0)
    return out


def both_hits(score_map: torch.Tensor, gr: int, gc: int) -> Tuple[int, int]:
    """Is the argmin / argmax within one cell of the GT cell?

    argmin / argmax 是否落在 GT 单元的 1 格邻域内(L-infinity <= 1)。

    Returns:
        ``(min_hit, max_hit)`` as 0/1 ints — the localizing extremum is
        chosen once per read-out, never per draw (see ``_pick_sign``).
    """
    h, w = score_map.shape[-2:]
    flat = score_map.reshape(-1)
    i_min = int(flat.argmin().item())
    i_max = int(flat.argmax().item())
    hit_min = int(max(abs(i_min // w - gr), abs(i_min % w - gc)) <= 1)
    hit_max = int(max(abs(i_max // w - gr), abs(i_max % w - gc)) <= 1)
    return hit_min, hit_max


def saturation_rank(curve: Sequence[float], full: float, floor: float,
                    frac: float, ranks: Sequence[int]) -> Optional[int]:
    """Smallest rank whose SUSTAINED chance-normalized recovery reaches ``frac``.

    达到 frac 恢复率且此后不再跌破的最小秩(要求"持续跨越",避免非单调抖动
    报出虚低的秩)。

    Args:
        curve: Hit rate per rank.
        full: Unrestricted hit rate of the same read-out.
        floor: Chance hit rate of the same score-map size.
        frac: Recovery threshold (0.90 in the paper).
        ranks: The swept ranks, ascending.

    Returns:
        The rank, or ``None`` when the curve never sustains ``frac``.
    """
    denom = max(full - floor, 1e-9)
    rec = [(v - floor) / denom for v in curve]
    for j, k in enumerate(ranks):
        if all(c >= frac for c in rec[j:]):
            return k
    return None


def restrict(fmap: torch.Tensor, q: torch.Tensor,
             mu: torch.Tensor) -> torch.Tensor:
    """Keep only the ``span(q)`` component of every token, mean added back.

    把每个 token 限制到 span(q) 子空间内(减均值->投影->加回均值)。

    NOTE (fact 1): this operates on the RAW 1024-d stride-16 map, exactly as
    ``probe_b_keystone.py`` does — the restriction is NOT applied to the
    stride-4 matching features, whatever the Fig. 8 caption says.
    注意:限制作用在原始 1024-d stride-16 特征上,与 keystone 一致。
    """
    _, c, h, w = fmap.shape
    x = fmap.reshape(c, h * w).t()
    x_r = (x - mu) @ q @ q.t() + mu
    return x_r.t().reshape(1, c, h, w)


# ==========================================================================
# Feature extraction + cache
# ==========================================================================
def cache_key(spec: EncoderSpec, csv_path: str, pool: int, seed: int) -> str:
    """Stable short hash of everything the cached features depend on.

    缓存键:权重 + 清单 + 池大小 + 种子(提取本身确定,种子仅入键以求严格)。
    """
    raw = f'{spec.config}|{spec.ckpt}|{csv_path}|{pool}|{seed}'
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]


def build_model(spec: EncoderSpec, device: torch.device) -> torch.nn.Module:
    """Build a variant from YAML and load its checkpoint into trainable_tail.

    按 YAML 构建变体并把权重加载进 trainable_tail(strict=False,与其他探针一致)。
    """
    model = build_model_from_yaml(spec.config, _ProbeArgs(spec.is_stage_2))
    ckpt = torch.load(spec.ckpt, map_location='cpu', weights_only=False)
    state = ckpt.get('model_0', ckpt)
    missing, unexpected = model.trainable_tail.load_state_dict(state,
                                                               strict=False)
    print(f'  [{spec.key}] loaded {spec.ckpt}: '
          f'missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    return model.to(device).eval()


def extract_features(model: torch.nn.Module, spec: EncoderSpec,
                     df: pd.DataFrame, pool: int, device: torch.device,
                     dtype: str) -> FeatureSet:
    """Run the encoders once over the pool and keep the stride-16 tokens.

    对整池图像跑一次编码器,保留 stride-16 token。这是全脚本唯一昂贵的一步,
    因此必须缓存 —— 换 RANKS 重跑时不应再碰 ViT-L。

    Args:
        model: Built model in eval mode.
        spec: Which encoder this is (stage decides the forward path).
        df: The eval manifest.
        pool: Number of image pairs to extract.
        device: CUDA device.
        dtype: ``float32`` (default, faithful) or ``float16`` (half the disk).

    Returns:
        The populated :class:`FeatureSet`.
    """
    np_dtype = np.float16 if dtype == 'float16' else np.float32
    rgb_list: List[np.ndarray] = []
    sar_list: List[np.ndarray] = []
    gy_list, gx_list, sar_paths, rgb_paths = [], [], [], []
    for i in range(pool):
        row = df.iloc[i]
        gx, gy = (int(v) for v in json.loads(row['Label']))
        sar_img = preprocess_image(
            read_image(row['SAR_Image'], ImageReadMode.GRAY),
            duplicate_channels=True).unsqueeze(0).to(device)
        rgb_img = preprocess_image(
            read_image(row['RGB_Image'], ImageReadMode.RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            if spec.is_stage_2:
                f_rgb = model.rgb_encoder(rgb_img)[-1]
                f_sar = model.sar_encoder(sar_img)['sar16']
            else:
                # Stage-1 forward returns the aligned code the paper's ~22
                # is measured on. Stage-1 前向给出论文 ~22 所测的对齐码。
                out = model(sar_img, rgb_img)
                f_rgb = out['rbg_feat'][3]          # key typo is upstream
                f_sar = out['sar_feat']['sar16']
        rgb_list.append(f_rgb[0].cpu().numpy().astype(np_dtype))
        sar_list.append(f_sar[0].cpu().numpy().astype(np_dtype))
        gy_list.append(gy)
        gx_list.append(gx)
        sar_paths.append(row['SAR_Image'])
        rgb_paths.append(row['RGB_Image'])
        if i == 0:
            print(f'  [{spec.key}] feat dims: rgb={tuple(f_rgb.shape)} '
                  f'sar={tuple(f_sar.shape)}', flush=True)
        if (i + 1) % 200 == 0:
            print(f'  [{spec.key}] extracted {i + 1}/{pool}', flush=True)
    return FeatureSet(rgb16=np.stack(rgb_list), sar16=np.stack(sar_list),
                      gy=np.asarray(gy_list), gx=np.asarray(gx_list),
                      sar_paths=np.asarray(sar_paths),
                      rgb_paths=np.asarray(rgb_paths))


def load_or_extract(spec: EncoderSpec, df: pd.DataFrame, pool: int,
                    device: torch.device, args: argparse.Namespace,
                    model: torch.nn.Module) -> FeatureSet:
    """Return the feature set from the ``.npz`` cache, extracting on a miss.

    命中缓存则直接读盘,未命中才跑编码器并落盘。
    """
    if args.no_cache:
        return extract_features(model, spec, df, pool, device,
                                args.cache_dtype)
    os.makedirs(args.cache_dir, exist_ok=True)
    key = cache_key(spec, args.image_csv, pool, args.seed)
    path = os.path.join(args.cache_dir, f'feats_{spec.key}_{key}.npz')
    if os.path.exists(path):
        print(f'  [{spec.key}] feature cache HIT: {path}', flush=True)
        blob = np.load(path, allow_pickle=False)
        return FeatureSet(rgb16=blob['rgb16'], sar16=blob['sar16'],
                          gy=blob['gy'], gx=blob['gx'],
                          sar_paths=blob['sar_paths'],
                          rgb_paths=blob['rgb_paths'])
    print(f'  [{spec.key}] feature cache MISS -> extracting to {path}',
          flush=True)
    fs = extract_features(model, spec, df, pool, device, args.cache_dtype)
    np.savez(path, rgb16=fs.rgb16, sar16=fs.sar16, gy=fs.gy, gx=fs.gx,
             sar_paths=fs.sar_paths, rgb_paths=fs.rgb_paths)
    return fs


# ==========================================================================
# Read-out construction
# ==========================================================================
def align_score(f_rgb: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
    """Stride-16 cosine alignment score map (per-token L2-norm + conv2d).

    stride-16 余弦对齐热图,与 ``eval_stage1_matching.score_match`` 一致。
    """
    rgb_n = F.normalize(f_rgb, p=2, dim=1)
    sar_n = F.normalize(f_sar, p=2, dim=1)
    _, _, s_h, s_w = sar_n.shape
    return F.conv2d(rgb_n, sar_n) / (s_h * s_w)


def make_supply_readouts(model: torch.nn.Module) -> List[Readout]:
    """Read-outs of the Stage-1 aligned code, at the common stride-16 level.

    供给侧读出:原始余弦对齐,以及(若变体有可学内积投影头)对齐目标真正所在
    的投影空间 —— 与需求侧"经各自训练好的算子读出"结构完全对称。
    """
    readouts = [Readout(name='align_raw', role='supply', stride=16,
                        fn=lambda fr, fs, ri, si: align_score(fr, fs),
                        note='stride-16 cosine on the raw 1024-d code')]
    align_ip = getattr(model, 'align_ip', None)
    if align_ip is not None:
        def _ip(f_rgb: torch.Tensor, f_sar: torch.Tensor,
                rgb_img: torch.Tensor, sar_img: torch.Tensor) -> torch.Tensor:
            s_pre, d_pre = align_ip(f_sar, f_rgb)
            return align_score(d_pre, s_pre)

        readouts.append(Readout(
            name='align_ip', role='supply', stride=16, fn=_ip,
            note='restricted 1024-d code read out through the trained '
                 'learned-IP projectors (the Stage-1 objective space)'))
    else:
        print('  [supply] no align_ip in this variant — '
              'align_ip read-out skipped', flush=True)
    return readouts


def make_match_readouts(model: torch.nn.Module, role: str,
                        suffix: str = '') -> List[Readout]:
    """Read-outs through the trained matching projections (the demand side).

    需求侧读出:限制后的 1024-d stride-16 特征经**真实训练好的** proj_*_4 /
    proj_*_8 读出 —— 这正是 keystone / granularity 探针的做法。
    """
    mh = model.matching

    def _s4(f_rgb: torch.Tensor, f_sar: torch.Tensor, rgb_img: torch.Tensor,
            sar_img: torch.Tensor) -> torch.Tensor:
        return F.conv2d(mh.proj_rgb_4(f_rgb, rgb_img),
                        mh.proj_sar_4(f_sar, sar_img))

    def _s8(f_rgb: torch.Tensor, f_sar: torch.Tensor, rgb_img: torch.Tensor,
            sar_img: torch.Tensor) -> torch.Tensor:
        return F.conv2d(mh.proj_rgb_8(f_rgb, rgb_img),
                        mh.proj_sar_8(f_sar, sar_img))

    return [
        Readout(name=f'match_4{suffix}', role=role, stride=4, fn=_s4,
                note='restricted stride-16 tokens through the trained '
                     'proj_*_4 (JBU x4) — the paper\'s ~64 demand curve'),
        Readout(name=f'match_8{suffix}', role=role, stride=8, fn=_s8,
                note='same estimator, proj_*_8 (JBU x2) — the ~48 curve'),
    ]


def make_control_readout() -> Readout:
    """The reverse direction: the trained matcher's own stride-16 similarity.

    反向对照:已训练匹配器自身的 stride-16 原始余弦互相关。它不定位(全特征
    命中 ~0.325 对 chance 0.36),因此在该层级**需求秩无定义** —— 这正是共同
    层级必须取 stride-16 **基底**而非 stride-4 的原因,必须带误差棒印出来。
    """
    return Readout(
        name='raw16', role='control', stride=16,
        fn=lambda fr, fs, ri, si: align_score(fr, fs),
        note='reverse direction: trained matcher at stride-16; a demand rank '
             'is UNDEFINED here (full-feature hit at or below chance)')


# ==========================================================================
# Per-draw estimation
# ==========================================================================
def gt_cell(gy: int, gx: int, stride: int, h: int, w: int) -> Tuple[int, int]:
    """GT pixel offset -> score-map cell at the read-out's own stride.

    GT 像素位移换算到该读出自身 stride 的热图单元。
    """
    gr = min(max(int(round(gy / stride)), 0), h - 1)
    gc = min(max(int(round(gx / stride)), 0), w - 1)
    return gr, gc


def fit_draw_cca(fs: FeatureSet, fit_idx: np.ndarray, eval_idx: np.ndarray,
                 args: argparse.Namespace,
                 rng: np.random.RandomState) -> Dict[str, object]:
    """De-leaked CCA for one draw, with both significance criteria.

    单次抽样的去泄漏 CCA:按**图像**划分拟合/评估半集,在拟合半集上拟合,
    在留出半集上报告典型相关;显著性同时给出置换零分布(默认)与旧的固定
    +0.05 边距(可比性)。

    Args:
        fs: The feature set.
        fit_idx: Image indices used to FIT the CCA.
        eval_idx: Held-out image indices used to EVALUATE it.
        args: Parsed CLI.
        rng: Seeded generator for this draw.

    Returns:
        Dict with the loadings, means, held-out rho, both significant-direction
        counts, the permutation threshold, and the sample counts to disclose.
    """
    s_h, s_w = fs.sar16.shape[-2:]
    n_tok = s_h * s_w
    o_h, o_w = fs.rgb16.shape[-2:]

    def _cells(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in idx:
            gr, gc = gt_cell(int(fs.gy[i]), int(fs.gx[i]), 16,
                             o_h - s_h + 1, o_w - s_w + 1)
            win = fs.rgb16[i][:, gr:gr + s_h, gc:gc + s_w]
            xs.append(win.reshape(win.shape[0], -1).T)
            ys.append(fs.sar16[i].reshape(fs.sar16.shape[1], -1).T)
        return (np.concatenate(xs).astype(np.float64),
                np.concatenate(ys).astype(np.float64))

    x_fit, y_fit = _cells(fit_idx)
    w_x, w_y, m_x, m_y = fit_cca(x_fit, y_fit, args.d_pca, args.cca_eps)
    n_fit_cells = int(x_fit.shape[0])
    del x_fit, y_fit                       # ~1.8 GiB at n=1500; free it early

    x_ev, y_ev = _cells(eval_idx)
    ax = standardize((x_ev - m_x) @ w_x)
    ay = standardize((y_ev - m_y) @ w_y)
    n_eval_cells = int(x_ev.shape[0])
    del x_ev, y_ev
    rho = (ax * ay).mean(0)

    # (a) permutation null — the disclosed criterion.
    null = permutation_null(ax, ay, len(eval_idx), n_tok, args.perm, rng)
    thr_dir = np.percentile(np.abs(null), args.perm_pct, axis=0)
    thr_global = float(np.percentile(np.abs(null), args.perm_pct))
    sig_perm = int(np.sum(rho > thr_dir))

    # (b) the legacy fixed-margin rule against ONE cyclic re-pairing, kept
    # verbatim for comparability with the published counts.
    # 旧准则:对单次循环重配对的零相关加固定 0.05 边距,原样保留以便对照。
    ay3 = ay.reshape(len(eval_idx), n_tok, ay.shape[1])
    cyc = np.roll(np.arange(len(eval_idx)), 1)
    rho_cyc = (ax * ay3[cyc].reshape(-1, ay.shape[1])).mean(0)
    sig_margin = int(np.sum(rho > (np.abs(rho_cyc) + args.margin)))

    return {
        'w_x': w_x, 'w_y': w_y, 'm_x': m_x, 'm_y': m_y,
        'rho': rho, 'thr_dir': thr_dir, 'thr_global': thr_global,
        'sig_perm': sig_perm, 'sig_margin': sig_margin,
        'top_rho': float(rho[0]),
        'n_fit_imgs': int(len(fit_idx)), 'n_eval_imgs': int(len(eval_idx)),
        'n_fit_cells': n_fit_cells, 'n_eval_cells': n_eval_cells,
        'n_tok': n_tok,
    }


def score_draw(fs: FeatureSet, readouts: Sequence[Readout],
               cca: Dict[str, object], scored_idx: np.ndarray,
               ranks: Sequence[int], device: torch.device
               ) -> Dict[str, Dict[str, np.ndarray]]:
    """Full-feature and rank-restricted hit rates for one draw.

    单次抽样的全特征与限秩命中率。每张图只读一次盘、只做一次子空间投影,
    然后所有读出复用同一份限制后的特征。

    Returns:
        ``{readout_name: {'full': (n, 2), 'rank<k>': (n, 2), 'map': (h, w)}}``
        where the trailing 2 is ``(argmin hit, argmax hit)``.
    """
    m_x = torch.tensor(cca['m_x'], dtype=torch.float32, device=device)
    m_y = torch.tensor(cca['m_y'], dtype=torch.float32, device=device)
    q_d = {k: torch.tensor(orthobasis(cca['w_x'], k), dtype=torch.float32,
                           device=device) for k in ranks}
    q_s = {k: torch.tensor(orthobasis(cca['w_y'], k), dtype=torch.float32,
                           device=device) for k in ranks}

    acc: Dict[str, Dict[str, list]] = {
        ro.name: {'full': [], **{f'rank{k}': [] for k in ranks}}
        for ro in readouts}
    map_size: Dict[str, Tuple[int, int]] = {}

    for count, i in enumerate(scored_idx):
        sar_img = preprocess_image(
            read_image(str(fs.sar_paths[i]), ImageReadMode.GRAY),
            duplicate_channels=True).unsqueeze(0).to(device)
        rgb_img = preprocess_image(
            read_image(str(fs.rgb_paths[i]), ImageReadMode.RGB)
        ).unsqueeze(0).to(device)
        f_rgb = torch.tensor(fs.rgb16[i], dtype=torch.float32,
                             device=device).unsqueeze(0)
        f_sar = torch.tensor(fs.sar16[i], dtype=torch.float32,
                             device=device).unsqueeze(0)
        with torch.no_grad():
            for ro in readouts:
                score = ro.fn(f_rgb, f_sar, rgb_img, sar_img)
                h, w = score.shape[-2:]
                map_size.setdefault(ro.name, (h, w))
                gr, gc = gt_cell(int(fs.gy[i]), int(fs.gx[i]), ro.stride, h, w)
                acc[ro.name]['full'].append(both_hits(score, gr, gc))
            for k in ranks:
                r_rgb = restrict(f_rgb, q_d[k], m_x)
                r_sar = restrict(f_sar, q_s[k], m_y)
                for ro in readouts:
                    score = ro.fn(r_rgb, r_sar, rgb_img, sar_img)
                    h, w = score.shape[-2:]
                    gr, gc = gt_cell(int(fs.gy[i]), int(fs.gx[i]), ro.stride,
                                     h, w)
                    acc[ro.name][f'rank{k}'].append(both_hits(score, gr, gc))
        if (count + 1) % 100 == 0:
            print(f'    scored {count + 1}/{len(scored_idx)}', flush=True)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for ro in readouts:
        out[ro.name] = {key: np.asarray(val)
                        for key, val in acc[ro.name].items()}
        out[ro.name]['map'] = np.asarray(map_size[ro.name])
    return out


def pick_sign(hits_full: np.ndarray) -> int:
    """Which extremum localizes: 0 = argmin (valley), 1 = argmax (peak).

    选定"定位极值":B-ep4 的 stride-4 原始互相关在 GT 处是**谷**(refine 卷积
    翻号),而对齐是**峰**;符号只在第一次抽样上选定一次并固定,避免逐抽样
    选边把命中率选高。
    """
    means = hits_full.mean(0)
    return 0 if means[0] >= means[1] else 1


# ==========================================================================
# Aggregation, CSV, reporting
# ==========================================================================
def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and (population) SD of a per-draw sample, NaNs dropped.

    逐抽样样本的均值与标准差(丢弃 NaN)。
    """
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def fmt_pm(mean: float, sd: float, digits: int = 3) -> str:
    """``mean +/- sd`` with a fixed number of digits, NaN-safe.

    统一的 "均值 +/- 标准差" 排版。
    """
    if np.isnan(mean):
        return 'n/a'
    return f'{mean:.{digits}f} +/- {sd:.{digits}f}'


def write_csvs(curve_rows: List[dict], summary_rows: List[dict],
               out_dir: str) -> Tuple[str, str]:
    """Write the two tidy CSVs the figure script and the letter consume.

    落两张长表:曲线表给出图脚本,汇总表给论文/回复信的标量。
    """
    os.makedirs(out_dir, exist_ok=True)
    curve_path = os.path.join(out_dir, 'probe_common_level_curves.csv')
    summary_path = os.path.join(out_dir, 'probe_common_level_summary.csv')
    curve_cols = ['arm', 'readout', 'role', 'stride', 'rank', 'hit_mean',
                  'hit_sd', 'rec_mean', 'rec_sd', 'full_mean', 'full_sd',
                  'chance', 'n_draws']
    summary_cols = ['arm', 'readout', 'role', 'stride', 'metric', 'mean',
                    'sd', 'n_draws', 'note']
    with open(curve_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_cols)
        writer.writeheader()
        writer.writerows(curve_rows)
    with open(summary_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_cols)
        writer.writeheader()
        writer.writerows(summary_rows)
    return curve_path, summary_path


def parse_ranks(text: str) -> List[int]:
    """Parse ``4,8,16`` into a sorted, de-duplicated list of ints.

    解析 ``--ranks`` 字符串。
    """
    return sorted({int(tok) for tok in text.split(',') if tok.strip()})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI. Every estimator knob is a flag; the OLD values are documented.

    命令行:所有估计器旋钮都可传参,旧值写在 help 里,复现已发表数字只需传旧值。
    """
    parser = argparse.ArgumentParser(
        description='Same-level supply-vs-demand capacity measurement with '
                    'permutation-null significance and across-draw stability.')
    parser.add_argument('--n', type=int, default=DEFAULT_N,
                        help=f'images per draw (published: {PUBLISHED_N})')
    parser.add_argument('--pool', type=int, default=DEFAULT_POOL,
                        help='images encoded once and drawn from; draws '
                             'sample --n of these without replacement')
    parser.add_argument('--d-pca', type=int, default=DEFAULT_D_PCA,
                        help=f'CCA/PCA dimension = the measurement ceiling '
                             f'(published: {PUBLISHED_D_PCA})')
    parser.add_argument('--ranks', type=str,
                        default=','.join(str(k) for k in DEFAULT_RANKS),
                        help='swept ranks (published: '
                             + ','.join(str(k) for k in PUBLISHED_RANKS) + ')')
    parser.add_argument('--draws', type=int, default=DEFAULT_DRAWS,
                        help='independent subset draws (published: 1)')
    parser.add_argument('--perm', type=int, default=DEFAULT_PERM,
                        help='permutation-null shuffles')
    parser.add_argument('--perm-pct', type=float, default=DEFAULT_PERM_PCT,
                        help='percentile of the null taken as the threshold')
    parser.add_argument('--sig-criterion', choices=('perm', 'margin'),
                        default='perm',
                        help='headline significance rule; "margin" is the '
                             'published fixed +0.05 rule (both are computed)')
    parser.add_argument('--margin', type=float, default=DEFAULT_MARGIN,
                        help='the legacy fixed margin')
    parser.add_argument('--cca-eps', type=float, default=1e-4,
                        help='CCA whitening eigenvalue floor (regularization)')
    parser.add_argument('--eval-cap', type=int, default=DEFAULT_EVAL_CAP,
                        help='held-out images SCORED per draw (0 = all); the '
                             'CCA is always fit on the whole fit half')
    parser.add_argument('--arms', type=str, default='supply,demand',
                        help='comma list of supply,demand,cross')
    parser.add_argument('--supply-config', type=str, default=SUPPLY_CONFIG)
    parser.add_argument('--supply-ckpt', type=str, default=SUPPLY_CKPT)
    parser.add_argument('--demand-config', type=str, default=DEMAND_CONFIG)
    parser.add_argument('--demand-ckpt', type=str, default=DEMAND_CKPT)
    parser.add_argument('--image-csv', type=str, default=DEFAULT_IMAGE_CSV)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--out-dir', type=str, default='DebugResult_probes',
                        help='where the two CSVs land')
    parser.add_argument('--cache-dir', type=str, default='.probe_cache')
    parser.add_argument('--cache-dtype', choices=('float32', 'float16'),
                        default='float32',
                        help='float16 halves the cache size at ~1e-3 relative '
                             'feature error')
    parser.add_argument('--no-cache', action='store_true',
                        help='never read or write the caches')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the same-level measurement end to end and report it.

    端到端跑完同层级测量并输出 CSV + 回复信段落。
    """
    args = parse_args(argv)
    ranks = [k for k in parse_ranks(args.ranks) if k <= args.d_pca]
    dropped = [k for k in parse_ranks(args.ranks) if k > args.d_pca]
    if dropped:
        print(f'WARNING: ranks {dropped} exceed --d-pca {args.d_pca} and were '
              f'dropped; raise --d-pca to sweep them.', flush=True)
    if not ranks:
        raise SystemExit('FATAL: no rank in --ranks is <= --d-pca.')
    if args.draws < 10:
        print(f'WARNING: --draws {args.draws} < 10; the reviewer asked for '
              f'stability over >=10 draws. Use this only to reproduce the '
              f'published single-run numbers.', flush=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    df = pd.read_csv(args.image_csv)
    pool = min(args.pool, len(df))
    n_draw = min(args.n, pool)
    if n_draw < args.n:
        print(f'WARNING: --n {args.n} > pool {pool}; using {n_draw}.',
              flush=True)
    print(f'manifest={args.image_csv}  pool={pool}  n/draw={n_draw}  '
          f'd_pca={args.d_pca}  draws={args.draws}  perm={args.perm}  '
          f'ranks={ranks}', flush=True)

    # ---- draws: identical image indices for EVERY arm, so supply and demand
    # are measured on the same draws (the reviewer's "from the same draws").
    # 抽样一次生成,所有 arm 复用 —— 供给与需求来自同一批抽样。
    draw_rng = np.random.RandomState(args.seed)
    draws = []
    for _ in range(args.draws):
        sel = draw_rng.choice(pool, size=n_draw, replace=False)
        draw_rng.shuffle(sel)
        draws.append((sel[:n_draw // 2], sel[n_draw // 2:]))

    # ---- build the arms ---------------------------------------------------
    wanted = [a.strip() for a in args.arms.split(',') if a.strip()]
    for arm_name in wanted:
        if arm_name not in ('supply', 'demand', 'cross'):
            raise SystemExit(f'FATAL: unknown arm "{arm_name}".')
    need_supply_fs = any(a in ('supply', 'cross') for a in wanted)
    need_demand_model = any(a in ('demand', 'cross') for a in wanted)

    specs: Dict[str, EncoderSpec] = {}
    if need_supply_fs:
        specs['supply'] = EncoderSpec('supply', args.supply_config,
                                      args.supply_ckpt, is_stage_2=False)
    if 'demand' in wanted:
        specs['demand'] = EncoderSpec('demand', args.demand_config,
                                      args.demand_ckpt, is_stage_2=True)

    models: Dict[str, torch.nn.Module] = {}
    for key, spec in specs.items():
        models[key] = build_model(spec, device)
    demand_model = models.get('demand')
    if need_demand_model and demand_model is None:
        demand_model = build_model(
            EncoderSpec('demand', args.demand_config, args.demand_ckpt, True),
            device)

    arms: List[Arm] = []
    if 'supply' in wanted:
        arms.append(Arm('supply', 'supply',
                        make_supply_readouts(models['supply'])))
    if 'demand' in wanted:
        arms.append(Arm('demand', 'demand',
                        make_match_readouts(demand_model, 'demand')
                        + [make_control_readout()]))
    if 'cross' in wanted:
        arms.append(Arm('cross', 'supply',
                        make_match_readouts(demand_model, 'control',
                                            suffix='_on_aligned')))

    features: Dict[str, FeatureSet] = {}
    for key, spec in specs.items():
        features[key] = load_or_extract(spec, df, pool, device, args,
                                        models[key])

    # ---- per-draw estimation ---------------------------------------------
    settings = json.dumps({
        'n': n_draw, 'pool': pool, 'd_pca': args.d_pca, 'ranks': ranks,
        'perm': args.perm, 'perm_pct': args.perm_pct, 'margin': args.margin,
        'cca_eps': args.cca_eps, 'eval_cap': args.eval_cap,
        'seed': args.seed, 'csv': args.image_csv,
        'ckpts': {k: s.ckpt for k, s in specs.items()},
    }, sort_keys=True)
    settings_hash = hashlib.sha1(settings.encode('utf-8')).hexdigest()[:12]
    draw_cache_dir = os.path.join(args.cache_dir, 'draws')
    if not args.no_cache:
        os.makedirs(draw_cache_dir, exist_ok=True)

    cca_stats: Dict[str, List[dict]] = {key: [] for key in features}
    # per (arm, readout): lists over draws
    curves: Dict[Tuple[str, str], Dict[str, list]] = {}
    signs: Dict[Tuple[str, str], int] = {}
    chances: Dict[Tuple[str, str], float] = {}

    for d_idx, (fit_idx, ev_idx) in enumerate(draws):
        print(f'\n===== draw {d_idx + 1}/{len(draws)} '
              f'(fit {len(fit_idx)} imgs / held-out {len(ev_idx)} imgs) =====',
              flush=True)
        rng = np.random.RandomState(args.seed + 1000 + d_idx)
        scored = ev_idx if args.eval_cap <= 0 else ev_idx[:args.eval_cap]
        for arm in arms:
            cache_path = os.path.join(
                draw_cache_dir, f'{arm.name}_{settings_hash}_d{d_idx}.json')
            if not args.no_cache and os.path.exists(cache_path):
                with open(cache_path, encoding='utf-8') as handle:
                    payload = json.load(handle)
                print(f'  [{arm.name}] draw cache HIT', flush=True)
            else:
                fs = features[arm.fs_key]
                cca = fit_draw_cca(fs, fit_idx, ev_idx, args,
                                   np.random.RandomState(args.seed + d_idx))
                scores = score_draw(fs, arm.readouts, cca, scored, ranks,
                                    device)
                payload = {'cca': {k: v for k, v in cca.items()
                                   if k in ('sig_perm', 'sig_margin',
                                            'thr_global', 'top_rho',
                                            'n_fit_imgs', 'n_eval_imgs',
                                            'n_fit_cells', 'n_eval_cells')},
                           'rho_top': [float(v) for v in cca['rho'][:16]],
                           'thr_top': [float(v) for v in cca['thr_dir'][:16]],
                           'readouts': {}}
                for ro in arm.readouts:
                    res = scores[ro.name]
                    payload['readouts'][ro.name] = {
                        'map': [int(res['map'][0]), int(res['map'][1])],
                        'full': res['full'].tolist(),
                        'ranks': {str(k): res[f'rank{k}'].tolist()
                                  for k in ranks},
                    }
                if not args.no_cache:
                    with open(cache_path, 'w', encoding='utf-8') as handle:
                        json.dump(payload, handle)

            cca_stats[arm.fs_key].append(payload['cca'])
            print(f'  [{arm.name}] CCA: sig(perm p{args.perm_pct:.0f})='
                  f'{payload["cca"]["sig_perm"]}  sig(margin +{args.margin})='
                  f'{payload["cca"]["sig_margin"]}  thr='
                  f'{payload["cca"]["thr_global"]:.3f}  '
                  f'top rho={payload["cca"]["top_rho"]:+.3f}', flush=True)
            for ro in arm.readouts:
                block = payload['readouts'][ro.name]
                full_arr = np.asarray(block['full'])
                key = (arm.name, ro.name)
                if key not in signs:
                    signs[key] = pick_sign(full_arr)
                    h, w = block['map']
                    chances[key] = 9.0 / (h * w)
                    curves[key] = {'full': [], 'curve': [], 'rec': [],
                                   'sat90': [], 'sat80': [], 'ro': ro}
                col = signs[key]
                floor = chances[key]
                full = float(full_arr.mean(0)[col])
                curve = [float(np.asarray(block['ranks'][str(k)]).mean(0)[col])
                         for k in ranks]
                denom = max(full - floor, 1e-9)
                rec = [(v - floor) / denom for v in curve]
                sat90 = saturation_rank(curve, full, floor, 0.90, ranks)
                sat80 = saturation_rank(curve, full, floor, 0.80, ranks)
                undefined = full <= floor + UNDEFINED_MARGIN
                curves[key]['full'].append(full)
                curves[key]['curve'].append(curve)
                curves[key]['rec'].append(rec)
                curves[key]['sat90'].append(
                    float('nan') if (sat90 is None or undefined) else sat90)
                curves[key]['sat80'].append(
                    float('nan') if (sat80 is None or undefined) else sat80)
                flag = '  [does NOT localize -> rank undefined]' if undefined \
                    else ''
                print(f'    {ro.name:<18s} full={full:.3f} '
                      f'chance={floor:.3f} sign='
                      f'{"MIN" if col == 0 else "MAX"} '
                      f'sat0.90={sat90}{flag}', flush=True)

    # ---- aggregate --------------------------------------------------------
    curve_rows: List[dict] = []
    summary_rows: List[dict] = []
    for (arm_name, ro_name), blob in curves.items():
        ro: Readout = blob['ro']
        n_d = len(blob['full'])
        full_m, full_s = mean_sd(blob['full'])
        floor = chances[(arm_name, ro_name)]
        hits = np.asarray(blob['curve'])          # (draws, ranks)
        recs = np.asarray(blob['rec'])
        for j, k in enumerate(ranks):
            h_m, h_s = mean_sd(hits[:, j])
            r_m, r_s = mean_sd(recs[:, j])
            curve_rows.append({
                'arm': arm_name, 'readout': ro_name, 'role': ro.role,
                'stride': ro.stride, 'rank': k,
                'hit_mean': round(h_m, 6), 'hit_sd': round(h_s, 6),
                'rec_mean': round(r_m, 6), 'rec_sd': round(r_s, 6),
                'full_mean': round(full_m, 6), 'full_sd': round(full_s, 6),
                'chance': round(floor, 6), 'n_draws': n_d})

        undefined = full_m <= floor + UNDEFINED_MARGIN
        def _add(metric: str, mean: float, sd: float, note: str = '') -> None:
            summary_rows.append({
                'arm': arm_name, 'readout': ro_name, 'role': ro.role,
                'stride': ro.stride, 'metric': metric,
                'mean': round(mean, 6) if not np.isnan(mean) else '',
                'sd': round(sd, 6) if not np.isnan(sd) else '',
                'n_draws': n_d, 'note': note})

        _add('full_hit', full_m, full_s, ro.note)
        _add('chance', floor, 0.0,
             f'9 / map cells; sign='
             f'{"argmin" if signs[(arm_name, ro_name)] == 0 else "argmax"}')
        for frac, keyname in ((0.90, 'sat90'), (0.80, 'sat80')):
            vals = blob[keyname]
            ok = int(np.sum(~np.isnan(vals)))
            m, s = mean_sd(vals)
            note = f'{ok}/{n_d} draws saturated'
            if undefined:
                note = ('read-out does not localize (full <= chance): a '
                        'recovery rank is UNDEFINED here')
            elif ok == 0:
                note = (f'no draw saturated: rank > {ranks[-1]} (the '
                        f'--d-pca ceiling) — quote as a lower bound')
            _add(f'sat_rank_{int(frac * 100):03d}', m, s, note)
        k_key_idx = ranks.index(K_KEY) if K_KEY in ranks else None
        if k_key_idx is not None:
            _add(f'recovery_at_k{K_KEY}', *mean_sd(recs[:, k_key_idx]),
                 note='the paper\'s headline rank')

    for fs_key, stats in cca_stats.items():
        if not stats:
            continue
        n_d = len(stats)

        def _cca_add(metric: str, values: Sequence[float],
                     note: str = '') -> None:
            m, s = mean_sd(values)
            summary_rows.append({
                'arm': fs_key, 'readout': '(cca)', 'role': 'estimator',
                'stride': 16, 'metric': metric, 'mean': round(m, 6),
                'sd': round(s, 6), 'n_draws': n_d, 'note': note})

        _cca_add('cca_sig_dirs_perm', [s['sig_perm'] for s in stats],
                 f'held-out rho above the p{args.perm_pct:.0f} per-direction '
                 f'permutation null ({args.perm} shuffles)')
        _cca_add('cca_sig_dirs_margin', [s['sig_margin'] for s in stats],
                 f'LEGACY criterion: rho > |single cyclic null| + '
                 f'{args.margin}')
        _cca_add('perm_threshold', [s['thr_global'] for s in stats],
                 f'global p{args.perm_pct:.0f} of |null rho|')
        _cca_add('top_rho', [s['top_rho'] for s in stats])
        _cca_add('n_fit_imgs', [s['n_fit_imgs'] for s in stats],
                 'CCA fit/eval split is BY IMAGE')
        _cca_add('n_eval_imgs', [s['n_eval_imgs'] for s in stats])
        _cca_add('n_fit_cells', [s['n_fit_cells'] for s in stats],
                 'token pairs the CCA is fit on')
        _cca_add('n_eval_cells', [s['n_eval_cells'] for s in stats])
        _cca_add('d_pca', [args.d_pca] * n_d, 'measurement ceiling')

    curve_path, summary_path = write_csvs(curve_rows, summary_rows,
                                          args.out_dir)

    # ---- report -----------------------------------------------------------
    print('\n' + '=' * 74, flush=True)
    print('RESPONSE-LETTER BLOCK (R2-4a same level / R2-4b estimator)',
          flush=True)
    print('=' * 74, flush=True)
    print(f'Common level: the RAW 1024-d stride-16 token space. Both the '
          f'rank-k\nrestriction and the CCA basis live there for every arm; '
          f'each side is then\nread out through its OWN trained operator '
          f'(alignment vs proj_*_4/8). This is\nwhat the published probes '
          f'already did — the paper\'s "stride-4 features"\nwording is the '
          f'error, not the measurement.', flush=True)
    print(f'\nEstimator: CCA fit/eval split BY IMAGE; D_PCA={args.d_pca} '
          f'(published {PUBLISHED_D_PCA});\nranks swept to {ranks[-1]} '
          f'(published {PUBLISHED_RANKS[-1]}); n={n_draw} images/draw '
          f'(published {PUBLISHED_N});\n{len(draws)} independent subset draws; '
          f'significance = {args.sig_criterion} '
          f'({args.perm} shuffles,\np{args.perm_pct:.0f} of the null) — '
          f'the published rule was a fixed +{args.margin} margin.', flush=True)

    for fs_key, stats in cca_stats.items():
        if not stats:
            continue
        head = 'sig_perm' if args.sig_criterion == 'perm' else 'sig_margin'
        print(f'\n[{fs_key}] significant shared directions '
              f'({args.sig_criterion}) = '
              f'{fmt_pm(*mean_sd([s[head] for s in stats]), digits=2)} '
              f'of {args.d_pca}', flush=True)
        print(f'  legacy(+{args.margin}) count = '
              f'{fmt_pm(*mean_sd([s["sig_margin"] for s in stats]), 2)}   '
              f'null threshold = '
              f'{fmt_pm(*mean_sd([s["thr_global"] for s in stats]), 3)}   '
              f'top held-out rho = '
              f'{fmt_pm(*mean_sd([s["top_rho"] for s in stats]), 3)}',
              flush=True)
        print(f'  fit {stats[0]["n_fit_imgs"]} imgs / '
              f'{stats[0]["n_fit_cells"]} token pairs; held out '
              f'{stats[0]["n_eval_imgs"]} imgs / '
              f'{stats[0]["n_eval_cells"]} token pairs', flush=True)

    print('\nSaturation ranks (mean +/- SD over draws), one ruler, one basis:',
          flush=True)
    for (arm_name, ro_name), blob in curves.items():
        ro: Readout = blob['ro']
        floor = chances[(arm_name, ro_name)]
        full_m, full_s = mean_sd(blob['full'])
        sat_m, sat_s = mean_sd(blob['sat90'])
        ok = int(np.sum(~np.isnan(blob['sat90'])))
        if full_m <= floor + UNDEFINED_MARGIN:
            verdict = ('UNDEFINED — this read-out does not localize at full '
                       'rank, so there is nothing to recover')
        elif ok == 0:
            verdict = f'> {ranks[-1]} (never saturates at the ceiling)'
        else:
            verdict = (f'{fmt_pm(sat_m, sat_s, 1)}  '
                       f'({ok}/{len(blob["sat90"])} draws saturated)')
        print(f'  {arm_name:<7s} {ro_name:<18s} ({ro.role:<7s}, stride-'
              f'{ro.stride:<2d})  full={fmt_pm(full_m, full_s)}  '
              f'chance={floor:.3f}\n           0.90-recovery rank = {verdict}',
              flush=True)

    supply_ranks = [mean_sd(b['sat90'])[0] for (a, _), b in curves.items()
                    if b['ro'].role == 'supply']
    demand_ranks = [mean_sd(b['sat90'])[0] for (a, _), b in curves.items()
                    if b['ro'].role == 'demand']
    sup = np.nanmax(supply_ranks) if supply_ranks else float('nan')
    dem = np.nanmin(demand_ranks) if demand_ranks else float('nan')
    print('\nORDERING CHECK (supply < demand at the common level):', flush=True)
    if np.isnan(sup) or np.isnan(dem):
        print('  NOT EVALUABLE from this run (a side has no defined rank). '
              'Report exactly that.', flush=True)
    elif sup < dem:
        print(f'  HOLDS: supply {sup:.1f} < demand {dem:.1f} at one level, '
              f'one basis, one estimator.', flush=True)
    else:
        print(f'  DOES NOT HOLD: supply {sup:.1f} >= demand {dem:.1f}. '
              f'REPORT AS-IS — the letter\n  commits to reporting whatever '
              f'the same-level measurement returns, and the\n  capacity-'
              f'mismatch wording in Sec. IV must be retracted, not re-tuned.',
              flush=True)

    print(f'\nCSV (figure script input): {curve_path}', flush=True)
    print(f'CSV (scalars for the text): {summary_path}', flush=True)
    print('\nCOMMON_LEVEL_DONE', flush=True)


if __name__ == '__main__':
    main()
