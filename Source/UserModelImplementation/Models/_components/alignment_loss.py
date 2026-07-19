# -*- coding: utf-8 -*-
"""Stage-1 alignment loss: PatchNCE on local window + global cosine.

Copied verbatim from TM_Model_A/Loss/alignment_loss_func.py so behavior
matches the in-progress baseline run exactly. The public `alignment_loss`
is registered for config-driven dispatch.

Optional `normalize_logits` flag (default False, so the legacy baseline is
byte-for-byte unchanged): when set True via the stage_1.loss yaml block, the
PatchNCE logits become a true cosine / tau by L2-normalizing the DINO teacher
side before the dot-product. See ``soft_window_patchnce_loss`` for why this
matters (the teacher arrives un-normalized, so the default logits are
``cos * ||r|| / tau``, not cosine).

可选 ``normalize_logits`` 开关（默认 False，旧基线逐字节不变）：经 stage_1.loss
的 yaml 置 True 后，对 DINO 教师侧做 L2 归一化，使 PatchNCE 的 logits 成为真正的
cosine / tau —— 否则教师侧未归一化，logits 实为 ``cos * ||r|| / tau``。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import register


def _grid_index(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(h, device=device)
    xs = torch.arange(w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)


def soft_window_patchnce_loss(
    sar16: torch.Tensor,
    rgb16: torch.Tensor,
    tau: float = 0.07,
    radius: int = 1,
    normalize_logits: bool = False,
) -> torch.Tensor:
    """Soft-window PatchNCE with batch negatives. See A's docstring.

    When ``normalize_logits`` is True, both feature maps are L2-normalized
    along the channel dim before the dot-product, so every logit is a true
    cosine / tau. The SAR side already arrives unit-norm from ``ProjHead``;
    the DINO teacher side does NOT (it is the raw ``norm=False`` block-23
    output with large, position-varying norms), so without this the logits
    are ``cos * ||r|| / tau`` rather than cosine. Default False keeps the
    legacy baseline behavior byte-for-byte.

    ``normalize_logits`` 为真时，先沿通道维对两路特征做 L2 归一化，使每个
    logit 都是真正的 cosine / tau。SAR 侧经 ``ProjHead`` 已是单位向量，但 DINO
    教师侧未归一化（``norm=False`` 的原始 block-23 输出，范数大且逐位置不均），
    否则 logits 实为 ``cos * ||r|| / tau``。默认 False，保持旧基线逐字节一致。
    """
    if sar16.shape != rgb16.shape:
        raise ValueError(f"Shape mismatch: sar16={sar16.shape}, rgb16={rgb16.shape}")

    b, d, h, w = sar16.shape
    n = h * w
    device = sar16.device

    if normalize_logits:
        # True cosine InfoNCE: unit-norm both sides along the channel dim so
        # the temperature tau actually controls sharpness on cos in [-1, 1].
        # 真 cosine InfoNCE：沿通道维把两路都归一化为单位向量，让温度 tau 真正
        # 控制 [-1, 1] 上 cosine 的锐度。SAR 侧已是单位向量，此处为 no-op。
        sar16 = F.normalize(sar16, p=2, dim=1)
        rgb16 = F.normalize(rgb16, p=2, dim=1)

    s = sar16.flatten(2).transpose(1, 2).contiguous()
    r = rgb16.flatten(2).transpose(1, 2).contiguous()

    s_all = s.view(b * n, d)
    r_all = r.view(b * n, d)
    logits = (s_all @ r_all.t()) / tau

    coords = _grid_index(h, w, device)
    yy = coords[:, 0]; xx = coords[:, 1]
    dy = (yy[:, None] - yy[None, :]).abs()
    dx = (xx[:, None] - xx[None, :]).abs()
    within = (dy <= radius) & (dx <= radius)

    pos_mask = torch.zeros((b * n, b * n), device=device, dtype=torch.bool)
    for bi in range(b):
        a = bi * n
        pos_mask[a:a + n, a:a + n] = within

    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    lse_pos = torch.logsumexp(pos_logits, dim=1)
    lse_all = torch.logsumexp(logits, dim=1)
    return (-(lse_pos - lse_all)).mean()


def global_cosine_loss(sar16: torch.Tensor, rgb16: torch.Tensor) -> torch.Tensor:
    g_sar = F.normalize(sar16.mean(dim=(2, 3)), p=2, dim=1)
    g_rgb = F.normalize(rgb16.mean(dim=(2, 3)), p=2, dim=1)
    return (1.0 - (g_sar * g_rgb).sum(dim=1)).mean()


def alignment_cos_accuracy(sar16: torch.Tensor, rgb16: torch.Tensor) -> torch.Tensor:
    """Per-position cosine similarity between SAR and RGB features.

    Normalize each position's feature vector along the channel dim, take
    the dot product per position, then mean over (B, H, W). Range
    [-1, 1]; higher means better aligned. Reported as Stage-1 accuracy
    so loss and accuracy carry distinct, comparable signals.
    """
    s = F.normalize(sar16, p=2, dim=1)
    r = F.normalize(rgb16, p=2, dim=1)
    return (s * r).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Learned-inner-product Stage-1 loss (PatchNCE on projected pair + VICReg
# anti-rank-collapse). 可学内积 Stage-1 损失:投影对上的 PatchNCE + VICReg 防秩塌缩。
# ---------------------------------------------------------------------------

def effective_rank(feat: torch.Tensor) -> torch.Tensor:
    """exp(entropy of normalized squared singular values) — probe formula.

    有效秩 = 归一化奇异值平方的熵的指数(与 probe_b_collapse.py 公式一致)。
    Range [1, D]: 1 = collapsed to one direction, D = perfectly spread.

    Args:
        feat: (N, D) feature matrix, PRE-norm (post-norm conceals collapse).
            (N, D) 特征矩阵,必须是 pre-norm。
    """
    fc = feat - feat.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(fc)
    p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
    return torch.exp(-(p * torch.log(p + 1e-12)).sum())


def _vicreg_variance(feat: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance hinge: push every dim's std up to 1.

    VICReg 方差铰链项:把每一维的 std 推到 1(低于 1 才有惩罚)。
    """
    std = torch.sqrt(feat.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()


def _vicreg_covariance(feat: torch.Tensor) -> torch.Tensor:
    """VICReg covariance penalty: squared off-diagonal covariances / D.

    VICReg 协方差项:协方差矩阵非对角元素平方和除以维度 D,逼各维去相关。
    """
    n, d = feat.shape
    fc = feat - feat.mean(dim=0, keepdim=True)
    cov = (fc.t() @ fc) / max(n - 1, 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / d


@register("alignment_ip_loss")
def alignment_ip_loss(
    s_pre: torch.Tensor,
    d_pre: torch.Tensor,
    tau: float = 0.07,
    radius: int = 1,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.04,
    erank_rows: int = 4096,
):
    """Learned-IP Stage-1 loss: PatchNCE(norm proj) + VICReg(pre-norm proj).

    可学内积 Stage-1 损失:归一化投影上的 PatchNCE + pre-norm 投影上的
    VICReg 方差/协方差正则,目标是把两侧 pre-norm 有效秩稳在 ≥64
    (Stage-2 匹配是 ~64 维分布式的;裸 InfoNCE 会理性地塌到 ~25 维)。

    Args:
        s_pre: (B, D, h, w) SAR-side pre-norm projection (W_s output).
            SAR 侧 pre-norm 投影。
        d_pre: (B, D, h, w) DINO-side pre-norm projection (W_d output).
            DINO 侧 pre-norm 投影。
        tau: PatchNCE temperature. 温度。
        radius: soft positive window radius in stride-16 cells. 软正窗半径。
        lambda_var: weight of the VICReg variance hinge. 方差项权重。
        lambda_cov: weight of the VICReg covariance penalty. 协方差项权重。
        erank_rows: row cap for the effective-rank SVD (logging only).
            有效秩 SVD 的采样行数上限(仅日志用)。

    Returns:
        Tuple ``(total, components)``; components log as l1..l5 =
        patch_nce, vic_var, vic_cov, erank_s, erank_d. erank_* are detached
        diagnostics, NOT optimized — watch erank_s: healthy ≥64, the
        rank-deficiency trap shows up as drift toward ~25.
        erank_* 只是诊断量,不参与优化 —— 盯 erank_s:健康 ≥64,
        往 ~25 漂就是秩缺陷陷阱复发。
    """
    # PatchNCE on the projected pair; normalize_logits=True makes every
    # logit a true cosine/τ in the learned space.
    # 投影对上的 PatchNCE;normalize_logits=True 使 logits 为该空间的真余弦/τ。
    l_nce = soft_window_patchnce_loss(
        s_pre, d_pre, tau=tau, radius=radius, normalize_logits=True)

    # Flatten to (N, D) for the statistics terms.
    # 展平成 (N, D) 给统计项。
    s_flat = s_pre.flatten(2).transpose(1, 2).reshape(-1, s_pre.shape[1])
    d_flat = d_pre.flatten(2).transpose(1, 2).reshape(-1, d_pre.shape[1])

    l_var = 0.5 * (_vicreg_variance(s_flat) + _vicreg_variance(d_flat))
    l_cov = 0.5 * (_vicreg_covariance(s_flat) + _vicreg_covariance(d_flat))

    total = l_nce + lambda_var * l_var + lambda_cov * l_cov

    with torch.no_grad():
        # .float(): svdvals is unhappy under fp16/AMP. 半精度下 SVD 不稳,转 fp32。
        er_s = effective_rank(s_flat[:erank_rows].detach().float())
        er_d = effective_rank(d_flat[:erank_rows].detach().float())

    components = {
        'patch_nce': l_nce.detach(),
        'vic_var': l_var.detach(),
        'vic_cov': l_cov.detach(),
        'erank_s': er_s.detach(),
        'erank_d': er_d.detach(),
    }
    return total, components


@register("alignment_loss")
def alignment_loss(
    sar16: torch.Tensor,
    rgb16: torch.Tensor,
    tau: float = 0.07,
    radius: int = 1,
    lambda_global: float = 0.1,
    normalize_logits: bool = False,
):
    """Combined Stage-1 alignment loss: local PatchNCE + global cosine.

    组合 Stage 1 对齐损失：局部 PatchNCE + 全局余弦。

    Returns:
        Tuple ``(total, components)`` where ``total`` is the backward-traced
        sum and ``components`` is a dict of detached per-term losses for
        logging via ``user_loss`` → l1, l2, ...
        返回 ``(total, components)``：total 是反传用的总损失；components
        是逐项 detach 后的 scalar dict，给框架显示成 l1、l2、...
    """
    l_local = soft_window_patchnce_loss(
        sar16, rgb16, tau=tau, radius=radius, normalize_logits=normalize_logits)
    l_global = global_cosine_loss(sar16, rgb16)
    total = l_local + lambda_global * l_global
    components = {
        'patch_nce': l_local.detach(),
        'global_cos': l_global.detach(),
    }
    return total, components
