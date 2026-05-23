# -*- coding: utf-8 -*-
"""Primary-head decoding for inference and test-time CSV export.

Shared between ``accuracy()`` (a0 branch in ``_interface_base.py``) and
``save_result()`` so the exported predictions exactly match the metric the
training-side accuracy reports.

主推理 head 解码工具，accuracy() 的 a0 分支和 save_result() 共用，
保证测试导出的预测坐标与训练评估口径完全一致。
"""
from __future__ import annotations

from typing import Tuple

import torch

from .sub_pixel import quadratic_subpixel_fit


def decode_primary(
    output: dict,
    argmax_type: str = 'hard',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode the primary head's heatmap to pixel coords + peak score.

    解码主 head 的 heatmap，输出像素坐标和峰值分数。

    Selection rule (same as ``MatchInterface.accuracy``):
      - CascadeHead (C/E variants): use ``heatmap_cascade[1]`` (stride-1
        cascade) with ``fine_anchor_pix`` anchor.
      - StandardMatchingHead (A/B/D variants): use ``heatmap_fine`` with
        ``fine_anchor_pix`` anchor.

    Args:
        output: Model output dict; expected to contain either
            ``heatmap_cascade`` (list of stride→tensor) or ``heatmap_fine``,
            plus ``fine_anchor_pix`` (B, 2) anchor in image pixels.
        argmax_type: ``'hard'`` (flat argmax) or ``'soft_subpixel'``
            (quadratic 3x3 sub-pixel fit on top of the hard argmax).

    Returns:
        pred_xy: (B, 2) float tensor of predicted (x, y) in image pixels.
        score:   (B,) float tensor of the raw heatmap value at the integer
                 peak (logit / similarity). Useful for confidence filtering
                 and PR curves downstream.

    Raises:
        RuntimeError: if neither ``heatmap_cascade`` nor ``heatmap_fine`` is
            present in ``output`` (test mode requires a fine head).
    """
    if 'heatmap_cascade' in output:
        logits = output['heatmap_cascade'][1]
    elif 'heatmap_fine' in output:
        logits = output['heatmap_fine']
    else:
        raise RuntimeError(
            "decode_primary: neither 'heatmap_cascade' nor 'heatmap_fine' "
            "present in model output; test mode requires a fine head.")

    if 'fine_anchor_pix' not in output:
        raise RuntimeError(
            "decode_primary: 'fine_anchor_pix' missing; cannot anchor the "
            "heatmap argmax back to image pixel coords.")
    anchor = output['fine_anchor_pix'].float()

    batch = logits.shape[0]
    h_f, w_f = logits.shape[-2:]
    flat = logits.reshape(batch, -1)
    idx = flat.argmax(dim=1)
    score = flat.gather(1, idx.unsqueeze(1)).squeeze(1).float()

    if argmax_type == 'soft_subpixel':
        peak_int, sub_off = quadratic_subpixel_fit(logits)
        pred = anchor + peak_int.float() + sub_off
    else:
        ux = (idx % w_f).float()
        vy = (idx // w_f).float()
        pred = anchor + torch.stack([ux, vy], dim=1)

    return pred, score
