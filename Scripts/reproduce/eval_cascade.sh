#!/usr/bin/env bash
# Reproduce the paper's headline result with the RELEASED cascade checkpoint.
# 用发布的 cascade 权重复现论文主结果。
#
#   Table I — in-domain SEN1-2, proposed method (TM_Model_C, direct-trained cascade)
#   表 I —— 域内 SEN1-2，本文方法（TM_Model_C，直接训练的级联）
#
# Usage / 用法:
#   bash Scripts/reproduce/eval_cascade.sh [ckpt] [gpu]
#
#   ckpt  default Checkpoint_C_phase2_8gpu/model_epoch_2.pth (the released model)
#         默认为发布的 Checkpoint_C_phase2_8gpu/model_epoch_2.pth
#   gpu   default 0
#
# Override the eval set / 更换评测集:
#   TM_EVAL_CSV=/path/list.csv TM_EVAL_N=20000 bash Scripts/reproduce/eval_cascade.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

CKPT="${1:-Checkpoint_C_phase2_8gpu/model_epoch_2.pth}"
GPU="${2:-0}"
TAG="repro_cascade"

tm_eval TM_Model_C "$CKPT" "$TAG" "$GPU" || exit 1
tm_pck "$TAG"

cat <<'NOTE'

Reference (paper Table I, full test set n=2,236,840):
论文表 I 参考值（全测试集 n=2,236,840）:
    CMR(1) 93.30 | CMR(3) 95.83 | CMR(5) 96.30 | mean 1.13 px | median 0 px
The default 2k subsample lands within sampling noise of these numbers.
默认的 2k 子采样结果应落在上述数字的采样噪声范围内。
NOTE
