#!/usr/bin/env bash
# Score only the missing final R/Fr checkpoint with bounded GPU memory.
# 在显存上限内，仅补测缺失的 R/Fr 末代权重。
set -eo pipefail
source /root/autodl-tmp/setup/activate.sh
source /root/autodl-tmp/repos/005_Template-Matching-DINOv3/Scripts/common.sh
export TM_CUDNN_BENCHMARK_LIMIT=10 TM_CORRELATION_CHANNELS_LAST=1
export TM_SEED_FAST=1
export PYTHONPATH="/root/autodl-tmp/setup/campaign_20260925/eval_memory_cap:${PYTHONPATH:-}"
export TM_EVAL_ALLOC_FRACTION=0.20
tm_eval TM_Model_F Checkpoint_F_rfr_256_s2026/model_epoch_4.pth \
    x2_rfr_s2026_20k "${TM_REPAIR_GPU:?}" \
    /root/autodl-tmp/setup/campaign_20260925/eval20k.csv 20000 2 8997
unset TM_EVAL_ALLOC_FRACTION
tm_pck x2_rfr_s2026_20k > DebugResult_x2_rfr_s2026_20k/pck_summary.txt
bash /root/autodl-tmp/setup/campaign_20260925/run_seed.sh 2026 eval
echo 'SEED2026_BACKFILL_COMPLETE'
