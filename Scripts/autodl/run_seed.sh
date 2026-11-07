#!/usr/bin/env bash
# Preserve the original recipe for each sequential seed and evaluate afterward.
# 每个 seed 保持原配方串行训练，训练结束后独立评估。
set -eo pipefail
source /root/autodl-tmp/setup/activate.sh
set -u
RUN=/root/autodl-tmp/setup/campaign_20260925
export TM_CUDNN_BENCHMARK_LIMIT=10 TM_CORRELATION_CHANNELS_LAST=1
unset TORCH_CUDNN_V8_API_DISABLED
SEED="${1:?seed}"
MODE="${2:?train or eval}"
case "$SEED" in 2025|2026|2027|2028) ;; *) exit 2 ;; esac
export TM_2X2_CONCURRENT=0 TM_2X2_GPUS=0,1
export TM_2X2_MANIFEST=/root/autodl-tmp/setup/shard_train_256_subset40.csv
export TM_2X2_STAGE1=Checkpoint_F_stage1_256/model_epoch_9.pth
export TM_2X2_EPOCHS=5 TM_DATALOADER_NUM=8 TM_SHUFFLE_BUFFER=100000
export TM_LR=1e-3 TM_SEED_FAST=1
export TM_2X2_FORCE=0 TM_2X2_RESTART_PARTIAL=0
export TM_2X2_POLL=30 TM_2X2_HEARTBEAT=300
export TM_EVAL_CSV="$RUN/eval20k.csv" TM_EVAL_N=20000
export TM_2X2_EVAL_BATCH=2 TM_2X2_TAG_SUFFIX=20k
case "$MODE" in
    train)
        TM_2X2_DRY_RUN=1 bash Scripts/train/train_2x2.sh "$SEED" 2 32 8901
        TM_2X2_DRY_RUN=0 bash Scripts/train/train_2x2.sh "$SEED" 2 32 8901
        ;;
    eval)
        /root/miniconda3/bin/python "$RUN/eval_complete_guard.py" wait "$SEED"
        TM_2X2_DRY_RUN=0 bash Scripts/reproduce/eval_2x2.sh "$SEED" auto all
        /root/miniconda3/bin/python "$RUN/eval_complete_guard.py" verify "$SEED"
        ;;
    *) exit 2 ;;
esac
