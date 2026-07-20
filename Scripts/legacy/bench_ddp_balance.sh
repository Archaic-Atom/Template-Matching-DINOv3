#!/usr/bin/env bash
# Integration test for the _balanced_split_by_node fix:
# 11 shards (not divisible by 4) -> floor(11/4)=2 shards per rank, all equal.
# Tests that dist.barrier() at epoch boundaries doesn't timeout AND that
# bar matches actual iter count.

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate stereoA

export COLUMNS=200
SHARDS='/data5/datasets/TM_shards/shard-{000000..000010}.tar'   # 11 shards
IMGNUM=256000     # 8 shards effectively used (4 ranks × 2 each)
BATCH=32
EPOCHS=2

rm -rf ./Checkpoint_BENCH_DDP/ ./bench_DDP_balance.log
mkdir -p ./Checkpoint_BENCH_DDP/

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u Source/main.py \
    --mode train \
    --batchSize ${BATCH} \
    --gpu 4 \
    --trainListPath "${SHARDS}" \
    --auto_save_num 1 \
    --dataloaderNum 8 \
    --dataloaderType iterable \
    --imgNum ${IMGNUM} \
    --maxEpochs ${EPOCHS} \
    --sampleNum 1 \
    --lr 0.0002 \
    --dist True \
    --modelName TM_Model_A \
    --dataset tmdataset \
    --is_stage_2 False \
    --shuffle_buffer 30000 \
    --log ./log_bench_ddp/ \
    --outputDir ./result_bench_ddp/ \
    --resultImgDir ./ResultImg_bench_ddp/ \
    --modelDir ./Checkpoint_BENCH_DDP/ 2>&1 | tee ./bench_DDP_balance.log

echo
echo '=== POST-RUN ==='
grep -E 'TrainProcess|Save model in|Finish training|timeout' ./bench_DDP_balance.log || echo '(no key events)'
ls -la ./Checkpoint_BENCH_DDP/
