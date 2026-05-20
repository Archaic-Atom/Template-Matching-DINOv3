#!/usr/bin/env bash
# Experiment 1: verify that imgNum + progress bar agree under the fixed
# WebDataset with_epoch / world_size dataloader.
#
# Single-GPU (GPU 4, doesn't compete with A's 0-3 training). Small dataset
# (4 shards = 128,000 samples). Under the fixed dataloader:
#     world_size = 1  (--dist False)
#     per_rank_epoch = 128000 / 1 = 128000 samples
#     iter cap per epoch = 128000 / batchSize(8) = 16000 iters
# So we expect bar = 16000/16000 to coincide with `Save model in :
# model_epoch_0.pth` being written. If save fires earlier or later, the
# dataloader → progress-bar contract is broken.

set -e

# Activate the env that has torch + JackFramework + webdataset.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate stereoA

GPU=4
SHARD_PATTERN='/data5/datasets/TM_shards/shard-{000000..000003}.tar'   # 4 shards
IMGNUM=128000           # 4 × 32000
BATCH=8                 # global batch (= per-rank since dist=False)
EPOCHS=2                # run two epochs so we see TWO saves
MODEL_DIR='./Checkpoint_BENCH_E1/'
LOG_FILE='./bench_E1_imgnum.log'

export COLUMNS=200
mkdir -p "${MODEL_DIR}"
rm -f "${MODEL_DIR}"/model_epoch_*.pth "${MODEL_DIR}"/checkpoint.list 2>/dev/null || true

echo "[E1] verify imgNum + progress bar correctness"
echo "     GPU: ${GPU}    shards: 4    imgNum: ${IMGNUM}    batch: ${BATCH}"
echo "     expected iter cap per epoch = ${IMGNUM}/${BATCH} = $((IMGNUM/BATCH))"
echo "     log: ${LOG_FILE}"
echo

CUDA_VISIBLE_DEVICES=${GPU} python -u Source/main.py \
    --mode train \
    --batchSize ${BATCH} \
    --gpu 1 \
    --trainListPath "${SHARD_PATTERN}" \
    --auto_save_num 1 \
    --dataloaderNum 8 \
    --dataloaderType iterable \
    --imgNum ${IMGNUM} \
    --maxEpochs ${EPOCHS} \
    --sampleNum 1 \
    --lr 0.0002 \
    --dist False \
    --modelName TM_Model_A \
    --dataset tmdataset \
    --is_stage_2 False \
    --log ./log_bench_E1/ \
    --outputDir ./result_bench_E1/ \
    --resultImgDir ./ResultImg_bench_E1/ \
    --modelDir "${MODEL_DIR}" 2>&1 | tee "${LOG_FILE}"

echo
echo "=== POST-RUN VERIFICATION ==="
echo "Expected: 2 saves at iters 16000 and 32000 (cumulative across epochs)"
echo "Bar 'total' field should show 16000 each epoch."
echo
grep -E '\[TrainProcess\]|Save model in' "${LOG_FILE}" || echo "(no save lines found — FAIL)"
ls -la "${MODEL_DIR}"
