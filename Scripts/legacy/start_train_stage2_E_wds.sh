#!/usr/bin/env bash
# TM_Model_E Stage 2 — full method: cascade fine head + sub-pixel refinement.
#
# Same architecture and training as TM_Model_C (3-level cascade). Only
# differs at eval time: soft sub-pixel quadratic refinement on the fine
# stride-1 heatmap. State_dict is byte-identical to C, so E reuses C's
# trained Stage-2 ckpt — no separate training run needed.
#
# This script bootstraps Checkpoint_E_stage2/ from Checkpoint_C_stage2/
# and launches with maxEpochs=10. If C is already at epoch 9, JackFramework
# will skip training and run val only (eval with E's soft_subpixel argmax
# → ablation row E "full method" numbers).
#
# To rule out training variance for the paper, set src_stage2_dir to
# ./Checkpoint_v3/ — that bootstraps from Stage 1 and re-trains Stage 2
# for E from scratch (4 days of additional GPU).
#
# E 变体：full method（cascade + sub-pixel）。复用 C 的 Stage 2 ckpt，
# 只在 eval 时把 argmax 切到 soft_subpixel。Paper 中 "Our final" 那一行。

set -e
set -u

cd ~/Documents/Programs/Template-Matching-DINOv3

train_gpus_id=0,1,2,3
train_list_path='Datasets/shard_training_path.csv'
val_list_path='Datasets/shard_testing_path.csv'
src_stage2_dir='./Checkpoint_C_stage2/'
model_dir='./Checkpoint_E_stage2/'
log_file='./start_train_stage2_E_wds.log'

export COLUMNS=300

mkdir -p "${model_dir}"

# Bootstrap from C's Stage 2 ckpt (same cascade head + SAR encoder → keys
# identical). If user wants E to retrain from scratch for variance control,
# change src_stage2_dir to ./Checkpoint_v3/ above.
ckpt_count=$(ls "${model_dir}"/model_epoch_*.pth 2>/dev/null | wc -l)
if [ "${ckpt_count}" -eq 0 ]; then
    latest=$(ls -t "${src_stage2_dir}"/model_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "${latest}" ]; then
        echo "ERROR: ${model_dir} empty AND ${src_stage2_dir} has no ckpt."
        echo "       Run start_train_stage2_C_wds.sh first to produce C's Stage 2 ckpt."
        exit 1
    fi
    echo "  bootstrapping ${model_dir} from ${latest}"
    cp "${latest}" "${model_dir}/"
    [ -f "${src_stage2_dir}/checkpoint.list" ] && cp "${src_stage2_dir}/checkpoint.list" "${model_dir}/"
fi

echo "[TM_Model_E] Stage 2 — cascade + sub-pixel (full method, eval-only on C's ckpt)"
echo "  train gpus:  ${train_gpus_id}"
echo "  train list:  ${train_list_path}"
echo "  val list:    ${val_list_path}"
echo "  modelDir:    ${model_dir}"
echo "  log file:    ${log_file}"

nohup bash -c "JF_PROGRESS_COLUMNS=300 CUDA_VISIBLE_DEVICES=${train_gpus_id} python -u Source/main.py \
    --mode train \
    --batchSize 32 \
    --gpu 4 \
    --port 8887 \
    --trainListPath '${train_list_path}' \
    --valListPath '${val_list_path}' \
    --auto_save_num 1 \
    --dataloaderNum 8 \
    --dataloaderType iterable \
    --imgNum 65275274 \
    --valImgNum 8145176 \
    --maxEpochs 10 \
    --sampleNum 1 \
    --lr 1e-3 \
    --dist True \
    --modelName TM_Model_E \
    --dataset tmdataset \
    --is_stage_2 True \
    --shuffle_buffer 100000 \
    --log ./log_E_stage2/ \
    --outputDir ./result_E_stage2/ \
    --resultImgDir ./ResultImg_E_stage2/ \
    --modelDir ${model_dir} > ${log_file} 2>&1" >/dev/null 2>&1 &

PID=$!
disown $PID 2>/dev/null
echo "started PID=${PID}; kill ${PID} to abort"
echo "log: ${log_file}"
