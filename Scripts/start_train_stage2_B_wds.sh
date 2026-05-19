#!/usr/bin/env bash
# TM_Model_B Stage 2 training — ablation: NO Stage 1 alignment.
#
# B uses the same architecture as TM_Model_A but the SAR encoder is initialized
# at random and trained jointly with the matching head in Stage 2 (NOT frozen).
# Used to show how much Stage 1 contrastive alignment contributes to final
# matching accuracy.
#
# config.yaml sets training.stage_2.freeze_sar_encoder: false, so the
# interface leaves sar_encoder.* parameters un-frozen.
#
# Prep: Checkpoint_B_stage2/ MUST be empty on first launch (no Stage 1
# inheritance). Aborts non-resume runs with leftover ckpts.
#
# B 变体训练：无 Stage 1 alignment 消融。SAR encoder 随机初始化、Stage 2
# 与 matching head 联合训练。modelDir 必须为空（或仅含 B 自身的 resume ckpt）。

set -e
set -u

cd ~/Documents/Programs/Template-Matching-DINOv3

train_gpus_id=0,1,2,3
train_list_path='Datasets/shard_training_path.csv'
val_list_path='Datasets/shard_testing_path.csv'
model_dir='./Checkpoint_B_stage2/'
log_file='./start_train_stage2_B_wds.log'

export COLUMNS=200

mkdir -p "${model_dir}"

# Sanity check: B starts from scratch. If user wants to resume an in-progress B
# run, they must have a model_epoch_*.pth from a previous B launch in this dir.
# If the dir has files from some OTHER variant (e.g. A's Stage 1 ckpt copied
# in by mistake), abort.
ckpt_count=$(ls "${model_dir}"/model_epoch_*.pth 2>/dev/null | wc -l)
if [ "${ckpt_count}" -gt 0 ]; then
    echo "WARNING: ${model_dir} contains ${ckpt_count} checkpoint(s)."
    echo "         If this is a B-run resume, OK. If you copied A's Stage-1"
    echo "         ckpt in here by mistake, that defeats the ablation."
    echo "         Press Ctrl-C in 5s to abort, otherwise we will resume."
    sleep 5
fi

echo "[TM_Model_B] Stage 2 — no Stage-1 alignment ablation"
echo "  train gpus:  ${train_gpus_id}"
echo "  train list:  ${train_list_path}"
echo "  val list:    ${val_list_path}"
echo "  modelDir:    ${model_dir}"
echo "  log file:    ${log_file}"

nohup bash -c "CUDA_VISIBLE_DEVICES=${train_gpus_id} python -u Source/main.py \
    --mode train \
    --batchSize 8 \
    --gpu 4 \
    --port 8887 \
    --trainListPath '${train_list_path}' \
    --valListPath '${val_list_path}' \
    --auto_save_num 1 \
    --dataloaderNum 8 \
    --dataloaderType iterable \
    --imgNum 65275274 \
    --valImgNum 8145176 \
    --maxEpochs 4 \
    --sampleNum 1 \
    --lr 1e-3 \
    --dist True \
    --modelName TM_Model_B \
    --dataset tmdataset \
    --is_stage_2 True \
    --shuffle_buffer 100000 \
    --log ./log_B_stage2/ \
    --outputDir ./result_B_stage2/ \
    --resultImgDir ./ResultImg_B_stage2/ \
    --modelDir ${model_dir} > ${log_file} 2>&1" >/dev/null 2>&1 &

PID=$!
disown $PID 2>/dev/null
echo "started PID=${PID}; kill ${PID} to abort"
echo "log: ${log_file}"
