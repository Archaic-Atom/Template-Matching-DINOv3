#!/usr/bin/env bash
# TM_Model_B Stage 2 training — ablation: NO Stage 1 alignment.
#
# B uses the same architecture as TM_Model_A but the SAR encoder is initialized
# at random and trained jointly with the matching head in Stage 2 (NOT frozen).
# This is the "no Stage 1 contrastive alignment" ablation — used to show how
# much Stage 1 contributes to final matching accuracy.
#
# config.yaml sets training.stage_2.freeze_sar_encoder: false, so the
# interface leaves sar_encoder.* parameters un-frozen.
#
# Prep: Checkpoint_B_stage2/ MUST be empty on first launch (no inheritance from
# any Stage 1 ckpt). The script aborts if it's non-empty and not a resume.

set -e

train_gpus_id=0,1,2,3
train_list_path='/data5/datasets/TM_shards/shard-{000000..000254}.tar'
model_dir='./Checkpoint_B_stage2/'
log_file='./start_train_B_stage2_wds.log'

# Force tqdm progress lines to render at full width in non-TTY logs.
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
echo "  shards:      ${train_list_path}"
echo "  modelDir:    ${model_dir}"
echo "  log file:    ${log_file}"

CUDA_VISIBLE_DEVICES=${train_gpus_id} python -u Source/main.py \
                        --mode train \
                        --batchSize 8 \
                        --gpu 4 \
                        --trainListPath "${train_list_path}" \
                        --auto_save_num 1 \
                        --dataloaderNum 8 \
                        --dataloaderType iterable \
                        --imgNum 8145177 \
                        --maxEpochs 30 \
                        --sampleNum 1 \
                        --lr 1e-3 \
                        --dist True \
                        --modelName TM_Model_B \
                        --dataset tmdataset \
                        --is_stage_2 True \
                        --log ./log_B_stage2/ \
                        --outputDir ./result_B_stage2/ \
                        --resultImgDir ./ResultImg_B_stage2/ \
                        --modelDir "${model_dir}" > "${log_file}" 2>&1 &

echo "started (PID $!); tailing log..."
tail -f "${log_file}"
