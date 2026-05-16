#!/usr/bin/env bash
# TM_Model_C Stage 2 training — main proposed method: 3-level cascade fine head.
#
# C reuses TM_Model_A's Stage-1-trained SAR encoder ckpt (same SarResNet16GN
# class → state_dict keys are identical, sar_encoder.* loads cleanly under
# strict=False). The cascade fine head is initialized fresh.
#
# Prep: Checkpoint_C_stage2/ must contain a Stage-1 ckpt (typically copied
# from TM_Model_A's ./Checkpoint/model_epoch_N.pth). The block below does
# that copy automatically if the dir is empty.
#
# PHASE-1 NOTE: CascadeHead.forward and cascade_loss are stubs that raise
# NotImplementedError. Running this script before Phase 2 will fail loudly
# at the first forward pass — by design.

set -e

train_gpus_id=0,1,2,3
train_list_path='/data5/datasets/TM_shards/shard-{000000..000254}.tar'
src_stage1_dir='./Checkpoint/'
model_dir='./Checkpoint_C_stage2/'
log_file='./start_train_C_stage2_wds.log'

export COLUMNS=200

mkdir -p "${model_dir}"

# Auto-copy A's best Stage-1 ckpt into C's modelDir if empty.
ckpt_count=$(ls "${model_dir}"/model_epoch_*.pth 2>/dev/null | wc -l)
if [ "${ckpt_count}" -eq 0 ]; then
    latest=$(ls -t "${src_stage1_dir}"/model_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "${latest}" ]; then
        echo "ERROR: ${model_dir} is empty AND ${src_stage1_dir} has no Stage-1 ckpt."
        echo "       Place a Stage-1 model_epoch_N.pth in either dir before launching."
        exit 1
    fi
    echo "  bootstrapping ${model_dir} from ${latest}"
    cp "${latest}" "${model_dir}/"
    # Also copy the framework's checkpoint.list so JackFramework knows what to load.
    [ -f "${src_stage1_dir}/checkpoint.list" ] && cp "${src_stage1_dir}/checkpoint.list" "${model_dir}/"
fi

echo "[TM_Model_C] Stage 2 — 3-level cascade fine head (Phase 2 implementation pending)"
echo "  train gpus:  ${train_gpus_id}"
echo "  shards:      ${train_list_path}"
echo "  modelDir:    ${model_dir}"
echo "  log file:    ${log_file}"
echo
echo "  WARNING: CascadeHead.forward is a Phase-1 stub. Training WILL fail on"
echo "  the first forward pass until Phase 2 fills in the cascade implementation."

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
                        --modelName TM_Model_C \
                        --dataset tmdataset \
                        --is_stage_2 True \
                        --log ./log_C_stage2/ \
                        --outputDir ./result_C_stage2/ \
                        --resultImgDir ./ResultImg_C_stage2/ \
                        --modelDir "${model_dir}" > "${log_file}" 2>&1 &

echo "started (PID $!); tailing log..."
tail -f "${log_file}"
