#!/usr/bin/env bash
# TM_Model_C Stage 2 training — main proposed method: 3-level cascade fine head.
#
# C reuses TM_Model_A's Stage-1-trained SAR encoder ckpt (same SarResNet16GN
# class → state_dict keys are identical, sar_encoder.* loads cleanly under
# strict=False). The cascade fine head is initialized fresh.
#
# Eval-time argmax_type is HARD in C's config.yaml — C isolates the cascade
# architecture contribution alone. TM_Model_E shares C's trained ckpt and
# eval's with soft sub-pixel argmax (combined contribution row).
#
# Prep: Checkpoint_C_stage2/ must contain a Stage-1 ckpt. Auto-bootstrapped
# from ./Checkpoint_v3/ if empty.
#
# C 变体训练：3-level cascade fine head 主方法。从 Stage 1 v3 ckpt 引导。

set -e
set -u

cd ~/Documents/Programs/Template-Matching-DINOv3

train_gpus_id=0,1,2,3
train_list_path='Datasets/shard_training_path.csv'
val_list_path='Datasets/shard_testing_path.csv'
src_stage1_dir='./Checkpoint_v3/'
model_dir='./Checkpoint_C_stage2/'
log_file='./start_train_stage2_C_wds.log'

export COLUMNS=200

mkdir -p "${model_dir}"

# Auto-bootstrap from latest Stage-1 ckpt if Stage-2 modelDir is empty.
# Stage 1 uses SGD, Stage 2 uses AdamW → opt state is incompatible.
# strip_stage1_opt.py drops opt_* and resets epoch=-1 so JackFramework
# treats this as a fresh Stage-2 start.
ckpt_count=$(ls "${model_dir}"/model_epoch_*.pth 2>/dev/null | wc -l)
if [ "${ckpt_count}" -eq 0 ]; then
    latest=$(ls -t "${src_stage1_dir}"/model_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "${latest}" ]; then
        echo "ERROR: ${model_dir} empty AND ${src_stage1_dir} has no Stage-1 ckpt."
        exit 1
    fi
    init_name="model_stage1_init.pth"
    echo "  bootstrapping ${model_dir}/${init_name} from ${latest} (strip opt + reset epoch)"
    python Source/Tools/strip_stage1_opt.py "${latest}" "${model_dir}/${init_name}"
    {
        echo "last model name:${init_name}"
        echo "${init_name}"
    } > "${model_dir}/checkpoint.list"
fi

echo "[TM_Model_C] Stage 2 — 3-level cascade fine head (cascade-only ablation row)"
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
    --modelName TM_Model_C \
    --dataset tmdataset \
    --is_stage_2 True \
    --shuffle_buffer 100000 \
    --log ./log_C_stage2/ \
    --outputDir ./result_C_stage2/ \
    --resultImgDir ./ResultImg_C_stage2/ \
    --modelDir ${model_dir} > ${log_file} 2>&1" >/dev/null 2>&1 &

PID=$!
disown $PID 2>/dev/null
echo "started PID=${PID}; kill ${PID} to abort"
echo "log: ${log_file}"
