#!/usr/bin/env bash
# TM_Model_A Stage 2 training — baseline matching head (simple, hard argmax).
#
# A is the ablation table's baseline row. Architecture: SAR encoder
# (Stage-1-aligned) + frozen DINOv3 + two-stage coarse heads (stride-4 +
# stride-8) + a single stride-1 fine head, eval with hard argmax.
#
# Prep: bootstraps ./Checkpoint_A_stage2/ from the latest Stage-1 ckpt in
# ./Checkpoint_v3/ if the dir is empty. JackFramework's ModelSaver auto-
# loads the latest ckpt from --modelDir; load_model() uses strict=False so
# the SAR encoder weights are read in and the new matching head is randomly
# initialized.
#
# nohup + disown so the run survives SSH disconnect. Prints PID + log path.
#
# A 变体训练：baseline matching head。从 Stage-1 ckpt 引导，启动 Stage 2 训练。

set -e
set -u

cd ~/Documents/Programs/Template-Matching-DINOv3

train_gpus_id=0,1,2,3
train_list_path='Datasets/shard_training_path.csv'
val_list_path='Datasets/shard_testing_path.csv'
src_stage1_dir='./Checkpoint_v3/'
model_dir='./Checkpoint_A_stage2/'
log_file='./start_train_stage2_A_wds.log'

# Force tqdm progress lines to render at full width in non-TTY logs.
export COLUMNS=300

mkdir -p "${model_dir}"

# Auto-bootstrap from latest Stage-1 ckpt if Stage-2 modelDir is empty.
# Stage 1 uses SGD optimizer, Stage 2 uses AdamW → opt state is incompatible.
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
    # checkpoint.list format JackFramework expects:
    #   line 1: "last model name:<filename>"
    #   line 2+: history filenames
    {
        echo "last model name:${init_name}"
        echo "${init_name}"
    } > "${model_dir}/checkpoint.list"
fi

echo "[TM_Model_A] Stage 2 — baseline matching head"
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
    --maxEpochs 6 \
    --sampleNum 1 \
    --lr 1e-3 \
    --dist True \
    --modelName TM_Model_A \
    --dataset tmdataset \
    --is_stage_2 True \
    --shuffle_buffer 100000 \
    --matching_dim 128 \
    --match_w4 1.0 \
    --match_w8 0.5 \
    --match_sigma4 1.0 \
    --match_sigma8 1.0 \
    --match_primary_stride 4 \
    --fine_dim 32 \
    --fine_margin 4 \
    --sar_size 64 \
    --match_w_fine 1.0 \
    --match_sigma_fine 1.0 \
    --fine_warmup_epochs 3 \
    --log ./log_A_stage2/ \
    --outputDir ./result_A_stage2/ \
    --resultImgDir ./ResultImg_A_stage2/ \
    --modelDir ${model_dir} > ${log_file} 2>&1" >/dev/null 2>&1 &

PID=$!
disown $PID 2>/dev/null
echo "started PID=${PID}; kill ${PID} to abort"
echo "log: ${log_file}"
