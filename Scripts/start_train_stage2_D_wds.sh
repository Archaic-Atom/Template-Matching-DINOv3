#!/usr/bin/env bash
# TM_Model_D Stage 2 — ablation: simple head + sub-pixel quadratic refinement.
#
# Same architecture and training as TM_Model_A. The only difference is at
# evaluation time: the fine heatmap's peak is refined via 2D quadratic fit
# over the 3×3 neighborhood around the integer argmax, giving sub-pixel
# offsets in [-0.5, 0.5].
#
# State_dict is byte-identical to TM_Model_A → D reuses A's trained Stage-2
# ckpt at eval time. This script bootstraps Checkpoint_D_stage2/ from
# Checkpoint_A_stage2/ and launches training with maxEpochs=10. If A is
# already at epoch 9, JackFramework will skip training and run val only
# (eval pass with D's soft_subpixel argmax → ablation row D's numbers).
#
# To rule out training variance for the paper, set src_stage2_dir to
# ./Checkpoint_v3/ instead — that bootstraps from Stage 1 and forces a
# fresh Stage 2 training run for D.
#
# D 变体：simple head + sub-pixel 评估精修。架构和训练与 A 完全相同；只在
# eval 时用 quadratic fit 精修到亚像素。默认复用 A 的 Stage-2 ckpt。

set -e
set -u

cd ~/Documents/Programs/Template-Matching-DINOv3

train_gpus_id=0,1,2,3
train_list_path='Datasets/shard_training_path.csv'
val_list_path='Datasets/shard_testing_path.csv'
src_stage2_dir='./Checkpoint_A_stage2/'
model_dir='./Checkpoint_D_stage2/'
log_file='./start_train_stage2_D_wds.log'

export COLUMNS=300

mkdir -p "${model_dir}"

# Bootstrap from A's Stage 2 ckpt (same SarResNet16GN + StandardMatchingHead
# → keys identical). If user wants D to retrain from scratch for variance
# control, change src_stage2_dir to ./Checkpoint_v3/ above.
ckpt_count=$(ls "${model_dir}"/model_epoch_*.pth 2>/dev/null | wc -l)
if [ "${ckpt_count}" -eq 0 ]; then
    latest=$(ls -t "${src_stage2_dir}"/model_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "${latest}" ]; then
        echo "ERROR: ${model_dir} empty AND ${src_stage2_dir} has no ckpt."
        echo "       Run start_train_stage2_A_wds.sh first to produce A's Stage 2 ckpt."
        exit 1
    fi
    echo "  bootstrapping ${model_dir} from ${latest}"
    cp "${latest}" "${model_dir}/"
    [ -f "${src_stage2_dir}/checkpoint.list" ] && cp "${src_stage2_dir}/checkpoint.list" "${model_dir}/"
fi

echo "[TM_Model_D] Stage 2 — soft argmax + sub-pixel quadratic (eval-only on A's ckpt)"
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
    --modelName TM_Model_D \
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
    --log ./log_D_stage2/ \
    --outputDir ./result_D_stage2/ \
    --resultImgDir ./ResultImg_D_stage2/ \
    --modelDir ${model_dir} > ${log_file} 2>&1" >/dev/null 2>&1 &

PID=$!
disown $PID 2>/dev/null
echo "started PID=${PID}; kill ${PID} to abort"
echo "log: ${log_file}"
