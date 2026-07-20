#!/bin/bash
# Reproducible 2k test on Datasets/image_test_2k.csv.
# 可复现的 2k 测试：固定随机种子，方便不同 ckpt / 变体之间对比。
#
# The test data is already deterministic (fixed images + CSV labels, no aug in
# Stage-2 eval); TM_SEED additionally pins the RNGs so the model forward pass is
# reproducible run-to-run.
# 测试数据本身已确定（固定图片 + CSV 标签，Stage-2 评测无增强）；TM_SEED 再固定
# 随机源，让模型前向逐次可复现。
#
# Usage: bash Scripts/start_test_2k.sh <model_name> <model_dir> [gpu] [seed]
#   <model_name>  registry key, e.g. TM_Model_A / TM_Model_B / TM_Model_C
#   <model_dir>   checkpoint folder,  e.g. ./Checkpoint_B_stage2/
#   [gpu]         CUDA device index   (default 0)
#   [seed]        fixed RNG seed      (default 2024)
#
# Example: bash Scripts/start_test_2k.sh TM_Model_B ./Checkpoint_B_stage2/ 0 2024
set -e

MODEL_NAME="${1:?usage: $0 <model_name> <model_dir> [gpu] [seed]}"
MODEL_DIR="${2:?usage: $0 <model_name> <model_dir> [gpu] [seed]}"
GPU="${3:-0}"
SEED="${4:-2024}"

mkdir -p DebugResult log_test_debug ResultImg_test_debug

# Fixed seed -> identical inputs + deterministic inference across runs.
# 固定种子 -> 每次输入一致、推理确定，结果可对比。
export TM_SEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CUDA_VISIBLE_DEVICES="$GPU" python -u Source/main.py \
    --mode test --batchSize 8 --gpu 1 --port 8889 \
    --trainListPath ./Datasets/image_test_2k.csv \
    --auto_save_num 1 --dataloaderNum 2 \
    --imgNum 2000 --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
    --dist False --modelName "$MODEL_NAME" --dataset tmdataset \
    --is_stage_2 True \
    --fine_dim 32 --fine_margin 4 --sar_size 64 \
    --log ./log_test_debug/ --outputDir ./DebugResult/ \
    --resultImgDir ./ResultImg_test_debug/ \
    --modelDir "$MODEL_DIR"
