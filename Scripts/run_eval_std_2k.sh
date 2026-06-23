#!/usr/bin/env bash
# Generic reproducible 2k eval for StandardMatchingHead variants (A / B / D) on
# the opt256/sar192 test set. A=hard, D=A's weights + sub-pixel (eval-only).
# 通用 2k 评测脚本(简单匹配头变体 A/B/D)。A=hard,D=复用 A 权重 + sub-pixel。
#
# Pins ONE checkpoint into a clean eval dir (so JF's checkpoint.list loads exactly
# that file), then runs --mode test at bs1.
#
# Usage: bash Scripts/run_eval_std_2k.sh <modelName> <src_ckpt.pth> <eval_tag> [gpu] [port]
#   <modelName>     TM_Model_A | TM_Model_B | TM_Model_D
#   <src_ckpt.pth>  e.g. ./Checkpoint_A_stage2_ep4/model_epoch_4.pth
#   <eval_tag>      e.g. a_e4   -> dirs eval_a_e4/ , DebugResult_a_e4/
#   [gpu]           CUDA device index (default 0)
#   [port]          DDP rendezvous port (default 8895; vary across concurrent runs)
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
MANIFEST=/data4/datasets/eval2k_opt256_sar192/image_test_2k_256.csv
cd "$PROJ" || exit 1

MODEL="${1:?usage: $0 <modelName> <src_ckpt.pth> <eval_tag> [gpu] [port]}"
SRC="${2:?usage: $0 <modelName> <src_ckpt.pth> <eval_tag> [gpu] [port]}"
TAG="${3:?usage: $0 <modelName> <src_ckpt.pth> <eval_tag> [gpu] [port]}"
GPU="${4:-0}"
PORT="${5:-8895}"

EVAL_DIR="eval_${TAG}"
OUT_DIR="DebugResult_${TAG}"
BASENAME="$(basename "$SRC")"

mkdir -p "$EVAL_DIR" "$OUT_DIR" log_test_debug ResultImg_test_debug
cp -f "$SRC" "$EVAL_DIR/$BASENAME"
printf 'last model name:%s\n%s\n' "$BASENAME" "$BASENAME" > "$EVAL_DIR/checkpoint.list"

export TM_SEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "==== EVAL $MODEL $TAG START $(date) (gpu $GPU, bs1) ===="
CUDA_VISIBLE_DEVICES="$GPU" $PY -u Source/main.py \
    --mode test --batchSize 1 --gpu 1 --port "$PORT" \
    --trainListPath "$MANIFEST" \
    --auto_save_num 1 --dataloaderNum 2 \
    --imgNum 2000 --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
    --dist False --modelName "$MODEL" --dataset tmdataset \
    --is_stage_2 True \
    --fine_dim 32 --fine_margin 4 --sar_size 192 \
    --log ./log_test_debug/ --outputDir "./$OUT_DIR/" \
    --resultImgDir ./ResultImg_test_debug/ \
    --modelDir "./$EVAL_DIR/"
echo "==== EVAL $MODEL $TAG DONE rc=$? $(date) ===="
