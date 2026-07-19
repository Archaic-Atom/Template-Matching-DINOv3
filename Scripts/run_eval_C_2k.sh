#!/usr/bin/env bash
# Reproducible 2k eval of TM_Model_C (cascade) on the opt256/sar192 test set.
# TM_Model_C 级联模型在 256/192 2k 测试集上的可复现评测。
#
# Pins ONE epoch checkpoint into a clean eval dir (so JF's checkpoint.list loads
# exactly that file), then runs --mode test at batch size 1 — small enough to
# share a training GPU's headroom (~3.9 GB free; C eval peaks ~2.5 GB).
# 把指定 epoch 的 ckpt 钉进干净目录（让 JF 的 checkpoint.list 精确加载它），
# 然后用 bs1 跑 --mode test —— 小到可挤进训练 GPU 的余量（约 3.9 GB 空闲）。
#
# Usage: bash Scripts/run_eval_C_2k.sh <src_ckpt.pth> <eval_tag> [gpu]
#   <src_ckpt.pth>  e.g. ./Checkpoint_C_direct_sub40_256/model_epoch_1.pth
#   <eval_tag>      e.g. e1   -> dirs eval_c_e1/ , DebugResult_c_e1/
#   [gpu]           CUDA device index (default 0)
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
MANIFEST=/data4/datasets/eval2k_opt256_sar192/image_test_2k_256.csv
cd "$PROJ" || exit 1

SRC="${1:?usage: $0 <src_ckpt.pth> <eval_tag> [gpu]}"
TAG="${2:?usage: $0 <src_ckpt.pth> <eval_tag> [gpu]}"
GPU="${3:-0}"

EVAL_DIR="eval_c_${TAG}"
OUT_DIR="DebugResult_c_${TAG}"
BASENAME="$(basename "$SRC")"

# --- pin the checkpoint into a clean dir + write checkpoint.list -------------
# 把 ckpt 钉进干净目录并写好 checkpoint.list。
mkdir -p "$EVAL_DIR" "$OUT_DIR" log_test_debug ResultImg_test_debug
cp -f "$SRC" "$EVAL_DIR/$BASENAME"
printf 'last model name:%s\n%s\n' "$BASENAME" "$BASENAME" > "$EVAL_DIR/checkpoint.list"

export TM_SEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "==== EVAL C $TAG START $(date) (gpu $GPU, bs1) ===="
CUDA_VISIBLE_DEVICES="$GPU" $PY -u Source/main.py \
    --mode test --batchSize 1 --gpu 1 --port 8893 \
    --trainListPath "$MANIFEST" \
    --auto_save_num 1 --dataloaderNum 2 \
    --imgNum 2000 --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
    --dist False --modelName TM_Model_C --dataset tmdataset \
    --is_stage_2 True \
    --fine_dim 32 --fine_margin 4 --sar_size 192 \
    --log ./log_test_debug/ --outputDir "./$OUT_DIR/" \
    --resultImgDir ./ResultImg_test_debug/ \
    --modelDir "./$EVAL_DIR/"
echo "==== EVAL C $TAG DONE rc=$? $(date) ===="
