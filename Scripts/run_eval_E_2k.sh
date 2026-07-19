#!/usr/bin/env bash
# Reproducible 2k eval of TM_Model_E (cascade + SUB-PIXEL) on the opt256/sar192 test set.
# TM_Model_E = TM_Model_C 的级联架构 + eval 时 soft_subpixel(2D 二次拟合精修),
# 与 C 共用 byte-identical 权重(直接复用 C 的 ckpt,无需重训)。本脚本把同一个
# C 的 ckpt 钉进 eval_e_<tag>/ 并以 --modelName TM_Model_E 跑 --mode test,唯一行为
# 差异来自 E 配置里的 inference.argmax_type: soft_subpixel。
#
# Pins ONE C-cascade checkpoint into a clean eval dir, runs --mode test at bs1
# under TM_Model_E (sub-pixel). Small enough to share GPU headroom (peaks ~2.5 GB).
#
# Usage: bash Scripts/run_eval_E_2k.sh <src_ckpt.pth> <eval_tag> [gpu]
#   <src_ckpt.pth>  e.g. ./Checkpoint_C_direct_sub40_256/model_epoch_4.pth
#   <eval_tag>      e.g. e4   -> dirs eval_e_e4/ , DebugResult_e_e4/
#   [gpu]           CUDA device index (default 0)
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
MANIFEST=/data4/datasets/eval2k_opt256_sar192/image_test_2k_256.csv
cd "$PROJ" || exit 1

SRC="${1:?usage: $0 <src_ckpt.pth> <eval_tag> [gpu]}"
TAG="${2:?usage: $0 <src_ckpt.pth> <eval_tag> [gpu]}"
GPU="${3:-0}"

EVAL_DIR="eval_e_${TAG}"
OUT_DIR="DebugResult_e_${TAG}"
BASENAME="$(basename "$SRC")"

# --- pin the checkpoint into a clean dir + write checkpoint.list -------------
# 把 C 的 ckpt 钉进干净目录并写好 checkpoint.list。
mkdir -p "$EVAL_DIR" "$OUT_DIR" log_test_debug ResultImg_test_debug
cp -f "$SRC" "$EVAL_DIR/$BASENAME"
printf 'last model name:%s\n%s\n' "$BASENAME" "$BASENAME" > "$EVAL_DIR/checkpoint.list"

export TM_SEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "==== EVAL E $TAG START $(date) (gpu $GPU, bs1, sub-pixel) ===="
CUDA_VISIBLE_DEVICES="$GPU" $PY -u Source/main.py \
    --mode test --batchSize 1 --gpu 1 --port 8894 \
    --trainListPath "$MANIFEST" \
    --auto_save_num 1 --dataloaderNum 2 \
    --imgNum 2000 --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
    --dist False --modelName TM_Model_E --dataset tmdataset \
    --is_stage_2 True \
    --fine_dim 32 --fine_margin 4 --sar_size 192 \
    --log ./log_test_debug/ --outputDir "./$OUT_DIR/" \
    --resultImgDir ./ResultImg_test_debug/ \
    --modelDir "./$EVAL_DIR/"
echo "==== EVAL E $TAG DONE rc=$? $(date) ===="
