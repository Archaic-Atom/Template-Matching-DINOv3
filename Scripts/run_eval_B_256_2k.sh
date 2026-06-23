#!/usr/bin/env bash
# Zero-shot eval of model B (trained at 128/64) on the opt256/sar192 2k set.
# B 在 256/192 上的零样本评测（B 用 128/64 训练）。nohup 抗断线。
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
mkdir -p DebugResult256 log_test_debug ResultImg_test_debug
export TM_SEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "==== EVAL B@256/192 START $(date) ===="
CUDA_VISIBLE_DEVICES=0 $PY -u Source/main.py \
    --mode test --batchSize 8 --gpu 1 --port 8899 \
    --trainListPath /data4/datasets/eval2k_opt256_sar192/image_test_2k_256.csv \
    --auto_save_num 1 --dataloaderNum 2 \
    --imgNum 2000 --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
    --dist False --modelName TM_Model_B --dataset tmdataset \
    --is_stage_2 True \
    --fine_dim 32 --fine_margin 4 --sar_size 192 \
    --log ./log_test_debug/ --outputDir ./DebugResult256/ \
    --resultImgDir ./ResultImg_test_debug/ \
    --modelDir ./Checkpoint_B_stage2/
echo "==== EVAL DONE rc=$? $(date) ===="
