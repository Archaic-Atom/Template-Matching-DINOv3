#!/usr/bin/env bash
# Zero-shot eval of B (128/64-trained) on the 10k opt256/sar192 set, batch 8.
# 需先把 TM_Model_B/config.yaml 的 sar_size 临时改成 192（builder 从 yaml 读）。
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
mkdir -p DebugResult256_10k log_test_debug ResultImg_test_debug
export TM_SEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "==== EVAL B@256/192 10k START $(date) ===="
CUDA_VISIBLE_DEVICES=0 $PY -u Source/main.py \
    --mode test --batchSize 8 --gpu 1 --port 8907 \
    --trainListPath /data4/datasets/eval10k_opt256_sar192/image_test_2k_256.csv \
    --auto_save_num 1 --dataloaderNum 2 \
    --imgNum 10000 --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
    --dist False --modelName TM_Model_B --dataset tmdataset \
    --is_stage_2 True \
    --fine_dim 32 --fine_margin 4 --sar_size 192 \
    --log ./log_test_debug/ --outputDir ./DebugResult256_10k/ \
    --resultImgDir ./ResultImg_test_debug/ \
    --modelDir ./Checkpoint_B_stage2/
echo "==== EVAL 10k DONE rc=$? $(date) ===="
