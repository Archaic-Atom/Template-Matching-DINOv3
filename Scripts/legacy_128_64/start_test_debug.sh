#!/bin/bash
# Smoke-test launcher for --mode test on Stage-2 A variant.
#
# Reads Datasets/image_test_100.csv (100 random samples; subsample via
# `python -c "import random, ..."` from image_test.csv), runs single-GPU
# inference under Checkpoint_A_stage2/, writes per-rank tmp CSVs then
# rank-0 merges them into DebugResult/test_tmdataset_<timestamp>.csv.
# Evaluate with: python Source/Tools/eval_test_csv.py --csv <that file> --n 0
#
# 跑 --mode test 的 smoke 脚本：100 张随机测试样本，单 GPU，输出合并 CSV。
mkdir -p DebugResult log_test_debug ResultImg_test_debug

CUDA_VISIBLE_DEVICES=0 python -u Source/main.py \
    --mode test \
    --batchSize 8 \
    --gpu 1 \
    --port 8889 \
    --trainListPath ./Datasets/image_test_100.csv \
    --auto_save_num 1 \
    --dataloaderNum 2 \
    --imgNum 100 \
    --sampleNum 1 \
    --maxEpochs 1 \
    --lr 1e-3 \
    --dist False \
    --modelName TM_Model_A \
    --dataset tmdataset \
    --is_stage_2 True \
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
    --log ./log_test_debug/ \
    --outputDir ./DebugResult/ \
    --resultImgDir ./ResultImg_test_debug/ \
    --modelDir ./Checkpoint_A_stage2/
