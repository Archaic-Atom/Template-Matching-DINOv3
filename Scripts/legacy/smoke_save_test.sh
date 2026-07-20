#!/usr/bin/env bash
# Smoke test: does Stage-1 actually WRITE a checkpoint at epoch end?
# 冒烟测试:Stage-1 在 epoch 末到底会不会写 checkpoint?
#
# Purpose: refute the earlier "epoch-end save hangs" diagnosis. We point a
# tiny 1-shard manifest (declares only 2000 samples) at the pipeline so one
# framework epoch = real_total/(bs*ws) = 2000/8 = 250 iters (~2 min on 1 GPU),
# then check Checkpoint_smoke_save/ for a .pth. If it appears, the save
# mechanism is fine and the only real issue was epoch size (csv real_total
# drives with_epoch, not imgNum).
# 目的:推翻之前"epoch 末存盘 hang"的判断。用只声明 2000 样本的单分片清单,
# 让一个 epoch = 250 iters(单卡 ~2 分钟),然后看 Checkpoint_smoke_save/ 是否
# 出现 .pth。出现即证明存盘机制正常,真正问题只是 epoch 大小(csv 的 real_total
# 驱动 with_epoch,而非 imgNum)。
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1

rm -rf Checkpoint_smoke_save
mkdir -p Checkpoint_smoke_save log_smoke result_smoke ResultImg_smoke

CUDA_VISIBLE_DEVICES=0 nohup "$PY" -u Source/main.py \
    --mode train \
    --batchSize 8 \
    --gpu 1 \
    --dist False \
    --trainListPath ./Datasets/smoke_save_test.csv \
    --auto_save_num 1 \
    --dataloaderNum 2 \
    --dataloaderType iterable \
    --imgNum 2000 \
    --maxEpochs 2 \
    --sampleNum 1 \
    --lr 0.005 \
    --modelName TM_Model_A \
    --dataset tmdataset \
    --is_stage_2 False \
    --sar_size 192 \
    --log ./log_smoke/ \
    --outputDir ./result_smoke/ \
    --resultImgDir ./ResultImg_smoke/ \
    --modelDir ./Checkpoint_smoke_save/ > ./smoke_save.log 2>&1 &
PID=$!
disown "$PID" 2>/dev/null
echo "SMOKE_PID=$PID log=./smoke_save.log"
