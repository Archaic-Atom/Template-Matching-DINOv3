#!/usr/bin/env bash
# Stage 1 (feature alignment) @ 256/192 geometry — WebDataset, 4-GPU DDP.
# 256/192 几何的 Stage-1 对齐训练(4 卡 DDP)。
#
# 从头训(全新 modelDir Checkpoint_A_stage1_256)。normalize_logits=True
# (A config),lr 5e-3,--sar_size 192 让 dataloader 把 RGB 裁成 192 模板
# (item-5 解耦)。DINO 冻结 → DDP 需要 find_unused_parameters=True(JF 远端 patch)。
#
# ⚠️ 关键约定(2026-06-06 踩坑):imgNum 必须 == manifest 的 real_total
# (sum(file_num))。框架用 imgNum 算进度条分母 training_iteration=imgNum/(bs*gpu),
# 但真正的 epoch 长度由 dataloader 的 with_epoch(=real_total/(bs*ws)) 决定。两者
# 不一致时进度条会在真 epoch 的 8% 处就显示 100%/Done,看起来像"卡在存盘"其实没存。
# 老的可用脚本(stage1_v3 / stage2_B)都满足 imgNum==real_total。
# IMPORTANT: imgNum MUST equal the manifest real_total or the progress bar lies.
#
# 本次为"先跑 10 代验证 lr/参数"的子集测试:用 stride-10 的 42-shard 子集
# (real_total=1,344,000,epoch≈10,500 iter≈~1.5h@4GPU,每代存一个 ckpt)。
# 参数没问题后再切回全量 manifest(shard_path.csv, imgNum=13182912)。
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1

train_gpus_id=0,1,2,3
# 42-shard 子集(stride-10,横跨全 416 shard);real_total=1,344,000。
train_list_path='./Datasets/shard_train_256_subset40.csv'
# ⚠️ 进度条列宽:必须用 JF_PROGRESS_COLUMNS(JF process_bar.py:272 直接读它),
# NOT `export COLUMNS` —— COLUMNS 是 shell 特殊变量,bash/zsh 会为子进程重置,
# nohup 下传不进 python,所以 a0 被截断只剩 l0/l1/l2。老的 stage2_B 脚本就是
# 用 JF_PROGRESS_COLUMNS=300 内联设置才显示全。这里设 400 给 a0/lr 留足宽度。
# Use JF_PROGRESS_COLUMNS (JF reads it directly); COLUMNS is shell-special & lost
# under nohup → a0 truncated. 400 wide so a0 (cos-align acc) shows in the bar.
export JF_PROGRESS_COLUMNS=400
mkdir -p log_stage1_256 result_stage1_256 ResultImg_stage1_256 Checkpoint_A_stage1_256

echo "Stage 1 @256/192 — gpus ${train_gpus_id}, shards ${train_list_path}"
CUDA_VISIBLE_DEVICES=${train_gpus_id} JF_PROGRESS_COLUMNS=400 nohup "$PY" -u Source/main.py \
    --mode train \
    --batchSize 32 \
    --gpu 4 \
    --port 8814 \
    --trainListPath "${train_list_path}" \
    --auto_save_num 1 \
    --dataloaderNum 8 \
    --dataloaderType iterable \
    --imgNum 1344000 \
    --maxEpochs 10 \
    --sampleNum 1 \
    --lr 0.005 \
    --dist True \
    --modelName TM_Model_A \
    --dataset tmdataset \
    --is_stage_2 False \
    --sar_size 192 \
    --log ./log_stage1_256/ \
    --outputDir ./result_stage1_256/ \
    --resultImgDir ./ResultImg_stage1_256/ \
    --modelDir ./Checkpoint_A_stage1_256/ > ./train_stage1_256_A.log 2>&1 &
PID=$!
disown "$PID" 2>/dev/null
echo "STAGE1_256_PID=$PID  log=./train_stage1_256_A.log"
