#!/bin/bash
# Stage 1 (feature alignment) v3 - 4 GPU DDP on real train + test as per-epoch val.
#
# Train: Datasets/shard_training_path.csv -> 2040 shards, 65,275,274 samples
# Val:   Datasets/shard_testing_path.csv  -> 255 shards,   8,145,176 samples
#
# CSV manifests (path,file_num) are the source of truth for shard sample
# counts — no disk probing on launch, instant startup.
#
# nohup + & + disown so the run survives SSH disconnect. PID is printed
# so you can `kill $PID` to abort. Tail the log to watch progress:
#   tail -f start_train_stage1_v3.log

set -u
cd ~/Documents/Programs/Template-Matching-DINOv3

train_gpus_id=0,1,2,3
train_list_path='Datasets/shard_training_path.csv'
val_list_path='Datasets/shard_testing_path.csv'

export COLUMNS=200
export CUDA_VISIBLE_DEVICES=${train_gpus_id}

echo "Stage 1 v3 - WebDataset DDP-4 (CSV manifests, floor with_epoch)"
echo "train gpus:    ${train_gpus_id}"
echo "train CSV:     ${train_list_path}"
echo "val   CSV:     ${val_list_path}"
echo "port: 8887"

nohup python -u Source/main.py \
                        --mode train \
                        --batchSize 32 \
                        --gpu 4 \
                        --port 8887 \
                        --trainListPath ${train_list_path} \
                        --valListPath ${val_list_path} \
                        --auto_save_num 1 \
                        --dataloaderNum 8 \
                        --dataloaderType iterable \
                        --imgNum 65275274 \
                        --valImgNum 8145176 \
                        --maxEpochs 8 \
                        --sampleNum 1 \
                        --lr 0.0002 \
                        --dist True \
                        --modelName TM_Model_A \
                        --dataset tmdataset \
                        --is_stage_2 False \
                        --shuffle_buffer 100000 \
                        --log ./log_v3/ \
                        --outputDir ./result_v3/ \
                        --resultImgDir ./ResultImg_v3/ \
                        --modelDir ./Checkpoint_v3/ > ./start_train_stage1_v3.log 2>&1 &

PID=$!
disown $PID 2>/dev/null
echo "PID=$PID"
echo "log: ./start_train_stage1_v3.log"
echo
echo "Tail log:  tail -f start_train_stage1_v3.log"
echo "Kill run:  kill $PID"
