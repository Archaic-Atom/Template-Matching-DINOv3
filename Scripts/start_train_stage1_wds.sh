#!/usr/bin/env bash
# Stage 1 (feature alignment) training - WebDataset version.
#
# Reads pre-packed shards from /data5 (sequential HDD reads), as opposed to
# the 16M scattered PNG files on /data1 used by start_train_dataset_model.sh.
# Stage 1 mode is the default (--is_stage_2 False), trains the SAR encoder
# to align with frozen DINOv3 features via InfoNCE-style alignment_loss.

train_gpus_id=0,1,2,3
train_list_path='/data5/datasets/TM_shards/shard-{000000..000254}.tar'

# Force tqdm progress lines to render at full width so e:/l0:/a0: don't get
# truncated when python writes to a non-TTY log file.
export COLUMNS=200

echo "Stage 1 (feature alignment) - WebDataset version"
echo "train gpus: ${train_gpus_id}"
echo "shards:     ${train_list_path}"

CUDA_VISIBLE_DEVICES=${train_gpus_id} python -u Source/main.py \
                        --mode train \
                        --batchSize 32 \
                        --gpu 4 \
                        --trainListPath ${train_list_path} \
                        --auto_save_num 1 \
                        --dataloaderNum 8 \
                        --dataloaderType iterable \
                        --imgNum 8145177 \
                        --maxEpochs 8 \
                        --sampleNum 1 \
                        --lr 0.0002 \
                        --dist True \
                        --modelName TM_Model_A \
                        --dataset tmdataset \
                        --is_stage_2 False \
                        --log ./log/ \
                        --outputDir ./result/ \
                        --resultImgDir ./ResultImg/ \
                        --modelDir ./Checkpoint/ > ./start_train_stage1_wds.log 2>&1 &

echo "PID=$\!"; echo "log: ./start_train_stage1_wds.log"
