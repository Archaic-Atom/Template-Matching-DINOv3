#!/usr/bin/env bash
# Stage 2 (template matching) training - WebDataset version.
#
# Reads pre-packed shards from /data5 (sequential HDD reads). Stage 2 freezes
# both DINOv3 and the Stage-1-trained SAR encoder, learning only the coarse
# (stride-4 + stride-8) and fine (stride-1) matching heads.
#
# IMPORTANT: copy your best Stage-1 checkpoint into ./Checkpoint_stage2/ BEFORE
# launching. JackFramework's ModelSaver auto-loads the latest checkpoint from
# --modelDir; load_model() uses strict=False, so the SAR encoder weights are
# read in and the new matching components are randomly initialized.

train_gpus_id=0,1,2,3
train_list_path='/data5/datasets/TM_shards/shard-{000000..000254}.tar'

# Force tqdm progress lines to render at full width so e:/l0:/a0: don't get
# truncated when python writes to a non-TTY log file.
export COLUMNS=200

echo "Stage 2 (template matching) - WebDataset version"
echo "train gpus: ${train_gpus_id}"
echo "shards:     ${train_list_path}"
echo "Make sure ./Checkpoint_stage2/ contains your latest Stage-1 model_epoch_N.pth."

CUDA_VISIBLE_DEVICES=${train_gpus_id} python -u Source/main.py \
                        --mode train \
                        --batchSize 8 \
                        --gpu 4 \
                        --trainListPath ${train_list_path} \
                        --auto_save_num 1 \
                        --dataloaderNum 8 \
                        --dataloaderType iterable \
                        --imgNum 8145177 \
                        --maxEpochs 30 \
                        --sampleNum 1 \
                        --lr 1e-3 \
                        --dist True \
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
                        --log ./log_stage2/ \
                        --outputDir ./result_stage2/ \
                        --resultImgDir ./ResultImg_stage2/ \
                        --modelDir ./Checkpoint_stage2/ > ./start_train_stage2_wds.log 2>&1 &

echo "started; tailing log..."
tail -f ./start_train_stage2_wds.log
