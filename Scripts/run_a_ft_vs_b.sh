#!/usr/bin/env bash
# A-finetune vs B @256/192 — the clean "does Stage-1 init help" ablation.
#   A-ft (TM_Model_A): SAR warm-starts from Stage-1 (Checkpoint_A_stage1_256),
#                      NOT frozen (config freeze_sar_encoder:false) -> trains.
#   B    (TM_Model_B): SAR random init (load_pretrained_sar:false) -> trains.
# ONLY difference = SAR init. Both 256/192, both trainable, same head/loss/data.
# Parallel 2-GPU each (A:0,1  B:2,3), subset40 (1.34M, imgNum=real_total honest),
# nohup'd. Eval PCK on per-epoch ckpts afterwards.
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
MAN=Datasets/shard_train_256_subset40.csv
RT=$(tail -n +2 "$MAN" | awk -F, '{s+=$2} END{print s}')
echo "manifest=$MAN real_total=$RT"

# --- bootstrap A-finetune from the Stage-1 aligned encoder ---
S1=Checkpoint_A_stage1_256
AFT=Checkpoint_A_ft_256
rm -rf "$AFT"; mkdir -p "$AFT"
latest=$(ls -t "$S1"/model_epoch_*.pth 2>/dev/null | head -1)
if [ -z "$latest" ]; then echo "ERROR: no Stage-1 ckpt in $S1"; exit 1; fi
echo "A-ft bootstrap from: $latest"
"$PY" Source/Tools/strip_stage1_opt.py "$latest" "$AFT/model_stage1_init.pth"
{ echo "last model name:model_stage1_init.pth"; echo "model_stage1_init.pth"; } > "$AFT/checkpoint.list"

# --- B starts fresh (random SAR) ---
rm -rf Checkpoint_B_256; mkdir -p Checkpoint_B_256

launch() {  # $1=modelName $2=gpus $3=port $4=modeldir $5=log
  CUDA_VISIBLE_DEVICES=$2 JF_PROGRESS_COLUMNS=400 nohup "$PY" -u Source/main.py \
    --mode train --batchSize 32 --gpu 2 --port "$3" --dist True \
    --trainListPath "./$MAN" --imgNum "$RT" --maxEpochs 5 --auto_save_num 1 \
    --dataloaderNum 8 --dataloaderType iterable --sampleNum 1 --lr 1e-3 \
    --modelName "$1" --dataset tmdataset --is_stage_2 True --sar_size 192 \
    --fine_margin 24 --matching_dim 256 --fine_dim 32 --shuffle_buffer 100000 \
    --match_w4 1.0 --match_w8 0.5 --match_sigma4 1.0 --match_sigma8 0.5 \
    --match_primary_stride 4 --match_w_fine 1.0 --match_sigma_fine 1.0 \
    --fine_warmup_epochs 3 \
    --log "./log_$4/" --outputDir "./result_$4/" --resultImgDir "./ResultImg_$4/" \
    --modelDir "./$4/" > "$5" 2>&1 &
  echo "$!"
}

PIDA=$(launch TM_Model_A 0,1 8840 "$AFT" train_a_ft.log)
echo "A_FT_PID=$PIDA  (TM_Model_A finetune, warm-start, gpu0,1, $AFT)"
PIDB=$(launch TM_Model_B 2,3 8841 Checkpoint_B_256 train_b256.log)
echo "B_PID=$PIDB  (TM_Model_B random, gpu2,3, Checkpoint_B_256)"
echo "AB_LAUNCHED A=$PIDA B=$PIDB"
