#!/usr/bin/env bash
# =====================================================================
# run_train_C_direct_full.sh — train TM_Model_C (3-level cascade) by
# DIRECT supervision on the FULL 256/192 training set.
# 用全部训练数据直接训练 C（级联）模型 —— 论文 Table I 的对标 SOTA 行。
#
# This is the paper's SOTA-comparison model: Table I row
# "Ours (cascade + sub-pixel, full set)" and the C2 demonstration claim.
# "Direct" = random SAR init + trainable encoder, NO Stage-1, NO freeze.
# Requires TM_Model_C/config.yaml edited (2026-06-14) to:
#     sar_size: 192   load_pretrained_sar: false   freeze_sar_encoder: false
# C is fully config-driven: the cascade-loss weights/sigmas/warmups
# (b4=1, b8=.5, w2=.5, w1=1, s4=1cell, s8=.5cell, s2=s1=1px, warmup 2/4 —
# matching sec_system.tex Eq. match_total) all come from config.yaml.
#
# Usage:
#   bash Scripts/run_train_C_direct_full.sh smoke        # 1-GPU dist:False build/save sanity
#   bash Scripts/run_train_C_direct_full.sh mem   32     # 1-GPU memory probe at batch 32
#   bash Scripts/run_train_C_direct_full.sh full  32     # 4-GPU nohup full run (batch defaults 32)
# =====================================================================
set -u
MODE="${1:-smoke}"
BS="${2:-}"
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1

MAN=Datasets/shard_train_256_full.csv
RT=$(tail -n +2 "$MAN" | awk -F, '{s+=$2} END{print s}')
echo "[C-direct] manifest=$MAN real_total=$RT (expect 13182912)"

# Args shared by every mode. sar_size/matching_dim passed for parity with the
# B launcher; the BUILDER actually reads them from config.yaml (hub §6.1).
COMMON=(--mode train --dataset tmdataset --modelName TM_Model_C --is_stage_2 True
        --dataloaderType iterable --dataloaderNum 8 --sampleNum 1
        --sar_size 192 --matching_dim 256 --shuffle_buffer 100000
        --lr 1e-3 --auto_save_num 1)

case "$MODE" in
  smoke)
    CKPT=Checkpoint_C_direct_smoke
    rm -rf "$CKPT" log_C_smoke result_C_smoke ResultImg_C_smoke; mkdir -p "$CKPT"
    echo "[C-direct] SMOKE: 1 GPU, dist False, batch ${BS:-4}, imgNum 2000, 1 epoch"
    CUDA_VISIBLE_DEVICES=0 JF_PROGRESS_COLUMNS=200 "$PY" -u Source/main.py \
      "${COMMON[@]}" --dist False --gpu 1 --batchSize "${BS:-4}" --port 8867 \
      --trainListPath "./$MAN" --imgNum 2000 --maxEpochs 1 \
      --log ./log_C_smoke/ --outputDir ./result_C_smoke/ \
      --resultImgDir ./ResultImg_C_smoke/ --modelDir "./$CKPT/" 2>&1 | tail -80
    echo "[C-direct] SMOKE finished — inspect above for build / loss / epoch-0 save."
    ;;
  mem)
    CKPT=Checkpoint_C_direct_mem
    rm -rf "$CKPT" log_C_mem result_C_mem ResultImg_C_mem; mkdir -p "$CKPT"
    echo "[C-direct] MEM PROBE: 1 GPU, dist False, batch ${BS:-32}, imgNum 512 (watch nvidia-smi peak)"
    CUDA_VISIBLE_DEVICES=0 JF_PROGRESS_COLUMNS=200 "$PY" -u Source/main.py \
      "${COMMON[@]}" --dist False --gpu 1 --batchSize "${BS:-32}" --port 8868 \
      --trainListPath "./$MAN" --imgNum 512 --maxEpochs 1 \
      --log ./log_C_mem/ --outputDir ./result_C_mem/ \
      --resultImgDir ./ResultImg_C_mem/ --modelDir "./$CKPT/" 2>&1 | tail -40
    echo "[C-direct] MEM PROBE finished — if no OOM, batch ${BS:-32} is safe per GPU."
    ;;
  full)
    CKPT=Checkpoint_C_direct_full_256
    if [ -d "$CKPT" ] && ls "$CKPT"/model_epoch_*.pth >/dev/null 2>&1; then
      echo "ERROR: $CKPT already has checkpoints — refusing to overwrite. Move it aside first."; exit 1
    fi
    mkdir -p "$CKPT"
    echo "[C-direct] FULL: 4 GPU, dist True, batch ${BS:-32}, imgNum $RT, 5 epochs, nohup"
    CUDA_VISIBLE_DEVICES=0,1,2,3 JF_PROGRESS_COLUMNS=400 nohup "$PY" -u Source/main.py \
      "${COMMON[@]}" --dist True --gpu 4 --batchSize "${BS:-32}" --port 8866 \
      --trainListPath "./$MAN" --imgNum "$RT" --maxEpochs 5 \
      --log ./log_C_direct_full/ --outputDir ./result_C_direct_full/ \
      --resultImgDir ./ResultImg_C_direct_full/ --modelDir "./$CKPT/" \
      > train_c_direct_full.log 2>&1 &
    echo "C_DIRECT_FULL_PID=$!  modelDir=$CKPT  log=train_c_direct_full.log"
    ;;
  *)
    echo "usage: $0 {smoke|mem|full} [batchSize]"; exit 1
    ;;
esac
