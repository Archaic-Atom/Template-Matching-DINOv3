#!/usr/bin/env bash
# =====================================================================
# run_train_C_direct.sh — DIRECT-trained cascade C (256/192), two-phase plan.
# 直接训练级联 C 模型(随机初始化 + 可训练编码器 + 无 Stage-1),分两阶段。
#
# Throughput reality (measured 2026-06-14): the cascade is COMPUTE-bound
# (~8 samples/s/GPU, ~1.4 s/step at bs10x4); full 13.18M ≈ 5 days/epoch.
# 级联是算力瓶颈(GPU 打满),全集 ~5 天/epoch,故采用两阶段:
#   Phase 1 (sub40):       subset40 1.34M x5 ep  ≈ 3 days  -> 快速收敛
#   Phase 2 (full_resume): full 13.18M x2-3 ep, warm-start from Phase-1 ckpt
#                          ≈ 11-16 days -> 见过全部数据,~7月头出结果
#
# C is config-driven (TM_Model_C/config.yaml edited 2026-06-14 to direct:
#   sar_size 192, load_pretrained_sar false, freeze_sar_encoder false,
#   cascade warmup 1/2). Cascade-loss weights live in the yaml (no CLI flag).
#
# Usage:
#   bash Scripts/run_train_C_direct.sh sub40                 # Phase 1
#   bash Scripts/run_train_C_direct.sh full_resume <ckpt>    # Phase 2 (later)
#   bash Scripts/run_train_C_direct.sh smoke                 # 1-GPU build check
# =====================================================================
set -u
MODE="${1:-sub40}"
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # fragmentation safety on the 24GB 3090s

# launch_4gpu <manifest> <ckptdir> <maxEpochs> <tag> [batch] [port]
launch_4gpu() {
  local MAN="$1" CKPT="$2" EP="$3" TAG="$4" BS="${5:-10}" PORT="${6:-8866}"
  local RT; RT=$(tail -n +2 "$MAN" | awk -F, '{s+=$2} END{print s}')
  echo "[C-direct/$TAG] manifest=$MAN real_total=$RT bs=$BS gpu=4 maxEpochs=$EP ckpt=$CKPT port=$PORT"
  CUDA_VISIBLE_DEVICES=0,1,2,3 JF_PROGRESS_COLUMNS=400 nohup "$PY" -u Source/main.py \
    --mode train --dataset tmdataset --modelName TM_Model_C --is_stage_2 True \
    --dataloaderType iterable --dataloaderNum 8 --sampleNum 1 \
    --sar_size 192 --matching_dim 256 --shuffle_buffer 100000 --lr 1e-3 \
    --auto_save_num 1 --dist True --gpu 4 --batchSize "$BS" --port "$PORT" \
    --trainListPath "./$MAN" --imgNum "$RT" --maxEpochs "$EP" \
    --log "./log_$TAG/" --outputDir "./result_$TAG/" --resultImgDir "./ResultImg_$TAG/" \
    --modelDir "./$CKPT/" > "train_$TAG.log" 2>&1 &
  echo "${TAG}_PID=$!  log=train_$TAG.log"
}

case "$MODE" in
  sub40)   # ---- Phase 1: subset40 1.34M x5 ----
    CKPT=Checkpoint_C_direct_sub40_256
    if [ -d "$CKPT" ] && ls "$CKPT"/model_epoch_*.pth >/dev/null 2>&1; then
      echo "ERROR: $CKPT already has checkpoints — move it aside first."; exit 1; fi
    rm -rf log_c_sub40 result_c_sub40 ResultImg_c_sub40; mkdir -p "$CKPT"
    launch_4gpu Datasets/shard_train_256_subset40.csv "$CKPT" 5 c_sub40 10 8866
    ;;

  full_resume)   # ---- Phase 2: full 13.18M x3, warm-start from Phase-1 ckpt ----
    SRC="${2:?usage: full_resume <phase1_ckpt.pth>}"
    [ -f "$SRC" ] || { echo "ERROR: phase-1 ckpt not found: $SRC"; exit 1; }
    CKPT=Checkpoint_C_direct_full_256
    if [ -d "$CKPT" ] && ls "$CKPT"/model_epoch_*.pth >/dev/null 2>&1; then
      echo "ERROR: $CKPT already has checkpoints — move it aside first."; exit 1; fi
    mkdir -p "$CKPT"
    # Warm-start: drop optimizer state + reset epoch so the full run starts clean
    # epoch counting (cascade warmup is already past by Phase 2 -> fine head at full weight).
    "$PY" Source/Tools/strip_stage1_opt.py "$SRC" "$CKPT/model_phase1_init.pth"
    { echo "last model name:model_phase1_init.pth"; echo "model_phase1_init.pth"; } > "$CKPT/checkpoint.list"
    echo "[C-direct/full] warm-started from $SRC"
    launch_4gpu Datasets/shard_train_256_full.csv "$CKPT" 3 c_full 10 8867
    ;;

  smoke)   # ---- 1-GPU dist:False build/shape sanity (imgNum=sub40 real_total so save fires) ----
    CKPT=Checkpoint_C_direct_smoke
    rm -rf "$CKPT" log_c_smoke result_c_smoke ResultImg_c_smoke; mkdir -p "$CKPT"
    MAN=Datasets/shard_train_256_subset40.csv
    RT=$(tail -n +2 "$MAN" | awk -F, '{s+=$2} END{print s}')
    echo "[C-direct/smoke] 1 GPU dist:False bs4 (Ctrl-C after a few hundred iters)"
    CUDA_VISIBLE_DEVICES=0 "$PY" -u Source/main.py \
      --mode train --dataset tmdataset --modelName TM_Model_C --is_stage_2 True \
      --dataloaderType iterable --dataloaderNum 4 --sampleNum 1 \
      --sar_size 192 --matching_dim 256 --shuffle_buffer 2000 --lr 1e-3 \
      --auto_save_num 1 --dist False --gpu 1 --batchSize 4 --port 8869 \
      --trainListPath "./$MAN" --imgNum "$RT" --maxEpochs 1 \
      --log ./log_c_smoke/ --outputDir ./result_c_smoke/ --resultImgDir ./ResultImg_c_smoke/ \
      --modelDir "./$CKPT/" 2>&1 | tail -60
    ;;

  *) echo "usage: $0 {sub40|full_resume <ckpt>|smoke}"; exit 1;;
esac
