#!/usr/bin/env bash
# Matched zero-shot WHU-OPT-SAR eval: A-ft-256 vs B-256 (both epoch_4).
# matched 跨传感器零样本 WHU eval:A-ft-256 vs B-256(都 epoch_4)。
#
#   A-ft (TM_Model_A): SAR trunk warm-started from Stage-1 (Checkpoint_A_stage1_256)
#                      then fine-tuned (freeze_sar_encoder:false). modelDir Checkpoint_A_ft_256.
#   B    (TM_Model_B): SAR random init, fine-tuned. modelDir Checkpoint_B_256.
# Both dim256 / sar_size 192 / fine_margin 24 — ARCHITECTURE-MATCHED, only the
# SAR init differs. This defuses the old dim128-A-vs-dim256-B confound and asks:
# does Stage-1 alignment help CROSS-SENSOR generalization (WHU = GF-1 optical +
# GF-3 SAR, resampled to 10 m to match the SEN1-2 training GSD)?
# 两者架构对齐、仅 SAR 初始化不同(A 有 Stage-1,B 随机),拆掉 dim128 混淆,
# 问:Stage-1 对齐是否帮助跨传感器泛化。
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
PCK=$(ls Source/Tools/pck_summary.py Scripts/pck_summary.py 2>/dev/null | head -1)
# Args: [CSV] [N] [TAG] — default = WHU cross-sensor. QXS cross-res:
#   bash Scripts/run_eval_whu_matched.sh /data5/datasets/cross_eval/qxs_saropt/cross_eval.csv 2000 qxs
CSV="${1:-/data5/datasets/cross_eval/whu_opt_sar/cross_eval.csv}"
N="${2:-6400}"
TAG="${3:-whu}"
export TM_SEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p log_test_debug ResultImg_test_debug

pin() {  # $1=srcdir $2=evaldir $3=epoch
  rm -rf "$2"; mkdir -p "$2"
  cp "$1/model_epoch_$3.pth" "$2/"
  { echo "last model name:model_epoch_$3.pth"; echo "model_epoch_$3.pth"; } > "$2/checkpoint.list"
}

eval_model() {  # $1=modelName $2=modelDir $3=outdir $4=port $5=log
  rm -rf "$3"; mkdir -p "$3"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u Source/main.py \
    --mode test --batchSize 8 --gpu 1 --port "$4" --dist False \
    --trainListPath "$CSV" --imgNum "$N" --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
    --dataloaderNum 2 --modelName "$1" --dataset tmdataset --is_stage_2 True \
    --fine_dim 32 --fine_margin 24 --sar_size 192 --matching_dim 256 \
    --log ./log_test_debug/ --outputDir "$3/" --resultImgDir ./ResultImg_test_debug/ \
    --modelDir "$2/" > "$5" 2>&1
}

report() {  # $1=outdir $2=tag $3=log
  echo "=== PCK $2 (zero-shot ${TAG}, 256/192, N=$N) ==="
  csv=$(ls -t "$1"/test_*.csv 2>/dev/null | head -1)
  if [ -n "$csv" ]; then "$PY" "$PCK" "$csv"; else echo "$2: NO test csv"; tail -15 "$3"; fi
}

pin Checkpoint_A_ft_256 "eval_aft_${TAG}" 4
pin Checkpoint_B_256    "eval_b_${TAG}"   4

echo "== A-ft-256 ${TAG} =="
eval_model TM_Model_A "eval_aft_${TAG}" "DebugResult_aft_${TAG}" 8925 "eval_aft_${TAG}.log"
report "DebugResult_aft_${TAG}" "A-ft-256-${TAG}" "eval_aft_${TAG}.log"

echo "== B-256 ${TAG} =="
eval_model TM_Model_B "eval_b_${TAG}" "DebugResult_b_${TAG}" 8926 "eval_b_${TAG}.log"
report "DebugResult_b_${TAG}" "B-256-${TAG}" "eval_b_${TAG}.log"

echo "CROSS_MATCHED_DONE_${TAG}"
