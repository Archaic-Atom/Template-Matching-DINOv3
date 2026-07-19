#!/usr/bin/env bash
# Full-pipeline (fine-head included) PCK eval for the frozen arms + B baseline.
# 全流程(含 fine head)PCK 评测:冻结双臂 epoch_$FEPOCH + B-e4 参照。
#
# Usage: bash Scripts/run_eval_frozen_full.sh [FEPOCH=4] [EVAL_CSV] [EVAL_N] [TAG]
#   default (2k, e4):  bash Scripts/run_eval_frozen_full.sh 4
#   20k larger sample: bash Scripts/run_eval_frozen_full.sh 4 \
#       /data4/datasets/eval20k_opt256_sar192/image_test_2k_256.csv 20000 e4_20k
#
# A bigger EVAL_N shrinks the test-sampling CI (~1/sqrt(N)) so a small but real
# gap (e.g. S1F vs RF ~1.5pt) becomes cleanly significant — much cheaper than
# multi-seed retraining, since it is inference on the already-trained models.
# 增大 EVAL_N 收窄测试采样 CI(~1/sqrt(N)),让小而真的差距变显著;比多 seed
# 重训便宜得多(只是对已训模型多跑推理)。
#
# Runs sequentially on GPU1 (now free; batch 4 kept identical to the 2k run for
# number-for-number comparability — matching inference is per-sample independent
# so batch size does not change predictions, but we keep it fixed anyway).
set -u
FEPOCH="${1:-4}"
EVAL_CSV="${2:-/data4/datasets/eval2k_opt256_sar192/image_test_2k_256.csv}"
EVAL_N="${3:-2000}"
TAG="${4:-e${FEPOCH}}"
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
# pck_summary.py is in Source/Tools/ locally but Scripts/ on the remote after the
# 06-12 reorg — locate it wherever it is so report() never silently no-ops.
PCK=$(ls Source/Tools/pck_summary.py Scripts/pck_summary.py 2>/dev/null | head -1)
export TM_SEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p log_test_debug ResultImg_test_debug
echo "FEPOCH=$FEPOCH EVAL_N=$EVAL_N TAG=$TAG CSV=$EVAL_CSV"

pin() {  # $1=srcdir $2=evaldir $3=epoch
  rm -rf "$2"; mkdir -p "$2"
  cp "$1/model_epoch_$3.pth" "$2/"
  { echo "last model name:model_epoch_$3.pth"; echo "model_epoch_$3.pth"; } > "$2/checkpoint.list"
}

eval_model() {  # $1=modelName $2=modelDir $3=outdir $4=port $5=log
  rm -rf "$3"; mkdir -p "$3"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u Source/main.py \
    --mode test --batchSize 4 --gpu 1 --port "$4" --dist False \
    --trainListPath "$EVAL_CSV" --imgNum "$EVAL_N" --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
    --dataloaderNum 2 --modelName "$1" --dataset tmdataset --is_stage_2 True \
    --fine_dim 32 --fine_margin 24 --sar_size 192 --matching_dim 256 \
    --log ./log_test_debug/ --outputDir "$3/" --resultImgDir ./ResultImg_test_debug/ \
    --modelDir "$2/" > "$5" 2>&1
}

report() {  # $1=outdir $2=tag $3=log
  echo "=== PCK $2 (full pipeline, 256/192, N=$EVAL_N) ==="
  csv=$(ls -t "$1"/test_*.csv 2>/dev/null | head -1)
  if [ -n "$csv" ]; then "$PY" "$PCK" "$csv"; else echo "$2: NO test csv"; tail -12 "$3"; fi
}

pin Checkpoint_F_s2_frozen_256  "eval_s1f_${TAG}" "${FEPOCH}"
pin Checkpoint_RF_s2_frozen_256 "eval_rf_${TAG}"  "${FEPOCH}"

echo "== S1F-${TAG} =="
eval_model TM_Model_F "eval_s1f_${TAG}" "DebugResult_s1f_${TAG}" 8920 "eval_s1f_${TAG}.log"
report "DebugResult_s1f_${TAG}" "S1F-${TAG}" "eval_s1f_${TAG}.log"

echo "== RF-${TAG} =="
eval_model TM_Model_F "eval_rf_${TAG}" "DebugResult_rf_${TAG}" 8921 "eval_rf_${TAG}.log"
report "DebugResult_rf_${TAG}" "RF-${TAG}" "eval_rf_${TAG}.log"

# B baseline is always pinned at epoch_4 (its final), regardless of FEPOCH.
echo "== B-${TAG} =="
pin Checkpoint_B_256 "eval_b_${TAG}" 4
eval_model TM_Model_B "eval_b_${TAG}" "DebugResult_b_${TAG}" 8922 "eval_b_${TAG}.log"
report "DebugResult_b_${TAG}" "B-${TAG}" "eval_b_${TAG}.log"

echo "FROZEN_FULL_PCK_DONE"
