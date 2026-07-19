#!/usr/bin/env bash
# Full-pipeline PCK for the frozen arms at epoch_3, run IN PARALLEL on RF's two
# GPUs (2,3) so it finishes in ~3min and never touches S1F (GPU0,1).
# 冻结双臂 epoch_3 全流程 PCK,并行铺在 RF 的两张卡(2,3),不碰 S1F。
# B-e4 reference reused from §10.5.3 (fixed converged ckpt) — not re-run here.
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
CSV=/data4/datasets/eval2k_opt256_sar192/image_test_2k_256.csv
export TM_SEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p log_test_debug ResultImg_test_debug

pin() {  # $1=srcdir $2=evaldir $3=epoch
  rm -rf "$2"; mkdir -p "$2"
  cp "$1/model_epoch_$3.pth" "$2/"
  { echo "last model name:model_epoch_$3.pth"; echo "model_epoch_$3.pth"; } > "$2/checkpoint.list"
}

eval_model() {  # $1=modelName $2=modelDir $3=outdir $4=port $5=log $6=gpu
  rm -rf "$3"; mkdir -p "$3"
  CUDA_VISIBLE_DEVICES="$6" "$PY" -u Source/main.py \
    --mode test --batchSize 4 --gpu 1 --port "$4" --dist False \
    --trainListPath "$CSV" --imgNum 2000 --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
    --dataloaderNum 2 --modelName "$1" --dataset tmdataset --is_stage_2 True \
    --fine_dim 32 --fine_margin 24 --sar_size 192 --matching_dim 256 \
    --log ./log_test_debug/ --outputDir "$3/" --resultImgDir ./ResultImg_test_debug/ \
    --modelDir "$2/" > "$5" 2>&1
}

report() {  # $1=outdir $2=tag $3=log
  echo "=== PCK $2 (full pipeline, 256/192, 2k) ==="
  csv=$(ls -t "$1"/test_*.csv 2>/dev/null | head -1)
  if [ -n "$csv" ]; then "$PY" Scripts/pck_summary.py "$csv"; else echo "$2: NO test csv"; tail -12 "$3"; fi
}

pin Checkpoint_F_s2_frozen_256  eval_s1f_e3 3
pin Checkpoint_RF_s2_frozen_256 eval_rf_e3  3

# Parallel on RF's GPUs / 并行铺在 RF 的两张卡
eval_model TM_Model_F eval_s1f_e3 DebugResult_s1f_e3 8930 eval_s1f_e3.log 2 &
PS=$!
eval_model TM_Model_F eval_rf_e3  DebugResult_rf_e3  8931 eval_rf_e3.log  3 &
PR=$!
wait $PS; echo "S1F-e3 test rc=$?"
wait $PR; echo "RF-e3 test rc=$?"

report DebugResult_s1f_e3 S1F-e3 eval_s1f_e3.log
report DebugResult_rf_e3  RF-e3  eval_rf_e3.log
echo "FROZEN_E3_PCK_DONE"
