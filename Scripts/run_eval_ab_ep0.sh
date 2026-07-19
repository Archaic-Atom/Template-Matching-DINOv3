#!/usr/bin/env bash
# Eval PCK of A-ft ep0 vs B ep0 @256/192 on the 2k in-domain set (GPU 6/7,
# parallel; trainings keep running on 0-3). Pins each ep0 into a clean dir so the
# live training's checkpoint.list writes don't race the test loader.
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
CSV=/data4/datasets/eval2k_opt256_sar192/image_test_2k_256.csv
export TM_SEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p log_test_debug ResultImg_test_debug

pin() {  # $1=srcdir $2=evaldir
  rm -rf "$2"; mkdir -p "$2"
  cp "$1/model_epoch_0.pth" "$2/"
  { echo "last model name:model_epoch_0.pth"; echo "model_epoch_0.pth"; } > "$2/checkpoint.list"
}

eval_model() {  # $1=modelName $2=evaldir $3=gpu $4=outdir $5=port $6=log
  rm -rf "$4"; mkdir -p "$4"
  CUDA_VISIBLE_DEVICES="$3" "$PY" -u Source/main.py \
    --mode test --batchSize 8 --gpu 1 --port "$5" --dist False \
    --trainListPath "$CSV" --imgNum 2000 --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
    --dataloaderNum 2 --modelName "$1" --dataset tmdataset --is_stage_2 True \
    --fine_dim 32 --fine_margin 24 --sar_size 192 \
    --log ./log_test_debug/ --outputDir "$4/" --resultImgDir ./ResultImg_test_debug/ \
    --modelDir "$2/" > "$6" 2>&1
}

pin Checkpoint_A_ft_256 eval_aft_ep0
pin Checkpoint_B_256    eval_b_ep0

echo "== running test-mode evals (A-ft on gpu6, B on gpu7) =="
eval_model TM_Model_A eval_aft_ep0 6 DebugResult_aft_ep0 8910 eval_aft.log &
PA=$!
eval_model TM_Model_B eval_b_ep0   7 DebugResult_b_ep0   8911 eval_b.log &
PB=$!
wait $PA; echo "A-ft eval rc=$?"
wait $PB; echo "B eval rc=$?"

echo "=== PCK A-ft ep0 (256/192, 2k) ==="
csvA=$(ls -t DebugResult_aft_ep0/test_*.csv 2>/dev/null | head -1)
[ -n "$csvA" ] && "$PY" Scripts/pck_summary.py "$csvA" || { echo "A-ft: NO test csv"; tail -8 eval_aft.log; }
echo "=== PCK B ep0 (256/192, 2k) ==="
csvB=$(ls -t DebugResult_b_ep0/test_*.csv 2>/dev/null | head -1)
[ -n "$csvB" ] && "$PY" Scripts/pck_summary.py "$csvB" || { echo "B: NO test csv"; tail -8 eval_b.log; }
echo "AB_PCK_DONE"
