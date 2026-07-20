#!/usr/bin/env bash
# Parallel loss sweep to lift Stage-1 discrimination off its plateau.
# A: lambda_global 0.1 -> 0.0  (drop the global-alignment term, top suspect)
# B: radius 1 -> 0             (tighten soft window to exact-GT-cell positive)
# Each: 2-GPU DDP, batch 64 (=128 total, matches baseline), subset40, lr 5e-3,
# normalize_logits unchanged. Config is read once at model construction, so we
# sequence: edit -> launch -> wait for "Successfully get user's model!" -> next.
# 并行扫 loss,把 Stage-1 判别从 plateau 拉起来。配置在构造时读一次,顺序编辑即可。
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
CFG="$PROJ/Source/UserModelImplementation/Models/TM_Model_A/config.yaml"
cd "$PROJ" || exit 1

echo '== killing the plateaued run (PID 1011966 + port-8814 workers) =='
kill 1011966 2>/dev/null
sleep 2
pkill -9 -f 'port 8814' 2>/dev/null
sleep 3
echo "GPU0-3 after kill:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n '1,4p'

cp "$CFG" "$CFG.sweepbak"
mkdir -p log_sweep result_sweep ResultImg_sweep

launch() {  # $1=gpus $2=port $3=modeldir $4=log
  CUDA_VISIBLE_DEVICES="$1" JF_PROGRESS_COLUMNS=400 nohup "$PY" -u Source/main.py \
    --mode train --batchSize 64 --gpu 2 --port "$2" --dist True \
    --trainListPath ./Datasets/shard_train_256_subset40.csv \
    --imgNum 448000 --maxEpochs 6 --auto_save_num 1 \
    --dataloaderNum 8 --dataloaderType iterable --sampleNum 1 --lr 0.005 \
    --modelName TM_Model_A --dataset tmdataset --is_stage_2 False --sar_size 192 \
    --log ./log_sweep/ --outputDir ./result_sweep/ --resultImgDir ./ResultImg_sweep/ \
    --modelDir "./$3/" > "$4" 2>&1 &
  echo "$!"
}

wait_built() {  # $1=log ; returns when the model is constructed (config consumed)
  for _ in $(seq 1 80); do
    grep -q "Successfully get user's model" "$1" 2>/dev/null && { echo "  built ($1)"; return 0; }
    grep -qiE "Traceback|Error|Exception" "$1" 2>/dev/null && { echo "  ERROR ($1):"; tail -5 "$1"; return 1; }
    sleep 3
  done
  echo "  TIMEOUT waiting build ($1)"; return 1
}

# ---- Variant A: lambda_global = 0.0 (radius stays 1) ----
cp "$CFG.sweepbak" "$CFG"
sed -i -E 's/^( *)lambda_global: 0\.1/\1lambda_global: 0.0/' "$CFG"
echo "A config:"; grep -E 'lambda_global|radius' "$CFG"
PIDA=$(launch 0,1 8820 Checkpoint_s1_lamg0 train_lamg0.log)
echo "VARIANT_A lamg0 PID=$PIDA gpus=0,1 dir=Checkpoint_s1_lamg0"
wait_built train_lamg0.log

# ---- Variant B: radius = 0 (lambda_global back to 0.1) ----
cp "$CFG.sweepbak" "$CFG"
sed -i -E 's/^( *)radius: 1$/\1radius: 0/' "$CFG"
echo "B config:"; grep -E 'lambda_global|radius' "$CFG"
PIDB=$(launch 2,3 8821 Checkpoint_s1_rad0 train_rad0.log)
echo "VARIANT_B rad0 PID=$PIDB gpus=2,3 dir=Checkpoint_s1_rad0"
wait_built train_rad0.log

# ---- restore canonical config ----
cp "$CFG.sweepbak" "$CFG"; rm -f "$CFG.sweepbak"
echo "config restored:"; grep -E 'lambda_global|radius' "$CFG"
echo "SWEEP_LAUNCHED A=$PIDA B=$PIDB"
