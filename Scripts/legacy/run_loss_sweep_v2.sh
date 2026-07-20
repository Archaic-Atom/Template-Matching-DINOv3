#!/usr/bin/env bash
# Loss sweep v2: FAST. Each variant on all 6 free GPUs (0,1,2,3,6,7) SEQUENTIALLY
# (ViT-L is GPU-bound -> more GPUs beats parallel-on-fewer), 1 epoch each on a
# 20-shard (640k) subset, then held-out eval inline. Everything nohup'd in ONE
# script so it survives tunnel drops. Compares A:lambda_global=0 vs B:radius=0
# vs baseline (d'=0.03/win0.54) vs oracle ceiling (d'=1.45/win0.98).
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
CFG="$PROJ/Source/UserModelImplementation/Models/TM_Model_A/config.yaml"
cd "$PROJ" || exit 1
GPUS=0,1,2,3,6,7
EVAL_CSV=Datasets/test_256_2k.csv

echo "== kill prior variants (ports 8820/8821) =="
pkill -9 -f 'port 8820' 2>/dev/null; pkill -9 -f 'port 8821' 2>/dev/null; sleep 4

echo "== build 20-shard (640k) manifest =="
MAN=Datasets/shard_train_256_sub20.csv
head -21 Datasets/shard_train_256_subset40.csv > "$MAN"
RT=$(tail -n +2 "$MAN" | awk -F, '{s+=$2} END{print s}')
echo "sub20 real_total=$RT shards=$(($(wc -l < "$MAN")-1))"

cp "$CFG" "$CFG.sweepbak"; mkdir -p log_sweep result_sweep ResultImg_sweep DebugResult

run_one() {  # $1=tag $2=modeldir $3=port  (blocks until the 1-epoch run exits)
  rm -rf "$2"; mkdir -p "$2"
  CUDA_VISIBLE_DEVICES=$GPUS JF_PROGRESS_COLUMNS=400 "$PY" -u Source/main.py \
    --mode train --batchSize 32 --gpu 6 --port "$3" --dist True \
    --trainListPath "./$MAN" --imgNum "$RT" --maxEpochs 1 --auto_save_num 1 \
    --dataloaderNum 8 --dataloaderType iterable --sampleNum 1 --lr 0.005 \
    --modelName TM_Model_A --dataset tmdataset --is_stage_2 False --sar_size 192 \
    --log ./log_sweep/ --outputDir ./result_sweep/ --resultImgDir ./ResultImg_sweep/ \
    --modelDir "./$2/" > "train_$1.log" 2>&1
  echo "  $1 training exited rc=$?"
}

eval_one() {  # $1=tag $2=modeldir
  CUDA_VISIBLE_DEVICES=0 "$PY" Source/Tools/stage1_separation_2k.py \
    --ckpt "$2/model_epoch_0.pth" --csv "$EVAL_CSV" \
    --out "DebugResult/sep_$1_ep0.csv" > "eval_$1.log" 2>&1
  echo "  $1 eval rc=$?"
}

echo "== VARIANT A: lambda_global=0 =="
cp "$CFG.sweepbak" "$CFG"; sed -i -E 's/^( *)lambda_global: 0\.1/\1lambda_global: 0.0/' "$CFG"
grep -E 'lambda_global|radius:' "$CFG"
run_one lamg0 Checkpoint_s1_lamg0 8830
eval_one lamg0 Checkpoint_s1_lamg0

echo "== VARIANT B: radius=0 =="
cp "$CFG.sweepbak" "$CFG"; sed -i -E 's/^( *)radius: 1$/\1radius: 0/' "$CFG"
grep -E 'lambda_global|radius:' "$CFG"
run_one rad0 Checkpoint_s1_rad0 8831
eval_one rad0 Checkpoint_s1_rad0

cp "$CFG.sweepbak" "$CFG"; rm -f "$CFG.sweepbak"
echo "== config restored =="; grep -E 'lambda_global|radius:' "$CFG"

echo "=== SWEEP RESULTS (held-out 2k 256/192) ==="
echo "baseline: top1<=1cell=0.597 gt>far=0.54 d'=0.03 | oracle: 1.000 0.98 1.45"
"$PY" - <<'PYEOF'
import pandas as pd, numpy as np
for tag in ('lamg0','rad0'):
    try:
        d=pd.read_csv(f"DebugResult/sep_{tag}_ep0.csv")
        g,f=d['cos_gt'],d['cos_max_far']
        dp=(g.mean()-f.mean())/np.sqrt(0.5*(g.var()+f.var())+1e-9)
        print(f"{tag}: top1<=1cell={(d.cell_dist_argmax<=1).mean():.4f} "
              f"top1exact={(d.cell_dist_argmax==0).mean():.4f} "
              f"gt>far={(g>f).mean():.3f} d'={dp:.3f}")
    except Exception as e:
        print(f"{tag}: ERROR {e}")
PYEOF
echo "SWEEP_RESULTS_DONE"
