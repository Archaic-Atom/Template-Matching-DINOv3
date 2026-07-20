#!/usr/bin/env bash
# WHU resolution sweep: regenerate WHU at multiple GSDs (downsample from 5m
# native), eval B@256/192 zero-shot on each, print PCK@3-vs-GSD curve.
# Controlled experiment isolating RESOLUTION (same sensor/scenes, only GSD
# varies) — cleaner than QXS which confounds resolution + sensor + opt source.
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
ADP="$PROJ/Source/Tools/cross_dataset_adapter.py"
CFG="$PROJ/Source/UserModelImplementation/Models/TM_Model_B/config.yaml"
RAW=/data5/datasets/whu-opt-sar
OUTBASE=/data5/datasets/cross_eval_resolution
cd "$PROJ" || exit 1
GSDS="5 8 10 16 24"      # m/px; all >= WHU native 5m (downsample only)
SPP=32                    # 100 scenes x 32 = 3200 samples / GSD

echo "==== WHU RES SWEEP START $(date) ===="

# ---- Phase 1: generate WHU at each GSD ----
for g in $GSDS; do
  echo "---- gen WHU @ ${g}m ----"
  $PY "$ADP" --dataset whu_opt_sar --raw_dir "$RAW" \
      --out_dir "$OUTBASE/whu_${g}m" --target_res_m "$g" \
      --samples_per_scene "$SPP" 2>&1 | grep -vE "DeprecationWarning|Image.fromarray" | tail -2
done

# ---- Phase 2: eval B@256/192 (sar_size 192 via yaml; trap restores on exit) ----
cp "$CFG" /tmp/B_cfg_sweep_backup.yaml
trap 'cp /tmp/B_cfg_sweep_backup.yaml "$CFG"; echo "[trap] B config restored"' EXIT
sed -i "s/      sar_size: 64/      sar_size: 192/" "$CFG"
export TM_SEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for g in $GSDS; do
  csv="$OUTBASE/whu_${g}m/cross_eval.csv"
  n=$(($(wc -l < "$csv") - 1))
  echo "---- eval B @ ${g}m (n=$n) ----"
  CUDA_VISIBLE_DEVICES=0 $PY -u Source/main.py \
      --mode test --batchSize 8 --gpu 1 --port 8931 \
      --trainListPath "$csv" --auto_save_num 1 --dataloaderNum 2 \
      --imgNum "$n" --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
      --dist False --modelName TM_Model_B --dataset tmdataset \
      --is_stage_2 True --fine_dim 32 --fine_margin 4 --sar_size 192 \
      --log ./log_test_debug/ --outputDir "./DebugResult_whu_${g}m_B/" \
      --resultImgDir ./ResultImg_test_debug/ --modelDir ./Checkpoint_B_stage2/ \
      2>&1 | tail -2
done

# ---- Phase 3: summary curve ----
echo "######## WHU RESOLUTION SWEEP — B@256/192 zero-shot (trained@10m) ########"
for g in $GSDS; do
  c=$(ls -t DebugResult_whu_${g}m_B/*.csv 2>/dev/null | head -1)
  echo "=== GSD ${g}m  ($c) ==="
  $PY Scripts/pck_summary.py "$c" 2>/dev/null | grep -E "n=|PCK@1px|PCK@3px|PCK@5px|HIGH-conf"
done
echo "==== WHU RES SWEEP DONE rc=$? $(date) ===="
