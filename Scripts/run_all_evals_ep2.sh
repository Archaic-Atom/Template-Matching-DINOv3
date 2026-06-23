#!/usr/bin/env bash
# Run all 4 eval suites on B model_epoch_2 (128/64-trained), sequentially.
# 在 B 的 epoch_2 上串行跑 4 套评测。SHARED config.yaml sar_size 决定必须串行
# (native=64, new-geom=192)。结果写到全新的 DebugResult_ep2_* 目录,不覆盖
# Jun-4 的 epoch_1 结果。汇总写 EVAL_EP2_SUMMARY.txt,完整日志 log_alleval_ep2.log。
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
CFG="$PROJ/Source/UserModelImplementation/Models/TM_Model_B/config.yaml"
SUMM="$PROJ/EVAL_EP2_SUMMARY.txt"
FULLLOG="$PROJ/log_alleval_ep2.log"
cd "$PROJ" || exit 1
export TM_SEED=2024 CUBLAS_WORKSPACE_CONFIG=:4096:8
: > "$SUMM"; : > "$FULLLOG"
log(){ echo "$@" | tee -a "$SUMM"; }
log "==== ALL-EVAL ep2 START $(date) ===="
log "checkpoint.list: $(tr '\n' ' ' < Checkpoint_B_stage2/checkpoint.list)"

# Restore B config sar_size on ANY exit (incl. crash) so it never stays dirty.
cp "$CFG" /tmp/Bcfg_alleval_bk.yaml
trap 'cp /tmp/Bcfg_alleval_bk.yaml "$CFG"; echo "[trap] B cfg restored"' EXIT
# Start from sar_size 64 (no-op if already 64).
sed -i "s/      sar_size: 192/      sar_size: 64/" "$CFG"

# $1=tag $2=csv $3=imgNum $4=port $5=outdir $6=sar_size
run_eval(){
  log "### [$1]  sar_size=$6  n=$3 ###"
  rm -rf "$5"; mkdir -p "$5"
  CUDA_VISIBLE_DEVICES=0 $PY -u Source/main.py \
    --mode test --batchSize 8 --gpu 1 --port "$4" \
    --trainListPath "$2" --auto_save_num 1 --dataloaderNum 2 \
    --imgNum "$3" --sampleNum 1 --maxEpochs 1 --lr 1e-3 --dist False \
    --modelName TM_Model_B --dataset tmdataset --is_stage_2 True \
    --fine_dim 32 --fine_margin 4 --sar_size "$6" \
    --log ./log_test_debug/ --outputDir "$5" \
    --resultImgDir ./ResultImg_test_debug/ --modelDir ./Checkpoint_B_stage2/ \
    >> "$FULLLOG" 2>&1
  local csv; csv=$(ls -t "$5"/*.csv 2>/dev/null | head -1)
  log "[$1] csv=$csv"
  if [ -n "$csv" ]; then $PY Scripts/pck_summary.py "$csv" 2>/dev/null | tee -a "$SUMM"; else log "[$1] NO CSV — eval failed, see $FULLLOG"; fi
}

# ---- 1) native 128/64 2k (sar_size 64) ----
run_eval 2k_128   ./Datasets/image_test_2k.csv 2000 8889 ./DebugResult_ep2_2k128 64

# ---- switch to 192 for all new-geometry evals ----
sed -i "s/      sar_size: 64/      sar_size: 192/" "$CFG"

# ---- 2) in-domain 256/192 10k ----
run_eval 256_10k  /data4/datasets/eval10k_opt256_sar192/image_test_2k_256.csv 10000 8907 ./DebugResult_ep2_256_10k 192

# ---- 3) cross-domain (WHU cross-sensor + QXS cross-resolution) ----
run_eval cross_whu /data5/datasets/cross_eval/whu_opt_sar/cross_eval.csv 6400 8923 ./DebugResult_ep2_cross_whu 192
run_eval cross_qxs /data5/datasets/cross_eval/qxs_saropt/cross_eval.csv 2000 8924 ./DebugResult_ep2_cross_qxs 192

# ---- 4) GSD sweep (WHU @ 5/8/10/16/24m). Reuse Jun-4 regen data if present. ----
p=8951
for g in 5 8 10 16 24; do
  RD=/data5/datasets/cross_eval_resolution/whu_${g}m
  if [ ! -f "$RD/cross_eval.csv" ]; then
    log "[gsd_${g}m] regenerating WHU @ ${g}m ..."
    $PY Source/Tools/cross_dataset_adapter.py --dataset whu_opt_sar \
      --raw_dir /data5/datasets/whu-opt-sar --out_dir "$RD" \
      --target_res_m "$g" --samples_per_scene 32 >> "$FULLLOG" 2>&1
  else
    log "[gsd_${g}m] reusing existing regen $RD"
  fi
  if [ -f "$RD/cross_eval.csv" ]; then
    n=$(($(wc -l < "$RD/cross_eval.csv") - 1))
    run_eval gsd_${g}m "$RD/cross_eval.csv" "$n" "$p" ./DebugResult_ep2_gsd_${g}m 192
  else
    log "[gsd_${g}m] regen FAILED — no cross_eval.csv, see $FULLLOG"
  fi
  p=$((p+1))
done

log "==== ALL-EVAL ep2 DONE $(date) ===="
