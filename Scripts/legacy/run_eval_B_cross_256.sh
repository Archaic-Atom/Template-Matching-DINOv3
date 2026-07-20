#!/usr/bin/env bash
# Zero-shot eval of B (dim256, ep1, trained@128/64) on the 256/192 cross-domain
# sets (WHU cross-sensor, QXS cross-resolution). Temporarily sets B yaml
# sar_size=192; a trap restores it on ANY exit (incl. crash) so B's config
# never stays dirty.
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
CFG="$PROJ/Source/UserModelImplementation/Models/TM_Model_B/config.yaml"
cd "$PROJ" || exit 1
cp "$CFG" /tmp/B_cfg_cross_backup.yaml
trap 'cp /tmp/B_cfg_cross_backup.yaml "$CFG"; echo "[trap] B config restored"' EXIT
sed -i "s/      sar_size: 64/      sar_size: 192/" "$CFG"
grep -n "sar_size" "$CFG"
export TM_SEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_one () {  # $1=tag $2=csv $3=imgNum $4=port
  echo "==== EVAL B@256/192 cross:$1 START $(date) ===="
  CUDA_VISIBLE_DEVICES=0 $PY -u Source/main.py \
      --mode test --batchSize 8 --gpu 1 --port "$4" \
      --trainListPath "$2" \
      --auto_save_num 1 --dataloaderNum 2 \
      --imgNum "$3" --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
      --dist False --modelName TM_Model_B --dataset tmdataset \
      --is_stage_2 True --fine_dim 32 --fine_margin 4 --sar_size 192 \
      --log ./log_test_debug/ --outputDir "./DebugResult_${1}_B/" \
      --resultImgDir ./ResultImg_test_debug/ \
      --modelDir ./Checkpoint_B_stage2/
  echo "==== EVAL cross:$1 rc=$? $(date) ===="
}

run_one whu /data5/datasets/cross_eval/whu_opt_sar/cross_eval.csv 6400 8923
run_one qxs /data5/datasets/cross_eval/qxs_saropt/cross_eval.csv 2000 8924
echo "==== ALL CROSS EVAL DONE $(date) ===="
