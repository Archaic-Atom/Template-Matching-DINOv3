#!/usr/bin/env bash
# Regenerate WHU-OPT-SAR + QXS-SAROPT cross-eval at the corrected 256/192
# geometry (deletes the old 64/128 outputs first). nohup-safe.
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
ADP="$PROJ/Source/Tools/cross_dataset_adapter.py"
cd "$PROJ" || exit 1
echo "==== CROSS REGEN 256/192 START $(date) ===="
# delete old 64/128 cross_eval (raw datasets are NOT touched)
rm -rf /data5/datasets/cross_eval/whu_opt_sar /data5/datasets/cross_eval/qxs_saropt
echo "---- WHU-OPT-SAR (cross-sensor, 10m) ----"
$PY "$ADP" --dataset whu_opt_sar \
    --raw_dir /data5/datasets/whu-opt-sar \
    --out_dir /data5/datasets/cross_eval/whu_opt_sar \
    --samples_per_scene 64
echo "---- QXS-SAROPT (cross-resolution, 1m) ----"
$PY "$ADP" --dataset qxs_saropt \
    --raw_dir /data5/datasets/QXSLAB_SAROPT \
    --out_dir /data5/datasets/cross_eval/qxs_saropt \
    --samples_per_scene 1 --max_scenes 2000
echo "==== CROSS REGEN DONE rc=$? $(date) ===="
