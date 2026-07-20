#!/usr/bin/env bash
# Sequential background generation: test split first (fast, validates),
# then train split. Survives ssh/network drop via nohup.
# 顺序后台生成：先 test（快、验证）再 train。nohup 抗断线。
set -u
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
GEN="$PROJ/Scripts/make_opt256_sar192_shards.py"
cd "$PROJ" || exit 1
echo "==== GEN START $(date) ===="
echo "---- TEST split ----"
$PY "$GEN" --split test  --workers 16 && \
echo "---- TRAIN split ----" && \
$PY "$GEN" --split train --workers 16
echo "==== GEN DONE rc=$? $(date) ===="
