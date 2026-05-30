#!/bin/bash
# Run test mode on an arbitrary cross-dataset CSV with caller-provided
# modelDir, total imgNum, output label, and GPU.
#
# 在跨数据集 CSV 上跑 JackFramework test 模式；调用者传入 modelDir、
# 样本数 imgNum、输出 label、GPU id。
#
# Usage:
#   bash Scripts/start_test_cross.sh <model_dir> <csv_path> <img_num> \
#        <out_label> [gpu]
#
# Output CSV: DebugResult/test_<out_label>.csv
set -euo pipefail
MODEL_DIR="${1:?usage: $0 <model_dir> <csv_path> <img_num> <out_label> [gpu]}"
CSV_PATH="${2:?usage: $0 <model_dir> <csv_path> <img_num> <out_label> [gpu]}"
IMG_NUM="${3:?usage: $0 <model_dir> <csv_path> <img_num> <out_label> [gpu]}"
OUT_LABEL="${4:?usage: $0 <model_dir> <csv_path> <img_num> <out_label> [gpu]}"
GPU="${5:-0}"

mkdir -p DebugResult log_test_debug ResultImg_test_debug
# Clear any prior dataloader-emitted temp/result file so the rename below
# picks up only this run's output.
# 清理上次留下的 dataloader 输出，避免误 cp 别人的 CSV。
rm -f DebugResult/test_tmdataset_*.csv \
      DebugResult/.tmp_test_tmdataset_rank0.csv \
      DebugResult/output.log

# Port 8893 chosen to avoid clashing with auto-epoch daemon's 8889.
# 端口 8893 避免与 daemon 的 8889 撞车。
CUDA_VISIBLE_DEVICES="$GPU" python -u Source/main.py \
    --mode test --batchSize 8 --gpu 1 --port 8893 \
    --trainListPath "$CSV_PATH" \
    --auto_save_num 1 --dataloaderNum 2 \
    --imgNum "$IMG_NUM" --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
    --dist False --modelName TM_Model_A --dataset tmdataset \
    --is_stage_2 True --matching_dim 128 \
    --match_w4 1.0 --match_w8 0.5 --match_sigma4 1.0 --match_sigma8 1.0 \
    --match_primary_stride 4 --fine_dim 32 --fine_margin 4 --sar_size 64 \
    --match_w_fine 1.0 --match_sigma_fine 1.0 --fine_warmup_epochs 3 \
    --log ./log_test_debug/ --outputDir ./DebugResult/ \
    --resultImgDir ./ResultImg_test_debug/ \
    --modelDir "$MODEL_DIR"

# Rename dataloader's timestamped output to our deterministic label.
# 把 dataloader 自动命名的 timestamped CSV 改成可读 label 命名。
if ls DebugResult/test_tmdataset_*.csv >/dev/null 2>&1; then
    cp DebugResult/test_tmdataset_*.csv "DebugResult/test_${OUT_LABEL}.csv"
    echo "Output CSV: DebugResult/test_${OUT_LABEL}.csv"
else
    echo "ERROR: no DebugResult/test_tmdataset_*.csv emitted" >&2
    exit 1
fi
