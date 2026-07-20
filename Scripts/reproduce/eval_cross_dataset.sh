#!/usr/bin/env bash
# Zero-shot cross-dataset evaluation of the RELEASED cascade (no retraining).
# 用发布的 cascade 做跨数据集零样本评测（不重新训练）。
#
#   Table III — WHU-OPT-SAR (cross-sensor, 10 m) and QXS-SAROPT (cross-resolution, 1 m)
#   表 III —— WHU-OPT-SAR（跨传感器 10m）与 QXS-SAROPT（跨分辨率 1m）
#
# Usage / 用法:
#   bash Scripts/reproduce/eval_cross_dataset.sh [ckpt] [gpu] [dataset]
#     dataset  whu | qxs ; omit to run both / 取 whu 或 qxs，不填则两个都跑
#
# The cross-dataset CSVs are built by Source/Tools/cross_dataset_adapter.py;
# point at them with TM_CROSS_WHU / TM_CROSS_QXS if they live elsewhere.
# 跨数据集 CSV 由 Source/Tools/cross_dataset_adapter.py 生成；路径不同就用
# TM_CROSS_WHU / TM_CROSS_QXS 覆盖。
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

CKPT="${1:-Checkpoint_C_phase2_8gpu/model_epoch_2.pth}"
GPU="${2:-0}"
ONLY="${3:-all}"

run_one() {  # <name> <csv> <n> <reference>
    local name="$1" csv="$2" num="$3" reference="$4"
    [ "$ONLY" != "all" ] && [ "$ONLY" != "$name" ] && return 0

    echo ""
    echo "################ ${name}  (paper CMR(3) = ${reference}) ################"
    if [ ! -f "$csv" ]; then
        echo "SKIP ${name}: eval csv not found: $csv" >&2
        echo "     build it with Source/Tools/cross_dataset_adapter.py, or override the path" >&2
        return 0
    fi
    tm_eval TM_Model_C "$CKPT" "repro_x_${name}" "$GPU" "$csv" "$num" || return 0
    tm_pck "repro_x_${name}"
}

run_one whu "$TM_CROSS_WHU" "${TM_CROSS_WHU_N:-6400}" "74.39 (high-conf 96.12, gross 4.64)"
run_one qxs "$TM_CROSS_QXS" "${TM_CROSS_QXS_N:-2000}" "10.40 (gross 31.35)"

cat <<'NOTE'

Reading the result / 结果怎么读:
  WHU (cross-sensor, same 10 m scale) transfers well on the confident half.
  WHU（跨传感器、同为 10m）在高置信半区迁移良好。
  QXS (1 m, never resampled) collapses — resolution, not sensor, is the binding
  constraint. This is a reported limitation, not a bug.
  QXS（1m，刻意不重采样）大幅退化——制约因素是分辨率而非传感器。这是论文中报告的局限，
  不是 bug。
NOTE
