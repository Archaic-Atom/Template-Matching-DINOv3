#!/usr/bin/env bash
# Cross-dataset eval prep: turn the three external SAR-optical datasets into
# SEN1-2-shaped patch pairs (SAR 64x64, RGB 128x128, label (x,y)) ready for
# zero-shot evaluation by the trained DOSR-Net.
#
# 跨数据集评测准备：把三个公开 SAR-optical 数据集切成 SEN1-2 同形状的
# patch 对，供训练好的 DOSR-Net zero-shot 评测使用。
#
# Pipeline per dataset:
#   raw scenes  →  resample (per-dataset target_res_m)  →  random crop
#                  64x64 SAR + 128x128 RGB with ±8 px GT  →  PNGs + manifest.pt
#
# Per-dataset choices:
#   * whu_opt_sar  native 5 m/px  → resample to 10 m  (matches SEN1-2)
#   * qxs_saropt   native 1 m/px  → keep 1 m (cross-resolution test;
#                                   256x256 patches are too small to
#                                   downsample to 10 m)
#   * sarptical    native 1 m/px  → resample to 10 m  (matches SEN1-2)
#
# Output layout:
#   ${OUT_BASE}/<dataset>/sar/00000000.png  (64x64 L)
#   ${OUT_BASE}/<dataset>/rgb/00000000.png  (128x128 RGB)
#   ${OUT_BASE}/<dataset>/manifest.pt        ({sar_path, rgb_path, label})
#   ${OUT_BASE}/<dataset>/meta.json          (provenance)
#
# After this, optionally pack each manifest.pt into WebDataset shards via
# Scripts/pack_to_webdataset.py.

set -e
set -u

# Resolve the project root so the script works regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

# ----------------------------------------------------------------------------
# Configuration — override via env vars before invoking the script.
#
#   RAW_BASE     parent of WHU-OPT-SAR/, QXS-SAROPT/, SARptical/
#   OUT_BASE     where this script writes <dataset>/sar+rgb+manifest
#   CONDA_ENV    optional, conda env to activate (default: stereoA)
#
# Example:
#   RAW_BASE=/data4/datasets OUT_BASE=/data4/datasets/cross_eval \
#       bash Scripts/build_cross_dataset_shards.sh
# ----------------------------------------------------------------------------
RAW_BASE="${RAW_BASE:-/data4/datasets}"
OUT_BASE="${OUT_BASE:-/data4/datasets/cross_eval}"
CONDA_ENV="${CONDA_ENV:-stereoA}"

# Per-dataset settings: (raw_subdir, samples_per_scene)
declare -A SUBDIR=(
    [whu_opt_sar]="WHU-OPT-SAR"
    [qxs_saropt]="QXS-SAROPT"
    [sarptical]="SARptical"
)
declare -A SAMPLES=(
    [whu_opt_sar]=64    # 100 scenes  x 64  = 6400 pairs
    [qxs_saropt]=4      # 20000 pairs x 4   = 80000 pairs
    [sarptical]=8       # 10000 pairs x 8   = 80000 pairs
)

ADAPTER_PY="Source/Tools/cross_dataset_adapter.py"

# Activate conda env if present (the remote box has tifffile and PIL inside
# `stereoA`).
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" || echo "warning: conda activate ${CONDA_ENV} failed; using current env"
fi

echo "================================================================"
echo "  build_cross_dataset_shards.sh"
echo "  project root : ${PROJ_ROOT}"
echo "  raw base     : ${RAW_BASE}"
echo "  out base     : ${OUT_BASE}"
echo "  adapter      : ${ADAPTER_PY}"
echo "================================================================"

mkdir -p "${OUT_BASE}"

for ds in whu_opt_sar qxs_saropt sarptical; do
    raw_dir="${RAW_BASE}/${SUBDIR[$ds]}"
    out_dir="${OUT_BASE}/${ds}"
    samples="${SAMPLES[$ds]}"

    echo ""
    echo "----------------------------------------------------------------"
    echo "[${ds}]  raw=${raw_dir}  out=${out_dir}  samples_per_scene=${samples}"
    echo "----------------------------------------------------------------"

    if [ ! -d "${raw_dir}" ]; then
        echo "[${ds}] SKIP — raw_dir not found: ${raw_dir}"
        continue
    fi

    mkdir -p "${out_dir}"

    t0=$(date +%s)
    python "${ADAPTER_PY}" \
        --dataset "${ds}" \
        --raw_dir "${raw_dir}" \
        --out_dir "${out_dir}" \
        --samples_per_scene "${samples}"
    t1=$(date +%s)
    echo "[${ds}] done in $((t1 - t0))s"
done

echo ""
echo "================================================================"
echo "  All cross-dataset shards prepared under ${OUT_BASE}"
echo "  Next step: pack each manifest.pt into WebDataset shards via"
echo "             Scripts/pack_to_webdataset.py if you want iterable"
echo "             dataloading at eval time."
echo "================================================================"
