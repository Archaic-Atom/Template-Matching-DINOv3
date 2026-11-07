#!/usr/bin/env bash
# Shared configuration and helpers for every launcher under Scripts/.
# Scripts/ 下所有启动脚本共用的配置与助手函数。
#
# Source it from a launcher (works from any working directory):
# 在启动脚本里 source 它（在任何工作目录下都可用）：
#
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
#
# Nothing here is hard-coded to one machine: every value can be overridden from
# the environment, so you should not need to edit this file.
# 这里没有任何写死到某台机器的值——所有变量都可用环境变量覆盖，正常情况下无需修改本文件。
#
#     TM_PY=/path/to/python \
#     TM_EVAL_CSV=/data/my_eval.csv \
#     bash Scripts/reproduce/eval_cascade.sh Checkpoint_C_phase2_8gpu/model_epoch_2.pth
set -u

# --- project root & interpreter ------------------------------------------
# Derived from THIS file's location, so a clone at any path just works.
# bash exposes BASH_SOURCE; zsh does not, so fall back to $0 to stay sourceable
# from an interactive zsh as well.
# 由本文件位置推导，因此仓库克隆到任何路径都能直接跑。bash 有 BASH_SOURCE，zsh 没有，
# 因此回退到 $0，保证在交互式 zsh 里 source 也不报错。
if [ -n "${BASH_SOURCE:-}" ]; then
    _tm_self="${BASH_SOURCE[0]}"
else
    _tm_self="$0"
fi
TM_PROJ="${TM_PROJ:-$(cd "$(dirname "$_tm_self")/.." && pwd)}"
TM_PY="${TM_PY:-python}"

cd "$TM_PROJ" || { echo "ERROR: project root not found: $TM_PROJ" >&2; exit 1; }

# --- datasets (NOT in the repo — see README "Data preparation") -----------
# 数据集不在仓库内，生成方式见 README「数据准备」。
TM_EVAL_CSV="${TM_EVAL_CSV:-/data4/datasets/eval2k_opt256_sar192/image_test_2k_256.csv}"
TM_EVAL_N="${TM_EVAL_N:-2000}"
TM_CROSS_WHU="${TM_CROSS_WHU:-/data4/datasets/cross_eval/whu_opt_sar/cross_eval.csv}"
TM_CROSS_QXS="${TM_CROSS_QXS:-/data4/datasets/cross_eval/qxs_saropt/cross_eval.csv}"
TM_GSD_BASE="${TM_GSD_BASE:-/data4/datasets/cross_eval_resolution}"
TM_TRAIN_LIST="${TM_TRAIN_LIST:-./Datasets/shard_train_256_full.csv}"

# --- geometry (256/192, ME-SSM protocol) ---------------------------------
TM_SAR_SIZE="${TM_SAR_SIZE:-192}"
TM_MATCHING_DIM="${TM_MATCHING_DIM:-256}"

# --- determinism ----------------------------------------------------------
export TM_SEED="${TM_SEED:-2024}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# --------------------------------------------------------------------------
# tm_require_ckpt <path>
#   Fail early with a readable message instead of a mid-run traceback.
#   提前报错，避免跑到一半才抛 traceback。
# --------------------------------------------------------------------------
tm_require_ckpt() {
    local ckpt="${1:-}"
    if [ -z "$ckpt" ]; then
        echo "ERROR: no checkpoint given." >&2
        return 1
    fi
    if [ ! -e "$ckpt" ]; then
        echo "ERROR: checkpoint not found: $ckpt" >&2
        echo "       Download the released models (see README 'Pretrained models')" >&2
        echo "       and unzip them at the project root: $TM_PROJ" >&2
        return 1
    fi
    if [ -d "$ckpt" ] && [ ! -f "$ckpt/checkpoint.list" ]; then
        echo "ERROR: '$ckpt' is a directory without checkpoint.list." >&2
        echo "       Point at the .pth file directly instead, e.g." >&2
        echo "       $ckpt/model_epoch_0.pth" >&2
        return 1
    fi
    return 0
}

# --------------------------------------------------------------------------
# tm_eval <modelName> <ckpt> <tag> [gpu] [eval_csv] [eval_n] [batch] [port]
#
#   Runs `--mode test` and leaves predictions in DebugResult_<tag>/.
#   `<ckpt>` may be a .pth FILE or a checkpoint directory holding a
#   checkpoint.list — JackFramework accepts both for --modelDir.
#   跑 `--mode test`，预测结果写入 DebugResult_<tag>/。<ckpt> 可以是 .pth 文件，
#   也可以是含 checkpoint.list 的目录——JackFramework 两种都接受。
# --------------------------------------------------------------------------
tm_eval() {
    local model="${1:?usage: tm_eval <modelName> <ckpt> <tag> [gpu] [csv] [n] [batch] [port]}"
    local ckpt="${2:?usage: tm_eval <modelName> <ckpt> <tag> [gpu] [csv] [n] [batch] [port]}"
    local tag="${3:?usage: tm_eval <modelName> <ckpt> <tag> [gpu] [csv] [n] [batch] [port]}"
    local gpu="${4:-0}"
    local csv="${5:-$TM_EVAL_CSV}"
    local num="${6:-$TM_EVAL_N}"
    local batch="${7:-1}"
    local port="${8:-8890}"

    tm_require_ckpt "$ckpt" || return 1
    if [ ! -f "$csv" ]; then
        echo "ERROR: eval manifest not found: $csv" >&2
        echo "       Override it with TM_EVAL_CSV=/path/to/list.csv" >&2
        return 1
    fi

    local out_dir="DebugResult_${tag}"
    rm -rf "$out_dir"; mkdir -p "$out_dir" log_eval ResultImg_eval

    echo "==== EVAL ${model} [${tag}] START $(date '+%F %T') (gpu ${gpu}, n=${num}) ===="
    echo "     ckpt: ${ckpt}"
    CUDA_VISIBLE_DEVICES="$gpu" "$TM_PY" -u Source/main.py \
        --root_path "$TM_PROJ" \
        --mode test --dist False --gpu 1 --port "$port" \
        --modelName "$model" --dataset tmdataset --is_stage_2 True \
        --trainListPath "$csv" --imgNum "$num" \
        --batchSize "$batch" --sampleNum 1 --maxEpochs 1 --lr 1e-3 \
        --dataloaderNum 2 --auto_save_num 1 \
        --sar_size "$TM_SAR_SIZE" --matching_dim "$TM_MATCHING_DIM" \
        --fine_dim 32 --fine_margin 4 \
        --log ./log_eval/ --outputDir "./${out_dir}/" \
        --resultImgDir ./ResultImg_eval/ \
        --modelDir "$ckpt"
    local rc=$?
    echo "==== EVAL ${model} [${tag}] DONE rc=${rc} $(date '+%F %T') ===="
    return $rc
}

# --------------------------------------------------------------------------
# tm_train <modelName> <manifest> <ckpt_dir> <epochs> <tag> [gpus] [batch] [port]
#
#   Launches DDP training under nohup so it survives an SSH drop, and prints the
#   PID + log path. `--imgNum` is the manifest's real sample total (JackFramework
#   uses it as the progress denominator).
#   用 nohup 起 DDP 训练（断开 SSH 也不会死），打印 PID 与日志路径。`--imgNum` 取清单
#   的真实样本总数（JackFramework 用它作进度分母）。
#
#   Host-RAM knob: `TM_DATALOADER_NUM` (default 8, unchanged) sets
#   `--dataloaderNum`. Host RAM ~= gpus x dataloaderNum x shuffle_buffer, so
#   concurrent runs must lower it — see Scripts/train/train_2x2.sh.
#   主机内存旋钮：`TM_DATALOADER_NUM`（默认 8，与原行为一致）控制 `--dataloaderNum`。
#   主机内存 ≈ gpus × dataloaderNum × shuffle_buffer，多臂并发时必须调小，
#   见 Scripts/train/train_2x2.sh。
# --------------------------------------------------------------------------
tm_train() {
    local model="${1:?usage: tm_train <modelName> <manifest> <ckpt_dir> <epochs> <tag> [gpus] [batch] [port]}"
    local manifest="${2:?manifest}"
    local ckpt_dir="${3:?ckpt_dir}"
    local epochs="${4:?epochs}"
    local tag="${5:?tag}"
    local gpus="${6:-4}"
    local batch="${7:-10}"
    local port="${8:-8866}"

    if [ ! -f "$manifest" ]; then
        echo "ERROR: shard manifest not found: $manifest" >&2
        echo "       Generate the shards first (README 'Data preparation')," >&2
        echo "       or override with TM_TRAIN_LIST=/path/manifest.csv" >&2
        return 1
    fi
    if [ -d "$ckpt_dir" ] && find "$ckpt_dir" -maxdepth 1 -name 'model_epoch_*.pth' | grep -q .; then
        echo "ERROR: $ckpt_dir already holds checkpoints — move it aside first." >&2
        return 1
    fi
    mkdir -p "$ckpt_dir"

    # real sample total across the shard manifest (header + path,file_num rows)
    local real_total
    real_total=$(tail -n +2 "$manifest" | awk -F, '{s+=$2} END{print s}')

    # TM_DEVICES pins the PHYSICAL GPUs (comma list). The default keeps the old
    # behaviour of taking the first `gpus` devices, which is silently wrong the
    # moment another job already holds GPU 0 -- the run then fights an existing
    # job for memory instead of using the idle cards.
    # TM_DEVICES 指定物理 GPU（逗号分隔）。默认沿用"取前 gpus 张"的旧行为，但只要
    # GPU 0 已被别的任务占着，这个默认就是错的：新任务会去和旧任务抢显存，而不是
    # 用空闲卡。
    local devices
    devices="${TM_DEVICES:-$(seq -s, 0 $((gpus - 1)))}"
    local n_dev
    n_dev=$(printf '%s' "$devices" | tr ',' '\n' | grep -c .)
    if [ "$n_dev" -ne "$gpus" ]; then
        echo "ERROR: TM_DEVICES lists ${n_dev} device(s) (${devices}) but gpus=${gpus}." >&2
        echo "       DDP would spawn ${gpus} ranks onto ${n_dev} visible GPUs and hang." >&2
        return 1
    fi

    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    echo "==== TRAIN ${model} [${tag}] gpus=${gpus} bs=${batch} epochs=${epochs} ===="
    echo "     manifest=${manifest} samples=${real_total} ckpt=${ckpt_dir}"
    CUDA_VISIBLE_DEVICES="$devices" JF_PROGRESS_COLUMNS=400 nohup "$TM_PY" -u Source/main.py \
        --root_path "$TM_PROJ" \
        --mode train --dataset tmdataset --modelName "$model" --is_stage_2 True \
        --dataloaderType iterable --dataloaderNum "${TM_DATALOADER_NUM:-8}" --sampleNum 1 \
        --sar_size "$TM_SAR_SIZE" --matching_dim "$TM_MATCHING_DIM" \
        --shuffle_buffer "${TM_SHUFFLE_BUFFER:-30000}" --lr "${TM_LR:-1e-3}" \
        --auto_save_num 1 --dist True --gpu "$gpus" --batchSize "$batch" --port "$port" \
        --trainListPath "$manifest" --imgNum "$real_total" --maxEpochs "$epochs" \
        --log "./log_${tag}/" --outputDir "./result_${tag}/" \
        --resultImgDir "./ResultImg_${tag}/" \
        --modelDir "./${ckpt_dir}/" > "train_${tag}.log" 2>&1 &
    echo "     PID=$!  log=train_${tag}.log"
    echo "     follow: tail -f train_${tag}.log"
}

# --------------------------------------------------------------------------
# tm_pck <tag>
#   Print CMR/PCK for the predictions written by tm_eval.
#   对 tm_eval 产出的预测计算 CMR/PCK。
#
#   Two robustness notes learned from real runs / 两点来自实跑的健壮性处理:
#   * `--mode test` occasionally hangs in the finalize step that merges the
#     per-rank temp CSVs into `test_<dataset>_<stamp>.csv`. No data is lost —
#     every prediction is already in `.tmp_test_*_rank0.csv`, so fall back to it.
#     `--mode test` 偶尔会卡在把各 rank 临时 CSV 合并成 test_*.csv 的收尾步骤。
#     数据不会丢——预测都已写入 .tmp_test_*_rank0.csv，因此回退读它。
#   * pck_summary.py sits in Source/Tools/ here but in Scripts/ on older
#     deployments; look in both.
#     pck_summary.py 在本仓库位于 Source/Tools/，旧部署里在 Scripts/，两处都找。
# --------------------------------------------------------------------------
tm_pck() {
    local tag="${1:?usage: tm_pck <tag>}"
    local out_dir="DebugResult_${tag}"
    local csv pck

    # `find` rather than a glob: zsh aborts on an unmatched glob, bash does not.
    # 用 find 而非 glob：zsh 遇到不匹配的 glob 会直接报错，bash 不会。
    csv=$(find "$out_dir" -maxdepth 1 -type f -name 'test_*.csv' -exec ls -t {} + 2>/dev/null | head -1)
    if [ -z "$csv" ]; then
        csv=$(find "$out_dir" -maxdepth 1 -type f -name '.tmp_test_*.csv' -exec ls -t {} + 2>/dev/null | head -1)
        [ -n "$csv" ] && echo "NOTE: merged csv absent (finalize did not complete); using per-rank temp csv." >&2
    fi
    if [ -z "$csv" ]; then
        echo "WARN: no prediction csv in ${out_dir}/ — did the eval fail?" >&2
        return 1
    fi

    pck=$(ls Source/Tools/pck_summary.py Scripts/pck_summary.py 2>/dev/null | head -1)
    if [ -z "$pck" ]; then
        echo "WARN: pck_summary.py not found (looked in Source/Tools/ and Scripts/)." >&2
        echo "      Predictions are in ${csv}" >&2
        return 1
    fi

    echo "==== PCK [${tag}] : ${csv} ===="
    "$TM_PY" "$pck" "$csv"
}
