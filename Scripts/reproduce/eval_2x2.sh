#!/usr/bin/env bash
# Score every finished arm of the REVISED 2x2 campaign on the 20k in-domain
# subsample, and emit the CSVs the two statistics tools consume.
# 在 20k 域内子采样上评测修订版 2x2 实验中每个已跑完的臂，产出两个统计工具要吃的 CSV。
#
#   arms (one seed = four arms) / 一个 seed = 四个臂:
#     S1/Tr  TM_Model_G  Checkpoint_G_s1tr_256_s<seed>   Stage-1 init, trainable
#     R/Tr   TM_Model_G  Checkpoint_G_rtr_256_s<seed>    random init,  trainable
#     S1/Fr  TM_Model_F  Checkpoint_F_s1fr_256_s<seed>   Stage-1 init, frozen
#     R/Fr   TM_Model_F  Checkpoint_F_rfr_256_s<seed>    random init,  frozen
#
#   These directory names, the arm tags and the final-epoch index all come from
#   Scripts/train/train_2x2.sh — that script is the producer, this one is the
#   consumer, so if you rename anything there, rename it here too.
#   目录名、arm tag、末代 epoch 编号都取自 Scripts/train/train_2x2.sh —— 那边是生产者，
#   这边是消费者；那边改名，这边必须同步改。
#
# ==========================================================================
# WHY THIS EXISTS / 为什么要有这个脚本
# ==========================================================================
# Reviewer 2, comment 5(b) asks for multiple independent runs per condition with
# mean, SD and confidence intervals, and the round-3 panel added that the test
# templates are NOT independent: the 20k rows are 7,740 held-out optical/SAR
# pairs x a 289-offset grid, and the pairs come from SEN1-2 scenes. So the
# statistics need TWO things this script must produce cleanly:
#   (a) one prediction CSV per (seed, arm), all on the SAME manifest, for the
#       across-seed mean/SD/CI  -> multiseed_summary.py
#   (b) prediction CSVs that still carry `rgb_name`, so the bootstrap can cluster
#       by pair and by scene instead of resampling templates i.i.d.
#       -> cluster_bootstrap.py
# An i.i.d. template bootstrap is what produced the implausible P(delta>0)=1.00
# in the submitted paper; do not go back to it.
# 审稿人 2 的 5(b) 要多次独立训练并报告均值/标准差/置信区间；第三轮内部评审补充：测试模板
# 并不独立（20k = 7740 对留出图像 × 289 个偏移网格，图像对又来自 SEN1-2 场景）。因此本脚本
# 必须干净地产出两样东西：(a) 每个 (seed, arm) 一份、全部在同一清单上的预测 CSV；
# (b) 保留 `rgb_name` 列的预测 CSV，让 bootstrap 能按 pair / scene 聚类重采样。
# 提交版里 P(delta>0)=1.00 就是 i.i.d. 模板 bootstrap 的产物，别再退回去。
#
# The prediction CSV schema (verified against a real DebugResult CSV in this
# repo, not assumed) is:
# 预测 CSV 的列（对着仓库里真实的 DebugResult CSV 核过，不是猜的）：
#     index,sar_name,rgb_name,gt_x,gt_y,pred_x,pred_y,score
#     0,20_44_ROIs1970_fall_s1_53_p415.png,ROIs1970_fall_s2_53_p415.png,20,44,...
# so the pair key is `rgb_name` and the scene key is its prefix up to the patch
# field: ROIs1970_fall_s2_53_p415.png -> ROIs1970_fall_s2_53.
# 因此 pair 键就是 `rgb_name`，scene 键是它去掉 _p<patch> 后的前缀。
#
# ==========================================================================
# ONE ARM AT A TIME, ON A GPU TRAINING IS NOT USING / 一次一个臂，且只用训练没占的卡
# ==========================================================================
# The box is training the campaign concurrently (~49 h per arm, sequential).
# This script therefore takes exactly ONE idle GPU per eval, re-queried at each
# arm's start, and REFUSES rather than contends when nothing is idle: a stolen
# GPU does not just slow the eval, it slows or OOMs a 49-hour training arm that
# cannot be resumed as the same seed (brief_experiments 7.4).
# 机器正在并发跑训练（每臂约 49 小时，串行）。本脚本每次评测只取一张空闲卡，且在每个臂
# 开始时**重新**查询；没有空闲卡就拒绝执行而不是抢占——抢卡不只是把评测拖慢，还可能拖慢
# 甚至撑爆一个跑了 49 小时、且无法以同一 seed 续训的训练臂。
#
# Usage / 用法:
#   bash Scripts/reproduce/eval_2x2.sh <seeds> [gpu] [arm] [--force] [--dry-run]
#
#     seeds  comma- or space-separated, e.g. 2024 or 2024,2025,2026
#            逗号或空格分隔的 seed 列表
#     gpu    device index, or `auto` (default) = take the first idle device
#            设备号，或 `auto`（默认）= 取第一张空闲卡
#     arm    s1tr | rtr | s1fr | rfr | all (default)
#     --force      re-evaluate arms that already have a prediction CSV, and
#                  override the GPU / port / freshness refusals
#                  重新评测已有预测 CSV 的臂，并越过 GPU/端口/新鲜度这几道闸
#     --dry-run    print the plan and the follow-up commands, run nothing
#                  只打印计划与后续命令，不实际跑
#
# Environment / 环境变量:
#   TM_EVAL_CSV          eval manifest; default is the 20k one, NOT common.sh's 2k
#                        评测清单；默认用 20k 那份，而不是 common.sh 的 2k
#   TM_EVAL_N            --imgNum, the progress-bar denominator (default 20000)
#   TM_2X2_TAG_SUFFIX    tag suffix, default `20k` — change it with the manifest
#   TM_2X2_EVAL_BATCH    --batchSize, default 2
#   TM_2X2_PORT_BASE     first DDP rendezvous port to try, default 8991
#   TM_2X2_GPU_MEM_MIB   a GPU counts as idle below this used-MiB, default 1024
#   TM_2X2_FINAL_EPOCH   epoch index to score, default 4 (= the 5th epoch)
#   TM_2X2_FORCE         same as --force
#   TM_2X2_DRY_RUN       same as --dry-run
#   TM_PY                interpreter (honoured by common.sh)
#
# Wall clock: ~1-1.5 h per arm at batch 2 on a 3090, so a full five-seed
# campaign is 20 arms ~= 25 h of one GPU, spread over whatever idle windows the
# training leaves. Re-run it as often as you like; it skips what it already did.
# 墙钟：3090 上 batch 2 每臂约 1~1.5 小时，五个 seed 共 20 臂 ≈ 单卡 25 小时，可以在训练
# 留下的空闲窗口里分批跑。重复运行是安全的：已经跑过的会跳过。

# --------------------------------------------------------------------------
# Capture the caller's manifest choice BEFORE sourcing common.sh, which assigns
# TM_EVAL_CSV/TM_EVAL_N its own 2k defaults. Without this we could not tell
# "the user asked for the 2k set" from "the user asked for nothing", and the 2x2
# would silently be scored on 2,000 templates instead of 20,000 — a tenfold loss
# of resolution in exactly the CI the reviewer is asking about.
# 在 source common.sh **之前**记住调用者的选择：common.sh 会把 TM_EVAL_CSV/TM_EVAL_N
# 填成它自己的 2k 默认值，届时就分不清"用户要 2k"和"用户什么都没说"了，2x2 会被悄悄地在
# 2000 条模板上评测——而那正是审稿人问的那个置信区间，分辨率直接掉十倍。
# --------------------------------------------------------------------------
_req_eval_csv="${TM_EVAL_CSV:-}"
_req_eval_n="${TM_EVAL_N:-}"
_req_tag_suffix="${TM_2X2_TAG_SUFFIX:-}"

# Absolute path to THIS file, resolved before common.sh cd's to the project root
# (--help reads the header back out of it).
# 本文件的绝对路径，在 common.sh cd 到项目根之前解析好（--help 要回读文件头）。
_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

# --------------------------------------------------------------------------
# Argument parsing. Flags may appear anywhere; positionals keep their order.
# 参数解析：flag 可以出现在任何位置，位置参数按顺序。
# --------------------------------------------------------------------------
FORCE="${TM_2X2_FORCE:-0}"
DRY_RUN="${TM_2X2_DRY_RUN:-0}"
SEEDS_RAW=""
GPU_ARG=""
ARM_ONLY=""
_pos=0
for _a in "$@"; do
    case "$_a" in
        --force)   FORCE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            # Print the leading comment block, stopping at the first line of code.
            # 打印文件头注释块，遇到第一行代码即停。
            awk 'NR > 1 && !/^#/ { exit } { sub(/^#[ ]?/, ""); print }' "$_self"
            exit 0
            ;;
        -*)
            echo "ERROR: unknown flag '$_a'." >&2
            echo "       usage: $0 <seeds> [gpu] [arm] [--force] [--dry-run]" >&2
            exit 1
            ;;
        *)
            case "$_pos" in
                0) SEEDS_RAW="$_a" ;;
                1) GPU_ARG="$_a" ;;
                2) ARM_ONLY="$_a" ;;
                *) echo "ERROR: too many positional arguments ('$_a')." >&2; exit 1 ;;
            esac
            _pos=$((_pos + 1))
            ;;
    esac
done
GPU_ARG="${GPU_ARG:-auto}"
ARM_ONLY="${ARM_ONLY:-all}"

if [ -z "$SEEDS_RAW" ]; then
    echo "ERROR: no seed list." >&2
    echo "       usage: $0 <seeds> [gpu] [arm] [--force] [--dry-run]" >&2
    echo "       e.g.:  $0 2024              # one replicate" >&2
    echo "              $0 2024,2025,2026 auto all" >&2
    exit 1
fi
SEEDS=$(printf '%s' "$SEEDS_RAW" | tr ',' ' ')
for s in $SEEDS; do
    case "$s" in
        ''|*[!0-9]*)
            echo "ERROR: seed '$s' is not a number." >&2
            exit 1
            ;;
    esac
done

case "$ARM_ONLY" in
    all|s1tr|rtr|s1fr|rfr) ;;
    *)
        echo "ERROR: arm must be one of s1tr|rtr|s1fr|rfr|all (got '$ARM_ONLY')." >&2
        exit 1
        ;;
esac

# --------------------------------------------------------------------------
# Configuration.
# --------------------------------------------------------------------------
FINAL_EPOCH="${TM_2X2_FINAL_EPOCH:-4}"
BATCH="${TM_2X2_EVAL_BATCH:-2}"
PORT_BASE="${TM_2X2_PORT_BASE:-8991}"
PORT_SPAN=8                       # ports scanned from PORT_BASE / 从 PORT_BASE 起扫描的端口数
GPU_MEM_MIB="${TM_2X2_GPU_MEM_MIB:-1024}"
FRESH_SECONDS=120                 # a ckpt younger than this may still be being written
INDEX_CSV="${TM_2X2_INDEX_CSV:-eval_2x2_index.csv}"

DEFAULT_20K_CSV=/data4/datasets/eval20k_opt256_sar192/image_test_2k_256.csv
EVAL_CSV="${_req_eval_csv:-$DEFAULT_20K_CSV}"
EVAL_N="${_req_eval_n:-20000}"
TAG_SUFFIX="${_req_tag_suffix:-20k}"

for _v in FINAL_EPOCH BATCH PORT_BASE GPU_MEM_MIB EVAL_N; do
    eval "_val=\${$_v}"
    case "$_val" in
        ''|*[!0-9]*) echo "ERROR: ${_v} must be a non-negative integer (got '${_val}')." >&2; exit 1 ;;
    esac
done

# The tag suffix is what keeps two eval sets from colliding in one DebugResult
# namespace. If the manifest was overridden but the suffix was not, the CSVs
# would land under `..._20k/` while holding something else entirely, and the
# statistics tools glob by tag — they would silently mix eval sets across seeds.
# tag 后缀是两套评测集在 DebugResult 命名空间里不互相覆盖的唯一保证。若换了清单却没换
# 后缀，CSV 会落在 `..._20k/` 里却装着别的东西；统计工具按 tag 通配，就会在不同 seed 之间
# 悄悄混用评测集。
#
# The common case — pointing at the 20k manifest where it actually lives on this
# box — is NOT a mismatch, so only refuse when the path gives no sign of holding
# the same eval set the suffix claims.
# 常见情况（把清单指到本机 20k 的真实路径）不算不一致，因此只有当路径里看不出它就是
# 后缀所声称的那套评测集时才拒绝。
if [ -n "$_req_eval_csv" ] && [ -z "$_req_tag_suffix" ]; then
    case "$_req_eval_csv" in
        *"$TAG_SUFFIX"*)
            echo "NOTE: manifest overridden; path mentions '${TAG_SUFFIX}', keeping that suffix."
            ;;
        *)
            echo "WARN: TM_EVAL_CSV was overridden but TM_2X2_TAG_SUFFIX was not." >&2
            echo "      Tags would still say '_${TAG_SUFFIX}' while holding a different eval set," >&2
            echo "      and the statistics tools glob by tag — they would mix eval sets." >&2
            echo "      Set an honest suffix, e.g. TM_2X2_TAG_SUFFIX=2k." >&2
            echo "      换了评测清单却没换 tag 后缀：tag 仍写着 _${TAG_SUFFIX}，内容却不是；" >&2
            echo "      统计工具按 tag 通配，会把两套评测集混在一起。请设一个诚实的后缀。" >&2
            [ "$FORCE" = "1" ] || exit 1
            echo "WARN: --force — continuing with a misleading tag. Disclose it." >&2
            ;;
    esac
fi

if [ ! -f "$EVAL_CSV" ]; then
    echo "ERROR: eval manifest not found: $EVAL_CSV" >&2
    echo "       Override it with TM_EVAL_CSV=/path/to/list.csv (and set" >&2
    echo "       TM_2X2_TAG_SUFFIX to match)." >&2
    echo "       评测清单不存在；用 TM_EVAL_CSV 指定，并同步改 TM_2X2_TAG_SUFFIX。" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Port safety.
#
# JackFramework binds --port for the DDP rendezvous even at --dist False, and a
# collision shows up as a HANG, not an error — the worst possible failure mode
# for a script meant to run unattended beside a 49-hour training arm. Two
# sources of collision have to be excluded before we even look at `ss`:
#
#   1. The training campaign. train_2x2.sh computes PORT_BASE = 8900 +
#      (seed % 10) * 10 + 1 and its four arms take PORT_BASE..PORT_BASE+3, so
#      seed 2024 -> 8941..8944, 2025 -> 8951..8954, ... 2029 -> 8991..8994.
#      Note that last one: the eval default 8991 collides with a seed whose last
#      digit is 9. We derive the blocks from the checkpoint dirs actually on
#      disk, so a replicate that exists is protected whether or not it is in the
#      seed list handed to this script.
#   2. Ports hard-coded in Scripts/legacy/* and probes_local/eval_arms_20k.sh
#      (that one owns 8961..8964 — the seed-2026 training block, a collision
#      already flagged in train_2x2.sh's header).
#
# 端口安全。JF 即使 --dist False 也会绑定 --port 做 DDP 会合，撞端口的表现是**挂起**而不是
# 报错——对一个要在 49 小时训练臂旁边无人值守运行的脚本，这是最糟的失败模式。因此在查 `ss`
# 之前先排除两类冲突：(1) 训练活动的端口块（由磁盘上真实存在的 checkpoint 目录反推，
# 注意 seed 末位为 9 时正好落在 8991）；(2) Scripts/legacy/* 与 probes_local/eval_arms_20k.sh
# 里写死的端口（后者占 8961..8964）。
# --------------------------------------------------------------------------
LEGACY_PORTS="8800 8814 8817 8820 8821 8830 8831 8832 8833 8852 8853 8866 8867
8868 8869 8871 8872 8887 8889 8893 8894 8895 8899 8900 8907 8911 8931
8961 8962 8963 8964"

# Training blocks of every replicate whose checkpoint dirs exist, plus every
# seed on this command line (a replicate about to start still owns its ports).
# 磁盘上存在的每个重复的训练端口块，加上命令行里的每个 seed（即将开跑的重复也占着端口）。
training_seeds() {
    local d s
    for d in Checkpoint_G_s1tr_256_s* Checkpoint_G_rtr_256_s* \
             Checkpoint_F_s1fr_256_s* Checkpoint_F_rfr_256_s*; do
        [ -d "$d" ] || continue
        s="${d##*_s}"
        case "$s" in ''|*[!0-9]*) continue ;; esac
        echo "$s"
    done
    echo $SEEDS
}

RESERVED_PORTS="$LEGACY_PORTS"
for _s in $(training_seeds | sort -u); do
    _base=$((8900 + (_s % 10) * 10 + 1))
    RESERVED_PORTS="$RESERVED_PORTS $_base $((_base + 1)) $((_base + 2)) $((_base + 3))"
done
RESERVED_PORTS=$(printf '%s\n' $RESERVED_PORTS | sort -un | tr '\n' ' ')

port_reserved() {   # <port> -> 0 if it belongs to training or a legacy script
    case " $RESERVED_PORTS " in *" $1 "*) return 0 ;; esac
    return 1
}

port_free() {       # <port> -> 0 if nothing is listening (or ss is unavailable)
    command -v ss >/dev/null 2>&1 || return 0
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${1}\$" && return 1
    return 0
}

# pick_port <what> -> a port from the scan window that is neither reserved nor
# bound, or fail. Re-queried per arm: evals span hours and a neighbour binds.
# 取一个既不在保留集、也没被监听的端口；每个臂重新取（评测跨小时，邻居随时会占）。
pick_port() {
    local what="$1" p i
    for i in $(seq 0 $((PORT_SPAN - 1))); do
        p=$((PORT_BASE + i))
        [ "$p" -gt 65535 ] && break
        port_reserved "$p" && continue
        port_free "$p" || continue
        echo "$p"
        return 0
    done
    echo "ERROR: no free port in ${PORT_BASE}..$((PORT_BASE + PORT_SPAN - 1)) [${what}]." >&2
    echo "       reserved (training + legacy): ${RESERVED_PORTS}" >&2
    echo "       Pick another window with TM_2X2_PORT_BASE=<port>." >&2
    echo "       该窗口内没有可用端口；用 TM_2X2_PORT_BASE 换一个窗口。" >&2
    return 1
}

if [ $((PORT_BASE + PORT_SPAN - 1)) -gt 65535 ] || [ "$PORT_BASE" -lt 1024 ]; then
    echo "ERROR: TM_2X2_PORT_BASE must be in 1024..$((65535 - PORT_SPAN + 1))" >&2
    echo "       (got ${PORT_BASE}; the scan window is ${PORT_SPAN} ports wide)." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# GPU helpers — the same free-device rule train_2x2.sh uses, so "free" means the
# same thing on both sides of the campaign.
# GPU 助手 —— 与 train_2x2.sh 用同一条空闲判据，保证"空闲"在训练侧与评测侧含义一致。
# --------------------------------------------------------------------------
show_gpu_state() {
    echo "  --- nvidia-smi (index, used MiB, total MiB, util %) ---"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    echo "  --- compute processes currently holding GPUs ---"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    echo "    (empty above = no compute apps reported)"
}

free_gpus() {   # -> space-separated indices of idle devices, right now
    command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return 0; }
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null |
        awk -F', *' -v thr="$GPU_MEM_MIB" '$2+0 <= thr {print $1}' | tr '\n' ' '
}

# pick_gpu <what> -> one device index, or fail. Honours the `gpu` positional.
# Refusing here is the point: see the ONE ARM AT A TIME block in the header.
# 取一张卡；拿不到就拒绝——见文件头"一次一个臂"那段，这正是本脚本存在的理由之一。
pick_gpu() {
    local what="$1" free g
    free=" $(free_gpus) "
    if [ "$GPU_ARG" != "auto" ]; then
        case "$GPU_ARG" in
            ''|*[!0-9]*)
                echo "ERROR: gpu must be a device index or 'auto' (got '${GPU_ARG}')." >&2
                return 1
                ;;
        esac
        if command -v nvidia-smi >/dev/null 2>&1; then
            case "$free" in
                *" $GPU_ARG "*) ;;
                *)
                    echo "ERROR: GPU ${GPU_ARG} is busy (>${GPU_MEM_MIB} MiB used) [${what}]." >&2
                    show_gpu_state >&2
                    echo "       Training is using it. Wait, pass another index, or 'auto'." >&2
                    echo "       训练正在用这张卡：等一等、换一张、或改用 auto。" >&2
                    [ "$FORCE" = "1" ] || return 1
                    echo "WARN: --force — evaluating on a busy GPU anyway." >&2
                    ;;
            esac
        fi
        echo "$GPU_ARG"
        return 0
    fi
    g=$(printf '%s' "$free" | tr ' ' '\n' | grep -v '^$' | head -1)
    # Only reachable once the operator has already accepted the blind-run gate
    # above: without nvidia-smi there is no free list to pick from, and handing
    # back an empty CUDA_VISIBLE_DEVICES makes torch see zero devices and die
    # with a device-count error unrelated to the real cause.
    # 只有操作者已经越过上面那道盲跑闸才会走到这里：没有 nvidia-smi 就没有空闲卡列表，
    # 而返回空的 CUDA_VISIBLE_DEVICES 会让 torch 看到 0 张卡，报一个与真实原因无关的错。
    if [ -z "$g" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "WARN: no nvidia-smi — falling back to device 0 blind [${what}]." >&2
        echo "0"
        return 0
    fi
    if [ -z "$g" ]; then
        echo "ERROR: no idle GPU (<= ${GPU_MEM_MIB} MiB used) [${what}]." >&2
        show_gpu_state >&2
        echo "       REFUSING rather than contending: the box is training a 49-hour" >&2
        echo "       arm that cannot be resumed as the same seed if it dies." >&2
        echo "       Arms already scored are safe — re-run when a GPU frees up and" >&2
        echo "       this script will pick up where it stopped." >&2
        echo "       宁可拒绝也不抢卡：机器上那个 49 小时的训练臂死了就无法以同一 seed 续训。" >&2
        echo "       已评完的臂不受影响，等有空闲卡再重跑本脚本即可接着往下走。" >&2
        return 1
    fi
    echo "$g"
    return 0
}

# --------------------------------------------------------------------------
# Prediction-CSV discovery — the same two-place lookup tm_pck does, because the
# `--mode test` finalize step that merges the per-rank temp CSVs can lag minutes
# behind the progress bar. A `.tmp_test_*_rank0.csv` holds every prediction and
# is a perfectly good input to the statistics; it just means finalize had not
# run yet. (brief_experiments 7.5)
# 预测 CSV 的定位 —— 与 tm_pck 同样查两处：`--mode test` 把各 rank 临时 CSV 合并成
# test_*.csv 的收尾步骤可能比进度条慢好几分钟。`.tmp_test_*_rank0.csv` 里预测是全的，
# 拿去做统计完全没问题，只是说明还没收尾。
# --------------------------------------------------------------------------
find_pred_csv() {   # <tag> -> path, or empty
    local out_dir="DebugResult_${1}" csv
    csv=$(find "$out_dir" -maxdepth 1 -type f -name 'test_*.csv' \
          -exec ls -t {} + 2>/dev/null | head -1)
    if [ -z "$csv" ]; then
        csv=$(find "$out_dir" -maxdepth 1 -type f -name '.tmp_test_*.csv' \
              -exec ls -t {} + 2>/dev/null | head -1)
    fi
    printf '%s' "$csv"
}

file_age_seconds() {   # <path> -> seconds since mtime, or 999999 if unknowable
    local f="$1" m now
    now=$(date +%s)
    m=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || echo "")
    case "$m" in
        ''|*[!0-9]*) echo 999999; return 0 ;;
    esac
    echo $((now - m))
}

# --------------------------------------------------------------------------
# Arm table. Index into the plan is seed_index * 4 + arm_index, so everything
# stays in plain indexed arrays — bash 3.2 has no associative arrays and the
# remote box is not guaranteed to be newer.
# 臂表。计划的下标 = seed 序号 * 4 + arm 序号，全部用普通索引数组：bash 3.2 没有关联数组，
# 远端机器的 bash 版本也不保证够新。
# --------------------------------------------------------------------------
A_KEY=(  "s1tr"       "rtr"        "s1fr"       "rfr"      )
A_NAME=( "S1/Tr"      "R/Tr"       "S1/Fr"      "R/Fr"     )
A_MODEL=("TM_Model_G" "TM_Model_G" "TM_Model_F" "TM_Model_F")
A_PREFIX=("Checkpoint_G_s1tr_256_s" "Checkpoint_G_rtr_256_s" \
          "Checkpoint_F_s1fr_256_s" "Checkpoint_F_rfr_256_s")

P_SEED=(); P_ARM=(); P_TAG=(); P_CKPT=(); P_STATE=(); P_NOTE=(); P_CSV=()
N_PLAN=0
N_TODO=0
N_SKIP=0
N_ABSENT=0

for seed in $SEEDS; do
    for a in 0 1 2 3; do
        [ "$ARM_ONLY" != "all" ] && [ "$ARM_ONLY" != "${A_KEY[$a]}" ] && continue
        ck_dir="${A_PREFIX[$a]}${seed}"
        ck="${ck_dir}/model_epoch_${FINAL_EPOCH}.pth"
        tag="x2_${A_KEY[$a]}_s${seed}_${TAG_SUFFIX}"
        state=""; note=""; csv=""

        if [ ! -d "$ck_dir" ]; then
            state="NO-DIR"
            note="${ck_dir} does not exist — arm never launched"
        elif [ ! -f "$ck" ]; then
            state="UNFINISHED"
            have=$(find "$ck_dir" -maxdepth 1 -name 'model_epoch_*.pth' \
                   -exec basename {} \; 2>/dev/null | sort | tr '\n' ' ')
            note="no model_epoch_${FINAL_EPOCH}.pth; has: ${have:-nothing}"
        elif [ ! -s "$ck" ]; then
            state="EMPTY-CKPT"
            note="model_epoch_${FINAL_EPOCH}.pth is zero bytes — a failed save"
        else
            csv=$(find_pred_csv "$tag")
            if [ -n "$csv" ] && [ "$FORCE" != "1" ]; then
                state="SCORED"
                note="$csv"
            else
                age=$(file_age_seconds "$ck")
                if [ "$age" -lt "$FRESH_SECONDS" ] && [ "$FORCE" != "1" ]; then
                    state="TOO-FRESH"
                    note="ckpt written ${age}s ago — may still be being flushed"
                else
                    state="TODO"
                    note="$ck"
                    [ -n "$csv" ] && note="$ck  (--force: re-scoring over $csv)"
                fi
            fi
        fi

        P_SEED[$N_PLAN]="$seed"
        P_ARM[$N_PLAN]="$a"
        P_TAG[$N_PLAN]="$tag"
        P_CKPT[$N_PLAN]="$ck"
        P_STATE[$N_PLAN]="$state"
        P_NOTE[$N_PLAN]="$note"
        P_CSV[$N_PLAN]="$csv"
        N_PLAN=$((N_PLAN + 1))

        case "$state" in
            TODO)   N_TODO=$((N_TODO + 1)) ;;
            SCORED) N_SKIP=$((N_SKIP + 1)) ;;
            *)      N_ABSENT=$((N_ABSENT + 1)) ;;
        esac
    done
done

# --------------------------------------------------------------------------
# The plan, printed before anything runs.
# --------------------------------------------------------------------------
echo "======================================================================"
echo " 2x2 EVALUATION — seeds: ${SEEDS}"
echo "======================================================================"
echo "  project        : ${TM_PROJ}"
echo "  manifest       : ${EVAL_CSV}"
echo "  --imgNum       : ${EVAL_N}   (progress-bar denominator, not a sample cap)"
echo "  epoch scored   : model_epoch_${FINAL_EPOCH}.pth"
echo "  batch          : ${BATCH}   (inference is batch-invariant here: frozen"
echo "                   DINOv3 + GroupNorm, so bs2 == bs8 numbers)"
echo "  tag pattern    : x2_<arm>_s<seed>_${TAG_SUFFIX}  ->  DebugResult_<tag>/"
echo "  gpu            : ${GPU_ARG}   (one at a time; idle = <= ${GPU_MEM_MIB} MiB used)"
echo "  port window    : ${PORT_BASE}..$((PORT_BASE + PORT_SPAN - 1))"
echo "  ports reserved : ${RESERVED_PORTS}"
echo "  force          : ${FORCE}    dry-run: ${DRY_RUN}"
echo ""
printf '  %-6s %-6s %-11s %-11s %s\n' SEED ARM MODEL STATE DETAIL
for i in $(seq 0 $((N_PLAN - 1))); do
    a="${P_ARM[$i]}"
    printf '  %-6s %-6s %-11s %-11s %s\n' \
        "${P_SEED[$i]}" "${A_NAME[$a]}" "${A_MODEL[$a]}" \
        "${P_STATE[$i]}" "${P_NOTE[$i]}"
done
echo ""
echo "  ${N_TODO} to score, ${N_SKIP} already scored, ${N_ABSENT} not ready."

# Naming what is missing is a requirement, not a courtesy: a silently absent arm
# turns a 2x2 into a 2x1 and the summary tool would report a mean over three
# arms as if it were four.
# 明确报出缺哪个臂是硬要求：默默少一个臂会把 2x2 变成 2x1，而汇总工具会把三个臂的均值
# 当成四个臂的报出来。
if [ "$N_ABSENT" -gt 0 ]; then
    echo ""
    echo "  NOT READY / 尚未就绪 — these arms are NOT evaluated and NOT counted:"
    for i in $(seq 0 $((N_PLAN - 1))); do
        case "${P_STATE[$i]}" in TODO|SCORED) continue ;; esac
        a="${P_ARM[$i]}"
        echo "    seed ${P_SEED[$i]} ${A_NAME[$a]} [${P_STATE[$i]}] — ${P_NOTE[$i]}"
    done
    echo "    A seed is only a replicate when ALL FOUR arms are scored. Re-run this"
    echo "    script after the missing arms finish training."
    echo "    四个臂全部评完，这个 seed 才算一个完整重复；缺的臂训完后重跑本脚本。"
fi

if [ "$N_TODO" = "0" ]; then
    echo ""
    echo "  Nothing to score."
    [ "$N_SKIP" -gt 0 ] && echo "  (use --force to re-score the ${N_SKIP} arm(s) marked SCORED)"
fi

# --------------------------------------------------------------------------
# Pre-flight — refuse now rather than after the first arm.
# --------------------------------------------------------------------------
# A dry run is meant to be readable anywhere — on the Mac while planning the
# campaign, not just on the box — so the GPU gate is a note there, not a refusal.
# dry run 的用途是在任何地方（比如在 Mac 上排计划）都能读，因此那里 GPU 闸只提示不拒绝。
if [ "$N_TODO" -gt 0 ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "" >&2
    if [ "$DRY_RUN" = "1" ]; then
        echo "NOTE: no nvidia-smi here — the GPU checks are skipped in this dry run." >&2
    else
        echo "ERROR: nvidia-smi not found — cannot tell which GPU training is using." >&2
        echo "       Use --force to evaluate blind (not recommended on a shared box)." >&2
        echo "       没有 nvidia-smi 就无法判断哪张卡在训练；--force 可盲跑，但共享机器上不推荐。" >&2
        [ "$FORCE" = "1" ] || exit 1
    fi
fi
if [ "$N_TODO" -gt 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo ""
    PREFLIGHT_GPU=$(pick_gpu "pre-flight") || exit 1
    PREFLIGHT_PORT=$(pick_port "pre-flight") || exit 1
    echo "  pre-flight     : GPU ${PREFLIGHT_GPU} idle, port ${PREFLIGHT_PORT} free"
    echo "                   (both re-checked at EACH arm's start, not pinned now)"
fi

# --------------------------------------------------------------------------
# Run — strictly one arm at a time. tm_eval/tm_pck come from common.sh; the eval
# invocation is NOT re-implemented here, so the 2x2 arms are scored by exactly
# the command that produced the published numbers.
# 执行 —— 严格一次一个臂。tm_eval/tm_pck 来自 common.sh，这里不重新实现评测命令，
# 保证 2x2 各臂用的就是当初产出已发表数字的那条命令。
# --------------------------------------------------------------------------
N_OK=0
N_FAIL=0
if [ "$N_TODO" -gt 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo ""
    echo "---- scoring ---------------------------------------------------------"
    for i in $(seq 0 $((N_PLAN - 1))); do
        [ "${P_STATE[$i]}" = "TODO" ] || continue
        a="${P_ARM[$i]}"
        tag="${P_TAG[$i]}"

        echo ""
        echo "  == seed ${P_SEED[$i]} ${A_NAME[$a]} (${A_MODEL[$a]}) -> ${tag} =="

        # Re-query both at THIS arm's start: an eval campaign spans hours and the
        # training arm beside it starts and finishes in that window.
        # 在**本臂**开始时重新取卡取端口：评测跨小时，旁边的训练臂在这期间会起会停。
        if ! gpu=$(pick_gpu "seed ${P_SEED[$i]} ${A_NAME[$a]}"); then
            echo "       Stopping here. ${N_OK} arm(s) scored so far are safe." >&2
            echo "       就此停下；已评完的 ${N_OK} 个臂不受影响。" >&2
            break
        fi
        if ! port=$(pick_port "seed ${P_SEED[$i]} ${A_NAME[$a]}"); then
            echo "       Stopping here. ${N_OK} arm(s) scored so far are safe." >&2
            break
        fi
        echo "     gpu=${gpu}  port=${port}  ckpt=${P_CKPT[$i]}"

        if ! tm_eval "${A_MODEL[$a]}" "${P_CKPT[$i]}" "$tag" \
                     "$gpu" "$EVAL_CSV" "$EVAL_N" "$BATCH" "$port"; then
            echo "  FAIL seed ${P_SEED[$i]} ${A_NAME[$a]}: eval returned non-zero." >&2
            echo "       Look at log_eval/ and DebugResult_${tag}/ before re-running." >&2
            P_STATE[$i]="FAILED"
            P_NOTE[$i]="eval returned non-zero"
            N_FAIL=$((N_FAIL + 1))
            continue
        fi

        # Keep the CMR/PCK printout beside the predictions. The statistics tools
        # recompute everything from the CSV, so this file is for humans reading
        # the campaign log — not an input to the analysis.
        # 把 CMR/PCK 输出留在预测旁边。统计工具一律从 CSV 重算，这个文件是给人看的，
        # 不是分析的输入。
        summary="DebugResult_${tag}/pck_summary.txt"
        if tm_pck "$tag" > "$summary" 2>&1; then
            cat "$summary"
        else
            cat "$summary" >&2
            echo "  WARN seed ${P_SEED[$i]} ${A_NAME[$a]}: tm_pck failed; see ${summary}" >&2
        fi

        csv=$(find_pred_csv "$tag")
        if [ -z "$csv" ]; then
            echo "  FAIL seed ${P_SEED[$i]} ${A_NAME[$a]}: no prediction CSV in DebugResult_${tag}/" >&2
            P_STATE[$i]="FAILED"
            P_NOTE[$i]="eval exited 0 but wrote no prediction CSV"
            N_FAIL=$((N_FAIL + 1))
            continue
        fi
        P_CSV[$i]="$csv"
        P_STATE[$i]="SCORED"
        P_NOTE[$i]="$csv"
        N_OK=$((N_OK + 1))
    done
fi

# --------------------------------------------------------------------------
# Index CSV — one row per scored (seed, arm), merged across invocations so a
# campaign evaluated over several idle windows still ends up as one table.
# This is the handover to the statistics tools: if their flags ever drift from
# what this script prints below, everything they need is still in here.
# 索引 CSV —— 每个已评测的 (seed, arm) 一行，跨多次调用合并，因此分几个空闲窗口跑完的
# campaign 最后仍是一张表。这是交给统计工具的交接物：即使那两个工具的参数以后变了，
# 它们要的东西都在这个文件里。
# --------------------------------------------------------------------------
INDEX_HEADER="seed,arm,arm_name,model,tag,ckpt,pred_csv,eval_csv,eval_n"
if [ "$DRY_RUN" != "1" ]; then
    tmp_index="${INDEX_CSV}.tmp.$$"
    printf '%s\n' "$INDEX_HEADER" > "$tmp_index"
    # carry over rows for tags this run did not touch / 保留本次没碰到的 tag 的旧行
    if [ -f "$INDEX_CSV" ]; then
        tail -n +2 "$INDEX_CSV" 2>/dev/null | while IFS= read -r row; do
            [ -n "$row" ] || continue
            old_tag=$(printf '%s' "$row" | cut -d, -f5)
            keep=1
            for i in $(seq 0 $((N_PLAN - 1))); do
                [ "$old_tag" = "${P_TAG[$i]}" ] && { keep=0; break; }
            done
            [ "$keep" = "1" ] && printf '%s\n' "$row"
        done >> "$tmp_index"
    fi
    for i in $(seq 0 $((N_PLAN - 1))); do
        [ "${P_STATE[$i]}" = "SCORED" ] || continue
        a="${P_ARM[$i]}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${P_SEED[$i]}" "${A_KEY[$a]}" "${A_NAME[$a]}" "${A_MODEL[$a]}" \
            "${P_TAG[$i]}" "${P_CKPT[$i]}" "${P_CSV[$i]}" "$EVAL_CSV" "$EVAL_N" \
            >> "$tmp_index"
    done
    mv "$tmp_index" "$INDEX_CSV"
fi

# --------------------------------------------------------------------------
# Summary.
# --------------------------------------------------------------------------
echo ""
echo "======================================================================"
if [ "$DRY_RUN" = "1" ]; then
    echo " 2x2 EVALUATION — DRY RUN, nothing was executed"
else
    echo " 2x2 EVALUATION — ${N_OK} scored this run, ${N_FAIL} failed"
fi
echo "======================================================================"
printf '  %-6s %-6s %-11s %s\n' SEED ARM STATE PRED_CSV
for i in $(seq 0 $((N_PLAN - 1))); do
    a="${P_ARM[$i]}"
    printf '  %-6s %-6s %-11s %s\n' \
        "${P_SEED[$i]}" "${A_NAME[$a]}" "${P_STATE[$i]}" "${P_CSV[$i]:-—}"
done
[ "$DRY_RUN" != "1" ] && echo "" && echo "  index: ${INDEX_CSV}"

# Which seeds are complete replicates? Only those go into the across-seed
# statistics; a seed with three scored arms is not a replicate.
# 哪些 seed 是完整重复？只有完整的才进跨 seed 统计；三个臂的 seed 不算一个重复。
COMPLETE_SEEDS=""
INCOMPLETE_SEEDS=""
for seed in $SEEDS; do
    n_have=0
    for i in $(seq 0 $((N_PLAN - 1))); do
        [ "${P_SEED[$i]}" = "$seed" ] || continue
        [ "${P_STATE[$i]}" = "SCORED" ] || continue
        n_have=$((n_have + 1))
    done
    if [ "$ARM_ONLY" = "all" ] && [ "$n_have" = "4" ]; then
        COMPLETE_SEEDS="${COMPLETE_SEEDS}${COMPLETE_SEEDS:+ }${seed}"
    else
        INCOMPLETE_SEEDS="${INCOMPLETE_SEEDS}${INCOMPLETE_SEEDS:+ }${seed}"
    fi
done

echo ""
if [ -n "$COMPLETE_SEEDS" ]; then
    echo "  complete replicates (all 4 arms scored): ${COMPLETE_SEEDS}"
else
    echo "  complete replicates: none yet"
fi
[ -n "$INCOMPLETE_SEEDS" ] && \
    echo "  incomplete (do NOT feed to the across-seed statistics): ${INCOMPLETE_SEEDS}"

# --------------------------------------------------------------------------
# Hand-off — the exact commands for the two statistics tools.
#
# Both are separate units of this campaign and may not be deployed here yet, so
# resolve their paths the way tm_pck resolves pck_summary.py (Source/Tools/,
# then probes_local/, then Scripts/) and say plainly when one is missing rather
# than printing a command that cannot run.
# 交接 —— 两个统计工具的确切命令。它们是本 campaign 的另外两个单元，可能还没部署到这里，
# 因此按 tm_pck 找 pck_summary.py 的方式解析路径，找不到就直说，而不是打印一条跑不起来的命令。
# --------------------------------------------------------------------------
find_tool() {   # <basename> -> path if it exists, else empty
    local n="$1" p
    for p in "Source/Tools/${n}" "../../010_Remote_Claude/probes_local/${n}" \
             "probes_local/${n}" "Scripts/${n}"; do
        [ -f "$p" ] && { printf '%s' "$p"; return 0; }
    done
    printf ''
}
CLUSTER_TOOL=$(find_tool cluster_bootstrap.py)
MULTISEED_TOOL=$(find_tool multiseed_summary.py)
CLUSTER_SHOW="${CLUSTER_TOOL:-Source/Tools/cluster_bootstrap.py}"
MULTISEED_SHOW="${MULTISEED_TOOL:-Source/Tools/multiseed_summary.py}"

csv_for() {   # <seed> <arm_key> -> prediction CSV path, or empty
    local s="$1" k="$2" i a
    for i in $(seq 0 $((N_PLAN - 1))); do
        [ "${P_SEED[$i]}" = "$s" ] || continue
        a="${P_ARM[$i]}"
        [ "${A_KEY[$a]}" = "$k" ] || continue
        [ "${P_STATE[$i]}" = "SCORED" ] || continue
        printf '%s' "${P_CSV[$i]}"
        return 0
    done
    printf ''
}

echo ""
echo "======================================================================"
echo " NEXT — the two statistics tools / 下一步：两个统计工具"
echo "======================================================================"
[ -z "$CLUSTER_TOOL" ] && \
    echo "  NOTE: ${CLUSTER_SHOW} is not on this box yet (separate unit); deploy it first."
[ -z "$MULTISEED_TOOL" ] && \
    echo "  NOTE: ${MULTISEED_SHOW} is not on this box yet (separate unit); deploy it first."

echo ""
echo "  (1) Layer (b) — cluster bootstrap, per seed, on the three decisive"
echo "      contrasts. One invocation reports ALL THREE levels (template / pair /"
echo "      scene) on the same resampled templates, so there is nothing to rerun."
echo "      Scene-level is the HEADLINE (most conservative: templates share SEN1-2"
echo "      scenes); pair-level is the secondary line. Do NOT quote the template"
echo "      (i.i.d.) row — that is what produced P(delta>0)=1.00."
echo "      一次调用就同时给出三个层级；场景级是主报（最保守），pair 级作次要行；"
echo "      不要引用 template（i.i.d.）那一行。"
if [ -n "$COMPLETE_SEEDS" ]; then
    for seed in $COMPLETE_SEEDS; do
        c_s1tr=$(csv_for "$seed" s1tr)
        c_rtr=$(csv_for "$seed" rtr)
        c_s1fr=$(csv_for "$seed" s1fr)
        c_rfr=$(csv_for "$seed" rfr)
        echo ""
        echo "      # seed ${seed}   (arms are given as LABEL:CSV; Delta = first - second)"
        echo "      ${TM_PY} ${CLUSTER_SHOW} \\"
        echo "          'S1/Tr:${c_s1tr}' 'R/Tr:${c_rtr}' \\"
        echo "          --B 10000 --seed ${seed} --ks 1,3 \\"
        echo "          --out cluster_ci_s1tr_vs_rtr_s${seed}.csv"
        echo "      ${TM_PY} ${CLUSTER_SHOW} \\"
        echo "          'S1/Fr:${c_s1fr}' 'R/Fr:${c_rfr}' \\"
        echo "          --B 10000 --seed ${seed} --ks 1,3 \\"
        echo "          --out cluster_ci_s1fr_vs_rfr_s${seed}.csv"
        echo "      ${TM_PY} ${CLUSTER_SHOW} \\"
        echo "          'S1/Fr:${c_s1fr}' 'R/Tr:${c_rtr}' \\"
        echo "          --B 10000 --seed ${seed} --ks 1,3 \\"
        echo "          --out cluster_ci_s1fr_vs_rtr_s${seed}.csv"
    done
    echo ""
    echo "      # each run ends with a self-test line; if it says FAIL, the"
    echo "      # resampling is wrong and the intervals must not be quoted."
else
    echo ""
    echo "      (no complete replicate yet — nothing to contrast)"
fi

echo ""
echo "  (2) Layer (a) — across-seed mean / SD / t-based 95% CI per arm and per"
echo "      contrast. Feed it only the COMPLETE replicates."
echo "      跨 seed 的均值/标准差/t 分布 95% CI；只喂完整的重复。"
if [ -n "$COMPLETE_SEEDS" ]; then
    echo ""
    echo "      ${TM_PY} ${MULTISEED_SHOW} \\"
    for seed in $COMPLETE_SEEDS; do
        for k in s1tr rtr s1fr rfr; do
            echo "          ${k}:${seed}:$(csv_for "$seed" "$k") \\"
        done
    done
    echo "          --csv multiseed_2x2_${TAG_SUFFIX}.csv \\"
    echo "          --latex tab_two_by_two_${TAG_SUFFIX}.tex"
    echo ""
    echo "      # arm:seed:path is spelled out on purpose. '--root .' would also"
    echo "      # sweep in DebugResult_* dirs from other campaigns and from"
    echo "      # INCOMPLETE seeds; this run's index has the same mapping:"
    echo "      #   ${INDEX_CSV}"
else
    echo ""
    echo "      (needs >= 2 complete replicates before an SD means anything;"
    echo "       the plan calls for 5 seeds, staged 2024-2026 then 2027-2028)"
fi

cat <<NOTE

  (3) Layer (c) — say plainly which variance dominates. If the across-seed SD is
      much larger than the cluster-bootstrap SE, the CI quoted in the paper must
      be the across-seed one, and the honest sentence is "run-to-run variation,
      not test-set sampling, is the binding uncertainty".
      明确指出哪一层方差占主导：若跨 seed 的标准差远大于聚类 bootstrap 的标准误，
      论文里引用的置信区间必须是跨 seed 那个。

  Verify the scene parsing on ~20 rows before trusting any cluster CI. In these
  CSVs the pair key is the \`rgb_name\` column and the scene key is its prefix
  before the patch field:
  信任任何聚类 CI 之前，先在 ~20 行上核对场景解析。本 CSV 里 pair 键是 \`rgb_name\` 列，
  scene 键是它去掉 _p<patch> 之后的前缀：
      ROIs1970_fall_s2_53_p415.png  ->  pair  ROIs1970_fall_s2_53_p415.png
                                    ->  scene ROIs1970_fall_s2_53
  If a manifest ever loses that structure, fall back to pair-level clustering and
  say so in the response letter.
  若某份清单没有这个结构，退回 pair 级聚类，并在回复信里写明。
NOTE

if [ "$N_FAIL" -gt 0 ]; then
    echo ""
    echo "ERROR: ${N_FAIL} arm(s) failed to score — see the FAIL lines above." >&2
    exit 1
fi
exit 0
