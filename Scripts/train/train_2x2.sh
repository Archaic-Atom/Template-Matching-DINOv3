#!/usr/bin/env bash
# One replicate of the REVISED 2x2 pre-alignment experiment: four arms, one seed.
# 修订版 2x2 预对齐实验的一个重复：四臂，一个 seed。
#
#   S1/Fr = TM_Model_F, modelDir seeded from the shared Stage-1 ckpt
#   R/Fr  = TM_Model_F, empty modelDir  (random SAR trunk, then frozen)
#   S1/Tr = TM_Model_G, modelDir seeded from the SAME Stage-1 ckpt
#   R/Tr  = TM_Model_G, empty modelDir  (random SAR trunk, trainable)
#
# TM_Model_G is TM_Model_F with `training.stage_2.freeze_sar_encoder` flipped to
# false and NOTHING else (`diff` the two config.yaml to see it). So this is a
# strict factorial: one model family, ONE Stage-1 checkpoint, two binary factors
# (initialization x trainability). The published 2x2 was not — its trainable
# arms were TM_Model_A (asymmetric PatchNCE Stage-1) and TM_Model_B (random)
# while both frozen arms were TM_Model_F (symmetric learned-IP Stage-1), so
# "initialization" was crossed with "which Stage-1 recipe" (Reviewer 2, 5a).
# TM_Model_G 就是把 F 的 `freeze_sar_encoder` 改成 false，其余一字未动（diff 两个
# config.yaml 即可验证）。因此这是严格析因设计：同一模型族、同一份 Stage-1 权重、
# 两个二值因子。原论文的 2x2 不是——可训行用 A/B、冻结行用 F，初始化因子与
# Stage-1 配方混淆（审稿人 2 意见 5a）。
#
# ==========================================================================
# SEQUENTIAL BY DEFAULT / 默认串行
# ==========================================================================
# The arms run ONE AT A TIME, in the order S1/Tr, R/Tr, S1/Fr, R/Fr. Each arm
# is launched, gated, waited on, and verified before the next one starts. That
# costs ~9.4 days of wall clock per replicate instead of ~2.4 (4 x ~56 h; the
# 56 h already includes the +15% cuDNN-determinism tax this script always pays
# — see the WALL-CLOCK block for the arithmetic), and buys the thing
# that matters: one arm at a time FITS IN HOST RAM AT THE PUBLISHED RECIPE
# (--dataloaderNum 8, --shuffle_buffer 100000), so the new arms are
# recipe-identical to the published ones and the response letter needs no
# data-loader disclosure at all. See §RAM below for the arithmetic.
# NOT "no disclosure at all": this script always sets TM_SEED, which forces
# deterministic cuDNN, and the published arms ran unseeded. That one difference
# remains and the summary block prints it. 不等于"完全无需披露"：本脚本必设 TM_SEED，
# cuDNN 变确定性，而已发表各臂是未设 seed 的——这一条差异仍在，总结段会打印。
# 四个臂一次跑一个，顺序 S1/Tr → R/Tr → S1/Fr → R/Fr：启动、验闸、等待退出、
# 校验完成，然后才起下一个。代价是每个重复约 9.4 天而不是 2.4 天；换来的是——单臂
# 在主机内存里放得下**论文原配方**（--dataloaderNum 8、--shuffle_buffer 100000），
# 于是新跑的四臂与已发表各臂配方完全一致，回复信里无需再披露 dataloader 偏差。
#
# Concurrency is still one env var away (TM_2X2_CONCURRENT=1): four arms on
# 8 GPUs, ~2.4 days, but at the REDUCED data-loader recipe, which then MUST be
# disclosed. 并发仍然一个环境变量可达（TM_2X2_CONCURRENT=1）：8 卡四臂约 2 天，
# 但只能用缩水的 dataloader 配方，那就必须在回复信里披露。
#
# A sequential replicate blocks this shell for ~9.4 days. Run it inside tmux, or
# under `nohup bash Scripts/train/train_2x2.sh 2024 > 2x2_s2024.driver.log 2>&1 &`.
# 串行一个重复会占住当前 shell 约 9.4 天：请放进 tmux，或用 nohup 起。
#
# It is RESUMABLE: re-running it with the same seed skips every arm that
# already has its final-epoch checkpoint and continues at the first incomplete
# one. 可续跑：同一 seed 重跑会跳过已经产出末代 ckpt 的臂，从第一个未完成的臂继续。
#
# Usage / 用法:
#   bash Scripts/train/train_2x2.sh <seed> [gpus_per_arm] [batch] [port_base]
#     seed          e.g. 2024; it names every checkpoint dir, tag and log
#                   例如 2024；它会写进每个 ckpt 目录、tag 和日志名
#     gpus_per_arm  default 2 (published recipe: --gpu 2) / 默认 2
#     batch         per-GPU batch, default 32 (published recipe) / 每卡 batch
#     port_base     default 89<seed%10>1, so replicates never collide
#                   默认 89<seed%10>1，跨重复不会撞端口
#                   KNOWN COLLISION: seed 2026 maps to 8961..8964, which is the
#                   block Scripts/legacy/eval_arms_20k.sh still hard-codes
#                   (brief_experiments 7.3). If that eval is ever running,
#                   pass an explicit port_base for seed 2026, e.g.
#                       bash Scripts/train/train_2x2.sh 2026 2 32 8901
#                   已知冲突：seed 2026 落在 8961..8964，与 legacy 的 eval_arms_20k.sh
#                   写死的端口块重叠；那个 eval 若在跑，请给 2026 显式指定 port_base。
#                   Sequential mode re-checks the port at each arm's launch, so a
#                   collision surfaces as a readable refusal, not as a DDP hang.
#                   串行模式在每个臂启动时重查端口，撞了会明确报错而不是静默挂起。
#
#   bash Scripts/train/train_2x2.sh 2024                     # sequential, ~9.4 days
#   TM_2X2_DRY_RUN=1 bash Scripts/train/train_2x2.sh 2024    # print the plan only
#   TM_2X2_CONCURRENT=1 bash Scripts/train/train_2x2.sh 2024 # old 4-wide behaviour
#
# Environment overrides (all optional) / 环境变量覆盖（均可选）:
#   TM_2X2_CONCURRENT      1 = all four arms at once (needs 4 x gpus_per_arm
#                          GPUs and the reduced recipe); default 0 = sequential
#   TM_2X2_DRY_RUN         1 = run every pre-flight, print the plan, launch nothing
#   TM_2X2_GPUS            explicit device list. SEQUENTIAL: exactly
#                          gpus_per_arm devices, reused by every arm.
#                          CONCURRENT: exactly (arms still to run) x gpus_per_arm
#                          devices — on a resumed replicate that is fewer than 8.
#   TM_2X2_MANIFEST        shard manifest (default: subset-40)
#   TM_2X2_STAGE1          the shared Stage-1 checkpoint (.pth)
#   TM_2X2_EPOCHS          default 5
#   TM_DATALOADER_NUM      default 8 sequential / 4 concurrent — see §RAM
#   TM_SHUFFLE_BUFFER      default 100000 sequential / 30000 concurrent — see §RAM
#   TM_2X2_GPU_MEM_MIB     a GPU is "free" below this used-MiB, default 1024
#   TM_2X2_LOAD_TIMEOUT    seconds to wait for the init gate, default 1800
#   TM_2X2_KILL_ON_FAIL    1 (default) kills the running arm(s) on a gate failure
#   TM_2X2_POLL            seconds between liveness polls, default 300
#   TM_2X2_HEARTBEAT       seconds between progress heartbeats, default 3600
#   TM_2X2_SPS_PER_GPU     throughput used for the schedule, default 19
#                          (measured: simple head, 3090, 2 GPU x bs32)
#   TM_2X2_RESTART_PARTIAL 1 = archive a half-finished arm and retrain it clean
#                          (default 0 = stop and make the human decide)
#   TM_2X2_FORCE           1 bypasses the GPU / host-RAM / port refusals
#
# Prerequisite / 前置条件: JackFramework's DDP `find_unused_parameters=True`
# patch must be in place — the two F arms freeze the SAR trunk and all-reduce
# aborts with an EMPTY ProcessRaisedException without it.
# 必须已打上 JackFramework 的 `find_unused_parameters=True` 补丁——两个 F 臂冻结
# SAR trunk，没有该补丁 all-reduce 会以空 ProcessRaisedException 中止。
set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

# --- arguments ------------------------------------------------------------
SEED="${1:?usage: $0 <seed> [gpus_per_arm] [batch] [port_base]}"
# Validate BEFORE any arithmetic: `set -u` turns $((SEED % 10)) on a
# non-numeric seed into an unreadable "unbound variable" error.
# 先校验再做算术：seed 非数字时，`set -u` 下的 $((SEED % 10)) 会报成看不懂的
# "unbound variable"。
case "$SEED" in
    ''|*[!0-9]*) echo "ERROR: seed must be a positive integer (got '$SEED')" >&2; exit 1 ;;
esac

GPUS_PER_ARM="${2:-2}"
BATCH="${3:-32}"
PORT_BASE="${4:-$((8900 + (SEED % 10) * 10 + 1))}"

for _v in GPUS_PER_ARM BATCH PORT_BASE; do
    eval "_val=\$$_v"
    case "$_val" in
        ''|*[!0-9]*) echo "ERROR: ${_v} must be a positive integer (got '${_val}')" >&2; exit 1 ;;
    esac
done
if [ "$GPUS_PER_ARM" -lt 1 ]; then
    echo "ERROR: gpus_per_arm must be >= 1 (got ${GPUS_PER_ARM})" >&2
    exit 1
fi
if [ "$BATCH" -lt 1 ]; then
    echo "ERROR: batch must be >= 1 (got ${BATCH})" >&2
    exit 1
fi
# The arms take PORT_BASE..PORT_BASE+3. Out-of-range ports do not fail at
# argument parsing — DDP rendezvous just never completes and the arm HANGS,
# which the load gate would only catch 30 minutes later.
# 四个臂用 PORT_BASE..PORT_BASE+3。端口越界不会在解析时报错，而是 DDP 会合永远不成功、
# 该臂挂起，加载闸要 30 分钟后才发现。
if [ "$PORT_BASE" -lt 1024 ] || [ "$((PORT_BASE + 3))" -gt 65535 ]; then
    echo "ERROR: port_base must be in 1024..65532 (got ${PORT_BASE}; arms use" >&2
    echo "       ${PORT_BASE}..$((PORT_BASE + 3)))." >&2
    exit 1
fi

# common.sh exports TM_SEED=${TM_SEED:-2024}; this replicate's seed must win.
# common.sh 会导出 TM_SEED（默认 2024），这里必须用本次重复的 seed 覆盖它。
export TM_SEED="$SEED"

# Reproducibility + host-RAM environment for every arm.
# MALLOC_ARENA_MAX caps glibc's per-thread arenas: with N dataloader workers per
# rank the default arena growth alone costs tens of GiB, and it creeps rather
# than plateaus — it is what killed an earlier full-corpus run.
# 每个臂共用的复现 + 主机内存环境。MALLOC_ARENA_MAX 限制 glibc 每线程 arena：
# 每 rank N 个 worker 时，光 arena 增长就要吃掉几十 GiB，而且是持续爬升不是稳定
# 平台——之前搞死过一次全量训练。
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- mode -----------------------------------------------------------------
CONCURRENT="${TM_2X2_CONCURRENT:-0}"
case "$CONCURRENT" in
    0|1) ;;
    *) echo "ERROR: TM_2X2_CONCURRENT must be 0 or 1 (got '${CONCURRENT}')" >&2; exit 1 ;;
esac
DRY_RUN="${TM_2X2_DRY_RUN:-0}"
if [ "$CONCURRENT" = "1" ]; then MODE_NAME="CONCURRENT (4 arms at once)"
else MODE_NAME="SEQUENTIAL (one arm at a time)"; fi

N_ARMS=4

# --------------------------------------------------------------------------
# §RAM — the host-RAM law, and why the two modes get different recipes.
#
# Measured law (project_multi_server_workflow, and §7.1 of brief_experiments):
#   workers = concurrent_arms x gpus_per_arm x dataloaderNum
#   one worker holds one shuffle buffer; at --shuffle_buffer 100000 a worker
#   plateaus at ~8.5 GiB (~85 KiB/sample), and the cost is LINEAR in the buffer.
#
# Machine 2 has 251 GiB of RAM. Arithmetic, done here so it is auditable:
#
#   SEQUENTIAL, published recipe (dataloaderNum 8, shuffle_buffer 100000):
#       1 arm x 2 GPU x 8 workers = 16 workers
#       16 x 8.5 GiB              = 136 GiB          <= 251 GiB   FITS
#
#   CONCURRENT, published recipe:
#       4 arms x 2 GPU x 8 workers = 64 workers
#       64 x 8.5 GiB               = 544 GiB         >> 251 GiB   DIES
#
#   CONCURRENT, reduced recipe (dataloaderNum 4, shuffle_buffer 30000):
#       4 arms x 2 GPU x 4 workers = 32 workers
#       per worker: 8.5 x 30000/100000 = 2.55 GiB
#       32 x 2.55 GiB              = 82 GiB          <= 251 GiB   FITS
#
# 主机内存定律：worker 数 = 并发臂数 × 每臂 GPU 数 × dataloaderNum，每个 worker 持有
# 一个 shuffle buffer（buffer=100000 时约 8.5 GiB，开销与 buffer 线性相关）。
# 251 GiB 的机器上：串行按原配方 = 136 GiB（放得下）；并发按原配方 = 544 GiB（必死）；
# 并发按缩水配方 = 82 GiB（放得下）。
#
# *** DISCLOSURE CONSEQUENCE / 对回复信的影响 ***
#   SEQUENTIAL mode is RECIPE-IDENTICAL to the published arms
#   (Scripts/legacy/run_frozen_arms_F.sh: --dataloaderNum 8 --shuffle_buffer
#   100000, everything else already matched). Nothing about the data loader
#   needs to be disclosed, and absolute CMR is directly comparable to the
#   published Table III.
#   串行模式与已发表各臂的配方逐字一致（见 run_frozen_arms_F.sh），dataloader 无需
#   任何披露，绝对 CMR 可直接与已发表的 Table III 比较。
#
#   CONCURRENT mode DEVIATES (dataloaderNum 8->4, shuffle_buffer 100000->30000).
#   It changes shuffling quality, not the data, and it applies UNIFORMLY to all
#   four arms, so the internal 2x2 contrast stays clean — but absolute CMR may
#   drift a few tenths from the published Table III, and THAT MUST BE DISCLOSED
#   IN THE RESPONSE LETTER.
#   并发模式与原配方不同：改变的是打乱质量而非数据本身，且四臂一致，2x2 的内部对比
#   仍然干净——但绝对 CMR 相比已发表的 Table III 可能有零点几的漂移，**必须在回复信
#   里披露**。
#
# dataloaderNum 8 is not a throughput risk either way: at ~19 samples/s/GPU each
# worker needs to produce ~2.4 samples/s and a 256^2+192^2 JPEG decode is <5 ms.
# 两种模式下 dataloaderNum 都不是吞吐瓶颈：每 worker 只需 ~2.4 samples/s，
# 而一次 256²+192² 的 JPEG 解码远低于 5 ms。
# --------------------------------------------------------------------------
if [ "$CONCURRENT" = "1" ]; then
    export TM_DATALOADER_NUM="${TM_DATALOADER_NUM:-4}"
    export TM_SHUFFLE_BUFFER="${TM_SHUFFLE_BUFFER:-30000}"
else
    export TM_DATALOADER_NUM="${TM_DATALOADER_NUM:-8}"
    export TM_SHUFFLE_BUFFER="${TM_SHUFFLE_BUFFER:-100000}"
fi
GIB_PER_WORKER_AT_100K=8.5

# The published recipe, for the "do we need to disclose?" check below.
# 论文原配方，供下面"是否需要披露"的判断使用。
#
# SCOPE OF THIS CHECK — read it before quoting the verdict.
# It compares the two DATA-LOADER flags only. The rest of the flag list is
# literal-identical to run_frozen_arms_F.sh by construction (see launch_arm),
# but the ENVIRONMENT is not, and that is where the remaining deviation lives:
#
#   run_frozen_arms_F.sh is standalone — it does NOT source common.sh, so the
#   published arms ran with TM_SEED UNSET. Source/main.py:_maybe_fix_seed()
#   is therefore a no-op for them and cuDNN keeps benchmark=True (fast,
#   nondeterministic algorithm selection).
#   This script sets TM_SEED unconditionally — that is the entire point of a
#   seed replicate (E2) — so every arm here runs with
#   cudnn.deterministic=True / benchmark=False, plus common.sh's
#   CUBLAS_WORKSPACE_CONFIG. That is a genuine, deliberate difference from the
#   published arms. It is uniform across all four arms, so the 2x2's internal
#   contrast is untouched, but it is NOT "nothing to disclose": the response
#   letter needs one sentence about determinism instead of one about the loader.
#
# 本检查只比对两个 dataloader 参数。其余参数按构造与 run_frozen_arms_F.sh 逐字一致
# （见 launch_arm），但**环境变量不一致**，剩下的偏差就在这里：run_frozen_arms_F.sh
# 是独立脚本、不 source common.sh，因此已发表各臂 TM_SEED 未设、cudnn.benchmark=True；
# 本脚本必设 TM_SEED（seed 重复的意义所在），于是 cudnn 变成 deterministic。
# 这是真实且有意的差异，四臂一致因而不影响 2x2 内部对比，但**不等于"无需披露"**：
# 回复信里要写的是一句关于确定性的说明，而不是一句关于 dataloader 的说明。
PUB_DATALOADER_NUM=8
PUB_SHUFFLE_BUFFER=100000
if [ "$TM_DATALOADER_NUM" = "$PUB_DATALOADER_NUM" ] && \
   [ "$TM_SHUFFLE_BUFFER" = "$PUB_SHUFFLE_BUFFER" ]; then
    LOADER_STATUS="data loader IDENTICAL to the published arms"
else
    LOADER_STATUS="data loader DEVIATES (published: dataloaderNum ${PUB_DATALOADER_NUM}, shuffle_buffer ${PUB_SHUFFLE_BUFFER})"
fi
# TM_SEED_FAST=1 (the default here, 2026-09-10): seed every RNG but LEAVE cuDNN
# autotuning on. Two reasons, both decided before the campaign launched:
#   (i)  kernel-selection nondeterminism is part of the run-to-run variance a
#        seed replicate exists to measure (Reviewer 2, comment 5b asks for the
#        variance from "random initialization and stochastic optimization");
#        suppressing it reports a spread narrower than the truth, which is the
#        anti-conservative direction for a NULL result;
#   (ii) it restores the setting the published arms ran under (they never set
#        TM_SEED), so the launch environment matches too, not just the flags.
# Set TM_SEED_FAST=0 to force deterministic kernels instead (costs ~15%).
# TM_SEED_FAST=1（本脚本默认，2026-09-10）：固定所有 RNG，但保留 cuDNN autotune。
#   (i) kernel 选择的非确定性本就属于"跨次运行方差"，正是种子重复要测的东西
#       （审稿人 2 第 5b 条要的就是随机初始化 + 随机优化带来的方差）；压掉它会让
#       方差偏小，对一个零结果而言是反保守方向；
#   (ii) 已发表各臂从未设 TM_SEED，保留 autotune 才连运行环境也一致。
# 需要确定性 kernel 就设 TM_SEED_FAST=0（约多花 15% 机时）。
export TM_SEED_FAST="${TM_SEED_FAST:-1}"
if [ "${TM_SEED_FAST}" = "1" ]; then
    DETERMINISM_STATUS="cuDNN autotune KEPT (TM_SEED=${TM_SEED} seeds the RNGs only) — as the published arms"
else
    DETERMINISM_STATUS="cuDNN DETERMINISTIC (TM_SEED=${TM_SEED}); the published arms were unseeded/benchmark=True"
fi
RECIPE_STATUS="${LOADER_STATUS}; ${DETERMINISM_STATUS}"

# --- fixed campaign inputs ------------------------------------------------
MANIFEST="${TM_2X2_MANIFEST:-${TM_TRAIN_LIST_SUB40:-./Datasets/shard_train_256_subset40.csv}}"
# The published launcher passed "./$MAN" with a relative MAN. Normalize once here
# instead of at the launch site: an ABSOLUTE override would otherwise be turned
# into ".//data4/..." — which POSIX resolves as the RELATIVE "./data4/...", so the
# arm dies at open() instead of reading the manifest the pre-flight just validated.
# 论文脚本传的是 "./$MAN"（MAN 为相对路径）。在这里统一归一化，而不是在启动处拼接：
# 否则绝对路径覆盖会被拼成 ".//data4/..."，POSIX 将其解析为相对路径 "./data4/..."，
# 于是该臂在 open() 处死掉，读的根本不是预检刚校验过的那个清单。
case "$MANIFEST" in
    /*) MANIFEST_ARG="$MANIFEST" ;;       # absolute — pass through untouched
    *)  MANIFEST_ARG="./${MANIFEST#./}" ;; # relative — exactly the published form
esac
STAGE1="${TM_2X2_STAGE1:-Checkpoint_F_stage1_256/model_epoch_9.pth}"
EPOCHS="${TM_2X2_EPOCHS:-5}"
GPU_MEM_MIB="${TM_2X2_GPU_MEM_MIB:-1024}"
LOAD_TIMEOUT="${TM_2X2_LOAD_TIMEOUT:-1800}"
KILL_ON_FAIL="${TM_2X2_KILL_ON_FAIL:-1}"
POLL="${TM_2X2_POLL:-300}"
HEARTBEAT="${TM_2X2_HEARTBEAT:-3600}"
SPS_PER_GPU="${TM_2X2_SPS_PER_GPU:-19}"
RESTART_PARTIAL="${TM_2X2_RESTART_PARTIAL:-0}"
FORCE="${TM_2X2_FORCE:-0}"
# CONCURRENT_ARMS / GPUS_NEEDED_NOW are computed after the resume classification
# below: a resumed replicate has fewer arms left to run, hence a smaller budget.
# CONCURRENT_ARMS / GPUS_NEEDED_NOW 在下面的续跑分类之后才算：续跑时剩下要跑的臂更少，
# 预算也更小。

case "$EPOCHS" in ''|*[!0-9]*) echo "ERROR: TM_2X2_EPOCHS must be a positive integer" >&2; exit 1 ;; esac
[ "$EPOCHS" -ge 1 ] || { echo "ERROR: TM_2X2_EPOCHS must be >= 1" >&2; exit 1; }
FINAL_EPOCH=$((EPOCHS - 1))

# The exact strings JackFramework logs. Keyed on the sources, not on memory:
#   Source/UserModelImplementation/Models/_interface_base.py:422
#       jf.log.info(f"[{cls}] loaded trainable_tail (missing=..., unexpected=...)")
#   JackFramework/Core/Graph/_meta_ops.py:151
#       log.warning('No checkpoint found; starting from scratch.')
#   JackFramework/Core/Mode/train_proc.py:117
#       log.info('Start iteration!')     <- printed AFTER restore_model(), so it
#                                           is the point at which the presence or
#                                           absence of the load line is decided.
# JF 打印的确切字符串（对着源码取，不靠记忆）。注意 'Start iteration!' 在
# restore_model() 之后打印，因此它就是"加载与否已成定局"的时刻。
LOAD_MARK='loaded trainable_tail'
START_MARK='Start iteration!'
NOCKPT_MARK='No checkpoint found'

echo "======================================================================"
echo " 2x2 replicate — seed ${SEED}   [${MODE_NAME}]"
echo "======================================================================"
echo "  project        : ${TM_PROJ}"
echo "  python         : ${TM_PY}"
echo "  manifest       : ${MANIFEST}"
echo "  shared Stage-1 : ${STAGE1}"
echo "  epochs         : ${EPOCHS} (final ckpt = model_epoch_${FINAL_EPOCH}.pth)"
echo "  batch/GPU      : ${BATCH}   GPUs/arm: ${GPUS_PER_ARM}"
echo "  dataloaderNum  : ${TM_DATALOADER_NUM}   shuffle_buffer: ${TM_SHUFFLE_BUFFER}"
echo "  recipe vs paper: ${LOADER_STATUS}"
echo "                   ${DETERMINISM_STATUS}"
echo "  port base      : ${PORT_BASE} (arms use ${PORT_BASE}..$((PORT_BASE + N_ARMS - 1)))"
echo ""

# --------------------------------------------------------------------------
# tm_2x2_sha256 <file>
#   Portable sha256 (sha256sum on Linux, shasum -a 256 on macOS).
#   跨平台 sha256（Linux 用 sha256sum，macOS 用 shasum -a 256）。
# --------------------------------------------------------------------------
tm_2x2_sha256() {
    local f="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$f" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$f" | awk '{print $1}'
    else
        echo "UNAVAILABLE(no sha256sum/shasum)"
    fi
}

# --------------------------------------------------------------------------
# tm_2x2_eta <minutes>
#   "now + N minutes" in local time. GNU date wants -d, BSD date wants -v.
#   本地时间的"此刻 + N 分钟"。GNU date 用 -d，BSD date 用 -v。
# --------------------------------------------------------------------------
tm_2x2_eta() {
    local mins="$1"
    if date -d "+${mins} minutes" '+%F %H:%M %Z' 2>/dev/null; then return 0; fi
    if date -v "+${mins}M" '+%F %H:%M %Z' 2>/dev/null; then return 0; fi
    echo "(no portable date arithmetic on this host)"
}

# --------------------------------------------------------------------------
# Pre-flight 1 — inputs exist. Fail before anything is created.
# 预检 1 —— 输入齐全。任何目录被创建之前就失败。
# --------------------------------------------------------------------------
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: shard manifest not found: ${MANIFEST}" >&2
    echo "       Override with TM_2X2_MANIFEST=/path/manifest.csv" >&2
    exit 1
fi
tm_require_ckpt "$STAGE1" || exit 1
if [ ! -f "$STAGE1" ]; then
    echo "ERROR: the shared Stage-1 seed must be a .pth FILE, not a directory:" >&2
    echo "       ${STAGE1}" >&2
    exit 1
fi
if [ ! -f "Source/Tools/strip_stage1_opt.py" ]; then
    echo "ERROR: Source/Tools/strip_stage1_opt.py missing — wrong project root?" >&2
    exit 1
fi

# `--imgNum` MUST be the manifest's real sample total: JackFramework uses it as
# the progress denominator, and a wrong value makes the bar hit 100% at a
# fraction of the epoch and look like a stuck save.
# `--imgNum` 必须等于清单的真实样本总数：JF 拿它当进度分母，填错会让进度条在
# 半个 epoch 就到 100%，看起来像卡在保存。
REAL_TOTAL=$(tail -n +2 "$MANIFEST" | awk -F, '{s+=$2} END{print s+0}')
if [ -z "$REAL_TOTAL" ] || [ "$REAL_TOTAL" -le 0 ] 2>/dev/null; then
    echo "ERROR: could not read real_total from ${MANIFEST} (got '${REAL_TOTAL}')." >&2
    echo "       Expected a shard manifest with header 'path,file_num'." >&2
    exit 1
fi
echo "  manifest real_total (--imgNum) : ${REAL_TOTAL}"


# --------------------------------------------------------------------------
# Pre-flight 0 — the interpreter. TM_PY defaults to bare `python`, which on a
# conda box is whatever env happens to be active — often `base`, which has no
# torch. Launching four arms with the wrong interpreter fails minutes in, or
# worse, succeeds against a different torch than the published arms used.
# 预检 0 —— 解释器。TM_PY 默认是裸 `python`，在 conda 机器上就是当前激活的环境，
# 往往是没有 torch 的 base。用错解释器要么几分钟后失败，要么更糟：跑起来了，但
# torch 版本与已发表各臂不同。
# --------------------------------------------------------------------------
if ! "$TM_PY" -c 'import torch, JackFramework' >/dev/null 2>&1; then
    echo "ERROR: TM_PY='${TM_PY}' cannot import torch + JackFramework." >&2
    echo "       Resolved to: $(command -v "$TM_PY" 2>/dev/null || echo 'not found')" >&2
    echo "       Pass the env's interpreter explicitly, e.g." >&2
    echo "         TM_PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python \\" >&2
    echo "         bash Scripts/train/train_2x2.sh ${SEED}" >&2
    exit 1
fi
echo "  interpreter    : $(command -v "$TM_PY") ($("$TM_PY" -c 'import torch;print("torch "+torch.__version__)' 2>/dev/null))"

# --------------------------------------------------------------------------
# Arm table — arm | modelName | ckpt dir | seeded? | tag
# Directory and tag names carry the seed so a replicate can never overwrite
# another one. 目录名与 tag 都带 seed，重复之间绝不会互相覆盖。
# --------------------------------------------------------------------------
CK_S1TR="Checkpoint_G_s1tr_256_s${SEED}"
CK_RTR="Checkpoint_G_rtr_256_s${SEED}"
CK_S1FR="Checkpoint_F_s1fr_256_s${SEED}"
CK_RFR="Checkpoint_F_rfr_256_s${SEED}"
SHARED_DIR="Checkpoint_2x2_shared_init_s${SEED}"
SHARED_INIT="${SHARED_DIR}/model_stage1_init.pth"
SHARED_SHA_FILE="${SHARED_INIT}.sha256"

ARM_NAME=(  "S1/Tr"      "R/Tr"       "S1/Fr"      "R/Fr"     )
ARM_MODEL=( "TM_Model_G" "TM_Model_G" "TM_Model_F" "TM_Model_F")
ARM_CKPT=(  "$CK_S1TR"   "$CK_RTR"    "$CK_S1FR"   "$CK_RFR"  )
ARM_SEEDED=("1"          "0"          "1"          "0"        )
ARM_TAG=(   "x2_s1tr_s${SEED}" "x2_rtr_s${SEED}" "x2_s1fr_s${SEED}" "x2_rfr_s${SEED}")
ARM_GPUS=("" "" "" "")
ARM_PORT=("" "" "" "")
ARM_PID=("" "" "" "")
ARM_LOG=("" "" "" "")
ARM_STATE=("" "" "" "")
for a in 0 1 2 3; do
    ARM_PORT[$a]=$((PORT_BASE + a))
    ARM_LOG[$a]="train_${ARM_TAG[$a]}.log"
done
RUNNING_ARMS=""

# --------------------------------------------------------------------------
# RESUMPTION — classify each arm before touching anything.
#
#   DONE    : the final-epoch checkpoint exists -> skip, never re-train.
#             A replicate spans ~9 days; an SSH drop, a reboot or a Ctrl-C must
#             not cost the arms that already finished.
#   PARTIAL : some model_epoch_*.pth but not the final one -> AMBIGUOUS, stop.
#             Resuming it in place is NOT allowed for a seed replicate: JF would
#             continue with a fresh data order, so the run is no longer the seed
#             it claims to be (brief_experiments §7.4). The human decides:
#             archive it and retrain clean, or keep it and investigate.
#   FRESH   : no epoch checkpoints (an empty dir, or one holding only the
#             stripped init from a launch that never got an epoch out) -> run it.
#
# 续跑分类：DONE=末代 ckpt 已存在，跳过，绝不重训；PARTIAL=有中间代但没末代，
# 有歧义，停下来交给人判断（seed 重复不允许原地续训：JF 会用新的数据顺序继续，
# 那这个 run 就不再是它自称的那个 seed 了）；FRESH=没有任何 epoch ckpt，正常跑。
# --------------------------------------------------------------------------
arm_epochs_present() {   # <ckpt_dir> -> "0,1,2" or ""
    local dir="$1"
    [ -d "$dir" ] || { echo ""; return 0; }
    find "$dir" -maxdepth 1 -name 'model_epoch_*.pth' 2>/dev/null |
        sed 's|.*/model_epoch_||; s|\.pth$||' | sort -n | paste -sd, - | tr -d ' '
}

# TRUNCATED-FINAL-CHECKPOINT GUARD.
#   `[ -s ]` only proves "not empty". A driver (or the box) killed DURING the
#   final torch.save leaves a short, non-empty model_epoch_<final>.pth — which
#   the classifier would call DONE, the script would skip forever, and the
#   campaign would evaluate a corrupt checkpoint as a headline arm. Epoch
#   checkpoints of one arm are the same state_dict + optimizer state and so are
#   within a few bytes of each other, which makes a size comparison against the
#   arm's own earlier epochs a cheap, false-positive-free truncation test.
#   Verifying the pickle itself would need torch and is not worth ~9 days of
#   latency on a check this decisive, so: any final ckpt under 90 % of the
#   largest sibling is treated as truncated.
#   截断末代 ckpt 的防护：`[ -s ]` 只能证明"非空"。若在最后一次 torch.save 期间被杀，
#   会留下一个非空但残缺的 model_epoch_<final>.pth，分类器会判为 DONE、之后永远跳过，
#   于是主臂拿一个损坏的权重去评测。同一个臂各代 ckpt 的内容结构相同、大小相差无几，
#   因此与本臂更早的 epoch 比大小，是既便宜又不会误报的截断检测。低于最大同类 90% 即判截断。
arm_final_truncated() {   # <ckpt_dir> -> 0 if the final ckpt looks truncated
    local dir="$1" final="$1/model_epoch_${FINAL_EPOCH}.pth" sz max=0 f
    [ -s "$final" ] || return 1
    sz=$(wc -c < "$final" 2>/dev/null | tr -d ' ')
    [ -n "$sz" ] || return 1
    for f in $(find "$dir" -maxdepth 1 -name 'model_epoch_*.pth' 2>/dev/null); do
        case "$f" in *"/model_epoch_${FINAL_EPOCH}.pth") continue ;; esac
        local other
        other=$(wc -c < "$f" 2>/dev/null | tr -d ' ')
        [ -n "$other" ] || continue
        [ "$other" -gt "$max" ] && max="$other"
    done
    [ "$max" -gt 0 ] || return 1          # no sibling to compare against
    awk -v s="$sz" -v m="$max" 'BEGIN{exit !(s < 0.9 * m)}'
}

classify_arm() {   # <ckpt_dir> -> DONE | PARTIAL | TRUNC | FRESH
    local dir="$1"
    [ -d "$dir" ] || { echo FRESH; return 0; }
    if [ -s "${dir}/model_epoch_${FINAL_EPOCH}.pth" ]; then
        if arm_final_truncated "$dir"; then echo TRUNC; else echo DONE; fi
        return 0
    fi
    if [ -n "$(arm_epochs_present "$dir")" ]; then echo PARTIAL; return 0; fi
    echo FRESH
}

for a in 0 1 2 3; do
    ARM_STATE[$a]=$(classify_arm "${ARM_CKPT[$a]}")
done

# --------------------------------------------------------------------------
# WALL-CLOCK ARITHMETIC — a computed schedule, not a magic number.
#
#   samples/epoch          = REAL_TOTAL              (read from the manifest)
#   throughput             = SPS_PER_GPU x GPUS_PER_ARM samples/s
#                            SPS_PER_GPU default 19 = the MEASURED rate for the
#                            simple-head arms on a 3090 at 2 GPU x bs32
#                            (brief_experiments §0.2; the 4080 anchor,
#                            23.5 s/s/GPU, implies a consistent 19/23.5 = 0.81).
#   seconds/arm            = REAL_TOTAL x EPOCHS / (SPS_PER_GPU x GPUS_PER_ARM)
#   worked example (subset-40, 1.34e6 samples, 5 epochs, 2 GPU):
#       1.34e6 x 5 / (19 x 2) = 176,300 s = 49.0 h   <- matches the measured
#                                                       "~49 h per 5-epoch arm"
#   determinism tax        = +15%: TM_SEED is always set here, and main.py's
#                            _maybe_fix_seed() then forces cudnn.deterministic=
#                            True / benchmark=False (brief_experiments §7.6).
#                            49.0 h x 1.15 = 56 h; the observed band is 49-52 h,
#                            so the tax is reported separately rather than baked
#                            into the headline number.
#   replicate (sequential) = 4 x per-arm  ~= 8.2 days raw / 9.4 days taxed  <- what
#                            the PLAN block prints, and the number to quote
#   replicate (concurrent) = 1 x per-arm  ~= 2.0 days raw / 2.4 days taxed
#
# 墙钟推算：吞吐取实测 19 samples/s/GPU（3090，简单头，2 卡 bs32），
# 每臂秒数 = 样本数 × epoch 数 ÷ (19 × 每臂 GPU 数)；subset-40 代入 = 49.0 h/臂，
# 与实测 "~49 h" 吻合。TM_SEED 会强制 cudnn.deterministic，另计 +15% 的确定性税。
# 串行 = 4 × 每臂 ≈ 8.2 天（含税 9.4 天，以含税为准）；并发 ≈ 2.0 天（含税 2.4 天）。
# --------------------------------------------------------------------------
ARM_SECONDS=$(awk -v n="$REAL_TOTAL" -v e="$EPOCHS" -v s="$SPS_PER_GPU" -v g="$GPUS_PER_ARM" \
    'BEGIN{printf "%.0f", n * e / (s * g)}')
ARM_HOURS=$(awk -v s="$ARM_SECONDS" 'BEGIN{printf "%.1f", s/3600.0}')
# The determinism tax is paid only when cuDNN is forced deterministic.
# 只有强制确定性 kernel 时才付这 15%。
if [ "${TM_SEED_FAST}" = "1" ]; then TAX=1.00; else TAX=1.15; fi
ARM_HOURS_TAXED=$(awk -v s="$ARM_SECONDS" -v t="$TAX" 'BEGIN{printf "%.1f", s*t/3600.0}')

N_TODO=0
for a in 0 1 2 3; do
    [ "${ARM_STATE[$a]}" = "DONE" ] || N_TODO=$((N_TODO + 1))
done
if [ "$CONCURRENT" = "1" ]; then WAVES=1; else WAVES="$N_TODO"; fi

# How many arms are actually in flight at once — the number every GPU and RAM
# budget below is computed from. On a resumed replicate a concurrent wave is
# narrower than four, so ask for fewer GPUs rather than refusing to start.
# 真正同时在跑的臂数——下面所有 GPU / 内存预算都由它算出。续跑时并发那一波不足四臂，
# 就少要几张卡，而不是直接拒绝启动。
if [ "$CONCURRENT" = "1" ]; then CONCURRENT_ARMS="$N_TODO"; else CONCURRENT_ARMS=1; fi
GPUS_NEEDED_NOW=$((CONCURRENT_ARMS * GPUS_PER_ARM))
TOTAL_SECONDS=$(awk -v s="$ARM_SECONDS" -v w="$WAVES" 'BEGIN{printf "%.0f", s*w}')
TOTAL_DAYS=$(awk -v s="$TOTAL_SECONDS" 'BEGIN{printf "%.1f", s/86400.0}')
TOTAL_DAYS_TAXED=$(awk -v s="$TOTAL_SECONDS" -v t="$TAX" 'BEGIN{printf "%.1f", s*t/86400.0}')
ETA_MINS=$(awk -v s="$TOTAL_SECONDS" -v t="$TAX" 'BEGIN{printf "%d", s*t/60.0}')

echo ""
echo "======================================================================"
echo " PLAN — check this before committing the box for days"
echo " 执行计划 —— 占用机器好几天之前，先把这段看一遍"
echo "======================================================================"
printf '  %-3s %-6s %-11s %-30s %-9s %-6s %s\n' \
    '#' ARM MODEL CKPT_DIR SEEDED PORT STATE
for a in 0 1 2 3; do
    _seeded="random"; [ "${ARM_SEEDED[$a]}" = "1" ] && _seeded="Stage-1"
    printf '  %-3s %-6s %-11s %-30s %-9s %-6s %s\n' \
        "$((a + 1))" "${ARM_NAME[$a]}" "${ARM_MODEL[$a]}" "${ARM_CKPT[$a]}" \
        "$_seeded" "${ARM_PORT[$a]}" "${ARM_STATE[$a]}"
done
echo ""
echo "  order          : ${ARM_NAME[0]} -> ${ARM_NAME[1]} -> ${ARM_NAME[2]} -> ${ARM_NAME[3]}"
echo "  arms to run    : ${N_TODO} of ${N_ARMS}  (DONE arms are skipped, see below)"
# The TAXED figure leads, because it is the one that applies: TM_SEED is set
# unconditionally by this script, so cudnn.deterministic is ALWAYS on and the
# +15% is always being paid. The untaxed number is kept beside it only because
# it is the one that matches the 19 samples/s/GPU anchor, which was measured on
# the published (unseeded, benchmark=True) arms.
# 以**含税**数字为准：本脚本必设 TM_SEED，cudnn.deterministic 恒为开，那 15% 一直在付。
# 保留未含税数字只是因为它对应 19 samples/s/GPU 这个锚，而该锚测自已发表的（未设 seed、
# benchmark=True）各臂。
echo "  per arm        : ~${ARM_HOURS_TAXED} h  (raw ~${ARM_HOURS} h; determinism tax x${TAX})"
echo "                   = ${REAL_TOTAL} samples x ${EPOCHS} epochs / (${SPS_PER_GPU} samples/s/GPU x ${GPUS_PER_ARM} GPU), x${TAX}"
echo "                   anchor measured on FROZEN-encoder arms; the two ${ARM_MODEL[0]} arms"
echo "                   train the SAR trunk too, so treat this as a LOWER bound for them."
echo "  this invocation: ~${TOTAL_DAYS_TAXED} days  (~${TOTAL_DAYS} days raw), ${WAVES} wave(s)"
echo "  projected end  : $(tm_2x2_eta "$ETA_MINS")"
echo "  GPUs at a time : ${GPUS_NEEDED_NOW}  (${CONCURRENT_ARMS} arm(s) x ${GPUS_PER_ARM})"
echo ""
for a in 0 1 2 3; do
    case "${ARM_STATE[$a]}" in
        DONE)
            echo "  SKIP ${ARM_NAME[$a]}: ${ARM_CKPT[$a]}/model_epoch_${FINAL_EPOCH}.pth exists"
            echo "       (epochs present: $(arm_epochs_present "${ARM_CKPT[$a]}")) — already trained, not re-running."
            ;;
        PARTIAL)
            echo "  STOP ${ARM_NAME[$a]}: ${ARM_CKPT[$a]} is HALF-FINISHED" >&2
            echo "       epochs present: $(arm_epochs_present "${ARM_CKPT[$a]}"), missing model_epoch_${FINAL_EPOCH}.pth" >&2
            ;;
        TRUNC)
            echo "  STOP ${ARM_NAME[$a]}: ${ARM_CKPT[$a]}/model_epoch_${FINAL_EPOCH}.pth is TRUNCATED" >&2
            echo "       it exists but is far smaller than this arm's earlier epochs — the" >&2
            echo "       run was killed during the final save. It would otherwise have been" >&2
            echo "       classified DONE, skipped forever, and evaluated as a headline arm." >&2
            echo "       末代 ckpt 存在但远小于本臂更早的 epoch：最后一次保存被打断。若不拦下，" >&2
            echo "       它会被判为 DONE、永远跳过，并作为主臂拿去评测。" >&2
            ls -l "${ARM_CKPT[$a]}"/model_epoch_*.pth 2>/dev/null | sed 's/^/         /' >&2
            ;;
    esac
done

# A half-finished or truncated arm is ambiguous by construction, so stop rather
# than guess. Both are handled identically: archive and retrain clean.
# 半截或被截断的臂天然有歧义，因此停下来而不是替人猜。两者处理方式相同：归档后重训。
PARTIAL_FOUND=0
for a in 0 1 2 3; do
    case "${ARM_STATE[$a]}" in PARTIAL|TRUNC) PARTIAL_FOUND=1 ;; esac
done
if [ "$PARTIAL_FOUND" = "1" ] && [ "$RESTART_PARTIAL" != "1" ]; then
    echo "" >&2
    echo "ERROR: at least one arm is half-finished; refusing to guess." >&2
    echo "       Resuming in place is NOT valid for a seed replicate: JackFramework" >&2
    echo "       would continue with a fresh data order, so the run would no longer be" >&2
    echo "       the seed it claims to be (brief_experiments 7.4)." >&2
    echo "       原地续训对 seed 重复无效：JF 会用新的数据顺序继续，这个 run 就不再是" >&2
    echo "       它自称的那个 seed 了。请二选一：" >&2
    echo "" >&2
    echo "       (a) archive it and retrain that arm clean:" >&2
    for a in 0 1 2 3; do
        case "${ARM_STATE[$a]}" in PARTIAL|TRUNC) ;; *) continue ;; esac
        echo "             mv ${ARM_CKPT[$a]} ${ARM_CKPT[$a]}.partial.\$(date +%Y%m%d_%H%M%S)" >&2
    done
    echo "           or re-run this script with TM_2X2_RESTART_PARTIAL=1, which does" >&2
    echo "           exactly that move (loudly) and then retrains the arm." >&2
    echo "       (b) keep it and investigate why it stopped (check its log tail)." >&2
    exit 1
fi

if [ "$N_TODO" = "0" ]; then
    echo ""
    echo "  Nothing to do — all ${N_ARMS} arms of seed ${SEED} already have model_epoch_${FINAL_EPOCH}.pth."
    echo "  seed ${SEED} 的四个臂都已有末代 ckpt，无需再训。"
    echo ""
    # Re-run the load gate after the fact, off the logs. Cheap, and it is the
    # one check that stands between "we trained four arms" and "one of the two
    # headline arms was silently a second random control".
    # 事后再跑一遍加载闸（对着日志）。很便宜，而它正是"训了四个臂"与"其中一个主臂其实
    # 是第二个随机对照"之间唯一的分界线。
    echo "  Integrity re-check from the logs / 日志事后复检:"
    _bad=0
    for a in 0 1 2 3; do
        if [ ! -f "${ARM_LOG[$a]}" ]; then
            echo "    WARN ${ARM_NAME[$a]}: ${ARM_LOG[$a]} is gone — cannot re-verify." >&2
            continue
        fi
        _want="random"; [ "${ARM_SEEDED[$a]}" = "1" ] && _want="loaded"
        if grep -q "$LOAD_MARK" "${ARM_LOG[$a]}" 2>/dev/null; then _got="loaded"; else _got="random"; fi
        if [ "$_got" = "$_want" ]; then
            echo "    OK   ${ARM_NAME[$a]}: ${_got} init, as designed."
        else
            echo "    FAIL ${ARM_NAME[$a]}: its log says '${_got}' init but this arm must be '${_want}'." >&2
            _bad=1
        fi
    done
    if [ "$_bad" = "1" ]; then
        echo "ERROR: seed ${SEED} is NOT a valid replicate — see the FAIL line(s) above." >&2
        echo "       Move the offending Checkpoint_*_s${SEED} dir aside and re-run." >&2
        exit 1
    fi
    echo ""
    echo "  Next: evaluate them (see the eval block at the end of this script)."
    exit 0
fi

# --------------------------------------------------------------------------
# GPU helpers. In sequential mode the free-device list is re-read AT EACH ARM'S
# START, not once: a replicate now spans ~9 days and a neighbouring user's job
# may come or go in that time, so a snapshot taken on day 0 is worthless on day 6.
# GPU 助手。串行模式下每个臂启动时**重新**读取空闲卡列表，而不是开头读一次：
# 一个重复要跨 ~9 天，隔壁同学的任务随时可能上下线，第 0 天的快照到第 6 天已经失效。
# --------------------------------------------------------------------------
tm_2x2_show_gpu_state() {
    echo "  --- nvidia-smi (index, used MiB, total MiB, util %) ---"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    echo "  --- compute processes currently holding GPUs ---"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    echo "    (empty above = no compute apps reported)"
}

tm_2x2_free_gpus() {   # -> space-separated indices of idle devices, right now
    command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return 0; }
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null |
        awk -F', *' -v thr="$GPU_MEM_MIB" '$2+0 <= thr {print $1}' | tr '\n' ' '
}

# pick_devices <n> <what> -> echoes a comma list of <n> device ids, or fails.
# Honours TM_2X2_GPUS (which must then hold exactly the number needed for the
# current mode) and re-queries nvidia-smi on every call.
# 取 <n> 张卡；TM_2X2_GPUS 若已指定，其长度必须正好等于当前模式所需张数。
pick_devices() {
    local n="$1" what="$2" free busy g
    if [ -n "${TM_2X2_GPUS:-}" ]; then
        local -a req
        IFS=',' read -r -a req <<< "$TM_2X2_GPUS"
        if [ "${#req[@]}" -ne "$GPUS_NEEDED_NOW" ]; then
            echo "ERROR: TM_2X2_GPUS lists ${#req[@]} devices but this mode needs ${GPUS_NEEDED_NOW}" >&2
            echo "       (${CONCURRENT_ARMS} arm(s) x ${GPUS_PER_ARM} GPUs)." >&2
            return 1
        fi
        free=" $(tm_2x2_free_gpus) "
        busy=""
        if command -v nvidia-smi >/dev/null 2>&1; then
            for g in "${req[@]}"; do
                case "$free" in *" $g "*) ;; *) busy="$busy $g" ;; esac
            done
        fi
        if [ -n "$busy" ]; then
            echo "ERROR: requested GPU(s)${busy} are busy (>${GPU_MEM_MIB} MiB used) [${what}]." >&2
            tm_2x2_show_gpu_state >&2
            if [ "$FORCE" != "1" ]; then return 1; fi
            echo "WARN: TM_2X2_FORCE=1 — launching onto busy GPUs anyway." >&2
        fi
        # Sequential reuses the same devices for every arm; concurrent slices them.
        # 串行每个臂复用同一组卡；并发则由调用方切片。
        echo "$TM_2X2_GPUS"
        return 0
    fi

    free=$(tm_2x2_free_gpus)
    local n_free
    n_free=$(echo "$free" | wc -w | tr -d ' ')
    if [ "${n_free:-0}" -lt "$n" ]; then
        echo "ERROR: need ${n} free GPUs [${what}], found ${n_free:-0}: ${free:-none}" >&2
        echo "       A GPU counts as free below ${GPU_MEM_MIB} MiB used (TM_2X2_GPU_MEM_MIB)." >&2
        echo "       What is running instead:" >&2
        tm_2x2_show_gpu_state >&2
        echo "       Options: wait, lower gpus_per_arm, or pin devices with TM_2X2_GPUS." >&2
        if [ "$FORCE" != "1" ]; then return 1; fi
        echo "WARN: TM_2X2_FORCE=1 — proceeding with insufficient free GPUs." >&2
    fi
    local picked
    picked=$(echo "$free" | tr ' ' '\n' | grep -v '^$' | head -n "$n" | paste -sd, -)
    # Never hand back an empty list: CUDA_VISIBLE_DEVICES="" makes torch see zero
    # devices, and the arm dies minutes in with a confusing device-count error
    # instead of the GPU shortage that actually caused it. Reachable only under
    # TM_2X2_FORCE=1 (or a missing nvidia-smi), which is exactly when the operator
    # has stopped reading warnings.
    # 绝不返回空列表：CUDA_VISIBLE_DEVICES="" 会让 torch 看到 0 张卡，几分钟后报一个
    # 与真实原因（卡不够）无关的设备数错误。只有 TM_2X2_FORCE=1（或没有 nvidia-smi）
    # 才会走到这里——而那正是操作者已经不看告警的时候。
    if [ -z "$picked" ]; then
        echo "ERROR: no usable GPU index to place [${what}] (free list was empty)." >&2
        echo "       TM_2X2_FORCE=1 cannot conjure devices — pin them with TM_2X2_GPUS." >&2
        return 1
    fi
    echo "$picked"
}

port_free() {   # <port> -> 0 if nothing is listening (or ss is unavailable)
    local p="$1"
    command -v ss >/dev/null 2>&1 || return 0
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${p}\$" && return 1
    return 0
}

# --------------------------------------------------------------------------
# Pre-flight 2 — GPUs. Refuse to start on a contended box rather than crawl or
# OOM halfway through a 50-hour arm. Sequential asks for gpus_per_arm ONLY;
# concurrent asks for four times that.
# 预检 2 —— GPU。宁可拒绝启动，也不要在 50 小时的臂中途被挤爆。串行只要每臂的卡数，
# 并发要四倍。
# --------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — cannot verify the GPUs are free." >&2
    echo "       Set TM_2X2_FORCE=1 to launch blind (not recommended)." >&2
    [ "$FORCE" = "1" ] || exit 1
fi
PREFLIGHT_DEVICES=$(pick_devices "$GPUS_NEEDED_NOW" "pre-flight") || exit 1
echo "  GPUs free now  : ${PREFLIGHT_DEVICES}  (need ${GPUS_NEEDED_NOW} at a time)"
if [ "$CONCURRENT" != "1" ]; then
    echo "                   sequential: re-checked at EACH arm's launch, not pinned now"
fi

# --------------------------------------------------------------------------
# Pre-flight 3 — host RAM (§RAM above). This is the failure that killed an
# earlier campaign, so it is a gate, not a note.
# 预检 3 —— 主机内存（见上面 §RAM）。这是之前搞死过一次实验的失败模式，所以是硬闸。
# --------------------------------------------------------------------------
N_WORKERS=$((CONCURRENT_ARMS * GPUS_PER_ARM * TM_DATALOADER_NUM))
GIB_PER_WORKER=$(awk -v buf="$TM_SHUFFLE_BUFFER" -v g="$GIB_PER_WORKER_AT_100K" \
    'BEGIN{printf "%.2f", g*(buf/100000.0)}')
RAM_EST_GIB=$(awk -v w="$N_WORKERS" -v p="$GIB_PER_WORKER" 'BEGIN{printf "%.0f", w * p}')

# check_host_ram <what> — the estimate is static; MemAvailable is not. Like the
# GPU check, this is re-run AT EACH ARM'S LAUNCH in sequential mode rather than
# once on day 0: over ~9 days a neighbour's job can take the headroom, and
# swap-thrash from an over-subscribed host is the failure that already cost this
# project two days (brief_experiments 7.1) — it degrades everything on the box,
# not just our arm, so it is worth re-asking 4 times instead of once.
# 内存估算是静态的，MemAvailable 不是。与 GPU 检查一样，串行模式下**每个臂启动时**重跑，
# 而不是第 0 天查一次：9 天里隔壁的任务随时可能吃掉余量，而主机超订导致的换页颠簸
# 已经害掉过本项目两天，且它拖垮的是整台机器而不只是我们这个臂。
check_host_ram() {
    local what="$1" avail
    command -v free >/dev/null 2>&1 || {
        echo "  host MemAvailable : (no 'free' on this host — estimate not checked)"
        return 0
    }
    avail=$(free -g | awk '/^Mem:/ {print $7+0}')
    echo "  host MemAvailable : ${avail} GiB  (need ~${RAM_EST_GIB} GiB) [${what}]"
    [ "$RAM_EST_GIB" -gt "$avail" ] || return 0
    echo "ERROR: estimated dataloader RAM (${RAM_EST_GIB} GiB) exceeds MemAvailable (${avail} GiB) [${what}]." >&2
    echo "       Lower TM_DATALOADER_NUM or TM_SHUFFLE_BUFFER, or set TM_2X2_FORCE=1." >&2
    echo "       NOTE: lowering either one in SEQUENTIAL mode reintroduces the" >&2
    echo "       recipe deviation this mode exists to avoid — see the RAM comment." >&2
    echo "       Arms already finished are safe; re-run when the box frees up." >&2
    echo "       已跑完的臂不受影响：等机器空出来再重跑。" >&2
    [ "$FORCE" = "1" ] || return 1
    echo "WARN: TM_2X2_FORCE=1 — launching into a probable OOM." >&2
    return 0
}

echo "  dataloader RAM : ${N_WORKERS} workers x ~${GIB_PER_WORKER} GiB = ~${RAM_EST_GIB} GiB estimated"
echo "                   (${CONCURRENT_ARMS} arm(s) x ${GPUS_PER_ARM} GPU x ${TM_DATALOADER_NUM} workers, buffer ${TM_SHUFFLE_BUFFER})"
check_host_ram "pre-flight" || exit 1
if [ "$CONCURRENT" != "1" ]; then
    echo "                      sequential: MemAvailable re-read at EACH arm's launch"
fi

# --------------------------------------------------------------------------
# Pre-flight 4 — ports. JF binds --port for DDP rendezvous; a collision shows
# up as a hang, not an error. Concurrent needs all four free at once;
# sequential only needs one at a time and re-checks at each launch.
# 预检 4 —— 端口。JF 用 --port 做 DDP 会合，撞端口表现为挂起而不是报错。并发要四个
# 端口同时可用；串行一次只用一个，且在每个臂启动时重新检查。
# --------------------------------------------------------------------------
if [ "$CONCURRENT" = "1" ]; then
    for a in 0 1 2 3; do
        if ! port_free "${ARM_PORT[$a]}"; then
            echo "ERROR: port ${ARM_PORT[$a]} is already bound. Pick another port_base (arg 4)." >&2
            [ "$FORCE" = "1" ] || exit 1
        fi
    done
fi

# --------------------------------------------------------------------------
# Pre-flight 5 — recipe uniformity ACROSS INVOCATIONS.
#
# The 2x2's internal contrast is only clean if all four arms saw the same
# training recipe. In sequential mode the four arms are launched DAYS apart and
# usually from different invocations of this script, so every knob that the
# environment can move has to be pinned on the first launch and re-checked on
# every later one. Anything left out of RECIPE_NOW is a knob that can silently
# differ BETWEEN ARMS — which is exactly the third factor the response letter
# claims does not exist.
#
# Recorded: the loader pair (sequential vs concurrent differ, §RAM), the launch
# geometry, AND the environment-driven flags — TM_SEED (data order + cuDNN
# determinism), TM_LR, TM_SAR_SIZE, TM_MATCHING_DIM. Those four are read from
# the environment at launch, so without them here a stray `TM_LR=5e-4` on the
# second invocation would train arms 3 and 4 at a different learning rate and
# nothing would notice.
#
# 预检 5 —— 跨调用的配方一致性。串行模式下四个臂相隔数天、通常分属不同次脚本调用，
# 因此凡是环境变量能改动的旋钮，都必须在首次启动时钉死、之后每次调用重新核对。
# 任何没写进 RECIPE_NOW 的旋钮，都是一个可以在**臂与臂之间**悄悄变化的量——而那正是
# 回复信声称不存在的第三个因子。除 dataloader 与几何外，这里额外记录 TM_SEED（数据
# 顺序与 cuDNN 确定性）、TM_LR、TM_SAR_SIZE、TM_MATCHING_DIM：它们都在启动时从环境
# 读取，不记录的话，第二次调用时一个漏掉的 `TM_LR=5e-4` 会让后两个臂用不同学习率训练，
# 而没有任何东西会发现。
# --------------------------------------------------------------------------
RECIPE_FILE="${SHARED_DIR}/recipe.env"
RECIPE_NOW="dataloaderNum=${TM_DATALOADER_NUM} shuffle_buffer=${TM_SHUFFLE_BUFFER} batch=${BATCH} gpus_per_arm=${GPUS_PER_ARM} epochs=${EPOCHS} manifest=${MANIFEST} lr=${TM_LR:-1e-3} sar_size=${TM_SAR_SIZE} matching_dim=${TM_MATCHING_DIM} tm_seed=${TM_SEED:-unset}"
if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$SHARED_DIR" || exit 1
fi
if [ -s "$RECIPE_FILE" ]; then
    RECIPE_WAS=$(cat "$RECIPE_FILE")
    if [ "$RECIPE_NOW" != "$RECIPE_WAS" ]; then
        echo "" >&2
        echo "ERROR: this invocation's recipe differs from the one seed ${SEED} started with." >&2
        echo "       recorded : ${RECIPE_WAS}" >&2
        echo "       now      : ${RECIPE_NOW}" >&2
        echo "       Arms trained under different recipes are not a factorial design." >&2
        echo "       Re-run in the original mode/settings, or start a fresh seed dir." >&2
        echo "       不同配方训出来的臂不构成析因设计：请用原设置重跑，或换一个新的 seed。" >&2
        [ "$FORCE" = "1" ] || exit 1
        echo "WARN: TM_2X2_FORCE=1 — continuing with a MIXED recipe. Disclose it." >&2
    else
        echo "  recipe check   : matches the one recorded for seed ${SEED}"
    fi
elif [ "$DRY_RUN" = "1" ]; then
    echo "  recipe check   : nothing recorded yet for seed ${SEED}; this run would record"
    echo "                   ${RECIPE_NOW}"
else
    printf '%s\n' "$RECIPE_NOW" > "$RECIPE_FILE"
    echo "  recipe recorded: ${RECIPE_FILE}"
fi

# --------------------------------------------------------------------------
# Pre-flight 5b — CODE identity across invocations.
#
# RECIPE_NOW pins the knobs. It cannot pin the thing the knobs are fed to: the
# four arms are launched days apart, and between arm 2 and arm 3 someone can
# `git pull`, edit a model file, or `conda activate` a different env. That would
# be a third factor — the exact thing the response letter asserts does not exist
# — and nothing else in this script would notice.
#
# Recorded as a WARNING, not a refusal, and deliberately so: a commit that only
# touches a README must not strand a replicate four days in, and the remote tree
# is deployed by file sync and may not be a git checkout at all. The warning is
# the audit trail; the human decides whether the change could have touched the
# model.
#
# 预检 5b —— 跨调用的**代码**一致性。RECIPE_NOW 钉的是旋钮，钉不住旋钮作用的对象：
# 四个臂相隔数天启动，其间完全可能 git pull、改模型文件、换 conda 环境——那就是第三个
# 因子，正是回复信声称不存在的东西，而本脚本其他任何检查都发现不了。
# 这里只告警不拒绝，是有意为之：一次只动 README 的提交不该让跑了四天的重复卡死，而且
# 远端是文件同步部署的、未必是 git 仓库。告警即留痕，是否影响模型由人判断。
# --------------------------------------------------------------------------
CODE_FILE="${SHARED_DIR}/code.env"
_code_head=$(git -C "$TM_PROJ" rev-parse HEAD 2>/dev/null || echo "not-a-git-checkout")
_code_dirty=""
if git -C "$TM_PROJ" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$TM_PROJ" diff --quiet 2>/dev/null || _code_dirty="+dirty"
fi
_code_py=$(command -v "$TM_PY" 2>/dev/null || printf '%s' "$TM_PY")
CODE_NOW="code=${_code_head}${_code_dirty} python=${_code_py}"
if [ -s "$CODE_FILE" ]; then
    CODE_WAS=$(cat "$CODE_FILE")
    if [ "$CODE_NOW" != "$CODE_WAS" ]; then
        echo "" >&2
        echo "WARN: the code/interpreter changed since seed ${SEED}'s first arm." >&2
        echo "      recorded : ${CODE_WAS}" >&2
        echo "      now      : ${CODE_NOW}" >&2
        echo "      If anything under Source/UserModelImplementation changed, the arms" >&2
        echo "      of this replicate did NOT train the same model and the 2x2 has a" >&2
        echo "      third factor. Check with:" >&2
        echo "        git -C ${TM_PROJ} diff --stat ${CODE_WAS#code=} -- Source/" >&2
        echo "      若改动碰到了模型代码，本重复的各臂训的就不是同一个模型，2x2 多出第三个因子。" >&2
    else
        echo "  code check     : matches the tree seed ${SEED} started with"
    fi
elif [ "$DRY_RUN" = "1" ]; then
    echo "  code check     : nothing recorded yet; this run would record ${CODE_NOW}"
else
    printf '%s\n' "$CODE_NOW" > "$CODE_FILE"
    echo "  code recorded  : ${CODE_FILE}  (${CODE_NOW})"
fi

# The dry run exits HERE, after pre-flight 5, not before it: the recipe
# mismatch is the one pre-flight a resumed replicate is most likely to trip,
# and a "plan" that cannot report it is not a plan. Nothing above this line
# creates or modifies state when DRY_RUN=1.
# 干跑在**预检 5 之后**退出，而不是之前：配方不一致是续跑时最容易踩的一道闸，
# 一个报不出它的"计划"不算计划。DRY_RUN=1 时本行以上不创建、不修改任何状态。
if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "  TM_2X2_DRY_RUN=1 — pre-flights passed, plan printed, nothing launched."
    echo "  TM_2X2_DRY_RUN=1 —— 预检通过、计划已打印，未启动任何训练。"
    exit 0
fi

# --------------------------------------------------------------------------
# Pre-flight 6 — SINGLE DRIVER PER SEED (a lock, not a convention).
#
# Sequential mode makes concurrent invocations LIKELY, not hypothetical: the
# driver blocks for ~9 days, so an SSH drop that kills the driver leaves the
# current arm running under nohup while the operator, seeing no driver, re-runs
# the script. That second driver would find the arm's dir still epoch-less
# (FRESH — the first epoch alone takes ~10 h), re-seed it, and launch a SECOND
# copy of the same arm onto the same modelDir. Two arms then cost
# 2 x 136 GiB = 272 GiB on a 251 GiB box, which is precisely the swap-thrash
# that already cost this project two days (brief_experiments 7.1).
#
# `mkdir` is atomic on POSIX filesystems, so it is the lock primitive here.
# A lock whose recorded PID is gone (same host) is stale and is taken over; a
# lock from another host cannot be adjudicated and is refused.
#
# 预检 6 —— 每个 seed 只允许一个 driver（用锁，不靠自觉）。串行模式让"并发调用"从
# 假想变成大概率事件：driver 要阻塞约 9 天，SSH 一掉 driver 就没了，而那个臂在 nohup
# 下继续跑；操作者看不到 driver，自然会重跑脚本。第二个 driver 会发现该臂目录还没有
# 任何 epoch（光第一个 epoch 就要 ~10 小时，被判为 FRESH），于是重新种权重、朝同一个
# modelDir 再起一份同样的臂。两个臂 = 272 GiB，251 GiB 的机器上必然换页颠簸——正是
# 之前已经害掉两天的那个故障。mkdir 在 POSIX 上是原子的，因此用它当锁原语。
# --------------------------------------------------------------------------
LOCK_DIR="${SHARED_DIR}/.driver.lock"
LOCK_HELD=0
tm_2x2_release_lock() {
    [ "$LOCK_HELD" = "1" ] || return 0
    rm -f "${LOCK_DIR}/owner" 2>/dev/null
    rmdir "$LOCK_DIR" 2>/dev/null
    LOCK_HELD=0
}
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    LOCK_OWNER=$(cat "${LOCK_DIR}/owner" 2>/dev/null || echo "")
    LOCK_PID=$(printf '%s' "$LOCK_OWNER" | awk '{print $1}')
    LOCK_HOST=$(printf '%s' "$LOCK_OWNER" | awk '{print $2}')
    THIS_HOST=$(hostname 2>/dev/null || echo unknown)
    if [ -n "$LOCK_PID" ] && [ "$LOCK_HOST" = "$THIS_HOST" ] && \
       ! kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "  driver lock    : stale (pid ${LOCK_PID} on ${THIS_HOST} is gone) — taking over"
        rm -f "${LOCK_DIR}/owner" 2>/dev/null
        rmdir "$LOCK_DIR" 2>/dev/null
        mkdir "$LOCK_DIR" 2>/dev/null || { echo "ERROR: could not take over ${LOCK_DIR}" >&2; exit 1; }
    else
        echo "" >&2
        echo "ERROR: another driver already owns seed ${SEED}." >&2
        echo "       lock  : ${LOCK_DIR}" >&2
        echo "       owner : ${LOCK_OWNER:-(unreadable)}" >&2
        echo "       Two drivers on one seed launch DUPLICATE arms onto the same" >&2
        echo "       modelDir and double host RAM. Attach to the running one instead" >&2
        echo "       (tmux / tail -f its log). If you are certain it is dead, remove" >&2
        echo "       the lock by hand: rmdir ${LOCK_DIR}" >&2
        echo "       同一个 seed 上跑两个 driver 会朝同一个 modelDir 起重复的臂，并让主机" >&2
        echo "       内存翻倍。请去接管正在跑的那个；确认它确实已死，再手工删锁。" >&2
        exit 1
    fi
fi
LOCK_HELD=1
printf '%s %s %s\n' "$$" "$(hostname 2>/dev/null || echo unknown)" "$(date '+%F %T')" \
    > "${LOCK_DIR}/owner"
trap 'tm_2x2_release_lock' EXIT
echo "  driver lock    : ${LOCK_DIR} (pid $$)"

# --------------------------------------------------------------------------
# assert_arm_not_running <a> — the lock covers drivers; this covers ARMS.
#
# An arm outlives its driver by design (nohup), so an arm can be alive with no
# lock anywhere: driver killed on day 3, lock released or stale-collected,
# arm still training. Re-seeding and re-launching that arm's modelDir while the
# first copy is still writing into it corrupts both. Ask the process table
# directly — it is the only thing that actually knows.
# assert_arm_not_running —— 锁管的是 driver，这里管的是**臂**。臂在设计上就会比
# driver 活得久（nohup），所以完全可能出现"没有锁、但臂还在跑"的状态。此时重新种权重
# 并再起一份，会让两份同时往同一个 modelDir 写，双双损坏。直接问进程表。
# --------------------------------------------------------------------------
assert_arm_not_running() {
    local a="$1" pids hits
    # A missing pgrep must NOT make this gate a silent pass. This is the only
    # thing standing between "the driver died and the arm survived" and two
    # processes writing one modelDir, so refuse rather than assume. procps is
    # present on the target box; if it ever is not, TM_2X2_FORCE=1 is the
    # deliberate, logged override.
    # pgrep 缺失时绝不能让这道闸静默放行——它是"driver 死了臂还活着"与"两个进程写同一个
    # modelDir"之间唯一的分界线，所以拒绝执行而不是默认没问题。目标机上 procps 是有的；
    # 万一没有，用 TM_2X2_FORCE=1 显式（且留痕地）绕过。
    if ! command -v pgrep >/dev/null 2>&1; then
        echo "ERROR: pgrep not found — cannot prove ${ARM_NAME[$a]} is not already" >&2
        echo "       running. Launching blind risks a SECOND copy writing into" >&2
        echo "       ${ARM_CKPT[$a]}. Install procps, or set TM_2X2_FORCE=1." >&2
        [ "$FORCE" = "1" ] || return 1
        echo "WARN: TM_2X2_FORCE=1 — duplicate-arm check skipped (no pgrep)." >&2
        return 0
    fi
    # `pgrep -f` (PIDs only) is portable; `-a` is GNU-only and fails silently on
    # BSD/macOS, which would turn this gate into a no-op exactly where a no-op is
    # worst. Get the PIDs portably, then render them with ps.
    # 只用可移植的 `pgrep -f` 取 PID；`-a` 是 GNU 扩展，在 BSD/macOS 上会静默失败，
    # 那会把这道闸变成空操作。取到 PID 后再用 ps 渲染。
    pids=$(pgrep -f "modelDir ./${ARM_CKPT[$a]}/" 2>/dev/null | tr '\n' ' ')
    pids=$(printf '%s' "$pids" | sed 's/ *$//')
    [ -n "$pids" ] || return 0
    hits=$(ps -o pid=,args= -p "$(printf '%s' "$pids" | tr ' ' ',')" 2>/dev/null)
    [ -n "$hits" ] || hits="  pids: ${pids}"
    echo "" >&2
    echo "ERROR: ${ARM_NAME[$a]} is ALREADY RUNNING — refusing to launch a second copy." >&2
    echo "       modelDir: ${ARM_CKPT[$a]}" >&2
    printf '%s\n' "$hits" | sed 's/^/         /' >&2
    echo "       This is the 'driver died, arm survived' case. Do NOT re-launch:" >&2
    echo "       let it finish (tail -f ${ARM_LOG[$a]}*), then re-run this script —" >&2
    echo "       it will classify the arm DONE and continue with the next one." >&2
    echo "       这是「driver 死了、臂还活着」的情况。别重起：等它跑完，再重跑本脚本，" >&2
    echo "       它会把该臂判为 DONE 并接着跑下一个。" >&2
    echo "       If those processes are genuinely dead wood, kill them first." >&2
    return 1
}

# --------------------------------------------------------------------------
# ONE strip pass per replicate, then copy. Both seeded arms must be provably
# identical at initialization: a second strip pass would be a second torch.save
# and a second opportunity to seed a different checkpoint. In sequential mode
# the two seeded arms start DAYS apart and possibly in different invocations of
# this script — so if the strip already exists, REUSE it and verify its sha256
# against the value recorded beside it, rather than stripping again.
# 每个重复只 strip 一次，然后复制。两个 seeded 臂在初始化时必须可证明地一致：第二次
# strip 就是第二次 torch.save，也就是第二次种错权重的机会。串行模式下这两个臂相隔
# 好几天、甚至分属不同次脚本调用——因此 strip 产物若已存在，就复用它并校验 sha256
# （与旁边记录的值比对），绝不重新 strip。
# --------------------------------------------------------------------------
# This block runs UNCONDITIONALLY, even when both seeded arms are already DONE
# and only the two random arms remain. That is deliberate: SHA_SHARED is what
# the DONE-arm audit below compares each finished arm's copy against, so
# skipping the strip to save a torch.load would silently turn that audit into a
# comparison with the empty string — i.e. it would "pass" by accident.
# 本段无条件执行，即使两个 seeded 臂都已 DONE、只剩随机臂要跑。这是有意的：下面对已完成
# 臂的审计要拿 SHA_SHARED 做比对，为省一次 torch.load 而跳过，会让该审计变成"与空串比对"
# ——于是它会莫名其妙地"通过"。
SHA_SHARED=""
echo ""
echo "---- shared Stage-1 init ---------------------------------------------"
mkdir -p "$SHARED_DIR" || exit 1
if [ -s "$SHARED_INIT" ]; then
    SHA_SHARED=$(tm_2x2_sha256 "$SHARED_INIT")
    echo "  reuse (already stripped by an earlier invocation of seed ${SEED}):"
    echo "    ${SHARED_INIT}"
    echo "    sha256 now      : ${SHA_SHARED}"
    if [ -s "$SHARED_SHA_FILE" ]; then
        SHA_RECORDED=$(cat "$SHARED_SHA_FILE")
        echo "    sha256 recorded : ${SHA_RECORDED}"
        if [ "$SHA_SHARED" != "$SHA_RECORDED" ]; then
            echo "ERROR: the shared Stage-1 init CHANGED since it was stripped." >&2
            echo "       Every seeded arm of seed ${SEED} must start from bit-identical" >&2
            echo "       weights; this replicate can no longer assert that. Investigate," >&2
            echo "       then move ${SHARED_DIR} aside and restart the replicate." >&2
            exit 1
        fi
        echo "    -> match; not re-stripping (a second strip is a second chance to seed the wrong weights)."
    else
        echo "    WARN: no recorded sha256 beside it (${SHARED_SHA_FILE} missing)." >&2
        echo "          Recording the current value now; it cannot prove the file is" >&2
        echo "          the one earlier arms used — check those arms' logs if in doubt." >&2
        printf '%s\n' "$SHA_SHARED" > "$SHARED_SHA_FILE"
    fi
else
    echo "  strip: ${STAGE1} -> ${SHARED_INIT}"
    if ! "$TM_PY" Source/Tools/strip_stage1_opt.py "$STAGE1" "$SHARED_INIT"; then
        echo "ERROR: strip_stage1_opt.py failed on ${STAGE1}" >&2
        exit 1
    fi
    if [ ! -s "$SHARED_INIT" ]; then
        echo "ERROR: ${SHARED_INIT} is missing or empty after the strip." >&2
        exit 1
    fi
    SHA_SHARED=$(tm_2x2_sha256 "$SHARED_INIT")
    # A recorded sha with no file beside it means an EARLIER invocation of this
    # seed already stripped, already seeded arms from that strip, and the .pth
    # was deleted afterwards. Overwriting the record here would destroy the only
    # evidence that the arms already trained and the arms still to train started
    # from the same weights. Compare instead, and refuse on a mismatch.
    # 有记录但没文件，说明本 seed 的**上一次**调用已经 strip 过、已经用它种过臂，之后
    # 文件被删了。此处直接覆写记录，就会销毁"已训的臂"与"待训的臂"出自同一份权重的唯一
    # 证据。因此改为比对；不一致就拒绝继续。
    if [ -s "$SHARED_SHA_FILE" ]; then
        SHA_RECORDED=$(cat "$SHARED_SHA_FILE")
        if [ "$SHA_SHARED" != "$SHA_RECORDED" ]; then
            echo "ERROR: re-stripping ${STAGE1} did NOT reproduce the checkpoint that seed" >&2
            echo "       ${SEED} was started with." >&2
            echo "         recorded : ${SHA_RECORDED}" >&2
            echo "         re-strip : ${SHA_SHARED}" >&2
            echo "       Arms already trained under seed ${SEED} were seeded from the" >&2
            echo "       RECORDED weights; arms trained from here would use different ones," >&2
            echo "       so this would stop being a shared-checkpoint design." >&2
            echo "       Either restore the original ${SHARED_INIT}, or move ${SHARED_DIR}" >&2
            echo "       and every Checkpoint_*_s${SEED} aside and restart the replicate." >&2
            echo "       要么恢复原来的 ${SHARED_INIT}，要么把 ${SHARED_DIR} 与所有" >&2
            echo "       Checkpoint_*_s${SEED} 一起挪走、整个重复重跑。" >&2
            exit 1
        fi
        echo "  sha256 shared  : ${SHA_SHARED}  (re-strip reproduced the recorded value)"
    else
        printf '%s\n' "$SHA_SHARED" > "$SHARED_SHA_FILE"
        echo "  sha256 shared  : ${SHA_SHARED}  (recorded in ${SHARED_SHA_FILE})"
    fi
fi

# Audit the seeded arms that are already DONE: their copy must still match, or
# a previous wave was seeded from something else and the control is fabricated.
# 审计已完成的 seeded 臂：它们的副本必须仍然一致，否则说明之前那一波种的是别的权重，
# 对照组就是伪造的。
for a in 0 1 2 3; do
    [ "${ARM_SEEDED[$a]}" = "1" ] || continue
    [ "${ARM_STATE[$a]}" = "DONE" ] || continue
    _copy="${ARM_CKPT[$a]}/model_stage1_init.pth"
    if [ -s "$_copy" ]; then
        _sha=$(tm_2x2_sha256 "$_copy")
        if [ "$_sha" = "$SHA_SHARED" ]; then
            echo "  audit ${ARM_NAME[$a]} (DONE): init still matches the shared sha256."
        else
            echo "  ERROR ${ARM_NAME[$a]} (DONE): its model_stage1_init.pth does NOT match" >&2
            echo "        the shared init (${_sha} != ${SHA_SHARED}). That arm was seeded" >&2
            echo "        from different weights — the 2x2 is not a shared-checkpoint design." >&2
            exit 1
        fi
    else
        echo "  WARN  ${ARM_NAME[$a]} (DONE): model_stage1_init.pth no longer present;" >&2
        echo "        cannot re-verify its seed. Its log's '${LOAD_MARK}' line is the" >&2
        echo "        remaining evidence (checked below)." >&2
    fi
done

seed_arm() {   # <ckpt_dir> — copy the ONE stripped init in and index it for JF
    local dir="$1"
    mkdir -p "$dir" || return 1
    cp "$SHARED_INIT" "$dir/model_stage1_init.pth" || return 1
    printf 'last model name:%s\n%s\n' model_stage1_init.pth model_stage1_init.pth \
        > "$dir/checkpoint.list" || return 1
    local sha
    sha=$(tm_2x2_sha256 "$dir/model_stage1_init.pth")
    echo "  sha256 ${dir}: ${sha}"
    if [ "$sha" != "$SHA_SHARED" ]; then
        echo "ERROR: ${dir}/model_stage1_init.pth does not match the shared init!" >&2
        return 1
    fi
    return 0
}

prepare_arm_dir() {   # <a> — make the modelDir the arm will actually launch with
    local a="$1"
    if [ "${ARM_SEEDED[$a]}" = "1" ]; then
        seed_arm "${ARM_CKPT[$a]}" || return 1
    else
        # Random init is enforced the way run_frozen_arms_F.sh does it: an
        # EMPTY modelDir with nothing for JF to load. No config flag involved.
        # 随机初始化沿用 run_frozen_arms_F.sh 的做法：给一个空 modelDir，JF 无
        # 权重可加载。与 config 开关无关。
        mkdir -p "${ARM_CKPT[$a]}" || return 1
        if [ -e "${ARM_CKPT[$a]}/checkpoint.list" ]; then
            echo "ERROR: random-control arm ${ARM_NAME[$a]} has a checkpoint.list in" >&2
            echo "       ${ARM_CKPT[$a]} — JF would load something and the control would" >&2
            echo "       be fabricated. Remove it or move the directory aside." >&2
            return 1
        fi
        echo "  empty modelDir : ${ARM_CKPT[$a]}  (random SAR trunk)"
    fi
    return 0
}

# --------------------------------------------------------------------------
# launch_arm <a> — start arm <a>, set ARM_PID[a].
#
# Not tm_train(): tm_train pins CUDA_VISIBLE_DEVICES to 0..N-1, which cannot
# place concurrent arms on disjoint devices, and it does not hand back the PID.
# The flag list below is the published arms' recipe VERBATIM
# (Scripts/legacy/run_frozen_arms_F.sh) — in sequential mode including
# --dataloaderNum 8 and --shuffle_buffer 100000, i.e. no deviation at all.
# 不用 tm_train()：它把 CUDA_VISIBLE_DEVICES 写死成 0..N-1，无法把并发臂放到互不
# 重叠的卡上，而且不回传 PID。下面的参数表逐字复制论文各臂的配方——串行模式下连
# --dataloaderNum 8 / --shuffle_buffer 100000 都一样，没有任何偏差。
#
# The background job is started here (not inside a command substitution) so the
# PID is a direct child of THIS shell and `wait` can reap its exit status.
# 后台任务在这里直接起（不放进命令替换），PID 就是本 shell 的直接子进程，`wait`
# 才能拿到退出码。
# --------------------------------------------------------------------------
launch_arm() {
    local a="$1"
    local model="${ARM_MODEL[$a]}" devices="${ARM_GPUS[$a]}" port="${ARM_PORT[$a]}"
    local ckpt_dir="${ARM_CKPT[$a]}" log="${ARM_LOG[$a]}"

    # Preserve any earlier log rather than clobbering the evidence.
    # 保留旧日志，不要把证据覆盖掉。
    if [ -f "$log" ]; then
        mv "$log" "${log}.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
    fi

    CUDA_VISIBLE_DEVICES="$devices" JF_PROGRESS_COLUMNS=400 \
    nohup "$TM_PY" -u Source/main.py \
        --root_path "$TM_PROJ" \
        --mode train --batchSize "$BATCH" --gpu "$GPUS_PER_ARM" --port "$port" \
        --dist True \
        --trainListPath "$MANIFEST_ARG" --imgNum "$REAL_TOTAL" \
        --maxEpochs "$EPOCHS" --auto_save_num 1 \
        --dataloaderNum "$TM_DATALOADER_NUM" --dataloaderType iterable \
        --sampleNum 1 --lr "${TM_LR:-1e-3}" \
        --modelName "$model" --dataset tmdataset --is_stage_2 True \
        --sar_size "$TM_SAR_SIZE" --matching_dim "$TM_MATCHING_DIM" \
        --fine_dim 32 --fine_margin 24 --shuffle_buffer "$TM_SHUFFLE_BUFFER" \
        --match_w4 1.0 --match_w8 0.5 --match_sigma4 1.0 --match_sigma8 0.5 \
        --match_primary_stride 4 --match_w_fine 1.0 --match_sigma_fine 1.0 \
        --fine_warmup_epochs 3 \
        --log "./log_${ckpt_dir}/" --outputDir "./result_${ckpt_dir}/" \
        --resultImgDir "./ResultImg_${ckpt_dir}/" \
        --modelDir "./${ckpt_dir}/" > "$log" 2>&1 &
    ARM_PID[$a]=$!
    RUNNING_ARMS="${RUNNING_ARMS}${RUNNING_ARMS:+ }${a}"
}

# --------------------------------------------------------------------------
# THE LOAD VERIFICATION GATE.
#
# A silently-missing Stage-1 init turns a headline arm into a SECOND random
# control — the arms still train, the numbers still look sane, and the paper's
# conclusion is fabricated. That is the one failure mode worth killing a
# 50-hour run over, so this gate is hard, and in sequential mode it runs
# IMMEDIATELY AFTER THE ARM STARTS, before hours are committed to it.
#
# The markers are read off the sources, not off memory:
#   _interface_base.py:422  -> "loaded trainable_tail (missing=..., unexpected=...)"
#   _meta_ops.py:151        -> "No checkpoint found; starting from scratch."
#   train_proc.py:117       -> "Start iteration!"   (logged AFTER restore_model(),
#                              so waiting for it removes the old blind grace sleep:
#                              once it appears, the load either happened or never will)
#
# Seeded arms MUST print the load line; random arms MUST NOT.
#
# 加载验证闸。Stage-1 init 静默缺失会把主臂退化成第二个随机对照——照样训练、数字
# 照样正常，而论文结论已被伪造。这是唯一值得为之杀掉 50 小时任务的失败模式，因此
# 这道闸是硬的；串行模式下它在臂刚启动时就跑完，绝不先押上几十小时。
# 三个标记全部取自源码。'Start iteration!' 在 restore_model() 之后打印，因此等到它
# 出现就可以断言"加载与否已成定局"，不再需要原来那个盲等的 grace sleep。
# --------------------------------------------------------------------------
kill_running_arms() {
    local a waited
    for a in $RUNNING_ARMS; do
        [ -n "${ARM_PID[$a]}" ] || continue
        echo "  killing ${ARM_NAME[$a]} (pid ${ARM_PID[$a]}) ..." >&2
        pkill -P "${ARM_PID[$a]}" 2>/dev/null
        kill "${ARM_PID[$a]}" 2>/dev/null
    done
    # Confirm, then escalate. A DDP rank that ignores SIGTERM keeps its GPU
    # memory, and the next invocation's GPU pre-flight then refuses to start on
    # a box that looks busy but is only haunted. Give them 30 s, then SIGKILL.
    # 先确认再升级：吃掉 SIGTERM 的 DDP rank 会一直占着显存，下次调用的 GPU 预检就会
    # 因为"机器很忙"而拒绝启动——其实只是尸体没收干净。给 30 秒，然后 SIGKILL。
    waited=0
    while [ "$waited" -lt 30 ]; do
        local alive=0
        for a in $RUNNING_ARMS; do
            [ -n "${ARM_PID[$a]}" ] || continue
            kill -0 "${ARM_PID[$a]}" 2>/dev/null && alive=1
        done
        [ "$alive" = "0" ] && return 0
        sleep 5
        waited=$((waited + 5))
    done
    for a in $RUNNING_ARMS; do
        [ -n "${ARM_PID[$a]}" ] || continue
        if kill -0 "${ARM_PID[$a]}" 2>/dev/null; then
            echo "  ${ARM_NAME[$a]} (pid ${ARM_PID[$a]}) ignored SIGTERM — SIGKILL" >&2
            pkill -9 -P "${ARM_PID[$a]}" 2>/dev/null
            kill -9 "${ARM_PID[$a]}" 2>/dev/null
        fi
    done
    echo "  NOTE: check for orphaned ranks holding GPU memory:" >&2
    echo "        pgrep -af 'Source/main.py' ; nvidia-smi" >&2
}

gate_fail() {
    echo "" >&2
    echo "!!!!!! 2x2 seed ${SEED} LOAD GATE FAILED: $* " >&2
    if [ "$KILL_ON_FAIL" = "1" ]; then
        kill_running_arms
        echo "  Running arm(s) killed. Investigate, remove the affected" >&2
        echo "  Checkpoint_*_s${SEED} dir(s), then re-run this script — completed arms" >&2
        echo "  will be skipped automatically." >&2
    else
        echo "  TM_2X2_KILL_ON_FAIL=0 — arm(s) LEFT RUNNING. Do not use their output" >&2
        echo "  until the initialization is explained." >&2
    fi
    exit 1
}

assert_running_alive() {   # any launched arm that died invalidates the wait
    local a
    for a in $RUNNING_ARMS; do
        if ! kill -0 "${ARM_PID[$a]}" 2>/dev/null; then
            echo "  arm ${ARM_NAME[$a]} (pid ${ARM_PID[$a]}) exited early; last 30 log lines:" >&2
            tail -n 30 "${ARM_LOG[$a]}" >&2
            gate_fail "arm ${ARM_NAME[$a]} died before its initialization was verified"
        fi
    done
}

# gate_arm <a> — block until arm <a>'s initialization is provably correct.
gate_arm() {
    local a="$1" log="${ARM_LOG[$a]}" waited=0 poll=10 line
    echo "  gate ${ARM_NAME[$a]}: waiting for '${START_MARK}' (timeout ${LOAD_TIMEOUT}s) ..."
    while ! grep -q "$START_MARK" "$log" 2>/dev/null; do
        assert_running_alive
        if [ "$waited" -ge "$LOAD_TIMEOUT" ]; then
            echo "  ${ARM_NAME[$a]} log tail:" >&2
            tail -n 30 "$log" >&2
            gate_fail "arm ${ARM_NAME[$a]} never reached '${START_MARK}' within ${LOAD_TIMEOUT}s"
        fi
        sleep "$poll"
        waited=$((waited + poll))
    done

    if [ "${ARM_SEEDED[$a]}" = "1" ]; then
        line=$(grep -m1 "$LOAD_MARK" "$log" 2>/dev/null)
        if [ -z "$line" ]; then
            echo "  ${ARM_NAME[$a]} log tail:" >&2
            tail -n 30 "$log" >&2
            gate_fail "seeded arm ${ARM_NAME[$a]} reached '${START_MARK}' WITHOUT logging '${LOAD_MARK}' — it is a fabricated control"
        fi
        echo "  OK  ${ARM_NAME[$a]}: ${line}"
        # Expected signature: missing=103 unexpected=2 (the 2 = align_ip.w_s/w_d,
        # dropped by design). missing=0 or unexpected=0 means the WRONG checkpoint
        # went in — warn loudly, but do not kill: it is a judgement call.
        # 预期签名 missing=103 unexpected=2（那 2 个是设计上丢弃的 align_ip 投影）。
        # missing=0 或 unexpected=0 说明种错了 ckpt——大声告警，但不自动杀（需人判断）。
        case "$line" in
            *"missing=0,"*|*"unexpected=0)"*)
                echo "  WARN ${ARM_NAME[$a]}: unexpected load signature — the shared" >&2
                echo "       Stage-1 ckpt should give missing=103, unexpected=2." >&2
                echo "       Verify BEFORE trusting this replicate." >&2
                ;;
        esac
        if grep -qi 'no checkpoint' "$log" 2>/dev/null; then
            gate_fail "seeded arm ${ARM_NAME[$a]} also reports 'no checkpoint'"
        fi
    else
        if grep -q "$LOAD_MARK" "$log" 2>/dev/null; then
            echo "  ${ARM_NAME[$a]} logged: $(grep -m1 "$LOAD_MARK" "$log")" >&2
            gate_fail "random-control arm ${ARM_NAME[$a]} LOADED a checkpoint (modelDir ${ARM_CKPT[$a]} was not empty?)"
        fi
        if grep -qi "$NOCKPT_MARK" "$log" 2>/dev/null; then
            echo "  OK  ${ARM_NAME[$a]}: random init confirmed ('${NOCKPT_MARK}', no '${LOAD_MARK}')"
        else
            echo "  OK  ${ARM_NAME[$a]}: random init confirmed (no '${LOAD_MARK}' in log)"
            echo "  WARN ${ARM_NAME[$a]}: '${NOCKPT_MARK}' not logged either — expected it" >&2
            echo "       for an empty modelDir; check JackFramework's version." >&2
        fi
    fi

    # The seed banner is printed by main.py only when TM_SEED is set. If it is
    # absent the run is unseeded and this replicate is not a controlled draw.
    # main.py 只有在设了 TM_SEED 时才打印该 banner。没有它就说明本次没固定随机源，
    # 这个重复不是一次受控抽样。
    local b
    b=$(grep -m1 'TM_SEED=' "$log" 2>/dev/null)
    if [ -z "$b" ]; then
        echo "  WARN ${ARM_NAME[$a]}: no 'TM_SEED=' banner — is the run unseeded?" >&2
    else
        echo "  seed ${ARM_NAME[$a]}: ${b}"
    fi
}

# audit_done_arm <a> — static re-check of an arm we are SKIPPING. The load gate
# applies to it too; it just runs against the log instead of a live process.
# 对被跳过的臂做静态复检：加载闸同样适用，只是对着日志而不是活进程跑。
audit_done_arm() {
    local a="$1" log="${ARM_LOG[$a]}"
    if [ ! -f "$log" ]; then
        echo "  WARN ${ARM_NAME[$a]} (DONE): ${log} is gone — cannot re-verify its" >&2
        echo "       initialization from the log. The sha256 audit above is the" >&2
        echo "       remaining evidence." >&2
        return 0
    fi
    if [ "${ARM_SEEDED[$a]}" = "1" ]; then
        if grep -q "$LOAD_MARK" "$log" 2>/dev/null; then
            echo "  audit ${ARM_NAME[$a]} (DONE): $(grep -m1 "$LOAD_MARK" "$log")"
        else
            echo "ERROR: completed seeded arm ${ARM_NAME[$a]} has no '${LOAD_MARK}' in" >&2
            echo "       ${log} — it trained as a SECOND RANDOM CONTROL. Its checkpoints" >&2
            echo "       are unusable; move ${ARM_CKPT[$a]} aside and re-run." >&2
            exit 1
        fi
    else
        if grep -q "$LOAD_MARK" "$log" 2>/dev/null; then
            echo "ERROR: completed random arm ${ARM_NAME[$a]} DID load a checkpoint" >&2
            echo "       ($(grep -m1 "$LOAD_MARK" "$log")). It is not a random control;" >&2
            echo "       move ${ARM_CKPT[$a]} aside and re-run." >&2
            exit 1
        fi
        echo "  audit ${ARM_NAME[$a]} (DONE): random init confirmed from the log."
    fi
}

# --------------------------------------------------------------------------
# wait_for_arm <a> — block until arm <a> exits, then prove it FINISHED:
#   * the final-epoch checkpoint exists and is non-empty, and
#   * the tail of the log does not end in a traceback.
# A heartbeat prints the last progress line so a days-long wait is observable.
# 等待某个臂退出，然后证明它是"跑完"而不是"跑挂"：末代 ckpt 存在且非空，且日志尾部
# 没有 traceback。心跳会定期打印最后一行进度，好让几天的等待可观察。
# --------------------------------------------------------------------------

# pid_still_this_arm <pid> — 0 if <pid> still looks like a JackFramework run.
#
# `kill -0` alone is not enough over a ~56 h wait: if the arm dies at hour 2 and
# the kernel recycles its PID (Linux wraps at /proc/sys/kernel/pid_max, and a box
# running dataloader workers churns through PIDs fast), the loop below would keep
# "waiting" on a stranger's process for days and the campaign would silently
# stall with three arms unlaunched. `ps` is the tiebreak. Two consecutive misses
# are required so a transient ps hiccup cannot abort a healthy arm, and an EMPTY
# ps result is treated as "still ours" — unreadable is not the same as reused.
# Breaking out is safe in every case: the completion check below trusts the final
# checkpoint, not the exit status.
# 光靠 `kill -0` 在 ~56 小时的等待里不够：若该臂第 2 小时就死了、PID 又被内核回收，
# 循环会对着别人的进程"继续等"好几天，整个 campaign 悄无声息地停摆、剩下三个臂永不启动。
# 用 `ps` 做仲裁；要求连续两次未命中，避免 ps 偶发抖动误杀健康任务；ps 输出为空按"仍是本臂"
# 处理——读不到不等于被复用。跳出循环在任何情况下都安全：下面的完成判定只认末代 ckpt。
pid_still_this_arm() {
    local pid="$1" args
    args=$(ps -o args= -p "$pid" 2>/dev/null | tr -d '\n')
    [ -n "$args" ] || return 0
    case "$args" in *"Source/main.py"*) return 0 ;; esac
    return 1
}

wait_for_arm() {
    local a="$1" pid="${ARM_PID[$a]}" log="${ARM_LOG[$a]}"
    local started elapsed since_beat=0 rc=0 last strikes=0
    started=$(date +%s)
    echo "  waiting for ${ARM_NAME[$a]} (pid ${pid}) — expected ~${ARM_HOURS_TAXED} h ..."
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL"
        if pid_still_this_arm "$pid"; then
            strikes=0
        else
            strikes=$((strikes + 1))
            if [ "$strikes" -ge 2 ]; then
                echo "  WARN ${ARM_NAME[$a]}: pid ${pid} is alive but is no longer a" >&2
                echo "       'Source/main.py' process — the arm exited and the PID was" >&2
                echo "       recycled. Not waiting on a stranger; verifying by checkpoint." >&2
                echo "       pid ${pid} 还活着但已不是训练进程（PID 被复用）：不再空等，" >&2
                echo "       改用末代 ckpt 判定该臂是否真的跑完。" >&2
                break
            fi
        fi
        since_beat=$((since_beat + POLL))
        if [ "$since_beat" -ge "$HEARTBEAT" ]; then
            since_beat=0
            elapsed=$(( ($(date +%s) - started) / 60 ))
            last=$(tail -c 4000 "$log" 2>/dev/null | tr '\r' '\n' | grep -v '^$' | tail -n 1)
            printf '  [%s] %s alive %dh%02dm | %s\n' \
                "$(date '+%F %H:%M')" "${ARM_NAME[$a]}" \
                "$((elapsed / 60))" "$((elapsed % 60))" "${last:-(no output yet)}"
        fi
    done
    # The process is a direct child, so `wait` still holds its exit status.
    # 进程是本 shell 的直接子进程，`wait` 仍能取到退出码。
    wait "$pid" 2>/dev/null || rc=$?
    RUNNING_ARMS=$(echo "$RUNNING_ARMS" | tr ' ' '\n' | grep -v "^${a}\$" | paste -sd' ' -)
    elapsed=$(( ($(date +%s) - started) / 60 ))
    echo "  ${ARM_NAME[$a]} exited rc=${rc} after $((elapsed / 60))h$((elapsed % 60))m at $(date '+%F %H:%M')"

    local final="${ARM_CKPT[$a]}/model_epoch_${FINAL_EPOCH}.pth"
    local tb=0
    if tail -n 80 "$log" 2>/dev/null | grep -qE 'Traceback \(most recent call last\)|ProcessRaisedException|CUDA out of memory|Killed'; then
        tb=1
    fi
    local trunc=0
    arm_final_truncated "${ARM_CKPT[$a]}" && trunc=1
    if [ ! -s "$final" ] || [ "$tb" = "1" ] || [ "$trunc" = "1" ]; then
        echo "" >&2
        echo "!!!!!! arm ${ARM_NAME[$a]} DID NOT FINISH" >&2
        [ -s "$final" ] || echo "       missing final checkpoint: ${final}" >&2
        [ "$tb" = "0" ] || echo "       the log tail contains a traceback / fatal error" >&2
        [ "$trunc" = "0" ] || echo "       final checkpoint is TRUNCATED (much smaller than this arm's earlier epochs)" >&2
        echo "       epochs present: $(arm_epochs_present "${ARM_CKPT[$a]}")" >&2
        echo "       last 30 log lines:" >&2
        tail -n 30 "$log" >&2
        echo "" >&2
        echo "       Stopping the replicate here — the remaining arms are NOT launched." >&2
        echo "       Fix the cause, move ${ARM_CKPT[$a]} aside, and re-run this script:" >&2
        echo "       finished arms are skipped automatically." >&2
        echo "       本次重复到此停止，后续臂不再启动。修好原因、把该目录挪走后重跑本脚本，" >&2
        echo "       已完成的臂会自动跳过。" >&2
        exit 1
    fi
    if [ "$rc" != "0" ]; then
        echo "  WARN ${ARM_NAME[$a]}: exit code ${rc} but ${final} exists and the log is" >&2
        echo "       clean — probably the known finalize/teardown lag. Treating as done." >&2
    fi
    echo "  DONE ${ARM_NAME[$a]}: ${final} ($(arm_epochs_present "${ARM_CKPT[$a]}") epochs present)"
}

# ==========================================================================
# RUN
# ==========================================================================
if [ "$CONCURRENT" = "1" ]; then
    # Concurrent: slice the pre-flight device list into GPUS_PER_ARM-sized groups,
    # one per arm that still has to run. Skipped arms consume nothing.
    # 并发：把预检拿到的卡列表按每臂 GPUS_PER_ARM 张切片，只分给还要跑的臂；
    # 被跳过的臂不占卡。
    IFS=',' read -r -a _dev <<< "$PREFLIGHT_DEVICES"
    _k=0
    for a in 0 1 2 3; do
        [ "${ARM_STATE[$a]}" = "DONE" ] && continue
        _grp=""
        for _j in $(seq 0 $((GPUS_PER_ARM - 1))); do
            _grp="${_grp}${_grp:+,}${_dev[$_k]:-}"
            _k=$((_k + 1))
        done
        case "$_grp" in
            *,,*|,*|*,) echo "ERROR: not enough devices to place ${ARM_NAME[$a]} (got '${_grp}')." >&2; exit 1 ;;
            '')          echo "ERROR: no device for ${ARM_NAME[$a]}." >&2; exit 1 ;;
        esac
        ARM_GPUS[$a]="$_grp"
    done
fi

echo ""
echo "---- running ---------------------------------------------------------"
for a in 0 1 2 3; do
    if [ "${ARM_STATE[$a]}" = "DONE" ]; then
        echo ""
        echo "  == ${ARM_NAME[$a]} — SKIPPED (already complete) =="
        audit_done_arm "$a"
        continue
    fi
    # LIVENESS BEFORE ANYTHING DESTRUCTIVE. This must precede the PARTIAL/TRUNC
    # `mv` below: a half-finished arm is the SAME state as "the driver died on
    # day 3 and the arm is still training" (its dir holds epochs 0..2 and no
    # final one). Archiving first would rename the modelDir out from under a
    # live process — on Linux the open directory handle follows the rename, so
    # the arm silently keeps saving into `<dir>.partial.<stamp>/` and its own
    # log path no longer matches its checkpoints. Ask the process table first.
    # 任何破坏性操作之前先查活性：必须排在下面 PARTIAL/TRUNC 的 `mv` 之前。"半截的臂"
    # 与"driver 死了、臂还在跑"是同一种目录状态；先归档就会把 modelDir 在活进程脚下改名
    # （Linux 上打开的目录句柄跟着 rename 走），该臂会继续默默往 `<dir>.partial.<stamp>/`
    # 里存权重，日志与 ckpt 从此对不上。先问进程表。
    assert_arm_not_running "$a" || exit 1

    case "${ARM_STATE[$a]}" in
        PARTIAL|TRUNC)
        # Only reachable with TM_2X2_RESTART_PARTIAL=1 (the plan section exits
        # otherwise). Archive loudly, never overwrite in place.
        # 只有 TM_2X2_RESTART_PARTIAL=1 才会走到这里。大声归档，绝不原地覆盖。
        _archive="${ARM_CKPT[$a]}.partial.$(date +%Y%m%d_%H%M%S)"
        echo ""
        echo "  == ${ARM_NAME[$a]} — ${ARM_STATE[$a]}, ARCHIVING AND RETRAINING CLEAN =="
        echo "     TM_2X2_RESTART_PARTIAL=1: mv ${ARM_CKPT[$a]} -> ${_archive}"
        mv "${ARM_CKPT[$a]}" "$_archive" || exit 1
        ;;
    esac

    echo ""
    echo "  == ${ARM_NAME[$a]} (${ARM_MODEL[$a]}) — arm $((a + 1)) of ${N_ARMS} =="

    if [ "$CONCURRENT" != "1" ]; then
        # Sequential: pick devices and check the port NOW, at this arm's start.
        # 串行：此刻现取卡、现查端口。
        if ! ARM_GPUS[$a]=$(pick_devices "$GPUS_PER_ARM" "arm ${ARM_NAME[$a]}"); then
            echo "       Arms finished so far are safe: re-run this script when the box" >&2
            echo "       frees up and it will resume at ${ARM_NAME[$a]}." >&2
            echo "       已跑完的臂不受影响：等机器空出来重跑本脚本，会从 ${ARM_NAME[$a]} 继续。" >&2
            exit 1
        fi
        if ! port_free "${ARM_PORT[$a]}"; then
            echo "ERROR: port ${ARM_PORT[$a]} is bound; ${ARM_NAME[$a]} cannot rendezvous." >&2
            echo "       Re-run with a different port_base (arg 4) once it frees up." >&2
            [ "$FORCE" = "1" ] || exit 1
        fi
        check_host_ram "arm ${ARM_NAME[$a]}" || exit 1
    fi

    prepare_arm_dir "$a" || exit 1
    launch_arm "$a"
    printf '  launched %-6s %-11s gpu=%-8s port=%-5s pid=%-8s log=%s\n' \
        "${ARM_NAME[$a]}" "${ARM_MODEL[$a]}" "${ARM_GPUS[$a]}" \
        "${ARM_PORT[$a]}" "${ARM_PID[$a]}" "${ARM_LOG[$a]}"

    if [ "$CONCURRENT" != "1" ]; then
        gate_arm "$a"
        wait_for_arm "$a"
    fi
done

if [ "$CONCURRENT" = "1" ]; then
    # Concurrent: everything that had to run is now up. Gate each arm, then hand
    # the box back — the caller waits on the PIDs (see the note printed below).
    # 并发：该起的臂都起来了。逐个过闸，然后把机器交还调用者，由它等这些 PID。
    echo ""
    echo "---- load verification gate (timeout ${LOAD_TIMEOUT}s each) -----------"
    for a in $RUNNING_ARMS; do
        gate_arm "$a"
    done
fi

# --------------------------------------------------------------------------
# Summary — the record to paste into the campaign log.
# 总结 —— 直接贴进实验记录的那一段。
# --------------------------------------------------------------------------
echo ""
echo "======================================================================"
if [ "$CONCURRENT" = "1" ]; then
    echo " 2x2 seed ${SEED} LAUNCHED (concurrent) — every arm passed the load gate"
else
    echo " 2x2 seed ${SEED} COMPLETE (sequential) — every arm finished and was verified"
fi
echo "======================================================================"
printf '  %-6s %-11s %-30s %-8s %-6s %-8s %s\n' \
    ARM MODEL CKPT_DIR GPUS PORT PID LOG
for a in 0 1 2 3; do
    printf '  %-6s %-11s %-30s %-8s %-6s %-8s %s\n' \
        "${ARM_NAME[$a]}" "${ARM_MODEL[$a]}" "${ARM_CKPT[$a]}" \
        "${ARM_GPUS[$a]:--}" "${ARM_PORT[$a]}" "${ARM_PID[$a]:--}" "${ARM_LOG[$a]}"
done
echo ""
echo "  seed            : ${SEED}"
echo "  mode            : ${MODE_NAME}"
echo "  shared Stage-1  : ${STAGE1}"
echo "  stripped init   : ${SHARED_INIT}"
echo "  sha256          : ${SHA_SHARED}   (identical in both seeded arm dirs)"
echo "  manifest        : ${MANIFEST}  (--imgNum ${REAL_TOTAL})"
echo "  data loader     : --dataloaderNum ${TM_DATALOADER_NUM} --shuffle_buffer ${TM_SHUFFLE_BUFFER}"
echo "  vs published    : ${LOADER_STATUS}"
echo "                    ${DETERMINISM_STATUS}"
echo "  DISCLOSE        : the determinism line above is a real difference from the"
echo "                    published Table III arms. Do NOT write 'recipe identical'"
echo "                    in the response letter without qualifying it."
echo "                    上面这行确定性差异是与已发表 Table III 各臂的真实差别，回复信里"
echo "                    不要不加限定地写\"配方完全一致\"。"
echo ""

if [ "$CONCURRENT" = "1" ]; then
cat <<NOTE
Next / 下一步 (concurrent):
  tail -f ${ARM_LOG[0]}                          # follow one arm / 跟一个臂
  pgrep -af "Source/main.py" | grep -c .          # DDP parents + ranks

  Wait for ALL FOUR PIDs to exit before the next replicate — overlapping
  replicates oversubscribe the GPUs and double host RAM:
  下一个重复必须等这四个 PID 全部退出，重叠会挤爆 GPU 并让主机内存翻倍：
      while kill -0 ${ARM_PID[0]:-0} ${ARM_PID[1]:-0} ${ARM_PID[2]:-0} ${ARM_PID[3]:-0} 2>/dev/null; do sleep 300; done

  Then re-run this script with the same seed: it verifies every arm's final
  checkpoint and reports what is missing.
  然后用同一 seed 重跑本脚本：它会校验每个臂的末代 ckpt 并报告缺什么。
NOTE
else
cat <<NOTE
Next / 下一步 (sequential):
  Nothing is running. All ${N_ARMS} arms of seed ${SEED} have
  model_epoch_${FINAL_EPOCH}.pth and passed the load gate.
  当前没有任务在跑；seed ${SEED} 的四个臂都已产出末代 ckpt 并通过加载闸。
NOTE
fi

cat <<NOTE

Evaluate each arm at model_epoch_${FINAL_EPOCH}.pth on the 20k manifest, then
summarize across seeds:
在 20k 清单上评测每个臂的 model_epoch_${FINAL_EPOCH}.pth，再跨 seed 汇总：
    TM_EVAL_CSV=/data4/datasets/eval20k_opt256_sar192/image_test_2k_256.csv \\
    TM_EVAL_N=20000 bash -c 'source Scripts/common.sh
        tm_eval ${ARM_MODEL[0]} ${ARM_CKPT[0]}/model_epoch_${FINAL_EPOCH}.pth ${ARM_TAG[0]}_20k 0
        tm_pck ${ARM_TAG[0]}_20k'
    python Source/Tools/multiseed_summary.py --seeds ${SEED} ...
NOTE
