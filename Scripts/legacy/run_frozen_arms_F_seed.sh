#!/usr/bin/env bash
# Multi-SEED replicate of the Stage-2 FROZEN-encoder double arms (S1F + RF).
# 冻结 encoder 双臂(S1F + RF)的多 seed 复现版,用于硬化 e4 的小而显著的
# S1F>RF 与 S1F≈B 差距(单 seed 不足以下结论)。
#
# Derived from run_frozen_arms_F.sh (the seed-1, UNSEEDED launch). The only
# substantive change: export TM_SEED=$SEED so main.py fixes every RNG +
# cudnn.deterministic — each seed is a reproducible independent draw of the
# matching-head init and data-shuffle order (the encoder is frozen, so those
# are the only seed-dependent factors). Config is otherwise byte-identical to
# seed-1 (--gpu 2, batch 32, subset40, maxEpochs 5, fine_warmup 3).
# 与 seed-1 唯一实质差别:导出 TM_SEED=$SEED,让 main.py 固定所有随机源;
# 其余配置与 seed-1 完全一致,保证跨 seed 可比。
#
# Usage: bash Scripts/run_frozen_arms_F_seed.sh SEED [PORT_S1F] [PORT_RF] [GPUS_S1F] [GPUS_RF]
#   e.g. round-1 (seed 2025):  bash Scripts/run_frozen_arms_F_seed.sh 2025
#        round-2 (seed 2026):  bash Scripts/run_frozen_arms_F_seed.sh 2026 8862 8863
set -u
SEED="${1:?usage: $0 SEED [PORT_S1F] [PORT_RF] [GPUS_S1F] [GPUS_RF]}"
PORT_S1F="${2:-8852}"
PORT_RF="${3:-8853}"
GPUS_S1F="${4:-0,1}"
GPUS_RF="${5:-2,3}"
export TM_SEED="$SEED"
SUF="_s${SEED}"

PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
MAN=Datasets/shard_train_256_subset40.csv
RT=$(tail -n +2 "$MAN" | awk -F, '{s+=$2} END{print s}')
echo "seed=$SEED suffix=$SUF manifest=$MAN real_total=$RT ports=$PORT_S1F/$PORT_RF gpus=$GPUS_S1F|$GPUS_RF"

S1=Checkpoint_F_stage1_256
S1F=Checkpoint_F_s2_frozen_256${SUF}
RF=Checkpoint_RF_s2_frozen_256${SUF}

# Re-run guard: refuse to wipe an existing seed run's checkpoints.
# 防重跑:该 seed 的目录已存在则拒绝,保护已训的 ckpt。
for d in "$S1F" "$RF"; do
    if [ -d "$d" ]; then
        echo "ERROR: $d exists — refusing to wipe. rm -rf it manually if intended."
        exit 1
    fi
done
mkdir -p "$S1F" "$RF"

# Pin the FINAL Stage-1 ckpt (model_epoch_9) — same source as seed-1.
# 钉死最终 Stage-1 ckpt(与 seed-1 同一 bootstrap 源)。
latest="$S1/model_epoch_9.pth"
if [ ! -f "$latest" ]; then
    echo "ERROR: $latest missing — Stage-1 F ckpt absent."
    exit 1
fi
echo "S1F bootstrap from: $latest"
"$PY" Source/Tools/strip_stage1_opt.py "$latest" "$S1F/model_stage1_init.pth" || {
    echo "ERROR: strip_stage1_opt.py failed"; exit 1; }
# A silently-missing init would degrade the headline arm into a 2nd random
# control — the one failure mode that fabricates the conclusion. Hard-stop.
# init 静默缺失会把主臂退化成第二个随机对照(恰好伪造结论),必须硬卡。
[ -s "$S1F/model_stage1_init.pth" ] || {
    echo "ERROR: model_stage1_init.pth missing/empty after strip"; exit 1; }
{ echo "last model name:model_stage1_init.pth"; echo "model_stage1_init.pth"; } > "$S1F/checkpoint.list"

launch() {  # $1=modelName $2=gpus $3=port $4=modeldir $5=log
  CUDA_VISIBLE_DEVICES=$2 JF_PROGRESS_COLUMNS=400 nohup "$PY" -u Source/main.py \
    --mode train --batchSize 32 --gpu 2 --port "$3" --dist True \
    --trainListPath "./$MAN" --imgNum "$RT" --maxEpochs 5 --auto_save_num 1 \
    --dataloaderNum 8 --dataloaderType iterable --sampleNum 1 --lr 1e-3 \
    --modelName "$1" --dataset tmdataset --is_stage_2 True --sar_size 192 \
    --fine_margin 24 --matching_dim 256 --fine_dim 32 --shuffle_buffer 100000 \
    --match_w4 1.0 --match_w8 0.5 --match_sigma4 1.0 --match_sigma8 0.5 \
    --match_primary_stride 4 --match_w_fine 1.0 --match_sigma_fine 1.0 \
    --fine_warmup_epochs 3 \
    --log "./log_$4/" --outputDir "./result_$4/" --resultImgDir "./ResultImg_$4/" \
    --modelDir "./$4/" > "$5" 2>&1 &
  echo "$!"
}

LOG_S1F=train_f_s1frozen${SUF}.log
LOG_RF=train_f_rfrozen${SUF}.log
PIDS=$(launch TM_Model_F "$GPUS_S1F" "$PORT_S1F" "$S1F" "$LOG_S1F")
echo "S1F_PID=$PIDS  (seed $SEED, strong-S1 init, frozen, gpu$GPUS_S1F, $S1F)"
PIDR=$(launch TM_Model_F "$GPUS_RF" "$PORT_RF" "$RF" "$LOG_RF")
echo "RF_PID=$PIDR  (seed $SEED, random init, frozen, gpu$GPUS_RF, $RF)"
echo "FROZEN_ARMS_SEED${SEED}_LAUNCHED S1F=$PIDS RF=$PIDR"

# Post-launch load verification: S1F log must show the trainable_tail load
# (unexpected=2 = dropped align_ip projectors) and must NOT start from scratch.
# 启动后验证:S1F 日志须出现 trainable_tail 加载(unexpected=2),且不得从零开始。
sleep 90
if grep -q 'loaded trainable_tail' "$LOG_S1F" && ! grep -qi 'no checkpoint' "$LOG_S1F"; then
    echo "S1F load check OK:"; grep 'loaded trainable_tail' "$LOG_S1F" | head -1
else
    echo "!!!!!! S1F(seed $SEED) DID NOT LOAD STAGE-1 CKPT — kill PID $PIDS and investigate"
fi
# Confirm the seed actually took (main.py prints the banner only when TM_SEED set).
# 确认 seed 真的生效(main.py 仅在设了 TM_SEED 时打印该行)。
echo "--- seed banners ---"; grep -h 'TM_SEED=' "$LOG_S1F" "$LOG_RF" | head -2
