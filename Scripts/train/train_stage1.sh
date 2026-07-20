#!/usr/bin/env bash
# Train a Stage-1 pre-alignment encoder (the thing the paper shows is redundant).
# 训练 Stage-1 预对齐编码器（论文正是要证明它在微调下是冗余的）。
#
# Two objectives / 两种目标:
#   A  contrastive PatchNCE on raw cosine   -> Checkpoint_A_stage1_256/
#      对比式 PatchNCE（原始余弦）
#   F  learned inner product + VICReg floor -> Checkpoint_F_stage1_256/
#      learned inner product + VICReg 秩下限（更强的 Stage-1 上界探针）
#
# These checkpoints are only useful as the warm-start source for the 2x2 arms —
# see Scripts/train/train_arms.sh. They are NOT matching models on their own.
# 这两个权重只作为 2x2 实验臂的热启动来源（见 train_arms.sh），本身不是匹配模型。
#
# Usage / 用法:
#   bash Scripts/train/train_stage1.sh A|F [gpus] [batch]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

VARIANT="${1:?usage: $0 A|F [gpus] [batch]}"
GPUS="${2:-4}"
BATCH="${3:-16}"
MANIFEST="${TM_TRAIN_LIST:-./Datasets/shard_train_256_full.csv}"

case "$VARIANT" in
    A) MODEL=TM_Model_A; CKPT=Checkpoint_A_stage1_256; EPOCHS="${TM_S1_EPOCHS:-6}";  PORT=8871 ;;
    F) MODEL=TM_Model_F; CKPT=Checkpoint_F_stage1_256; EPOCHS="${TM_S1_EPOCHS:-10}"; PORT=8872 ;;
    *) echo "ERROR: variant must be A or F (got '$VARIANT')" >&2; exit 1 ;;
esac

# Stage-1 is the pre-alignment stage: --is_stage_2 False.
# Stage-1 是预对齐阶段：--is_stage_2 False。
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: shard manifest not found: $MANIFEST" >&2
    exit 1
fi
if [ -d "$CKPT" ] && find "$CKPT" -maxdepth 1 -name 'model_epoch_*.pth' | grep -q .; then
    echo "ERROR: $CKPT already holds checkpoints — move it aside first." >&2
    exit 1
fi
mkdir -p "$CKPT"

REAL_TOTAL=$(tail -n +2 "$MANIFEST" | awk -F, '{s+=$2} END{print s}')
DEVICES=$(seq -s, 0 $((GPUS - 1)))
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "==== STAGE-1 ${VARIANT} (${MODEL}) gpus=${GPUS} bs=${BATCH} epochs=${EPOCHS} ===="
echo "     manifest=${MANIFEST} samples=${REAL_TOTAL} ckpt=${CKPT}"
CUDA_VISIBLE_DEVICES="$DEVICES" JF_PROGRESS_COLUMNS=400 nohup "$TM_PY" -u Source/main.py \
    --mode train --dataset tmdataset --modelName "$MODEL" --is_stage_2 False \
    --dataloaderType iterable --dataloaderNum 8 --sampleNum 1 \
    --sar_size "$TM_SAR_SIZE" --matching_dim "$TM_MATCHING_DIM" \
    --shuffle_buffer "${TM_SHUFFLE_BUFFER:-30000}" --lr "${TM_LR:-1e-3}" \
    --auto_save_num 1 --dist True --gpu "$GPUS" --batchSize "$BATCH" --port "$PORT" \
    --trainListPath "$MANIFEST" --imgNum "$REAL_TOTAL" --maxEpochs "$EPOCHS" \
    --log "./log_s1_${VARIANT}/" --outputDir "./result_s1_${VARIANT}/" \
    --resultImgDir "./ResultImg_s1_${VARIANT}/" --modelDir "./${CKPT}/" \
    > "train_s1_${VARIANT}.log" 2>&1 &
echo "     PID=$!  log=train_s1_${VARIANT}.log"

cat <<NOTE

Next: use this encoder as the Stage-1 warm start for the 2x2 arms.
下一步：把它作为 2x2 实验臂的 Stage-1 热启动来源。
    bash Scripts/train/train_arms.sh ${VARIANT}
NOTE
