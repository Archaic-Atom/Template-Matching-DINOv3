#!/usr/bin/env bash
# Train the PROPOSED method from scratch: the direct-trained cascade (TM_Model_C).
# 从零训练本文方法：直接训练的级联（TM_Model_C）。
#
# No Stage-1 pre-alignment: the SAR encoder starts random and is trained jointly
# with the cascade under matching supervision alone.
# 不做 Stage-1 预对齐：SAR 编码器随机初始化，仅用匹配监督与级联联合训练。
#
# Two phases, as in the paper / 论文中的两阶段:
#   phase1  subset warm-start   (~1.34M samples x 5 epochs)
#   phase2  full corpus         (~13.18M samples x 3 epochs, warm-started from phase1)
#
# Usage / 用法:
#   bash Scripts/train/train_cascade.sh phase1 [gpus] [batch]
#   bash Scripts/train/train_cascade.sh phase2 <phase1_ckpt.pth> [gpus] [batch]
#   bash Scripts/train/train_cascade.sh smoke            # 1-GPU shape/build check
#
# Always run `smoke` first: single-process gives real tracebacks, DDP does not.
# 先跑 smoke：单进程才会给出真实 traceback，DDP 不会。
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

MODE="${1:?usage: $0 phase1|phase2 <ckpt>|smoke [gpus] [batch]}"

SUB40="${TM_TRAIN_LIST_SUB40:-./Datasets/shard_train_256_subset40.csv}"
FULL="${TM_TRAIN_LIST:-./Datasets/shard_train_256_full.csv}"

case "$MODE" in
    phase1)
        GPUS="${2:-4}"; BATCH="${3:-10}"
        tm_train TM_Model_C "$SUB40" Checkpoint_C_direct_sub40_256 5 c_sub40 "$GPUS" "$BATCH" 8866
        ;;

    phase2)
        SRC="${2:?usage: $0 phase2 <phase1_ckpt.pth> [gpus] [batch]}"
        GPUS="${3:-4}"; BATCH="${4:-10}"
        [ -f "$SRC" ] || { echo "ERROR: phase-1 checkpoint not found: $SRC" >&2; exit 1; }

        CKPT=Checkpoint_C_direct_full_256
        mkdir -p "$CKPT"
        # Warm start: drop optimizer state and reset the epoch counter so phase 2
        # starts cleanly at epoch 0 (the cascade warmup is already past).
        # 热启动：丢掉优化器状态并复位 epoch 计数，让 phase2 干净地从 epoch 0 开始
        # （级联的 warmup 在 phase1 已经走完）。
        "$TM_PY" Source/Tools/strip_stage1_opt.py "$SRC" "$CKPT/model_phase1_init.pth" || exit 1
        printf 'last model name:%s\n%s\n' model_phase1_init.pth model_phase1_init.pth \
            > "$CKPT/checkpoint.list"
        echo "[phase2] warm-started from $SRC"
        tm_train TM_Model_C "$FULL" "$CKPT" 3 c_full "$GPUS" "$BATCH" 8867
        ;;

    smoke)
        echo "[smoke] 1 GPU, dist False — Ctrl-C after a few hundred iterations."
        rm -rf Checkpoint_C_smoke log_c_smoke result_c_smoke ResultImg_c_smoke
        mkdir -p Checkpoint_C_smoke
        REAL_TOTAL=$(tail -n +2 "$SUB40" 2>/dev/null | awk -F, '{s+=$2} END{print s}')
        CUDA_VISIBLE_DEVICES="${2:-0}" "$TM_PY" -u Source/main.py \
            --mode train --dataset tmdataset --modelName TM_Model_C --is_stage_2 True \
            --dataloaderType iterable --dataloaderNum 2 --sampleNum 1 \
            --sar_size "$TM_SAR_SIZE" --matching_dim "$TM_MATCHING_DIM" \
            --shuffle_buffer 2000 --lr 1e-3 --auto_save_num 1 \
            --dist False --gpu 1 --batchSize 4 --port 8868 \
            --trainListPath "$SUB40" --imgNum "${REAL_TOTAL:-1000}" --maxEpochs 1 \
            --log ./log_c_smoke/ --outputDir ./result_c_smoke/ \
            --resultImgDir ./ResultImg_c_smoke/ --modelDir ./Checkpoint_C_smoke/
        ;;

    *)
        echo "ERROR: unknown mode '$MODE' (expected phase1 | phase2 | smoke)" >&2
        exit 1
        ;;
esac

cat <<'NOTE'

When training finishes, evaluate the result with:
训练结束后这样评测：
    bash Scripts/reproduce/eval_cascade.sh Checkpoint_C_direct_full_256/model_epoch_2.pth
NOTE
