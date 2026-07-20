#!/usr/bin/env bash
# Train the 2x2 pre-alignment arms: init (Stage-1 vs random) x encoder (trainable vs frozen).
# 训练 2x2 预对齐实验臂：初始化（Stage-1 vs 随机）× 编码器（可训 vs 冻结）。
#
#   trainable  S1/Tr = TM_Model_A warm-started from Checkpoint_A_stage1_256
#              R/Tr  = TM_Model_B random init            (the only difference is the init)
#   frozen     S1/Fr = TM_Model_F warm-started from Checkpoint_F_stage1_256, encoder frozen
#              R/Fr  = TM_Model_F random init,           encoder frozen
#
# The trainable pair is the paper's key null (S1/Tr ~= R/Tr); the frozen pair is
# where pre-alignment finally shows a gain (S1/Fr > R/Fr).
# 可训一对是论文的关键零结果（S1/Tr ≈ R/Tr）；冻结一对才显出预对齐的增益（S1/Fr > R/Fr）。
#
# Usage / 用法:
#   bash Scripts/train/train_arms.sh trainable [gpus] [batch]   # needs Checkpoint_A_stage1_256
#   bash Scripts/train/train_arms.sh frozen    [gpus] [batch]   # needs Checkpoint_F_stage1_256
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

PAIR="${1:?usage: $0 trainable|frozen [gpus] [batch]}"
GPUS="${2:-4}"
BATCH="${3:-16}"
MANIFEST="${TM_TRAIN_LIST:-./Datasets/shard_train_256_full.csv}"
EPOCHS="${TM_ARM_EPOCHS:-5}"

# seed_from_stage1 <stage1_dir> <arm_ckpt_dir>
#   Strip optimizer state out of the newest Stage-1 checkpoint and pin it as the
#   arm's starting point, exactly how the paper's runs were warm-started.
#   把最新的 Stage-1 权重剥掉优化器状态后钉为该臂的起点——与论文的做法一致。
seed_from_stage1() {
    local s1_dir="$1" arm_dir="$2" latest
    latest=$(find "$s1_dir" -maxdepth 1 -name 'model_epoch_*.pth' -exec ls -t {} + 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        echo "ERROR: no Stage-1 checkpoint in ${s1_dir}/" >&2
        echo "       train it first: bash Scripts/train/train_stage1.sh ${s1_dir:11:1}" >&2
        return 1
    fi
    mkdir -p "$arm_dir"
    "$TM_PY" Source/Tools/strip_stage1_opt.py "$latest" "$arm_dir/model_stage1_init.pth" || return 1
    printf 'last model name:%s\n%s\n' model_stage1_init.pth model_stage1_init.pth \
        > "$arm_dir/checkpoint.list"
    echo "[seed] ${arm_dir} <- ${latest}"
}

case "$PAIR" in
    trainable)
        seed_from_stage1 Checkpoint_A_stage1_256 Checkpoint_A_ft_256 || exit 1
        tm_train TM_Model_A "$MANIFEST" Checkpoint_A_ft_256 "$EPOCHS" a_ft "$GPUS" "$BATCH" 8881
        tm_train TM_Model_B "$MANIFEST" Checkpoint_B_256    "$EPOCHS" b_rand "$GPUS" "$BATCH" 8882
        ;;

    frozen)
        seed_from_stage1 Checkpoint_F_stage1_256 Checkpoint_F_s2_frozen_256 || exit 1
        # RF is the control: same frozen config, but no Stage-1 seed at all.
        # RF 是对照：同样冻结，但完全不给 Stage-1 种子。
        mkdir -p Checkpoint_RF_s2_frozen_256
        tm_train TM_Model_F "$MANIFEST" Checkpoint_F_s2_frozen_256  "$EPOCHS" s1_frozen "$GPUS" "$BATCH" 8883
        tm_train TM_Model_F "$MANIFEST" Checkpoint_RF_s2_frozen_256 "$EPOCHS" r_frozen  "$GPUS" "$BATCH" 8884
        ;;

    *)
        echo "ERROR: expected 'trainable' or 'frozen' (got '$PAIR')" >&2
        exit 1
        ;;
esac

cat <<'NOTE'

Both arms of a pair run concurrently — watch train_*.log. When they finish:
一对中的两臂并发训练——看 train_*.log。跑完后：
    bash Scripts/reproduce/eval_arms.sh 0        # evaluates all four arms
NOTE
