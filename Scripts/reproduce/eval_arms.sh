#!/usr/bin/env bash
# Reproduce the 2x2 pre-alignment causal test with the RELEASED checkpoints.
# 用发布的权重复现 2x2 预对齐因果实验。
#
#   Table II — init (Stage-1 aligned vs random) x Stage-2 encoder (trainable vs frozen)
#   表 II —— 初始化（Stage-1 对齐 vs 随机）× Stage-2 编码器（可训 vs 冻结）
#
# Usage / 用法:
#   bash Scripts/reproduce/eval_arms.sh [gpu] [arm]
#     arm  one of s1tr | rtr | s1fr | rfr ; omit to run all four
#          取 s1tr | rtr | s1fr | rfr 之一；不填则四个全跑
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

GPU="${1:-0}"
ONLY="${2:-all}"

# arm | model | checkpoint | paper CMR(1)/CMR(3) @20k
ARMS=(
    "s1tr|TM_Model_A|Checkpoint_A_ft_256/model_epoch_4.pth|68.30 / 86.08"
    "rtr |TM_Model_B|Checkpoint_B_256/model_epoch_4.pth|68.28 / 86.35"
    "s1fr|TM_Model_F|Checkpoint_F_s2_frozen_256/model_epoch_4.pth|68.63 / 86.30"
    "rfr |TM_Model_F|Checkpoint_RF_s2_frozen_256/model_epoch_4.pth|66.94 / 85.17"
)

for entry in "${ARMS[@]}"; do
    IFS='|' read -r arm model ckpt reference <<< "$entry"
    arm="$(echo "$arm" | tr -d ' ')"
    [ "$ONLY" != "all" ] && [ "$ONLY" != "$arm" ] && continue

    echo ""
    echo "################ arm ${arm}  (paper CMR(1)/CMR(3) @20k = ${reference}) ################"
    if ! tm_eval "$model" "$ckpt" "repro_${arm}" "$GPU"; then
        echo "SKIP ${arm}: eval failed (missing checkpoint?)" >&2
        continue
    fi
    tm_pck "repro_${arm}"
done

cat <<'NOTE'

The paper's numbers above come from a 20,000-template subsample; this script's
default eval set is the 2k one, so expect sampling noise of a few tenths.
上面的论文数字来自 20,000 模板的子采样；本脚本默认用 2k 评测集，允许零点几的采样噪声。
Run all four on the same set to reproduce the key comparison:
在同一评测集上跑完四臂即可复现关键对比：
    S1/Tr ~= R/Tr  (pre-alignment redundant when the encoder is fine-tuned)
    S1/Fr  >  R/Fr  (pre-alignment helps only when the encoder is frozen)
NOTE
