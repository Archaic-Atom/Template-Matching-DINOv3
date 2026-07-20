#!/usr/bin/env bash
# Stage-2 FROZEN-encoder double arms @256/192 — run AFTER Stage-1 F finishes.
# Stage-2 冻结 encoder 双臂实验,在 F 的 Stage-1 训完后启动。
#
#   Arm S1F (TM_Model_F): SAR trunk warm-starts from the learned-IP Stage-1
#       ckpt and is FROZEN (config freeze_sar_encoder:true) -> only the
#       matching head trains. align_ip.* keys in the ckpt are dropped at
#       load (unexpected, strict=False) — projector discarded by design.
#   Arm RF  (TM_Model_F): fresh empty modelDir -> random-init SAR trunk,
#       same freeze. Random-frozen CONTROL — isolates "freezing costs
#       capacity" from "Stage-1 information content". Without this arm the
#       S1F result is uninterpretable.
#   臂 S1F:强 Stage-1 初始化 + 冻结;臂 RF:随机初始化 + 冻结(对照,把
#   "冻结的容量代价"与"Stage-1 信息含量"分离 —— 没有它结果无法解读)。
#
# Readout vs the existing A-ft/B (unfrozen) pair:
#   S1F >> RF and S1F ~ B  -> strong protected alignment WORKS -> method back.
#   S1F ~ RF               -> even strong protected alignment is useless ->
#                             F0 upgrades to categorical.
#
# DDP + frozen trunk => most params get no grad => REQUIRES the remote JF
# find_unused_parameters=True patch (_meta_ops.py). Verify it's in place
# before launching (the empty-ProcessRaisedException crash = patch missing).
# 冻结 + DDP 依赖远端 JF 的 find_unused_parameters patch,启动前确认。
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
MAN=Datasets/shard_train_256_subset40.csv
RT=$(tail -n +2 "$MAN" | awk -F, '{s+=$2} END{print s}')
echo "manifest=$MAN real_total=$RT"

# --- bootstrap arm S1F from the learned-IP Stage-1 encoder ---
S1=Checkpoint_F_stage1_256
S1F=Checkpoint_F_s2_frozen_256
RF=Checkpoint_RF_s2_frozen_256

# Re-run guard (review 2026-06-11): refuse to wipe trained arms — an
# accidental second invocation would destroy GPU-days of checkpoints.
# 防重跑:目录已存在就拒绝,误触不再清掉已训了几天的双臂 ckpt。
for d in "$S1F" "$RF"; do
    if [ -d "$d" ]; then
        echo "ERROR: $d exists — refusing to wipe. rm -rf it manually if intended."
        exit 1
    fi
done
mkdir -p "$S1F" "$RF"

# Pin the FINAL Stage-1 ckpt (maxEpochs 10 -> model_epoch_9). mtime-latest
# would silently bootstrap from a half-trained run (review 2026-06-11).
# 钉死最终代 ckpt;按 mtime 选会静默用上半截训练的权重。
latest="$S1/model_epoch_9.pth"
if [ ! -f "$latest" ]; then
    echo "ERROR: $latest missing — Stage-1 F not finished (need all 10 epochs)."
    exit 1
fi
echo "S1F bootstrap from: $latest"
"$PY" Source/Tools/strip_stage1_opt.py "$latest" "$S1F/model_stage1_init.pth" || {
    echo "ERROR: strip_stage1_opt.py failed"; exit 1; }
# A silently-missing init turns the headline arm into a second random-frozen
# control — the one failure mode that would FABRICATE our conclusion.
# init 静默缺失会把主臂退化成第二个随机对照 —— 恰好伪造结论,必须硬卡。
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

PIDS=$(launch TM_Model_F 0,1 8842 "$S1F" train_f_s1frozen.log)
echo "S1F_PID=$PIDS  (TM_Model_F, strong-S1 init, frozen trunk, gpu0,1, $S1F)"
PIDR=$(launch TM_Model_F 2,3 8843 "$RF" train_f_rfrozen.log)
echo "RF_PID=$PIDR  (TM_Model_F, random init, frozen trunk, gpu2,3, $RF)"
echo "FROZEN_ARMS_LAUNCHED S1F=$PIDS RF=$PIDR"

# Post-launch load verification (review 2026-06-11): the S1F log must show the
# trainable_tail load (expected signature: unexpected=2, the dropped
# align_ip.w_s/w_d) and must NOT show a from-scratch start.
# 启动后验证:S1F 日志必须出现 trainable_tail 加载行(unexpected=2,即被丢弃
# 的两个投影头 key),且不得出现"从零开始"。
sleep 90
if grep -q 'loaded trainable_tail' train_f_s1frozen.log \
        && ! grep -qi 'no checkpoint' train_f_s1frozen.log; then
    echo "S1F load check OK:"
    grep 'loaded trainable_tail' train_f_s1frozen.log | head -1
else
    echo "!!!!!! S1F DID NOT LOAD THE STAGE-1 CKPT — kill PID $PIDS and investigate"
fi
