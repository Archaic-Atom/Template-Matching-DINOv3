#!/usr/bin/env bash
# Stage 1 learned-IP (TM_Model_F) @ 256/192 — WebDataset, 4-GPU DDP.
# 可学内积 Stage-1(TM_Model_F)训练,256/192 几何,4 卡 DDP。
#
# Controlled comparison vs the weak Stage-1 baseline (Checkpoint_A_stage1_256):
# same data (subset40), same epochs (10), same lr/optimizer (SGD 5e-3), same
# batch — the ONLY variable is the objective (alignment_ip_loss + W_s/W_d
# projectors + VICReg rank floor).
# 与弱 Stage-1 基线严格对照:同数据/代数/lr/优化器,唯一变量是目标函数。
#
# ⚠️ imgNum MUST equal manifest real_total (progress-bar contract, see
# start_train_stage1_256_A.sh header). subset40: real_total=1,344,000.
#
# WATCH during epoch 0: l4 (erank_s) must hold ≥64 — drift toward ~25 means
# the rank-deficiency trap is back; raise lambda_var/lambda_cov in F's yaml.
# (NOTE: probe yardsticks were measured POST-norm at proj_dim=64; calibrate
# the pre-norm reading once at smoke test before panicking.)
# a0 = cosine in the learned projected space; a1 = cosine in the RAW trunk
# space — the trunk is all that transfers to Stage-2, so a0 high + a1 flat
# means the projector heads absorbed the alignment and the run is wasted.
# epoch 0 盯 l4(erank_s)≥64(注意探针阈值是 post-norm@64 维测的,先冒烟校准);
# a0=投影空间余弦,a1=trunk 原始余弦 —— a0 高 a1 平 = 对齐被投影头吸走,白训。
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1

train_gpus_id=0,1,2,3
train_list_path='./Datasets/shard_train_256_subset40.csv'
export JF_PROGRESS_COLUMNS=400

# Resume trap guard (review 2026-06-11): F's yaml has load_pretrained_sar:true,
# so a leftover ckpt (e.g. from a smoke test) makes JF silently RESUME instead
# of training fresh — breaking the single-variable control vs the A baseline.
# Smoke tests must use a throwaway modelDir (Checkpoint_F_smoke/).
# 续训陷阱防护:目录里有残留 ckpt 时 JF 会静默续训,毁掉与 A 基线的对照。
# 冒烟测试一律用一次性目录 Checkpoint_F_smoke/。
if [ -e Checkpoint_F_stage1_256/checkpoint.list ]; then
    echo "ERROR: Checkpoint_F_stage1_256 not empty — JF would silently resume,"
    echo "breaking the fresh-init control. Clear/move it first."
    exit 1
fi
mkdir -p log_stage1_256_F result_stage1_256_F ResultImg_stage1_256_F Checkpoint_F_stage1_256

echo "Stage 1 learned-IP (F) @256/192 — gpus ${train_gpus_id}, shards ${train_list_path}"
CUDA_VISIBLE_DEVICES=${train_gpus_id} JF_PROGRESS_COLUMNS=400 nohup "$PY" -u Source/main.py \
    --mode train \
    --batchSize 32 \
    --gpu 4 \
    --port 8817 \
    --trainListPath "${train_list_path}" \
    --auto_save_num 1 \
    --dataloaderNum 8 \
    --dataloaderType iterable \
    --imgNum 1344000 \
    --maxEpochs 10 \
    --sampleNum 1 \
    --lr 0.005 \
    --dist True \
    --modelName TM_Model_F \
    --dataset tmdataset \
    --is_stage_2 False \
    --sar_size 192 \
    --log ./log_stage1_256_F/ \
    --outputDir ./result_stage1_256_F/ \
    --resultImgDir ./ResultImg_stage1_256_F/ \
    --modelDir ./Checkpoint_F_stage1_256/ > ./train_stage1_256_F.log 2>&1 &
PID=$!
disown "$PID" 2>/dev/null
echo "STAGE1_256_F_PID=$PID  log=./train_stage1_256_F.log"
