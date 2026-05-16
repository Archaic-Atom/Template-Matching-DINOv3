#!/usr/bin/env bash
# TM_Model_D Stage 2 training — ablation: soft argmax + sub-pixel refinement.
#
# Architecture and training procedure identical to TM_Model_A. The only
# difference is at evaluation time: the fine heatmap's peak is refined via
# 2D quadratic fit (config.inference.argmax_type: soft_subpixel), recovering
# sub-pixel offsets in [-0.5, 0.5] from the discrete argmax.
#
# Therefore D's TRAINING produces the same weights as A's — the script exists
# so the paper's ablation table has a dedicated row, and so we have a separate
# log/result dir to compare cleanly. If you want to save compute, you can also
# point D at A's trained Checkpoint dir for inference-only evaluation, but for
# the paper we re-run training with a fixed seed to rule out variance.

set -e

train_gpus_id=0,1,2,3
train_list_path='/data5/datasets/TM_shards/shard-{000000..000254}.tar'
src_stage1_dir='./Checkpoint/'
model_dir='./Checkpoint_D_stage2/'
log_file='./start_train_D_stage2_wds.log'

export COLUMNS=200

mkdir -p "${model_dir}"

# Bootstrap from A's Stage-1 ckpt (same SarResNet16GN → keys load cleanly).
ckpt_count=$(ls "${model_dir}"/model_epoch_*.pth 2>/dev/null | wc -l)
if [ "${ckpt_count}" -eq 0 ]; then
    latest=$(ls -t "${src_stage1_dir}"/model_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "${latest}" ]; then
        echo "ERROR: ${model_dir} is empty AND ${src_stage1_dir} has no Stage-1 ckpt."
        echo "       Place a Stage-1 model_epoch_N.pth in either dir before launching."
        exit 1
    fi
    echo "  bootstrapping ${model_dir} from ${latest}"
    cp "${latest}" "${model_dir}/"
    [ -f "${src_stage1_dir}/checkpoint.list" ] && cp "${src_stage1_dir}/checkpoint.list" "${model_dir}/"
fi

echo "[TM_Model_D] Stage 2 — soft argmax + sub-pixel quadratic fit (eval-time)"
echo "  train gpus:  ${train_gpus_id}"
echo "  shards:      ${train_list_path}"
echo "  modelDir:    ${model_dir}"
echo "  log file:    ${log_file}"

CUDA_VISIBLE_DEVICES=${train_gpus_id} python -u Source/main.py \
                        --mode train \
                        --batchSize 8 \
                        --gpu 4 \
                        --trainListPath "${train_list_path}" \
                        --auto_save_num 1 \
                        --dataloaderNum 8 \
                        --dataloaderType iterable \
                        --imgNum 8145177 \
                        --maxEpochs 30 \
                        --sampleNum 1 \
                        --lr 1e-3 \
                        --dist True \
                        --modelName TM_Model_D \
                        --dataset tmdataset \
                        --is_stage_2 True \
                        --log ./log_D_stage2/ \
                        --outputDir ./result_D_stage2/ \
                        --resultImgDir ./ResultImg_D_stage2/ \
                        --modelDir "${model_dir}" > "${log_file}" 2>&1 &

echo "started (PID $!); tailing log..."
tail -f "${log_file}"
