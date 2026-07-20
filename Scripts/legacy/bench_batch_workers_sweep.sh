#!/usr/bin/env bash
# Experiment 3: sweep (batchSize, dataloaderNum) settings for best throughput
# on a single GPU. Each setting runs for one short epoch on 1 shard
# (32,000 samples), step time is read from the JackFramework progress bar.
#
# Single-GPU (GPU 4) so we don't compete with A on 0-3. The 200-sample
# epoch is intentionally short so each setting takes 30-60 s, not minutes.

set -e

# Activate the env that has torch + JackFramework + webdataset.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate stereoA

GPU=4
SHARD_PATTERN='/data5/datasets/TM_shards/shard-{000000..000000}.tar'    # 1 shard
IMGNUM=32000
EPOCHS=1
RESULTS_FILE='./bench_E3_sweep_results.tsv'
TMP_DIR='./Checkpoint_BENCH_E3/'

export COLUMNS=200
mkdir -p "${TMP_DIR}"
echo -e "batch\tworkers\titers\twall_s\timg_per_s\tstep_ms\tlast_loss" > "${RESULTS_FILE}"

# Settings sweep (batch, workers). Kept conservative to avoid OOM on a single
# 3090 (24 GB) competing with A's residual.
for batch in 4 8 16 32; do
  for workers in 4 8 16; do
    log="./bench_E3_b${batch}_w${workers}.log"
    iters_expected=$((IMGNUM / batch))

    echo "[E3] batch=${batch}  workers=${workers}  expect ${iters_expected} iters"
    rm -f "${TMP_DIR}"/model_epoch_*.pth "${TMP_DIR}"/checkpoint.list 2>/dev/null || true

    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=${GPU} python -u Source/main.py \
        --mode train \
        --batchSize ${batch} \
        --gpu 1 \
        --trainListPath "${SHARD_PATTERN}" \
        --auto_save_num 1 \
        --dataloaderNum ${workers} \
        --dataloaderType iterable \
        --imgNum ${IMGNUM} \
        --maxEpochs ${EPOCHS} \
        --sampleNum 1 \
        --lr 0.0002 \
        --dist False \
        --modelName TM_Model_A \
        --dataset tmdataset \
        --is_stage_2 False \
        --log ./log_bench_E3/ \
        --outputDir ./result_bench_E3/ \
        --resultImgDir ./ResultImg_bench_E3/ \
        --modelDir "${TMP_DIR}" > "${log}" 2>&1 || {
            echo -e "${batch}\t${workers}\t-\t-\t-\tFAILED\t-" >> "${RESULTS_FILE}"
            continue
        }
    t1=$(date +%s)
    wall=$((t1 - t0))

    # Pull final step time + last loss from the log
    step_ms=$(tail -c 5000 "${log}" | tr "\r" "\n" | sed -E "s/\x1b\[[0-9;]*m//g" | grep -oE "step: [0-9]+ms" | tail -1 | sed -E "s/step: ([0-9]+)ms/\1/" || true)
    last_loss=$(tail -c 5000 "${log}" | tr "\r" "\n" | sed -E "s/\x1b\[[0-9;]*m//g" | grep -oE "l0: [0-9.]+" | tail -1 | sed -E "s/l0: //" || true)

    if [ -n "${step_ms}" ] && [ "${step_ms}" -gt 0 ]; then
        img_per_s=$(awk "BEGIN {printf \"%.1f\", ${batch} * 1000.0 / ${step_ms}}")
    else
        img_per_s="?"
        step_ms="?"
    fi

    echo "     wall=${wall}s  step=${step_ms}ms  img/s=${img_per_s}  loss=${last_loss}"
    echo -e "${batch}\t${workers}\t${iters_expected}\t${wall}\t${img_per_s}\t${step_ms}\t${last_loss}" >> "${RESULTS_FILE}"
  done
done

echo
echo "=== SWEEP DONE ==="
column -t -s $'\t' "${RESULTS_FILE}"
