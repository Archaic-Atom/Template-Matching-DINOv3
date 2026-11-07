#!/usr/bin/env bash
# Validate real samples and the original two-GPU batch before the campaign.
# 正式实验前，用真实样本验证原双卡 batch；结果只写入独立检查目录。
set -eo pipefail
source /root/autodl-tmp/setup/activate.sh
set -u
RUN=/root/autodl-tmp/setup/campaign_20260925
export TM_CUDNN_BENCHMARK_LIMIT=10 TM_CORRELATION_CHANNELS_LAST=1
unset TORCH_CUDNN_V8_API_DISABLED
mkdir -p "$RUN/preflight"
export TM_SEED=2025 TM_SEED_FAST=1
export MALLOC_ARENA_MAX=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4
"$TM_PY" - "$RUN" <<'PY'
import csv
import sys
from pathlib import Path
run = Path(sys.argv[1])
source = Path("/root/autodl-tmp/setup/shard_train_256_subset40.csv")
rows = list(csv.DictReader(source.open()))[:2]
# Only the smoke manifest uses a short epoch cap; the formal manifest is untouched.
# 仅冒烟清单缩短 epoch 上限；正式清单保持原样。
for row in rows:
    assert Path(row["path"]).stat().st_size > 4_000_000_000
    row["file_num"] = "64"
with (run / "preflight/manifest.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["path", "file_num"])
    writer.writeheader()
    writer.writerows(rows)
PY
for VARIANT in G F; do
    OUT="$RUN/preflight/$VARIANT"
    if [ -e "$OUT" ]; then
        echo "Refusing to overwrite preflight output: $OUT" >&2
        exit 1
    fi
    mkdir -p "$OUT/checkpoint"
    "$TM_PY" Source/Tools/strip_stage1_opt.py \
        Checkpoint_F_stage1_256/model_epoch_9.pth \
        "$OUT/checkpoint/model_stage1_init.pth"
    printf 'last model name:model_stage1_init.pth\nmodel_stage1_init.pth\n' > "$OUT/checkpoint/checkpoint.list"
    "$TM_PY" -u Source/main.py \
        --root_path "$TM_PROJ" \
        --mode train --batchSize 32 --gpu 2 --port 8981 --dist True \
        --trainListPath "$RUN/preflight/manifest.csv" --imgNum 128 \
        --maxEpochs 4 --auto_save_num 1 --dataloaderNum 1 \
        --dataloaderType iterable --sampleNum 1 --lr 1e-3 \
        --modelName "TM_Model_$VARIANT" --dataset tmdataset --is_stage_2 True \
        --sar_size 192 --matching_dim 256 --fine_dim 32 --fine_margin 24 \
        --shuffle_buffer 64 --match_w4 1.0 --match_w8 0.5 \
        --match_sigma4 1.0 --match_sigma8 0.5 --match_primary_stride 4 \
        --match_w_fine 1.0 --match_sigma_fine 1.0 --fine_warmup_epochs 3 \
        --log "$OUT/log/" --outputDir "$OUT/result/" \
        --resultImgDir "$OUT/images/" --modelDir "$OUT/checkpoint/" \
        > "$OUT/train.log" 2>&1
    test -s "$OUT/checkpoint/model_epoch_3.pth"
    grep -q 'loaded trainable_tail' "$OUT/train.log"
    if grep -Eq 'Traceback|CUDA out of memory|ProcessRaisedException|(^|[^a-z])(nan|inf)([^a-z]|$)' "$OUT/train.log"; then
        echo "Preflight failure in $OUT/train.log" >&2
        exit 1
    fi
    "$TM_PY" - "$OUT" <<'PYVERIFY'
import sys
from pathlib import Path
import torch
out = Path(sys.argv[1])
initial = torch.load(out / "checkpoint/model_stage1_init.pth",
                     map_location="cpu", weights_only=False)
final = torch.load(out / "checkpoint/model_epoch_3.pth",
                   map_location="cpu", weights_only=False)
assert final["epoch"] == 3
steps = [float(item["step"]) for item in final["opt_0"]["state"].values()]
assert steps and min(steps) == max(steps) == 8, steps
assert torch.equal(final["model_0"]["fusion.logits"], torch.zeros(2))
assert all(torch.isfinite(tensor).all() for tensor in final["model_0"].values())
trunk_changed = any(
    not torch.equal(value, final["model_0"][name])
    for name, value in initial["model_0"].items()
    if name.startswith("sar_encoder.") and name in final["model_0"])
assert trunk_changed == (out.name == "G"), (out.name, trunk_changed)
print("CHECKPOINT_VERIFIED", out.name, "optimizer_steps=8",
      "trunk_changed=" + str(trunk_changed))
PYVERIFY
    echo "PREFLIGHT_PASS TM_Model_$VARIANT batch32 x 2"
done
"$TM_PY" - "$RUN" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
run = Path(sys.argv[1])
report = {
    "status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
    "models": ["TM_Model_G", "TM_Model_F"], "gpus": 2, "batch_per_gpu": 32,
    "steps_per_model": 8, "real_data": True, "optimizer_step": True,
    "smoke_only_workers": 1, "smoke_only_shuffle": 64,
    "fine_warmup_epochs": 3, "smoke_epochs": 4,
    "formal_recipe_unchanged": True,
    "cudnn_benchmark_limit": 10, "correlation_channels_last": True,
}
(run / "preflight_pass.json").write_text(json.dumps(report, indent=2))
PY
