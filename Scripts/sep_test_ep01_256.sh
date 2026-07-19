#!/usr/bin/env bash
# Held-out Stage-1 separation test on epoch_0 + epoch_1 @ 256/192, on the FREE
# GPU 6 (training keeps running on 0-3). Measures Stage-1's own objective
# (cos-sim heatmap relative ranking) on a fixed 2k held-out set — the trustworthy
# version of the in-training a0.
# 在空闲的 GPU 6 上对 ep0/ep1 跑 held-out separation 测试(训练继续在 0-3 跑)。
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
mkdir -p DebugResult
CSV=Datasets/test_256_2k.csv

for ep in 0 1; do
  echo "=== SEPARATION ep${ep} ==="
  CUDA_VISIBLE_DEVICES=6 "$PY" Source/Tools/stage1_separation_2k.py \
    --ckpt Checkpoint_A_stage1_256/model_epoch_${ep}.pth \
    --csv "$CSV" \
    --out DebugResult/sep_256_ep${ep}.csv 2>&1 | tail -4
done

echo "=== SUMMARY (held-out 2k, 256/192) ==="
"$PY" - <<'PYEOF'
import pandas as pd, numpy as np
for ep in (0, 1):
    df = pd.read_csv(f"DebugResult/sep_256_ep{ep}.csv")
    g, f = df['cos_gt'], df['cos_max_far']
    dprime = (g.mean() - f.mean()) / np.sqrt(0.5*(g.var()+f.var()) + 1e-9)
    top1_cell1 = (df['cell_dist_argmax'] <= 1).mean()   # fair top-1 (<=1 cell)
    top1_exact = (df['cell_dist_argmax'] == 0).mean()
    rank1 = (df['rank_gt'] == 1).mean()
    print(f"ep{ep}: n={len(df)} cos_gt={g.mean():.4f} cos_far={f.mean():.4f} "
          f"d'={dprime:.3f} top1(<=1cell)={top1_cell1:.4f} top1(exact)={top1_exact:.4f} "
          f"rank_gt==1={rank1:.4f}")
PYEOF
echo "SEP_DONE"
