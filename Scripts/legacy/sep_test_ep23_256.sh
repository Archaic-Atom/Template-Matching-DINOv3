#!/usr/bin/env bash
# Held-out separation on ep2 + ep3 @256/192 (GPU 6), then chart the full ep0-3
# trajectory. Tests under-training (held-out d'/top-1 climbing toward oracle 1.45)
# vs stuck (plateau ~0.03). ep0/ep1 CSVs already exist from the earlier run.
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python
cd "$PROJ" || exit 1
CSV=Datasets/test_256_2k.csv
for ep in 2 3; do
  echo "=== SEPARATION ep${ep} ==="
  CUDA_VISIBLE_DEVICES=6 "$PY" Source/Tools/stage1_separation_2k.py \
    --ckpt Checkpoint_A_stage1_256/model_epoch_${ep}.pth \
    --csv "$CSV" --out DebugResult/sep_256_ep${ep}.csv 2>&1 | tail -3
done
echo "=== HELD-OUT TRAJECTORY ep0-3 (oracle ceiling: top1<=1cell=1.0 d'=1.45) ==="
"$PY" - <<'PYEOF'
import pandas as pd, numpy as np
for ep in (0,1,2,3):
    try:
        d=pd.read_csv(f"DebugResult/sep_256_ep{ep}.csv")
    except FileNotFoundError:
        print(f"ep{ep}: (no csv)"); continue
    g,f=d['cos_gt'],d['cos_max_far']
    dp=(g.mean()-f.mean())/np.sqrt(0.5*(g.var()+f.var())+1e-9)
    print(f"ep{ep}: top1(<=1cell)={(d.cell_dist_argmax<=1).mean():.4f} "
          f"top1(exact)={(d.cell_dist_argmax==0).mean():.4f} "
          f"gt>far={(g>f).mean():.3f} d'={dp:.3f}")
PYEOF
echo "SEP23_DONE"
