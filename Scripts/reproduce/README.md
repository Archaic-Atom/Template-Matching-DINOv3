# `Scripts/reproduce/` — evaluate the released models / 用发布的模型直接测试

Everything here **only runs inference** on checkpoints you downloaded. Nothing is
trained and no released checkpoint is modified.
这里的脚本**只做推理**，用的是你下载的权重；不训练、也不会改动已发布的权重。

## Setup / 准备

1. Download the models (README → *Pretrained models*) and unzip at the project root,
   so `Checkpoint_*/` sit next to `Source/`. Keep the DINOv3 backbone in `Weights/`.
   下载模型并解压到项目根目录，使各 `Checkpoint_*/` 与 `Source/` 同级；DINOv3 骨干留在 `Weights/`。
2. Point the scripts at your evaluation data (it is not in the repo):
   把脚本指向你的评测数据（数据不在仓库里）：

```bash
export TM_EVAL_CSV=/path/to/image_test_2k_256.csv   # in-domain eval manifest
export TM_PY=/path/to/python                        # optional, defaults to `python`
```

`Scripts/common.sh` derives the project root from its own location, so a clone at
any path works without editing anything.
`Scripts/common.sh` 由自身位置推导项目根目录，仓库放在任何路径都无需改动。

## Scripts / 脚本

| script | reproduces | default checkpoint |
|---|---|---|
| `eval_cascade.sh [ckpt] [gpu]` | **Table I** — in-domain SOTA, proposed cascade | `Checkpoint_C_phase2_8gpu/model_epoch_2.pth` |
| `eval_arms.sh [gpu] [arm]` | **Table IV** — submitted single-run comparison | the four arm checkpoints |
| `eval_2x2.sh <seeds> [gpu] [arm] [--dry-run]` | **Table V / Figure 9** — repeated shared-initialization four-arm test | revision checkpoints |
| `eval_cross_dataset.sh [ckpt] [gpu] [whu\|qxs]` | **Table VIII** — zero-shot cross-dataset | released cascade |

```bash
# headline number
bash Scripts/reproduce/eval_cascade.sh

# one arm only, on GPU 1
bash Scripts/reproduce/eval_arms.sh 1 s1tr

# cross-sensor transfer
TM_CROSS_WHU=/path/whu/cross_eval.csv bash Scripts/reproduce/eval_cross_dataset.sh
```

Each run writes predictions to `DebugResult_<tag>/` and prints CMR/PCK.
每次运行把预测写入 `DebugResult_<tag>/` 并打印 CMR/PCK。

## Expected output / 预期结果

Verified on an RTX 4080 with the released cascade, default 2k eval set:
在 RTX 4080 上用发布的 cascade、默认 2k 评测集实测：

```
n=2000  mean=1.10px  median=0.00px
  PCK@1px = 93.20%   PCK@3px = 95.70%   PCK@5px = 96.05%
```

which matches the paper's 2k figures for the shipped checkpoint. The Table I
headline (CMR(1) 93.30 / CMR(3) 95.83) is the same model on the **full** test set.
与论文中该权重的 2k 结果一致。表 I 的 93.30 / 95.83 是同一模型在**全测试集**上的值。

## Gotchas / 注意

- **`--imgNum` is only the progress-bar denominator**, not the number of samples
  evaluated — the map-style backend always walks the whole manifest. A bar reading
  `64/64` can still have scored all 2000 rows.
  **`--imgNum` 只是进度条分母**，不是实际评测样本数——map-style 后端始终遍历整个清单。
  进度条显示 `64/64` 时仍可能已评完 2000 条。
- The finalize step that merges per-rank CSVs can lag well after the bar hits 100%.
  No data is at risk; `tm_pck` falls back to `.tmp_test_*_rank0.csv` if needed.
  合并各 rank CSV 的收尾步骤可能在进度条满格后还要跑一会。数据不会丢；必要时 `tm_pck`
  会回退读 `.tmp_test_*_rank0.csv`。
