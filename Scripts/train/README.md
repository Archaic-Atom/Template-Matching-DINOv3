# `Scripts/train/` — train it yourself / 自己训练和测试

These scripts train models from scratch on your own data, then hand off to
`Scripts/reproduce/` for evaluation.
这些脚本用你自己的数据从零训练，训练完交给 `Scripts/reproduce/` 做评测。

## Setup / 准备

You need the WebDataset shards, which are **not** in the repo — build them with
`Source/Tools/make_opt256_sar192_shards.py` (README → *Data preparation*), then:
你需要 WebDataset 分片（**不在**仓库里），用 `Source/Tools/make_opt256_sar192_shards.py`
生成后：

```bash
export TM_TRAIN_LIST=./Datasets/shard_train_256_full.csv        # full corpus
export TM_TRAIN_LIST_SUB40=./Datasets/shard_train_256_subset40.csv  # subset warm-start
export TM_EVAL_CSV=/path/to/image_test_2k_256.csv               # for the eval afterwards
```

Tunables (all optional): `TM_LR`, `TM_SHUFFLE_BUFFER`, `TM_S1_EPOCHS`, `TM_ARM_EPOCHS`.
可调项（均可选）：`TM_LR`、`TM_SHUFFLE_BUFFER`、`TM_S1_EPOCHS`、`TM_ARM_EPOCHS`。

## The proposed method / 本文方法

`TM_Model_C` needs **no Stage-1** — random SAR init, trained jointly with the
cascade under matching supervision alone. That is the paper's point.
`TM_Model_C` **不需要 Stage-1**——SAR 随机初始化，仅用匹配监督与级联联合训练。这正是论文的论点。

```bash
bash Scripts/train/train_cascade.sh smoke                 # 1 GPU, real tracebacks — do this first
bash Scripts/train/train_cascade.sh phase1 4 10           # subset warm-start, ~1.34M x 5 epochs
bash Scripts/train/train_cascade.sh phase2 \
     Checkpoint_C_direct_sub40_256/model_epoch_4.pth 4 10 # full corpus, ~13.18M x 3 epochs

bash Scripts/reproduce/eval_cascade.sh Checkpoint_C_direct_full_256/model_epoch_2.pth
```

**Always run `smoke` first.** Under DDP a shape/config error surfaces as an empty
`ProcessRaisedException` with no traceback; single-process shows the real error.
**务必先跑 `smoke`。** DDP 下形状/配置错误会表现为没有 traceback 的空
`ProcessRaisedException`，单进程才能看到真正的报错。

## The pre-alignment ablation / 预对齐消融

Only needed to reproduce the 2×2 causal test — the proposed method does not use it.
只有复现 2×2 因果实验才需要——本文方法并不使用它。

```bash
bash Scripts/train/train_stage1.sh A          # contrastive PatchNCE   -> Checkpoint_A_stage1_256/
bash Scripts/train/train_stage1.sh F          # learned IP + VICReg    -> Checkpoint_F_stage1_256/

bash Scripts/train/train_arms.sh trainable    # S1/Tr + R/Tr  (needs Stage-1 A)
bash Scripts/train/train_arms.sh frozen       # S1/Fr + R/Fr  (needs Stage-1 F)

bash Scripts/reproduce/eval_arms.sh 0         # evaluate all four
```

Each pair launches both arms concurrently, so they differ only in the seed/freeze
setting — that is what makes the comparison causal.
每一对的两臂并发启动，差别只在初始化/是否冻结——这正是该对比具备因果性的原因。

## Notes / 说明

- Training runs under `nohup`; the script prints `PID` and a log path, so an SSH
  drop will not kill the run. Follow with `tail -f train_<tag>.log`.
  训练用 `nohup` 起，脚本会打印 `PID` 与日志路径，SSH 断开不会中断训练。
  用 `tail -f train_<tag>.log` 跟踪。
- The scripts refuse to start if the target `Checkpoint_*/` already contains
  epochs — move it aside rather than silently mixing runs.
  若目标 `Checkpoint_*/` 已有 epoch 权重，脚本会拒绝启动——请先移走，避免不同次训练混在一起。
- Frozen-encoder variants need `find_unused_parameters=True` in JackFramework's
  DDP wrap, or all-reduce aborts with an empty exception (see `CODE_STRUCTURE.md`).
  冻结编码器的变体需要 JackFramework 的 DDP 包装启用 `find_unused_parameters=True`，
  否则 all-reduce 会以空异常中止（见 `CODE_STRUCTURE.md`）。
