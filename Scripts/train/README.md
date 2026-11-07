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

Tunables (all optional): `TM_LR`, `TM_SHUFFLE_BUFFER`, `TM_DATALOADER_NUM`,
`TM_S1_EPOCHS`, `TM_ARM_EPOCHS`. The last two data-loader knobs drive host RAM —
see the 2×2 section below before lowering them.
可调项（均可选）：`TM_LR`、`TM_SHUFFLE_BUFFER`、`TM_DATALOADER_NUM`、`TM_S1_EPOCHS`、
`TM_ARM_EPOCHS`。其中两个 dataloader 旋钮决定主机内存占用，调小之前先看下面的 2×2 一节。

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

## Shared-initialization four-arm runs / 共享初始化四臂实验

`train_2x2.sh` uses one shared Stage-1 source for both pre-aligned arms.
F implements the frozen arms, G the trainable arms. Initialization and
encoder trainability are varied within the same matching model family.
两个预对齐臂使用同一 Stage-1 来源；F 对应冻结臂，G 对应可训练臂。

```bash
TM_2X2_DRY_RUN=1 bash Scripts/train/train_2x2.sh 2025
bash Scripts/train/train_2x2.sh 2025
```

Sequential execution is the default. `TM_2X2_CONCURRENT=1` launches the four
arms concurrently and changes the loader-resource defaults. Check the printed
plan and set worker/buffer limits for the available host memory.
默认串行执行；并发模式会调整 loader 资源默认值，使用前检查计划及主机内存。

Each seed records its recipe in `Checkpoint_2x2_shared_init_s<seed>/recipe.env`
and its source/interpreter in `code.env`. A completed arm is skipped. A partial
arm stops the driver; `TM_2X2_RESTART_PARTIAL=1` archives it before a fresh run.
The shared initialization and its SHA are retained for resumption. The load gate checks that
pre-aligned arms loaded their source and random arms did not.
每个 seed 记录配方和代码来源。完整臂跳过，半途臂停止；显式重启时先归档。

`TM_SEED` fixes the random generators. By default it also selects deterministic
cuDNN settings. `TM_SEED_FAST=1` leaves the existing cuDNN settings unchanged.
Use the settings recorded for the run you want to reproduce.
`TM_SEED` 固定随机源；默认还设置确定性 cuDNN，`TM_SEED_FAST=1` 保留现有后端设置。

Evaluate completed arms with `Scripts/reproduce/eval_2x2.sh`; it defaults to
the common 20k manifest. See [MODELS_REVISION.md](../../MODELS_REVISION.md) for
released checkpoint names and revised table/figure references.
完成后使用四臂评测入口，权重及修订版表图编号见上述说明。

## Notes / 说明

- Training runs under `nohup`; the script prints `PID` and a log path, so an SSH
  drop will not kill the run. Follow with `tail -f train_<tag>.log`.
  训练用 `nohup` 起，脚本会打印 `PID` 与日志路径，SSH 断开不会中断训练。
  用 `tail -f train_<tag>.log` 跟踪。
- The scripts refuse to start if the target `Checkpoint_*/` already contains
  epochs — move it aside rather than silently mixing runs. The one exception is
  `train_2x2.sh`, which is resumable: it *skips* arms that already hold their
  final-epoch checkpoint and stops on half-finished ones (see above).
  若目标 `Checkpoint_*/` 已有 epoch 权重，脚本会拒绝启动——请先移走，避免不同次训练混在一起。
  唯一例外是可续跑的 `train_2x2.sh`：已产出末代 ckpt 的臂会被*跳过*，半截的臂会让它停下（见上）。
- Frozen-encoder variants need `find_unused_parameters=True` in JackFramework's
  DDP wrap, or all-reduce aborts with an empty exception (see `CODE_STRUCTURE.md`).
  冻结编码器的变体需要 JackFramework 的 DDP 包装启用 `find_unused_parameters=True`，
  否则 all-reduce 会以空异常中止（见 `CODE_STRUCTURE.md`）。
