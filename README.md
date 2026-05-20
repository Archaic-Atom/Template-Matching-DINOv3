# Template-Matching-DINOv3

![Python](https://img.shields.io/badge/python-%3E=3.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

A cross-modal **SAR ↔ RGB template matching** framework that uses a frozen
**DINOv3** ViT as the RGB feature teacher and trains a lightweight SAR CNN
encoder to align SAR features into the DINO feature space. The project is
built on top of [JackFramework](https://github.com/Archaic-Atom/JackFramework),
which handles the training loop, DDP, logging and checkpointing.

---

## 1. Overview

Given a small SAR patch (64×64) and a larger optical RGB image, the goal is
to locate the SAR patch inside the RGB image and produce a pixel-wise
localization heatmap.

Training is split into two **logical** stages, both runnable end-to-end
from a single launch script:

1. **Stage 1 — feature alignment** (`--is_stage_2 False`): freeze the
   DINOv3 RGB encoder and train a SAR encoder so the SAR feature map at
   stride-16 matches the DINO RGB feature map at the corresponding
   location. Trained once over the full ~8M-sample contrastive dataset;
   the checkpoint is then shared across the four model variants below.
2. **Stage 2 — matching / localization** (`--is_stage_2 True`): freeze
   the aligned encoders, learn a matching head that regresses a
   stride-aware Gaussian heatmap whose peak is at the true SAR position
   inside the RGB image.

### Method at a glance

- **RGB branch**: `dinov3_vitl16` with `sat493m` pretrained weights, frozen.
  Intermediate-layer tokens are extracted via `get_intermediate_layers`.
- **SAR branch**: `SarResNet16GN`, a ResNet-34-style CNN with **GroupNorm**
  (GN preferred over BN for SAR due to unstable batch statistics) down to
  stride-16, followed by a 1×1 projection that L2-normalizes onto DINO's
  1024-d space.
- **Stage-1 alignment loss**: soft-window PatchNCE (`tau=0.07`, 3×3 positive
  window) with in-batch negatives, plus a small global cosine term.
- **Stage-2 matching loss**: BCE on multi-stride Gaussian heatmaps. Optional
  cascade (stride 4 → 2 → 1) refinement in variant C; optional sub-pixel
  quadratic peak fit in variant D.

---

## 2. Architecture — four yaml-driven variants

Each variant is selected at launch time via `--modelName TM_Model_X`. The
shared building blocks live in `_components/`; `builder.py` reads a
per-variant `config.yaml` and assembles a `ComposedTM`. The four variants
share the **same `trainable_tail.sar_encoder.*` state-dict subtree** so the
expensive Stage-1 checkpoint loads cleanly into any of them
(`Scripts/verify_state_dict_compat.py` enforces this).

| Variant | Folder         | Purpose                                              | Key config diff                                                       |
|---------|----------------|------------------------------------------------------|-----------------------------------------------------------------------|
| **A**   | `TM_Model_A`   | Baseline (full method)                               | —                                                                     |
| **B**   | `TM_Model_B`   | Ablation: no Stage-1 alignment                       | `load_pretrained_sar: false` + `freeze_sar_encoder: false`            |
| **C**   | `TM_Model_C`   | **Main proposed** — 3-level cascade fine head        | `matching.type: CascadeHead`, `loss.type: cascade_loss`               |
| **D**   | `TM_Model_D`   | Soft argmax + sub-pixel refinement                   | `inference.argmax_type: soft_subpixel`                                |

Pre-refactor TM_Model_A code is preserved in
`Source/UserModelImplementation/Models/_legacy/legacy_A/` for the
state-dict compatibility verification script.

---

## 3. Environment

### Hardware
- NVIDIA GPU (CUDA). Multi-GPU training uses PyTorch DDP. Tested with
  4× RTX 3090 (24 GB).

### Software
- Python ≥ 3.8
- PyTorch matching your CUDA driver (tested with recent 2.x)
- `JackFramework` (dev install from
  [Archaic-Atom/JackFramework](https://github.com/Archaic-Atom/JackFramework);
  see install below)
- `webdataset` (for the iterable dataloader)
- `numpy`, `pandas`, `pyarrow`, `Pillow`, `opencv-python`, `torchvision`

---

## 4. Installation

```bash
# 1. clone the repo together with the DINOv3 submodule
git clone --recurse-submodules <this-repo-url>
cd Template-Matching-DINOv3

# 2. install JackFramework (dev / editable install)
git clone https://github.com/Archaic-Atom/JackFramework.git
cd JackFramework && ./install.sh && cd ..

# 3. place DINOv3 satellite-pretrained weights under Weights/
#    default filename: dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

Default paths live in `Source/UserModelImplementation/user_define.py`:

```python
ROOT_PATH   = '/data/raozhibo/Documents/Programs/Template-Matching-DINOv3/'
DINOv3_PATH = 'Source/Libs/dinov3/'
WEIGHTS_PATH = 'Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
```

Update `ROOT_PATH` for your environment, or override it on the CLI
(`--root_path ...`).

---

## 5. Data

### 5.1 WebDataset shards (recommended; `--dataloaderType iterable`)

Production runs use a pre-packed WebDataset on `/data5`:

- 8,145,177 SAR–optical pairs total
- 255 tar shards (`shard-000000.tar` … `shard-000254.tar`); 32,000 samples
  per shard except the last (17,176)
- Per sample: SAR PNG (~3 KB raw) + RGB PNG (~31 KB) + `label.cls`
- Sequential reads from `/data5` give ~10× the throughput of the legacy
  scattered-PNG layout

Repack from a `.pt` source list:

```bash
python Scripts/pack_to_webdataset.py \
  --src_pt   ./Datasets/image_test.pt \
  --dst_dir  /data5/datasets/TM_shards \
  --samples_per_shard 32000
```

### 5.2 Legacy CSV / `.pt` (`--dataloaderType mapstyle`)

Still supported for small experiments. `TM_Dataset` reads a CSV with
columns `SAR_Image, RGB_Image, Label` (where `Label` is JSON
`"[x, y]"`), or a `.pt` dict with parallel `sar_path`, `rgb_path`,
`label` lists. Use `Source/Tools/csv2pt.py` to convert.

---

## 6. Dataloader contracts (read before training)

These are not obvious from the code in isolation and have bitten the project
several times. They live in the docstring of `_tm_webdataset.py` and are
re-summarized here.

1. **No `.with_epoch(N)` on the WebDataset pipeline.** `with_epoch` caps
   each iterator instance via `islice`, but PyTorch's `DataLoader` gives
   each worker a fresh copy of the dataset, so a per-worker cap can't
   express the global "one framework epoch = one true pass" contract.
   Without it, `split_by_node + split_by_worker` partition shards per
   rank/worker and each worker drains its share once; `StopIteration`
   propagates naturally. **1 framework epoch ≡ 1 true pass over the
   dataset** (modulo at most one shard of rounding loss per rank).

2. **`_balanced_split_by_node` instead of `wds.split_by_node`.** The
   vanilla splitter (`islice(src, rank, None, world_size)`) gives some
   ranks one extra shard when `num_shards % world_size != 0`. Under DDP,
   that imbalance combined with JackFramework's epoch-end `dist.barrier()`
   triggers an NCCL `ALLREDUCE` timeout: the short rank waits at the
   barrier past the 600 s default while the longer ranks finish their
   extra iterations. The balanced version truncates every rank to
   `floor(num_shards / world_size)`. At our 255-shard / 4-rank setup each
   rank ends up with 63 shards (3 tail shards dropped, ≈ 1.2 % of data).

3. **No `DistributedSampler` in iterable mode.** Dev JackFramework's
   `__build_iterable_dataloader` correctly skips `sampler`/`shuffle` for
   `--dataloaderType iterable`.

4. **`persistent_workers=True`.** Set in dev JackFramework's iterable
   branch so the ~1-minute 100 K-buffer warmup doesn't repeat at every
   epoch boundary.

5. **`shuffle_buffer = 100 000` default (CLI: `--shuffle_buffer`).** Sized
   to mix ~3 shards per worker. Cost is `num_workers × buffer × ~35 KB raw
   bytes per sample`; under DDP-4 with `dataloaderNum=8` (= 32 workers
   total) that's ~112 GB host RAM. Required because shard internal order
   is structured (per-shard SAR mean drifts from ~88 at the head to ~184
   at the tail), so the default 10 K buffer mixes only one third of a
   shard — too local for batched contrastive negatives.

---

## 7. Training

### 7.1 Stage 1 — feature alignment

```bash
bash Scripts/start_train_stage1_wds.sh
```

Defaults: `--modelName TM_Model_A`, `--is_stage_2 False`, 4 GPUs DDP,
`batchSize 32`, `maxEpochs 8`, `dataloaderNum 8`, `lr 0.0002`. Reads/writes
`./Checkpoint/`. Predicted wall time ~70 min per framework epoch ≈
~9–10 h on 4× RTX 3090 (HDD-bound for `/data5`; faster if shards are on
SSD).

The training is `nohup`-friendly: the launch script puts `python -u main.py`
into the background and exits, so SSH drops don't kill the training. Tail
`./start_train_stage1_wds.log` to monitor.

### 7.2 Stage 2 — matching head

Three variant-specific scripts under `Scripts/`:

```bash
bash Scripts/start_train_stage2_B_wds.sh    # B: no Stage-1 inheritance
bash Scripts/start_train_stage2_C_wds.sh    # C: cascade fine head
bash Scripts/start_train_stage2_D_wds.sh    # D: soft argmax / sub-pixel
```

C and D auto-copy a Stage-1 `Checkpoint/model_epoch_N.pth` into
`Checkpoint_C_stage2/` / `Checkpoint_D_stage2/`. B intentionally starts
from scratch (its `modelDir` is empty and the variant config sets
`load_pretrained_sar: false`).

### 7.3 Frequently-used CLI flags

JackFramework flags (set in the launch scripts):

| Flag                | Typical value | Notes |
|---------------------|---------------|-------|
| `--mode`            | `train`       | `train` / `test`. |
| `--batchSize`       | 32 (Stage 1)  | Per-rank batch size. |
| `--gpu`             | 4             | Number of GPUs participating in DDP. |
| `--dist`            | `True`        | Enable DDP. |
| `--dataloaderNum`   | 8             | Number of DataLoader workers per rank. |
| `--dataloaderType`  | `iterable`    | `mapstyle` (legacy CSV) or `iterable` (WebDataset). |
| `--imgNum`          | 8145177       | Dataset size; used by the progress bar denominator. |
| `--maxEpochs`       | 8             | Number of framework epochs. With the fixed dataloader, 1 framework epoch = 1 true pass. |
| `--auto_save_num`   | 1             | Save every framework epoch. |
| `--trainListPath`   | shard glob    | `/data5/datasets/TM_shards/shard-{000000..000254}.tar` for iterable mode. |
| `--modelDir`        | `./Checkpoint/` | Checkpoint directory. |

Project-specific flags (added in `user_interface.py`):

| Flag                  | Default | Meaning |
|-----------------------|---------|---------|
| `--is_stage_2`        | `False` | Stage 1 alignment if `False`, Stage 2 matching if `True`. |
| `--shuffle_buffer`    | 100000  | WebDataset sample shuffle buffer size (see §6.5). |
| `--root_path`         | (see `user_define.py`) | Repo root used to resolve `dinov3_path` / `pre_trained_weights_path`. |
| `--dinov3_path`       | `Source/Libs/dinov3/` | Local DINOv3 hub path. |
| `--pre_trained_weights_path` | `Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | DINOv3 checkpoint. |
| `--pre_trained`       | `True`  | Deprecated, kept for backward-compat — the forward path is now selected by `--is_stage_2` alone. |
| `--matching_dim` / `--match_w4` / `--match_w8` / `--match_sigma4` / `--match_sigma8` / `--match_primary_stride` / `--match_pos_weight` | various | Stage-2 matching head hyperparameters. |
| `--fine_dim` / `--fine_margin` / `--sar_size` / `--match_w_fine` / `--match_sigma_fine` / `--match_fine_pos_weight` / `--fine_warmup_epochs` | various | Stage-2 fine-stage refinement (used by variants C/D). |

### 7.4 What gets saved
Only `trainable_tail` (the SAR encoder + projection / matching head) is
stored in each checkpoint. The frozen DINOv3 weights are not duplicated.
Across variants A/B/C/D, the `trainable_tail.sar_encoder.*` subtree is
byte-identical so a Stage-1 ckpt loads into any variant via
`strict=False`.

---

## 8. Benchmark / verification scripts

| Script | Purpose | Where to run |
|--------|---------|--------------|
| `Scripts/bench_imgnum_progress.sh` | E1: verify `imgNum` ↔ progress-bar alignment (single GPU). | Single idle GPU. |
| `Scripts/inspect_shard_order.py`   | E2: report intra-shard SAR pixel-mean drift (used to choose `shuffle_buffer`). | CPU only. |
| `Scripts/bench_batch_workers_sweep.sh` | E3: `(batchSize × dataloaderNum)` sweep for max throughput. | Single GPU. |
| `Scripts/bench_ddp_balance.sh` | DDP integration: small unbalanced-shard run that exercises `_balanced_split_by_node` + `dist.barrier()` across two epochs. | 4 GPUs. |
| `Scripts/verify_state_dict_compat.py` | All four variants share the same `sar_encoder.*` key layout. | CPU. |
| `Scripts/verify_variants_smoke.py` | End-to-end forward + loss + backward smoke test for all four variants on dummy tensors. | CPU. |

### Measured throughput (E3, single RTX 3090)

| batch ↓ \\ workers → | 4 | 8 | 16 |
|---|---|---|---|
| 4 | 90.9 | 88.9 | 87.0 |
| 8 | 173.9 | 170.2 | 177.8 |
| 16 | 340.4 | 347.8 | 347.8 |
| **32** | **581.8** | **581.8** | **581.8** |

img/s saturates around `batch = 32`. Under DDP-4 the step time picks up an
extra ~9 ms from NCCL all-reduce; with 32 workers per node hammering the
HDD that backs `/data5` we settle around 60–80 ms/step in practice.

---

## 9. Repository layout

```
Template-Matching-DINOv3
├── Datasets/                                 # training/testing lists (csv / pt) — gitignored
├── Scripts/
│   ├── bench_batch_workers_sweep.sh          # E3 throughput sweep
│   ├── bench_ddp_balance.sh                  # DDP barrier integration test
│   ├── bench_imgnum_progress.sh              # E1 imgNum / progress-bar verification
│   ├── inspect_shard_order.py                # E2 intra-shard structure probe
│   ├── pack_to_webdataset.py                 # repack .pt list → tar shards
│   ├── start_train_stage1_wds.sh             # Stage 1 (alignment) launcher
│   ├── start_train_stage2_{B,C,D}_wds.sh     # Stage 2 launchers (per variant)
│   ├── verify_state_dict_compat.py
│   └── verify_variants_smoke.py
├── Source/
│   ├── main.py                               # JackFramework entry
│   ├── Libs/dinov3/                          # DINOv3 submodule
│   ├── Tools/                                # csv2pt, path utilities
│   └── UserModelImplementation/
│       ├── user_define.py                    # global paths
│       ├── user_interface.py                 # CLI + UserInterface wiring
│       ├── Dataloaders/
│       │   ├── tm_dataloader.py              # JackFramework data interface
│       │   ├── _tm_webdataset.py             # WebDataset pipeline + _balanced_split_by_node
│       │   ├── _tm_datasets.py               # legacy map-style TM_Dataset
│       │   └── _data_aug.py                  # rotate / flip augmentation
│       └── Models/
│           ├── builder.py                    # ComposedTM = builder(config.yaml)
│           ├── _interface_base.py            # ComponentBasedInterface
│           ├── _components/                  # shared building blocks
│           │   ├── registry.py               # @register decorator
│           │   ├── alignment_loss.py
│           │   ├── coarse_matching.py
│           │   ├── dino_backbone.py
│           │   ├── fine_head_cascade.py      # 3-level cascade (variant C)
│           │   ├── fine_head_simple.py
│           │   ├── matching_loss.py
│           │   └── cascade_loss.py           # multi-level cascade loss
│           ├── _legacy/legacy_A/             # pre-refactor TM_Model_A code (for compat verify)
│           ├── TM_Model_A/                   # baseline — config.yaml + __init__.py
│           ├── TM_Model_B/                   # no Stage-1 inheritance
│           ├── TM_Model_C/                   # cascade fine head
│           └── TM_Model_D/                   # soft argmax + sub-pixel
├── Weights/                                  # DINOv3 checkpoints (not tracked)
└── README.md
```

---

## 10. Resume / diagnose a long-running training

```bash
pgrep -af main.py                                                                                   # process alive?
ls -la Checkpoint/model_epoch_*.pth | sort -k 9                                                     # most recent ckpt
tail -c 1000 start_train_stage1_wds.log | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' | tail -5        # current iter / loss
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv                               # GPU utilization
iostat -x 1 2                                                                                       # disk pressure on /data5
```

If the training crashes with an NCCL `ALLREDUCE` timeout at an epoch boundary,
suspect rank length mismatch first — confirm `_balanced_split_by_node` is the
nodesplitter in `_tm_webdataset.py` and `len(shards) % world_size == 0` after
the truncation.

---

## 11. License

MIT. See [`LICENSE`](LICENSE). DINOv3 is redistributed under its own
license; refer to the upstream repository.
