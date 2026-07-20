# Template-Matching-DINOv3 (DOSR-Net)

![Python](https://img.shields.io/badge/python-%3E=3.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

**DOSR-Net** is a cross-modal **SAR ↔ optical template-matching** framework for
remote-sensing image registration (TGRS 2026). It localizes a small SAR template
inside a larger optical reference patch by pairing a **frozen DINOv3 ViT-L/16**
optical backbone (satellite-pretrained) with a **jointly-trained SAR CNN encoder**,
upsampling both feature maps with **FeatUp / JBU** image-guided upsamplers, and
matching them through a **coarse-to-fine cascade** of normalized cross-correlations.
The whole model family is YAML-driven and trained with
[JackFramework](https://github.com/Archaic-Atom/JackFramework), which owns the
training loop, DDP launcher, logging and checkpoint I/O.

<p align="center"><img src="assets/pipeline.png" width="100%" alt="DOSR-Net pipeline"></p>

> **Paper (IEEE TGRS 2026):** *Rethinking SAR–Optical Pre-Alignment with
> Foundation Models for Template Matching.* The central finding is that a usable
> cross-modal similarity does **not pre-exist** in the encoders — it is
> *constructed* inside a task-specific matching projection and is
> high-dimensional (needs ≳64 directions), whereas contrastive pre-alignment
> supplies only a low-rank (≈22) code that fine-tuning **overwrites**. So a
> direct-trained cascade, with no pre-alignment stage, reaches state of the art.
> See [Results](#results).

The repository is organized as an **analysis-led ablation family (`TM_Model_A..F`)**.
The headline/proposed model is **`TM_Model_C`** — a direct-trained cascade matching
head. One of the paper's findings is that contrastive Stage-1 *pre-alignment* of the
SAR encoder does **not** improve in-domain matching over direct training; several
variants exist precisely to measure that (see [Model variants](#model-variants)).

> For the authoritative, deep developer guide (entry-point chain, config-vs-CLI
> precedence, dataloader contracts, the cascade head internals, and a list of known
> pitfalls) see **[`CODE_STRUCTURE.md`](CODE_STRUCTURE.md)**. This README is a
> code-usage overview and links into that document rather than duplicating it.

---

## Key features

- **256/192 geometry.** Optical reference patch **256×256**, SAR template **192×192**;
  the ground-truth label `(x, y)` is the **top-left corner** of the SAR template
  inside the optical patch, so displacement lives in `[0, 64]`. (Migrated from the
  earlier 128/64 geometry to match the ME-SSM evaluation protocol.)
- **Frozen DINOv3 ViT-L/16 optical backbone** (`dinov3_vitl16`, satellite-pretrained
  `sat493m` weights, 1024-d stride-16 features) paired with a **jointly-trained
  `SarResNet16GN`** SAR encoder (ResNet-style, GroupNorm, no BatchNorm).
- **FeatUp / JBU upsampling.** Image-guided `JBULearnedRange` upsamplers
  (`featup.upsamplers`) lift stride-16 features to stride-4/8 for matching; trained
  end-to-end (FeatUp's pretrained DINOv2 weights are **not** reused).
- **Coarse-to-fine cascade matching.** Normalized cross-correlation (SAR template =
  conv kernel, optical search region = input) at coarse stride-4 + stride-8, then a
  windowed stride-2 and stride-1 refinement around the detached coarse argmax.
- **JackFramework + WebDataset training.** Iterable WebDataset shard pipeline for the
  ~13.2M-sample training set, with custom DDP-balanced shard splitting, large sample
  shuffle buffers, and a map-style backend for evaluation.
- **`TM_Model_A..F` ablation family** isolating Stage-1 pre-alignment-as-init, cascade
  vs standard-fine, sub-pixel decode, and a learned-inner-product (VICReg) probe.
- **Cross-dataset zero-shot evaluation** adapters for **WHU-OPT-SAR** (cross-sensor)
  and **QXS-SAROPT** (cross-resolution), plus a controlled GSD-sweep harness.
- **Analysis / probe tooling** (`Source/Tools/`, `Source/Tools/probes/`) that produces
  the mechanistic numbers in the paper (where similarity lives, control batteries,
  effective-rank trajectories, oracle ceilings).

---

## Repository structure

```
Template-Matching-DINOv3/
├── CODE_STRUCTURE.md              # authoritative developer guide (read this)
├── Source/
│   ├── main.py                    # the ONLY entry point (python Source/main.py ...)
│   ├── Libs/dinov3/               # DINOv3 git submodule (frozen optical backbone)
│   ├── UserModelImplementation/
│   │   ├── user_interface.py      # CLI surface + dataset/model zoo dispatch
│   │   ├── user_define.py         # ROOT_PATH / DINOv3 repo + weights path constants
│   │   ├── Models/
│   │   │   ├── __init__.py        # model_zoo: TM_Model_A..F → interface classes
│   │   │   ├── builder.py         # YAML → ComposedTM assembly
│   │   │   ├── _interface_base.py # generic JackFramework model handler
│   │   │   ├── TM_Model_{A..F}/   # per-variant config.yaml (+ thin __init__.py)
│   │   │   ├── _components/        # backbones, JBU, matching heads, losses, decode
│   │   │   └── _legacy/legacy_A/   # A's original hand-written code (reference only)
│   │   └── Dataloaders/            # TMDataloader (map-style + WebDataset backends)
│   └── Tools/                      # eval/metric/probe scripts (+ probes/)
├── Scripts/                        # train / eval / data-gen launchers (256/192)
│   └── legacy_128_64/              # superseded 128/64-geometry launchers
├── Datasets/                       # only a stale example is tracked; real manifests live on servers
├── Weights/                        # DINOv3 checkpoint (gitignored, download separately)
├── assets/                         # README figures (pipeline, qualitative, ...)
├── MODELS.md                       # pretrained-model manifest (download + placement)
├── README.md / README_zh.md
└── LICENSE
```

See **[`CODE_STRUCTURE.md` §1](CODE_STRUCTURE.md)** for the full, current tree and the
"JackFramework lives outside this repo" mental model.

---

## Model variants

All six variants share the **same backbones** (frozen DINOv3 ViT-L/16 +
`SarResNet16GN`), the **same coarse stride-4 + stride-8 JBU matching**, and the
**256/192 geometry** (`matching_dim = 256`, `sar_size = 192`). Each `TM_Model_X/` is a
thin `ComponentBasedInterface` subclass that only sets `CONFIG_PATH`; all behavior is
in its `config.yaml`. Variants differ along three axes: the **Stage-1 objective**, the
**Stage-2 SAR-encoder freeze**, and the **fine/decode head**.

| Variant | Matching head | Stage-1 (`load_pretrained_sar`) | Stage-2 SAR (`freeze_sar_encoder`) | Stage-2 loss | Decode | What it isolates |
|---|---|---|---|---|---|---|
| **A** | `StandardMatchingHeadJBU` | `true` (warm-start from Stage-1 aligned ckpt) | `false` (keeps training) | `two_stage_matching_loss` | hard argmax | Stage-1 pre-alignment **as initialization** (only diff vs B) |
| **B** | `StandardMatchingHeadJBU` | `false` (random SAR init) | `false` | `two_stage_matching_loss` | hard argmax | No-Stage-1 control / baseline for A |
| **C** | `CascadeHeadJBU` | `false` (direct-trained, no Stage-1) | `false` (joint) | `cascade_loss` | hard argmax | **The proposed method** — direct-trained cascade |
| **D** | `StandardMatchingHeadJBU` | `true` | `true` (frozen) | `two_stage_matching_loss` | `soft_subpixel` | Sub-pixel decode on A's head (reuses A's ckpt; eval-only) |
| **E** | `CascadeHeadJBU` | `true` * | `true` * | `cascade_loss` | `soft_subpixel` | Cascade + sub-pixel; **evaluated on C's checkpoint** |
| **F** | `StandardMatchingHeadJBU` (+ `AlignmentIPHead` in Stage-1) | `true` | `true` (frozen) | `two_stage_matching_loss` | hard argmax | Learned-inner-product Stage-1 (W_s/W_d + VICReg rank floor), frozen-encoder Stage-2 |

**Checkpoint-reuse:** D reuses A's checkpoint and E reuses C's checkpoint — both add
only an eval-time `argmax_type` (sub-pixel) change, with `matching.args` kept
byte-identical so the state_dicts match. Only `matching.args` is byte-identical, not
the whole config: like E, D's own training block differs from the model it borrows from
(D sets `freeze_sar_encoder: true` vs A's `false`), so D is meant to be *evaluated on*
A's checkpoint, not trained from its own config. Sub-pixel exists in code
(`_components/sub_pixel.py`) but is **not** a headline contribution of the paper.

> \* **TODO / caveat:** E's architecture and state_dict are byte-compatible with C, and
> E is intended to be *evaluated on C's checkpoint* with sub-pixel decode. However, E's
> own `config.yaml` training block (`load_pretrained_sar: true`, `freeze_sar_encoder:
> true`, non-zero warmups) is **stale** relative to C's direct-training pivot and does
> not reproduce C if launched as written. Treat E as "C's architecture + sub-pixel
> decode, evaluated on C's checkpoint"; confirm intent before training E from its own
> config.

The paired comparisons the table supports: **A vs B** isolates Stage-1-as-init;
**C vs B** isolates cascade vs standard-fine; **A vs D** and **C vs E** isolate
sub-pixel decode; **F** isolates a protected, strong-Stage-1 objective. A 2×2
"frozen-arms" causal study (init × freeze) is documented in
[`CODE_STRUCTURE.md §7`](CODE_STRUCTURE.md).

---

## Results

All numbers follow **ME-SSM's evaluation protocol** on SEN1-2 (optical reference
256×256, SAR template 192×192): per-template **CMR@k px** (correct-match rate — one
`(x, y)` per template, dense zero-reject).

### In-domain SOTA (SEN1-2, full test set)

The proposed direct-trained cascade (`TM_Model_C`) beats prior work at **both**
thresholds. Full test set = all 7,740 pairs × 289 offsets (≈2.24M positions).

| Method | Venue | CMR(1) ↑ | CMR(3) ↑ | CMR(5) ↑ |
|---|---|---|---|---|
| ME-SSM (protocol source) | TGRS 2026 | 62.14 | 89.19 | 93.04 |
| VU-Net | ICSIDP 2024 | 63.0 | 87.0 | 90.0 |
| RMSO-ConvNeXt | — | — | 94.36 | — |
| **Ours (cascade C, direct-trained)** | — | **93.30** | **95.83** | **96.30** |

Mean error 1.13 px, median 0 px. `Checkpoint_C_phase2_8gpu/model_epoch_2.pth`.

### Pre-alignment is redundant under fine-tuning (2×2 causal test, 20k)

Stage-1 contrastive pre-alignment helps **only when the encoder is frozen**; once the
encoder is fine-tuned, aligned init (S1/Tr) ≈ random init (R/Tr).

| Arm | Init | Stage-2 encoder | CMR(1) | CMR(3) | checkpoint |
|---|---|---|---|---|---|
| S1/Tr | Stage-1 aligned | trainable | 68.30 | 86.08 | `Checkpoint_A_ft_256/model_epoch_4.pth` |
| R/Tr  | random | trainable | 68.28 | 86.35 | `Checkpoint_B_256/model_epoch_4.pth` |
| S1/Fr | Stage-1 aligned | **frozen** | 68.63 | 86.30 | `Checkpoint_F_s2_frozen_256/model_epoch_4.pth` |
| R/Fr  | random | **frozen** | 66.94 | 85.17 | `Checkpoint_RF_s2_frozen_256/model_epoch_4.pth` |

### Why (mechanism, Sec. IV)

- **Similarity is constructed, not pre-existing.** Raw DINOv3×SAR features localize no
  better than chance; matching similarity is built inside the projection and is
  **high-dimensional** — restricting the matcher to the top 22 shared directions drops
  the GT-hit rate from 0.897 to 0.565 (needs ≳64 directions at stride-4).
- **Pre-alignment supplies only a low-rank code.** Contrastive alignment saturates at
  effective rank ≈22 (four independent confirmations; stronger regularization cannot
  raise it).
- **Fine-tuning overwrites it.** End-to-end training drives the aligned encoder's
  effective rank ≈23 → ≈12 (a from-scratch matcher's regime), so the aligned code is
  overwritten rather than built upon.

Reproduce every mechanism number with the probes in
[`Source/Tools/probes/`](Source/Tools/probes/) (default specimen
`Checkpoint_B_256/model_epoch_4.pth`).

### Cross-dataset zero-shot (cascade, no retraining)

The specialized cascade transfers strongly on its confident half but degrades on the
coarse tail; OOD robustness *decreases* with training specialization (see
[`CODE_STRUCTURE.md`](CODE_STRUCTURE.md) and the paper's Discussion).

| Dataset | shift | CMR(3) | high-conf CMR(3) | gross-fail % |
|---|---|---|---|---|
| WHU-OPT-SAR | cross-sensor (10 m) | 74.39 | 96.12 | 4.64 |
| QXS-SAROPT | cross-resolution (1 m) | 10.40 | — | 31.35 |

<p align="center"><img src="assets/specialization.png" width="55%" alt="Specialization scissors: in-domain rises, OOD falls"></p>

### Qualitative (SEN1-2 in-domain)

SAR template (grey) pasted at the **predicted** location; green box = ground truth.

<p align="center"><img src="assets/qualitative_sen12.png" width="90%" alt="Qualitative SEN1-2 matches"></p>

---

## Pretrained models

The full **key-set** of checkpoints (proposed cascade + the six ablation / Stage-1
models backing Tables I–VII) is released as two archives. See
[`MODELS.md`](MODELS.md) for the per-model manifest (role, paper claim, placement).

| archive | size | sha256 (first 16) | contents |
|---|---|---|---|
| `PaperModels_TGRS2026_m5.zip` | 272 MB | `be9d348822477b8f` | proposed cascade `TM_Model_C` (headline) |
| `PaperModels_TGRS2026_m2.zip` | 698 MB | `5df5a6712a3311f5` | A-ft, B, S1F, RF, Stage-1 A, Stage-1 F |

**Placement:** unzip **into the project root** so the `Checkpoint_*/` directories sit
next to `Source/`. The launchers and configs already reference these exact names, e.g.:

```bash
bash Scripts/run_eval_C_2k.sh Checkpoint_C_phase2_8gpu/model_epoch_2.pth c_ep2 0
```

The **DINOv3 ViT-L/16 satellite backbone** (`dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`,
1.2 GB) is the **public** Meta weights — not trained by us and not in the archives above;
download it separately into `Weights/` (see [Installation](#installation) step 5).

> **Download (Baidu netdisk / 百度网盘):**
> <https://pan.baidu.com/s/1lhXTqkXKwzg5H9PVtt01EQ?pwd=ug7r> — extraction code: `ug7r`

---

## Installation

> **No `requirements.txt` / `environment.yml` / `setup.py` ships in this repo.** The
> dependency set below is reconstructed from imports and from the launch scripts (all
> of which target a conda env named **`stereoA`** on the training server). Versions are
> not pinned here — install into a recent PyTorch 2.x environment.

```bash
# 1. Clone with the DINOv3 submodule (frozen optical backbone).
git clone --recurse-submodules <this-repo-url>
cd Template-Matching-DINOv3
# SSH-less? The submodule URL is git@github.com:facebookresearch/dinov3.git;
# rewrite it to https://github.com/facebookresearch/dinov3.git if you lack SSH keys.

# 2. Install JackFramework (not vendored here; owns the training loop / DDP / I/O).
git clone https://github.com/Archaic-Atom/JackFramework.git
cd JackFramework && ./install.sh && cd ..

# 3. Install FeatUp (REQUIRED — external dependency, not a submodule).
#    The JBU upsamplers (every current variant uses them) import
#    `featup.upsamplers.JBULearnedRange`. Install from facebookresearch/FeatUp.

# 4. Other Python deps (into the same env):
#    torch, torchvision, numpy, pandas, Pillow, webdataset
#    (webdataset is required for the iterable training path).

# 5. Place the DINOv3 satellite-pretrained weights under Weights/ (gitignored):
#    Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth   (DINOv3 ViT-L/16, ~1.2 GB)
```

**Path constants** live in `Source/UserModelImplementation/user_define.py`:
`ROOT_PATH` (default `/data/raozhibo/Documents/Programs/Template-Matching-DINOv3/`),
`DINOv3_PATH = Source/Libs/dinov3/`, and
`WEIGHTS_PATH = Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`. Edit `ROOT_PATH`
per machine, or override it at launch with `--root_path` (it is the value substituted
into each `config.yaml`'s `${root_path}` placeholder for the DINO repo/weights paths).

**DDP caveat:** because the optical backbone (and, in some variants, the SAR encoder)
is frozen, JackFramework's DDP wrap needs `find_unused_parameters=True`. That patch
lives only on the training servers' JackFramework install — see
[`CODE_STRUCTURE.md §2.5`](CODE_STRUCTURE.md).

---

## Data preparation

### In-domain SEN1-2 (256/192) WebDataset shards

The production data generator is **`Scripts/make_opt256_sar192_shards.py`** (a
byte-identical copy lives at `Source/Tools/make_opt256_sar192_shards.py`). It crops and
packs co-registered 256×256 SEN1-2 (Sentinel-1 SAR `s1_*` + Sentinel-2 optical `s2_*`,
four seasons) directly into WebDataset `.tar` shards — no intermediate PNG dump:

- **Optical reference** = the original 256×256 `s2` PNG bytes, copied verbatim.
- **SAR template** = the `s1` image cropped to 192×192 at top-left `(cx, ry)` and
  re-encoded as PNG.
- Each sample = three tar entries sharing a key:
  `<key>.sar.png` (192×192 grayscale), `<key>.rgb.png` (256×256 RGB),
  `<key>.label.cls` (ASCII `"<cx> <ry>"` — top-left corner, **not** centered, **not**
  normalized).

Train/test are **disjoint at the scene/patch level** (per-season fixed-seed split, no
scene leakage). Offsets multiply the pairs: **TEST** uses a dense stride-4 grid
(`17×17 = 289` offsets per pair); **TRAIN** uses random offsets (default 48 per pair).

```bash
# Generate test split first (fast), then train split (nohup-friendly wrapper):
bash Scripts/run_gen_opt256_sar192.sh
# or directly:
python Scripts/make_opt256_sar192_shards.py --split test  --workers 16
python Scripts/make_opt256_sar192_shards.py --split train --workers 16
```

This writes `TM_shards_opt256_sar192_{train,test}/` plus a **shard-manifest CSV**
`shard_path.csv` with header `path,file_num` — the iterable backend reads this to learn
each shard's sample count (and uses `sum(file_num)` as the authoritative on-disk
total). With default settings the test set is ≈2.24M samples (7740 pairs × 289) and the
full train set is ≈13.18M samples; the exact train count depends on the SEN1-2 tree.

A small **map-style** evaluation CSV (`SAR_Image,RGB_Image,Label` with
`Label = "[cx, ry]"` JSON) is built by **`Scripts/make_2k_eval_256.py`** from the same
disjoint `test_pairs.txt`, defaulting to 2000 samples.

### Cross-dataset zero-shot adapters

**`Source/Tools/cross_dataset_adapter.py`** resamples external SAR-optical datasets to a
target GSD and emits SEN1-2-shaped 256/192 eval pairs (`SAR_SIZE=192`, `RGB_SIZE=256`;
note the module/class docstrings still say "64/128" — that text is stale, trust the
code):

| Adapter (`--dataset`) | Source | Native → target GSD | Role |
|---|---|---|---|
| `whu_opt_sar` | WHU-OPT-SAR (GF-1 + GF-3) | 5 m → 10 m (downsample) | cross-sensor |
| `qxs_saropt`  | QXS-SAROPT (GF-3 + Google Earth, native 256×256) | 1 m → 1 m (no resample) | cross-resolution |
| `sarptical`   | SARptical (TerraSAR-X + aerial) | 1 m → 10 m | cross-sensor (aux) |

```bash
bash Scripts/run_regen_cross_256.sh     # regen WHU + QXS at 256/192
```

Each run writes `sar/`, `rgb/`, a `manifest.pt` (for `Scripts/pack_to_webdataset.py`), a
`cross_eval.csv` (`SAR_Image,RGB_Image,Label` — same schema as the in-domain map-style
eval CSV, directly consumable by `--mode test`), and a `meta.json` provenance file. A
`--target_res_m` flag (downsample-only, guarded) drives a controlled GSD sweep
(`Scripts/run_whu_res_sweep.sh`).

### Manifest CSV formats (do not confuse them)

| | Shard-manifest CSV (iterable backend) | Path-list CSV (map-style backend) |
|---|---|---|
| Header | `path,file_num` | `SAR_Image,RGB_Image,Label` |
| One row = | one `.tar` shard | one sample |
| `Label` | — | JSON `[x, y]` string |

> The only git-tracked file under `Datasets/` is
> `dataset_example_training_list.csv`, which is a **stale stereo-disparity example**
> (`left_img,right_img,gt_disp`) — **not** a TM manifest. The real manifests
> (`shard_train_256_full.csv`, the subset warm-start manifests, the eval CSVs) are
> simply **not committed** — they live only on the training servers. Note they are
> **not** covered by `.gitignore` (only `Datasets/image_*.csv|.pt` and `Datasets/*.txt`
> are), so take care not to accidentally commit a shard/eval manifest.

---

## Training

There is **no separate `train.py`** — everything runs through
`python Source/main.py <flags>`, with `--mode train|test` selecting behavior. The CLI
has two flag families: a small set of **project flags** (e.g. `--is_stage_2`,
`--sar_size`, `--root_path`, `--shuffle_buffer`) and the **JackFramework builtins**
(`--mode --modelName --dataset --dist --gpu --batchSize --port --lr --maxEpochs
--imgNum --trainListPath --modelDir ...`).

> **Important precedence rule:** on the YAML path most architecture/loss CLI flags
> (`--matching_dim`, `--fine_dim`, `--fine_margin`, `--match_w*`, `--fine_warmup_epochs`)
> are **inert** — head dims and loss weights come from each variant's `config.yaml`.
> The launch scripts pass them anyway "for parity." The flags that *do* take effect
> include `--is_stage_2`, `--lr`, `--maxEpochs`, `--root_path`, and `--sar_size`
> (dataloader side). See [`CODE_STRUCTURE.md §2.3 / §11`](CODE_STRUCTURE.md).
>
> **`sar_size` dual-channel trap:** the matching head reads
> `architecture.matching.sar_size` from `config.yaml` (a literal `192`, not
> CLI-substitutable) while the dataloader reads `--sar_size` from the CLI. **Both must
> agree.** Always pass `--sar_size 192` and keep the config at `192`.
>
> **`--imgNum` is only the progress-bar denominator**, not the true epoch length (the
> WebDataset path uses `with_epoch(per_worker_samples)` derived from the manifest's
> `real_total`). The convention baked into the scripts is `--imgNum == sum(file_num)`;
> the launchers compute it via `awk` over the manifest.

### The proposed method (`TM_Model_C`) — two-phase recipe

`TM_Model_C` is direct-trained (random SAR init, no Stage-1, SAR encoder trained
jointly with the cascade). The scientific run is **two phases**, both via
`Scripts/run_train_C_direct.sh`:

```bash
# Phase 1 — subset warm-start (subset40 manifest ~1.34M, 4-GPU DDP, bs10, 5 epochs).
bash Scripts/run_train_C_direct.sh sub40

# Phase 2 — full set ×3 epochs, warm-started from the Phase-1 checkpoint.
# Internally: strip_stage1_opt.py drops optimizer state + resets epoch=-1 so JF
# starts cleanly at epoch 0, then pins it via a hand-written checkpoint.list.
bash Scripts/run_train_C_direct.sh full_resume Checkpoint_C_direct_sub40_256/model_epoch_4.pth
```

`Scripts/chain_to_phase2.sh` automates the handoff (polls until Phase 1 exits, verifies
`model_epoch_4.pth`, waits for the GPUs to free, then launches Phase 2 under `nohup` so
it survives an SSH drop). An alternative one-shot launcher,
`Scripts/run_train_C_direct_full.sh`, skips the subset phase (modes `smoke` / `mem N` /
`full N`).

> Note: `run_train_C_direct.sh full_resume` runs **3** epochs while
> `run_train_C_direct_full.sh full` runs **5**, and both write to
> `Checkpoint_C_direct_full_256/`. They are two alternative routes to the full run, not
> interchangeable — pick one deliberately.

**Always smoke-test single-GPU first** (`--dist False` surfaces real tracebacks):

```bash
bash Scripts/run_train_C_direct.sh smoke
```

The literal flags emitted by `run_train_C_direct.sh` (substitute your own python /
project root / paths):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u Source/main.py \
  --mode train --dataset tmdataset --modelName TM_Model_C --is_stage_2 True \
  --dataloaderType iterable --dataloaderNum 8 --sampleNum 1 \
  --sar_size 192 --matching_dim 256 --shuffle_buffer 100000 --lr 1e-3 \
  --auto_save_num 1 --dist True --gpu 4 --batchSize 10 --port 8867 \
  --trainListPath ./Datasets/shard_train_256_full.csv \
  --imgNum <sum of file_num over the manifest, ~13.18M> --maxEpochs 3 \
  --log ./log_c_full/ --outputDir ./result_c_full/ \
  --resultImgDir ./ResultImg_c_full/ --modelDir ./Checkpoint_C_direct_full_256/
```

> `--dataloaderType` is a **JackFramework core flag that defaults to `mapstyle`** — you
> must pass `--dataloaderType iterable` explicitly to use the WebDataset shard pipeline.

### Other variants / ablations (256/192 launchers)

- `Scripts/run_a_ft_vs_b.sh` — the clean "does Stage-1 init help" ablation: A-ft
  (warm-started, trainable SAR) vs B (random SAR), only the SAR init differs.
- `Scripts/start_train_stage1_256_A.sh` / `start_train_stage1_256_F.sh` — Stage-1
  contrastive pre-alignment (A) and learned-inner-product/VICReg pre-alignment (F).
- `Scripts/run_frozen_arms_F.sh` / `run_frozen_arms_F_seed.sh` — the frozen-encoder
  double-arms experiment (and its multi-seed replicate).

> The `Scripts/start_train_stage2_{A..E}_wds.sh` launchers and everything under
> `Scripts/legacy_128_64/` are **legacy 128/64-geometry** launchers, superseded by the
> 256/192 scripts above. The legacy Stage-2 launchers are still the clearest in-repo
> statement of the original A–E ablation semantics, but should not be used to train at
> the current geometry.

### Checkpoints

Only `model.trainable_tail.state_dict()` is saved (SAR encoder + matching head +
fusion; the frozen DINO backbone is not duplicated), every `--auto_save_num` epochs,
into `--modelDir`. JackFramework picks which checkpoint to load via a plain-text
`checkpoint.list` in that dir (two lines: `last model name:<file>` then `<file>`). To
resume, point `--modelDir` at a dir whose `checkpoint.list` names the desired file; all
loads are `strict=False`. Warm-starting a *fresh* run from a prior checkpoint uses
`Source/Tools/strip_stage1_opt.py` to drop the optimizer state and reset the epoch
counter.

---

## Evaluation

Evaluation runs `--mode test` on the **map-style** backend (eval scripts intentionally
do **not** pass `--dataloaderType iterable`) against a map-style CSV
(`SAR_Image,RGB_Image,Label`). Each eval script pins one checkpoint into a clean dir,
writes a 2-line `checkpoint.list`, and runs at batch 1 with `TM_SEED=2024` +
`CUBLAS_WORKSPACE_CONFIG=:4096:8` for determinism.

```bash
# In-domain 2k (proposed cascade, hard argmax):
bash Scripts/run_eval_C_2k.sh <src_ckpt.pth> <tag> [gpu]

# Cascade + sub-pixel (E), evaluated on a C checkpoint:
bash Scripts/run_eval_E_2k.sh <C_ckpt.pth> <tag> [gpu]

# Standard-head variants A / B / D:
bash Scripts/run_eval_std_2k.sh <TM_Model_A|B|D> <src_ckpt.pth> <tag> [gpu] [port]

# Larger in-domain eval sets (10k / 20k) and frozen-arms full pipeline:
bash Scripts/run_eval_B_256_10k.sh
bash Scripts/run_eval_A_256_10k.sh
bash Scripts/run_eval_frozen_full.sh [FEPOCH] [EVAL_CSV] [EVAL_N] [TAG]

# Cross-dataset zero-shot (WHU cross-sensor, QXS cross-resolution):
bash Scripts/run_eval_whu_matched.sh [CSV] [N] [TAG]
bash Scripts/run_whu_res_sweep.sh      # controlled GSD sweep (PCK vs GSD)
```

`--mode test` writes per-rank temp CSVs that rank-0 merges into
`test_<dataset>_<timestamp>.csv` (columns
`index,sar_name,rgb_name,gt_x,gt_y,pred_x,pred_y,score`). Turn that into metrics with:

```bash
python Source/Tools/pck_summary.py <merged_test.csv>     # PCK/CMR@{1,2,3,5,10}px + splits
python Source/Tools/eval_test_csv.py --csv <merged_test.csv>   # L2 stats + PCK@1..5 (+ --n/--seed)
```

The reported metric is **per-template PCK / CMR** (correct-match rate at a pixel
threshold — one `(x, y)` per template). For paired model comparisons,
`Source/Tools/paired_bootstrap_pck.py` reports point estimate, 95% CI, and
`P(A better than B)`.

---

## Analysis & probe tooling

`Source/Tools/` holds the analysis and metric scripts; `Source/Tools/probes/` holds the
mechanism probes that produce the paper's analysis-section numbers (default specimen
`Checkpoint_B_256/model_epoch_4.pth`). They fall in three groups — see
[`Source/Tools/probes/README.md`](Source/Tools/probes/README.md) for the full table.
The core *constructed-similarity* set:

- `probe_b_similarity.py` — **where** B's learned cross-modal similarity lives (raw
  stride-16 cosine vs the projected stride-4 matching subspace; sign check).
- `probe_b_controls.py` — is the localization **genuine** cross-modal similarity or a
  content-blind shortcut? (control battery, incl. foreign-template swap).
- `probe_b_content.py` — why are raw DINO/SAR features orthogonal yet projectable?
  (CCA shared subspace + edge-structure correlation).
- `probe_b_keystone.py` — does matching actually **use** the shared subspace, and is
  that subspace real vs spatial-autocorrelation leakage (de-leaked CCA)?
- `probe_b_stage1sim.py` — would a **learned inner product** beat raw cosine as the
  Stage-1 metric?
- `probe_b_collapse.py` — does a learned-IP Stage-1 collapse, and is a learnable `W_d`
  safe? (effective-rank capacity probe).
- `probe_f0_mechanism.py` — does Stage-2 fine-tuning **erase** Stage-1's alignment?
  (the "F0" mechanism: S1 vs A-ft vs B encoder alignment + cross-encoder CKA).

Plus two more groups (folded in from the analysis round): the
**granularity–dimensionality** recovery sweeps (`probe_b_granularity_dim.py`,
`probe_c_granularity_dim.py`, `probe_b_matchdim{,_ood}.py`,
`probe_b_conditional_dims.py`, `probe_recovery_by_stride.py`,
`probe_alignment_recovery.py` — matching needs ≳64 distributed dims) and the
**effective-rank / overwrite** probes (`probe_erank_extractors.py`,
`probe_erank_cascade.py`, `probe_erank_granularity.py` — aligned 23 → fine-tuned 12).

Supporting tools: `oracle_geom_ceiling.py` (theoretical matching ceiling),
`stage1_separation_2k.py` / `analyze_stage1_separation.py` (Stage-1 separability, d′),
`coarse_eval_ab.py` (does Stage-1 help the coarse level), `ckpt_inspect.py` (dim128 vs
dim256 / JBU presence), `compare_dataset_stats.py` (domain-gap statistics),
`parse_erank_logs.py` / `plot_frozen_arms.py` (paper-figure data prep),
`stage_ablation.py` (cascade-stage ablation), `ci_reject.py` (risk–coverage /
rejection curve), `profile_cascade.py` (runtime / efficiency profiling).

---

## Citation

If you use this code, please cite the paper (TGRS 2026):

```bibtex
@article{rao2026rethinking,
  title   = {Rethinking {SAR}--Optical Pre-Alignment with Foundation Models
             for Template Matching},
  author  = {Rao, Zhibo and He, Mingyi and Li, Xing and Dai, Yuchao and
             Zhang, Congxuan and Shen, Zhelun and Ding, Biyun},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {To appear}
}
```

---

## Acknowledgements

- **DINOv3** (Meta AI / FAIR) — the frozen optical backbone, vendored as a git
  submodule at `Source/Libs/dinov3/` and redistributed under its own license.
  <https://github.com/facebookresearch/dinov3>
- **FeatUp** — the JBU image-guided upsamplers
  (`featup.upsamplers.JBULearnedRange`). <https://github.com/facebookresearch/FeatUp>
- **JackFramework** — the training framework (loop, DDP, logging, checkpointing).
  <https://github.com/Archaic-Atom/JackFramework>

---

## License

MIT. See [`LICENSE`](LICENSE). DINOv3 is redistributed under its own license; refer to
the upstream repository.

> **TODO:** the `LICENSE` file currently reads `Copyright (c) 2018 BeautyOfWeb`, which
> appears to be an inherited template artifact rather than this project's authors — the
> copyright line likely needs correcting.
