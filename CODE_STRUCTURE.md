# TM-DINOv3 / DOSR-Net — Code Structure (developer guide)

A two-stage SAR↔optical template-matching registration system built on the
user's home-grown **JackFramework (JF)** training framework. This document maps
the codebase for a developer who needs to launch, debug, or extend training —
with an emphasis on the **config-vs-CLI precedence** rules that bite hardest.

> Path convention: all paths below are relative to the repo root
> `/Users/rhc/WorkSpace/002_Programs_our/005_Template-Matching-DINOv3`.
> Remote copy lives at `~/Documents/Programs/Template-Matching-DINOv3` (or
> `/data/raozhibo/Documents/Programs/Template-Matching-DINOv3`).

---

## 1. Overview & repo layout

```
Source/
  main.py                          # process entrypoint (tiny: seed + hand off to JF)
  UserModelImplementation/
    user_interface.py              # ALL project CLI flags (user_parser); JF owns the rest
    user_define.py                 # path constants (ROOT_PATH, DINOv3_PATH, WEIGHTS_PATH)
    Models/
      __init__.py                  # model_zoo: --modelName -> TM_Model_A..F interfaces
      _interface_base.py           # ComponentBasedInterface: THE shared JF ModelHandler
      builder.py                   # YAML -> ComposedTM; ${var} substitution; freeze plumbing
      TM_Model_{A..F}/config.yaml  # per-variant config (architecture + training + inference)
      _components/                 # encoders, matching heads, losses, decode
      _legacy/legacy_A/            # OLD argparse-driven path — NOT in the zoo, dead code
    Dataloaders/
      __init__.py                  # dataloaders_zoo: 'tmdataset' -> TMDataloader
      tm_dataloader.py             # JF DataHandlerTemplate; backend select; test CSV export
      _tm_webdataset.py            # iterable WebDataset pipeline (the LIVE path)
      _tm_datasets.py              # legacy map-style Dataset (mirror of the iterable processor)
      _data_aug.py                 # rot90/flip aug (Stage-1 train only)
  Tools/                           # probes, eval, shard packer, ckpt utilities
  Libs/dinov3/                     # vendored DINOv3 repo (torch.hub local load)
Scripts/                           # launcher .sh + eval .sh
  legacy_128_64/                   # superseded 128/64-geometry launchers
Datasets/                          # only dataset_example_training_list.csv locally; manifests are remote
Weights/                           # dinov3_vitl16 pretrained weights (remote)
```

**Mental model.** `main.py` does almost nothing — it hands the entire CLI, DDP
launcher, training loop, progress bar, checkpoint I/O, and resume to JF. The
project supplies: (1) one `UserInterface` (CLI flags + zoo dispatch), (2) one
shared `ComponentBasedInterface` model handler that is *configured by YAML*, and
(3) one `TMDataloader`. The six variants A–F are **4-line subclasses** that only
set `CONFIG_PATH`; all behaviour comes from each variant's `config.yaml`.

> **JackFramework is NOT in this repo.** It is installed only on the remote
> servers. Its arg parser, DDP launcher, training loop, progress bar, and
> `checkpoint.list` resume logic cannot be read locally — their semantics below
> are reconstructed from the launch scripts, the README, and how `args.*` is
> consumed in user code. To inspect JF internals you must read it on the remote.

---

## 2. Training entrypoint & JackFramework glue

### 2.1 Entrypoint chain

`Source/main.py:37-40`: `main()` calls `_maybe_fix_seed()` then
`jf.Application(UserInterface(), "Template Matching model").start()`.

`_maybe_fix_seed()` (`main.py:12-34`) reads env var `TM_SEED`. If unset →
no-op (training keeps `cudnn.benchmark=True` for speed). If set → seeds
`random`/`numpy`/`torch`/`cuda` and forces `cudnn.deterministic=True`,
`benchmark=False`. **Only eval/probe scripts set `TM_SEED`; normal training runs
unseeded.**

### 2.2 Two flag families — the override boundary

`UserInterface.user_parser()` (`user_interface.py:22-75`) is the **only** place
the project adds argparse flags. Anything not added there is a **JF-builtin
flag** parsed by JF before `user_parser` runs.

**PROJECT flags** (added in `user_parser`):
`--root_path --dinov3_path --pre_trained_weights_path --pre_trained`
(deprecated), `--is_stage_2` (default False), `--matching_dim` (128),
`--match_w4` (1.0), `--match_w8` (0.5), `--match_sigma4` (1.0),
`--match_sigma8` (1.0), `--match_primary_stride` (4), `--match_pos_weight`
(None), `--fine_dim` (32), `--fine_margin` (4), `--sar_size` (64),
`--match_w_fine` (1.0), `--match_sigma_fine` (1.0), `--match_fine_pos_weight`
(None), `--fine_warmup_epochs` (3), `--shuffle_buffer` (100000).

**JF-builtin flags** (in launch scripts/README, NOT in `user_parser`):
`--mode --batchSize --gpu --port --dist --lr --maxEpochs --imgNum --valImgNum
--auto_save_num --dataloaderNum --dataloaderType --sampleNum --trainListPath
--valListPath --modelName --dataset --log --outputDir --resultImgDir --modelDir`.

### 2.3 config-vs-CLI precedence — THE most important rule

On the active (YAML-driven) code path, **most architecture/loss CLI flags are
DEAD**. `builder._resolve_args` (`builder.py:66-87`) substitutes only `${var}`
placeholders into config values, and **the only placeholder used in any of the
six `config.yaml` files is `${root_path}`** (for the DINOv3 repo/weights paths).
Head dims come from `config.yaml` (e.g. `TM_Model_B/config.yaml:39-44`
`matching_dim:256, fine_dim:32, fine_margin:24, sar_size:192`); loss weights come
from `loss_cfg.get(...)` inside the loss functions, where `loss_cfg =
self._config['training'][stage_key]['loss']` (`_interface_base.py:180`).

**Therefore passing `--matching_dim 128 --match_w4 0.3 --fine_warmup_epochs 5`
on the command line has NO effect** — to change them you must edit the variant's
`config.yaml`. The same-named argparse flags are consumed only in
`_legacy/legacy_A/` (`model.py:43-44`, `inference.py:72-113`), which is not
imported by the zoo. (The 128/64 launch scripts still pass these flags out of
habit; they are inert.)

**CLI flags that DO take effect:**

| Flag | Where consumed | Effect |
|---|---|---|
| `--is_stage_2` | `_interface_base.py:55`, `builder.py:217` | selects Stage-1 vs Stage-2 forward path / loss block / optimizer block / freeze logic. Single source of stage truth. |
| `--lr` | `optimizer()` `_interface_base.py:91,97` | overrides any `lr` in config. |
| `--maxEpochs` | `optimizer()` `_interface_base.py:111,115` | feeds the cosine scheduler `T_max`. |
| `--root_path` | `builder._resolve_args` | substituted into `${root_path}` for DINO paths. |
| `--sar_size` | `tm_dataloader.py:123,128,148,153` → `template_size` | drives the **dataloader** RGB-crop/template geometry (NOT the head). See §6 + §9. |
| `--shuffle_buffer` | `tm_dataloader.py:120,145` → `wds.shuffle()` | sample shuffle buffer (train only). |

> **Two `sar_size` channels.** The matching head reads `sar_size` from
> `config.yaml` (a hard-coded literal, e.g. `192` in B, `64` in C); the
> dataloader reads `--sar_size` from the CLI. They are wired independently and
> **can legitimately diverge** — but for a coherent run they MUST agree. See the
> pitfalls in §11.

### 2.4 imgNum / batchSize / gpu → iterations & epoch length

The **real per-epoch length is `with_epoch(per_worker_samples)`**, derived from
the shard manifest's `real_total`, NOT from `--imgNum`
(`_tm_webdataset.py:318-330`):

```
world_size       = dist.get_world_size()                       # from JF DDP
real_total       = sum(file_num) over manifest rows            # _tm_webdataset.py:282
per_rank_iter    = max(1, real_total // (batchSize * world_size))
per_worker_samples = max(1, (per_rank_iter * batchSize) // num_workers)
pipeline.with_epoch(per_worker_samples)
```

`--imgNum` is only JF's **progress-bar denominator**:
`training_iteration ≈ imgNum / (batchSize * gpu)`.

> **GOTCHA:** if `imgNum != real_total`, the bar shows 100%/Done at a fraction
> of the true epoch and looks like a stuck save. **Rule: `--imgNum` MUST equal
> the manifest `real_total` (= `sum(file_num)`).** Documented in
> `start_train_stage1_256_A.sh:9-14`.

### 2.5 DDP / world_size

`--dist True --gpu 4` → JF spawns 4 DDP ranks. **`--gpu N` is a COUNT, not a
device list** (JF convention); devices come from `CUDA_VISIBLE_DEVICES`.
`--port` is the DDP rendezvous port — vary it across concurrent runs.

Frozen DINO/SAR params require JF's remote-only patch
`find_unused_parameters=True` in `_meta_ops.py` (see
[[project-jackframework-ddp-patch]]; NOT in this repo's git). Without it, frozen
encoders cause an **empty `ProcessRaisedException` at all_reduce**.

`_balanced_split_by_node` (`_tm_webdataset.py:220-246`) truncates every rank to
`floor(num_shards/world_size)` shards to avoid the NCCL barrier timeout that
JF's epoch-end `dist.barrier()` would hit on uneven shard counts.

### 2.6 Checkpoint save (auto_save_num) & resume

JF saves every `--auto_save_num` framework epochs by calling
`save_model(epoch, model_list, opt_list)` (`_interface_base.py:466-473`), which
stores **only `model.trainable_tail.state_dict()`** per model as `model_i` plus
`opt_i` — the frozen DINO backbone is NOT duplicated.

Resume is JF-driven via a `checkpoint.list` index file in `--modelDir` (JF reads
the latest entry). The user hooks:
- `load_model` (`_interface_base.py:374-425`):
  `trainable_tail.load_state_dict(state, strict=False)`, with the
  `load_pretrained_sar` SAR-strip logic (§7).
- `load_opt` (`_interface_base.py:427-464`): always returns True; tolerates
  Stage1-SGD → Stage2-AdamW mismatch and missing `opt_0` by starting the
  optimizer fresh.

`--mode test` reuses the train-dataset code path and writes per-rank CSVs merged
by `post_process` → `finalize_active_test()` (`tm_dataloader.py`).

---

## 3. Model interface hooks & metric semantics

The shared `ComponentBasedInterface` (`_interface_base.py:33`) implements every
JF `ModelHandlerTemplate` hook. Variants A–F inherit it unchanged.

- **`inference(model, input_data, model_id) -> [output_dict]`**
  (`_interface_base.py:149`). `output = model(sar, rgb)`; **returns `[output]`**
  (list-wrapped; `ID_PRED=0`). In `--mode test` + Stage-2 it also attaches
  `_test_pred_xy`/`_test_score` via `decode_primary` (`argmax_type` from config).
- **`loss(output_data, label_data, model_id) -> [total, *components]`**
  (`_interface_base.py:168`). Unwraps `output_data[self.ID_PRED]`; selects
  `loss_cfg = self._config['training'][stage_key]['loss']` (`:180`); dispatches
  by `loss_cfg['type']`. JF backprops `loss[0]` (total); the rest are detached
  log scalars shown as `l1, l2, ...` in dict-insertion order.
- **`accuracy(output_data, label_data, model_id) -> [metrics]`**
  (`_interface_base.py:218`). Same list-unwrap.
- **`pretreatment(epoch, rank)`** (`_interface_base.py:351`): sets
  `self._current_epoch = int(epoch)` — drives all loss warmups.
- **`load_model` / `load_opt` / `save_model`**: §2.6 + §7.

### Metric semantics (the polarity trap)

- **Stage-2 `a0/a1/a2(/a3)` are Euclidean PIXEL ERRORS — lower is better.**
  `matching_loss.py:163,181` compute `((pred-gt)**2).sum(1).sqrt().mean()`.
  - StandardMatchingHead (A/B/D): `a0`=fine-head px err, `a1`=coarse fused
    (stride-4+stride-8), `a2`=stride-4 alone.
  - CascadeHead (C/E): `a0`=cascade stride-1, `a1`=stride-2, `a2`=coarse fused,
    `a3`=stride-4.
- **Stage-1 `a0` is a COSINE SIMILARITY — higher is better** (opposite polarity).
  Don't compare metrics across stages, and don't read a rising Stage-1 `a0` as a
  regression.

### Loss component layout (`l0/l1/l2...`)

`loss()` returns `[total, *components.values()]`; JF shows index 0 as `l0`:
- `cascade_loss` (C/E): `l0`=total, `l1`=coarse_s4, `l2`=coarse_s8, `l3`=cascade_s2,
  `l4`=cascade_s1. **The `l1..ln` are RAW per-level (detached, before weight &
  warmup)** — a flat `cascade_s1` while `epoch < warmup_epoch_1` is expected, not
  a dead head.
- `two_stage_matching_loss` (A/B/D): `l0`=total, `l1`=coarse, `l2`=fine.
- `alignment_loss` (Stage-1): `l0`=total, `l1`=patch_nce, `l2`=global_cos.
- `alignment_ip_loss` (Stage-1 learned-IP, F): `l0`=total, `l1`=patch_nce,
  `l2`=vic_var, `l3`=vic_cov, `l4`=erank_s, `l5`=erank_d (erank are diagnostics).

---

## 4. TM_Model_C cascade architecture (the paper's main method)

`TM_Model_C/__init__.py:13` is a 4-line `ComponentBasedInterface` subclass that
only sets `CONFIG_PATH`. All behaviour is in `TM_Model_C/config.yaml` +
`_components/`.

**Architecture** (`TM_Model_C/config.yaml:15-45`):
- `rgb_encoder`: `DinoV3Backbone` (`dinov3_vitl16`), `freeze: true`. Returns 4
  intermediate stride-16 feature maps; the cascade head uses only `rgb_feat[-1]`
  (`dino_backbone.py`).
- `sar_encoder`: `SarResNet16GN` (`in_chans:3, dino_dim:1024, layers:[3,4,6],
  channels:[64,128,256]`). GN ResNet → stride-16, returns `{sar16 (L2-normed,
  proj to 1024), raw16, C2(stride4), C3(stride8)}`, exposes `.channels`
  (`sar_encoder.py:73`).
- `matching`: `CascadeHeadJBU` with **hard-coded literals**
  `dino_dim:1024, matching_dim:256, sar_size:64, raw_stem_dim:16,
  use_stride_8_coarse:true, cascade_window_2:4, cascade_window_1:4`.

**`CascadeHeadJBU.forward`** (`fine_head_cascade.py:435-499`):
1. **Coarse stride-4**: `ProjMatchJBU(levels=2)` (×4 upsample) on both `sar16` and
   `rgb16` (each guided by its own raw image) → `MatchingHead` grouped-conv
   cross-correlation → `H_4`.
2. **Coarse stride-8** (optional, on): `ProjMatchJBU(levels=1)` (×2) → `H_8`.
3. **Fused coarse argmax** (detached, `no_grad`) via `softmax(fusion_logits)`
   blend → stride-4 cell anchor `uv_4` (`fine_head_cascade.py:460`).
4. **Cascade stride-2**: `LearnedUpsampler ×2` (bilinear+2conv) on the PROJECTED
   stride-4 feats (JBU intentionally skipped here, comment `:387`), re-L2-norm,
   windowed RGB crop around `2×uv_4` (±`cascade_window_2` cells) → `H_2`.
5. **Cascade stride-1**: `LearnedUpsampler ×2` + `RawImageStem` (3-conv stride-1
   CNN, 16ch) concatenated, `ProjMatchSAR` 1×1 proj + L2-norm, windowed crop
   around `2×uv_2` (±`cascade_window_1` px) → `H_1`.
6. Output dict: `heatmap_per_stride {4,8}`, `heatmap_cascade {2,1}`,
   `heatmap_fine=H_1`, `fine_anchor_pix`, `cascade_anchor_2_cells`,
   `fusion_weights`.

Building blocks live in `fine_head_cascade.py`: `LearnedUpsampler` (`:56`),
`RawImageStem` (`:81`), `_windowed_search_crop` (`:102`, per-sample slice+stack
to keep autograd through the search crop), `_argmax_xy` (`:151`). Cross-correlation
requires **search ≥ template** in HxW (`coarse_matching.py:129`), and the
windowed crops require **search ≥ template_size + 2·margin**
(`fine_head_cascade.py:130`).

**ComposedTM wiring** (`builder.py`): components are auto-wired; `sar_ch_c2/c3`
are injected only if the head's `__init__` signature advertises them —
`CascadeHeadJBU` does NOT (it eats `sar16`), so no injection
(`builder.py:256-268`). `trainable_tail` = `ModuleDict{sar_encoder, matching,
fusion}` in Stage-2 (`builder.py:131`), mirroring A's layout so ckpts transfer
via `strict=False`. Note C's state_dict is byte-compatible with E but **NOT**
with B (different matching sub-keys).

**Stage-2 forward freeze routing** (`builder.py:174-197`): RGB encoder always
under `no_grad` (frozen); SAR encoder under `no_grad` + `.detach()` only when
`freeze_sar_in_stage2` is true.

**As shipped, C = Stage-1-bootstrapped, frozen-encoder cascade** (the
F0-vulnerable design): `load_pretrained_sar:true` + `freeze_sar_encoder:true`.
See §10 for direct (random + trainable) training.

---

## 5. Losses & where each knob is set

Three loss families under `_components/`, registered by `@register` and
dispatched by `loss_cfg['type']` from each variant's config. **Every loss knob
is read from `config.yaml`, never from the CLI** (the same-named CLI flags are
dead — §2.3).

### 5.1 `cascade_loss` (C/E Stage-2) — `cascade_loss.py:60`

Multi-level Gaussian-target BCE; returns `(total, {coarse_s4, coarse_s8,
cascade_s2, cascade_s1})`.
- `match_w4`/`match_sigma4` (`cascade_loss.py:78-79`)
- `match_w8`/`match_sigma8` (`:87-88`)
- `match_w_2`/`match_sigma_2`/`warmup_epoch_2` (`:102-104`) — **config-only, no
  CLI flag at all**
- `match_w_1`/`match_sigma_1`/`warmup_epoch_1` (`:112-114`) — **config-only**
- `match_pos_weight` (coarse, `:76`), `match_fine_pos_weight` (cascade, `:96`)

C's config (`TM_Model_C/config.yaml:71-81`): `w4=1.0, w8=0.5, sigma4=1.0,
sigma8=0.5, w_2=0.5, sigma_2=1.0, warmup_epoch_2=2, w_1=1.0, sigma_1=1.0,
warmup_epoch_1=4`.

Coarse levels divide GT by `stride` in `gaussian_target` (`matching_loss.py:27`),
so **sigmas are in CELL units of that stride**: `sigma4=1.0`=4px, `sigma8=0.5`=4px,
`sigma_2=1.0`=2px, `sigma_1=1.0`=1px (physical-px ladder 4→4→2→1). Cascade levels
use `_windowed_bce` (`cascade_loss.py:39`): samples whose GT lands outside the
crop window are masked to zero contribution (covered by coarser levels).
`_linear_warmup(e,N)=min(1,e/N)`, and **=1.0 when N≤0** (no ramp = full weight,
NOT zero weight).

### 5.2 `two_stage_matching_loss` (A/B/D Stage-2) — `matching_loss.py:189`

Coarse multi-stride loss + single fine-head loss with a `fine_weight` linear
warmup. Returns `(total, {coarse, fine})`. Reads `match_w4/w8/sigma4/sigma8/
sigma_fine/w_fine/fine_warmup_epochs` from config (`matching_loss.py:214-233`);
`fine_weight = match_w_fine * min(1, epoch/fine_warmup_epochs)`
(`:221-226`). B's config: `w4=1.0, w8=0.5, sigma4=1.0, sigma8=0.5,
sigma_fine=1.0, w_fine=1.0, fine_warmup_epochs=3`.

### 5.3 Stage-1 `alignment_loss` / `alignment_ip_loss` — `alignment_loss.py`

- `alignment_loss` (`:223`): soft-window PatchNCE + `lambda_global`·global cosine.
  Params `tau, radius, lambda_global` from `stage_1.loss` block.
- `alignment_ip_loss` (`:156`, F only): PatchNCE on a learned projected pair +
  VICReg variance/covariance, with effective-rank diagnostics. Params
  `lambda_var, lambda_cov, erank_rows`.

Stage-1 params are passed as `**{k:v for k in loss_cfg if k!='type'}`
(`_interface_base.py:196`), so **a stray/misspelled key becomes an unexpected
kwarg and raises `TypeError`** — no silent skip here. There is also a hard guard
(`_interface_base.py:202`) requiring `architecture.alignment_ip` to be present
when `loss.type == alignment_ip_loss`.

---

## 6. Dataloader / WebDataset

Zoo (`Dataloaders/__init__.py:6-13`): the single key `'tmdataset'` →
`TMDataloader` (`tm_dataloader.py:84`), a JF `DataHandlerTemplate`. Backend
selected by `__use_webdataset` (`tm_dataloader.py:156`): **iterable** iff
`--dataloaderType == 'iterable'` (the live production path — all 256/192 runs use
it); otherwise legacy map-style `TM_Dataset` (`_tm_datasets.py:16`).
**`--dataloaderType` defaults to `mapstyle`, so you MUST pass `iterable`.**

### 6.1 Two CSV formats (do not confuse them)

1. **Shard manifest CSV** (iterable backend) — header `path,file_num`; one row
   per `.tar` shard. Parsed by `_read_path_csv` (`_tm_webdataset.py:199-217`);
   produced by `make_opt256_sar192_shards.py:285-290`. **`real_total =
   sum(file_num)`** (`_tm_webdataset.py:282`) is the authoritative on-disk count.
2. **Path-list CSV** (legacy map-style) — `SAR_Image,RGB_Image,Label` where
   `Label` is a JSON `[x,y]` string (`_tm_datasets.py:69`). The checked-in
   `Datasets/dataset_example_training_list.csv` uses stale stereo columns
   (`left_img,right_img,gt_disp`) — **NOT** the TM schema; ignore it as a
   reference.

### 6.2 Pipeline (`build_tm_webdataset`, `_tm_webdataset.py:249-337`)

```
manifest CSV -> shard list -> wds.WebDataset(shardshuffle)
  -> [train only] .shuffle(shuffle_buffer)
  -> .with_epoch(per_worker_samples)            # §2.4 epoch-length control
  -> .to_tuple('sar.png','rgb.png','label.cls','__key__')
  -> .map(TMSampleProcessor)                    # decode, aug, sanitize, normalize
  -> (sar[3,H,W], rgb[1or3,H,W], label_out, sar_name, rgb_name)
```

`get_train_dataset(path, is_training)` (`tm_dataloader.py:112`) reads
`args.trainListPath` / `args.valListPath` **directly — the `path` argument from
JF is ignored** (`tm_dataloader.py:115`). If `trainListPath` ends in `.csv` it is
a manifest; anything else triggers a slow per-shard disk probe.

### 6.3 `TMSampleProcessor` (`_tm_webdataset.py:90-160`)

- `h=w=64` heatmap grid is **hard-coded** in `__init__`; `template_size` comes
  from `--sar_size`.
- `__call__`: decodes PNGs; optional aug (Stage-1 train only,
  `_tm_webdataset.py:142` — Stage-2 disables geometric aug deliberately);
  `_sanitize_label` clips `(x,y)` to `[0, img-template]`; Stage-1 pre_train crops
  RGB to the template; normalizes; SAR always duplicated to 3 channels.
- **Stage-2 returns a raw float `(x,y)` label** (`:156`) and never touches the
  h/w grid; **Stage-1 returns a softmax heatmap** (`:158`).

### 6.4 Label convention

`<key>.label.cls` = ASCII `"<x> <y>"` (two ints, comma normalized to space at
`_tm_webdataset.py:45-48`). Semantics: `x`=column, `y`=row, **TOP-LEFT corner**
of the SAR template inside the RGB reference = the GT displacement. Range
`[0, rgb_size - sar_size]`. At opt256/sar192 that is `[0, 64]` (asserted in the
packer `make_opt256_sar192_shards.py:230`), which is why the 64-grid stays
valid for Stage-2. NOT a center coord, NOT normalized.

### 6.5 `sar_size` handling (dual use — see §2.3 / §11)

The model head reads `architecture.matching.sar_size` from `config.yaml`; the
dataloader reads `--sar_size` from the CLI and threads it as `template_size`.
**Both must agree.** Eval scripts `sed`-patch the yaml `sar_size` around each run
with a restore trap, so **concurrent runs against the same config.yaml must be
serialized**.

---

## 7. The A–F variant matrix & 2×2 causal arms

Two orthogonal axes define the causal study:
- **Axis-1 (init):** `training.stage_1.load_pretrained_sar` — Stage-1-warm-started
  vs random SAR init. Consumed in `load_model` (`_interface_base.py:383-418`):
  if false AND the ckpt is a pure Stage-1 ckpt (no `matching.*` keys, `:403`),
  all `sar_encoder.*` keys are stripped before `load_state_dict(strict=False)`.
  **Stage-2 resume/test ckpts keep their trained SAR even when the flag is
  false** (the `:391-409` bug fix).
- **Axis-2 (freeze):** `training.stage_2.freeze_sar_encoder` — SAR trainable vs
  frozen in Stage-2. Consumed in TWO places: `get_model()`
  (`_interface_base.py:65-74`, sets `requires_grad=False`) and `builder.py:274-283`
  (→ `ComposedTM(freeze_sar_in_stage2=...)`, `no_grad`+detach in forward).

| Variant | Head | matching_dim | sar_size (head) | geometry | load_pretrained_sar | freeze_sar_encoder | Stage-2 loss | argmax |
|---|---|---|---|---|---|---|---|---|
| **A** | StandardMatchingHeadJBU | 256 | 192 | 256/192 | true | false | two_stage_matching_loss | hard |
| **B** | StandardMatchingHeadJBU | 256 | 192 | 256/192 | **false** (the ablation) | false | two_stage_matching_loss | hard |
| **C** | **CascadeHeadJBU** | 256 | **64** | **128/64** | true | true | **cascade_loss** | hard |
| **D** | StandardMatchingHeadJBU | 256 | 64 | 128/64 | true | true | two_stage_matching_loss | soft_subpixel |
| **E** | CascadeHeadJBU | 256 | 64 | 128/64 | true | true | cascade_loss | soft_subpixel |
| **F** | StandardMatchingHeadJBU (+ AlignmentIPHead, Stage-1 only) | 256 | 192 | 256/192 | true | true | two_stage_matching_loss | hard |

(Flags cited from each `TM_Model_*/config.yaml`.)

**The 2×2 "frozen-arms" experiment** (init × freeze):

| Arm | Name | modelName | init | freeze | Checkpoint dir | Launcher |
|---|---|---|---|---|---|---|
| S1/Tr | A-ft | TM_Model_A | Stage-1 | trainable | `Checkpoint_A_ft_256/` | `run_a_ft_vs_b.sh` |
| R/Tr | B | TM_Model_B | random | trainable | `Checkpoint_B_256/` | `run_a_ft_vs_b.sh` |
| S1/Fr | S1F | TM_Model_F | Stage-1 | frozen | `Checkpoint_F_s2_frozen_256/` | `run_frozen_arms_F.sh` |
| R/Fr | RF | TM_Model_F | random | frozen | `Checkpoint_RF_s2_frozen_256/` | `run_frozen_arms_F.sh` |

> **S1F vs RF use the SAME config (TM_Model_F).** F's config has
> `load_pretrained_sar:true` unconditionally (for Stage-1 resume support). The
> init axis is enforced by the **launcher**: S1F seeds its `modelDir` with the
> stripped Stage-1 ckpt; RF starts with an empty `modelDir`. Reading the configs
> alone would miss the RF arm.

**C/E are still 128/64** not by design but because their `config.yaml` files
(`sar_size:64`) and full-manifest launchers predate the 4-Jun 256/192 pivot and
were never migrated. A/B/F were edited to `sar_size:192 + fine_margin:24`; C/D/E
were not. Any C-vs-A/B comparison is therefore *also* a geometry comparison
unless C is retrained at 256/192 (see §10).

---

## 8. Launcher scripts

Live (256/192) launchers in `Scripts/`:
- `run_a_ft_vs_b.sh` — the unfrozen 2×2 row. `launch()` (`:30-43`) is the
  canonical 256/192 Stage-2 recipe (`--batchSize 32 --gpu 2 --port ... --dist
  True --maxEpochs 5 --auto_save_num 1 --dataloaderNum 8 --dataloaderType
  iterable --lr 1e-3 --is_stage_2 True --sar_size 192 ...`), subset40 manifest
  (`real_total=1,344,000`), `nohup`. A-ft bootstrapped via `strip_stage1_opt.py`.
- `run_frozen_arms_F.sh` — the frozen 2×2 row (S1F + RF), both TM_Model_F.
- `start_train_stage1_256_{A,F}.sh` — Stage-1 alignment @256/192.

Legacy / superseded (128/64) in `Scripts/` (and `Scripts/legacy_128_64/`):
- `start_train_stage2_{A,B,C,D,E}_wds.sh` — full manifests
  `shard_training_path.csv` (`imgNum=65275274`) + `shard_testing_path.csv`
  (`valImgNum=8145176`), `maxEpochs 10`, NO `--sar_size`. `start_train_stage2_C_wds.sh`
  auto-bootstraps from `Checkpoint_v3/` Stage-1 ckpt via `strip_stage1_opt.py`.
- `start_train_stage2_E_wds.sh` does **no training** — it copies C's ckpt
  (byte-identical state_dict) and runs val-only with `soft_subpixel`. Same
  relationship as A↔D.

`strip_stage1_opt.py` (`Source/Tools/`) drops optimizer state + resets `epoch`
so a Stage-1 ckpt is treated as a fresh Stage-2 start (Stage-1 SGD ≠ Stage-2
AdamW). Referenced by all bootstrap paths.

---

## 9. Worked geometry check for C @ 256/192

Before retraining C at the 256/192 geometry, verify the cascade crop asserts
(`coarse_matching.py:129`, `fine_head_cascade.py:130`). With `opt=256`,
`sar=192`:

| Level | SAR template | RGB search | constraint |
|---|---|---|---|
| coarse s4 | sar_4 = 12·4 = 48 | rgb_4 = 16·4 = 64 | 64 ≥ 48 ✔ |
| coarse s8 | sar_8 = 12·2 = 24 | rgb_8 = 16·2 = 32 | 32 ≥ 24 ✔ |
| cascade s2 | sar_2 = 48·2 = 96 (template) | crop = `sar_size//2 + 2·win2` = 96+8 = 104; rgb_2_full = 64·2 = 128 | 104 ≤ 128 and 104 ≥ 96 ✔ |
| cascade s1 | sar_1 = 96·2 = 192 (template) | crop = `sar_size + 2·win1` = 192+8 = 200; rgb_1_full = 128·2 = 256 | 200 ≤ 256 and 200 ≥ 192 ✔ |

So C at 256/192 is geometrically valid with `cascade_window_2 = cascade_window_1
= 4`. **Anyone bumping `sar_size` must re-run this check** — a too-large window
or template trips a runtime `ValueError`, not a silent failure.

---

## 10. Training C "direct" (random SAR, trainable, no Stage-1)

To run C as a *direct* (joint-from-scratch) model rather than the
Stage-1-bootstrapped frozen default, you must flip BOTH flags in its config
(see the launch recipe section of the structured output for the exact edits):
- `training.stage_2.freeze_sar_encoder: false` (SAR trains jointly)
- `training.stage_1.load_pretrained_sar: false` (random SAR init; also prevents
  bootstrapping from any Stage-1 ckpt left in `modelDir`)
- `architecture.matching.sar_size: 192` + pass `--sar_size 192` (both channels)

Flipping `freeze_sar_encoder` alone is NOT enough — if a Stage-1 ckpt sits in
`modelDir` and `load_pretrained_sar` stays true, C still bootstraps from it. A
mutated-C run also breaks E's ckpt-reuse contract; prefer a **new variant dir**
to keep C's published config intact.

---

## 11. Known pitfalls (in rough order of how often they bite)

1. **config-overrides-CLI for arch dims & loss weights.** `--matching_dim`,
   `--fine_dim`, `--fine_margin`, `--match_w4/w8/sigma4/sigma8/w_fine/sigma_fine/
   pos_weight`, `--fine_warmup_epochs`, `--match_primary_stride` are **DEAD** on
   the active YAML path (only `${root_path}` is substituted). Edit `config.yaml`,
   not the CLI. `match_w_2/sigma_2/warmup_epoch_2` and `match_w_1/sigma_1/
   warmup_epoch_1` have **no CLI flag at all**.

2. **`sar_size` is configured through two channels.** The head reads it from
   `config.yaml` (a literal, NOT a `${sar_size}` placeholder — the CLI can't fix
   it); the dataloader reads `--sar_size`. To train at 192 you must change BOTH:
   the literal in `config.yaml` AND pass `--sar_size 192`. Changing only one
   desyncs the SAR template the model expects vs what the loader delivers.

3. **`imgNum` is NOT the epoch length** — only JF's progress-bar denominator.
   Real epoch = `with_epoch(per_worker_samples)` from manifest `real_total`. If
   `imgNum != sum(file_num)` the bar lies (100%/Done at a fraction of the epoch,
   looks like a stuck save). Rule: `--imgNum == real_total`.

4. **`with_epoch` per-rank trap (and its fix).** `with_epoch` caps EACH worker on
   EACH rank, so `real_total` is divided by `world_size` FIRST
   (`_tm_webdataset.py:328`). This is the deliberate fix for JF's
   `with_epoch`-double-counting under DDP. Forgetting the `//world_size` yields
   `world_size`× oversized epochs. The legacy map-style backend has no
   `with_epoch` and relies on natural `StopIteration`.

5. **DDP + frozen encoders requires `find_unused_parameters=True`** — a patch in
   the remote JF's `_meta_ops.py`, NOT in this repo's git. Without it, frozen
   encoders crash with an empty `ProcessRaisedException` at all_reduce. Always
   reproduce on `--dist False --gpu 1` first to get a real traceback.

6. **JF silent name dispatch.** JF dispatches user hooks by string name via
   `getattr` with no validation: `post_process` not `postprocess`; `--dataset`
   must be `'tmdataset'`; `--modelName` must be a zoo key. The zoo asserts on
   unknown keys (clear error), but a **misnamed hook method fails silently**.

7. **`--dataloaderType` defaults to `mapstyle`.** You MUST pass `iterable` to get
   the WebDataset path; otherwise you silently fall into the legacy map-style
   backend (which expects a different CSV schema).

8. **inference() list-wrap.** `inference()` returns `[output]`; `loss`,
   `accuracy`, `save_result` all receive that list and must unwrap via
   `output_data[self.ID_PRED]` (=0).

9. **Stage-1 vs Stage-2 metric polarity is OPPOSITE** (cosine higher-better vs
   pixel-error lower-better). Cascade loss `l1..ln` are RAW per-level (pre-warmup)
   — a flat finer level during its warmup window is expected.

10. **Stage-1 loss kwargs are strict.** A misspelled key in the `stage_1.loss`
    block raises `TypeError` at call time (passed as `**kwargs`), unlike the
    silently-ignored CLI flags.

11. **`--gpu N` is a COUNT, not a device list.** Devices come from
    `CUDA_VISIBLE_DEVICES`. `--port` must be unique across concurrent runs.

12. **Manifests are remote-only.** `Datasets/` locally holds only the stale
    example CSV. `shard_training_path.csv` / `shard_train_256_subset40.csv` /
    `TM_shards_opt256_sar192_*/shard_path.csv` exist only on the remote; all
    `imgNum`/`real_total` values come from the scripts, not verifiable locally.
</content>
</invoke>
