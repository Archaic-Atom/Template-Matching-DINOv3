# PaperModels — TGRS 2026 checkpoint release / 论文模型清单

Checkpoints for **DOSR-Net / Template-Matching-DINOv3** (SAR ↔ optical template
matching). Every checkpoint maps to the paper claim or table it supports.
每个 checkpoint 对应它支撑的论文结论/表格。

**Geometry / 几何**: optical reference **256×256**, SAR template **192×192**
(ME-SSM protocol). All Stage-2 models use `matching_dim = 256`.

**How to use / 如何使用**: the netdisk share is one `Weights` folder holding both
the backbone and every checkpoint. Keep the DINOv3 backbone inside `Weights/`, and
move the `Checkpoint_*/` directories to the **project root** next to `Source/`,
`Scripts/`. 网盘里是一个 `Weights` 文件夹，含骨干与全部 checkpoint：DINOv3 骨干留在
`Weights/` 内，各 `Checkpoint_*/` 目录移到项目根目录（与 `Source/` 同级）。

**Directory names are load-bearing.** Only the cascade is passed to its eval launcher as
a path argument; every other checkpoint is opened by its hard-coded directory name:
`Checkpoint_A_stage1_256/` and `Checkpoint_F_stage1_256/` are the Stage-1 warm-start
sources in `run_a_ft_vs_b.sh` / `run_frozen_arms_F.sh`, and
`Checkpoint_{F,RF}_s2_frozen_256/` plus `Checkpoint_B_256/` (pinned at epoch 4) are read
by name in `run_eval_frozen_full.sh`. Keep the names as shipped.
**目录名是"承重"的**：只有级联是按路径参数传给评测脚本，其余权重都由脚本按写死的目录名打开
（两个 Stage-1 目录是热启动来源；两个 frozen 臂与 `Checkpoint_B_256/` 由
`run_eval_frozen_full.sh` 按名读取，B 固定 epoch 4）。请保持原目录名。

See the README's *"Which checkpoint goes with which launcher"* table for the full
checkpoint ↔ script mapping. 完整的权重↔脚本对应表见 README。

> ⚠️ The launchers hard-code our servers' `PROJ` / `PY` / `/data4|/data5` manifest paths
> (29 of 41 scripts) — see the README's *"Paths you must edit"* section before running.
> 脚本写死了服务器的 `PROJ`/`PY`/数据清单路径（41 个里有 29 个），运行前请先看 README 的
> “需要修改的路径”一节。

```
Template-Matching-DINOv3/
├── Source/  Scripts/ ...
├── Weights/
│   └── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth   # ← backbone stays here
├── Checkpoint_C_phase2_8gpu/        # ← moved out of the downloaded Weights/
├── Checkpoint_A_ft_256/
├── Checkpoint_B_256/
└── ...
```

Run any variant from the project root, e.g. the proposed cascade on the 2k eval set:
```bash
bash Scripts/run_eval_C_2k.sh Checkpoint_C_phase2_8gpu/model_epoch_2.pth c_ep2 0
```

---

## Download / 下载

All checkpoints (plus the DINOv3 backbone) are in one Baidu netdisk share — the
`Weights` folder holds everything. 全部 checkpoint（连同 DINOv3 骨干）都在一个百度网盘
分享里，`Weights` 文件夹即为全部内容。

> **Baidu netdisk / 百度网盘:**
> <https://pan.baidu.com/s/1lhXTqkXKwzg5H9PVtt01EQ?pwd=ug7r> — extraction code: `ug7r`

The same key-set is also staged on the two training servers as two archives (each
preserves the `Checkpoint_*/model_epoch_*.pth` layout). 关键集在两台训练机上也各打了一个包：

| archive | size | sha256 (first 16) | contents |
|---|---|---|---|
| `PaperModels_TGRS2026_m5.zip` | 272 MB | `be9d348822477b8f` | the **proposed cascade** (headline model) |
| `PaperModels_TGRS2026_m2.zip` | 698 MB | `5df5a6712a3311f5` | the **6 ablation / Stage-1 models** (Tables II–VII) |

Full sha256:
- m5: `be9d348822477b8fd8f72b7f58a8cc2a10d200df5a6f5e729b68811b11be319b`
- m2: `5df5a6712a3311f598775e3952ff926cd5c69a4961f58796dcb48e08aceac6e7`

---

## Model table / 模型清单

| checkpoint (place at project root) | model / `--modelName` | role — paper claim/table | key result | size |
|---|---|---|---|---|
| `Checkpoint_C_phase2_8gpu/model_epoch_2.pth` | **TM_Model_C** (CascadeHeadJBU) | **★ Proposed method** — direct-trained coarse-to-fine cascade. Table I headline / SOTA claim. | in-domain full-set **CMR(1) 93.30 / CMR(3) 95.83**; beats ME-SSM (62.14/89.19) on both thresholds | 237 MB |
| `Checkpoint_A_ft_256/model_epoch_4.pth` | TM_Model_A (Standard head) | **S1/Tr** — Stage-1 aligned init, encoder trainable. 2×2 pre-alignment test (Table II). | CMR(1) 68.30 / CMR(3) 86.08 (20k) — ≈ R/Tr → pre-alignment redundant | 178 MB |
| `Checkpoint_B_256/model_epoch_4.pth` | TM_Model_B (Standard head) | **R/Tr** — random init, encoder trainable. Table II control **and the subject model for all mechanism probes** (Sec. IV). | CMR(1) 68.28 / CMR(3) 86.35 (20k) | 178 MB |
| `Checkpoint_F_s2_frozen_256/model_epoch_4.pth` | TM_Model_F (frozen encoder) | **S1/Fr** — Stage-1 aligned init, encoder **frozen**. Frozen row of Table II. | CMR(1) 68.63 / CMR(3) 86.30 (20k) | 102 MB |
| `Checkpoint_RF_s2_frozen_256/model_epoch_4.pth` | TM_Model_F (random, frozen) | **R/Fr** — random init, encoder **frozen**. Frozen-row control. | CMR(1) 66.94 / CMR(3) 85.17 (20k) | 102 MB |
| `Checkpoint_A_stage1_256/model_epoch_5.pth` | Stage-1 (raw-cosine PatchNCE) | **Aligned Stage-1 encoder** — the warm-start source for A-ft / S1/Fr. Effective-rank analysis (Sec. IV, Fig. erank/overwrite). | alignable effective rank ≈ 22–23 | 76 MB |
| `Checkpoint_F_stage1_256/model_epoch_9.pth` | Stage-1 (learned-IP + VICReg) | **Learned-inner-product Stage-1** — best-case alignment probe; warm-start source for S1/Fr. Capacity-ceiling evidence. | erank → 23, alignment a0 → 0.80 (yet still redundant in Stage-2) | 80 MB |

Convenience copies also inside the archives:
- `Checkpoint_C_phase2_8gpu/model_phase1_init.pth` (79 MB) — the Phase-1 warm-start weights the cascade Phase-2 resumes from.
- `Checkpoint_A_ft_256/model_stage1_init.pth` / `Checkpoint_F_s2_frozen_256/model_stage1_init.pth` — the Stage-1 weights those arms were initialized/frozen from.

---

## DINOv3 optical backbone / DINOv3 光学骨干

The frozen optical encoder is the **public** DINOv3 ViT-L/16, satellite-pretrained
(`sat493m`) weights — **not** trained by us, but included in the netdisk share above
for convenience (it is *not* inside the two server archives).
冻结的光学骨干是公开的 DINOv3 ViT-L/16 卫星预训练权重，非本文训练；为方便起见已放在上面的
网盘分享里（但**不在**两台服务器的压缩包内）。

- file: `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` (1.21 GB)
- place at: `Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`
- source: Meta DINOv3 release (satellite `sat493m` variant); also mirrored in the
  netdisk share. The path is set in `Source/UserModelImplementation/user_define.py`.

---

*Generated for the TGRS 2026 release. Key set = the 6 models backing Tables I–VII
plus the proposed cascade; per-epoch/eval-copy checkpoints are omitted.*
