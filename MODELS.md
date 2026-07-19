# PaperModels — TGRS 2026 checkpoint release / 论文模型清单

Checkpoints for **DOSR-Net / Template-Matching-DINOv3** (SAR ↔ optical template
matching). Every checkpoint maps to the paper claim or table it supports.
每个 checkpoint 对应它支撑的论文结论/表格。

**Geometry / 几何**: optical reference **256×256**, SAR template **192×192**
(ME-SSM protocol). All Stage-2 models use `matching_dim = 256`.

**How to use / 如何使用**: unzip **into the project root** so the
`Checkpoint_*/` directories sit next to `Source/`, `Scripts/`, `Weights/`. The
training/eval launchers and configs already reference these exact directory
names. 解压到项目根目录即可，启动脚本/配置已按这些目录名引用。

```
Template-Matching-DINOv3/
├── Source/  Scripts/  Weights/ ...
├── Checkpoint_C_phase2_8gpu/        # ← from PaperModels_*_m5.zip
├── Checkpoint_A_ft_256/             # ← from PaperModels_*_m2.zip
├── Checkpoint_B_256/                # ...
└── ...
```

Run any variant from the project root, e.g. the proposed cascade on the 2k eval set:
```bash
bash Scripts/run_eval_C_2k.sh Checkpoint_C_phase2_8gpu/model_epoch_2.pth c_ep2 0
```

---

## The two download archives / 两个下载包

Models live on two training servers; each archive preserves the
`Checkpoint_*/model_epoch_*.pth` layout, so you can unzip both into the same
project root. 模型分布在两台训练机，两个包解压到同一项目根即可合并。

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

## DINOv3 optical backbone (download separately) / DINOv3 光学骨干（单独下载）

The frozen optical encoder is the **public** DINOv3 ViT-L/16, satellite-pretrained
(`sat493m`) weights — **not** trained by us, so it is not in the archives above.
冻结的光学骨干是公开的 DINOv3 ViT-L/16 卫星预训练权重，非本文训练，不在上面两个包内。

- file: `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` (1.21 GB)
- place at: `Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`
- source: Meta DINOv3 release (satellite `sat493m` variant). The path is set in
  `Source/UserModelImplementation/user_define.py`.

---

*Generated for the TGRS 2026 release. Key set = the 6 models backing Tables I–VII
plus the proposed cascade; per-epoch/eval-copy checkpoints are omitted.*
