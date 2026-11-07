# Revision checkpoints

Download the weights from the same share as the original release.

> <https://pan.baidu.com/s/1lhXTqkXKwzg5H9PVtt01EQ?pwd=ug7r> — extraction code: `ug7r`

The `Weights` folder keeps the original 11 files and adds 66 checkpoint files
under `Revision_20261107/`. The [file index](docs/checkpoints_revision.csv)
records every added filename, SHA-256 checksum, seed, epoch and paper reference.
Each downloaded model folder also includes its README. Table and figure numbers
below refer to the revised TGRS manuscript.

| Experiment | Checkpoints | Paper reference |
|---|---|---|
| Shared-initialization four-arm test | Seeds 2025–2028; epoch 4 for S1/Tr, R/Tr, S1/Fr and R/Fr | Table IV; Figure 9 |
| Early training | Epochs 0–4 for S1/Tr and R/Tr, seeds 2025–2028 | Figure 13(a,b) |
| Reduced matching pools | Epoch 4 for both arms at 10 and 20 shards, seed 2028 | Figure 13(c,d) |
| Partial encoder freezing | Epoch 4, random and pre-aligned arms | Table V |
| 100/500/1000 training pairs | Matching epoch 4 and Stage-1 epoch 9 for each set | Table VI |
| Coarse JBU versus bilinear | Epoch 4 for both cascades | Tables X and XIII |
| Shared SAR initialization | `shared_stage1/model_stage1_init.pth` | Table IV setup |

Seed 2024 uses the original trained weights. This folder adds the later runs
for seeds 2025–2028. The submitted single-run results remain separately listed
in [MODELS.md](MODELS.md).

The headline cascade is still
`Checkpoint_C_phase2_8gpu/model_epoch_2.pth`. It supports Tables I, II, VII,
VIII, IX, XI and XII. The new matched JBU cascade is a different run used in
Tables X/XIII. Table XIII's single-shot profile uses the added seed-2028 R/Tr
endpoint. Both coarse-upsampling arms retain the learned fine upsamplers.

## Placement and evaluation

Keep the frozen DINOv3 SAT-493M backbone in `Weights/`. The filenames use
zero-based epochs: matching epoch 4 is the fifth epoch; Stage-1 epoch 9 is
the tenth epoch.

For the four-arm launcher, move the downloaded `Checkpoint_F_*` and
`Checkpoint_G_*` folders from `Revision_20261107` to the project root. Keep
their names, then supply the common 20k evaluation list:

```bash
export TM_EVAL_CSV=/path/to/eval20k.csv
export TM_PY=/path/to/python
bash Scripts/reproduce/eval_2x2.sh 2025 auto all --dry-run
bash Scripts/reproduce/eval_2x2.sh 2025 auto all
```

For a bilinear cascade, the shared helper accepts a checkpoint path directly:

```bash
source Scripts/common.sh
tm_eval TM_Model_CB Weights/Revision_20261107/r1_7_cascade_bilinear/model_epoch_4.pth revision_bilinear 0 /path/to/eval20k.csv 20000
```

The package's `REPRODUCTION_CONFIGS` preserves the recorded command and
model-contract snapshots. Update server and data paths before using them.
The held-out 20k test list is unchanged. The 100/500/1000-pair lists are nested
training subsets, with matching Stage-1 and Stage-2 pair budgets.
