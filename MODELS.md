# Pretrained models

Weights for DOSR-Net / Template-Matching-DINOv3 use 256-pixel optical references
and 192-pixel SAR templates. Matching features have 256 channels.

## Download

The address and extraction code are unchanged:

> <https://pan.baidu.com/s/1lhXTqkXKwzg5H9PVtt01EQ?pwd=ug7r> — extraction code: `ug7r`

The shared `Weights` folder contains the original 11 weight files and the
66-file `Revision_20261107` addition. See [MODELS_REVISION.md](MODELS_REVISION.md)
and the [per-file revision index](docs/checkpoints_revision.csv) for the added
four-arm, early-training, budget, partial-freezing, few-pair and upsampling runs.

## Original checkpoints

References use the revised manuscript numbering. The submitted single-run
comparison and the repeated shared-initialization comparison are reported
separately.

| Checkpoint | Model | Use in the revised manuscript |
|---|---|---|
| `Checkpoint_C_phase2_8gpu/model_epoch_2.pth` | `TM_Model_C` | Headline direct-trained cascade; Tables I/II/VIII/IX/X/XII/XIII |
| `Checkpoint_A_ft_256/model_epoch_4.pth` | `TM_Model_A` | Submitted pre-aligned trainable arm; Table IV, Figure 8 |
| `Checkpoint_B_256/model_epoch_4.pth` | `TM_Model_B` | Submitted random trainable arm; Tables I/IV/XII and representation probes |
| `Checkpoint_F_s2_frozen_256/model_epoch_4.pth` | `TM_Model_F` | Submitted pre-aligned frozen arm; Table IV, Figure 12 |
| `Checkpoint_RF_s2_frozen_256/model_epoch_4.pth` | `TM_Model_F` | Submitted random frozen arm; Table IV, Figure 12 |
| `Checkpoint_A_stage1_256/model_epoch_5.pth` | asymmetric Stage 1 | Initialization source for A-ft; Figure 8 |
| `Checkpoint_F_stage1_256/model_epoch_9.pth` | symmetric learned-IP Stage 1 | Initialization and recovery diagnostics; Figure 7 |
| `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | frozen DINOv3 ViT-L/16 | Shared satellite-pretrained optical provider |

The original set also includes the Phase-1 warm start in
`Checkpoint_C_phase2_8gpu/model_phase1_init.pth` and the initialization files
inside `Checkpoint_A_ft_256` and `Checkpoint_F_s2_frozen_256`.

The submitted trainable and frozen S1 arms used different Stage-1 recipes.
The revised Table V uses a shared Stage-1 source for both pre-aligned arms
and reports five seeds. Seed 2024 uses the original trained weights.

## Placement

Keep the DINOv3 backbone in `Weights/`. Move original `Checkpoint_*/` folders
to the project root alongside `Source/` and `Scripts/`. For the repeated
four-arm launcher, move its revision `Checkpoint_F_*` and `Checkpoint_G_*`
folders to the project root too. Other helpers accept checkpoint paths.

```text
Template-Matching-DINOv3/
├── Source/
├── Scripts/
├── Weights/
│   ├── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
│   └── Revision_20261107/
├── Checkpoint_C_phase2_8gpu/
├── Checkpoint_A_ft_256/
├── Checkpoint_B_256/
└── ...
```

Evaluate the headline cascade with your evaluation manifest:

```bash
export TM_PY=/path/to/python
export TM_EVAL_CSV=/path/to/image_test_2k_256.csv
bash Scripts/reproduce/eval_cascade.sh Checkpoint_C_phase2_8gpu/model_epoch_2.pth 0
```

The default helper uses a 2k sample list. The Table I headline uses all
2,236,840 scored placements. Select the same manifest when comparing numbers.
For revised Table V, use the common 20k list and `eval_2x2.sh`.
The DINOv3 backbone is a separate dependency; learned-component checkpoints
exclude its frozen parameters. The `CW` configuration expects the web-pretrained
LVD-1689M provider at the path in its YAML file.
