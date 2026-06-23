# Mechanism probes / 机制探针

The seven probes that produce every mechanistic number in the paper's
analysis section (Sec. IV). 论文分析章全部机制数字的来源脚本。

**Invocation convention / 调用约定**: all probes resolve paths relative to
the PROJECT ROOT (`sys.path.insert(0, 'Source')`, checkpoint paths like
`Checkpoint_B_256/model_epoch_4.pth`). Always run FROM the project root:

```bash
cd <project-root>
CUDA_VISIBLE_DEVICES=<gpu> python -u Source/Tools/probes/<probe>.py
```

| probe | measures | paper evidence |
|---|---|---|
| `probe_b_similarity.py` | raw stride-16 vs projected stride-4 similarity + sign check | F1 (raw gap), F2 core |
| `probe_b_controls.py` | adversarial control battery C0–C7 incl. one-/two-sided orthogonal rotation | F2 proof (constructed similarity) |
| `probe_b_content.py` | CCA shared subspace + structure correlation (NOTE: superseded row-split; de-leaked version in keystone) | F3 (legacy) |
| `probe_b_keystone.py` | image-holdout CCA + subspace-restricted matching | F3 (de-leaked), E2 (~64-d) |
| `probe_b_stage1sim.py` | raw cosine vs CCA learned-IP stride-16 localization | learnable-alignment evidence |
| `probe_b_collapse.py` | mini-train W_s/W_d on frozen features, effective-rank trajectory | capacity probe (~25) |
| `probe_f0_mechanism.py` | S1 vs A-ft vs B encoder alignment + CKA | F0 overwrite mechanism |

Default checkpoint: `Checkpoint_B_256/model_epoch_4.pth` (the analyzed
specimen) unless the docstring says otherwise. Numbers ledger with status
tags: `PAPER_DATA_AND_ANALYSIS.md` §10.6 (one level above the project).
