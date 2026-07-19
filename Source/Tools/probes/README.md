# Mechanism probes / 机制探针

The probes that produce every mechanistic number in the paper's analysis
section (Sec. IV) — the *constructed-similarity* battery, the
*granularity–dimensionality* recovery sweeps, and the *effective-rank /
overwrite* measurements. 论文分析章全部机制数字的来源脚本。

**Invocation convention / 调用约定**: all probes resolve paths relative to
the PROJECT ROOT (`sys.path.insert(0, 'Source')`, checkpoint paths like
`Checkpoint_B_256/model_epoch_4.pth`). Always run FROM the project root:

```bash
cd <project-root>
CUDA_VISIBLE_DEVICES=<gpu> python -u Source/Tools/probes/<probe>.py
```

## Constructed similarity — F1/F2/F3/E2 (similarity is built, not pre-existing)
| probe | measures | paper evidence |
|---|---|---|
| `probe_b_similarity.py` | raw stride-16 vs projected stride-4 similarity + sign check | F1 (raw gap), F2 core |
| `probe_b_controls.py` | adversarial control battery C0–C7 incl. one-/two-sided orthogonal rotation | F2 proof (constructed similarity), Fig. controls |
| `probe_b_content.py` | CCA shared subspace + structure correlation (NOTE: superseded row-split; de-leaked version in keystone) | F3 (legacy) |
| `probe_b_keystone.py` | image-holdout CCA + subspace-restricted matching | F3 (de-leaked), E2 (~64-d) |
| `probe_b_stage1sim.py` | raw cosine vs CCA learned-IP stride-16 localization | learnable-alignment evidence |
| `probe_b_collapse.py` | mini-train W_s/W_d on frozen features, effective-rank trajectory | capacity probe (~25) |
| `probe_f0_mechanism.py` | S1 vs A-ft vs B encoder alignment + CKA | F0 overwrite mechanism |

## Granularity–dimensionality — matching needs ≥64 distributed dims
| probe | measures | paper evidence |
|---|---|---|
| `probe_b_granularity_dim.py` | B recovery-rank sweep across matching strides (str4≈64, str8≈48, both ≫ 22) | Fig. recovery_rank |
| `probe_c_granularity_dim.py` | same recovery sweep on the direct-trained cascade C | Fig. recovery_rank (C) |
| `probe_b_matchdim.py` | B matching-dimensionality recovery (in-domain) | supply-vs-demand (matching demand) |
| `probe_b_matchdim_ood.py` | B matching-dimensionality recovery (cross-dataset / OOD) | robustness of the dim demand |
| `probe_b_conditional_dims.py` | conditional / incremental dimension contribution | distributed-not-low-rank support |
| `probe_recovery_by_stride.py` | recovery rank across strides 16/8/4/2/1 (GT-centred ±4 window for fine) | diagnostic (inverted-U; not in paper) |
| `probe_alignment_recovery.py` | Stage-1 encoder de-leaked CCA → top-k restricted stride-16 alignment hit-rate (lands ≈22 by construction) | Fig. recovery_rank (stride-16 curve) |

## Effective rank & overwrite — pre-alignment supplies only a low-rank code
| probe | measures | paper evidence |
|---|---|---|
| `probe_erank_extractors.py` | SAR-extractor erank by training state: random 34 → aligned(locked) 23 → fine-tuned 12 | Fig. overwrite panel (c) |
| `probe_erank_cascade.py` | erank across cascade strides 8/4/2/1 (non-monotone, inverted-U) | diagnostic (not in paper) |
| `probe_erank_granularity.py` | erank as a function of task granularity | diagnostic support |

Related analysis scripts live one level up in `Source/Tools/`:
`stage_ablation.py` (Table cascade-stage ablation), `ci_reject.py`
(risk–coverage / rejection curve, Fig. risk_coverage),
`profile_cascade.py` (runtime/efficiency profiling, Table efficiency).

Default checkpoint: `Checkpoint_B_256/model_epoch_4.pth` (the analyzed
specimen) unless the docstring says otherwise; erank/overwrite probes also
load the Stage-1 (`Checkpoint_A_stage1_256`) and A-ft (`Checkpoint_A_ft_256`)
encoders. Numbers ledger with status tags: `PAPER_DATA_AND_ANALYSIS.md`
§10 (one level above the project).
