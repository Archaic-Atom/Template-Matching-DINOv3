# Recorded revision launchers

These scripts preserve the finite AutoDL experiment campaigns. They use the
original Linux work directories. Set those paths, restore the datasets and
provide the required checkpoints before launching a campaign.
For portable entry points, start with `Scripts/train` and `Scripts/reproduce`.

| Script | Purpose |
|---|---|
| `run_seed.sh` | Shared-initialization four-arm training and evaluation |
| `r1_followup_20261011.py` | Early-checkpoint evaluation and smaller matching pools |
| `review_queue_20261020.py` | Matched coarse-upsampling and partial-freezing runs |
| `review_few_prepare_20261020.py` | Nested 100/500/1000-pair training subsets |
| `review_few_queue_20261020.py` | Stage-1 and matching runs within each few-pair budget |
| `review_overlap_20261027.py` | Run registered groups on free GPUs while preserving active work |
| `profile_review_cost_20261028.py` | Inference-only latency and memory profiles |
| `diagnostics` | Hardware-specific cuDNN/DDP probes; opt-in through their entry points |

Controllers write command, configuration and checkpoint records. Their
prediction CSVs retain sample indices, image names and ground truth. The
common 20k list is used for the matching comparisons. The queue may also
contain canceled or preflight jobs; use the formal endpoint records for
paper results.

The overlap coordinator has a CPU-only check that uses temporary directories
and mocked workers. It does not start training or require cloud access:

```bash
python Scripts/autodl/test_review_overlap_20261027.py
```
