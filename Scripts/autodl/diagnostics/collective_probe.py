"""Check large NCCL collectives before a real model's backward pass.
检查较大 NCCL 通信，区分真实模型反向传播中的通信与计算问题。
"""
import csv
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=45))
rows = []
for count in (1, 1024, 8 * 1024 * 1024, 64 * 1024 * 1024):
    print(f"rank={rank} begin elements={count}", flush=True)
    value = torch.full((count,), rank + 1.0, device=f"cuda:{rank}")
    start = time.monotonic()
    dist.all_reduce(value)
    torch.cuda.synchronize()
    assert value[0].item() == 3.0 and value[-1].item() == 3.0
    seconds = time.monotonic() - start
    print(f"rank={rank} PASS elements={count} seconds={seconds:.6f}", flush=True)
    rows.append({"rank": rank, "elements": count, "seconds": seconds})
    del value
dist.destroy_process_group()
out = Path("/root/autodl-tmp/setup/campaign_20260925")
with (out / f"collective_rank{rank}.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["rank", "elements", "seconds"])
    writer.writeheader()
    writer.writerows(rows)
