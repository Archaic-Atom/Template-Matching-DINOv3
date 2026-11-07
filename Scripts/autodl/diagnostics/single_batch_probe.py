"""Locate real-batch backward latency without modifying the model.
不修改模型，用真实 batch 定位反向传播耗时。
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT = Path("/root/autodl-tmp/repos/005_Template-Matching-DINOv3")
RUN = Path("/root/autodl-tmp/setup/campaign_20260925")
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "Source"))
sys.path.insert(0, str(PROJECT / "Source/Tools"))
from smoke_factorial import _args
from UserModelImplementation.Models import model_zoo
from UserModelImplementation.Dataloaders._tm_webdataset import build_tm_webdataset

torch.set_num_threads(4)
torch.cuda.set_device(0)
torch.manual_seed(2025)
torch.backends.cudnn.benchmark = os.environ.get("TM_DIAG_BENCHMARK", "1") == "1"
print("BENCHMARK", torch.backends.cudnn.benchmark, flush=True)
if "TM_DIAG_BENCHMARK_LIMIT" in os.environ:
    torch.backends.cudnn.benchmark_limit = int(os.environ["TM_DIAG_BENCHMARK_LIMIT"])
print("BENCHMARK_LIMIT", torch.backends.cudnn.benchmark_limit, flush=True)
args = _args()
args.batchSize = 32
args.mode = "train"
args.fine_warmup_epochs = 0
interface = model_zoo(args, "TM_Model_G")
interface._current_epoch = 4
model = interface.get_model()[0].cuda().train()
checkpoint = torch.load(
    RUN / "preflight_backward_trace/G/checkpoint/model_stage1_init.pth",
    map_location="cpu", weights_only=False)
interface.load_model(model, checkpoint, 0)
dataset = build_tm_webdataset(
    str(RUN / "preflight_backward_trace/manifest.csv"),
    is_train=True, is_pre_train=False, is_stage_2=True,
    shuffle_buffer=64, distributed=False, batch_size=32,
    num_workers=1, template_size=192)
batch = next(iter(DataLoader(dataset, batch_size=32, num_workers=0)))
sar, rgb, label = (item.cuda() for item in batch[:3])
assert sar.shape == (32, 3, 192, 192)
assert rgb.shape == (32, 3, 256, 256)
start = time.monotonic()
seen = set()


def gradient_hook(name: str):
    """Print the first gradient per component. 记录各模块首个梯度到达时间。"""
    def report(gradient: torch.Tensor) -> torch.Tensor:
        prefix = ".".join(name.split(".")[:3])
        if prefix not in seen:
            seen.add(prefix)
            print(f"GRAD {time.monotonic() - start:.3f}s {prefix}", flush=True)
        return gradient
    return report


for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        parameter.register_hook(gradient_hook(name))
print("FORWARD_BEGIN", flush=True)
output = interface.inference(model, [sar, rgb], 0)
torch.cuda.synchronize()
forward_s = time.monotonic() - start
print("FORWARD_DONE", forward_s, flush=True)
loss = interface.loss(output, [label], 0)[0]
print("LOSS", loss.item(), flush=True)
backward_start = time.monotonic()
loss.backward()
torch.cuda.synchronize()
backward_s = time.monotonic() - backward_start
print("BACKWARD_DONE", backward_s, flush=True)
assert torch.isfinite(loss)
with (RUN / ("single_batch_timing_" + os.environ.get("TM_DIAG_BENCHMARK", "1") + "_limit" + os.environ.get("TM_DIAG_BENCHMARK_LIMIT", "default") + ".csv")).open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "batch", "forward_seconds", "backward_seconds", "peak_allocated_bytes"])
    writer.writeheader()
    writer.writerow({"batch": 32, "forward_seconds": forward_s,
                     "backward_seconds": backward_s,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
print("UNUSED_PARAMETERS", [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None], flush=True)
print("SINGLE_REAL_BATCH_PASS", flush=True)
