"""Controlled cuDNN benchmark on a cached real batch.
固定真实输入和模型权重，对比 cuDNN 冷启动与稳态耗时。
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/root/autodl-tmp/repos/005_Template-Matching-DINOv3")
RUN = Path("/root/autodl-tmp/setup/campaign_20260925/cudnn_audit")
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "Source"))
sys.path.insert(0, str(PROJECT / "Source/Tools"))
from smoke_factorial import _args
from UserModelImplementation.Models import model_zoo
from UserModelImplementation.Dataloaders._tm_webdataset import build_tm_webdataset


def seed_all() -> None:
    """Fix all relevant RNGs. 固定所有相关随机源。"""
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)


def prepare() -> None:
    """Cache one batch so all modes see identical samples. 缓存同一真实批次。"""
    from torch.utils.data import DataLoader
    seed_all()
    dataset = build_tm_webdataset(
        "/root/autodl-tmp/setup/campaign_20260925/preflight/manifest.csv",
        is_train=True, is_pre_train=False, is_stage_2=True,
        shuffle_buffer=64, distributed=False, batch_size=32,
        num_workers=1, template_size=192)
    batch = next(iter(DataLoader(dataset, batch_size=32, num_workers=0)))
    path = RUN / "fixed_batch.pt"
    assert not path.exists(), "Refuse to overwrite audit fixture"
    torch.save([item.cpu() for item in batch[:3]], path)
    print("FIXTURE", hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)


def main() -> None:
    """Measure isolated forward/backward/optimizer times. 分段测量算子执行耗时。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--name", default="limit1")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    opts = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if opts.prepare:
        prepare()
        return
    seed_all()
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = opts.limit >= 0
    torch.backends.cudnn.benchmark_limit = max(opts.limit, 1)
    args = _args()
    args.batchSize = 32
    args.mode = "train"
    args.maxEpochs = 5
    args.fine_warmup_epochs = 3
    interface = model_zoo(args, "TM_Model_G")
    interface._current_epoch = 3
    model = interface.get_model()[0].cuda().train()
    checkpoint = torch.load(
        PROJECT / "Checkpoint_G_s1tr_256_s2025/model_stage1_init.pth",
        map_location="cpu", weights_only=False)
    interface.load_model(model, checkpoint, 0)
    optimizer = interface.optimizer([model], args.lr)[0][0]
    fixture = torch.load(RUN / "fixed_batch.pt", weights_only=False)
    sar, rgb, label = [item.cuda() for item in fixture]
    assert sar.shape == (32, 3, 192, 192)
    assert rgb.shape == (32, 3, 256, 256)
    # Restore original weights outside timing to eliminate trajectory drift.
    # 每步计时外恢复固定权重，避免优化轨迹差异污染对比。
    fixed = {k: v.detach().clone() for k, v in model.trainable_tail.state_dict().items()}
    model_hash = hashlib.sha256()
    for key, val in sorted(fixed.items()):
        model_hash.update(key.encode())
        model_hash.update(val.cpu().numpy().tobytes())
    report = {
        "name": opts.name, "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(), "cuda": torch.version.cuda,
        "benchmark": torch.backends.cudnn.benchmark,
        "limit": torch.backends.cudnn.benchmark_limit,
        "tf32_conv": torch.backends.cudnn.allow_tf32,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "input_sha256": hashlib.sha256((RUN / "fixed_batch.pt").read_bytes()).hexdigest(),
        "tail_sha256": model_hash.hexdigest(),
        "batch_per_gpu": 32, "single_gpu": True,
        "fine_warmup_epoch": 3, "warmup_steps": 3,
        "restored_weights_each_step": True,
    }
    print("CONFIG", json.dumps(report), flush=True)
    rows = []

    def step(index: int, snapshot: bool = False) -> None:
        """Run one original optimization step. 执行一次原配方优化。"""
        model.trainable_tail.load_state_dict(fixed)
        seed_all()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.profiler.record_function("MODEL_FORWARD"):
            output = interface.inference(model, [sar, rgb], 0)
            loss = interface.loss(output, [label], 0)[0]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print("FORWARD", index, round(t1 - t0, 4), flush=True)
        with torch.profiler.record_function("LOSS_BACKWARD"):
            loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if snapshot:
            saved = {"output": {
                "h4": output[0]["heatmap_per_stride"][4].detach().cpu(),
                "h8": output[0]["heatmap_per_stride"][8].detach().cpu(),
                "fine": output[0]["heatmap_fine"].detach().cpu(),
                "anchor": output[0]["fine_anchor_pix"].detach().cpu(),
                "loss": loss.detach().cpu(),
            }, "grads": {
                name: p.grad.detach().cpu()
                for name, p in model.named_parameters() if p.grad is not None
            }}
            torch.save(saved, RUN / (opts.name + "_numerics.pt"))
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        with torch.profiler.record_function("ADAM_STEP"):
            optimizer.step()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in model.parameters())
        row = {"step": index, "forward_s": t1-t0, "backward_s": t2-t1,
               "optimizer_s": t4-t3, "total_s": t1-t0+t2-t1+t4-t3,
               "loss": loss.item()}
        rows.append(row)
        print("STEP", json.dumps(row), flush=True)

    torch.cuda.reset_peak_memory_stats()
    for index in range(3 + opts.steps):
        step(index, snapshot=(index == 0))
    report["steps"] = rows
    measured = rows[3:]
    for key in ("forward_s", "backward_s", "optimizer_s", "total_s"):
        report["median_" + key] = statistics.median(x[key] for x in measured)
        report["mean_" + key] = statistics.mean(x[key] for x in measured)
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    (RUN / (opts.name + ".json")).write_text(json.dumps(report, indent=2))
    print("STEADY_RESULT", json.dumps({k:v for k,v in report.items() if k!="steps"}), flush=True)
    if opts.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
            step(999)
        table = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=30)
        (RUN / (opts.name + "_profile.txt")).write_text(table)
        prof.export_chrome_trace(str(RUN / (opts.name + "_trace.json")))
        print("PROFILE_DONE", flush=True)
    print("AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
