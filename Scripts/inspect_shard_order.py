#!/usr/bin/env python3
"""E2: inspect packed shard order to decide if shuffle_buffer matters.

The current pipeline uses `shardshuffle=100` (shuffles 100 of 255 shards
in a buffer) AND a per-worker `.shuffle(10000)` on raw tar entries. This
matters for contrastive learning: same-batch samples should be diverse;
if shards group adjacent geographic regions or time-series neighbors,
small shuffle buffer means correlated batches → weaker negatives.

This script inspects a single shard and reports:
  1. The sample keys (filenames) in order — look for runs of adjacent
     IDs that suggest geographic clustering.
  2. Decode the first 16 samples of the shard and the last 16, plot
     a per-pixel mean comparison. If start vs end of shard look very
     different (different regions), the shard ordering is structured;
     if they look similar (random samples from across the dataset),
     packing was random and small shuffle_buffer is fine.

Run on remote (no GPU needed):
    conda activate stereoA
    python Scripts/inspect_shard_order.py
"""
from __future__ import annotations

import io
import os
import sys
import tarfile

SHARD = '/data5/datasets/TM_shards/shard-000000.tar'
N_KEYS_SHOW = 30        # how many keys to print from start of shard
N_DECODE = 8            # how many samples to decode for content check


def main() -> int:
    if not os.path.exists(SHARD):
        print(f"ERROR: shard not found at {SHARD}")
        return 1

    print(f"Inspecting shard: {SHARD}")
    print()

    # ── 1. List sample keys ───────────────────────────────────────────────
    with tarfile.open(SHARD, 'r') as tar:
        names = tar.getnames()

    # Pull just the .sar.png filenames (each sample has 3 entries)
    sar_keys = [n for n in names if n.endswith('.sar.png')]
    print(f"Total samples in shard: {len(sar_keys)}")
    print()
    print(f"First {N_KEYS_SHOW} sample keys (look for sequential numbering or "
          f"shared prefixes):")
    for k in sar_keys[:N_KEYS_SHOW]:
        print(f"  {k}")
    print()
    print(f"Last {N_KEYS_SHOW//2} sample keys:")
    for k in sar_keys[-N_KEYS_SHOW//2:]:
        print(f"  {k}")
    print()

    # ── 2. Inspect content at start vs end of shard ───────────────────────
    try:
        import torch
        from torchvision.io import decode_image, ImageReadMode
    except ImportError:
        print("[skip content check — torch/torchvision unavailable]")
        return 0

    print(f"Decoding first {N_DECODE} and last {N_DECODE} SAR samples; "
          f"computing per-sample mean pixel values:")
    print(f"  (if start vs end have very different means/stds, packing is "
          f"structured by some axis like time/geography)")
    print()

    def _decode_at(tar_path: str, key: str):
        with tarfile.open(tar_path, 'r') as tar:
            m = tar.getmember(key)
            f = tar.extractfile(m)
            buf = f.read()
        tensor = decode_image(
            torch.frombuffer(bytearray(buf), dtype=torch.uint8),
            mode=ImageReadMode.GRAY,
        ).float()
        return tensor.mean().item(), tensor.std().item()

    print(f"{'pos':>6}  {'key':>34}  {'mean':>8}  {'std':>8}")
    starts = []
    ends = []
    for i, key in enumerate(sar_keys[:N_DECODE]):
        m, s = _decode_at(SHARD, key)
        starts.append((m, s))
        print(f"{i:>6}  {key:>34}  {m:>8.2f}  {s:>8.2f}")
    print("  ...")
    for i, key in enumerate(sar_keys[-N_DECODE:]):
        m, s = _decode_at(SHARD, key)
        ends.append((m, s))
        print(f"{len(sar_keys)-N_DECODE+i:>6}  {key:>34}  {m:>8.2f}  {s:>8.2f}")

    import statistics
    start_mean = statistics.mean(m for m, _ in starts)
    end_mean = statistics.mean(m for m, _ in ends)
    start_std = statistics.mean(s for _, s in starts)
    end_std = statistics.mean(s for _, s in ends)

    print()
    print(f"Start-of-shard:  mean(mean)={start_mean:.2f}  mean(std)={start_std:.2f}")
    print(f"End-of-shard:    mean(mean)={end_mean:.2f}  mean(std)={end_std:.2f}")

    mean_drift = abs(start_mean - end_mean)
    std_drift = abs(start_std - end_std)
    print()
    if mean_drift < 5 and std_drift < 5:
        print("CONCLUSION: start ≈ end → packing looks RANDOM → "
              "10K shuffle_buffer is fine.")
    elif mean_drift < 15 and std_drift < 10:
        print("CONCLUSION: mild drift → packing might be loosely structured → "
              "consider bumping shuffle_buffer to 50K for safety.")
    else:
        print("CONCLUSION: strong drift → packing is structured (probably by "
              "geographic region or capture time) → 10K is TOO SMALL, bump to "
              "100K or larger to mix shards properly.")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
