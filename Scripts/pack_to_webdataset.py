#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pack the SAR / RGB / label triples listed in an ``image_*.pt`` manifest
into WebDataset TAR shards.

Reads paths from the source ``.pt`` (random IO on the source HDD), writes
shards sequentially to the destination directory. A ProducerN -> Writer
queue keeps the disk-side parallel reads decoupled from the single-threaded
tar writer so the bottleneck is the slowest disk rather than thread
synchronisation.

Output layout (one tar per shard)::

    shard-000000.tar
        ├── 00000000.sar.png
        ├── 00000000.rgb.png     (or .rgb.jpg if --rgb_format jpg)
        ├── 00000000.label.cls
        ├── 00000001.sar.png
        ...

Usage example (~1GB shards, parallel reads from /data1, sequential writes
to /data5)::

    python Scripts/pack_to_webdataset.py \
        --src_pt ./Datasets/image_test.pt \
        --dst_dir /data5/datasets/TM_shards \
        --samples_per_shard 32000 \
        --read_threads 64
"""

import argparse
import io
import os
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Optional

import torch


def _load_manifest(pt_path: str):
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    return data['sar_path'], data['rgb_path'], data['label']


def _read_bytes(path: str) -> bytes:
    with open(path, 'rb') as fh:
        return fh.read()


def _maybe_recode_rgb(rgb_bytes: bytes, rgb_format: str, jpeg_quality: int) -> bytes:
    if rgb_format == 'png':
        return rgb_bytes
    from PIL import Image  # local import keeps PNG-only mode dependency-free
    img = Image.open(io.BytesIO(rgb_bytes)).convert('RGB')
    out = io.BytesIO()
    img.save(out, format='JPEG', quality=jpeg_quality)
    return out.getvalue()


def _producer(idx: int, sar_paths, rgb_paths, labels,
              rgb_format: str, jpeg_quality: int):
    sar_bytes = _read_bytes(sar_paths[idx])
    rgb_bytes = _read_bytes(rgb_paths[idx])
    rgb_bytes = _maybe_recode_rgb(rgb_bytes, rgb_format, jpeg_quality)
    lbl = labels[idx]
    if hasattr(lbl, 'tolist'):
        lbl = lbl.tolist()
    label_text = f"{int(lbl[0])} {int(lbl[1])}\n".encode('utf-8')
    return idx, sar_bytes, rgb_bytes, label_text


class ShardWriter:
    def __init__(self, dst_dir: str, samples_per_shard: int, rgb_ext: str) -> None:
        self.dst_dir = dst_dir
        self.samples_per_shard = samples_per_shard
        self.rgb_ext = rgb_ext
        self._shard_idx = 0
        self._sample_in_shard = 0
        self._tar: Optional[tarfile.TarFile] = None

    def _shard_path(self) -> str:
        return os.path.join(self.dst_dir, f"shard-{self._shard_idx:06d}.tar")

    def _ensure_open(self) -> None:
        if self._tar is None:
            self._tar = tarfile.open(self._shard_path(), mode='w')

    def _add_entry(self, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = 0
        self._tar.addfile(info, io.BytesIO(data))

    def write(self, key: str, sar_bytes: bytes, rgb_bytes: bytes,
              label_bytes: bytes) -> None:
        self._ensure_open()
        self._add_entry(f"{key}.sar.png", sar_bytes)
        self._add_entry(f"{key}.rgb.{self.rgb_ext}", rgb_bytes)
        self._add_entry(f"{key}.label.cls", label_bytes)
        self._sample_in_shard += 1
        if self._sample_in_shard >= self.samples_per_shard:
            self._tar.close()
            self._tar = None
            self._shard_idx += 1
            self._sample_in_shard = 0

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None


def pack(args) -> None:
    os.makedirs(args.dst_dir, exist_ok=True)
    print(f"[pack] loading manifest: {args.src_pt}", flush=True)
    sar_paths, rgb_paths, labels = _load_manifest(args.src_pt)
    total = len(sar_paths)
    if args.limit > 0:
        total = min(total, args.limit)
    print(f"[pack] N={total}, dst={args.dst_dir}, "
          f"per_shard={args.samples_per_shard}, "
          f"shards_expected={(total + args.samples_per_shard - 1) // args.samples_per_shard}",
          flush=True)

    rgb_ext = 'jpg' if args.rgb_format == 'jpg' else 'png'
    writer = ShardWriter(args.dst_dir, args.samples_per_shard, rgb_ext)

    queue: Queue = Queue(maxsize=args.queue_size)
    next_to_write = 0
    pending = {}

    def submit_range(start: int, end: int, pool: ThreadPoolExecutor):
        for i in range(start, end):
            future = pool.submit(_producer, i, sar_paths, rgb_paths, labels,
                                 args.rgb_format, args.jpeg_quality)
            future.add_done_callback(lambda fut: queue.put(fut.result()))

    t0 = time.time()
    last_report = t0
    report_every = max(args.report_every, 1)
    n_written = 0

    with ThreadPoolExecutor(max_workers=args.read_threads) as pool:
        # warm up: queue ~4x queue_size of producers, then keep topping up
        in_flight_target = max(args.read_threads * 4, args.queue_size)
        next_submit = 0
        # initial submit
        end = min(in_flight_target, total)
        submit_range(0, end, pool)
        next_submit = end

        while n_written < total:
            idx, sar_b, rgb_b, lbl_b = queue.get()
            pending[idx] = (sar_b, rgb_b, lbl_b)
            # write any in-order samples currently available
            while next_to_write in pending:
                sar_b, rgb_b, lbl_b = pending.pop(next_to_write)
                key = f"{next_to_write:08d}"
                writer.write(key, sar_b, rgb_b, lbl_b)
                next_to_write += 1
                n_written += 1
                # top up submissions
                if next_submit < total:
                    submit_range(next_submit, min(next_submit + 1, total), pool)
                    next_submit += 1
                if n_written % report_every == 0 or n_written == total:
                    now = time.time()
                    dt = now - t0
                    rate = n_written / max(dt, 1e-6)
                    print(f"[pack] {n_written}/{total} "
                          f"({100.0 * n_written / total:.2f}%) "
                          f"rate={rate:.1f} pair/s "
                          f"elapsed={dt:.0f}s eta={(total - n_written) / max(rate, 1e-6):.0f}s",
                          flush=True)
                    last_report = now
    writer.close()
    dt = time.time() - t0
    print(f"[pack] done: {n_written}/{total} in {dt:.0f}s "
          f"avg={n_written / max(dt, 1e-6):.1f} pair/s", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Pack TM dataset into WebDataset shards")
    p.add_argument('--src_pt', required=True,
                   help='source manifest .pt (sar_path/rgb_path/label lists)')
    p.add_argument('--dst_dir', required=True,
                   help='destination directory for shard-NNNNNN.tar files')
    p.add_argument('--samples_per_shard', type=int, default=32000,
                   help='samples per tar shard (~1GB at PNG-RGB 31KB)')
    p.add_argument('--read_threads', type=int, default=64,
                   help='concurrent disk read threads')
    p.add_argument('--queue_size', type=int, default=2048,
                   help='reader -> writer queue depth (backpressure)')
    p.add_argument('--rgb_format', choices=['png', 'jpg'], default='png',
                   help='keep RGB as PNG (lossless) or re-encode to JPEG '
                        '(smaller; lets dataset fit in RAM page cache)')
    p.add_argument('--jpeg_quality', type=int, default=85,
                   help='JPEG quality when --rgb_format=jpg')
    p.add_argument('--limit', type=int, default=0,
                   help='if >0, only pack this many samples (for smoke tests)')
    p.add_argument('--report_every', type=int, default=10000,
                   help='print progress every N samples')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pack(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
