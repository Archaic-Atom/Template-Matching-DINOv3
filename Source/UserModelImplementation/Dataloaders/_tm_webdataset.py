# -*- coding: utf-8 -*-
"""WebDataset-backed IterableDataset for Template-Matching training.

The on-disk shards are produced by ``Scripts/pack_to_webdataset.py``. Each
sample inside a shard is a triple of entries sharing the same key::

    <key>.sar.png    bytes of the SAR PNG
    <key>.rgb.png    bytes of the RGB PNG (may also be .jpg if packed as JPEG)
    <key>.label.cls  "x y" ASCII text (two integers separated by a space)

The augmentation and preprocessing logic is reused verbatim from
``_tm_datasets.TM_Dataset`` via the helper functions exposed here so behaviour
remains identical between the mapstyle and iterable paths.
"""

import io
import os
import glob
import tarfile
from typing import Iterable, Optional

import torch
from torch import Tensor
from torchvision.io import decode_image, ImageReadMode

try:
    import webdataset as wds
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "webdataset is required for --dataloaderType iterable; "
        "install with `pip install webdataset`"
    ) from exc

try:
    from ._data_aug import TrainAug
except ImportError:
    from _data_aug import TrainAug


def _decode_png_bytes(buf: bytes, gray: bool) -> Tensor:
    mode = ImageReadMode.GRAY if gray else ImageReadMode.RGB
    return decode_image(torch.frombuffer(bytearray(buf), dtype=torch.uint8), mode=mode)


def _parse_label_bytes(buf: bytes) -> Tensor:
    text = buf.decode('utf-8').strip()
    parts = text.replace(',', ' ').split()
    return torch.tensor([int(parts[0]), int(parts[1])], dtype=torch.long)


def _build_heatmap_cache(h: int, w: int) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij',
    )
    coords = torch.stack([xx, yy], dim=0).view(2, -1).permute(1, 0)
    target_x = coords[:, 0].view(-1, 1, 1)
    target_y = coords[:, 1].view(-1, 1, 1)
    dist = torch.sqrt((xx.unsqueeze(0) - target_x) ** 2
                      + (yy.unsqueeze(0) - target_y) ** 2)
    heatmaps = torch.exp(-dist / 1.2)
    heatmaps /= heatmaps.sum(dim=(1, 2), keepdim=True)
    return heatmaps.share_memory_()


def _generate_heatmap(x: int, y: int, h: int, w: int) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij',
    )
    dist = torch.sqrt((xx - float(x)) ** 2 + (yy - float(y)) ** 2)
    heatmap = torch.exp(-dist / 1.2)
    heatmap /= heatmap.sum()
    return heatmap


def _preprocess_image(img: Tensor, duplicate_channels: bool) -> Tensor:
    if img.dtype != torch.float32:
        img = img.float()
    img = img.div(255.0)
    if img.shape[0] == 1 and duplicate_channels:
        img = img.repeat(3, 1, 1)
    mean = img.mean(dim=(1, 2), keepdim=True)
    std = img.std(dim=(1, 2), keepdim=True).clamp_min(1e-7)
    return (img - mean) / std


class TMSampleProcessor:
    """Map function applied to each WebDataset sample.

    Performs the same work as ``TM_Dataset.__getitem__`` minus the
    file-system read (the bytes already arrive from the tar stream).
    """

    def __init__(self, is_train: bool, is_pre_train: bool, is_stage_2: bool,
                 h: int = 64, w: int = 64, template_size: int = 64) -> None:
        # ``h/w`` size the displacement heatmap grid (Stage-1 label) and stay at
        # the displacement range (= ref - template = 64 for BOTH 128/64 and
        # 256/192). ``template_size`` sizes the SAR template / RGB crop and the
        # label-sanitize clip — this is what grows 64 -> 192 at the new geometry.
        # At 128/64 the two coincide (64); at 256/192 they decouple (crop=192,
        # grid=64). Threaded from ``args.sar_size`` by the dataloader.
        # ``h/w`` 是位移热图网格(Stage-1 标签),保持位移范围(ref-template=64,
        # 两种几何都是 64);``template_size`` 是 SAR 模板/RGB 裁剪与 label clip
        # 范围,新几何下 64->192。128/64 两者都=64;256/192 解耦(裁剪 192、网格 64)。
        self.is_train = is_train
        self.is_pre_train = is_pre_train
        self.is_stage_2 = is_stage_2
        self.h = h
        self.w = w
        self.template = template_size
        self._heatmap_cache = _build_heatmap_cache(h, w)

    def _sanitize_label(self, label: Tensor, image_hw):
        x = int(label[0])
        y = int(label[1])
        max_x = max(0, image_hw[1] - self.template)
        max_y = max(0, image_hw[0] - self.template)
        return (min(max(x, 0), max_x), min(max(y, 0), max_y))

    def _label_to_heatmap(self, x: int, y: int) -> Tensor:
        if 0 <= x < self.w and 0 <= y < self.h:
            return self._heatmap_cache[y * self.w + x]
        return _generate_heatmap(x, y, self.h, self.w)

    def __call__(self, sample):
        sar_bytes, rgb_bytes, label_bytes, key = sample
        sar_img = _decode_png_bytes(sar_bytes, gray=True)
        rgb_img = _decode_png_bytes(rgb_bytes, gray=False)
        label = _parse_label_bytes(label_bytes)
        sar_name = f'{key}.sar.png'
        rgb_name = f'{key}.rgb.png'

        # Stage 1 alignment keeps geometric augmentation (rotate / flip).
        # Stage 2 matching disables it: user empirically observed that the
        # rotate/flip aug is too strong for the matching head and slows /
        # destabilises convergence. So we only augment when NOT in stage 2.
        # Stage 1 训练保留几何增强；Stage 2 匹配关闭——经验上增强太强,
        # 影响匹配头收敛。
        if self.is_train and not self.is_stage_2:
            sar_img, rgb_img, label = TrainAug(
                sar_img, rgb_img, label, rgb_img.shape[-2:],
                (self.template, self.template))

        x, y = self._sanitize_label(label, rgb_img.shape[-2:])

        if self.is_pre_train and not self.is_stage_2:
            rgb_img = rgb_img[:, y:y + self.template, x:x + self.template]

        sar_img = _preprocess_image(sar_img, duplicate_channels=True)
        rgb_img = _preprocess_image(rgb_img, duplicate_channels=False)

        if self.is_stage_2:
            label_out = torch.tensor([x, y], dtype=torch.float32)
        else:
            label_out = self._label_to_heatmap(x, y)

        return sar_img, rgb_img, label_out, sar_name, rgb_name


def _expand_shards_to_list(pattern):
    """Resolve a brace pattern (``...{lo..hi}.tar``), a glob, or an
    iterable of paths into a concrete sorted list of shard file paths.

    Returns an empty list if nothing matches; the caller decides whether
    that's an error.
    """
    if isinstance(pattern, str):
        if '{' in pattern and '}' in pattern:
            try:
                import braceexpand
                return list(braceexpand.braceexpand(pattern))
            except ImportError:
                import re
                m = re.search(r'\{(\d+)\.\.(\d+)(?:\.\.\d+)?\}', pattern)
                if not m:
                    return [pattern]
                lo, hi = int(m.group(1)), int(m.group(2))
                width = len(m.group(1))
                prefix = pattern[:m.start()]
                suffix = pattern[m.end():]
                return [f"{prefix}{i:0{width}d}{suffix}" for i in range(lo, hi + 1)]
        return sorted(glob.glob(pattern))
    return list(pattern)


def _count_one_shard(path: str) -> int:
    """Count samples in a single shard (1 sample = 3 tar entries).

    Slow (~5 s cold on HDD): only used in the legacy code path when the
    caller passes a brace pattern instead of a CSV manifest.
    """
    with tarfile.open(path, 'r') as tf:
        return sum(1 for _ in tf) // 3


def _read_path_csv(csv_path: str):
    """Parse a shard manifest CSV (header: ``path,file_num``).

    Returns ``(shard_paths, sample_counts)`` parallel lists. The CSV is
    the source of truth for sample counts; we do not re-probe disk.
    Generate / hand-maintain ``Datasets/shard_training_path.csv`` and
    ``Datasets/shard_testing_path.csv`` to keep this in sync with the
    on-disk shards.
    """
    import csv
    paths, counts = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row['path'])
            counts.append(int(row['file_num']))
    if not paths:
        raise ValueError(f"empty shard manifest: {csv_path}")
    return paths, counts


def _balanced_split_by_node(src):
    """Custom nodesplitter ensuring all DDP ranks get the same number of shards.

    wds.split_by_node uses islice(src, rank, None, world_size) which
    gives some ranks one extra shard when len(src) % world_size != 0.
    Combined with JackFramework's epoch-end dist.barrier(), that
    imbalance causes NCCL ALLREDUCE timeouts: the rank with fewer shards
    reaches barrier first and waits past the 600 s default while the
    longer ranks finish their extra iterations.

    Here we apply the same interleave then truncate each rank's list to
    floor(num_shards / world_size). Tail shards (at most
    world_size - 1 of them) are dropped.
    """
    from itertools import islice
    try:
        from webdataset import utils as _wu
        rank, world_size, _, _ = _wu.pytorch_worker_info()
    except Exception:
        rank, world_size = 0, 1
    shards = list(src)
    if world_size <= 1:
        yield from shards
        return
    per_rank = len(shards) // world_size
    interleaved = list(islice(shards, rank, None, world_size))[:per_rank]
    yield from interleaved


def build_tm_webdataset(shard_pattern: str, is_train: bool, is_pre_train: bool,
                        is_stage_2: bool,
                        shuffle_buffer: int = 100000,
                        distributed: bool = False,
                        batch_size: int = 1,
                        num_workers: int = 1,
                        template_size: int = 64) -> object:
    """Build the WebDataset pipeline ready to hand to ``DataLoader``.

    ``shard_pattern`` accepts two forms:
      * Path ending in ``.csv`` — a shard manifest with header
        ``path,file_num``. Sample counts are read directly; no disk
        probing. Generate / hand-maintain
        ``Datasets/shard_training_path.csv`` and
        ``Datasets/shard_testing_path.csv``.
      * Anything else — a brace pattern (``shard-{lo..hi}.tar``) or
        glob. Legacy path: probes first + last shard for sample counts
        (~5 s each on cold HDD).

    A per-worker ``with_epoch(N)`` cap is applied so every rank emits
    the same number of samples per epoch regardless of shard size
    imbalance (e.g. a partial last shard):
        per_rank_iter = floor(real_total / (batch_size * world_size))
    Per-epoch total = per_rank_iter * batch_size * world_size, dropping
    at most (batch_size * world_size - 1) random samples to reach a
    clean multiple of N*BS. All shards are used; no shard is wholesale
    dropped. The dropped samples differ each epoch because of shuffle.

    Independent of caller-provided imgNum/valImgNum (those are
    framework display/control hints, not ground truth on disk).
    """
    if isinstance(shard_pattern, str) and shard_pattern.endswith('.csv'):
        shard_list, sample_counts = _read_path_csv(shard_pattern)
        real_total = sum(sample_counts)
    else:
        shard_list = _expand_shards_to_list(shard_pattern)
        if not shard_list:
            raise FileNotFoundError(f"No shards matched: {shard_pattern}")
        nominal = _count_one_shard(shard_list[0])
        last_count = _count_one_shard(shard_list[-1]) if len(shard_list) > 1 else nominal
        real_total = nominal * (len(shard_list) - 1) + last_count

    num_shards = len(shard_list)
    # shardshuffle = num_shards gives a true random permutation per epoch
    # (the default sliding-window shuffle only mixes locally, which
    # combined with structured intra-shard packing produces non-stationary
    # loss within an epoch). Memory cost is one URL string per shard.
    shard_shuffle = num_shards if is_train and num_shards > 1 else False

    nodesplitter = _balanced_split_by_node if distributed else wds.single_node_only
    pipeline = wds.WebDataset(
        shard_list,
        shardshuffle=shard_shuffle,
        nodesplitter=nodesplitter,
        workersplitter=wds.split_by_worker,
        handler=wds.warn_and_continue,
        empty_check=False,
    )
    if is_train and shuffle_buffer > 0:
        pipeline = pipeline.shuffle(shuffle_buffer)

    processor = TMSampleProcessor(
        is_train=is_train, is_pre_train=is_pre_train,
        is_stage_2=is_stage_2, template_size=template_size)

    # Per-worker sample cap = floor(real_total / (bs * ws)) * bs // nw.
    # ``real_total`` came from the CSV manifest (preferred) or two-shard
    # probe (legacy). Drops < (bs * ws) random samples to hit a multiple
    # of N*BS; partial last shard is included.
    world_size = 1
    if distributed:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
        except Exception:
            world_size = 1
    nw = max(1, int(num_workers))
    bs = max(1, int(batch_size))
    per_rank_iter = max(1, real_total // (bs * world_size))
    per_worker_samples = max(1, (per_rank_iter * bs) // nw)
    pipeline = pipeline.with_epoch(per_worker_samples)

    pipeline = (
        pipeline
        .to_tuple('sar.png', 'rgb.png', 'label.cls', '__key__')
        .map(processor)
    )
    return pipeline
