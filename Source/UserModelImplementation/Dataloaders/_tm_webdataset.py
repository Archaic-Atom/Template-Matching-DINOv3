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
                 h: int = 64, w: int = 64) -> None:
        self.is_train = is_train
        self.is_pre_train = is_pre_train
        self.is_stage_2 = is_stage_2
        self.h = h
        self.w = w
        self._heatmap_cache = _build_heatmap_cache(h, w)

    def _sanitize_label(self, label: Tensor, image_hw):
        x = int(label[0])
        y = int(label[1])
        max_x = max(0, image_hw[1] - self.w)
        max_y = max(0, image_hw[0] - self.h)
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

        if self.is_train:
            sar_img, rgb_img, label = TrainAug(
                sar_img, rgb_img, label, rgb_img.shape[-2:], (self.h, self.w))

        x, y = self._sanitize_label(label, rgb_img.shape[-2:])

        if self.is_pre_train and not self.is_stage_2:
            rgb_img = rgb_img[:, y:y + self.h, x:x + self.w]

        sar_img = _preprocess_image(sar_img, duplicate_channels=True)
        rgb_img = _preprocess_image(rgb_img, duplicate_channels=False)

        if self.is_stage_2:
            label_out = torch.tensor([x, y], dtype=torch.float32)
        else:
            label_out = self._label_to_heatmap(x, y)

        return sar_img, rgb_img, label_out, sar_name, rgb_name


def _expand_shard_pattern(pattern: str) -> Iterable[str]:
    """Accept either a brace pattern (``shard-{000000..000255}.tar``) or a
    glob (``shards/*.tar``). WebDataset handles braces natively; we only
    fall back to glob if no brace is present."""
    if '{' in pattern and '}' in pattern:
        return pattern
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No shards matched: {pattern}")
    return matches


def _count_shards(pattern: str) -> int:
    """Count shards covered by ``pattern`` so callers can size the
    ``shardshuffle`` buffer to cover the entire list (= a true random
    permutation per epoch, not a sliding-window shuffle).

    Supports brace patterns (``...{000000..000254}.tar``) and globs.
    Returns 1 (the no-shuffle sentinel) if the count can't be resolved.
    """
    if '{' in pattern and '}' in pattern:
        try:
            import braceexpand
            return sum(1 for _ in braceexpand.braceexpand(pattern))
        except ImportError:
            import re
            m = re.search(r'\{(\d+)\.\.(\d+)(?:\.\.\d+)?\}', pattern)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                return hi - lo + 1
            return 1
    return len(glob.glob(pattern))


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
    world_size - 1 of them) are dropped — < 1.2% of the data at our
    255-shard / 4-rank setup.
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
                        is_stage_2: bool, dataset_size: int,
                        shuffle_buffer: int = 100000,
                        distributed: bool = False) -> object:
    """Build the WebDataset pipeline ready to hand to ``DataLoader``.

    No ``with_epoch(N)`` is applied: that caps each *iterator instance*
    via ``islice``, but PyTorch DataLoader copies the dataset to every
    worker, so a per-worker cap cannot express "one framework epoch ==
    one true pass." Instead we rely on natural exhaustion:
    ``split_by_node`` narrows shards per rank, ``split_by_worker``
    narrows them per worker, then each worker drains its share and
    StopIteration propagates.

    Coverage per framework epoch: every shard assigned to this rank is
    read exactly once (only integer-division rounding from
    ``num_shards / world_size`` can drop a shard, which is at most one).

    Caveat: the bar denominator JackFramework derives from
    ``imgNum / batchSize`` may not match the actual per-rank iter count
    for the first epoch; the framework logs "input images numbers is
    different the number of datasets" and self-corrects via
    ``set_training_iteration`` for subsequent epochs.
    """

    shards = _expand_shard_pattern(shard_pattern)

    # shardshuffle = num_shards gives a true random permutation per epoch
    # (the default sliding-window shuffle only mixes locally, which combined
    # with structured intra-shard packing produces non-stationary loss
    # within an epoch). Memory cost: one URL string per shard ≈ negligible.
    num_shards = _count_shards(shard_pattern)
    shard_shuffle = num_shards if is_train and num_shards > 1 else False

    nodesplitter = _balanced_split_by_node if distributed else wds.single_node_only
    pipeline = wds.WebDataset(
        shards,
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
        is_stage_2=is_stage_2)

    # Per-rank epoch size: split_by_node already narrowed shards across ranks;
    # with_epoch should match that narrowing.
    world_size = 1
    if distributed:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
        except Exception:
            world_size = 1
    per_rank_epoch = max(1, dataset_size // world_size)
    del per_rank_epoch  # iter cap removed; pipeline drains naturally

    pipeline = (
        pipeline
        .to_tuple('sar.png', 'rgb.png', 'label.cls', '__key__')
        .map(processor)
    )
    return pipeline
