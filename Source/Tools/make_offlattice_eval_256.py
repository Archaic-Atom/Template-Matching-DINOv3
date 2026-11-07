#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a paired on-lattice / off-lattice eval set at opt256/sar192.

构建 opt256/sar192 的"格点 / 非格点"配对评测集。

Why this exists
---------------
Every SEN1-2 evaluation in the paper puts the ground-truth displacement on the
stride-4 lattice, because ``make_opt256_sar192_shards._offsets_test`` and
``make_2k_eval_256`` both build offsets with ``range(0, 65, 4)``. That lattice
is exactly the coarse head's 17x17 output grid, so the strict thresholds are
read under a condition the random-crop protocol our competitors use never
sees. Training itself draws arbitrary integers. This tool measures the gap.

论文里所有 SEN1-2 评测的 GT 位移都落在 stride-4 格点上，而该格点恰好就是粗
匹配 head 的 17x17 输出格点；训练用的却是任意整数位移。本工具量化二者差距。

How the comparison is kept clean
--------------------------------
Every test pair contributes one row under each condition. The two rows share
the optical reference and the source SAR image, and differ only in where the
template was cut, so pair difficulty cancels exactly and a paired bootstrap
over pairs is valid.

每个测试配对在两种条件下各产出一行。两行共享同一张 optical 和同一张源 SAR，
只有模板裁剪位置不同，因此配对难度完全抵消，可按配对做成对 bootstrap。

The test split is replayed from ``make_opt256_sar192_shards`` rather than read
from a manifest, so the pairs are provably the same ones held out of training.
测试划分由 ``make_opt256_sar192_shards`` 确定性重放，而非读清单，因此可证明
这些配对与训练集不重叠。

Output
------
Recover the condition from the ``_g``/``_r`` suffix of the template name, not
from ``gt % 4``, since a uniform draw lands on the lattice anyway with
probability ``(17/65)^2``.
用文件名的 ``_g``/``_r`` 后缀区分条件，不要用 ``gt % 4``。

Example::

    python make_offlattice_eval_256.py \\
        --out_dir /data/raozhibo/datasets/eval_lattice_ab_256 --per_pair 1
"""

import argparse
import json
import os
import random
import sys
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

def _locate_shard_builder() -> str:
    """Find the directory holding ``make_opt256_sar192_shards``.

    定位 ``make_opt256_sar192_shards`` 所在目录。

    Deployments disagree on where it lives, so probe both known locations.
    Importing it rather than re-implementing the split is the point: the eval
    pairs are then provably the ones the shard builder held out of training.
    不同部署里它的位置不一致，因此两处都探测。坚持 import 而不是重写划分逻辑
    的意义在于：这样才能证明评测配对就是 shard 构建器留出训练集的那一批。

    Returns:
        Absolute directory to prepend to ``sys.path``.
        应插入 ``sys.path`` 的绝对目录。

    Raises:
        ImportError: if no candidate location holds the module.
            所有候选位置都没有该模块时抛出。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    for cand in (here, os.path.join(root, "Scripts", "legacy")):
        if os.path.isfile(os.path.join(cand, "make_opt256_sar192_shards.py")):
            return cand
    raise ImportError(
        "make_opt256_sar192_shards.py not found in Source/Tools or "
        "Scripts/legacy; the test split cannot be replayed")


sys.path.insert(0, _locate_shard_builder())
from make_opt256_sar192_shards import enumerate_pairs, split_pairs  # noqa: E402

# Offsets are drawn from their own RNG so the pair order never shifts when the
# number of draws per pair changes.
# 位移用独立 RNG，改变每对抽样次数不会打乱配对顺序。
OFFSET_SEED_STRIDE = 1_000_003


def draw_offsets(rng: random.Random, offset_max: int,
                 stride: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Draw one on-lattice and one uniform-integer offset.

    抽一个格点位移和一个均匀整数位移。

    The second draw is deliberately not resampled away from the lattice, so it
    reproduces the competitors' random-crop protocol as-is. Subsetting it to
    the draws that miss the lattice isolates the mechanism.
    第二个位移刻意不重抽排除格点，这样它原样等同于竞品的随机裁剪协议；把它筛
    成不落格点的子集则可隔离机制。

    Args:
        rng: Offset random source. 位移随机源。
        offset_max: Largest displacement on either axis. 单轴最大位移。
        stride: Lattice step of the coarse head. 粗匹配 head 的格点步长。

    Returns:
        ``(grid_offset, random_offset)`` as ``(cx, ry)``.
        ``(格点位移, 随机位移)``，均为 ``(cx, ry)``。
    """
    lattice = range(0, offset_max + 1, stride)
    return ((rng.choice(lattice), rng.choice(lattice)),
            (rng.randint(0, offset_max), rng.randint(0, offset_max)))


def test_pairs(src_root: str, test_per_season: int,
               seed: int) -> List[Sequence[str]]:
    """Replay the held-out test split deterministically.

    确定性重放留出的测试划分。

    Args:
        src_root: SEN1-2 source tree. SEN1-2 源目录。
        test_per_season: Pairs held out per season. 每季留出的配对数。
        seed: Split seed. 划分随机种子。

    Returns:
        The test pairs, in the same order the shard builder used.
        测试配对，顺序与 shard 构建器一致。
    """
    _, test = split_pairs(enumerate_pairs(src_root), test_per_season, seed)
    return test


def main() -> None:
    """Build the paired eval set. 构建配对评测集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_root",
                        default="/data1/datasets/Template_matching/SEN1-2")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--per_pair", type=int, default=1,
                        help="paired (grid, random) draws per test pair")
    parser.add_argument("--max_pairs", type=int, default=0,
                        help="0 = every test pair")
    parser.add_argument("--sar_size", type=int, default=192)
    parser.add_argument("--offset_max", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--test_per_season", type=int, default=1935)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    sar_out = os.path.join(args.out_dir, "sar")
    rgb_out = os.path.join(args.out_dir, "rgb")
    os.makedirs(sar_out, exist_ok=True)
    os.makedirs(rgb_out, exist_ok=True)

    pairs = test_pairs(args.src_root, args.test_per_season, args.split_seed)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"[ab] {len(pairs)} held-out test pairs", flush=True)

    off_rng = random.Random(args.seed + OFFSET_SEED_STRIDE)
    size = args.sar_size
    rows = []
    for done, (season, scene, pid, sar_path, opt_path) in enumerate(pairs):
        sar = np.asarray(Image.open(sar_path).convert("L"))
        tag = f"{season}_{scene}_p{pid}"

        # One optical reference per pair, shared by both conditions.
        # 每对一张 optical，两种条件共享。
        rgb_dst = os.path.join(rgb_out, f"{tag}.png")
        with open(opt_path, "rb") as src, open(rgb_dst, "wb") as dst:
            dst.write(src.read())

        for draw in range(args.per_pair):
            grid_xy, rand_xy = draw_offsets(
                off_rng, args.offset_max, args.stride)
            for (cx, ry), suffix in ((grid_xy, "g"), (rand_xy, "r")):
                name = f"{tag}_d{draw}_{cx}_{ry}_{suffix}.png"
                dst_path = os.path.join(sar_out, name)
                Image.fromarray(sar[ry:ry + size, cx:cx + size]).save(dst_path)
                rows.append((dst_path, rgb_dst, json.dumps([cx, ry])))

        if (done + 1) % 1000 == 0:
            print(f"[ab] {done + 1}/{len(pairs)} pairs", flush=True)

    csv_path = os.path.join(args.out_dir, "image_test_lattice_ab_256.csv")
    with open(csv_path, "w") as handle:
        handle.write("SAR_Image,RGB_Image,Label\n")
        for sar_p, rgb_p, label in rows:
            handle.write(f'{sar_p},{rgb_p},"{label}"\n')

    n_grid = sum(1 for r in rows if r[0].endswith("_g.png"))
    print(f"[ab] {len(pairs)} pairs -> {len(rows)} rows "
          f"({n_grid} on-lattice / {len(rows) - n_grid} random) -> {csv_path}")


if __name__ == "__main__":
    main()
