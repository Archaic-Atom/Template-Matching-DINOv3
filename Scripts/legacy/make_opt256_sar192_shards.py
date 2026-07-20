#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate opt256/sar192 WebDataset shards from the original SEN1-2.

从原始 256x256 SEN1-2（co-registered s1 SAR + s2 optical）直接裁剪 + 打包成
WebDataset shards，对齐 ME-SSM 协议（optical reference 256x256，SAR template
192x192，top-left = GT）。一步到位（裁剪即打包），不落 1500 万中间 PNG。

Geometry / 几何:
    optical (RGB) reference = 256x256  (s2 patch, 原样字节复制)
    SAR template           = 192x192  (s1 patch 在 (ry, cx) 处裁剪)
    label.cls              = "cx ry"   (x=列, y=行; top-left 坐标, GT)
    offset 范围            = [0, 64]    (256 - 192 = 64)

Splits / 划分（train/test 严格不相交，避免泄漏）:
    TEST  = 每季 ``--test_per_season`` 对（固定 seed 抽），× stride-4 289 网格
    TRAIN = 全量减去 TEST 的所有对，× 每对 ``--train_offsets`` 个随机偏移

Shard 样本格式（与 _tm_webdataset.py 完全兼容）:
    <key>.sar.png    192x192 8-bit 灰度 PNG
    <key>.rgb.png    256x256 8-bit RGB PNG（原 s2 字节）
    <key>.label.cls  "cx ry" ASCII

Usage（先 test 小规模 smoke，再全量后台跑）:
    python make_opt256_sar192_shards.py --split test  --workers 16
    python make_opt256_sar192_shards.py --split train --workers 16
"""

import argparse
import io
import json
import os
import random
import sys
import tarfile
import time
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from PIL import Image

SEASONS = ["ROIs1158_spring", "ROIs1868_summer", "ROIs1970_fall",
           "ROIs2017_winter"]


# --------------------------------------------------------------------------
# Pair enumeration / 配对枚举
# --------------------------------------------------------------------------
def enumerate_pairs(src_root: str) -> List[Tuple[str, str, str, str, str]]:
    """Enumerate all co-registered (SAR, optical) pairs across 4 seasons.

    Returns a deterministically-sorted list of tuples:
        (season, scene_id, patch_id, sar_png_path, opt_png_path)
    """
    pairs = []
    for season in SEASONS:
        sdir = os.path.join(src_root, season)
        if not os.path.isdir(sdir):
            continue
        # scene dirs: s1_<i> (SAR) and s2_<i> (optical) / 场景目录
        scenes = sorted(
            d[3:] for d in os.listdir(sdir)
            if d.startswith("s1_") and os.path.isdir(os.path.join(sdir, d)))
        for scene_id in sorted(scenes, key=lambda x: int(x)):
            s1_dir = os.path.join(sdir, f"s1_{scene_id}")
            s2_dir = os.path.join(sdir, f"s2_{scene_id}")
            if not os.path.isdir(s2_dir):
                continue
            for fn in sorted(os.listdir(s1_dir)):
                if not fn.endswith(".png"):
                    continue
                # s1 name: ROIs..._s1_<scene>_p<pid>.png; optical swaps s1->s2
                # s1 名转 s2 名（同一 patch id，共配准）
                opt_fn = fn.replace("_s1_", "_s2_")
                opt_path = os.path.join(s2_dir, opt_fn)
                if not os.path.isfile(opt_path):
                    continue
                pid = fn.rsplit("_p", 1)[-1].rsplit(".", 1)[0]
                pairs.append((season, scene_id, pid,
                              os.path.join(s1_dir, fn), opt_path))
    return pairs


def split_pairs(pairs, test_per_season: int, seed: int):
    """Deterministically select TEST pairs per season; rest = TRAIN.

    每季用固定 seed 打乱后取前 N 个做 TEST，其余 TRAIN。两者不相交。
    """
    by_season = {s: [] for s in SEASONS}
    for p in pairs:
        by_season[p[0]].append(p)
    test, train = [], []
    for s in SEASONS:
        lst = sorted(by_season[s])  # deterministic base order
        rng = random.Random(f"{seed}:{s}")
        rng.shuffle(lst)
        test.extend(lst[:test_per_season])
        train.extend(lst[test_per_season:])
    return sorted(train), sorted(test)


# --------------------------------------------------------------------------
# Crop + encode / 裁剪与编码
# --------------------------------------------------------------------------
def _offsets_test(stride: int, omax: int):
    """Dense stride-grid offsets: (cx, ry) in {0,stride,...,omax}^2."""
    vals = list(range(0, omax + 1, stride))
    return [(cx, ry) for ry in vals for cx in vals]


def _offsets_train(n: int, omax: int, seed_key: str):
    """``n`` random integer (cx, ry) offsets in [0, omax]^2, seeded per pair."""
    rng = random.Random(seed_key)
    return [(rng.randint(0, omax), rng.randint(0, omax)) for _ in range(n)]


def _sar_template_png(sar_arr: np.ndarray, cx: int, ry: int, t: int) -> bytes:
    """Crop a t×t grayscale template at top-left (ry, cx) and PNG-encode."""
    crop = sar_arr[ry:ry + t, cx:cx + t]
    buf = io.BytesIO()
    # 2-D uint8 array -> Pillow infers mode 'L' (grayscale) on its own.
    Image.fromarray(crop).save(buf, format="PNG")
    return buf.getvalue()


def process_chunk(task) -> Tuple[int, int, list]:
    """Worker: write the shard-set for this worker's slice of pairs.

    Materialises the full ordered (pair, offset) sample plan, slices it into
    ``samples_per_shard`` shards with deterministic boundaries, then writes
    each shard. Resumable: a finished shard has a sidecar ``.done`` JSON with
    its count; on re-run such shards are skipped without re-emitting samples.
    返回 (n_samples_written, n_shards, manifest_rows)。
    """
    (wid, pair_slice, cfg) = task
    split = cfg["split"]
    t = cfg["sar_size"]
    spsh = cfg["samples_per_shard"]
    dst_dir = cfg["dst_dir"]
    os.makedirs(dst_dir, exist_ok=True)

    # 1) Build the ordered sample plan: list of (pair_index, cx, ry).
    #    物化样本计划（轻量元组，确定顺序 -> 确定 shard 边界）。
    plan = []
    for pi, (season, scene_id, pid, _sp, _op) in enumerate(pair_slice):
        if split == "test":
            offs = _offsets_test(cfg["test_stride"], cfg["offset_max"])
        else:
            offs = _offsets_train(cfg["train_offsets"], cfg["offset_max"],
                                  f'{cfg["seed"]}:{season}:{scene_id}:{pid}')
        for (cx, ry) in offs:
            plan.append((pi, cx, ry))

    def _add(tar, key, name, data: bytes):
        info = tarfile.TarInfo(f"{key}.{name}")
        info.size = len(data)
        info.mtime = 0  # fixed mtime -> byte-stable tars
        tar.addfile(info, io.BytesIO(data))

    # Single-entry source cache. The sample plan is grouped by pair and
    # monotonic in ``pi``, so each s1/s2 is read exactly once while bounding
    # memory to one pair (vs accumulating all of a worker's pairs).
    # 单条源图缓存：plan 按 pair 分组且 pi 单调，每张源图只读一次，内存只占一对。
    cache = {"pi": -1, "data": None}

    def _get_src(pi):
        if cache["pi"] != pi:
            _se, _sc, _pid, sar_path, opt_path = pair_slice[pi]
            cache["pi"] = pi
            cache["data"] = (np.asarray(Image.open(sar_path).convert("L")),
                             open(opt_path, "rb").read())
        return cache["data"]

    manifest = []
    n_written = 0
    n_shards = (len(plan) + spsh - 1) // spsh
    for s in range(n_shards):
        path = os.path.join(dst_dir, f"shard-{split}-{wid:03d}-{s:06d}.tar")
        done = path + ".done"
        chunk = plan[s * spsh:(s + 1) * spsh]
        if os.path.isfile(done):
            try:
                cnt = json.load(open(done))["count"]
                manifest.append((path, cnt))
                continue  # already complete; skip
            except Exception:
                pass
        tar = tarfile.open(path, "w")
        count = 0
        for (pi, cx, ry) in chunk:
            try:
                sar_arr, opt_bytes = _get_src(pi)
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"[w{wid}] skip pair {pi}: {exc}\n")
                continue
            key = f"{count:08d}"
            _add(tar, key, "sar.png", _sar_template_png(sar_arr, cx, ry, t))
            _add(tar, key, "rgb.png", opt_bytes)
            _add(tar, key, "label.cls", f"{cx} {ry}".encode())
            count += 1
        tar.close()
        json.dump({"count": count}, open(done, "w"))
        manifest.append((path, count))
        n_written += count
    return n_written, n_shards, manifest


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root",
                    default="/data1/datasets/Template_matching/SEN1-2")
    ap.add_argument("--dst_root", default="/data4/datasets")
    ap.add_argument("--patch_dir_name", default="SEN_opt256_sar192")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--sar_size", type=int, default=192)
    ap.add_argument("--rgb_size", type=int, default=256)
    ap.add_argument("--offset_max", type=int, default=64)  # rgb - sar
    ap.add_argument("--test_stride", type=int, default=4)
    ap.add_argument("--test_per_season", type=int, default=1935)  # 6450*0.3
    ap.add_argument("--train_offsets", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples_per_shard", type=int, default=32000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_pairs", type=int, default=0,
                    help="0=all; >0 limits pairs (smoke test)")
    args = ap.parse_args()

    assert args.offset_max == args.rgb_size - args.sar_size, \
        "offset_max must equal rgb_size - sar_size"

    t0 = time.time()
    print(f"[enum] scanning {args.src_root} ...", flush=True)
    pairs = enumerate_pairs(args.src_root)
    print(f"[enum] {len(pairs)} total pairs in {time.time()-t0:.1f}s",
          flush=True)

    train, test = split_pairs(pairs, args.test_per_season, args.seed)
    chosen = train if args.split == "train" else test
    print(f"[split] train={len(train)} test={len(test)} "
          f"-> using {args.split}={len(chosen)}", flush=True)

    # persist the split for reproducibility / 持久化 split
    dst_shard_dir = os.path.join(
        args.dst_root, f"TM_shards_opt256_sar192_{args.split}")
    os.makedirs(dst_shard_dir, exist_ok=True)
    split_meta = os.path.join(args.dst_root, args.patch_dir_name)
    os.makedirs(split_meta, exist_ok=True)
    with open(os.path.join(split_meta, f"{args.split}_pairs.txt"), "w") as f:
        for p in chosen:
            f.write("\t".join(map(str, p)) + "\n")

    if args.max_pairs > 0:
        chosen = chosen[:args.max_pairs]
        print(f"[smoke] limited to {len(chosen)} pairs", flush=True)

    cfg = dict(
        split=args.split, sar_size=args.sar_size, rgb_size=args.rgb_size,
        offset_max=args.offset_max, test_stride=args.test_stride,
        train_offsets=args.train_offsets, seed=args.seed,
        samples_per_shard=args.samples_per_shard, dst_dir=dst_shard_dir,
    )

    # split pairs into ``workers`` contiguous chunks / 分块给 worker
    W = max(1, args.workers)
    chunks = [chosen[i::W] for i in range(W)]  # round-robin = balanced
    tasks = [(wid, chunks[wid], cfg) for wid in range(W) if chunks[wid]]

    print(f"[run] {len(tasks)} workers, ~{len(chosen)//max(1,len(tasks))} "
          f"pairs/worker", flush=True)
    t1 = time.time()
    if W == 1:
        results = [process_chunk(tasks[0])]
    else:
        with Pool(len(tasks)) as pool:
            results = pool.map(process_chunk, tasks)

    total_samples = sum(r[0] for r in results)
    total_shards = sum(r[1] for r in results)
    all_rows = [row for r in results for row in r[2]]
    all_rows.sort()

    # write shard manifest CSV (header: path,file_num) / 写 shard 清单
    csv_path = os.path.join(dst_shard_dir, "shard_path.csv")
    with open(csv_path, "w") as f:
        f.write("path,file_num\n")
        for path, count in all_rows:
            if count > 0:
                f.write(f"{path},{count}\n")

    dt = time.time() - t1
    print(f"[done] split={args.split} samples={total_samples} "
          f"shards={total_shards} in {dt/60:.1f} min", flush=True)
    print(f"[done] manifest: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
