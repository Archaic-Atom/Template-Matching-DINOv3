#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a 2k mapstyle eval set at opt256/sar192 from the TEST split.

从 opt256/sar192 的 TEST 配对（已由 make_opt256_sar192_shards.py 存到
test_pairs.txt）随机抽 2000 个 (pair, offset) 样本，裁出 192 SAR 模板 + 复制
256 optical，写成与 image_test_2k.csv 同格式的 mapstyle CSV，供 start_test
零样本评测 B 模型（128/64 训练）在 256/192 上的表现。

CSV columns（与 _tm_datasets._read_csv 完全一致）:
    SAR_Image, RGB_Image, Label   (Label = "[cx, ry]" = [x=col, y=row], JSON)
"""

import argparse
import json
import os
import random

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_txt",
                    default="/data4/datasets/SEN_opt256_sar192/test_pairs.txt")
    ap.add_argument("--out_dir",
                    default="/data4/datasets/eval2k_opt256_sar192")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--sar_size", type=int, default=192)
    ap.add_argument("--offset_max", type=int, default=64)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    t = args.sar_size
    sar_out = os.path.join(args.out_dir, "sar")
    rgb_out = os.path.join(args.out_dir, "rgb")
    os.makedirs(sar_out, exist_ok=True)
    os.makedirs(rgb_out, exist_ok=True)

    # load test pairs / 读测试配对
    pairs = []
    with open(args.pairs_txt) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 5:
                pairs.append(cols)  # season, scene, pid, sar_path, opt_path
    print(f"[2k] {len(pairs)} test pairs loaded")

    grid = list(range(0, args.offset_max + 1, args.stride))
    rng = random.Random(args.seed)

    rows = []
    rgb_cache = set()  # which pair opticals already copied
    for i in range(args.n):
        season, scene, pid, sar_path, opt_path = rng.choice(pairs)
        cx = rng.choice(grid)
        ry = rng.choice(grid)
        # crop 192 SAR template at (ry, cx) / 裁 SAR 模板
        sar_arr = np.asarray(Image.open(sar_path).convert("L"))
        crop = sar_arr[ry:ry + t, cx:cx + t]
        sar_name = f"{season}_s1_{scene}_p{pid}_{cx}_{ry}.png"
        Image.fromarray(crop).save(os.path.join(sar_out, sar_name))
        # optical 256 copied once per pair (shared) / optical 每对存一次
        rgb_name = f"{season}_s2_{scene}_p{pid}.png"
        rgb_dst = os.path.join(rgb_out, rgb_name)
        if rgb_name not in rgb_cache:
            with open(opt_path, "rb") as src, open(rgb_dst, "wb") as dst:
                dst.write(src.read())
            rgb_cache.add(rgb_name)
        # Label = [x=col=cx, y=row=ry] / 标签
        rows.append((os.path.join(sar_out, sar_name), rgb_dst,
                     json.dumps([cx, ry])))

    csv_path = os.path.join(args.out_dir, "image_test_2k_256.csv")
    with open(csv_path, "w") as f:
        f.write("SAR_Image,RGB_Image,Label\n")
        for sar_p, rgb_p, lab in rows:
            f.write(f'{sar_p},{rgb_p},"{lab}"\n')
    print(f"[2k] wrote {len(rows)} rows -> {csv_path}")
    print(f"[2k] unique opticals copied: {len(rgb_cache)}")


if __name__ == "__main__":
    main()
