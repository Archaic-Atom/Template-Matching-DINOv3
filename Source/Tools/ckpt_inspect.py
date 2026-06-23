#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect a checkpoint's matching-head / proj dims to detect dim128 vs dim256.

探测 ckpt 的匹配头维度，判断 dim128 还是 dim256，以及是否含 JBU 组件。
"""
import sys
import torch


def find_state(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "net"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # maybe already a state dict (values are tensors)
        if all(torch.is_tensor(v) for v in list(obj.values())[:5]):
            return obj
    return obj


def main():
    path = sys.argv[1]
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = find_state(obj)
    keys = list(sd.keys())
    print(f"== {path} ==")
    print(f"total keys: {len(keys)}")
    has_match = [k for k in keys if "matching" in k]
    has_jbu = [k for k in keys if "jbu" in k.lower() or "JBU" in k]
    has_sar = [k for k in keys if k.startswith("sar_encoder") or "sar_encoder" in k]
    print(f"matching.* keys: {len(has_match)}  jbu keys: {len(has_jbu)}  sar_encoder keys: {len(has_sar)}")
    # print a few matching proj shapes to read the dim
    shown = 0
    for k in keys:
        if "matching" in k and ("proj" in k or "fine" in k or "head" in k) and k.endswith("weight"):
            v = sd[k]
            if torch.is_tensor(v):
                print(f"  {k}: {tuple(v.shape)}")
                shown += 1
        if shown >= 12:
            break
    # sample any 10 keys for orientation
    print("sample keys:", keys[:6])


if __name__ == "__main__":
    main()
