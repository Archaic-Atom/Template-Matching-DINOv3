# -*- coding: utf-8 -*-
"""Strip Stage-1 optimizer state when bootstrapping Stage 2 from Stage 1.

Stage 1 trains the SAR encoder with SGD. Stage 2 trains the matching head
with AdamW (and freezes the encoder). JackFramework's ModelSaver.load_opt
unconditionally calls opt.load_state_dict(checkpoint[key]) — so loading
Stage-1's SGD state into Stage-2's AdamW raises ValueError on optimizer
type mismatch.

Fix: produce a "clean" ckpt that has only model_0 (the SAR encoder weights)
and sets epoch=-1 so JackFramework starts Stage 2 from epoch 0 fresh.

Stage 1 用 SGD、Stage 2 用 AdamW，二者 state_dict 结构不兼容。本脚本剥掉
opt_0 字段并把 epoch 置 -1，让 JackFramework 把它当成"刚开始"的初始化点。

Usage:
    python Source/Tools/strip_stage1_opt.py <src.pth> <dst.pth>
"""
import sys

import torch


def main() -> None:
    """CLI: read src ckpt, drop opt_*, set epoch=-1, save to dst."""
    if len(sys.argv) != 3:
        print('usage: strip_stage1_opt.py <src.pth> <dst.pth>', file=sys.stderr)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]

    d = torch.load(src, map_location='cpu', weights_only=False)
    kept = {k: v for k, v in d.items() if not k.startswith('opt_')}
    kept['epoch'] = -1  # JackFramework: start_epoch = ckpt['epoch']+1 = 0
    torch.save(kept, dst)
    dropped = sorted(set(d.keys()) - set(kept.keys()))
    print(f"stripped {src} -> {dst}")
    print(f"  kept   : {sorted(kept.keys())}")
    print(f"  dropped: {dropped}")
    print(f"  epoch reset to -1 (Stage 2 will start from epoch 0)")


if __name__ == '__main__':
    main()
