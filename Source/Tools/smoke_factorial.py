# -*- coding: utf-8 -*-
"""Prove the 2x2 factorial at the parameter level, without training.

在不训练的前提下，从参数层面证明 2x2 的析因性质。

Builds TM_Model_F and TM_Model_G through the real model zoo and reports, for
each, how many SAR-trunk parameters carry ``requires_grad``. The freeze factor
is real if and only if F reports 0 trainable trunk parameters and G reports the
full count, with every other module identical between them.

用真实的 model zoo 分别构建 TM_Model_F 与 TM_Model_G，统计各自 SAR 主干中
``requires_grad`` 的参数量。只有当 F 的主干可训参数为 0、G 为全量，且其余模块
完全一致时，冻结因子才是真的。

Usage / 用法:
    python Source/Tools/smoke_factorial.py
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, 'Source')

import torch  # noqa: E402
from UserModelImplementation.Models import model_zoo  # noqa: E402


def _args() -> object:
    """Minimal arg object the interfaces read at build time.

    构建时接口会读取的最小参数对象。
    """
    return SimpleNamespace(
        # `${root_path}` appears in the configs' weights paths, so the builder
        # requires it even for a build-only smoke test.
        # config 里的权重路径含 `${root_path}`，构建时必须提供，哪怕只是 smoke。
        root_path=os.getcwd(),
        is_stage_2=True, sar_size=192, matching_dim=256, fine_dim=32,
        fine_margin=24, batchSize=1, gpu=1, dist=False, lr=1e-3,
        match_w4=1.0, match_w8=0.5, match_sigma4=1.0, match_sigma8=0.5,
        match_primary_stride=4, match_w_fine=1.0, match_sigma_fine=1.0,
        fine_warmup_epochs=3, modelDir='./', outputDir='./', log='./',
    )


def _trunk_stats(model: torch.nn.Module) -> tuple:
    """Return (trainable, total) parameter counts of the SAR trunk.

    返回 SAR 主干的（可训, 总计）参数量。
    """
    tr = tot = 0
    for name, p in model.named_parameters():
        if 'sar_encoder' in name or 'sar_trunk' in name:
            tot += p.numel()
            if p.requires_grad:
                tr += p.numel()
    return tr, tot


def main() -> int:
    out = {}
    for variant in ('TM_Model_F', 'TM_Model_G'):
        interface = model_zoo(_args(), variant)
        nets = interface.get_model()
        model = nets[0] if isinstance(nets, (list, tuple)) else nets
        tr, tot = _trunk_stats(model)
        names = sorted(n for n, _ in model.named_parameters())
        out[variant] = (tr, tot, names)
        print(f'{variant}: SAR trunk trainable={tr:,} / total={tot:,}')

    f_tr, f_tot, f_names = out['TM_Model_F']
    g_tr, g_tot, g_names = out['TM_Model_G']
    ok = True
    if f_names != g_names:
        print('FAIL: F and G expose DIFFERENT parameter names — not a factorial.')
        only_f = set(f_names) - set(g_names)
        only_g = set(g_names) - set(f_names)
        print(f'  only in F: {sorted(only_f)[:5]}')
        print(f'  only in G: {sorted(only_g)[:5]}')
        ok = False
    else:
        print(f'PASS: identical parameter set ({len(f_names)} tensors).')
    if f_tr != 0:
        print(f'FAIL: TM_Model_F trunk is NOT frozen ({f_tr:,} trainable).')
        ok = False
    if g_tr != g_tot or g_tot == 0:
        print(f'FAIL: TM_Model_G trunk is not fully trainable '
              f'({g_tr:,}/{g_tot:,}).')
        ok = False
    if ok:
        print('SMOKE_FACTORIAL_PASS')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
