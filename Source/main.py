# -*coding: utf-8 -*-
import os
import random

import numpy as np
import torch

import JackFramework as jf
from UserModelImplementation import UserInterface


def _configure_cudnn_search() -> None:
    """Apply an optional runtime-only cuDNN candidate-search limit.

    在新硬件上显式限制 cuDNN 候选搜索；不改变模型、batch 或优化器。
    Run at import time so torch.multiprocessing spawn workers inherit it.
    在导入时执行，确保 spawn 子进程也应用相同设置。

    Raises:
        ValueError: If the supplied limit is not a nonnegative integer.
            当限制不是非负整数时抛出。
    """
    raw = os.environ.get("TM_CUDNN_BENCHMARK_LIMIT")
    if raw is None:
        return
    limit = int(raw)
    if limit < 0:
        raise ValueError("TM_CUDNN_BENCHMARK_LIMIT must be nonnegative")
    torch.backends.cudnn.benchmark_limit = limit
    print(f"[main] cuDNN benchmark candidate limit={limit}", flush=True)
    print("[main] correlation channels_last="
          + str(os.environ.get("TM_CORRELATION_CHANNELS_LAST") == "1"), flush=True)


_configure_cudnn_search()


def _maybe_fix_seed() -> None:
    """Seed every RNG from the ``TM_SEED`` env var for reproducible runs.

    从 ``TM_SEED`` 环境变量固定所有随机源，让测试/评测结果可复现、可对比。

    Only active when ``TM_SEED`` is explicitly set (the test scripts set it),
    so normal training keeps cudnn auto-tuning and is not slowed down.
    仅在显式设置 ``TM_SEED`` 时生效（测试脚本会设它），训练默认不固定，
    保留 cudnn benchmark 的速度。
    """
    seed_env = os.environ.get('TM_SEED')
    if seed_env is None:
        return
    seed = int(seed_env)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN determinism is OPT-OUT via TM_SEED_FAST=1.
    #
    # Default (TM_SEED_FAST unset): deterministic kernels — right for an
    # evaluation you want to reproduce bit for bit.
    #
    # TM_SEED_FAST=1: seed the RNGs but leave cuDNN autotuning ON. Use this for
    # a multi-seed TRAINING campaign, for two reasons. (i) Kernel-selection
    # nondeterminism is part of the run-to-run variance a seed replicate is
    # meant to measure; suppressing it makes the spread look smaller than it is,
    # which is the anti-conservative direction. (ii) It restores the setting the
    # already-published arms ran under (they never set TM_SEED), keeping the new
    # arms comparable with them. Costs nothing; saves the ~15% determinism tax.
    #
    # cuDNN 确定性改为可选退出：TM_SEED_FAST=1。
    # 默认（不设 TM_SEED_FAST）用确定性 kernel，适合要逐位复现的评测。
    # 多种子训练战役请设 TM_SEED_FAST=1：只固定 RNG、保留 cudnn autotune。理由有二：
    # (i) kernel 选择的非确定性本就是种子重复要测的"跨次运行方差"，压掉它会让方差
    # 看起来更小，属于反保守方向；(ii) 已发表各臂从未设 TM_SEED，保留 autotune 才与
    # 它们同配置、可比。顺带省掉约 15% 的确定性开销。
    fast = os.environ.get('TM_SEED_FAST', '') == '1'
    if fast:
        print(f"[main] TM_SEED={seed} — RNGs seeded; cudnn autotune KEPT "
              f"(TM_SEED_FAST=1, published-arm setting).")
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[main] TM_SEED={seed} — RNGs seeded, cudnn set deterministic.")


def main() -> None:
    _maybe_fix_seed()
    app = jf.Application(UserInterface(), "Template Matching model")
    app.start()


# Spawned DDP workers construct models themselves, so seed them as well.
# DDP spawn 子进程独立构造模型，因此也必须固定随机源。
if __name__ == "__mp_main__":
    _maybe_fix_seed()

# execute the main function
if __name__ == "__main__":
    main()
