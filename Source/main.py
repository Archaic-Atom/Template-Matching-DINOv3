# -*coding: utf-8 -*-
import os
import random

import numpy as np
import torch

import JackFramework as jf
from UserModelImplementation import UserInterface


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
    # Best-effort determinism for the matching forward pass.
    # 尽量让匹配前向也确定（数据本身已固定）。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[main] TM_SEED={seed} — RNGs seeded, cudnn set deterministic.")


def main() -> None:
    _maybe_fix_seed()
    app = jf.Application(UserInterface(), "Template Matching model")
    app.start()


# execute the main function
if __name__ == "__main__":
    main()
