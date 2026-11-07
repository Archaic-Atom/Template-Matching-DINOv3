"""Limit only the repair evaluation's CUDA allocator.

仅限制补测进程的 CUDA 分配器，不影响正在运行的训练。
"""

import os

if os.environ.get('TM_EVAL_ALLOC_FRACTION'):
    import torch

    fraction = float(os.environ['TM_EVAL_ALLOC_FRACTION'])
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f'[repair-eval] CUDA allocator fraction={fraction}', flush=True)
