"""Require stable checkpoints and four complete evaluation arms.

要求权重保存稳定，并确认四个评测臂全部完成。
"""

import argparse
import csv
import time
from pathlib import Path


ROOT = Path('/root/autodl-tmp/repos/005_Template-Matching-DINOv3')
PREFIXES = {
    's1tr': 'Checkpoint_G_s1tr_256_s',
    'rtr': 'Checkpoint_G_rtr_256_s',
    's1fr': 'Checkpoint_F_s1fr_256_s',
    'rfr': 'Checkpoint_F_rfr_256_s',
}


def wait_checkpoints(seed: int) -> None:
    """Wait for all final checkpoints to age past the flushing guard.

    等待全部末代权重超过文件刷新保护时间。

    Args:
        seed: Training seed. 训练随机种子。

    Raises:
        RuntimeError: A checkpoint is missing, empty or keeps changing.
            权重缺失、为空或持续变动。
    """
    deadline = time.monotonic() + 300
    paths = [ROOT / f'{prefix}{seed}/model_epoch_4.pth'
             for prefix in PREFIXES.values()]
    while True:
        stats = [path.stat() for path in paths]
        if any(stat.st_size == 0 for stat in stats):
            raise RuntimeError('Final checkpoint is empty')
        youngest_age = min(time.time() - stat.st_mtime for stat in stats)
        if youngest_age >= 125:
            before = [(stat.st_size, stat.st_mtime_ns) for stat in stats]
            time.sleep(2)
            after = [(path.stat().st_size, path.stat().st_mtime_ns)
                     for path in paths]
            if before == after:
                print(f'CHECKPOINTS_STABLE seed={seed}', flush=True)
                return
        if time.monotonic() >= deadline:
            raise RuntimeError('Final checkpoints did not stabilize in 300s')
        print(f'WAIT_CHECKPOINTS seed={seed} youngest_age={youngest_age:.1f}s',
              flush=True)
        time.sleep(min(30, max(2, 126 - youngest_age)))


def verify_outputs(seed: int) -> None:
    """Reject incomplete replicates before the controller advances.

    控制器进入下一种子前，拒绝不完整的重复实验。

    Args:
        seed: Training seed. 训练随机种子。

    Raises:
        RuntimeError: An arm, sample index or shared basis is incomplete.
            臂、样本索引或共享评测口径不完整。
    """
    with (ROOT / 'eval_2x2_index.csv').open(newline='') as stream:
        records = [row for row in csv.DictReader(stream)
                   if row['seed'] == str(seed)]
    if len(records) != 4 or {row['arm'] for row in records} != set(PREFIXES):
        raise RuntimeError(f'INCOMPLETE_EVAL seed={seed}: expected four arms')
    reference = None
    for record in records:
        if int(record['eval_n']) != 20000:
            raise RuntimeError('Evaluation manifest size is not 20000')
        if record['ckpt'] != f"{PREFIXES[record['arm']]}{seed}/model_epoch_4.pth":
            raise RuntimeError('Wrong checkpoint in evaluation index')
        with (ROOT / record['pred_csv']).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 20000 or {int(row['index']) for row in rows} != set(range(20000)):
            raise RuntimeError('Prediction CSV is incomplete')
        rows.sort(key=lambda row: int(row['index']))
        basis = [(row['sar_name'], row['rgb_name'], row['gt_x'], row['gt_y'])
                 for row in rows]
        if reference is None:
            reference = basis
        elif reference != basis:
            raise RuntimeError('Evaluation bases differ between arms')
    print(f'FOUR_ARM_EVAL_VERIFIED seed={seed} n=20000', flush=True)


def main() -> None:
    """Run a checkpoint or output guard. 执行权重或输出检查。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('wait', 'verify'))
    parser.add_argument('seed', type=int)
    args = parser.parse_args()
    if args.mode == 'wait':
        wait_checkpoints(args.seed)
    else:
        verify_outputs(args.seed)


if __name__ == '__main__':
    main()
