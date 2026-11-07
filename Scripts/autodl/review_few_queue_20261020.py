"""Continue with strict low-pair experiments after the matched module queue.

匹配模块队列释放GPU后，继续原始配对数据受限的Stage1/Stage2实验。
"""

from __future__ import annotations

import concurrent.futures
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import review_queue_20261020 as queue


PARENT = Path('/root/autodl-tmp/setup/tmm_review_20261020')
WORK = PARENT / 'few_runs'
PAIR_DATA = PARENT / 'few_pairs'
REPO = queue.REPO


def environment(gpu: int, job: str) -> dict:
    """Return the paired numerical environment. 返回配对数值环境。

    Args:
        gpu: Physical GPU. 物理GPU。
        job: Job name. 任务名。

    Returns:
        Controlled environment. 受控环境。
    """
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), TM_SEED='2028', TM_SEED_FAST='1',
               TM_CUDNN_BENCHMARK_LIMIT='10', TM_CORRELATION_CHANNELS_LAST='1',
               TM_REVIEW_OVERRIDE=str(PARENT / 'configs/standard.json'),
               TM_REVIEW_RECORD=str(WORK / 'jobs' / job),
               TM_REVIEW_SMOKE_UPDATES='0', OMP_NUM_THREADS='8',
               MALLOC_ARENA_MAX='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               JF_PROGRESS_COLUMNS='250', TMPDIR='/root/autodl-tmp/tmp',
               TORCH_HOME='/root/autodl-tmp/cache/torch')
    return env


def train(job: str, count: int, model: str, gpu: int, initial: Path,
          stage2: bool, epochs: int, lr: float) -> Path:
    """Train with a fixed virtual epoch. 用相同虚拟epoch训练。

    Args:
        job: Job name. 任务名。
        count: Original pair count. 原始图像对数。
        model: Existing model registry key. 现有模型注册键。
        gpu: Physical GPU. 物理GPU。
        initial: Initial checkpoint. 初始权重。
        stage2: Whether this is matching. 是否为匹配训练。
        epochs: Fixed virtual epochs. 固定虚拟epoch数。
        lr: Registered learning rate. 已登记学习率。

    Returns:
        Final checkpoint. 最终检查点。
    """
    target = WORK / 'jobs' / job
    directory = target / 'checkpoint'
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f'model_epoch_{epochs - 1}.pth'
    if final.exists():
        return final
    if queue.shutil.disk_usage(WORK).free < 20 * 2**30:
        raise RuntimeError('Data-disk guard: less than 20 GiB free')
    if not (directory / 'checkpoint.list').exists():
        (directory / 'review_init.pth').symlink_to(initial)
        (directory / 'checkpoint.list').write_text(
            'last model name:review_init.pth\nreview_init.pth\n')
    command = [queue.PYTHON, '-u', str(queue.ENTRY), '--root_path', str(REPO),
               '--mode', 'train', '--dataset', 'tmdataset', '--modelName', model,
               '--is_stage_2', str(stage2), '--dist', 'False', '--gpu', '1',
               '--port', str(9510 + gpu), '--sar_size', '192',
               '--matching_dim', '256', '--fine_dim', '32', '--fine_margin', '24',
               '--sampleNum', '1', '--auto_save_num', '1',
               '--dataloaderNum', '2', '--dataloaderType', 'mapstyle',
               '--trainListPath', str(PAIR_DATA / f'train_{count}_virtual.csv'),
               '--imgNum', '48000', '--maxEpochs', str(epochs),
               '--batchSize', '8', '--lr', str(lr), '--modelDir', str(directory),
               '--log', str(target / 'train_jf'),
               '--outputDir', str(target / 'train_output'),
               '--resultImgDir', str(target / 'images')]
    queue.execute(job, 'train', command, environment(gpu, job))
    assert final.is_file()
    return final


def evaluate(job: str, count: int, arm: str, gpu: int, checkpoint: Path) -> None:
    """Validate full canonical evaluation. 核验完整标准评测。

    Args:
        job: Job name. 任务名。
        count: Training pairs. 训练图像对。
        arm: Random or S1. 随机或预对齐。
        gpu: Physical GPU. 物理GPU。
        checkpoint: Final matching checkpoint. 最终匹配检查点。
    """
    target = WORK / 'jobs' / job
    output = target / 'evaluation'
    output.mkdir(exist_ok=True)
    command = [queue.PYTHON, '-u', str(queue.ENTRY), '--root_path', str(REPO),
               '--mode', 'test', '--dataset', 'tmdataset', '--modelName', 'TM_Model_G',
               '--is_stage_2', 'True', '--dist', 'False', '--gpu', '1',
               '--port', str(9520 + gpu), '--sar_size', '192',
               '--matching_dim', '256', '--sampleNum', '1', '--lr', '0.001',
               '--auto_save_num', '1', '--dataloaderNum', '2',
               '--trainListPath', str(queue.EVAL), '--imgNum', '20000',
               '--maxEpochs', '1', '--batchSize', '2',
               '--modelDir', str(checkpoint), '--log', str(target / 'test_jf'),
               '--outputDir', str(output), '--resultImgDir', str(target / 'eval_images')]
    files = list(output.glob('test_tmdataset_*.csv'))
    if not files:
        queue.execute(job, 'test', command, environment(gpu, job))
        files = list(output.glob('test_tmdataset_*.csv'))
    assert len(files) == 1
    basis, metrics = queue.predictions(files[0])
    assert basis == queue.REFERENCE
    result = {'job': job, 'pairs': count, 'arm': arm, 'seed': 2028,
              'n': 20000, 'checkpoint': str(checkpoint),
              'checkpoint_sha256': queue.sha256(checkpoint),
              'prediction': str(files[0]), 'prediction_sha256': queue.sha256(files[0]),
              'status': 'single_seed_strict_pair_budget', **metrics}
    with queue.LOCK:
        queue.STATE['jobs'][job].update(phase='complete', result=result,
                                        completed_at=queue.now())
        results = [r['result'] for r in queue.STATE['jobs'].values() if 'result' in r]
        with (WORK / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result))
            writer.writeheader()
            writer.writerows(results)
        queue.save_state()


def run_group(count: int, gpu: int) -> None:
    """Use the same scarce pairs in both stages. 两个阶段限制在同一小子集。

    Args:
        count: Original pair count. 原始图像对数。
        gpu: Physical GPU. 物理GPU。
    """
    import torch

    source = PARENT / 'initialization/standard_random.pth'
    original = torch.load(source, map_location='cpu', weights_only=False)['model_0']
    sar_keys = [key for key in original if key.startswith('sar_encoder.')]
    initial = WORK / f'stage1_random_sar_{count}.pth'
    if not initial.exists():
        torch.save({'epoch': -1, 'model_0': {k: original[k] for k in sar_keys}}, initial)
    stage1 = f'few_{count}_stage1'
    checkpoint = train(stage1, count, 'TM_Model_F', gpu, initial, False, 10, 0.005)
    aligned = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_0']
    state = {k: v.clone() for k, v in original.items()}
    for key in sar_keys:
        assert key in aligned and aligned[key].shape == state[key].shape
        assert torch.isfinite(aligned[key]).all()
        state[key] = aligned[key].clone()
    merged = WORK / f'stage2_s1_init_{count}.pth'
    torch.save({'epoch': -1, 'model_0': state}, merged)
    with queue.LOCK:
        queue.STATE['jobs'][stage1].update(
            phase='complete', artifact_kind='initialization_only',
            checkpoint=str(checkpoint), checkpoint_sha256=queue.sha256(checkpoint))
        queue.save_state()
    for arm, start in [('rtr', source), ('s1tr', merged)]:
        job = f'few_{count}_{arm}'
        final = train(job, count, 'TM_Model_G', gpu, start, True, 5, 0.001)
        evaluate(job, count, arm, gpu, final)


def main() -> None:
    """Wait for the first phase, then run the finite low-data continuation.

    等待第一阶段队列释放后，执行有限的低数据量接续任务。
    """
    WORK.mkdir(exist_ok=True)
    wait_record = {'pid': os.getpid(), 'phase': 'WAIT_MATCHED_QUEUE_AND_DATA',
                   'registered_pair_counts': [100, 500, 1000],
                   'stage1_epochs': 10, 'stage2_epochs': 5,
                   'virtual_epoch_samples': 48000, 'batch': 8,
                   'data_scope': 'Both Stage1 and Stage2 restricted to the same pairs',
                   'updated_at': queue.now()}
    (WORK / 'wait_status.json').write_text(json.dumps(wait_record, indent=2) + '\n')
    # The same file lock prevents overlap during the GPU lease handoff.
    # 使用同一文件锁，避免交接GPU租约时与第一阶段重叠。
    handle = (PARENT / 'controller.lock').open('a')
    while True:
        ready = ((PARENT / 'status.json').exists()
                 and json.loads((PARENT / 'status.json').read_text())['phase']
                 in ('COMPLETE', 'COMPLETE_WITH_FAILURES')
                 and (PAIR_DATA / 'verification.json').exists())
        if ready:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
        time.sleep(30)
    dataset = json.loads((PAIR_DATA / 'verification.json').read_text())
    assert dataset['status'] == 'passed'
    for row in dataset['nested_datasets'].values():
        assert queue.sha256(Path(row['list'])) == row['list_sha256']
    values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    assert all(int(v) < 1024 for v in values.splitlines())
    old = json.loads(Path('/root/autodl-tmp/setup/r1_followup_20261011/job_results.json').read_text())
    queue.REFERENCE, _ = queue.predictions(Path(next(iter(old.values()))['prediction']))
    queue.WORK = WORK
    queue.STATE.update(controller_pid=os.getpid(), phase='RUNNING',
                       started_at=queue.now(), children={}, jobs={})
    for count in (100, 500, 1000):
        for arm in ('stage1', 'rtr', 's1tr'):
            queue.STATE['jobs'][f'few_{count}_{arm}'] = {'phase': 'queued',
                                                       'reviewer': 'R1-1/R1-6'}
    queue.save_state()
    pending = iter((100, 500, 1000))

    def worker(gpu: int) -> None:
        """Run one dataset budget at a time. 每次完成一档数据预算。"""
        while True:
            with queue.LOCK:
                count = next(pending, None)
            if count is None:
                return
            try:
                run_group(count, gpu)
            except Exception as error:
                with queue.LOCK:
                    queue.STATE.setdefault('failed_groups', {})[str(count)] = repr(error)
                    queue.save_state()
                print(f'FAILED PAIR GROUP {count}: {error!r}', flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, (0, 1)))
    queue.STATE['phase'] = ('COMPLETE' if all(r['phase'] == 'complete'
                                            for r in queue.STATE['jobs'].values())
                            else 'COMPLETE_WITH_FAILURES')
    queue.save_state()
    lease = json.loads(queue.LEASE.read_text())
    lease.update(state='released', released_at=queue.now())
    queue.LEASE.write_text(json.dumps(lease, indent=2) + '\n')


if __name__ == '__main__':
    main()
