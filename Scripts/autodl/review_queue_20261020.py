"""Run the finite matched reviewer queue on two GPUs, with final evaluation.

两张GPU分别运行有限的配对审稿任务；每次训练后自动完成标准20k评测。
"""

from __future__ import annotations

import concurrent.futures
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


REPO = Path(__file__).resolve().parents[2]
WORK = Path('/root/autodl-tmp/setup/tmm_review_20261020')
PYTHON = '/root/autodl-tmp/conda/envs/tm-dinov3/bin/python'
ENTRY = REPO / 'Scripts/autodl/review_entry_20261020.py'
TRAIN = Path('/root/autodl-tmp/setup/shard_train_256_subset40.csv')
EVAL = Path('/root/autodl-tmp/setup/campaign_20260925/eval20k.csv')
LEASE = Path('/root/autodl-tmp/gpu_lease.json')
COLUMNS = ('index', 'sar_name', 'rgb_name', 'gt_x', 'gt_y',
           'pred_x', 'pred_y', 'score')
JOBS = [
    ('r1_7_cascade_jbu', 'TM_Model_C', 'cascade_jbu', 'cascade_jbu', 'R1-7'),
    ('r1_7_cascade_bilinear', 'TM_Model_CB', 'cascade_bilinear',
     'cascade_bilinear', 'R1-7'),
    ('shared_standard_random', 'TM_Model_G', 'standard_random', 'standard',
     'R1-7/R1-6'),
    ('shared_standard_s1', 'TM_Model_G', 'standard_s1', 'standard', 'R1-6'),
    ('r2_3_partial_random', 'TM_Model_G', 'standard_random', 'partial', 'R2-3'),
    ('r2_3_partial_s1', 'TM_Model_G', 'standard_s1', 'partial', 'R2-3'),
]
LOCK = threading.Lock()
STATE = {}
REFERENCE = None


def now() -> str:
    """Return the local ISO time. 返回本地ISO时间。"""
    return datetime.datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    """Hash one file. 校验一个文件。

    Args:
        path: File to hash. 文件路径。

    Returns:
        SHA-256 hexadecimal value. SHA-256十六进制值。
    """
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save_state() -> None:
    """Atomically save queue liveness. 原子保存队列状态。"""
    STATE['checked_at'] = now()
    temporary = WORK / 'status.tmp'
    temporary.write_text(json.dumps(STATE, indent=2) + '\n')
    temporary.replace(WORK / 'status.json')
    lease = {'owner': 'tmm_review_20261020', 'state': 'active',
             'pid': os.getpid(), 'gpu_indices': [0, 1], 'work_dir': str(WORK),
             'release_after': 'finite_queue_and_final_evaluations',
             'updated_at': now(), 'children': STATE.get('children', {})}
    lease_tmp = LEASE.with_suffix('.tmp')
    lease_tmp.write_text(json.dumps(lease, indent=2) + '\n')
    lease_tmp.replace(LEASE)


def predictions(path: Path) -> tuple[list, dict]:
    """Validate indices and calculate canonical pixel metrics. 核验并计算像素指标。

    Args:
        path: Canonical prediction CSV. 标准预测CSV。

    Returns:
        Ground-truth basis and metrics. 真值口径与指标。
    """
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == COLUMNS
        rows = list(reader)
    assert len(rows) == 20000
    assert {int(r['index']) for r in rows} == set(range(20000))
    rows.sort(key=lambda r: int(r['index']))
    basis, errors = [], []
    for row in rows:
        values = [float(row[k]) for k in
                  ('gt_x', 'gt_y', 'pred_x', 'pred_y', 'score')]
        assert all(math.isfinite(v) for v in values)
        basis.append((row['sar_name'], row['rgb_name'], row['gt_x'], row['gt_y']))
        errors.append(math.hypot(values[2] - values[0], values[3] - values[1]))
    metrics = {f'cmr{k}': 100 * sum(e <= k for e in errors) / len(errors)
               for k in (1, 3, 5)}
    metrics['mee_px'] = sum(errors) / len(errors)
    metrics['gross_ge20_pct'] = 100 * sum(e >= 20 for e in errors) / len(errors)
    return basis, metrics


def execute(job: str, phase: str, command: list[str], env: dict) -> None:
    """Run one owned child and retain its exit. 运行子进程并保留退出证据。

    Args:
        job: Registered job name. 已登记任务名。
        phase: Train or test. 训练或测试。
        command: Exact argument vector. 精确参数列表。
        env: Controlled environment. 受控环境。
    """
    target = WORK / 'jobs' / job
    target.mkdir(parents=True, exist_ok=True)
    (target / f'{phase}_command.json').write_text(
        json.dumps({'args': command, 'environment': {k: v for k, v in env.items()
                    if k.startswith('TM_') or k in ('CUDA_VISIBLE_DEVICES',
                    'PYTORCH_CUDA_ALLOC_CONF', 'CUBLAS_WORKSPACE_CONFIG')}},
                   indent=2) + '\n')
    with (target / f'{phase}.log').open('a') as log:
        process = subprocess.Popen(command, cwd=REPO, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        with LOCK:
            STATE['children'][job] = process.pid
            STATE['jobs'][job].update(phase=phase, pid=process.pid,
                                      started_at=now())
            save_state()
        while process.poll() is None:
            time.sleep(15)
            with LOCK:
                save_state()
        code = process.returncode
    with LOCK:
        STATE['children'].pop(job, None)
        STATE['jobs'][job][f'{phase}_exit'] = code
        save_state()
    if code != 0:
        raise RuntimeError(f'{job} {phase} exited {code}')


def run_job(record: tuple, gpu: int) -> None:
    """Train and then fully evaluate one condition. 训练并完整评测一个条件。

    Args:
        record: Registered condition. 已登记实验条件。
        gpu: Physical GPU index. 物理GPU编号。
    """
    job, variant, initial_name, config_name, reviewer = record
    target = WORK / 'jobs' / job
    checkpoint_dir = target / 'checkpoint'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial = WORK / 'initialization' / f'{initial_name}.pth'
    final = checkpoint_dir / 'model_epoch_4.pth'
    if not (checkpoint_dir / 'checkpoint.list').exists():
        (checkpoint_dir / 'review_init.pth').symlink_to(initial)
        (checkpoint_dir / 'checkpoint.list').write_text(
            'last model name:review_init.pth\nreview_init.pth\n')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), TM_SEED='2028', TM_SEED_FAST='1',
               TM_CUDNN_BENCHMARK_LIMIT='10', TM_CORRELATION_CHANNELS_LAST='1',
               TM_REVIEW_OVERRIDE=str(WORK / 'configs' / f'{config_name}.json'),
               TM_REVIEW_RECORD=str(target), TM_REVIEW_SMOKE_UPDATES='0',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', MALLOC_ARENA_MAX='2',
               OMP_NUM_THREADS='8', JF_PROGRESS_COLUMNS='250',
               TMPDIR='/root/autodl-tmp/tmp',
               TORCH_HOME='/root/autodl-tmp/cache/torch')
    common = [PYTHON, '-u', str(ENTRY), '--root_path', str(REPO),
              '--dataset', 'tmdataset', '--modelName', variant,
              '--is_stage_2', 'True', '--dist', 'False', '--gpu', '1',
              '--port', str(9400 + gpu), '--sar_size', '192',
              '--matching_dim', '256', '--fine_dim', '32', '--fine_margin', '24',
              '--sampleNum', '1', '--lr', '1e-3', '--auto_save_num', '1',
              '--dataloaderNum', '2']
    if not final.exists():
        if shutil.disk_usage(WORK).free < 20 * 2**30:
            raise RuntimeError('Data-disk guard: less than 20 GiB free')
        train = common + ['--mode', 'train', '--dataloaderType', 'iterable',
                          '--trainListPath', str(TRAIN), '--imgNum', '1344000',
                          '--maxEpochs', '5', '--batchSize', '8',
                          '--shuffle_buffer', '100000',
                          '--modelDir', str(checkpoint_dir),
                          '--log', str(target / 'train_jf'),
                          '--outputDir', str(target / 'train_output'),
                          '--resultImgDir', str(target / 'images')]
        execute(job, 'train', train, env)
    assert final.is_file() and final.stat().st_size > 0
    evaluation = target / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    candidates = list(evaluation.glob('test_tmdataset_*.csv'))
    if not candidates:
        test = common + ['--mode', 'test', '--trainListPath', str(EVAL),
                         '--imgNum', '20000', '--maxEpochs', '1',
                         '--batchSize', '2', '--modelDir', str(final),
                         '--log', str(target / 'test_jf'),
                         '--outputDir', str(evaluation),
                         '--resultImgDir', str(target / 'eval_images')]
        execute(job, 'test', test, env)
        candidates = list(evaluation.glob('test_tmdataset_*.csv'))
    assert len(candidates) == 1
    basis, metrics = predictions(candidates[0])
    assert basis == REFERENCE
    result = {'job': job, 'reviewer': reviewer, 'seed': 2028, 'n': 20000,
              'checkpoint': str(final), 'checkpoint_sha256': sha256(final),
              'prediction': str(candidates[0]),
              'prediction_sha256': sha256(candidates[0]),
              'status': 'single_seed_matched_budget', **metrics}
    with LOCK:
        STATE['jobs'][job].update(phase='complete', completed_at=now(), result=result)
        results = [r['result'] for r in STATE['jobs'].values() if 'result' in r]
        with (WORK / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result))
            writer.writeheader()
            writer.writerows(results)
        save_state()


def main() -> None:
    """Own both GPUs for the finite authorized queue. 管理有限授权队列的两张GPU。"""
    global REFERENCE
    WORK.mkdir(exist_ok=True)
    lock_handle = (WORK / 'controller.lock').open('w')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('cascade_jbu_b8', 'cascade_bilinear_b8'):
        result = json.loads((WORK / 'smoke' / name / 'smoke_result.json').read_text())
        assert result['status'] == 'passed'
    initial = json.loads((WORK / 'initialization/verification.json').read_text())
    assert all(sha256(Path(r['path'])) == r['sha256'] for r in initial.values())
    old_results = json.loads(Path(
        '/root/autodl-tmp/setup/r1_followup_20261011/job_results.json').read_text())
    reference_path = Path(next(iter(old_results.values()))['prediction'])
    REFERENCE, _ = predictions(reference_path)
    (WORK / 'configs/standard.json').write_text('{}\n')
    (WORK / 'configs/partial.json').write_text(json.dumps({
        'training': {'stage_2': {'freeze_sar_modules':
                                ['stem', 'layer1', 'layer2']}}}, indent=2) + '\n')
    plan = {'created_at': now(), 'seed': 2028, 'jobs': JOBS,
            'train_manifest': str(TRAIN), 'train_sha256': sha256(TRAIN),
            'train_samples': 1344000, 'epochs': 5, 'batch_global': 8,
            'workers_per_gpu': 2, 'eval_csv': str(EVAL),
            'eval_sha256': sha256(EVAL), 'reference_prediction': str(reference_path),
            'initialization': initial, 'entry_sha256': sha256(ENTRY),
            'boundary': 'One matched seed; independent process data order is not '
                        'claimed bitwise identical. Do not pool with old n=5 '
                        'runs using different batches. R1-1 pair subsets are '
                        'not in this queue until pair identities are verified.'}
    (WORK / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    previous = (json.loads((WORK / 'status.json').read_text())
                if (WORK / 'status.json').exists() else {})
    STATE.update(controller_pid=os.getpid(), started_at=now(), phase='RUNNING',
                 children={}, jobs=previous.get('jobs', {}))
    for job in JOBS:
        STATE['jobs'].setdefault(job[0], {'phase': 'queued', 'reviewer': job[4]})
    save_state()
    pending = iter(job for job in JOBS
                   if STATE['jobs'][job[0]]['phase'] != 'complete')

    def worker(gpu: int) -> None:
        """Drain registered jobs without changing conditions. 按登记条件领取任务。"""
        while True:
            with LOCK:
                record = next(pending, None)
            if record is None:
                return
            try:
                run_job(record, gpu)
            except Exception as error:
                with LOCK:
                    STATE['jobs'][record[0]].update(
                        phase='failed', error=repr(error), failed_at=now())
                    save_state()
                print(f'FAILED {record[0]}: {error!r}', flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, (0, 1)))
    STATE['phase'] = ('COMPLETE' if all(r['phase'] == 'complete'
                                      for r in STATE['jobs'].values())
                      else 'COMPLETE_WITH_FAILURES')
    save_state()
    lease = json.loads(LEASE.read_text())
    lease.update(state='released', released_at=now())
    LEASE.write_text(json.dumps(lease, indent=2) + '\n')


if __name__ == '__main__':
    main()
