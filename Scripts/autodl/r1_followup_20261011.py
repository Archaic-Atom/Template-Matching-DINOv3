"""Continue the approved TMM experiments after the final four-arm evaluation.

最终四臂评测完成后，继续已批准的 TMM 实验。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import threading
import time


ARMS = ('s1tr', 'rtr', 's1fr', 'rfr')
COLUMNS = ('index', 'sar_name', 'rgb_name', 'gt_x', 'gt_y',
           'pred_x', 'pred_y', 'score')


def now() -> str:
    """Return an ISO timestamp. 返回 ISO 时间。"""
    return datetime.datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    """Return a streaming file digest. 返回流式文件校验值。"""
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_prediction(path: Path, expected_n: int = 20000) -> dict:
    """Validate a prediction file and calculate canonical metrics.

    核验预测文件，并计算标准指标。

    Args:
        path: Prediction CSV. 预测 CSV。
        expected_n: Required observations. 要求的样本数。

    Returns:
        Basis, metrics and provenance. 样本口径、指标和来源。

    Raises:
        ValueError: Output is incomplete or invalid. 输出不完整或无效。
    """
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f'Wrong schema: {path}')
        rows = list(reader)
    if len(rows) != expected_n:
        raise ValueError(f'Incomplete CSV: {path}: {len(rows)}/{expected_n}')
    indices = [int(row['index']) for row in rows]
    if set(indices) != set(range(expected_n)):
        raise ValueError(f'Missing/duplicate indices: {path}')
    rows.sort(key=lambda row: int(row['index']))
    basis = [(row['sar_name'], row['rgb_name'], row['gt_x'], row['gt_y'])
             for row in rows]
    errors = []
    for row in rows:
        values = [float(row[key]) for key in
                  ('gt_x', 'gt_y', 'pred_x', 'pred_y', 'score')]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f'Nonfinite coordinates: {path}')
        errors.append(math.hypot(values[2] - values[0], values[3] - values[1]))
    metrics = {f'cmr{threshold}': 100 * sum(
        error <= threshold for error in errors) / expected_n
        for threshold in (1, 3, 5)}
    metrics['mee_px'] = sum(errors) / expected_n
    metrics['gross_ge20_pct'] = 100 * sum(
        error >= 20 for error in errors) / expected_n
    return {'basis': basis, 'metrics': metrics, 'n': expected_n,
            'prediction': str(path), 'prediction_sha256': sha256(path)}


def main_replicate_ready(repo: Path, seed: int,
                         expected_n: int = 20000) -> bool:
    """Require four complete, matched evaluation outputs.

    要求四臂输出完整且样本口径一致。

    Args:
        repo: Repository path. 仓库路径。
        seed: Main training seed. 主训练种子。
        expected_n: Required row count. 要求行数。

    Returns:
        Whether the replicate is ready. 重复实验是否完成。
    """
    try:
        with (repo / 'eval_2x2_index.csv').open(newline='') as handle:
            rows = [row for row in csv.DictReader(handle)
                    if row['seed'] == str(seed)]
        if len(rows) != 4 or {row['arm'] for row in rows} != set(ARMS):
            return False
        reference = None
        for row in rows:
            checkpoint = repo / row['ckpt']
            if checkpoint.name != 'model_epoch_4.pth':
                return False
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                return False
            record = read_prediction(repo / row['pred_csv'], expected_n)
            if reference is None:
                reference = record['basis']
            elif record['basis'] != reference:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        return False


class Supervisor:
    """Own the finite, approved follow-up queue. 管理有限的已批准后续队列。"""

    def __init__(self, repo: Path, work: Path, poll: int = 60) -> None:
        """Initialize the controller. 初始化控制器。"""
        self.repo = repo
        self.work = work
        self.poll = poll
        self.seed = 2028
        self.python = Path('/root/autodl-tmp/conda/envs/tm-dinov3/bin/python')
        self.campaign = Path('/root/autodl-tmp/setup/campaign_20260925')
        self.eval_csv = self.campaign / 'eval20k.csv'
        self.shared = repo / 'Checkpoint_2x2_shared_init_s2028/model_stage1_init.pth'
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / 'logs').mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.state = {'controller_pid': os.getpid(), 'started_at': now(),
                      'phase': 'PREPARING', 'completed_jobs': [], 'children': {}}
        self.results_path = self.work / 'job_results.json'
        self.results = (json.loads(self.results_path.read_text())
                        if self.results_path.exists() else {})
        # Retain completed evaluations when resuming after a restart.
        # 重启恢复时保留已经完成的评测状态。
        self.state['completed_jobs'] = list(self.results)
        self.reference = None

    def set_state(self, phase: str, **fields: object) -> None:
        """Atomically retain liveness and phase evidence.

        原子保存存活和阶段证据。
        """
        with self.lock:
            self.state.update(phase=phase, checked_at=now(), **fields)
            path = self.work / 'status.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.state, indent=2) + '\n')
            temporary.replace(path)
        print(json.dumps({'phase': phase, 'at': now(), **fields}), flush=True)

    def env(self, seed: int) -> dict[str, str]:
        """Preserve the existing numerical environment.

        保持现有数值环境。
        """
        env = os.environ.copy()
        env.update(TM_SEED=str(seed), TM_SEED_FAST='1',
                   TM_CUDNN_BENCHMARK_LIMIT='10',
                   TM_CORRELATION_CHANNELS_LAST='1', MALLOC_ARENA_MAX='2',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   JF_PROGRESS_COLUMNS='400')
        env.pop('TORCH_CUDNN_V8_API_DISABLED', None)
        return env

    def plan(self) -> dict:
        """Prepare nested manifests before looking at budget outcomes.

        在读取预算结果前准备嵌套清单。
        """
        parent = Path('/root/autodl-tmp/setup/shard_train_256_subset40.csv')
        with parent.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 42 or any(int(row['file_num']) != 32000 for row in rows):
            raise ValueError('Expected the verified 42-by-32000 parent pool')
        order = list(range(42))
        random.Random(20261011).shuffle(order)
        manifests = {}
        for budget in (10, 20):
            selected = [rows[index] for index in sorted(order[:budget])]
            path = self.work / f'shard_train_sub{budget}_s2028.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=('path', 'file_num'))
                writer.writeheader()
                writer.writerows(selected)
            if not all(Path(row['path']).is_file() for row in selected):
                raise FileNotFoundError(f'Missing shard in {path}')
            manifests[budget] = {'path': str(path), 'sha256': sha256(path),
                                 'samples': 32000 * budget,
                                 'parent_indices': sorted(order[:budget])}
        plan = {'seed': self.seed, 'selection_seed': 20261011,
                'parent_manifest': str(parent), 'parent_sha256': sha256(parent),
                'shared_init': str(self.shared), 'shared_sha256': sha256(self.shared),
                'eval_csv': str(self.eval_csv), 'eval_sha256': sha256(self.eval_csv),
                'manifests': manifests, 'workers': 5, 'gpus': 2,
                'batch_per_gpu': 32, 'epochs': 5, 'shuffle_buffer': 100000,
                'early_seeds': [2025, 2026, 2027, 2028],
                'early_epochs': [0, 1, 2, 3],
                'boundary': '42-shard reference uses 8 workers; new budgets '
                'use 5. Stage-2 pool and update count vary at fixed epochs. '
                'WDS data order is not claimed bitwise matched.'}
        import importlib.metadata
        import inspect
        import torch
        import webdataset as wds

        initial = torch.load(self.shared, map_location='cpu', weights_only=False)
        if initial.get('epoch') != -1 or any(
                key.startswith('opt_') for key in initial):
            raise ValueError('Shared initialization is not a fresh warm-start')
        plan['versions'] = {'torch': torch.__version__,
                            'webdataset': wds.__version__,
                            'jackframework': importlib.metadata.version('JackFramework')}
        plan['runtime_environment'] = {key: self.env(self.seed)[key] for key in (
            'TM_SEED', 'TM_SEED_FAST', 'TM_CUDNN_BENCHMARK_LIMIT',
            'TM_CORRELATION_CHANNELS_LAST', 'MALLOC_ARENA_MAX',
            'CUBLAS_WORKSPACE_CONFIG', 'PYTORCH_CUDA_ALLOC_CONF')}
        plan['webdataset_epoch_source'] = inspect.getsource(wds.DataPipeline.with_epoch)
        plan['webdataset_shuffle_source'] = inspect.getsource(wds.filters._shuffle)
        plan['code_sha256'] = {}
        for relative in ('Source/main.py', 'Scripts/common.sh',
                         'Source/UserModelImplementation/Dataloaders/_tm_webdataset.py'):
            plan['code_sha256'][relative] = sha256(self.repo / relative)
        (self.work / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        return plan

    def wait_free(self) -> None:
        """Wait for both GPUs without interrupting any other task.

        等待两张GPU空闲，不中断其他任务。
        """
        while True:
            result = subprocess.check_output([
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'], text=True)
            if all(int(value) < 1024 for value in result.splitlines()):
                return
            self.set_state('WAIT_GPU_RELEASE', gpu_memory_mib=result.splitlines())
            time.sleep(self.poll)

    def execute(self, job_id: str, args: list[str], env: dict[str, str],
                expected_load: bool | None = None) -> None:
        """Run a child and preserve its exit and load evidence.

        运行子进程，保留退出码和加载证据。
        """
        log = self.work / 'logs' / f'{job_id}.log'
        with log.open('a') as output:
            child = subprocess.Popen(args, cwd=self.repo, env=env,
                                     stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            with self.lock:
                self.state['children'][job_id] = child.pid
            started = time.monotonic()
            heartbeat = started
            gated = expected_load is None
            while child.poll() is None:
                if not gated:
                    text = log.read_text(errors='replace')
                    if 'Start iteration!' in text:
                        loaded = 'loaded trainable_tail' in text
                        if loaded != expected_load:
                            os.killpg(child.pid, signal.SIGTERM)
                            raise RuntimeError(f'Wrong initialization: {job_id}')
                        gated = True
                    elif time.monotonic() - started > 1800:
                        os.killpg(child.pid, signal.SIGTERM)
                        raise RuntimeError(f'Load gate timeout: {job_id}')
                if time.monotonic() - heartbeat >= self.poll:
                    self.set_state(self.state['phase'])
                    heartbeat = time.monotonic()
                time.sleep(5)
            code = child.returncode
        with self.lock:
            self.state['children'].pop(job_id, None)
        if code != 0:
            raise RuntimeError(f'{job_id} exited {code}; see {log}')
        if expected_load is not None and not gated:
            raise RuntimeError(f'No verified training start: {job_id}')

    def evaluate(self, seed: int, arm: str, epoch: int, gpu: int,
                 budget: int = 42) -> None:
        """Score one retained checkpoint on the same 20k list.

        在同一20k清单上评测一个保留checkpoint。
        """
        if budget == 42:
            directory = self.repo / f'Checkpoint_G_{arm}_256_s{seed}'
        else:
            directory = self.repo / f'Checkpoint_G_r1sub{budget}_{arm}_s{seed}'
        checkpoint = directory / f'model_epoch_{epoch}.pth'
        job_id = f'r1_b{budget}_s{seed}_{arm}_e{epoch}_20k'
        output = self.repo / f'DebugResult_{job_id}'
        if job_id in self.results:
            record = read_prediction(Path(self.results[job_id]['prediction']))
            if record['basis'] != self.reference:
                raise ValueError(f'Saved sample basis changed: {job_id}')
            return
        output.mkdir(exist_ok=True)
        candidates = sorted(output.glob('test_tmdataset_*.csv'))
        if not candidates:
            env = self.env(seed)
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            args = [str(self.python), '-u', 'Source/main.py',
                    '--root_path', str(self.repo), '--mode', 'test',
                    '--dist', 'False', '--gpu', '1', '--port', str(9130 + gpu),
                    '--modelName', 'TM_Model_G', '--dataset', 'tmdataset',
                    '--is_stage_2', 'True', '--trainListPath', str(self.eval_csv),
                    '--imgNum', '20000', '--batchSize', '2', '--sampleNum', '1',
                    '--maxEpochs', '1', '--lr', '1e-3', '--dataloaderNum', '2',
                    '--auto_save_num', '1', '--sar_size', '192',
                    '--matching_dim', '256', '--fine_dim', '32',
                    '--fine_margin', '4', '--log', f'./log_{job_id}/',
                    '--outputDir', str(output),
                    '--resultImgDir', f'./ResultImg_{job_id}/',
                    '--modelDir', str(checkpoint)]
            self.execute(job_id, args, env)
            candidates = sorted(output.glob('test_tmdataset_*.csv'))
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one final CSV: {job_id}')
        record = read_prediction(candidates[0])
        if record.pop('basis') != self.reference:
            raise ValueError(f'Unexpected test samples: {job_id}')
        record.update(seed=seed, arm=arm, epoch=epoch, train_shards=budget,
                      checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                      status='single_seed' if budget != 42 else 'checkpoint_eval_complete')
        with self.lock:
            self.results[job_id] = record
            self.results_path.write_text(json.dumps(self.results, indent=2) + '\n')
            self.state['completed_jobs'].append(job_id)
            metrics_path = self.work / 'r1_checkpoint_metrics.csv'
            with metrics_path.open('w', newline='') as handle:
                fields = ('job', 'seed', 'arm', 'train_shards', 'epoch', 'n',
                          'cmr1', 'cmr3', 'cmr5', 'mee_px', 'gross_ge20_pct',
                          'status', 'prediction', 'prediction_sha256',
                          'checkpoint', 'checkpoint_sha256')
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for key, result in self.results.items():
                    writer.writerow({'job': key, **result['metrics'],
                                     **{field: result[field] for field in fields
                                        if field not in ('job', *result['metrics'])}})

    def train(self, arm: str, budget: int, manifest: dict) -> None:
        """Train one uniquely named small-budget arm.

        训练一个独立命名的小预算臂。
        """
        directory = self.repo / f'Checkpoint_G_r1sub{budget}_{arm}_s{self.seed}'
        if ((directory / 'model_epoch_4.pth').is_file()
                and (directory / 'model_epoch_4.pth').stat().st_size > 0):
            return
        if list(directory.glob('model_epoch_*.pth')):
            raise RuntimeError(f'Partial training needs review: {directory}')
        directory.mkdir(exist_ok=True)
        if arm == 's1tr':
            target = directory / 'model_stage1_init.pth'
            shutil.copy2(self.shared, target)
            if sha256(target) != sha256(self.shared):
                raise RuntimeError('Shared initialization copy differs')
            (directory / 'checkpoint.list').write_text(
                'last model name:model_stage1_init.pth\nmodel_stage1_init.pth\n')
        elif (directory / 'checkpoint.list').exists():
            raise RuntimeError('Random-control directory is not empty')
        job_id = f'train_r1_b{budget}_s{self.seed}_{arm}'
        self.set_state('TRAIN_BUDGET', job=job_id, model_dir=str(directory))
        env = self.env(self.seed)
        env['CUDA_VISIBLE_DEVICES'] = '0,1'
        args = [str(self.python), '-u', 'Source/main.py',
                '--root_path', str(self.repo), '--mode', 'train',
                '--batchSize', '32', '--gpu', '2', '--port', '9141',
                '--dist', 'True', '--trainListPath', manifest['path'],
                '--imgNum', str(manifest['samples']), '--maxEpochs', '5',
                '--auto_save_num', '1', '--dataloaderNum', '5',
                '--dataloaderType', 'iterable', '--sampleNum', '1', '--lr', '1e-3',
                '--modelName', 'TM_Model_G', '--dataset', 'tmdataset',
                '--is_stage_2', 'True', '--sar_size', '192',
                '--matching_dim', '256', '--fine_dim', '32', '--fine_margin', '24',
                '--shuffle_buffer', '100000', '--match_w4', '1.0',
                '--match_w8', '0.5', '--match_sigma4', '1.0', '--match_sigma8', '0.5',
                '--match_primary_stride', '4', '--match_w_fine', '1.0',
                '--match_sigma_fine', '1.0', '--fine_warmup_epochs', '3',
                '--log', f'./log_{directory.name}/',
                '--outputDir', f'./result_{directory.name}/',
                '--resultImgDir', f'./ResultImg_{directory.name}/',
                '--modelDir', str(directory)]
        self.execute(job_id, args, env, expected_load=arm == 's1tr')
        if not (directory / 'model_epoch_4.pth').is_file():
            raise RuntimeError(f'No final checkpoint: {job_id}')
        self.state['completed_jobs'].append(job_id)

    def run(self) -> None:
        """Check completion periodically and run only registered tasks.

        定时检查完成状态，只运行已登记任务。
        """
        plan = self.plan()
        while not main_replicate_ready(self.repo, self.seed):
            self.set_state('WAIT_MAIN_FOUR_ARM_EVAL', seed=self.seed)
            time.sleep(self.poll)
        with (self.repo / 'eval_2x2_index.csv').open(newline='') as handle:
            row = next(row for row in csv.DictReader(handle)
                       if row['seed'] == '2028' and row['arm'] == 's1tr')
        self.reference = read_prediction(self.repo / row['pred_csv'])['basis']
        self.set_state('MAIN_FOUR_ARM_EVAL_VERIFIED', seed=self.seed)
        self.wait_free()
        jobs = [(seed, arm, epoch) for seed in plan['early_seeds']
                for epoch in plan['early_epochs'] for arm in ('s1tr', 'rtr')]
        self.set_state('SINGLE_GPU_EVAL_CHECK', job=jobs[0])
        self.evaluate(*jobs[0], gpu=0)
        for start in range(1, len(jobs), 2):
            self.wait_free()
            pair = jobs[start:start + 2]
            self.set_state('EVAL_EARLY', jobs=pair)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.evaluate, seed, arm, epoch, gpu)
                           for gpu, (seed, arm, epoch) in enumerate(pair)]
                for future in futures:
                    future.result()
        for budget in (10, 20):
            for arm in ('s1tr', 'rtr'):
                self.wait_free()
                self.train(arm, budget, plan['manifests'][budget])
            self.wait_free()
            self.set_state('EVAL_BUDGET', budget=budget)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.evaluate, self.seed, arm, 4, gpu, budget)
                           for gpu, arm in enumerate(('s1tr', 'rtr'))]
                for future in futures:
                    future.result()
        self.wait_free()
        self.set_state('GPU_EXPERIMENTS_COMPLETE', results=len(self.results),
                       next_action='collect_data_then_review_remaining_tasks_before_restart')


def main() -> None:
    """Start one locked controller or print its plan.

    启动单个加锁控制器，或显示计划。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    supervisor = Supervisor(args.repo, args.work, args.poll_seconds)
    if args.dry_run:
        print(json.dumps(supervisor.plan(), indent=2))
        return
    with (args.work / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            supervisor.run()
        except Exception as error:
            supervisor.set_state('FAILED', error=str(error))
            raise


if __name__ == '__main__':
    main()
