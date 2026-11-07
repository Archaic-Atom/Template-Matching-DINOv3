"""Fill free GPUs from the registered queue while preserving live training.

保留正在训练的进程，按物理GPU接续已有小样本队列，统一写入GPU租约。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import review_few_queue_20261020 as few
from review_resume_20261024 import alive, identity


PARENT = few.PARENT
LOCK = threading.RLock()
COUNTS = (100, 500, 1000)
LIVE_JOB = 'r2_3_partial_s1'
CANCELLED = {'shared_standard_random', 'shared_standard_s1'}


def load_primary() -> object:
    """Load primary helpers with separate module globals. 隔离主队列全局状态。

    Returns:
        Independent helper module. 独立的工具模块。
    """
    path = Path(__file__).with_name('review_queue_20261020.py')
    spec = importlib.util.spec_from_file_location('review_primary_overlap', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict) -> None:
    """Replace a JSON record atomically. 原子替换JSON记录。

    Args:
        path: Destination file. 目标文件。
        value: Serializable record. 可序列化记录。
    """
    temporary = path.with_suffix(path.suffix + '.overlap.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def preflight(primary: object) -> dict:
    """Check the exact live process and unused queue before takeover.

    交接前核验运行进程、空闲卡、数据和原等待队列。

    Args:
        primary: Isolated primary helpers. 独立主队列工具。

    Returns:
        Verified identities and immutable source hashes. 已核验身份和源码哈希。
    """
    state = json.loads((PARENT / 'status.json').read_text())
    waiter_state = json.loads((few.WORK / 'wait_status.json').read_text())
    parent = identity(state['controller_pid'])
    waiter = identity(waiter_state['pid'])
    child = identity(state['children'][LIVE_JOB])
    assert state['phase'] == 'RUNNING'
    assert set(state['children']) == {LIVE_JOB}
    assert set(state['cancelled_jobs']) == CANCELLED
    assert state['jobs'][LIVE_JOB]['phase'] == 'train'
    assert all(row['phase'] == 'complete' for name, row in state['jobs'].items()
               if name != LIVE_JOB)
    assert parent and alive(parent)
    assert any(arg.endswith('/review_resume_20261024.py') for arg in parent['args'])
    assert waiter and alive(waiter)
    assert any(arg.endswith('/review_few_queue_20261020.py') for arg in waiter['args'])
    assert waiter_state['phase'] == 'WAIT_MATCHED_QUEUE_AND_DATA'
    assert child and alive(child) and child['ppid'] == parent['pid']
    assert child['session'] == child['pid']
    assert str(PARENT / 'jobs' / LIVE_JOB / 'checkpoint') in child['args']
    assert child['args'][child['args'].index('--mode') + 1] == 'train'
    environment = (Path('/proc') / str(child['pid']) / 'environ').read_bytes()
    assert b'CUDA_VISIBLE_DEVICES=0' in environment.split(b'\0')
    lease = json.loads(primary.LEASE.read_text())
    assert lease['pid'] == parent['pid'] and lease['state'] == 'active'
    assert lease['owner'] == 'tmm_review_20261020'
    assert not (few.WORK / 'status.json').exists(), 'Few queue already started'
    assert not list((few.WORK / 'jobs').glob('*/train_command.json'))
    data = json.loads((few.PAIR_DATA / 'verification.json').read_text())
    assert data['status'] == 'passed'
    for row in data['nested_datasets'].values():
        assert primary.sha256(Path(row['list'])) == row['list_sha256']
    assert primary.shutil.disk_usage(PARENT).free > 20 * 2**30
    gpus = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used',
         '--format=csv,noheader,nounits'], text=True)
    rows = {int(row.split(',')[0]): row.split(',') for row in gpus.splitlines()}
    assert int(rows[1][2]) < 1024
    applications = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
         '--format=csv,noheader'], text=True)
    assert {(int(row.split(',')[0]), row.split(',')[1].strip())
            for row in applications.splitlines()} == {
                (child['pid'], rows[0][1].strip())}
    sources = {name: primary.sha256(Path(__file__).with_name(name)) for name in (
        'review_queue_20261020.py', 'review_resume_20261024.py',
        'review_few_queue_20261020.py', 'review_entry_20261020.py',
        Path(__file__).name)}
    return {'state': state, 'parent': parent, 'waiter': waiter, 'child': child,
            'sources': sources, 'checked_at': primary.now()}


def stop_controller(record: dict) -> None:
    """Stop only an identity-checked controller PID. 仅停止已核对的控制器PID。

    Args:
        record: Verified Linux process identity. 已核验的Linux进程身份。
    """
    assert alive(record), 'Controller identity changed'
    os.kill(record['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 15
    while alive(record) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not alive(record), 'Controller did not stop'


class Coordinator:
    """Own one GPU lease and two isolated queue states. 统一租约与两份独立状态。"""

    def __init__(self, primary: object, verified: dict) -> None:
        """Bind original functions without changing experiment conditions.

        绑定原有函数，保留全部实验条件。

        Args:
            primary: Isolated primary helpers. 独立主队列工具。
            verified: Preflight evidence. 交接前证据。
        """
        self.primary = primary
        self.low = few.queue
        self.verified = verified
        self.pending = iter(COUNTS)
        self.primary.LOCK = self.low.LOCK = LOCK
        self.primary.STATE = copy.deepcopy(verified['state'])
        self.primary.STATE.update(controller_pid=os.getpid(),
                                  overlap_controller=str(Path(__file__)))
        self.low.WORK = few.WORK
        self.low.STATE = {
            'controller_pid': os.getpid(), 'phase': 'RUNNING',
            'started_at': primary.now(), 'children': {},
            'jobs': {f'few_{count}_{arm}': {'phase': 'queued', 'reviewer': 'R1-1/R1-6'}
                     for count in COUNTS for arm in ('stage1', 'rtr', 's1tr')},
            'registered_pair_counts': list(COUNTS), 'seed': 2028,
            'stage1_epochs': 10, 'stage2_epochs': 5,
            'virtual_epoch_samples': 48000, 'batch': 8,
            'data_scope': 'Both Stage1 and Stage2 restricted to the same pairs',
        }
        self.primary.save_state = self.save
        self.low.save_state = self.save

    def save(self, release: bool = False) -> None:
        """Save both states and their union under one lock. 同锁保存两队列和统一租约。

        Args:
            release: Release only after all workers finish. 全部工作结束才释放。
        """
        with LOCK:
            queues = (self.primary, self.low)
            children = {job: pid for module in queues
                        for job, pid in module.STATE.get('children', {}).items()}
            if release:
                assert not children
                assert all(module.STATE['phase'] in (
                    'COMPLETE', 'COMPLETE_WITH_FAILURES') for module in queues)
            for module in queues:
                module.STATE['checked_at'] = self.primary.now()
                write_json(module.WORK / 'status.json', module.STATE)
            lease = {
                'owner': 'tmm_review_20261020',
                'state': 'released' if release else 'active',
                'pid': os.getpid(), 'gpu_indices': [0, 1],
                'work_dir': str(PARENT), 'children': children,
                'queue_statuses': [str(m.WORK / 'status.json') for m in queues],
                'release_after': 'both_queues_and_all_final_evaluations',
                'updated_at': self.primary.now(),
            }
            if release:
                lease['released_at'] = self.primary.now()
            write_json(self.primary.LEASE, lease)

    def finish_primary(self) -> None:
        """Wait for the adopted S1 and run its unchanged full evaluation.

        等待保留的S1训练，再调用原函数完成完整评测。
        """
        child = self.verified['child']
        while alive(child):
            time.sleep(15)
            self.save()
        final = PARENT / 'jobs' / LIVE_JOB / 'checkpoint/model_epoch_4.pth'
        with LOCK:
            self.primary.STATE['children'].pop(LIVE_JOB, None)
            self.primary.STATE['jobs'][LIVE_JOB].update(
                train_exit=None, train_exit_provenance='not_reapable',
                train_observed_ended_at=self.primary.now(),
                final_checkpoint_exists=final.is_file())
            self.save()
        assert final.is_file() and final.stat().st_size > 0, 'No final S1 checkpoint'
        record = next(row for row in self.primary.JOBS if row[0] == LIVE_JOB)
        self.primary.run_job(record, 0)

    def worker(self, gpu: int) -> None:
        """Drain registered groups after a GPU's existing work. 每卡完成原任务后领取分组。

        Args:
            gpu: Physical GPU index. 物理GPU编号。
        """
        if gpu == 0:
            try:
                self.finish_primary()
            except Exception as error:
                with LOCK:
                    self.primary.STATE['jobs'][LIVE_JOB].update(
                        phase='failed', error=repr(error))
                    self.save()
                print(f'PRIMARY FAILED: {error!r}', flush=True)
            with LOCK:
                self.primary.STATE['phase'] = (
                    'COMPLETE' if all(row['phase'] == 'complete' for row in
                                      self.primary.STATE['jobs'].values())
                    else 'COMPLETE_WITH_FAILURES')
                self.save()
        while True:
            with LOCK:
                count = next(self.pending, None)
                if count is None:
                    return
                self.low.STATE.setdefault('group_gpu', {})[str(count)] = gpu
                self.save()
            try:
                few.run_group(count, gpu)
            except Exception as error:
                with LOCK:
                    self.low.STATE.setdefault('failed_groups', {})[str(count)] = repr(error)
                    self.save()
                print(f'FAILED PAIR GROUP {count}: {error!r}', flush=True)


def main(apply: bool) -> None:
    """Replace both supervisors and fill the free GPU. 替换两个管理进程并接续空闲卡。

    Args:
        apply: Perform the verified handoff. 执行已核验的交接。
    """
    primary = load_primary()
    lock_handle = (PARENT / 'overlap_controller.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verified = preflight(primary)
    old = json.loads(Path('/root/autodl-tmp/setup/r1_followup_20261011/'
                         'job_results.json').read_text())
    reference, _ = primary.predictions(Path(next(iter(old.values()))['prediction']))
    primary.REFERENCE = few.queue.REFERENCE = reference
    print(json.dumps({'preflight': 'passed', 'apply': apply,
                      'preserved_child': verified['child'],
                      'replaced_controllers': [verified['parent'], verified['waiter']],
                      'source_sha256': verified['sources']}, indent=2), flush=True)
    if not apply:
        return
    archive = PARENT / 'queue_overlap_20261027' / primary.now().replace(':', '')
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('status.json', 'pipeline.json', 'few_runs/wait_status.json'):
        (archive / Path(name).name).write_bytes((PARENT / name).read_bytes())
    (archive / 'gpu_lease.json').write_bytes(primary.LEASE.read_bytes())
    write_json(archive / 'handoff.json', verified)
    # Stop waiting first; stopping parent must never signal the child session.
    # 先停等待器；停止父进程时不向训练子会话发送信号。
    stop_controller(verified['waiter'])
    stop_controller(verified['parent'])
    assert alive(verified['child'])
    primary_lock = (PARENT / 'controller.lock').open('a')
    fcntl.flock(primary_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    current = json.loads((PARENT / 'status.json').read_text())
    assert current['children'] == verified['state']['children']
    verified['state'] = current
    coordinator = Coordinator(primary, verified)
    assert primary is not few.queue
    assert primary.WORK != few.queue.WORK
    coordinator.save()
    pipeline = json.loads((PARENT / 'pipeline.json').read_text())
    pipeline.update(phase='PRIMARY_AND_FEW_PAIR_OVERLAP',
                    primary_controller_pid=os.getpid(),
                    continuation_controller_pid=os.getpid(),
                    overlap_handoff=str(archive / 'handoff.json'),
                    overlap_source_sha256=primary.sha256(Path(__file__)),
                    scheduling='GPU1 starts few-pair now; GPU0 joins after S1 test')
    write_json(PARENT / 'pipeline.json', pipeline)
    wait_record = json.loads((few.WORK / 'wait_status.json').read_text())
    wait_record.update(pid=os.getpid(), phase='RUNNING_OVERLAP',
                       replaced_waiter_pid=verified['waiter']['pid'],
                       updated_at=primary.now())
    write_json(few.WORK / 'wait_status.json', wait_record)
    print('HANDOFF_APPLIED_S1_PRESERVED_FREE_GPU_STARTING_FEW_PAIRS', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(coordinator.worker, (1, 0)))
    coordinator.low.STATE['phase'] = (
        'COMPLETE' if all(row['phase'] == 'complete' for row in
                          coordinator.low.STATE['jobs'].values())
        else 'COMPLETE_WITH_FAILURES')
    coordinator.save(release=True)
    print('FINITE_QUEUES_FINISHED_AND_LEASE_RELEASED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    main(parser.parse_args().apply)
