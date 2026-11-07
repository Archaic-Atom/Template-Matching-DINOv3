"""Remove redundant queued baselines while preserving live training processes.

取消尚未启动的重复基线；接管现有训练进程并保留原评测和后续实验。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import threading
import time

import review_queue_20261020 as queue


CANCELLED = {'shared_standard_random', 'shared_standard_s1'}
ACTIVE_GPUS = {'r1_7_cascade_jbu': '0', 'r1_7_cascade_bilinear': '1'}
SOURCE_SHA256 = 'de64e1d635d7c26e4456e8f943284bb50736282437c0f9d376f6ec28db26208c'


def identity(pid: int) -> dict | None:
    """Read Linux process identity, including PID-reuse protection.

    读取Linux进程身份，使用启动时钟防止PID复用。

    Args:
        pid: Process identifier. 进程编号。

    Returns:
        Identity or None for an exited process. 进程身份，退出后为None。
    """
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (root / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return None
    return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
            'session': int(fields[3]), 'start_ticks': fields[19],
            'args': [item.decode() for item in command if item]}


def alive(record: dict) -> bool:
    """Check that the original live process still exists.

    检查原进程是否仍存活，僵尸状态不算运行。

    Args:
        record: Previously verified identity. 已核对的进程身份。

    Returns:
        Whether the same process is running. 同一进程是否仍运行。
    """
    current = identity(record['pid'])
    return bool(current and current['start_ticks'] == record['start_ticks']
                and current['state'] != 'Z')


def preflight() -> tuple[dict, dict, dict]:
    """Verify cancellation and process boundaries before any mutation.

    任何变更前核验取消范围和进程隔离。

    Returns:
        Current state, parent identity and children. 当前状态、父进程与子进程。

    Raises:
        AssertionError: If the expected state has changed. 预期状态已改变。
    """
    source = Path(queue.__file__)
    assert queue.sha256(source) == SOURCE_SHA256, 'Queue source changed'
    state = json.loads((queue.WORK / 'status.json').read_text())
    assert state['phase'] == 'RUNNING'
    parent = identity(state['controller_pid'])
    assert parent and alive(parent)
    assert str(source) in parent['args']
    assert set(state['children']) == set(ACTIVE_GPUS)
    for job in CANCELLED:
        assert state['jobs'][job]['phase'] == 'queued', job
        target = queue.WORK / 'jobs' / job
        assert not (target / 'train_command.json').exists(), job
        assert not (target / 'train.log').exists(), job
    children = {}
    for job, gpu in ACTIVE_GPUS.items():
        assert state['jobs'][job]['phase'] == 'train', job
        child = identity(state['children'][job])
        assert child and alive(child), job
        assert child['ppid'] == parent['pid'], job
        assert child['session'] == child['pid'], job
        assert '--mode' in child['args']
        assert child['args'][child['args'].index('--mode') + 1] == 'train'
        assert str(queue.WORK / 'jobs' / job / 'checkpoint') in child['args']
        env = (Path('/proc') / str(child['pid']) / 'environ').read_bytes()
        assert f'CUDA_VISIBLE_DEVICES={gpu}'.encode() in env.split(b'\0')
        children[job] = child
    lease = json.loads(queue.LEASE.read_text())
    assert lease['pid'] == parent['pid'] and lease['state'] == 'active'
    assert lease['owner'] == 'tmm_review_20261020'
    return state, parent, children


def install_adoption(children: dict, revision: dict) -> None:
    """Wrap existing helpers to await live jobs without launching them again.

    包装原控制器函数，等待既有训练结束，避免重复启动。

    Args:
        children: Verified original training processes. 已核验的原训练进程。
        revision: Persisted scheduling decision. 已落盘的调度决定。
    """
    original_run = queue.run_job
    original_save = queue.save_state
    worker_gpu = threading.local()

    def save_state() -> None:
        """Retain explicit cancellations in each snapshot. 每次快照保留取消记录。"""
        queue.STATE['cancelled_jobs'] = revision['cancelled_jobs']
        queue.STATE['queue_revision'] = revision['record_path']
        original_save()

    def run_job(record: tuple, gpu: int) -> None:
        """Adopt an existing training process, then use the standard evaluator.

        接管既有训练进程，结束后调用原有标准评测。

        Args:
            record: Existing registered condition. 原登记条件。
            gpu: Assigned physical GPU. 指定物理GPU。
        """
        job = record[0]
        if job in children:
            child = children[job]
            # Worker scheduling order is not guaranteed by ThreadPoolExecutor.
            # 线程启动顺序不确定，后续任务须沿用实际接管的物理GPU。
            worker_gpu.value = int(ACTIVE_GPUS[job])
            with queue.LOCK:
                queue.STATE['children'][job] = child['pid']
                queue.STATE['jobs'][job].update(
                    phase='train', pid=child['pid'], adopted_at=queue.now(),
                    original_start_ticks=child['start_ticks'])
                queue.save_state()
            while alive(child):
                time.sleep(15)
                with queue.LOCK:
                    queue.save_state()
            # An orphan's exit code cannot be reaped by this controller.
            # 孤儿进程退出码无法由本控制器回收，不伪造成功退出记录。
            final = queue.WORK / 'jobs' / job / 'checkpoint/model_epoch_4.pth'
            with queue.LOCK:
                queue.STATE['children'].pop(job, None)
                queue.STATE['jobs'][job].update(
                    train_exit=None, train_exit_provenance='not_reapable',
                    train_observed_ended_at=queue.now(),
                    final_checkpoint_exists=final.is_file())
                queue.save_state()
            if not final.is_file() or final.stat().st_size == 0:
                raise RuntimeError(f'{job}: adopted training lacks final checkpoint')
        original_run(record, getattr(worker_gpu, 'value', gpu))

    queue.save_state = save_state
    queue.run_job = run_job


def main(apply: bool) -> None:
    """Replace only the queue supervisor, preserving all live GPU children.

    仅替换队列管理进程，保留全部正在运行的GPU训练。

    Args:
        apply: Whether to perform the handoff. 是否执行交接。
    """
    state, parent, children = preflight()
    print(json.dumps({'preflight': 'passed', 'apply': apply,
                      'old_controller': parent, 'children': children,
                      'cancel_jobs': sorted(CANCELLED)}, indent=2), flush=True)
    if not apply:
        return
    stamp = queue.now().replace(':', '').replace('+', '_')
    archive = queue.WORK / 'queue_revision_20261024' / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('status.json', 'plan.json', 'pipeline.json'):
        source = queue.WORK / name
        if source.exists():
            (archive / name).write_bytes(source.read_bytes())
    (archive / 'gpu_lease.json').write_bytes(queue.LEASE.read_bytes())
    cancelled = {job: {**state['jobs'][job], 'phase': 'cancelled',
                       'cancelled_at': queue.now(),
                       'reason': 'Reuse completed five-seed single-head results; '
                                 'author prioritizes rented GPU cost'}
                 for job in sorted(CANCELLED)}
    revision = {'created_at': queue.now(), 'new_controller_pid': os.getpid(),
                'old_controller': parent, 'preserved_children': children,
                'cancelled_jobs': cancelled, 'archive': str(archive),
                'record_path': str(archive / 'revision.json'),
                'source_sha256': queue.sha256(Path(__file__)),
                'scientific_boundary': 'Old single-head and new cascade recipes '
                    'differ; do not claim a strictly matched head comparison. '
                    'C versus CB retains its matched recipe.'}
    (archive / 'revision.json').write_text(json.dumps(revision, indent=2) + '\n')
    # Kill only the parent PID. Child sessions and GPU training stay intact.
    # 仅终止父进程PID，子会话及GPU训练保持运行。
    assert alive(parent)
    os.kill(parent['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while alive(parent) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not alive(parent), 'Original controller did not exit'
    assert all(alive(child) for child in children.values())
    adjusted = copy.deepcopy(state)
    adjusted['jobs'] = {job: row for job, row in state['jobs'].items()
                        if job not in CANCELLED}
    adjusted.update(controller_pid=os.getpid(), cancelled_jobs=cancelled,
                    queue_revision=revision['record_path'])
    temporary = queue.WORK / 'handoff_status.tmp'
    temporary.write_text(json.dumps(adjusted, indent=2) + '\n')
    temporary.replace(queue.WORK / 'status.json')
    pipeline_path = queue.WORK / 'pipeline.json'
    if pipeline_path.exists():
        pipeline = json.loads(pipeline_path.read_text())
        pipeline['queue_revision'] = revision['record_path']
        pipeline['primary_controller_pid'] = os.getpid()
        pipeline['finite_training_jobs'] -= len(CANCELLED)
        pipeline['formal_final_matching_evaluations'] -= len(CANCELLED)
        pipeline['primary_resume_script_sha256'] = revision['source_sha256']
        pipeline_path.write_text(json.dumps(pipeline, indent=2) + '\n')
    queue.JOBS = [row for row in queue.JOBS if row[0] not in CANCELLED]
    install_adoption(children, revision)
    print('HANDOFF_APPLIED_KEEPING_ORIGINAL_TRAINING_PIDS', flush=True)
    queue.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    arguments = parser.parse_args()
    main(arguments.apply)
