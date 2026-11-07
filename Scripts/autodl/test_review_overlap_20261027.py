"""Exercise queue isolation, device ownership, and final lease release on CPU.

使用CPU验证队列隔离、物理卡归属和最终租约释放，不启动实际训练。
"""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import review_overlap_20261027 as overlap


def main() -> None:
    """Run meaningful scheduler checks with fake training. 使用模拟训练验证调度边界。"""
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        overlap.PARENT = root
        overlap.few.WORK = root / 'few_runs'
        overlap.few.WORK.mkdir()
        primary = overlap.load_primary()
        primary.WORK = root
        primary.LEASE = root / 'lease.json'
        initial = {'phase': 'RUNNING', 'jobs': {
            overlap.LIVE_JOB: {'phase': 'train'},
            'completed_control': {'phase': 'complete', 'result': {'mee_px': 3.0}},
        }, 'children': {overlap.LIVE_JOB: 123},
            'cancelled_jobs': {name: {'phase': 'cancelled'}
                               for name in overlap.CANCELLED}}
        coordinator = overlap.Coordinator(primary, {'state': initial,
                                                   'child': {'pid': 123}})
        assert primary is not overlap.few.queue
        assert primary.STATE is not overlap.few.queue.STATE
        assert primary.WORK != overlap.few.queue.WORK
        results['queue_globals_isolated'] = True
        coordinator.low.STATE['children']['few_100_stage1'] = 456
        coordinator.save()
        lease = json.loads(primary.LEASE.read_text())
        assert lease['children'] == {overlap.LIVE_JOB: 123,
                                     'few_100_stage1': 456}
        assert primary.STATE['jobs']['completed_control']['result']['mee_px'] == 3.0
        assert primary.STATE['cancelled_jobs'] == initial['cancelled_jobs']
        results['lease_tracks_union_and_preserves_results_cancellations'] = True
        try:
            coordinator.save(release=True)
            raise RuntimeError('Active work was incorrectly released')
        except AssertionError:
            results['release_rejects_active_children'] = True
        coordinator.low.STATE['children'].clear()
        coordinator.low.STATE['phase'] = 'COMPLETE'
        primary.STATE['children'].clear()
        primary.STATE['phase'] = 'COMPLETE'
        coordinator.save(release=False)
        assert json.loads(primary.LEASE.read_text())['state'] == 'active'
        coordinator.save(release=True)
        assert json.loads(primary.LEASE.read_text())['state'] == 'released'
        results['release_requires_explicit_all_workers_done'] = True

        # A free GPU starts without waiting for the primary training.
        # 空闲卡直接开始任务，不等待主训练结束。
        primary.STATE['phase'] = 'RUNNING'
        calls = []
        with mock.patch.object(overlap.few, 'run_group',
                               side_effect=lambda count, gpu: calls.append((count, gpu))), \
                mock.patch.object(coordinator, 'finish_primary') as finish:
            coordinator.worker(1)
            finish.assert_not_called()
        assert calls == [(100, 1), (500, 1), (1000, 1)]
        results['free_gpu_drains_only_registered_counts_once'] = True
        coordinator.pending = iter((500, 1000))
        calls.clear()

        def finish_primary() -> None:
            """Record completion before the GPU can claim a group. 先完成主任务再领分组。"""
            calls.append(('primary_finished', 0))
            primary.STATE['jobs'][overlap.LIVE_JOB]['phase'] = 'complete'

        with mock.patch.object(coordinator, 'finish_primary', side_effect=finish_primary), \
                mock.patch.object(overlap.few, 'run_group',
                                  side_effect=lambda count, gpu: calls.append((count, gpu))):
            coordinator.worker(0)
        assert calls == [('primary_finished', 0), (500, 0), (1000, 0)]
        assert coordinator.low.STATE['phase'] == 'COMPLETE'
        results['gpu0_joins_after_primary_completion_on_same_device'] = True

        # Once adopted training exits, a missing endpoint must not trigger retraining.
        # 接管训练退出后，缺少最终检查点时不能自动重训。
        primary.STATE['children'][overlap.LIVE_JOB] = 123
        with mock.patch.object(overlap, 'alive', return_value=False), \
                mock.patch.object(primary, 'run_job') as run_job:
            try:
                coordinator.finish_primary()
                raise RuntimeError('Missing endpoint was incorrectly accepted')
            except AssertionError:
                pass
            run_job.assert_not_called()
        assert primary.STATE['jobs'][overlap.LIVE_JOB]['train_exit'] is None
        results['missing_endpoint_never_retrains_or_forges_exit_code'] = True
    print(json.dumps({'status': 'passed', 'checks': results}, indent=2))


if __name__ == '__main__':
    main()
