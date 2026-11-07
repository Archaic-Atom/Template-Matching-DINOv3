"""Measure three existing matchers on one GPU under a common protocol.

用同一GPU、真实输入和计时口径测量三种既有匹配结构，不进行训练。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any


REPO = Path(__file__).resolve().parents[2]
WORK = Path('/root/autodl-tmp/setup/tmm_inference_cost_20261028')
LEASE = Path('/root/autodl-tmp/gpu_lease.json')
PYTHON = '/root/autodl-tmp/conda/envs/tm-dinov3/bin/python'
OWNER = 'tmm_inference_cost_20261028'
SOURCE_JOBS = Path('/root/autodl-tmp/setup/tmm_review_20261020/jobs')
CHECKPOINTS = {
    'G': REPO / 'Checkpoint_G_rtr_256_s2028/model_epoch_4.pth',
    'C': SOURCE_JOBS / 'r1_7_cascade_jbu/checkpoint/model_epoch_4.pth',
    'CB': SOURCE_JOBS / 'r1_7_cascade_bilinear/checkpoint/model_epoch_4.pth',
}


def now() -> str:
    """Return local ISO time. 返回本地ISO时间。"""
    return datetime.datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    """Hash a file without loading it all. 流式校验文件。

    Args:
        path: Input file. 输入文件。

    Returns:
        SHA-256 digest. SHA-256摘要。
    """
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def save_json(path: Path, value: dict) -> None:
    """Replace a JSON record atomically. 原子替换JSON记录。

    Args:
        path: Destination file. 目标文件。
        value: Serializable record. 可序列化记录。
    """
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def summarize(values: list[float]) -> dict:
    """Summarize repeated latencies using a linear 95th percentile.

    汇总重复时延，95分位采用线性插值。

    Args:
        values: Millisecond measurements. 毫秒测量值。

    Returns:
        Count, mean, median, sample SD and 95th percentile. 数量及统计量。
    """
    assert len(values) > 1 and all(math.isfinite(v) and v > 0 for v in values)
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    percentile = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {'n': len(values), 'mean_ms': statistics.mean(values),
            'median_ms': statistics.median(values),
            'sd_ms': statistics.stdev(values), 'p95_ms': percentile}


def count_parameters(parameters: Any) -> dict:
    """Count unique tensors, including shared-module aliases only once.

    共享模块别名只计一次，分别记录总参数与可训练参数。

    Args:
        parameters: Parameter iterable. 参数迭代器。

    Returns:
        Total and configured trainable counts. 总数与配置可训练数。
    """
    unique = {id(parameter): parameter for parameter in parameters}
    return {'total_params': sum(p.numel() for p in unique.values()),
            'trainable_params': sum(p.numel() for p in unique.values()
                                    if p.requires_grad)}


def prepare_plan() -> dict:
    """Record fixed inputs and conditions before measurements.

    测量前记录固定输入及条件，不使用测试成绩选择方案。

    Returns:
        Protocol and file provenance. 测量协议及来源。
    """
    manifest = Path('/root/autodl-tmp/setup/campaign_20260925/eval20k.csv')
    with manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20000
    inputs = []
    for index in range(0, 20000, 2500):
        row = rows[index]
        inputs.append({'index': index, 'sar': row['SAR_Image'],
                       'rgb': row['RGB_Image'],
                       'sar_sha256': sha256(Path(row['SAR_Image'])),
                       'rgb_sha256': sha256(Path(row['RGB_Image']))})
    configs = {name: REPO / 'Source/UserModelImplementation/Models'
               / f'TM_Model_{name}/config.yaml' for name in CHECKPOINTS}
    return {
        'created_at': now(), 'purpose': 'Common inference cost for G, C/JBU, CB/bilinear',
        'reviewer': 'R1-7 optional inference-only supplement',
        'variant_order': ['G', 'C', 'CB'], 'gpu_index': 0, 'batch': 1,
        'rgb_shape': [1, 3, 256, 256], 'sar_shape': [1, 3, 192, 192],
        'precision': 'float32, autocast disabled, TF32 disabled',
        'seed': 2028, 'warmup_first_input': 20, 'warmup_per_stage': 2,
        'timing_rounds': 3, 'iterations_per_input_round': 10,
        'inputs': inputs, 'eval_manifest': str(manifest),
        'eval_manifest_sha256': sha256(manifest),
        'configurations': {name: {'path': str(path), 'sha256': sha256(path)}
                           for name, path in configs.items()},
        'checkpoints': {name: {'path': str(path), 'sha256': sha256(path)}
                        for name, path in CHECKPOINTS.items()},
        'provider': {'path': str(REPO / 'Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'),
                     'sha256': sha256(REPO / 'Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')},
        'script_sha256': sha256(Path(__file__)),
        'timing_scope': 'GPU-resident inputs through model and hard primary decode; '
                        'disk IO, CPU preprocessing, H2D, checkpoint load excluded',
        'cached_scope': 'Cache only four raw DINOv3 optical feature maps; '
                        'retain RGB image guidance, all matching work and SAR encoding',
        'memory_scope': 'PyTorch peak allocated bytes including resident model, '
                        'one input pair and, in cached mode, its optical feature cache',
        'interpretation': 'Runtime comparison only. Different training budgets '
                          'do not establish a matched accuracy effect of architecture.',
    }


def worker(variant: str) -> None:
    """Measure one existing checkpoint in an isolated CUDA process.

    使用独立CUDA进程测量一个既有检查点。

    Args:
        variant: G, C or CB. 模型变体。
    """
    sys.path.insert(0, str(REPO / 'Source'))
    import torch
    from torchvision.io import ImageReadMode, read_image
    from Tools.eval_stage1_matching import preprocess_image
    from UserModelImplementation.Models.builder import build_model_from_yaml
    from UserModelImplementation.Models._components.decode import decode_primary

    plan = json.loads((WORK / 'plan.json').read_text())
    torch.set_num_threads(8)
    torch.manual_seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 10
    args = SimpleNamespace(is_stage_2=True, root_path=str(REPO))
    config = plan['configurations'][variant]
    checkpoint = plan['checkpoints'][variant]
    assert sha256(Path(config['path'])) == config['sha256']
    assert sha256(Path(checkpoint['path'])) == checkpoint['sha256']
    model = build_model_from_yaml(config['path'], args)
    model.rgb_encoder.requires_grad_(False)
    if variant == 'G':
        model.fusion_logits.requires_grad_(False)
    state = torch.load(checkpoint['path'], map_location='cpu', weights_only=False)
    model.trainable_tail.load_state_dict(state['model_0'], strict=True)
    del state
    parameters = count_parameters(model.parameters())
    components = {name: count_parameters(module.parameters()) for name, module in
                  [('provider', model.rgb_encoder), ('sar', model.sar_encoder),
                   ('matching', model.matching)]}
    components['fusion'] = count_parameters([model.fusion_logits])
    assert sum(row['total_params'] for row in components.values()) == parameters['total_params']
    model = model.cuda().eval()
    cpu_inputs = []
    for row in plan['inputs']:
        sar = preprocess_image(read_image(row['sar'], ImageReadMode.GRAY), True)
        rgb = preprocess_image(read_image(row['rgb'], ImageReadMode.RGB))
        assert list(sar.shape) == plan['sar_shape'][1:]
        assert list(rgb.shape) == plan['rgb_shape'][1:]
        cpu_inputs.append((sar.unsqueeze(0), rgb.unsqueeze(0)))
    records, memory, correctness = [], [], []
    raw_path = WORK / f'{variant}_timings.csv'
    fields = ['variant', 'mode', 'sample_index', 'round', 'iteration',
              'wall_ms', 'cuda_ms']
    with torch.inference_mode(), raw_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample, (sar_cpu, rgb_cpu) in zip(plan['inputs'], cpu_inputs):
            sar, rgb = sar_cpu.cuda(), rgb_cpu.cuda()

            def infer(features: Any = None) -> Any:
                """Use full model or its exact cached optical branch.

                调用完整模型或精确复用缓存的光学分支。

                Args:
                    features: Optional raw optical features. 可选原始光学特征。

                Returns:
                    Primary coordinates and score. 主预测坐标及分数。
                """
                if features is None:
                    output = model(sar, rgb)
                else:
                    output = model.matching(
                        sar_img=sar, rgb_img=rgb,
                        sar_feat=model.sar_encoder(sar), rgb_feat=features,
                        fusion_logits=model.fusion_logits)
                return decode_primary(output, argmax_type='hard')

            if sample['index'] == plan['inputs'][0]['index']:
                for _ in range(plan['warmup_first_input']):
                    infer()
                torch.cuda.synchronize()
            reference = [tensor.cpu() for tensor in infer()]
            for round_id in range(plan['timing_rounds']):
                for mode in ('full', 'cached'):
                    features = model.rgb_encoder(rgb) if mode == 'cached' else None
                    observed = [tensor.cpu() for tensor in infer(features)]
                    assert torch.equal(observed[0], reference[0])
                    assert torch.allclose(observed[1], reference[1], rtol=1e-5, atol=1e-6)
                    assert all(torch.isfinite(tensor).all() for tensor in observed)
                    for _ in range(plan['warmup_per_stage']):
                        infer(features)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    baseline = torch.cuda.memory_allocated()
                    for iteration in range(plan['iterations_per_input_round']):
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        wall_start = time.perf_counter_ns()
                        begin.record()
                        result = infer(features)
                        end.record()
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter_ns() - wall_start) / 1e6
                        row = {'variant': variant, 'mode': mode,
                               'sample_index': sample['index'], 'round': round_id,
                               'iteration': iteration, 'wall_ms': elapsed,
                               'cuda_ms': begin.elapsed_time(end)}
                        writer.writerow(row)
                        records.append(row)
                        del result, begin, end
                    memory.append({'mode': mode, 'sample_index': sample['index'],
                                   'round': round_id, 'baseline_bytes': baseline,
                                   'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
                    del features
                    stream.flush()
            correctness.append({'sample_index': sample['index'],
                                'pred_xy': reference[0].tolist(),
                                'full_cached_equal': True})
            del sar, rgb
            print(f'PROFILE {variant} input {sample["index"]} finished', flush=True)
    summary = []
    for mode in ('full', 'cached'):
        selected = [row for row in records if row['mode'] == mode]
        row = {'variant': variant, 'mode': mode, **parameters,
               **summarize([r['wall_ms'] for r in selected]),
               'cuda_median_ms': summarize([r['cuda_ms'] for r in selected])['median_ms'],
               'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in memory
                                           if r['mode'] == mode)}
        summary.append(row)
    result = {'status': 'complete', 'variant': variant, 'finished_at': now(),
              'summary': summary, 'components': components, 'memory_samples': memory,
              'correctness': correctness, 'checkpoint': checkpoint, 'configuration': config,
              'raw_timings_sha256': sha256(raw_path),
              'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'cudnn': torch.backends.cudnn.version(),
                          'gpu': torch.cuda.get_device_name(0),
                          'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                          'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                          'cpu_threads': torch.get_num_threads(),
                          'autocast': False, 'inference_mode': True}}
    save_json(WORK / f'{variant}_result.json', result)
    print(json.dumps({'variant': variant, 'status': 'complete', 'summary': summary}), flush=True)


def controller(run: bool) -> None:
    """Reserve one free GPU and profile all three models serially.

    预留一张空闲GPU，串行测量三个模型并在结束后释放。

    Args:
        run: Execute rather than prepare only. 是否真正执行测量。
    """
    WORK.mkdir(parents=True, exist_ok=True)
    handle = (WORK / 'controller.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (WORK / 'summary.csv').exists(), 'Benchmark already complete'
    plan = prepare_plan()
    save_json(WORK / 'plan.json', plan)
    if not run:
        print(json.dumps(plan, indent=2), flush=True)
        return
    lease_lock = (LEASE.parent / 'gpu_lease.lock').open('a')
    fcntl.flock(lease_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = json.loads(LEASE.read_text())
    assert before['state'] == 'released', 'GPU lease already held'
    gpus = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    gpu_uuid = next(line.split(',')[1].strip() for line in gpus.splitlines()
                    if line.split(',')[0].strip() == '0')
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    assert gpu_uuid not in apps, 'GPU0 has a compute process'
    save_json(WORK / 'lease_before.json', before)
    lease = {'owner': OWNER, 'state': 'active', 'pid': os.getpid(),
             'gpu_indices': [0], 'work_dir': str(WORK), 'children': {},
             'started_at': now(), 'release_after': 'three_finite_inference_profiles'}
    save_json(LEASE, lease)
    status = {'phase': 'RUNNING', 'pid': os.getpid(), 'jobs': {}, 'started_at': now()}
    save_json(WORK / 'status.json', status)
    try:
        for variant in plan['variant_order']:
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8',
                       TM_CORRELATION_CHANNELS_LAST='1', PYTHONDONTWRITEBYTECODE='1',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                       TORCH_HOME='/root/autodl-tmp/cache/torch',
                       TMPDIR='/root/autodl-tmp/tmp')
            command = [PYTHON, '-u', str(Path(__file__)), '--variant', variant]
            with (WORK / f'{variant}.log').open('a') as stream:
                child = subprocess.Popen(command, cwd=REPO, env=env,
                                         stdin=subprocess.DEVNULL, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                lease.update(children={variant: child.pid}, updated_at=now())
                save_json(LEASE, lease)
                status['jobs'][variant] = {'pid': child.pid, 'phase': 'running'}
                save_json(WORK / 'status.json', status)
                code = child.wait()
            lease['children'] = {}
            status['jobs'][variant].update(exit_code=code, phase='complete' if code == 0 else 'failed')
            save_json(WORK / 'status.json', status)
            assert code == 0, f'{variant} failed: inspect {WORK / (variant + ".log")}'
        rows = []
        for variant in plan['variant_order']:
            result = json.loads((WORK / f'{variant}_result.json').read_text())
            assert result['status'] == 'complete'
            rows.extend(result['summary'])
        with (WORK / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        status.update(phase='COMPLETE', completed_at=now())
    except Exception as error:
        status.update(phase='FAILED', error=repr(error), ended_at=now())
        raise
    finally:
        save_json(WORK / 'status.json', status)
        # This controller waits every child to exit before releasing its GPU.
        # 本控制器等待每个子进程退出后才释放所属GPU。
        assert not lease['children'], 'Retain active lease for an uncollected child'
        lease.update(state='released', released_at=now())
        save_json(LEASE, lease)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=list(CHECKPOINTS))
    parser.add_argument('--run', action='store_true')
    options = parser.parse_args()
    if options.variant:
        worker(options.variant)
    else:
        controller(options.run)
