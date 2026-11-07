"""Run an isolated reviewer experiment without changing the shared model code.

通过进程内配置覆盖运行审稿实验，不修改已验证的公共模型实现。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'Source'))


def merge_config(target: dict, extra: dict) -> dict:
    """Merge a registered JSON override. 合并已登记的JSON配置覆盖。

    Args:
        target: Original configuration. 原配置。
        extra: Experiment overrides. 实验覆盖项。

    Returns:
        Updated configuration. 更新后的配置。
    """
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_config(target[key], value)
        else:
            target[key] = value
    return target


def configure() -> None:
    """Apply process-local experiment settings. 施加本进程的实验设置。"""
    from UserModelImplementation.Models import builder
    from UserModelImplementation.Models import _interface_base as base
    import torch

    override_path = Path(os.environ['TM_REVIEW_OVERRIDE'])
    override = json.loads(override_path.read_text())
    original_load = builder.load_config
    original_get = base.ComponentBasedInterface.get_model
    original_inference = base.ComponentBasedInterface.inference
    smoke_steps = int(os.environ.get('TM_REVIEW_SMOKE_UPDATES', '0'))
    record_dir = Path(os.environ['TM_REVIEW_RECORD'])
    record_dir.mkdir(parents=True, exist_ok=True)
    observed_gradients = {}

    def load_config(path: str) -> dict:
        """Load the exact override for both config readers. 两个读取点使用同一覆盖。"""
        return merge_config(original_load(path), override)

    def capture_gradient(name: str, gradient: torch.Tensor) -> torch.Tensor:
        """Record finite training gradients. 记录有限的训练梯度。"""
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f'Nonfinite gradient: {name}')
        observed_gradients[name] = float(gradient.detach().norm())
        return gradient

    def get_model(self: object) -> list:
        """Freeze only the registered SAR prefix. 只冻结登记的SAR前缀。"""
        models = original_get(self)
        settings = self._config.get('training', {}).get('stage_2', {})
        prefix = settings.get('freeze_sar_modules', [])
        for model in models:
            if prefix and (not self._is_stage_2 or model.freeze_sar_in_stage2):
                raise ValueError('Partial freezing requires the SAR grad path')
            for name in prefix:
                if name not in ('stem', 'layer1', 'layer2', 'layer3', 'proj16'):
                    raise ValueError(f'Unknown SAR module: {name}')
                getattr(model.sar_encoder, name).requires_grad_(False)
            names = [name for name, p in model.named_parameters() if p.requires_grad]
            contract = {
                'frozen_sar_modules': prefix,
                'trainable_sar_modules': sorted({name.split('.')[1] for name in names
                                                if name.startswith('sar_encoder.')}),
                'trainable_parameters': sum(p.numel() for p in model.parameters()
                                            if p.requires_grad),
                'override': str(override_path),
                'override_sha256': hashlib.sha256(override_path.read_bytes()).hexdigest(),
            }
            (record_dir / 'model_contract.json').write_text(
                json.dumps(contract, indent=2) + '\n')
            print('REVIEW_MODEL_CONTRACT ' + json.dumps(contract), flush=True)
            if smoke_steps:
                for group in ('sar_encoder.', 'matching.'):
                    name, parameter = next((name, p) for name, p in model.named_parameters()
                                           if name.startswith(group) and p.requires_grad)
                    parameter.register_hook(
                        lambda grad, key=name: capture_gradient(key, grad))
        return models

    def inference(self: object, model: object, input_data: list,
                  model_id: int) -> list:
        """End the smoke only after optimizer updates. 完成反传后结束小试。"""
        count = getattr(self, '_review_steps', 0)
        if smoke_steps and self._args.mode == 'train' and count >= smoke_steps:
            assert len(observed_gradients) == 2, observed_gradients
            assert all(value > 0 for value in observed_gradients.values())
            report = {'completed_updates': count, 'gradients': observed_gradients,
                      'max_cuda_memory_mib': torch.cuda.max_memory_allocated() / 2**20,
                      'status': 'passed', 'at': time.time()}
            (record_dir / 'smoke_result.json').write_text(
                json.dumps(report, indent=2) + '\n')
            print('REVIEW_SMOKE_PASSED ' + json.dumps(report), flush=True)
            raise SystemExit(0)
        self._review_steps = count + 1
        return original_inference(self, model, input_data, model_id)

    builder.load_config = load_config
    base.load_config = load_config
    base.ComponentBasedInterface.get_model = get_model
    base.ComponentBasedInterface.inference = inference


if __name__ == '__main__':
    configure()
    runpy.run_path(str(REPO / 'Source/main.py'), run_name='__main__')
