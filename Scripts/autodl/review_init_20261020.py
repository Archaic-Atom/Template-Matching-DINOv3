"""Prepare matched initial states for the finite reviewer queue.

为有限的审稿实验队列准备共同初始权重，不加载冻结光学模型到GPU。
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'Source'))
from UserModelImplementation.Models.builder import build_model_from_config
from UserModelImplementation.Models.builder import load_config


def main() -> None:
    """Build identical shared tensors and audit S1 transfer. 核验共同参数和S1迁移。"""
    work = Path(os.environ['TM_REVIEW_WORK'])
    destination = work / 'initialization'
    destination.mkdir(exist_ok=True)
    shared_path = REPO / 'Checkpoint_2x2_shared_init_s2028/model_stage1_init.pth'
    shared = torch.load(shared_path, map_location='cpu', weights_only=False)['model_0']
    args = SimpleNamespace(root_path=str(REPO), is_stage_2=True, sar_size=192,
                           matching_dim=256, fine_dim=32, fine_margin=24)
    reference = None
    records = {}
    for variant, name in [('TM_Model_C', 'cascade_jbu'),
                          ('TM_Model_CB', 'cascade_bilinear'),
                          ('TM_Model_G', 'standard_random')]:
        random.seed(2028)
        np.random.seed(2028)
        torch.manual_seed(2028)
        config = load_config(str(REPO / 'Source/UserModelImplementation/Models'
                                 / variant / 'config.yaml'))
        model = build_model_from_config(config, args)
        state = {key: value.detach().clone().cpu()
                 for key, value in model.trainable_tail.state_dict().items()}
        matched = []
        if reference is None:
            reference = {key: value.clone() for key, value in state.items()}
        else:
            for key, value in state.items():
                source_key = key if key in reference else key.replace('.up.', '.jbu.')
                if source_key in reference and value.shape == reference[source_key].shape:
                    state[key] = reference[source_key].clone()
                    matched.append(key)
            assert all(torch.equal(state[key], reference[key if key in reference
                             else key.replace('.up.', '.jbu.')]) for key in matched)
        sar_keys = [key for key in state if key.startswith('sar_encoder.')]
        assert all(torch.equal(state[key], reference[key]) for key in sar_keys)
        path = destination / f'{name}.pth'
        torch.save({'epoch': -1, 'model_0': state,
                    'review_initialization': 'matched_random_seed2028'}, path)
        records[name] = {'path': str(path), 'sha256': hashlib.sha256(
            path.read_bytes()).hexdigest(), 'sar_tensors': len(sar_keys),
            'matched_common_tensors': len(matched), 'total_tensors': len(state)}
        print(json.dumps({name: records[name]}), flush=True)
        if name == 'standard_random':
            prealigned = {key: value.clone() for key, value in state.items()}
            for key in sar_keys:
                assert key in shared and shared[key].shape == state[key].shape, key
                prealigned[key] = shared[key].detach().clone()
            assert any(not torch.equal(state[k], prealigned[k]) for k in sar_keys)
            assert all(torch.equal(state[k], prealigned[k]) for k in state
                       if not k.startswith('sar_encoder.'))
            aligned_path = destination / 'standard_s1.pth'
            torch.save({'epoch': -1, 'model_0': prealigned,
                        'review_initialization': 'shared_s1_with_matched_head'},
                       aligned_path)
            records['standard_s1'] = {
                'path': str(aligned_path), 'sha256': hashlib.sha256(
                    aligned_path.read_bytes()).hexdigest(),
                'head_identical_to_random': True,
                'sar_identical_to_registered_stage1': True,
                'source': str(shared_path)}
        del model, state
        gc.collect()
    (destination / 'verification.json').write_text(
        json.dumps(records, indent=2) + '\n')
    print('MATCHED_INITIALIZATIONS_READY', flush=True)


if __name__ == '__main__':
    main()
