"""Check revision components without CUDA or downloaded weights.

不依赖CUDA或下载权重，检查返修组件的数学与配置契约。
"""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import random
import sys
from types import ModuleType
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as functional
import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ROOT / 'Source/UserModelImplementation/Models/_components'
MODELS = COMPONENTS.parent
PACKAGE = '_tmm_cpu_components'
namespace = ModuleType(PACKAGE)
namespace.__path__ = [str(COMPONENTS)]
sys.modules[PACKAGE] = namespace
coarse = importlib.import_module(PACKAGE + '.coarse_matching')
cascade = importlib.import_module(PACKAGE + '.fine_head_cascade')
bilinear = importlib.import_module(PACKAGE + '.bilinear_upsampler')
torch.set_num_threads(1)


def configuration(name: str) -> dict:
    """Load a variant without model construction. 仅加载配置，不构建模型。

    Args:
        name: Model variant. 模型变体名称。

    Returns:
        Configuration without display metadata. 移除展示字段后的配置。
    """
    value = yaml.safe_load((MODELS / name / 'config.yaml').read_text())
    value.pop('name', None)
    value.pop('description', None)
    return value


class RevisionComponentTests(unittest.TestCase):
    """Verify independent numerical and configuration properties. 核验独立性质。"""

    def test_bilinear_shape_and_guidance_independence(self) -> None:
        """Bilinear output ignores guidance and has the expected size. 检查尺寸及无引导性。"""
        torch.manual_seed(19)
        source = torch.randn(2, 16, 3, 4, requires_grad=True)
        for levels in (1, 2):
            layer = bilinear.BilinearStack(16, 8, levels=levels).eval()
            first = layer(source, torch.randn(2, 3, 12, 16))
            second = layer(source, torch.randn(2, 3, 12, 16) * 100)
            self.assertEqual(tuple(first.shape), (2, 8, 3 * 2**levels, 4 * 2**levels))
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            first.square().mean().backward()
            self.assertTrue(torch.isfinite(source.grad).all())

    def test_projector_normalizes_channels(self) -> None:
        """Matching projection returns unit channel norms. 检查匹配特征归一化。"""
        layer = coarse.ProjMatchBilinear(16, 8, levels=2).eval()
        result = layer(torch.randn(2, 16, 3, 4), None)
        torch.testing.assert_close(result.norm(dim=1), torch.ones(2, 12, 16))

    def test_grouped_correlation_matches_individual_samples(self) -> None:
        """Batch grouping and optional layout preserve correlations. 检查分组与布局。"""
        torch.manual_seed(23)
        template = torch.randn(3, 4, 3, 4)
        search = torch.randn(3, 4, 7, 8)
        expected = torch.cat([
            functional.conv2d(search[i:i + 1], template[i:i + 1])
            for i in range(3)
        ])
        for layout in ('0', '1'):
            with mock.patch.dict(os.environ, {'TM_CORRELATION_CHANNELS_LAST': layout}):
                actual = coarse.MatchingHead._cross_correlate(template, search)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_correlation_rejects_incompatible_shapes(self) -> None:
        """Invalid batch/channel contracts fail clearly. 检查形状错误出口。"""
        with self.assertRaises(ValueError):
            coarse.MatchingHead._cross_correlate(
                torch.randn(2, 4, 3, 3), torch.randn(1, 4, 8, 8)
            )

    def test_bilinear_cascade_all_scales_and_gradients(self) -> None:
        """The full control produces fine/coarse maps and gradients. 检查完整对照前反向。"""
        head = cascade.CascadeHeadBilinear(
            dino_dim=16, matching_dim=8, sar_size=32, raw_stem_dim=4,
            cascade_window_2=2, cascade_window_1=2,
        )
        fusion = torch.zeros(2, requires_grad=True)
        result = head(
            torch.randn(1, 3, 32, 32), torch.randn(1, 3, 64, 64),
            {'sar16': torch.randn(1, 16, 2, 2)},
            [torch.randn(1, 16, 4, 4)], fusion,
        )
        self.assertEqual(set(result['heatmap_per_stride']), {4, 8})
        self.assertEqual(set(result['heatmap_cascade']), {1, 2})
        self.assertEqual(tuple(result['heatmap_per_stride'][4].shape), (1, 1, 9, 9))
        self.assertEqual(tuple(result['heatmap_cascade'][1].shape), (1, 1, 5, 5))
        self.assertEqual(tuple(result['fine_anchor_pix'].shape), (1, 2))
        loss = sum(x.square().mean() for x in result['heatmap_per_stride'].values())
        loss = loss + sum(x.square().mean() for x in result['heatmap_cascade'].values())
        loss.backward()
        self.assertIsNone(fusion.grad)
        self.assertTrue(torch.isfinite(head.head_1.refine[0].weight.grad).all())

    def test_shared_family_only_changes_trainability(self) -> None:
        """F/G keep every configuration factor except freezing equal. 检查冻结单因子。"""
        frozen = configuration('TM_Model_F')
        trainable = configuration('TM_Model_G')
        self.assertTrue(frozen['training']['stage_2']['freeze_sar_encoder'])
        frozen['training']['stage_2']['freeze_sar_encoder'] = False
        self.assertEqual(frozen, trainable)

    def test_bilinear_variant_only_changes_coarse_head(self) -> None:
        """C/CB keep the same source and fine-stage settings. 检查粗尺度替换范围。"""
        original = configuration('TM_Model_C')
        control = configuration('TM_Model_CB')
        original['architecture']['matching']['type'] = 'CascadeHeadBilinear'
        self.assertEqual(original, control)

    def test_web_variant_only_changes_provider_weights(self) -> None:
        """C/CW differ only in the optical provider checkpoint. 检查provider单因子。"""
        original = configuration('TM_Model_C')
        web = configuration('TM_Model_CW')
        original['architecture']['rgb_encoder']['args']['weights_path'] = (
            web['architecture']['rgb_encoder']['args']['weights_path']
        )
        self.assertEqual(original, web)

    def test_runtime_seed_and_invalid_cudnn_limit(self) -> None:
        """Runtime hooks seed repeatably and reject negative limits. 检查运行设置。"""
        tree = ast.parse((ROOT / 'Source/main.py').read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'_maybe_fix_seed', '_configure_cudnn_search'}]
        scope = {'os': os, 'torch': torch, 'random': random, 'np': np}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'main_hooks', 'exec'), scope)
        previous = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        try:
            with mock.patch.dict(os.environ, {'TM_SEED': '41', 'TM_SEED_FAST': '0'}):
                scope['_maybe_fix_seed']()
                first = torch.randn(4)
                scope['_maybe_fix_seed']()
                torch.testing.assert_close(first, torch.randn(4), rtol=0, atol=0)
            with mock.patch.dict(os.environ, {'TM_CUDNN_BENCHMARK_LIMIT': '-1'}):
                with self.assertRaises(ValueError):
                    scope['_configure_cudnn_search']()
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous


if __name__ == '__main__':
    unittest.main()
