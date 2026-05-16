#!/usr/bin/env python3
"""Smoke test for all 4 TM_Model variants — full forward + loss + backward path.

Runs on CPU (so it doesn't compete with A's GPU training). Builds each
variant, runs one forward pass on random tensors of the expected shape,
computes loss + accuracy, and runs backward. Confirms every code path
works end-to-end without crashes.

For TM_Model_C in Phase 1, expects NotImplementedError on stage-2 forward
(CascadeHead.forward is a stub). Counted as a positive result.

Usage (on remote):
    cd /data/raozhibo/Documents/Programs/Template-Matching-DINOv3
    python Scripts/verify_variants_smoke.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch


def _add_paths(repo_root: str) -> None:
    src = os.path.join(repo_root, 'Source')
    if src not in sys.path:
        sys.path.insert(0, src)


def _mock_args(repo_root: str, is_stage_2: bool) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.root_path = repo_root + '/'
    ns.dinov3_path = 'Source/Libs/dinov3/'
    ns.pre_trained_weights_path = 'Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
    ns.pre_trained = True
    ns.is_stage_2 = is_stage_2
    ns.matching_dim = 128
    ns.fine_dim = 32
    ns.fine_margin = 4
    ns.sar_size = 64
    ns.lr = 2e-4
    # Stage-2 loss-related args used by A's legacy interface (yaml variants ignore them).
    ns.match_w4 = 1.0
    ns.match_w8 = 0.5
    ns.match_sigma4 = 1.0
    ns.match_sigma8 = 1.0
    ns.match_pos_weight = None
    ns.match_w_fine = 1.0
    ns.match_sigma_fine = 1.0
    ns.match_fine_pos_weight = None
    ns.match_primary_stride = 4
    ns.fine_warmup_epochs = 3
    return ns


def _section(title: str) -> None:
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def _expected_shapes(stage_2: bool) -> dict:
    if stage_2:
        return {'sar_in': (2, 3, 64, 64), 'rgb_in': (2, 3, 128, 128)}
    return {'sar_in': (2, 3, 64, 64), 'rgb_in': (2, 3, 64, 64)}


def smoke_test_stage1(variant_name: str, iface_cls, repo_root: str) -> bool:
    """Build variant in stage 1; forward + alignment loss + backward."""
    _section(f"[Stage 1] {variant_name}")
    args = _mock_args(repo_root, is_stage_2=False)
    try:
        iface = iface_cls(args)
        model = iface.get_model()[0].train()
        shapes = _expected_shapes(stage_2=False)
        sar = torch.randn(*shapes['sar_in'])
        rgb = torch.randn(*shapes['rgb_in'])

        t0 = time.perf_counter()
        out = model(sar, rgb)
        fwd_ms = (time.perf_counter() - t0) * 1000.0
        assert 'sar_feat' in out and 'rbg_feat' in out, f"missing keys: {list(out)}"
        sar16 = out['sar_feat']['sar16']
        assert sar16.shape == (2, 1024, 4, 4), f"unexpected sar16 shape: {sar16.shape}"
        print(f"  forward OK in {fwd_ms:.0f} ms — sar16={tuple(sar16.shape)}, rbg layers={len(out['rbg_feat'])}")

        disp = torch.zeros(2, 2)
        loss = iface.loss([out], [disp], 0)[0]
        print(f"  loss = {loss.item():.4f}")
        loss.backward()

        trainable = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        print(f"  backward OK — {trainable} trainable params received grad")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc(limit=5)
        return False


def smoke_test_stage2(variant_name: str, iface_cls, repo_root: str,
                      expect_nie: bool = False) -> bool:
    """Build variant in stage 2; forward + two-stage loss + accuracy + backward.

    If expect_nie=True (TM_Model_C in Phase 1), forward should raise
    NotImplementedError; that is the success condition.
    """
    _section(f"[Stage 2] {variant_name}{' (expect NotImplementedError)' if expect_nie else ''}")
    args = _mock_args(repo_root, is_stage_2=True)
    try:
        iface = iface_cls(args)
        model = iface.get_model()[0].train()
        shapes = _expected_shapes(stage_2=True)
        sar = torch.randn(*shapes['sar_in'])
        rgb = torch.randn(*shapes['rgb_in'])

        t0 = time.perf_counter()
        try:
            out = model(sar, rgb)
        except NotImplementedError as e:
            if expect_nie:
                print(f"  NotImplementedError raised (expected): {e}")
                return True
            raise
        fwd_ms = (time.perf_counter() - t0) * 1000.0

        assert 'heatmap_per_stride' in out, f"missing heatmap_per_stride; keys: {list(out)}"
        assert 'heatmap_fine' in out, f"missing heatmap_fine; keys: {list(out)}"
        h4 = out['heatmap_per_stride'][4]
        h8 = out['heatmap_per_stride'][8]
        hf = out['heatmap_fine']
        print(f"  forward OK in {fwd_ms:.0f} ms — "
              f"H4={tuple(h4.shape)}, H8={tuple(h8.shape)}, H_fine={tuple(hf.shape)}")

        # Loss expects integer pixel displacement; keep it inside the fine window.
        disp = torch.tensor([[8.0, 12.0], [20.0, 32.0]])
        loss = iface.loss([out], [disp], 0)[0]
        print(f"  loss = {loss.item():.4f}")

        # Accuracy: D's path runs sub_pixel; B/A run hard argmax.
        with torch.no_grad():
            err = iface.accuracy([out], [disp], 0)[0]
        print(f"  accuracy (pixel-error) = {err.item():.4f}")

        loss.backward()
        trainable_with_grad = sum(
            1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        print(f"  backward OK — {trainable_with_grad} trainable params received grad")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc(limit=5)
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_root', default=os.getcwd())
    parser.add_argument('--skip_stage1', action='store_true',
                        help='Skip Stage-1 forward (saves ~30s of DINO load)')
    parser.add_argument('--variants', default='A,B,C,D',
                        help='Comma-separated list of variants to test')
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    _add_paths(repo_root)

    # Force CPU so we don't compete with running training on GPUs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    from UserModelImplementation.Models import _get_model_dict
    model_dict = _get_model_dict()

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    name_map = {'A': 'TM_Model_A', 'B': 'TM_Model_B', 'C': 'TM_Model_C', 'D': 'TM_Model_D'}

    results: dict[str, bool] = {}
    for short in variants:
        full = name_map.get(short, short)
        if full not in model_dict:
            print(f"  SKIP — unknown variant {full!r}")
            continue
        iface_cls = model_dict[full]
        # Phase 2: CascadeHead now implemented; no variant should raise NIE.
        expect_nie = False

        if not args.skip_stage1:
            results[f"{full}-s1"] = smoke_test_stage1(full, iface_cls, repo_root)
        results[f"{full}-s2"] = smoke_test_stage2(full, iface_cls, repo_root, expect_nie=expect_nie)

    _section("SUMMARY")
    ok_count = sum(1 for v in results.values() if v)
    total = len(results)
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\n  {ok_count}/{total} {'PASS' if ok_count == total else 'FAIL'}")
    return 0 if ok_count == total else 1


if __name__ == '__main__':
    raise SystemExit(main())
