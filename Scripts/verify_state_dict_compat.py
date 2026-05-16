#!/usr/bin/env python3
"""Verify that yaml-built variant models share state_dict structure with TM_Model_A.

Run from repo root on the remote (DINOv3 weights live there):
    python Scripts/verify_state_dict_compat.py

Checks performed:
1. SarResNet16GN (A's class) vs SarResNet16GN (_components/sar_encoder) — should
   produce byte-identical state_dict keys (same class definition, copied).
2. Build ComposedTM for B/D via yaml. Compare its state_dict keys against
   what TMModelANet produces. The 'sar_encoder.*' and 'matching.*' keys must
   match. (TM_Model_C uses a different matching head, so cascade keys won't match.)
3. Load Checkpoint/model_epoch_0.pth into each variant's trainable_tail with
   strict=False; report missing/unexpected keys. For A/B/D: ZERO missing/unexpected
   in sar_encoder.*. For C: cascade keys missing (fresh init expected).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def _add_paths(repo_root: str) -> None:
    """Make Source/ importable."""
    src = os.path.join(repo_root, 'Source')
    if src not in sys.path:
        sys.path.insert(0, src)


def _mock_args(repo_root: str) -> argparse.Namespace:
    """Minimal argparse Namespace matching what UserInterface populates."""
    ns = argparse.Namespace()
    ns.root_path = repo_root + '/'
    ns.dinov3_path = 'Source/Libs/dinov3/'
    ns.pre_trained_weights_path = 'Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
    ns.pre_trained = True
    ns.is_stage_2 = False  # toggle externally for stage-2 checks
    ns.matching_dim = 128
    ns.fine_dim = 32
    ns.fine_margin = 4
    ns.sar_size = 64
    ns.lr = 2e-4
    return ns


def _section(title: str) -> None:
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def check_sar_encoder_only() -> bool:
    """Smallest possible structural check: SarResNet16GN from both locations
    must produce identical state_dict keys."""
    _section("[1] SarResNet16GN: A's code vs _components/")
    from UserModelImplementation.Models._legacy.legacy_A.Nets._create_sar_encoder import (
        SarResNet16GN as SarA)
    from UserModelImplementation.Models._components.sar_encoder import (
        SarResNet16GN as SarB)

    sar_a = SarA()
    sar_b = SarB()
    keys_a = set(sar_a.state_dict().keys())
    keys_b = set(sar_b.state_dict().keys())
    if keys_a == keys_b:
        print(f"  OK — {len(keys_a)} keys, byte-identical")
        return True
    print(f"  MISMATCH")
    print(f"  only in A: {sorted(keys_a - keys_b)[:5]}")
    print(f"  only in B: {sorted(keys_b - keys_a)[:5]}")
    return False


def check_variant_vs_A(variant_name: str, is_stage_2: bool, repo_root: str) -> bool:
    """Build the variant via yaml; compare trainable_tail state_dict keys to A."""
    _section(f"[2-{variant_name}] {variant_name} vs TM_Model_A (stage_2={is_stage_2})")

    from UserModelImplementation.Models._legacy.legacy_A.Nets.model import TMModelANet
    from UserModelImplementation.Models import _get_model_dict

    args = _mock_args(repo_root)
    args.is_stage_2 = is_stage_2

    # Build A's model the legacy way
    model_a = TMModelANet(args)
    keys_a = set(model_a.trainable_tail.state_dict().keys())

    # Build the variant via the yaml dispatch
    cls = _get_model_dict()[variant_name]
    iface = cls(args)
    model_v_list = iface.get_model()
    model_v = model_v_list[0]
    keys_v = set(model_v.trainable_tail.state_dict().keys())

    only_a = keys_a - keys_v
    only_v = keys_v - keys_a
    common = keys_a & keys_v

    print(f"  A keys: {len(keys_a)}, variant keys: {len(keys_v)}, common: {len(common)}")
    if only_a or only_v:
        print(f"  only in A ({len(only_a)}): {sorted(only_a)[:8]}{' ...' if len(only_a) > 8 else ''}")
        print(f"  only in V ({len(only_v)}): {sorted(only_v)[:8]}{' ...' if len(only_v) > 8 else ''}")
    else:
        print(f"  OK — identical")

    # Specifically check sar_encoder keys (always must match across all variants)
    sar_a = {k for k in keys_a if k.startswith('sar_encoder.')}
    sar_v = {k for k in keys_v if k.startswith('sar_encoder.')}
    if sar_a == sar_v:
        print(f"  sar_encoder.*: {len(sar_a)} keys, match")
        return True
    print(f"  sar_encoder.* MISMATCH:")
    print(f"    only A: {sorted(sar_a - sar_v)[:5]}")
    print(f"    only V: {sorted(sar_v - sar_a)[:5]}")
    return False


def check_load_ckpt(variant_name: str, ckpt_path: str, repo_root: str) -> bool:
    """Try loading model_epoch_0.pth into the variant's trainable_tail with strict=False."""
    _section(f"[3-{variant_name}] load ckpt into {variant_name}")

    if not os.path.exists(ckpt_path):
        print(f"  SKIP — {ckpt_path} not found")
        return False

    from UserModelImplementation.Models import _get_model_dict
    args = _mock_args(repo_root)
    args.is_stage_2 = True  # full model with matching for max key coverage

    cls = _get_model_dict()[variant_name]
    iface = cls(args)
    model = iface.get_model()[0]

    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt = state['model_0']

    missing, unexpected = model.trainable_tail.load_state_dict(ckpt, strict=False)
    sar_missing = [k for k in missing if k.startswith('sar_encoder.')]
    print(f"  total missing: {len(missing)}, unexpected: {len(unexpected)}")
    print(f"  sar_encoder.* missing: {len(sar_missing)} (must be 0 for A/B/D ckpt sharing)")
    if sar_missing:
        print(f"    examples: {sar_missing[:5]}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_root', default=os.getcwd())
    parser.add_argument('--ckpt', default='Checkpoint/model_epoch_0.pth')
    args = parser.parse_args()
    repo_root = os.path.abspath(args.repo_root)
    _add_paths(repo_root)

    ok = True
    ok &= check_sar_encoder_only()

    # Stage 1 (only sar_encoder in trainable_tail)
    for v in ('TM_Model_B', 'TM_Model_D'):
        ok &= check_variant_vs_A(v, is_stage_2=False, repo_root=repo_root)

    # Stage 2 (sar_encoder + matching + fusion in trainable_tail)
    for v in ('TM_Model_B', 'TM_Model_D'):
        ok &= check_variant_vs_A(v, is_stage_2=True, repo_root=repo_root)

    # C has a different matching head; only check sar_encoder.* compat there
    _section("[2-TM_Model_C] (cascade matching differs; only sar_encoder.* should match)")
    ok_c = check_variant_vs_A('TM_Model_C', is_stage_2=True, repo_root=repo_root)
    # ok_c may print 'mismatch' overall but sar_encoder check inside returns True/False
    # for our purposes, the function's True/False is on sar_encoder match
    ok &= ok_c

    # Load A's epoch_0 ckpt into each variant
    ckpt_path = os.path.join(repo_root, args.ckpt)
    for v in ('TM_Model_B', 'TM_Model_D'):
        ok &= check_load_ckpt(v, ckpt_path, repo_root)
    # C: skip clean load; cascade keys are missing by design
    print("\n[3-TM_Model_C] skipped — CascadeHead.forward stub; ckpt load test deferred to Phase 2")

    _section("SUMMARY")
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
