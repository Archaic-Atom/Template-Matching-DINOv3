"""Recover verified pair groups and create nested low-data training lists.

从已留存训练分片恢复并核验真实图像对分组，生成嵌套低数据量清单。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import tarfile

import numpy as np
from PIL import Image


WORK = Path('/root/autodl-tmp/setup/tmm_review_20261020/few_pairs')
TRAIN = Path('/root/autodl-tmp/setup/shard_train_256_subset40.csv')
EVAL = Path('/root/autodl-tmp/setup/campaign_20260925/eval20k.csv')
SEED = 20261020


def digest(data: bytes) -> str:
    """Hash one encoded image. 校验一幅编码图像。

    Args:
        data: Original bytes. 原始字节。

    Returns:
        SHA-256 digest. SHA-256校验值。
    """
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    """Verify all 48 views within each selected source pair. 核验每对的48个裁剪。"""
    WORK.mkdir(exist_ok=True)
    assert not (WORK / 'verification.json').exists(), 'Already prepared'
    with EVAL.open() as stream:
        evaluation = list(csv.DictReader(stream))
    test_paths = sorted({r['RGB_Image'] for r in evaluation})
    test_hashes = {digest(Path(p).read_bytes()) for p in test_paths}
    with TRAIN.open() as stream:
        shards = list(csv.DictReader(stream))
    candidates = []
    for row in shards:
        path = Path(row['path'])
        match = re.fullmatch(r'shard-train-(\d+)-(\d+)\.tar', path.name)
        assert match and int(row['file_num']) == 32000
        worker, shard = map(int, match.groups())
        first = shard * 32000
        # The original packer groups 48 views per pair and resets tar keys.
        # 原生成器每pair连续48个视图，tar内部key从零开始。
        for pair in range(math.ceil(first / 48), (first + 32000) // 48):
            candidates.append({'id': f'w{worker:03d}_p{pair:06d}',
                               'source': str(path), 'start': pair * 48 - first})
    assert len({r['id'] for r in candidates}) == len(candidates)
    random.Random(SEED).shuffle(candidates)
    selected = candidates[:1000]
    grouped = {}
    for rank, row in enumerate(selected):
        row['rank'] = rank
        grouped.setdefault(row['source'], []).append(row)
    (WORK / 'selection_plan.json').write_text(json.dumps({
        'selection_seed': SEED, 'candidate_pairs': len(candidates),
        'selected': selected, 'method': 'Complete 48-view groups; source pair '
        'ordinals reconstructed from the retained packer and validated by '
        'identical RGB bytes and overlapping SAR pixels.',
        'stratification': 'Uniform random pair sampling; original season '
        'metadata was not retained, so geographic/season stratification is '
        'not claimed.'}, indent=2) + '\n')
    samples, ledger = {}, []
    for source, groups in grouped.items():
        with tarfile.open(source) as archive:
            members = {m.name: m for m in archive.getmembers()}
            for row in groups:
                target = WORK / 'images' / row['id']
                target.mkdir(parents=True, exist_ok=True)
                canvas = np.zeros((256, 256), dtype=np.uint8)
                known = np.zeros((256, 256), dtype=bool)
                rgb_bytes = None
                examples = []
                for index in range(48):
                    key = f'{row["start"] + index:08d}'
                    rgb = archive.extractfile(members[f'{key}.rgb.png']).read()
                    sar = archive.extractfile(members[f'{key}.sar.png']).read()
                    label = archive.extractfile(members[f'{key}.label.cls']).read()
                    x, y = map(int, label.decode().split())
                    assert 0 <= x <= 64 and 0 <= y <= 64
                    if rgb_bytes is None:
                        rgb_bytes = rgb
                        rgb_hash = digest(rgb)
                        assert rgb_hash not in test_hashes, 'Train/test RGB overlap'
                        assert Image.open(io.BytesIO(rgb)).size == (256, 256)
                        (target / 'rgb.png').write_bytes(rgb)
                    else:
                        assert rgb == rgb_bytes, f'Pair grouping mismatch: {row}'
                    values = np.array(Image.open(io.BytesIO(sar)).convert('L'))
                    assert values.shape == (192, 192)
                    region = canvas[y:y + 192, x:x + 192]
                    mask = known[y:y + 192, x:x + 192]
                    assert np.array_equal(region[mask], values[mask]), row
                    region[:] = values
                    mask[:] = True
                    sar_path = target / f'sar_{index:02d}.png'
                    sar_path.write_bytes(sar)
                    examples.append({'SAR_Image': str(sar_path),
                                     'RGB_Image': str(target / 'rgb.png'),
                                     'Label': json.dumps([x, y])})
                samples[row['rank']] = examples
                ledger.append({'selection_rank': row['rank'], 'pair_id': row['id'],
                               'source_tar': source, 'first_sample_key': row['start'],
                               'views': 48, 'rgb_sha256': rgb_hash,
                               'sar_overlap_consistent': True})
        print(json.dumps({'source_complete': source,
                          'verified_pairs': len(samples)}), flush=True)
    assert len(samples) == 1000
    metadata = {}
    for count in (100, 500, 1000):
        unique = [r for index in range(count) for r in samples[index]]
        assert len(unique) == count * 48
        virtual = unique * (48000 // len(unique))
        assert len(virtual) == 48000
        random.Random(SEED + count).shuffle(virtual)
        for name, rows in [('unique', unique), ('virtual', virtual)]:
            path = WORK / f'train_{count}_{name}.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream,
                                        fieldnames=['SAR_Image', 'RGB_Image', 'Label'])
                writer.writeheader()
                writer.writerows(rows)
        metadata[count] = {'pairs': count, 'generated_samples': len(unique),
                           'virtual_epoch_samples': len(virtual),
                           'list': str(WORK / f'train_{count}_virtual.csv'),
                           'list_sha256': digest((WORK / f'train_{count}_virtual.csv').read_bytes())}
    ledger.sort(key=lambda r: r['selection_rank'])
    with (WORK / 'pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    report = {'status': 'passed', 'selection_seed': SEED,
              'test_rgb_files_checked': len(test_paths),
              'selected_pair_rgb_overlap_with_test': 0,
              'pair_groups_checked': 1000, 'views_per_group_checked': 48,
              'sar_pixel_overlap_consistent': True, 'nested_datasets': metadata,
              'original_pair_name_mapping': 'Reconstructed source ordinals; '
              'original scene names not retained in training tars.',
              'original_train_manifest_sha256': digest(TRAIN.read_bytes()),
              'eval_csv_sha256': digest(EVAL.read_bytes())}
    (WORK / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    with (WORK / 'dataset_counts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata[100]))
        writer.writeheader()
        writer.writerows(metadata.values())
    print('FEW_PAIR_DATA_READY', flush=True)


if __name__ == '__main__':
    main()
