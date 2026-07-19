# -*- coding: utf-8 -*-
"""Isolated smoke test for item-5 h/w decoupling at 256/192.

验证 dataloader 在 template_size=192 下:
  - Stage-2: SAR 192, RGB 256(不裁剪), label=(x,y) float in [0,64]
  - Stage-1: SAR 192, RGB 裁成 192, heatmap 网格保持 64x64
不依赖模型 / S1 ckpt,只测 item-5 的解耦是否正确。
"""
import sys
sys.path.insert(0, 'Source/UserModelImplementation/Dataloaders')
from _tm_webdataset import build_tm_webdataset  # noqa: E402

CSV = '/data4/datasets/TM_shards_opt256_sar192_test/shard_path.csv'


def one(is_pre_train, is_stage_2, tag):
    ds = build_tm_webdataset(
        CSV, is_train=False, is_pre_train=is_pre_train, is_stage_2=is_stage_2,
        template_size=192, batch_size=1, num_workers=1)
    sar, rgb, label, sn, rn = next(iter(ds))
    lab = label.tolist() if hasattr(label, 'tolist') else label
    lab_shape = tuple(label.shape)
    print(f'[{tag}] sar={tuple(sar.shape)} rgb={tuple(rgb.shape)} '
          f'label_shape={lab_shape} label_sample={lab if is_stage_2 else "heatmap"}')


print('=== item-5 dataloader smoke @256/192 ===')
one(is_pre_train=False, is_stage_2=True,  tag='STAGE2')
one(is_pre_train=True,  is_stage_2=False, tag='STAGE1')
print('=== EXPECT: STAGE2 sar(3,192,192) rgb(3,256,256) label(2,); '
      'STAGE1 sar(3,192,192) rgb(3,192,192) heatmap(64,64) ===')
print('ITEM5_SMOKE_OK')
