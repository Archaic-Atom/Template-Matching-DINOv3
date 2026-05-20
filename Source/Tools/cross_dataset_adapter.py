#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-dataset adapter for SAR-optical template matching evaluation.

Builds SEN1-2-shaped patch pairs (SAR 64x64, RGB 128x128, label (x,y))
from three external SAR-optical datasets after resampling to 10 m/px
to match the training distribution:

    * WHU-OPT-SAR  (GF-1 optical + GF-3 SAR, 5 m native)
    * QXS-SAROPT   (Google Earth optical + GF-3 SAR, 1 m native)
    * SARptical    (UltraCam aerial + TerraSAR-X SAR, 1 m native)

跨数据集评测 adapter：把三个公开 SAR-optical 数据集下采样到 10 m/px,
切成 SAR 64x64 + RGB 128x128 + label (x, y) 的样本对，输出到 PNG +
.pt manifest 供 pack_to_webdataset.py 后续打包。

Output layout (per dataset)::

    <out_dir>/
        sar/00000000.png          (64x64, mode=L, uint8)
        sar/00000001.png
        ...
        rgb/00000000.png          (128x128, mode=RGB, uint8)
        rgb/00000001.png
        ...
        manifest.pt               {sar_path, rgb_path, label}
        meta.json                 {native_res_m, scale, n_pairs, ...}

Usage::

    python Source/Tools/cross_dataset_adapter.py \\
        --dataset whu_opt_sar \\
        --raw_dir /data4/datasets/WHU-OPT-SAR \\
        --out_dir /data4/datasets/cross_eval/whu_opt_sar \\
        --samples_per_scene 64

Each `--samples_per_scene` controls how many (random center + random offset)
patches we extract from each native scene. With 100 WHU scenes and 64 samples
each you get 6400 eval pairs — enough for a reliable MEE/SR@k estimate.
每个原始大图采几个 patch 由 --samples_per_scene 控制。例如 WHU 100 张大图
× 64 patch = 6400 pair，足够稳定算 MEE / SR@k。
"""

import argparse
import io
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


# =============================================================================
# Sample container
# =============================================================================

@dataclass
class Sample:
    """One generated patch pair.

    一个生成的 patch 对。
    """
    sar_patch: np.ndarray   # (H, W) uint8 — 64x64 SAR template
    rgb_patch: np.ndarray   # (H, W, 3) uint8 — 128x128 RGB search image
    label_xy: Tuple[int, int]  # SAR top-left position inside RGB, in pixels


# =============================================================================
# Base class
# =============================================================================

class BaseCrossDatasetAdapter:
    """Base adapter: resampling + cropping pipeline shared by all datasets.

    所有 dataset 的公共处理流水线（下采样 + 切 patch + 生成 label）。
    """

    # Subclasses MUST override.
    DATASET_NAME: str = ''
    NATIVE_RES_M: float = 0.0

    # Defaults from the SEN1-2 training pipeline. Override if you ever change
    # the model input shape.
    TARGET_RES_M: float = 10.0
    SAR_SIZE: int = 64
    RGB_SIZE: int = 128
    MAX_OFFSET: int = 8        # GT displacement range: uniform in [-8, +8]

    def __init__(self, raw_dir: str, out_dir: str,
                 samples_per_scene: int = 64, seed: int = 12345) -> None:
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.samples_per_scene = samples_per_scene
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        if self.NATIVE_RES_M <= 0:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set NATIVE_RES_M"
            )

    # -------- Methods subclasses must implement ----------------------------

    def list_scenes(self) -> List[Tuple[str, str]]:
        """Return list of (sar_path, optical_path) per native scene.

        子类返回每张原始大图的 (sar_path, optical_path) 元组列表。
        """
        raise NotImplementedError

    def load_scene(self, sar_path: str, opt_path: str
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Load one native scene as (sar_2d_uint8, rgb_3d_uint8) at native res.

        子类负责读原始文件并归一化到 uint8。
            - sar: (H, W) grayscale uint8
            - rgb: (H, W, 3) RGB uint8
        """
        raise NotImplementedError

    # -------- Shared pipeline -----------------------------------------------

    @staticmethod
    def _resample_uint8(arr: np.ndarray, scale: float, is_rgb: bool
                        ) -> np.ndarray:
        """Resample uint8 image by ``scale`` factor (< 1 = downsample)."""
        if abs(scale - 1.0) < 1e-6:
            return arr
        if is_rgb:
            img = Image.fromarray(arr, mode='RGB')
            resample = Image.BILINEAR
        else:
            img = Image.fromarray(arr, mode='L')
            # SAR speckle: area downsample preserves the multiplicative-noise
            # statistics better than bilinear. PIL's BOX ~ area average.
            resample = Image.BOX if scale < 1.0 else Image.BILINEAR
        new_w = max(1, int(round(img.width * scale)))
        new_h = max(1, int(round(img.height * scale)))
        return np.asarray(img.resize((new_w, new_h), resample=resample))

    def resample_pair(self, sar: np.ndarray, rgb: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Bring both arrays to TARGET_RES_M m/px."""
        scale = self.NATIVE_RES_M / self.TARGET_RES_M
        sar_10m = self._resample_uint8(sar, scale, is_rgb=False)
        rgb_10m = self._resample_uint8(rgb, scale, is_rgb=True)
        # Align shapes (rounding can produce a 1-px mismatch).
        h = min(sar_10m.shape[0], rgb_10m.shape[0])
        w = min(sar_10m.shape[1], rgb_10m.shape[1])
        return sar_10m[:h, :w], rgb_10m[:h, :w]

    def sample_patches(self, sar_10m: np.ndarray, rgb_10m: np.ndarray
                       ) -> Iterable[Sample]:
        """Yield ``samples_per_scene`` (SAR, RGB, label) tuples from one
        co-registered scene at 10 m/px.

        For each draw:
            1. Pick a random center ``(cy, cx)`` in the scene such that both
               the 64x64 SAR template and the 128x128 RGB search window fit.
            2. Pick a random integer offset ``(dx, dy) ~ U(-MAX_OFFSET, +MAX_OFFSET)``.
            3. Crop SAR around the center; crop RGB so that the matched SAR
               region lands at top-left ``(rgb_center_offset + dx,
               rgb_center_offset + dy)`` inside the RGB patch.

        Yields ``Sample(sar_patch, rgb_patch, label_xy)`` where ``label_xy``
        is the SAR top-left coordinate inside the RGB patch (in pixels).
        """
        H, W = sar_10m.shape[:2]
        sar_s = self.SAR_SIZE
        rgb_s = self.RGB_SIZE
        center_offset = (rgb_s - sar_s) // 2  # = 32 for 128/64
        # The RGB window of size rgb_s must fully fit, so we need
        # rgb_s/2 of slack on each side from the scene boundary.
        half_rgb = rgb_s // 2
        min_c = half_rgb + self.MAX_OFFSET
        max_cy = H - half_rgb - self.MAX_OFFSET - 1
        max_cx = W - half_rgb - self.MAX_OFFSET - 1
        if max_cy <= min_c or max_cx <= min_c:
            return  # scene too small after downsampling

        for _ in range(self.samples_per_scene):
            cy = self.rng.randint(min_c, max_cy)
            cx = self.rng.randint(min_c, max_cx)
            dx = self.rng.randint(-self.MAX_OFFSET, self.MAX_OFFSET)
            dy = self.rng.randint(-self.MAX_OFFSET, self.MAX_OFFSET)

            # SAR template: centered on (cy, cx).
            sar_y0, sar_x0 = cy - sar_s // 2, cx - sar_s // 2
            sar_patch = sar_10m[sar_y0:sar_y0 + sar_s, sar_x0:sar_x0 + sar_s]

            # RGB search: shift the window by (dx, dy) so the GT label
            # position of SAR inside RGB equals (center_offset + dx,
            # center_offset + dy).
            rgb_y0 = cy - half_rgb - dy
            rgb_x0 = cx - half_rgb - dx
            rgb_patch = rgb_10m[rgb_y0:rgb_y0 + rgb_s,
                                rgb_x0:rgb_x0 + rgb_s, :]

            if sar_patch.shape != (sar_s, sar_s):
                continue
            if rgb_patch.shape != (rgb_s, rgb_s, 3):
                continue

            label_xy = (center_offset + dx, center_offset + dy)
            yield Sample(sar_patch, rgb_patch, label_xy)

    def build(self) -> None:
        """Top-level: list scenes, resample, sample patches, write to disk."""
        scenes = self.list_scenes()
        if not scenes:
            print(f"[{self.DATASET_NAME}] no scenes found under {self.raw_dir}",
                  file=sys.stderr)
            return

        sar_out_dir = self.out_dir / 'sar'
        rgb_out_dir = self.out_dir / 'rgb'
        sar_out_dir.mkdir(parents=True, exist_ok=True)
        rgb_out_dir.mkdir(parents=True, exist_ok=True)

        sar_paths: List[str] = []
        rgb_paths: List[str] = []
        labels: List[Tuple[int, int]] = []
        n_emitted = 0

        for i, (sar_path, opt_path) in enumerate(scenes):
            try:
                sar, rgb = self.load_scene(sar_path, opt_path)
            except Exception as exc:
                print(f"[{self.DATASET_NAME}] skip scene {i}: {exc}",
                      file=sys.stderr)
                continue
            sar_10m, rgb_10m = self.resample_pair(sar, rgb)
            for sample in self.sample_patches(sar_10m, rgb_10m):
                idx = n_emitted
                sar_fp = sar_out_dir / f'{idx:08d}.png'
                rgb_fp = rgb_out_dir / f'{idx:08d}.png'
                Image.fromarray(sample.sar_patch, mode='L').save(sar_fp)
                Image.fromarray(sample.rgb_patch, mode='RGB').save(rgb_fp)
                sar_paths.append(str(sar_fp))
                rgb_paths.append(str(rgb_fp))
                labels.append(sample.label_xy)
                n_emitted += 1
            if (i + 1) % 10 == 0:
                print(f"[{self.DATASET_NAME}] processed {i + 1}/{len(scenes)} "
                      f"scenes; emitted {n_emitted} samples so far",
                      flush=True)

        # Manifest (.pt) so pack_to_webdataset.py can read directly.
        import torch
        torch.save({
            'sar_path': sar_paths,
            'rgb_path': rgb_paths,
            'label': torch.tensor(labels, dtype=torch.long),
        }, self.out_dir / 'manifest.pt')

        # Human-readable meta.
        with open(self.out_dir / 'meta.json', 'w') as fh:
            json.dump({
                'dataset': self.DATASET_NAME,
                'native_res_m': self.NATIVE_RES_M,
                'target_res_m': self.TARGET_RES_M,
                'scale': self.NATIVE_RES_M / self.TARGET_RES_M,
                'sar_size': self.SAR_SIZE,
                'rgb_size': self.RGB_SIZE,
                'max_offset': self.MAX_OFFSET,
                'samples_per_scene': self.samples_per_scene,
                'n_scenes_seen': len(scenes),
                'n_samples_emitted': n_emitted,
            }, fh, indent=2)
        print(f"[{self.DATASET_NAME}] done. {n_emitted} samples written to "
              f"{self.out_dir}", flush=True)


# =============================================================================
# WHU-OPT-SAR
# =============================================================================

class WHUOPTSARAdapter(BaseCrossDatasetAdapter):
    """WHU-OPT-SAR (Hubei, GF-1 RGB+NIR + GF-3 SAR, native 5 m/px).

    Native layout (from https://github.com/AmberHen/WHU-OPT-SAR-dataset)::

        WHU-OPT-SAR/
            optical/  XX.tif    # 4-band (R,G,B,NIR), 5556x3704
            sar/      XX.tif    # 1-band  amplitude (or log-mag), 5556x3704

    We read the first three bands of the optical image as RGB; SAR is
    rescaled to 0-255 with a per-scene 1/99-percentile stretch (typical
    for displaying SAR amplitude).
    取 optical 的前 3 波段当 RGB；SAR 用 1/99 percentile stretch 到 uint8。
    """

    DATASET_NAME = 'whu_opt_sar'
    NATIVE_RES_M = 5.0

    def list_scenes(self) -> List[Tuple[str, str]]:
        opt_dir = self.raw_dir / 'optical'
        sar_dir = self.raw_dir / 'sar'
        if not opt_dir.is_dir() or not sar_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {opt_dir} and {sar_dir} under {self.raw_dir}"
            )
        scenes = []
        for opt_fp in sorted(opt_dir.glob('*.tif')) + sorted(opt_dir.glob('*.tiff')):
            sar_fp = sar_dir / opt_fp.name
            if sar_fp.exists():
                scenes.append((str(sar_fp), str(opt_fp)))
        return scenes

    @staticmethod
    def _stretch_to_uint8(arr: np.ndarray) -> np.ndarray:
        a = arr.astype(np.float32)
        lo, hi = np.percentile(a, [1.0, 99.0])
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.uint8)
        a = (a - lo) / (hi - lo)
        return np.clip(a * 255.0, 0, 255).astype(np.uint8)

    def load_scene(self, sar_path: str, opt_path: str
                   ) -> Tuple[np.ndarray, np.ndarray]:
        # Use tifffile if available (more robust on 4-band scientific TIFFs);
        # fall back to PIL for 3-band cases.
        try:
            import tifffile
            opt = tifffile.imread(opt_path)
            sar = tifffile.imread(sar_path)
        except ImportError:
            opt = np.asarray(Image.open(opt_path))
            sar = np.asarray(Image.open(sar_path))

        # Optical: take first 3 channels as RGB.
        if opt.ndim == 3 and opt.shape[-1] >= 3:
            rgb = opt[..., :3]
        elif opt.ndim == 2:
            rgb = np.stack([opt] * 3, axis=-1)
        else:
            raise ValueError(f"Unexpected optical shape: {opt.shape}")
        rgb = self._stretch_to_uint8(rgb)

        # SAR: 1-band amplitude → uint8 via percentile stretch.
        if sar.ndim == 3:
            sar = sar[..., 0]
        sar_u8 = self._stretch_to_uint8(sar)
        return sar_u8, rgb


# =============================================================================
# QXS-SAROPT
# =============================================================================

class QXSSAROPTAdapter(BaseCrossDatasetAdapter):
    """QXS-SAROPT (GF-3 SAR + Google Earth optical, 1 m/px, already 256x256).

    Native layout (from https://github.com/yaoxu008/QXS-SAROPT)::

        QXS-SAROPT/
            opt_256/  *.tif | *.png      # already 256x256
            sar_256/  *.tif | *.png      # already 256x256

    Because the patches are already tiny (256x256 at 1 m), we treat each
    pair as a single ``scene`` and dial samples_per_scene to something
    small (e.g. 4 — anything more starts to re-sample the same pixels).
    一对图就是一个 scene，每对采几个就够了，太多会过度采样同一像素。

    Note (cross-resolution evaluation): we deliberately keep this adapter
    at 1 m/px (TARGET_RES_M = NATIVE_RES_M) rather than down-sampling to
    10 m/px to match SEN1-2. The motivation is that down-sampling
    256x256@1m to 25.6x25.6@10m leaves too few pixels to crop a 64x64
    SAR + 128x128 RGB sample. By keeping native resolution here we
    additionally probe how DOSR-Net behaves on a 10x finer scale than it
    was trained on, which is a useful cross-resolution generalization
    signal to discuss alongside the cross-sensor signal.
    保持 native 1 m/px：(1) 下采样到 10 m 后 patch 只剩 25.6x25.6 不够切；
    (2) 顺便测一下模型在比训练分辨率细 10 倍的尺度下的泛化能力。
    """

    DATASET_NAME = 'qxs_saropt'
    NATIVE_RES_M = 1.0
    TARGET_RES_M = 1.0       # ★ override: no resampling for cross-resolution eval

    def list_scenes(self) -> List[Tuple[str, str]]:
        opt_dir = next((self.raw_dir / d for d in ('opt_256', 'optical', 'opt')
                        if (self.raw_dir / d).is_dir()), None)
        sar_dir = next((self.raw_dir / d for d in ('sar_256', 'sar')
                        if (self.raw_dir / d).is_dir()), None)
        if opt_dir is None or sar_dir is None:
            raise FileNotFoundError(
                f"Expected opt_256/ and sar_256/ (or opt/ and sar/) under "
                f"{self.raw_dir}"
            )
        scenes = []
        for opt_fp in sorted(opt_dir.glob('*')):
            if opt_fp.suffix.lower() not in ('.tif', '.tiff', '.png', '.jpg'):
                continue
            sar_fp = sar_dir / opt_fp.name
            if not sar_fp.exists():
                # Try same stem, .png/.tif swap.
                for ext in ('.tif', '.tiff', '.png', '.jpg'):
                    cand = sar_dir / (opt_fp.stem + ext)
                    if cand.exists():
                        sar_fp = cand
                        break
            if sar_fp.exists():
                scenes.append((str(sar_fp), str(opt_fp)))
        return scenes

    def load_scene(self, sar_path: str, opt_path: str
                   ) -> Tuple[np.ndarray, np.ndarray]:
        opt = np.asarray(Image.open(opt_path).convert('RGB'))
        sar = np.asarray(Image.open(sar_path).convert('L'))
        return sar, opt


# =============================================================================
# SARptical
# =============================================================================

class SARpticalAdapter(BaseCrossDatasetAdapter):
    """SARptical (TerraSAR-X spotlight + UltraCam aerial, Berlin, ~1 m/px).

    Native layout (the dataset is distributed as pre-extracted patches around
    individual buildings; structure varies by release)::

        SARptical/
            optical/  *.png         # 112x112 typical
            sar/      *.png         # 112x112 typical
            pairs.txt               # optional list of (sar_id, opt_id)

    If a ``pairs.txt`` exists we trust it; otherwise we pair by filename stem.
    """

    DATASET_NAME = 'sarptical'
    NATIVE_RES_M = 1.0

    def list_scenes(self) -> List[Tuple[str, str]]:
        opt_dir = next((self.raw_dir / d for d in ('optical', 'opt')
                        if (self.raw_dir / d).is_dir()), None)
        sar_dir = self.raw_dir / 'sar' if (self.raw_dir / 'sar').is_dir() else None
        if opt_dir is None or sar_dir is None:
            raise FileNotFoundError(
                f"Expected optical/ and sar/ under {self.raw_dir}"
            )

        pairs_txt = self.raw_dir / 'pairs.txt'
        if pairs_txt.exists():
            scenes = []
            for line in pairs_txt.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                sar_fp = sar_dir / parts[0]
                opt_fp = opt_dir / parts[1]
                if sar_fp.exists() and opt_fp.exists():
                    scenes.append((str(sar_fp), str(opt_fp)))
            return scenes

        # Fallback: filename stem match.
        scenes = []
        for opt_fp in sorted(opt_dir.glob('*.png')):
            sar_fp = sar_dir / opt_fp.name
            if sar_fp.exists():
                scenes.append((str(sar_fp), str(opt_fp)))
        return scenes

    def load_scene(self, sar_path: str, opt_path: str
                   ) -> Tuple[np.ndarray, np.ndarray]:
        opt = np.asarray(Image.open(opt_path).convert('RGB'))
        sar = np.asarray(Image.open(sar_path).convert('L'))
        return sar, opt


# =============================================================================
# CLI
# =============================================================================

ADAPTERS = {
    'whu_opt_sar': WHUOPTSARAdapter,
    'qxs_saropt': QXSSAROPTAdapter,
    'sarptical': SARpticalAdapter,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', required=True, choices=sorted(ADAPTERS))
    parser.add_argument('--raw_dir', required=True,
                        help='path to the unpacked dataset root')
    parser.add_argument('--out_dir', required=True,
                        help='where to write SAR/RGB PNGs + manifest.pt')
    parser.add_argument('--samples_per_scene', type=int, default=64,
                        help='# patches drawn per native scene (default 64; '
                             'for QXS-SAROPT recommend 4)')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    cls = ADAPTERS[args.dataset]
    adapter = cls(args.raw_dir, args.out_dir,
                  samples_per_scene=args.samples_per_scene, seed=args.seed)
    adapter.build()


if __name__ == '__main__':
    main()
