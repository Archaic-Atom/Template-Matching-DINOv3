# -*- coding: utf-8 -*-
import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


try:
    from ._data_aug import TrainAug
except ImportError:
    from _data_aug import TrainAug


class TM_Dataset(Dataset):
    def __init__(self, train_path: str, train: bool = True,
                 is_pre_train: bool = True, is_stage_2: bool = False):
        """
        Args:
            train_path:   path to .csv or .pt list file.
            train:        True for training augmentation, False for eval.
            is_pre_train: Stage 1 mode — crops RGB to the SAR-aligned 64x64 patch
                          and returns a heatmap label. Ignored when is_stage_2=True.
            is_stage_2:   Stage 2 (template matching) mode — keeps the full RGB
                          search region and returns the displacement (x, y) as a
                          float tensor instead of a heatmap.
        """
        self.is_train = train
        self.is_pre_train = is_pre_train
        self.is_stage_2 = is_stage_2

        _, ext = os.path.splitext(train_path)
        if ext == ".csv":
            self.sar, self.rgb, labels = self._read_csv(train_path)
        else:
            self.sar, self.rgb, labels = self._read_pt(train_path)

        self.label = torch.as_tensor(labels, dtype=torch.long)
        if self.label.is_floating_point():
            self.label = self.label.long()
        self.label = self.label.contiguous()
        self.label.share_memory_()
        self.sar_name = [os.path.basename(path) for path in self.sar]
        self.rgb_name = [os.path.basename(path) for path in self.rgb]

        self.h, self.w = 64, 64
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32),
            torch.arange(self.w, dtype=torch.float32),
            indexing="ij",
        )
        self._grid_y = yy.share_memory_()
        self._grid_x = xx.share_memory_()
        self._heatmap_cache = self._build_heatmap_cache()

    @staticmethod
    def _read_pt(pt_path: str) -> tuple:
        data = torch.load(pt_path, map_location="cpu")
        return data["sar_path"], data["rgb_path"], data["label"]

    @staticmethod
    def _read_csv(csv_path: str) -> pd:
        usecols = ["SAR_Image", "RGB_Image", "Label"]
        dtype = {"SAR_Image": "string", "RGB_Image": "string", "Label": "string"}

        df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype, engine="pyarrow",)
        sar = df.iloc[:, 0].tolist()
        rgb = df.iloc[:, 1].tolist()
        label = [json.loads(label) for label in df.iloc[:, 2].tolist()]
        return sar, rgb, label

    def __getitem__(self, idx):
        sar_img = self._load_image(self.sar[idx])
        rgb_img = self._load_image(self.rgb[idx])
        sar_img_name = self.sar_name[idx]
        rgb_img_name = self.rgb_name[idx]
        label = self.label[idx]

        # Stage 1 alignment keeps geometric augmentation; Stage 2 matching
        # disables it (rotate/flip aug empirically too strong, hurts matching
        # convergence). Mirrors the same gate in _tm_webdataset.py so both
        # dataloader backends agree.
        # 与 _tm_webdataset.py 一致：Stage 2 关闭几何增强。
        if self.is_train and not self.is_stage_2:
            sar_img, rgb_img, label = TrainAug(
                sar_img, rgb_img, label, rgb_img.shape[-2:], (self.h, self.w)
            )

        x, y = self._sanitize_label(label, rgb_img.shape[-2:])

        # Stage 1 (pre-train): crop RGB to the matched 64x64 patch.
        # Stage 2: keep the full RGB search region.
        if self.is_pre_train and not self.is_stage_2:
            rgb_img = rgb_img[:, y:y + self.h, x:x + self.w]

        sar_img = self.preprocess_image(sar_img, True)
        rgb_img = self.preprocess_image(rgb_img, False)

        if self.is_stage_2:
            label_out = torch.tensor([x, y], dtype=torch.float32)
        else:
            label_out = self.label_handler(x, y)

        return sar_img, rgb_img, label_out, sar_img_name, rgb_img_name

    def __len__(self):
        return len(self.label)

    def calculate_mean_std(self, img: torch.Tensor):
        mean = img.mean(dim=(1, 2), keepdim=True)
        std = img.std(dim=(1, 2), keepdim=True).clamp_min(1e-7)
        return mean, std

    def preprocess_image(self, img: torch.Tensor, duplicate_channels: bool = True):
        if img.dtype != torch.float32:
            img = img.float()
        img = img.div(255.0)

        if img.shape[0] == 1 and duplicate_channels:
            img = img.repeat(3, 1, 1)

        mean, std = self.calculate_mean_std(img)
        img = (img - mean) / std
        return img

    def _sanitize_label(self, label: torch.Tensor, image_hw: tuple[int, int]) -> tuple[int, int]:
        x = int(label[0])
        y = int(label[1])
        max_x = max(0, image_hw[1] - self.w)
        max_y = max(0, image_hw[0] - self.h)
        x = min(max(x, 0), max_x)
        y = min(max(y, 0), max_y)
        return x, y

    def label_handler(self, x: int, y: int) -> torch.Tensor:
        if 0 <= x < self.w and 0 <= y < self.h:
            idx = y * self.w + x
            return self._heatmap_cache[idx]
        return self._generate_heatmap(x, y)

    def _build_heatmap_cache(self) -> torch.Tensor:
        grid = torch.stack([self._grid_x, self._grid_y], dim=0)
        coords = grid.view(2, -1).permute(1, 0)
        xx = grid[0].unsqueeze(0)
        yy = grid[1].unsqueeze(0)
        target_x = coords[:, 0].view(-1, 1, 1)
        target_y = coords[:, 1].view(-1, 1, 1)
        dist = torch.sqrt((xx - target_x) ** 2 + (yy - target_y) ** 2)
        heatmaps = torch.exp(-dist / 1.2)
        heatmaps /= heatmaps.sum(dim=(1, 2), keepdim=True)
        return heatmaps.share_memory_()

    def _generate_heatmap(self, x: int, y: int) -> torch.Tensor:
        dist = torch.sqrt((self._grid_x - float(x)) ** 2 + (self._grid_y - float(y)) ** 2)
        heatmap = torch.exp(-dist / 1.2)
        heatmap /= heatmap.sum()
        return heatmap

    def _load_image(self, path: str) -> torch.Tensor:
        return read_image(path)


def main():
    data = TM_Dataset(r"./Datasets/image_debug.csv", train=True)
    print("len(data):", len(data))

    training_dataloader = torch.utils.data.DataLoader(
        data, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, sampler=None
    )
    print(training_dataloader)
    for iteration, batch_data in enumerate(training_dataloader):
        print(iteration)
        print(batch_data[0].size())
        print(batch_data[1].size())
        print(batch_data[2].size())
        print("___________")


if __name__ == "__main__":
    main()
