# -*- coding: utf-8 -*-
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


try:
    from ._data_aug import TrainAug
except ImportError:
    from _data_aug import TrainAug


class TM_Dataset(Dataset):
    def __init__(self, csv_path: str, train: bool = True):
        self.is_train = train
        self.transform = transforms.ToTensor()

        df = pd.read_csv(csv_path)

        self.sar = df.iloc[:, 0].tolist()
        self.rgb = df.iloc[:, 1].tolist()
        self.label = [json.loads(label) for label in df.iloc[:, 2].tolist()]

        h, w = 64, 64
        self.y = torch.arange(h).view(-1, 1)
        self.x = torch.arange(w).view(1, -1)

    def __getitem__(self, idx):
        sar_img = Image.open(self.sar[idx])
        rgb_img = Image.open(self.rgb[idx])
        sar_img_name = self.sar[idx].split("/")[-1]
        rgb_img_name = self.rgb[idx].split("/")[-1]
        label = self.label[idx]

        if self.is_train:
            sar_img, rgb_img, label = TrainAug(sar_img, rgb_img, label)

        sar_img = self.preprocess_image(sar_img, True)
        rgb_img = self.preprocess_image(rgb_img, False)
        label = self.label_handler(label)

        return sar_img, rgb_img, label, sar_img_name, rgb_img_name

    def __len__(self):
        return len(self.label)

    def calculate_mean_std(self, img: torch.Tensor):
        mean = img.mean([1, 2])
        std = img.std([1, 2])
        return mean, std

    def preprocess_image(self, img, duplicate_channels: bool = True):
        if isinstance(img, np.ndarray):
            img = np.copy(img)
        img = self.transform(img)

        if img.shape[0] == 1 and duplicate_channels:
            img = img.repeat(3, 1, 1)

        mean, std = self.calculate_mean_std(img)
        std = [max(i, 1e-7) for i in std]
        img = transforms.Normalize(mean=mean.tolist(), std=std)(img)
        return img

    def label_handler(self, label):
        heatmap = self.GenHeatmap(label)
        heatmap = torch.Tensor(heatmap)
        return heatmap

    def GenHeatmap(self, label: list):
        def Softmax(x):
            exp_x = torch.exp(x)
            sum_exp_x = torch.sum(exp_x)
            y = exp_x.div_(sum_exp_x)
            return y

        WEIGHT_DISTANCE = 1.2
        d = (self.x - label[0]) ** 2 + (self.y - label[1]) ** 2
        heatmap = -torch.sqrt(d) / WEIGHT_DISTANCE
        heatmap = Softmax(heatmap)
        return heatmap


def main():
    data = TM_Dataset(r"./Datasets/image_test.csv", train=True)
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
