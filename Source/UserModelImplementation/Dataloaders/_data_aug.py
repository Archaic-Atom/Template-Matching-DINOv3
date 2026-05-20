# -*- coding: utf-8 -*-
import random
from typing import Tuple

import torch
from torch import Tensor

ROTATE_PROB = 0.5
FLIP_PROB = 0.5


def TrainAug(
        img1: Tensor,
        img2: Tensor,
        label: Tensor,
        image_hw: Tuple[int, int],
        crop_hw: Tuple[int, int]) -> Tuple[Tensor, Tensor, Tensor]:
    new_label = label
    if random.random() < ROTATE_PROB:
        img1, img2, new_label = RandomRotate(img1, img2, new_label, image_hw, crop_hw)
    if random.random() < FLIP_PROB:
        img1, img2, new_label = RandomFlip(img1, img2, new_label, image_hw, crop_hw)
    return img1, img2, new_label


def RandomRotate(
        img1: Tensor,
        img2: Tensor,
        label: Tensor,
        image_hw: Tuple[int, int],
        crop_hw: Tuple[int, int]) -> Tuple[Tensor, Tensor, Tensor]:
    k = random.randint(1, 3)
    rotated_img1 = torch.rot90(img1, k, dims=(-2, -1))
    rotated_img2 = torch.rot90(img2, k, dims=(-2, -1))

    height, width = image_hw
    crop_h, crop_w = crop_hw
    x = int(label[0])
    y = int(label[1])

    if k == 1:
        new_x = y
        new_y = width - crop_w - x
    elif k == 2:
        new_x = width - crop_w - x
        new_y = height - crop_h - y
    else:
        new_x = height - crop_h - y
        new_y = x

    return rotated_img1, rotated_img2, label.new_tensor((new_x, new_y))


def RandomFlip(
        img1: Tensor,
        img2: Tensor,
        label: Tensor,
        image_hw: Tuple[int, int],
        crop_hw: Tuple[int, int]) -> Tuple[Tensor, Tensor, Tensor]:
    height, width = image_hw
    crop_h, crop_w = crop_hw
    x = int(label[0])
    y = int(label[1])

    if random.random() < 0.5:
        img1 = torch.flip(img1, dims=(-2,))
        img2 = torch.flip(img2, dims=(-2,))
        y = height - crop_h - y

    if random.random() < 0.5:
        img1 = torch.flip(img1, dims=(-1,))
        img2 = torch.flip(img2, dims=(-1,))
        x = width - crop_w - x

    return img1, img2, label.new_tensor((x, y))
