# -*- coding: utf-8 -*-
import random
import numpy as np


def TrainAug(img1, img2, label):
    img1, img2, label = RandomRotate(img1, img2, label)
    img1, img2, label = RandomFlip(img1, img2, label)
    return img1, img2, label


def RandomRotate(img1, img2, label):

    rotate_prob = np.random.randint(low=1, high=100)

    if rotate_prob > 101:
        return img1, img2, label

    k = random.choice((1, 2, 3, 4))
    new_img1 = np.rot90(img1, k)
    new_img2 = np.rot90(img2, k)

    x = label[1]
    y = label[0]

    if k == 1:
        new_x = 128 - 64 - y
        new_y = x
    elif k == 2:
        new_x = 128 - 64 - x
        new_y = 128 - 64 - y
    elif k == 3:
        new_x = y
        new_y = 128 - 64 - x
    elif k == 4:
        new_x = x
        new_y = y
    else:
        print(f"{k}")
        print("error")

    return new_img1, new_img2, [new_y, new_x]


def RandomFlip(img1, img2, label):

    flip_prob = np.random.randint(low=1, high=100)
    if flip_prob > 101:
        return img1, img2, label

    choice_H = random.choice((0, 1))
    choice_W = random.choice((0, 1))

    x = label[0]
    y = label[1]

    if choice_H == 1:
        new_img1 = np.flip(img1, 0)
        new_img2 = np.flip(img2, 0)
        new_x = x
        new_y = 128 - 64 - y
    elif choice_H == 0:
        new_img1 = img1
        new_img2 = img2
        new_x = x
        new_y = y
    else:
        print("chioce_H:{}".format(choice_H))
        print("error")

    if choice_W == 1:
        new_img1 = np.flip(new_img1, 1)
        new_img2 = np.flip(new_img2, 1)
        new_x = 128 - 64 - new_x
        new_y = new_y
    elif choice_W == 0:
        new_img1 = new_img1
        new_img2 = new_img2
        new_x = new_x
        new_y = new_y
    else:
        print("chioce_W:{}".format(choice_W))
        print("error")

    return new_img1, new_img2, [new_x, new_y]


if __name__ == '__main__':
    img1 = np.random.rand(128, 128)
    img2 = np.random.rand(64, 64)
    label = [24, 8]
    img1, img2, label = TrainAug(img1, img2, label)
    print(type(img1))
    print(img1.shape)
    print(img2.shape)
    print(label)
