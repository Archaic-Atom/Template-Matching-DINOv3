# -*- coding: utf-8 -*-
import json
import torch
import pandas as pd


def _read_csv(csv_path: str) -> pd:
    usecols = ["SAR_Image", "RGB_Image", "Label"]
    dtype = {"SAR_Image": "string", "RGB_Image": "string", "Label": "string"}
    return pd.read_csv(csv_path, usecols=usecols, dtype=dtype, engine="pyarrow",)


cvs_path = r"./Datasets/image_test.csv"
pt_path = r"./Datasets/image_test.pt"


df = _read_csv(cvs_path)
sar = df.iloc[:, 0].tolist()
rgb = df.iloc[:, 1].tolist()
label = [json.loads(label) for label in df.iloc[:, 2].tolist()]


torch.save(
    {
        "sar_path": sar,
        "rgb_path": rgb,
        "label": label
    },
    pt_path,
)
print("saved:", pt_path)


data = torch.load(pt_path, map_location="cpu")
sar_paths = data["sar_path"]
rgb_paths = data["rgb_path"]
label = data["label"]

print(sar_paths[1], rgb_paths[1], label[1])
print(label[1][1], label[1][0])
