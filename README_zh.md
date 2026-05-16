# Template-Matching-DINOv3（中文文档）

![Python](https://img.shields.io/badge/python-%3E=3.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

一个基于 **DINOv3** 的 **SAR ↔ RGB 跨模态模板匹配** 框架：以冻结的 DINOv3 ViT
作为 RGB 特征教师，训练一个轻量 SAR CNN 编码器，将 SAR 特征对齐到 DINO 的
特征空间。训练循环、DDP、日志与 checkpoint 由
[JackFramework](https://github.com/Archaic-Atom/JackFramework) 负责。

> English version: [README.md](README.md)。

---

## 1. 概述

输入一张 64×64 的 SAR 小图和一张较大的光学 RGB 图，目标是在 RGB 图中定位
SAR 小图的位置，并产生逐像素的定位 heatmap。

整个流水线分为两个阶段：

1. **特征对齐预训练**（当前阶段，`--pre_trained True`）：冻结 DINOv3 RGB
   编码器，只训练 SAR 编码器，使其在 stride-16 的特征图与 DINO RGB 特征图
   在对应位置上对齐。
2. **匹配 / 定位微调**（后续阶段）：复用对齐后的编码器，学习一个匹配头，
   回归出以真实位置为峰值的 heatmap。

### 方法要点

- **RGB 分支**：`dinov3_vitl16`，使用 `sat493m` 卫星预训练权重，**参数冻结**。
  通过 `get_intermediate_layers(..., reshape=True)` 取出
  `[4, 11, 17, 23]` 层的 token 特征。
- **SAR 分支**：`SarResNet16GN`，类 ResNet-34 的 CNN，用 **GroupNorm**
  （SAR 的 batch 统计不稳，GN 比 BN 更合适）下采样到 stride-16，尾端 1×1
  `ProjHead` 投影到 DINO 的 1024 维并 L2 归一化。
- **对齐损失**（`alignment_loss`）：
  - **Soft-window PatchNCE**（`tau=0.07`、`radius=1`，即 3×3 正样本窗），
    负样本取自整个 batch。
  - **全局 cosine loss**（对 mean-pool 后的特征，权重 `0.1`）。
- **标签 / heatmap**：以真实 `(x, y)` 为中心的高斯 heatmap，按位置预计算并
  缓存以加速。heatmap 供后续定位阶段使用；对齐阶段本身是自监督的。

---

## 2. 环境要求

### 硬件
- NVIDIA GPU（CUDA）。多卡训练使用 PyTorch DDP。

### 软件
- Python ≥ 3.8
- PyTorch（与你的 CUDA 驱动匹配，建议 2.x 近期版本）
- `JackFramework`（从 GitHub 单独安装）
- `numpy`、`pandas`、`pyarrow`、`Pillow`、`opencv-python`、`torchvision`

---

## 3. 安装

```bash
# 1. 克隆本仓库并拉取 DINOv3 子模块
git clone --recurse-submodules <本仓库地址>
cd Template-Matching-DINOv3

# 2. 安装 JackFramework
git clone https://github.com/Archaic-Atom/JackFramework.git
cd JackFramework && ./install.sh && cd ..

# 3. 将 DINOv3 卫星预训练权重放到 Weights/ 目录下
#    默认文件名：dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

默认路径定义在 `Source/UserModelImplementation/user_define.py`：

```python
ROOT_PATH   = '/data/raozhibo/Documents/Programs/Template-Matching-DINOv3/'
DINOv3_PATH = 'Source/Libs/dinov3/'
WEIGHTS_PATH = 'Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
```

请按你自己的环境修改 `ROOT_PATH`，或在命令行通过 `--root_path` 覆盖。

---

## 4. 数据格式

数据集类 `TM_Dataset` 同时支持 CSV 列表和 PyTorch `.pt` 列表。

### CSV
列名为：`SAR_Image, RGB_Image, Label`，其中 `Label` 为 JSON 编码的
`[x, y]`：

```csv
SAR_Image,RGB_Image,Label
/path/sar_0001.png,/path/rgb_0001.png,"[128, 64]"
...
```

### `.pt`
一个字典，包含键：`sar_path: List[str]`、`rgb_path: List[str]`、
`label: List[[x, y]]`。可用 `Source/Tools/csv2pt.py` 把 CSV 转成 `.pt`。

### patch 大小
SAR patch 为 64×64。预训练阶段会在 RGB 图的 `(x, y)` 位置裁出 64×64，与
SAR patch 同位置对齐；定位阶段则使用完整 RGB。

---

## 5. 训练

入口脚本位于 `Scripts/`：

- `start_debug_dataset_model.sh` — 小规模调试（2 卡，5 epoch）。
- `start_train_dataset_model.sh`  — 完整训练（4 卡 DDP，50 epoch）。
- `start_test_dataset_model.sh`   — 推理。

最小化的训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u Source/main.py \
  --mode train \
  --batchSize 4 \
  --gpu 2 \
  --trainListPath ./Datasets/image_test.pt \
  --dataloaderNum 8 \
  --maxEpochs 50 \
  --lr 0.001 \
  --dist True \
  --modelName TM_Model_A \
  --dataset tmdataset \
  --pre_trained True \
  --log ./log/ \
  --outputDir ./result/ \
  --modelDir ./Checkpoint/
```

`user_interface.py` 新增的用户参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--root_path` | 见 `user_define.py` | 工程根目录，用于拼出 `dinov3_path` 与 `pre_trained_weights_path`。 |
| `--dinov3_path` | `Source/Libs/dinov3/` | 本地 DINOv3 hub 目录（子模块）。 |
| `--pre_trained_weights_path` | `Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | DINOv3 权重文件。 |
| `--pre_trained` | `True` | 为 `True` 时只跑对齐预训练阶段。 |

### 保存内容
checkpoint 只保存 `trainable_tail`（即 SAR encoder + ProjHead），不会重复
存储冻结的 DINOv3 权重。

---

## 6. 目录结构

```
Template-Matching-DINOv3
├── Datasets/                        # 训练/测试列表 (csv / pt)
├── Scripts/                         # shell 入口脚本
│   ├── start_debug_dataset_model.sh
│   ├── start_train_dataset_model.sh
│   └── start_test_dataset_model.sh
├── Source/
│   ├── main.py                      # JackFramework 入口
│   ├── Libs/dinov3/                 # DINOv3 子模块
│   ├── Tools/                       # csv2pt、路径工具
│   └── UserModelImplementation/
│       ├── user_define.py           # 全局路径 / 开关
│       ├── user_interface.py        # CLI & 模型/数据接口组装
│       ├── Models/TM_Model_A/
│       │   ├── inference.py         # JackFramework 模型接口
│       │   ├── Nets/
│       │   │   ├── model.py             # TMModelANet（RGB 教师 + SAR 学生）
│       │   │   ├── _create_backbone.py  # DINOv3/DINOv2 加载器
│       │   │   └── _create_sar_encoder.py  # SarResNet16GN + ProjHead
│       │   └── Loss/
│       │       └── alignment_loss_func.py  # soft-window PatchNCE + cosine
│       └── Dataloaders/
│           ├── tm_dataloader.py     # JackFramework 数据接口
│           ├── _tm_datasets.py      # TM_Dataset（csv/pt、heatmap 缓存）
│           └── _data_aug.py         # 旋转 / 翻转增强
├── Weights/                         # DINOv3 权重（不纳入版本控制）
└── README.md
```

---

## 7. License

MIT，详见 [`LICENSE`](LICENSE)。DINOv3 按其自身 license 再分发，请以其
上游仓库为准。
