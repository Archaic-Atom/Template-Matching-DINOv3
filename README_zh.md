# Template-Matching-DINOv3 (DOSR-Net) · 中文文档

![Python](https://img.shields.io/badge/python-%3E=3.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

**DOSR-Net** 是面向遥感图像配准的 **SAR ↔ 光学跨模态模板匹配**框架（TGRS 2026）。
它把一小块 SAR 模板定位到较大的光学参考图块内：用**冻结的 DINOv3 ViT-L/16**
（卫星预训练）作为光学骨干，配一个**联合训练的 SAR CNN 编码器**，再用
**FeatUp / JBU** 图像引导上采样把两路特征图提分辨率，最后通过**由粗到细的级联**
归一化互相关完成匹配。整个模型族由 YAML 配置驱动，训练基于
[JackFramework](https://github.com/Archaic-Atom/JackFramework)（负责训练循环、DDP
启动、日志与 checkpoint 读写）。

仓库以**分析驱动的消融模型族 `TM_Model_A..F`** 组织，主方法（提出的模型）是
**`TM_Model_C`**——一个直接训练的级联匹配头。论文的一个结论是：对 SAR 编码器做
对比式 Stage-1 *预对齐*，相比直接训练**并不会**提升域内匹配效果；多个变体的存在
正是为了量化这一点（见[模型变体](#模型变体)）。

> 面向开发者的权威详解（入口链路、config-vs-CLI 优先级、dataloader 约定、级联头内部
> 结构、已知坑点清单）请见 **[`CODE_STRUCTURE.md`](CODE_STRUCTURE.md)**。本文档是
> 代码使用层面的概览，细节链接到该文档而不重复展开。

---

## 主要特性

- **256/192 几何。**光学参考图块 **256×256**，SAR 模板 **192×192**；真值标签
  `(x, y)` 是 SAR 模板在光学图块内的**左上角坐标**，因此位移范围为 `[0, 64]`。
  （由旧的 128/64 几何迁移而来，以匹配 ME-SSM 评测协议。）
- **冻结的 DINOv3 ViT-L/16 光学骨干**（`dinov3_vitl16`，卫星预训练 `sat493m` 权重，
  1024 维 stride-16 特征），搭配**联合训练的 `SarResNet16GN`** SAR 编码器
  （ResNet 风格，GroupNorm，无 BatchNorm）。
- **FeatUp / JBU 上采样。**图像引导的 `JBULearnedRange` 上采样器
  （`featup.upsamplers`）将 stride-16 特征提到 stride-4/8 供匹配使用；端到端训练
  （**不复用** FeatUp 的 DINOv2 预训练权重）。
- **由粗到细级联匹配。**归一化互相关（SAR 模板作为卷积核，光学搜索区作为输入）：
  先在 stride-4 + stride-8 做粗匹配，再围绕脱离梯度的粗匹配 argmax 做窗口化的
  stride-2、stride-1 细化。
- **JackFramework + WebDataset 训练。**面向约 1320 万样本训练集的可迭代
  WebDataset 分片管线，含自定义的 DDP 均衡分片切分、大样本 shuffle 缓冲，以及
  评测用的 map-style 后端。
- **`TM_Model_A..F` 消融族**，分别隔离 Stage-1 预对齐作为初始化、级联 vs 标准细化头、
  亚像素解码、以及一个 learned-inner-product（VICReg）探针。
- **跨数据集零样本评测**适配器：**WHU-OPT-SAR**（跨传感器）、**QXS-SAROPT**
  （跨分辨率），外加可控的 GSD 扫描工具。
- **分析 / 探针工具**（`Source/Tools/`、`Source/Tools/probes/`），产出论文分析部分的
  机理数字（相似性存在于何处、对照实验组、有效秩轨迹、oracle 上界等）。

---

## 仓库结构

```
Template-Matching-DINOv3/
├── CODE_STRUCTURE.md              # 权威开发者文档（建议先读）
├── Source/
│   ├── main.py                    # 唯一入口（python Source/main.py ...）
│   ├── Libs/dinov3/               # DINOv3 git 子模块（冻结光学骨干）
│   ├── UserModelImplementation/
│   │   ├── user_interface.py      # CLI 接口 + dataset/model zoo 分发
│   │   ├── user_define.py         # ROOT_PATH / DINOv3 仓库 + 权重路径常量
│   │   ├── Models/
│   │   │   ├── __init__.py        # model_zoo：TM_Model_A..F → 接口类
│   │   │   ├── builder.py         # YAML → ComposedTM 组装
│   │   │   ├── _interface_base.py # 通用 JackFramework 模型 handler
│   │   │   ├── TM_Model_{A..F}/   # 各变体 config.yaml（+ 极薄 __init__.py）
│   │   │   ├── _components/        # 骨干、JBU、匹配头、损失、解码
│   │   │   └── _legacy/legacy_A/   # A 的原始手写实现（仅供参考）
│   │   └── Dataloaders/            # TMDataloader（map-style + WebDataset 双后端）
│   └── Tools/                      # 评测/指标/探针脚本（+ probes/）
├── Scripts/                        # 训练 / 评测 / 造数据 启动脚本（256/192）
│   └── legacy_128_64/              # 已弃用的 128/64 几何脚本
├── Datasets/                       # 仅一个过时示例被跟踪；真实清单在服务器上
├── Weights/                        # DINOv3 权重（已 gitignore，需单独下载）
├── README.md / README_zh.md
└── LICENSE
```

完整且最新的目录树见 **[`CODE_STRUCTURE.md` §1](CODE_STRUCTURE.md)**。

---

## 模型变体

六个变体共享**相同骨干**（冻结 DINOv3 ViT-L/16 + `SarResNet16GN`）、**相同的
stride-4 + stride-8 JBU 粗匹配**、以及 **256/192 几何**（`matching_dim = 256`、
`sar_size = 192`）。每个 `TM_Model_X/` 都是只设置 `CONFIG_PATH` 的极薄
`ComponentBasedInterface` 子类，行为全在其 `config.yaml`。变体在三个维度上不同：
**Stage-1 目标**、**Stage-2 是否冻结 SAR 编码器**、**细化/解码头**。

| 变体 | 匹配头 | Stage-1（`load_pretrained_sar`） | Stage-2 SAR（`freeze_sar_encoder`） | Stage-2 损失 | 解码 | 隔离的因素 |
|---|---|---|---|---|---|---|
| **A** | `StandardMatchingHeadJBU` | `true`（从 Stage-1 对齐权重热启动） | `false`（继续训练） | `two_stage_matching_loss` | hard argmax | Stage-1 预对齐**作为初始化**（与 B 唯一差别） |
| **B** | `StandardMatchingHeadJBU` | `false`（随机 SAR 初始化） | `false` | `two_stage_matching_loss` | hard argmax | 无 Stage-1 的对照 / A 的基线 |
| **C** | `CascadeHeadJBU` | `false`（直接训练，无 Stage-1） | `false`（联合） | `cascade_loss` | hard argmax | **提出的方法**——直接训练的级联 |
| **D** | `StandardMatchingHeadJBU` | `true` | `true`（冻结） | `two_stage_matching_loss` | `soft_subpixel` | A 头上加亚像素解码（复用 A 权重，仅评测） |
| **E** | `CascadeHeadJBU` | `true` * | `true` * | `cascade_loss` | `soft_subpixel` | 级联 + 亚像素；**在 C 的权重上评测** |
| **F** | `StandardMatchingHeadJBU`（Stage-1 加 `AlignmentIPHead`） | `true` | `true`（冻结） | `two_stage_matching_loss` | hard argmax | learned-inner-product Stage-1（W_s/W_d + VICReg 秩下限）+ 冻结编码器 Stage-2 |

**权重复用：**D 复用 A 的权重、E 复用 C 的权重——两者只在评测期改动 `argmax_type`
（亚像素），并保持 `matching.args` 逐字节一致以便 state_dict 对齐。注意只有
`matching.args` 逐字节一致、而非整份 config：与 E 类似，D 自己的训练段也不同于它借用的
模型（D 设 `freeze_sar_encoder: true`，而 A 为 `false`），所以 D 的设计意图是**在 A 的
权重上评测**，而非按它自己的 config 训练。亚像素在代码里存在
（`_components/sub_pixel.py`），但**不是**论文的核心贡献。

> \* **TODO / 注意：**E 的架构与 state_dict 与 C 逐字节兼容，设计意图是**在 C 的权重上**
> 用亚像素解码评测。但 E 自己 `config.yaml` 的训练段（`load_pretrained_sar: true`、
> `freeze_sar_encoder: true`、非零 warmup）相对 C 转向直接训练后已**过时**，按原样启动
> 并不能复现 C。请把 E 理解为“C 的架构 + 亚像素解码、在 C 权重上评测”；若要按 E 自己的
> 配置训练，请先与作者确认意图。

成对对比：**A vs B** 隔离 Stage-1 作为初始化；**C vs B** 隔离级联 vs 标准细化头；
**A vs D**、**C vs E** 隔离亚像素解码；**F** 隔离一个受保护的强 Stage-1 目标。2×2 的
“frozen-arms”因果实验（初始化 × 冻结）见 [`CODE_STRUCTURE.md §7`](CODE_STRUCTURE.md)。

---

## 安装

> **本仓库不含 `requirements.txt` / `environment.yml` / `setup.py`。**下列依赖是从
> import 和启动脚本（均指向训练服务器上名为 **`stereoA`** 的 conda 环境）反推得到的，
> 版本未在此固定——请装入较新的 PyTorch 2.x 环境。

```bash
# 1. 连同 DINOv3 子模块一起克隆（冻结光学骨干）。
git clone --recurse-submodules <this-repo-url>
cd Template-Matching-DINOv3
# 没有 SSH 密钥？子模块 URL 是 git@github.com:facebookresearch/dinov3.git，
# 可改写为 https://github.com/facebookresearch/dinov3.git。

# 2. 安装 JackFramework（未内置于本仓库；负责训练循环 / DDP / I/O）。
git clone https://github.com/Archaic-Atom/JackFramework.git
cd JackFramework && ./install.sh && cd ..

# 3. 安装 FeatUp（必需——外部依赖，不是子模块）。
#    JBU 上采样器（当前所有变体都用）会 import
#    `featup.upsamplers.JBULearnedRange`。从 facebookresearch/FeatUp 安装。

# 4. 其它 Python 依赖（装入同一环境）：
#    torch, torchvision, numpy, pandas, Pillow, webdataset
#    （可迭代训练路径需要 webdataset）。

# 5. 把 DINOv3 卫星预训练权重放到 Weights/（已 gitignore）：
#    Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth   （DINOv3 ViT-L/16，约 1.2 GB）
```

**路径常量**在 `Source/UserModelImplementation/user_define.py`：
`ROOT_PATH`（默认 `/data/raozhibo/Documents/Programs/Template-Matching-DINOv3/`）、
`DINOv3_PATH = Source/Libs/dinov3/`、
`WEIGHTS_PATH = Weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`。请按机器修改
`ROOT_PATH`，或在启动时用 `--root_path` 覆盖（它会被替换进各 `config.yaml` 的
`${root_path}` 占位符，用于 DINO 仓库/权重路径）。

**DDP 注意：**由于光学骨干（部分变体下还有 SAR 编码器）被冻结，JackFramework 的 DDP
封装需要 `find_unused_parameters=True`。该补丁只存在于训练服务器上的 JackFramework
安装中——见 [`CODE_STRUCTURE.md §2.5`](CODE_STRUCTURE.md)。

---

## 数据准备

### 域内 SEN1-2（256/192）WebDataset 分片

生产用造数据脚本是 **`Scripts/make_opt256_sar192_shards.py`**（`Source/Tools/` 下有一份
逐字节相同的副本）。它把共配准的 256×256 SEN1-2（Sentinel-1 SAR `s1_*` + Sentinel-2
光学 `s2_*`，四季）直接裁剪打包成 WebDataset `.tar` 分片——无中间 PNG：

- **光学参考** = 原始 256×256 的 `s2` PNG 字节，逐字节复制。
- **SAR 模板** = `s1` 在左上角 `(cx, ry)` 裁剪到 192×192 后重新编码为 PNG。
- 每个样本 = 共享同一 key 的三个 tar 条目：
  `<key>.sar.png`（192×192 灰度）、`<key>.rgb.png`（256×256 RGB）、
  `<key>.label.cls`（ASCII `"<cx> <ry>"`——左上角，**非**中心，**未**归一化）。

训练/测试在**场景/图块级别互不相交**（按季固定种子切分，无场景泄漏）。偏移用于样本扩增：
**TEST** 用密集的 stride-4 网格（每对 `17×17 = 289` 个偏移）；**TRAIN** 用随机偏移
（默认每对 48 个）。

```bash
# 先造 test（快），再造 train（适合 nohup 的封装脚本）：
bash Scripts/run_gen_opt256_sar192.sh
# 或直接：
python Scripts/make_opt256_sar192_shards.py --split test  --workers 16
python Scripts/make_opt256_sar192_shards.py --split train --workers 16
```

输出 `TM_shards_opt256_sar192_{train,test}/`，并附一个**分片清单 CSV**
`shard_path.csv`（表头 `path,file_num`）——可迭代后端据此得知每个分片的样本数，并以
`sum(file_num)` 作为权威的磁盘样本总数。默认设置下，测试集约 224 万样本
（7740 对 × 289），完整训练集约 1318 万样本；训练集精确数量取决于 SEN1-2 目录树。

一个小的 **map-style** 评测 CSV（`SAR_Image,RGB_Image,Label`，`Label = "[cx, ry]"`
JSON）由 **`Scripts/make_2k_eval_256.py`** 从同一互不相交的 `test_pairs.txt` 生成，
默认 2000 个样本。

### 跨数据集零样本适配器

**`Source/Tools/cross_dataset_adapter.py`** 把外部 SAR-光学数据集重采样到目标 GSD，
并产出 SEN1-2 形状的 256/192 评测对（`SAR_SIZE=192`、`RGB_SIZE=256`；注意模块/类
docstring 仍写“64/128”——那是过时文字，以代码为准）：

| 适配器（`--dataset`） | 来源 | 原生 → 目标 GSD | 角色 |
|---|---|---|---|
| `whu_opt_sar` | WHU-OPT-SAR（GF-1 + GF-3） | 5 m → 10 m（下采样） | 跨传感器 |
| `qxs_saropt`  | QXS-SAROPT（GF-3 + Google Earth，原生 256×256） | 1 m → 1 m（不重采样） | 跨分辨率 |
| `sarptical`   | SARptical（TerraSAR-X + 航拍） | 1 m → 10 m | 跨传感器（辅助） |

```bash
bash Scripts/run_regen_cross_256.sh     # 在 256/192 下重造 WHU + QXS
```

每次运行写出 `sar/`、`rgb/`、一个 `manifest.pt`（供 `Scripts/pack_to_webdataset.py` 使用）、
一个 `cross_eval.csv`（`SAR_Image,RGB_Image,Label`，与域内 map-style 评测 CSV 同 schema，
可直接被 `--mode test` 消费），以及一个 `meta.json` 溯源文件。`--target_res_m`
（仅下采样、有保护）驱动可控 GSD 扫描（`Scripts/run_whu_res_sweep.sh`）。

### 清单 CSV 格式（切勿混淆）

| | 分片清单 CSV（可迭代后端） | 路径清单 CSV（map-style 后端） |
|---|---|---|
| 表头 | `path,file_num` | `SAR_Image,RGB_Image,Label` |
| 每行 = | 一个 `.tar` 分片 | 一个样本 |
| `Label` | — | JSON `[x, y]` 字符串 |

> `Datasets/` 下唯一被 git 跟踪的文件是 `dataset_example_training_list.csv`，它是一个
> **过时的立体视差示例**（`left_img,right_img,gt_disp`）——**不是** TM 清单。真实清单
> （`shard_train_256_full.csv`、子集热启动清单、各评测 CSV）**并未提交**，只存在于
> 训练服务器上。注意它们**并不**被 `.gitignore` 覆盖（只有 `Datasets/image_*.csv|.pt`
> 与 `Datasets/*.txt` 被覆盖），所以小心不要误提交某个分片/评测清单。

---

## 训练

**没有单独的 `train.py`**——一切都通过 `python Source/main.py <flags>` 运行，由
`--mode train|test` 选择行为。CLI 有两类 flag：少量**项目 flag**（如 `--is_stage_2`、
`--sar_size`、`--root_path`、`--shuffle_buffer`）和 **JackFramework 内置 flag**
（`--mode --modelName --dataset --dist --gpu --batchSize --port --lr --maxEpochs
--imgNum --trainListPath --modelDir ...`）。

> **重要的优先级规则：**在 YAML 路径下，大多数架构/损失 CLI flag
> （`--matching_dim`、`--fine_dim`、`--fine_margin`、`--match_w*`、`--fine_warmup_epochs`）
> **是失效的**——头维度与损失权重来自各变体的 `config.yaml`。脚本仍传它们只是“为了对齐
> 写法”。**真正生效**的包括 `--is_stage_2`、`--lr`、`--maxEpochs`、`--root_path`、
> `--sar_size`（dataloader 侧）。见 [`CODE_STRUCTURE.md §2.3 / §11`](CODE_STRUCTURE.md)。
>
> **`sar_size` 双通道坑：**匹配头从 `config.yaml` 读 `architecture.matching.sar_size`
> （写死的 `192`，CLI 无法替换），dataloader 从 CLI 读 `--sar_size`。**两者必须一致。**
> 始终传 `--sar_size 192` 并保持 config 为 `192`。
>
> **`--imgNum` 只是进度条分母**，不是真实 epoch 长度（WebDataset 路径用
> `with_epoch(per_worker_samples)`，由清单的 `real_total` 推出）。脚本里固化的约定是
> `--imgNum == sum(file_num)`，启动脚本用 `awk` 对清单求和得到。

### 提出的方法（`TM_Model_C`）——两阶段流程

`TM_Model_C` 直接训练（随机 SAR 初始化、无 Stage-1、SAR 编码器与级联联合训练）。科学
运行分**两阶段**，均通过 `Scripts/run_train_C_direct.sh`：

```bash
# 阶段一 —— 子集热启动（subset40 清单 ~1.34M，4 卡 DDP，bs10，5 epoch）。
bash Scripts/run_train_C_direct.sh sub40

# 阶段二 —— 全量 ×3 epoch，从阶段一 checkpoint 热启动。
# 内部：strip_stage1_opt.py 丢掉优化器状态并把 epoch 重置为 -1，使 JF 从干净的
# epoch 0 开始，再用手写的 checkpoint.list 固定它。
bash Scripts/run_train_C_direct.sh full_resume Checkpoint_C_direct_sub40_256/model_epoch_4.pth
```

`Scripts/chain_to_phase2.sh` 自动衔接（轮询直到阶段一退出，确认 `model_epoch_4.pth`
存在，等 GPU 释放后在 `nohup` 下启动阶段二，可扛 SSH 断连）。另一个一次性脚本
`Scripts/run_train_C_direct_full.sh` 跳过子集阶段（模式 `smoke` / `mem N` / `full N`）。

> 注意：`run_train_C_direct.sh full_resume` 跑 **3** 个 epoch，而
> `run_train_C_direct_full.sh full` 跑 **5** 个，两者都写到
> `Checkpoint_C_direct_full_256/`。它们是通往全量训练的两条不同路线，不可互换——请
> 明确选择其一。

**务必先做单卡 smoke 测试**（`--dist False` 才能看到真实 traceback）：

```bash
bash Scripts/run_train_C_direct.sh smoke
```

`run_train_C_direct.sh` 实际发出的命令（请替换为你自己的 python / 项目根 / 路径）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u Source/main.py \
  --mode train --dataset tmdataset --modelName TM_Model_C --is_stage_2 True \
  --dataloaderType iterable --dataloaderNum 8 --sampleNum 1 \
  --sar_size 192 --matching_dim 256 --shuffle_buffer 100000 --lr 1e-3 \
  --auto_save_num 1 --dist True --gpu 4 --batchSize 10 --port 8867 \
  --trainListPath ./Datasets/shard_train_256_full.csv \
  --imgNum <清单 file_num 之和，约 13.18M> --maxEpochs 3 \
  --log ./log_c_full/ --outputDir ./result_c_full/ \
  --resultImgDir ./ResultImg_c_full/ --modelDir ./Checkpoint_C_direct_full_256/
```

> `--dataloaderType` 是 **JackFramework 核心 flag，默认 `mapstyle`**——要用 WebDataset
> 分片管线，必须显式传 `--dataloaderType iterable`。

### 其它变体 / 消融（256/192 脚本）

- `Scripts/run_a_ft_vs_b.sh` —— 干净的“Stage-1 初始化是否有帮助”消融：A-ft（热启动、
  SAR 可训）vs B（随机 SAR），唯一差别是 SAR 初始化。
- `Scripts/start_train_stage1_256_A.sh` / `start_train_stage1_256_F.sh` —— Stage-1
  对比式预对齐（A）与 learned-inner-product/VICReg 预对齐（F）。
- `Scripts/run_frozen_arms_F.sh` / `run_frozen_arms_F_seed.sh` —— 冻结编码器的
  双臂实验（及其多种子复现）。

> `Scripts/start_train_stage2_{A..E}_wds.sh` 以及 `Scripts/legacy_128_64/` 里的一切
> 都是**遗留的 128/64 几何**脚本，已被上面的 256/192 脚本取代。遗留的 Stage-2 脚本仍是
> 仓库内对原始 A–E 消融语义最清晰的说明，但不应再用于当前几何下的训练。

### Checkpoint

只保存 `model.trainable_tail.state_dict()`（SAR 编码器 + 匹配头 + fusion；冻结的 DINO
骨干不重复保存），每 `--auto_save_num` 个 epoch 存一次到 `--modelDir`。JackFramework
通过该目录下的纯文本 `checkpoint.list`（两行：`last model name:<file>` 再 `<file>`）选择
加载哪个 checkpoint。续训只需把 `--modelDir` 指向其 `checkpoint.list` 指明目标文件的目录；
所有加载均为 `strict=False`。从既有 checkpoint **全新热启动**时用
`Source/Tools/strip_stage1_opt.py` 丢掉优化器状态并重置 epoch 计数。

---

## 评测

评测以 `--mode test` 跑在 **map-style** 后端上（评测脚本故意**不**传
`--dataloaderType iterable`），针对 map-style CSV（`SAR_Image,RGB_Image,Label`）。每个评测
脚本把单个 checkpoint 固定进一个干净目录、写一个两行 `checkpoint.list`，并以 batch 1、
`TM_SEED=2024` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` 确定性运行。

```bash
# 域内 2k（提出的级联，hard argmax）：
bash Scripts/run_eval_C_2k.sh <src_ckpt.pth> <tag> [gpu]

# 级联 + 亚像素（E），在 C 的 checkpoint 上评测：
bash Scripts/run_eval_E_2k.sh <C_ckpt.pth> <tag> [gpu]

# 标准头变体 A / B / D：
bash Scripts/run_eval_std_2k.sh <TM_Model_A|B|D> <src_ckpt.pth> <tag> [gpu] [port]

# 更大的域内评测集（10k / 20k）与 frozen-arms 全流程：
bash Scripts/run_eval_B_256_10k.sh
bash Scripts/run_eval_A_256_10k.sh
bash Scripts/run_eval_frozen_full.sh [FEPOCH] [EVAL_CSV] [EVAL_N] [TAG]

# 跨数据集零样本（WHU 跨传感器，QXS 跨分辨率）：
bash Scripts/run_eval_whu_matched.sh [CSV] [N] [TAG]
bash Scripts/run_whu_res_sweep.sh      # 可控 GSD 扫描（PCK vs GSD）
```

`--mode test` 写出按 rank 的临时 CSV，由 rank-0 合并为
`test_<dataset>_<timestamp>.csv`（列
`index,sar_name,rgb_name,gt_x,gt_y,pred_x,pred_y,score`）。用以下脚本转成指标：

```bash
python Source/Tools/pck_summary.py <merged_test.csv>     # PCK/CMR@{1,2,3,5,10}px + 分组
python Source/Tools/eval_test_csv.py --csv <merged_test.csv>   # L2 统计 + PCK@1..5（支持 --n/--seed）
```

报告指标是**逐模板 PCK / CMR**（在某像素阈值下的正确匹配率——每个模板一个 `(x, y)`）。
成对模型比较用 `Source/Tools/paired_bootstrap_pck.py`，给出点估计、95% 置信区间与
`P(A 优于 B)`。

---

## 分析与探针工具

`Source/Tools/` 存放分析与指标脚本；`Source/Tools/probes/` 存放产出论文分析部分数字的
七个机理探针（默认样本 `Checkpoint_B_256/model_epoch_4.pth`）：

- `probe_b_similarity.py` —— B 学到的跨模态相似性**存在于何处**（原始 stride-16 余弦 vs
  投影后的 stride-4 匹配子空间；含符号检查）。
- `probe_b_controls.py` —— 定位是**真实**的跨模态相似性还是内容无关的捷径？
  （对照实验组，含外来模板替换）。
- `probe_b_content.py` —— 为什么原始 DINO/SAR 特征正交却可投影？
  （CCA 共享子空间 + 边缘结构相关性）。
- `probe_b_keystone.py` —— 匹配是否真的**用到**了共享子空间，且该子空间是真实的还是
  空间自相关泄漏（去泄漏 CCA）？
- `probe_b_stage1sim.py` —— **learned inner product** 是否优于原始余弦作为 Stage-1 指标？
- `probe_b_collapse.py` —— learned-IP Stage-1 是否塌缩，可学习的 `W_d` 是否安全？
  （有效秩容量探针）。
- `probe_f0_mechanism.py` —— Stage-2 微调是否会**抹掉** Stage-1 的对齐？
  （“F0”机理：S1 vs A-ft vs B 编码器对齐 + 跨编码器 CKA）。

辅助工具：`oracle_geom_ceiling.py`（理论匹配上界）、
`stage1_separation_2k.py` / `analyze_stage1_separation.py`（Stage-1 可分性、d′）、
`coarse_eval_ab.py`（Stage-1 是否对粗匹配层有帮助）、`ckpt_inspect.py`（dim128 vs dim256
/ 是否含 JBU）、`compare_dataset_stats.py`（域间差异统计）、
`parse_erank_logs.py` / `plot_frozen_arms.py`（论文配图数据准备）。

---

## 引用

如果使用本代码，请引用论文（TGRS 2026）：

```bibtex
@article{TODO_dosrnet_tgrs2026,
  title   = {TODO: paper title},
  author  = {TODO: authors},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {TODO: volume/number/pages/DOI}
}
```

---

## 致谢

- **DINOv3**（Meta AI / FAIR）—— 冻结光学骨干，以 git 子模块形式置于
  `Source/Libs/dinov3/`，按其自身许可证再分发。
  <https://github.com/facebookresearch/dinov3>
- **FeatUp** —— JBU 图像引导上采样器（`featup.upsamplers.JBULearnedRange`）。
  <https://github.com/facebookresearch/FeatUp>
- **JackFramework** —— 训练框架（训练循环、DDP、日志、checkpoint）。
  <https://github.com/Archaic-Atom/JackFramework>

---

## 许可证

MIT，见 [`LICENSE`](LICENSE)。DINOv3 按其自身许可证再分发，详见上游仓库。

> **TODO：**`LICENSE` 文件当前写的是 `Copyright (c) 2018 BeautyOfWeb`，看起来是继承自
> 模板的残留，而非本项目作者——版权行可能需要修正。
