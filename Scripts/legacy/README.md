# `Scripts/legacy/` — the original per-experiment launchers / 原始的逐实验启动脚本

These are the scripts that actually produced the paper's runs. They are kept for
provenance and for the many one-off analyses that never became a headline result.
这些是真正跑出论文结果的脚本，保留下来是为了可追溯，也因为其中很多一次性分析没有进入正文。

**They are not the recommended entry point.** Use `Scripts/reproduce/` (evaluate the
released models) or `Scripts/train/` (train your own) instead — those are portable
and documented.
**它们不是推荐入口。** 请改用 `Scripts/reproduce/`（评测发布的模型）或 `Scripts/train/`
（自己训练）——那两个是可移植且有文档的。

## Why they are not portable / 为什么不可直接运行

Each file hard-codes our training servers at the top:

- `PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3` (most also `cd "$PROJ"`)
- `PY=/data/raozhibo/anaconda3/envs/stereoA/bin/python`
- dataset manifests under `/data4/...` or `/data5/...` (both appear — the data moved
  between machines partway through the project)

To run one, edit those lines. The new launchers avoid the problem entirely by
deriving the project root from the script's own location and reading data paths
from the environment (see `Scripts/common.sh`).
要运行其中某个脚本，请自行修改这几行。新的启动脚本通过"由脚本自身位置推导项目根 +
数据路径走环境变量"从根本上避免了这个问题（见 `Scripts/common.sh`）。

`legacy_128_64/` one level up holds the even older 128/64-geometry launchers,
superseded when the project moved to the 256/192 ME-SSM protocol.
上一级的 `legacy_128_64/` 是更早的 128/64 几何脚本，在项目切到 256/192 的 ME-SSM 协议后已废弃。
