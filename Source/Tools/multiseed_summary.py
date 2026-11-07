#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate N seed replicates of the 2x2 into mean / SD / 95% CI.

把 2x2 析因实验的 N 个 seed 重复汇总为均值 / 标准差 / 95% 置信区间，
也就是审稿人 2 意见 5(b) 明确要求的东西。

Reviewer 2, comment 5(b): "each experimental arm appears to involve only a
single training run ... execute multiple independent runs per condition with
different random seeds, reporting the mean, standard deviation, and confidence
intervals." This tool reads the per-arm prediction CSVs written by ``tm_eval``
(``DebugResult_<tag>/test_*.csv``), groups them by (arm, seed) and reports:

* per arm: CMR(1), CMR(3), MEE — mean, SD and a t-based 95% CI, with n stated;
* per contrast: S1/Fr − R/Fr, S1/Tr − R/Tr and the interaction, each computed
  PER SEED and then aggregated — not by differencing the pooled arm means
  (the two disagree whenever the seeds are unbalanced, and the per-seed form
  is the one the reviewer asked for);
* a power line: given the observed SD, the smallest effect this many seeds can
  resolve at 95% confidence — the honest way to state a null result;
* optionally the within-seed paired bootstrap, which measures a DIFFERENT
  quantity (test-set sampling), printed side by side with the across-seed CI
  and never merged with it.

审稿人 2 的 5(b) 要求「每个条件跑多次独立训练，报告均值/标准差/置信区间」。
本工具读 ``tm_eval`` 产出的每臂预测 CSV，按 (臂, seed) 分组后输出：每臂三项
指标的均值 / SD / t 分布 95% CI；三个关键对比量（先按 seed 求差、再跨 seed
汇总，而不是拿汇总后的均值相减）；给定当前 SD 时这些 seed 能分辨的最小效应；
以及可选的配对 bootstrap —— 它量的是测试集抽样噪声，与跨 seed 区间不是同一
个量，两者绝不能混为一谈。

The CSV schema is the one every TM test CSV uses:
``index,sar_name,rgb_name,gt_x,gt_y,pred_x,pred_y,score``.
本工具只依赖 TM 所有 test CSV 共用的这套列名。

Usage / 用法:
    # scan a project root for DebugResult_* dirs of every arm and seed
    python Source/Tools/multiseed_summary.py --root .

    # or name the files; each is auto-parsed, or pinned as arm:seed:path
    python Source/Tools/multiseed_summary.py \
        DebugResult_x2_s1tr_s2024_20k/test_tmdataset_0.csv \
        s1fr:2024:/tmp/other.csv

    # paste-ready fragments + the (different-quantity) paired bootstrap
    python Source/Tools/multiseed_summary.py --root . \
        --latex tab_two_by_two.tex --csv multiseed.csv --bootstrap 10000
"""

import argparse
import csv
import math
import os
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# --- the 2x2 arms, in table order / 2x2 四臂，按表格顺序 --------------------
ARM_KEYS: Tuple[str, ...] = ('s1tr', 'rtr', 's1fr', 'rfr')
ARM_LABEL: Dict[str, str] = {
    's1tr': 'S1/Tr', 'rtr': 'R/Tr', 's1fr': 'S1/Fr', 'rfr': 'R/Fr',
}

# Contrast name -> per-arm coefficients. The interaction is the frozen-row
# effect minus the trainable-row effect, i.e. "does freezing change whether
# pre-alignment helps".
# 对比名 -> 各臂系数。交互项 = 冻结行效应 − 可训行效应，即「冻结是否改变了
# 预对齐的作用」。
CONTRASTS: Tuple[Tuple[str, Dict[str, float]], ...] = (
    ('S1/Fr - R/Fr', {'s1fr': 1.0, 'rfr': -1.0}),
    ('S1/Tr - R/Tr', {'s1tr': 1.0, 'rtr': -1.0}),
    ('(S1/Fr - R/Fr) - (S1/Tr - R/Tr)',
     {'s1fr': 1.0, 'rfr': -1.0, 's1tr': -1.0, 'rtr': 1.0}),
)

# Metrics: name, unit, and whether larger is better.
# 指标：名称、单位、是否越大越好。
METRICS: Tuple[Tuple[str, str, bool], ...] = (
    ('CMR(1)', 'pt', True),
    ('CMR(3)', 'pt', True),
    ('MEE', 'px', False),
)
CMR_KS: Dict[str, int] = {'CMR(1)': 1, 'CMR(3)': 3}

# Two-sided 95% Student-t quantiles by df, and the one-sided 80% quantiles
# used for the power line. Hard-coded so the tool needs no scipy.
# 双侧 95% 与单侧 80% 的 t 分位数表（按自由度）。写死是为了不依赖 scipy。
T95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
T80: Dict[int, float] = {
    1: 1.376, 2: 1.061, 3: 0.978, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896,
    8: 0.889, 9: 0.883, 10: 0.879, 11: 0.876, 12: 0.873, 13: 0.870,
    14: 0.868, 15: 0.866, 16: 0.865, 17: 0.863, 18: 0.862, 19: 0.861,
    20: 0.860, 21: 0.859, 22: 0.858, 23: 0.858, 24: 0.857, 25: 0.856,
    26: 0.856, 27: 0.855, 28: 0.855, 29: 0.854, 30: 0.854,
}

# Bootstrap resamples are drawn in chunks so a 20k-template CSV never
# materialises a B x n matrix. 分块抽样，避免 B x n 的大矩阵。
BOOT_CHUNK: int = 200

# Path tokens. Arm first (``s1tr`` must win over a bare ``tr``), then seed.
# 路径 token：先认臂名（``s1tr`` 优先于裸的 ``tr``），再认 seed。
ARM_RE = re.compile(
    r'(?<![a-z0-9])(s1[_-]?tr|s1[_-]?fr|r[_-]?tr|r[_-]?fr)(?![a-z0-9])')
SEED_RES = (re.compile(r'seed[_-]?(\d{2,6})(?!\d)'),
            re.compile(r'(?<![a-z0-9])s(\d{2,6})(?!\d)'))


class ArmRun(NamedTuple):
    """One evaluated arm of one seed replicate.

    一个 seed 重复中的一个臂的评测结果。
    """

    arm: str
    seed: int
    path: str
    index: np.ndarray
    error: np.ndarray


class Stats(NamedTuple):
    """Across-seed summary of one metric of one arm or contrast.

    某个臂 / 对比量在某项指标上的跨 seed 汇总。
    """

    n: int
    mean: float
    sd: Optional[float]
    lo: Optional[float]
    hi: Optional[float]
    half: Optional[float]
    seeds: Tuple[int, ...]
    values: Tuple[float, ...]


def t_quantile(table: Dict[int, float], df: int, tail: float) -> float:
    """Look up a t quantile, falling back to the normal one past df=30.

    查 t 分位数表，自由度超过 30 时退化为正态分位数。

    Args:
        table: T95 or T80. / T95 或 T80 表。
        df: degrees of freedom. / 自由度。
        tail: the normal fallback for this table. / 该表对应的正态分位数。

    Returns:
        The quantile. / 分位数。
    """
    if df <= 0:
        return float('nan')
    return table.get(df, tail)


def load_errors(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load (template index, L2 pixel error) from a TM test CSV.

    从 TM test CSV 读出（模板 index, L2 误差(px)），按 index 升序返回。

    Args:
        path: a ``test_*.csv`` / ``.tmp_test_*.csv`` with columns
            ``index,sar_name,rgb_name,gt_x,gt_y,pred_x,pred_y,score``.
            列名为上述固定 schema 的预测 CSV 路径。

    Returns:
        Tuple of (index array, error array), both sorted by index.
        （index 数组, 误差数组），均按 index 升序。

    Raises:
        SystemExit: the file is empty, unreadable or missing a column.
            文件为空、读不了、或缺列时直接退出并说明原因。
    """
    idx: List[int] = []
    err: List[float] = []
    try:
        with open(path, newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                idx.append(int(row['index']))
                err.append(math.hypot(
                    float(row['pred_x']) - float(row['gt_x']),
                    float(row['pred_y']) - float(row['gt_y'])))
    except KeyError as exc:
        raise SystemExit(
            f'ERROR: {path} is missing column {exc}; expected '
            'index,sar_name,rgb_name,gt_x,gt_y,pred_x,pred_y,score') from exc
    except OSError as exc:
        raise SystemExit(f'ERROR: cannot read {path}: {exc}') from exc
    if not idx:
        raise SystemExit(f'ERROR: {path} holds no prediction rows.')

    index = np.asarray(idx, dtype=np.int64)
    error = np.asarray(err, dtype=np.float64)
    order = np.argsort(index, kind='stable')
    index, error = index[order], error[order]
    if np.unique(index).size != index.size:
        # A duplicated index means two ranks wrote the same template; the
        # merged CSV should never contain one.
        # index 重复说明两个 rank 写了同一模板；合并后的 CSV 不该出现。
        raise SystemExit(
            f'ERROR: {path} has duplicate template indices '
            f'({index.size} rows, {np.unique(index).size} unique) — this is '
            'a per-rank temp CSV concatenated twice, not a merged result.')
    return index, error


def parse_identity(path: str) -> Tuple[Optional[str], Optional[int]]:
    """Recover (arm, seed) from a path such as ``x2_s1fr_s2024_20k``.

    从形如 ``DebugResult_x2_s1fr_s2024_20k/test_*.csv`` 的路径里解析
    （臂, seed）。目录名优先于文件名，二者都找不到就返回 None。

    Args:
        path: any path under the eval output tree. / 评测输出树下的任意路径。

    Returns:
        Tuple of (arm key or None, seed or None).
        （臂名或 None, seed 或 None）。
    """
    text = path.lower()
    arms = {m.group(1).replace('_', '').replace('-', '')
            for m in ARM_RE.finditer(text)}
    arm = arms.pop() if len(arms) == 1 else None

    seed: Optional[int] = None
    # Search the deepest directory first, then the whole path: the tag lives
    # in the directory name, and seeds like 2024 are unambiguous there.
    # 先看最深一级目录，再看整条路径：tag 在目录名里，seed 在那儿没有歧义。
    haystacks = [os.path.basename(os.path.dirname(text)), text]
    for hay in haystacks:
        for pattern in SEED_RES:
            found = pattern.findall(hay)
            if found:
                seed = int(found[-1])
                break
        if seed is not None:
            break
    return arm, seed


def pick_prediction_csv(directory: str) -> Optional[str]:
    """Pick the prediction CSV of one eval output directory.

    从一个评测输出目录里挑出预测 CSV。优先合并后的 ``test_*.csv``；
    ``--mode test`` 的收尾合并偶尔会卡住，此时回退到 ``.tmp_test_*``
    （数据不丢，与 ``tm_pck`` 的处理一致）。

    Args:
        directory: a ``DebugResult_*`` directory. / 一个评测输出目录。

    Returns:
        Path of the chosen CSV, or None if the directory holds none.
        选中的 CSV 路径；目录里没有则返回 None。
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    merged = [n for n in names
              if n.startswith('test_') and n.endswith('.csv')]
    temps = [n for n in names
             if n.startswith('.tmp_test_') and n.endswith('.csv')]
    pool = merged or temps
    if not pool:
        return None
    paths = [os.path.join(directory, n) for n in pool]
    return max(paths, key=os.path.getmtime)


def discover_root(root: str) -> List[str]:
    """Collect one prediction CSV per ``DebugResult_*`` directory under root.

    在 root 下每个 ``DebugResult_*`` 目录里各取一份预测 CSV。

    Args:
        root: project root, or any directory holding the eval outputs.
            项目根目录，或任何存放评测输出的目录。

    Returns:
        List of CSV paths. / CSV 路径列表。
    """
    out: List[str] = []
    try:
        entries = sorted(os.listdir(root))
    except OSError as exc:
        raise SystemExit(f'ERROR: cannot list {root}: {exc}') from exc
    for name in entries:
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if not name.startswith('DebugResult'):
            continue
        chosen = pick_prediction_csv(full)
        if chosen is not None:
            out.append(chosen)
    return out


def build_runs(specs: Sequence[str]) -> List[ArmRun]:
    """Turn path specs into ArmRun records, refusing ambiguous ones.

    把路径（或 ``arm:seed:path``）解析成 ArmRun；解析不出臂/seed，或同一
    （臂, seed）出现两份不同的 CSV，都直接报错退出 —— 静默取其一会让整个
    汇总变成不可复现的东西。

    Args:
        specs: paths, optionally prefixed ``arm:seed:``.
            路径列表，可加 ``arm:seed:`` 前缀显式指定。

    Returns:
        One ArmRun per (arm, seed). / 每个（臂, seed）一条记录。

    Raises:
        SystemExit: on an unparseable path or a duplicate (arm, seed).
            路径解析失败或（臂, seed）重复时退出。
    """
    seen: Dict[Tuple[str, int], ArmRun] = {}
    problems: List[str] = []
    for spec in specs:
        arm: Optional[str] = None
        seed: Optional[int] = None
        path = spec
        parts = spec.split(':')
        if len(parts) >= 3 and parts[0].lower() in ARM_KEYS:
            arm = parts[0].lower()
            try:
                seed = int(parts[1])
            except ValueError:
                problems.append(f'{spec}: seed "{parts[1]}" is not an int')
                continue
            path = ':'.join(parts[2:])
        if arm is None or seed is None:
            guess_arm, guess_seed = parse_identity(os.path.abspath(path))
            arm = arm or guess_arm
            seed = seed if seed is not None else guess_seed
        if arm is None or seed is None:
            problems.append(
                f'{path}: could not read '
                f'{"arm" if arm is None else ""}'
                f'{" and " if arm is None and seed is None else ""}'
                f'{"seed" if seed is None else ""} from the path')
            continue
        key = (arm, seed)
        if key in seen and os.path.abspath(seen[key].path) != \
                os.path.abspath(path):
            problems.append(
                f'{ARM_LABEL[arm]} seed {seed} matches two CSVs:\n'
                f'      {seen[key].path}\n      {path}')
            continue
        if key in seen:
            continue
        index, error = load_errors(path)
        seen[key] = ArmRun(arm, seed, path, index, error)

    if problems:
        print('ERROR: cannot build an unambiguous (arm, seed) table.',
              file=sys.stderr)
        for line in problems:
            print(f'  - {line}', file=sys.stderr)
        print('  Pass the files explicitly as arm:seed:path, e.g.\n'
              '    s1fr:2024:DebugResult_x2_s1fr_s2024_20k/test_x.csv',
              file=sys.stderr)
        raise SystemExit(2)
    return sorted(seen.values(), key=lambda r: (ARM_KEYS.index(r.arm),
                                                r.seed))


def check_evaluation_basis(runs: Sequence[ArmRun], tolerance: int,
                           allow_mismatch: bool) -> None:
    """Refuse to aggregate arms evaluated on different template sets.

    拒绝把评测基准不同的臂混在一起汇总 —— 例如 2,000 模板的 CSV 混进
    20,000 模板的一批。这是唯一一种会「悄无声息地」污染整个 campaign 的
    错误：数字照样能算出来，只是没有任何意义。

    Two checks, in order: the row count (catches a 2k CSV among 20k ones),
    then the template index set itself (catches two different 20k subsamples,
    which the row count alone cannot see).
    先查行数（抓 2k 混进 20k），再查 index 集合本身（抓行数相同但抽样不同
    的两份 20k，光看行数是看不出来的）。

    ``tolerance`` slackens the FIRST check only. The index-set check always
    runs, because a row-count slack says nothing about whether two CSVs
    describe the same templates; with a nonzero tolerance it reports rather
    than refuses, so the caller still sees it.
    ``tolerance`` 只放宽第一项检查。index 集合检查始终执行 —— 行数容差与
    「是不是同一批模板」无关；容差非零时它只报告不中止，但绝不静默跳过。

    Args:
        runs: every loaded arm. / 已加载的全部臂。
        tolerance: row-count slack, 0 = exact. / 行数容差，0 表示精确相等。
        allow_mismatch: downgrade the refusal to a warning.
            把「拒绝」降级为警告。

    Raises:
        SystemExit: when the basis differs and the override is not set.
            基准不一致且未加 override 时退出。
    """
    if len(runs) < 2:
        return

    counts = [r.index.size for r in runs]
    modal = max(set(counts), key=counts.count)
    odd = [r for r in runs if abs(r.index.size - modal) > tolerance]
    if odd:
        print(f'ERROR: mixed evaluation basis — {counts.count(modal)} of '
              f'{len(runs)} CSVs have {modal:,} templates, but these do '
              f'not:', file=sys.stderr)
        for run in odd:
            print(f'  - {ARM_LABEL[run.arm]:>5} seed {run.seed}: '
                  f'{run.index.size:,} templates  {run.path}',
                  file=sys.stderr)
        print(f'  A CSV from a different manifest mixed into a {modal:,}-'
              'template campaign\n  yields numbers that look fine and mean '
              'nothing. Re-evaluate the odd arm\n  on the same manifest, or '
              'pass --allow-basis-mismatch if this is\n  deliberate.',
              file=sys.stderr)
        if not allow_mismatch:
            raise SystemExit(2)
        print('  --allow-basis-mismatch given: continuing anyway.',
              file=sys.stderr)
        return

    # The index-set check runs at EVERY tolerance. It is the only thing that
    # can see two disjoint subsamples of the same size, and the contrast
    # arithmetic below indexes the arms positionally, so a silent mismatch
    # would compare different templates and still print clean numbers.
    # index 集合检查在任何容差下都执行：它是唯一能发现「同样大小、却是两批
    # 不同模板」的检查；后面的对比量按位置取用各臂，静默不一致会拿不同模板
    # 相减，还照样打印出干净的数字。
    reference = runs[0]
    differing = [r for r in runs[1:]
                 if r.index.size != reference.index.size
                 or not np.array_equal(r.index, reference.index)]
    if differing:
        fatal = tolerance == 0 and not allow_mismatch
        head = 'ERROR' if fatal else 'WARN'
        print(f'{head}: same template COUNT but a different template SET '
              '— these CSVs are\n  two different subsamples, not one '
              'evaluation basis:', file=sys.stderr)
        print(f'  reference: {ARM_LABEL[reference.arm]:>5} seed '
              f'{reference.seed}  {reference.path}', file=sys.stderr)
        for run in differing:
            shared = np.intersect1d(run.index, reference.index).size
            print(f'  - {ARM_LABEL[run.arm]:>5} seed {run.seed}: '
                  f'{shared:,}/{run.index.size:,} indices shared  '
                  f'{run.path}', file=sys.stderr)
        if fatal:
            raise SystemExit(2)
        why = ('--allow-basis-mismatch given'
               if allow_mismatch else f'--basis-tol {tolerance} given')
        print(f'  {why}: continuing anyway — the per-seed contrasts below '
              'difference\n  arms template-by-template, so they are NOT '
              'trustworthy in this state.', file=sys.stderr)


def metric_value(error: np.ndarray, metric: str) -> float:
    """Compute one metric from a template-error vector.

    从每模板误差向量算出一项指标。CMR(k) 为 <= k px 的百分比，MEE 为
    平均 L2 误差(px)。

    Args:
        error: per-template L2 error in px. / 每模板 L2 误差(px)。
        metric: one of the METRICS names. / METRICS 里的指标名。

    Returns:
        The metric value. / 指标值。
    """
    if metric == 'MEE':
        return float(np.mean(error))
    return 100.0 * float(np.mean(error <= CMR_KS[metric]))


def summarize(values: Sequence[float],
              seeds: Sequence[int]) -> Stats:
    """Mean, SD and a t-based 95% CI across seeds.

    跨 seed 的均值、样本标准差与 t 分布 95% 置信区间。n=1 时 SD 与 CI
    不可估，返回 None，而不是伪造一个 0。

    Args:
        values: one value per seed. / 每个 seed 一个值。
        seeds: the matching seed ids. / 对应的 seed 编号。

    Returns:
        A Stats record. / 一条 Stats 记录。
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    mean = float(np.mean(arr))
    if n < 2:
        return Stats(n, mean, None, None, None, None,
                     tuple(seeds), tuple(float(v) for v in arr))
    sd = float(np.std(arr, ddof=1))
    half = t_quantile(T95, n - 1, 1.960) * sd / math.sqrt(n)
    return Stats(n, mean, sd, mean - half, mean + half, half,
                 tuple(seeds), tuple(float(v) for v in arr))


def power_line(stats: Stats, unit: str) -> str:
    """One sentence stating the smallest effect this many seeds can resolve.

    用一句话说明：以当前观测到的 SD，这么多 seed 能分辨的最小效应是多少。
    这是把一个 null 结果说清楚的唯一诚实方式 —— 「没测到差异」必须配上
    「能测到多小的差异」。

    Args:
        stats: the across-seed summary of a contrast. / 对比量的跨 seed 汇总。
        unit: "pt" or "px". / 单位。

    Returns:
        The formatted power line. / 格式化后的一行文字。
    """
    if stats.sd is None or stats.n < 2:
        return ('      power   : not estimable from a single seed — an SD '
                'needs n>=2.')
    if stats.sd == 0.0:
        return ('      power   : observed SD is exactly 0 over '
                f'{stats.n} seeds; treat the interval as unresolved, not '
                'as certainty.')
    df = stats.n - 1
    t95 = t_quantile(T95, df, 1.960)
    t80 = t_quantile(T80, df, 0.842)
    mde_ci = t95 * stats.sd / math.sqrt(stats.n)
    mde_80 = (t95 + t80) * stats.sd / math.sqrt(stats.n)
    projection = []
    for n_seeds in (3, 5, 8):
        half = (t_quantile(T95, n_seeds - 1, 1.960) * stats.sd
                / math.sqrt(n_seeds))
        marker = '*' if n_seeds == stats.n else ''
        projection.append(f'n={n_seeds}{marker}:+-{half:.2f}')
    return ('      power   : at SD={sd:.3f} {u}, n={n} resolves only '
            '|d| > {ci:.2f} {u} (95% CI\n'
            '                excludes zero); {p80:.2f} {u} is needed for '
            '80% power. Projected\n'
            '                half-width by seed count: {proj}'
            ).format(sd=stats.sd, u=unit, n=stats.n, ci=mde_ci,
                     p80=mde_80, proj='  '.join(projection))


def paired_bootstrap(runs: Dict[str, ArmRun], coeffs: Dict[str, float],
                     metric: str, n_boot: int,
                     rng: np.random.Generator) -> Optional[Tuple[float,
                                                                float, float]]:
    """Within-seed paired bootstrap of one contrast: (delta, lo, hi).

    单个 seed 内、对同一批模板做配对 bootstrap，给出该对比量的点估计与
    95% 百分位区间。

    NOTE — this is NOT the across-seed interval and must never be reported in
    its place. It resamples TEMPLATES i.i.d., so it quantifies test-set
    sampling for one fixed pair of trained networks. It also UNDERSTATES even
    that: the 20k eval set is 7,740 held-out pairs x a 289-offset grid, so
    templates are correlated within a pair and within a SEN1-2 scene. Use a
    scene-level cluster bootstrap for the test-set axis.
    注意：这不是跨 seed 区间，绝不能拿来替代它。它按 i.i.d. 重采样「模板」，
    量的是固定一对已训练网络在测试集抽样下的波动；而且连这一点都低估了 ——
    20k 评测集是 7,740 个留出配对 x 289 个偏移格点，同一配对、同一 SEN1-2
    场景内的模板互相相关。测试集这条轴应当用场景级 cluster bootstrap。

    Args:
        runs: arm key -> ArmRun, all from the same seed.
            臂名 -> ArmRun，必须同属一个 seed。
        coeffs: arm key -> +-1 coefficient. / 臂名 -> ±1 系数。
        metric: one of the METRICS names. / 指标名。
        n_boot: number of resamples. / 重采样次数。
        rng: the generator. / 随机数发生器。

    Returns:
        (delta, ci_lo, ci_hi), or None if the arms share no templates.
        （点估计, CI 下界, CI 上界）；无公共模板时返回 None。
    """
    keys = list(coeffs)
    common = runs[keys[0]].index
    for key in keys[1:]:
        common = np.intersect1d(common, runs[key].index)
    if common.size == 0:
        return None
    aligned: Dict[str, np.ndarray] = {}
    for key in keys:
        run = runs[key]
        pos = np.searchsorted(run.index, common)
        aligned[key] = run.error[pos]

    point = sum(coeffs[k] * metric_value(aligned[k], metric) for k in keys)
    n_rows = common.size
    draws = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        chunk = min(BOOT_CHUNK, n_boot - done)
        take = rng.integers(0, n_rows, size=(chunk, n_rows))
        acc = np.zeros(chunk, dtype=np.float64)
        for key in keys:
            sampled = aligned[key][take]
            if metric == 'MEE':
                stat = sampled.mean(axis=1)
            else:
                stat = 100.0 * (sampled <= CMR_KS[metric]).mean(axis=1)
            acc += coeffs[key] * stat
        draws[done:done + chunk] = acc
        done += chunk
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def format_stats(stats: Stats, unit: str, signed: bool = False) -> str:
    """Render "mean +- SD  95% CI [lo, hi]  (n=N)" for the console.

    把一条 Stats 渲染成控制台一行。n=1 时明说 SD/CI 不可估。

    Args:
        stats: the summary. / 汇总记录。
        unit: "pt" or "px". / 单位。
        signed: force a leading +/-, used for contrasts. / 对比量强制带符号。

    Returns:
        The formatted string. / 格式化字符串。
    """
    fmt = '{:+.2f}' if signed else '{:.2f}'
    head = fmt.format(stats.mean)
    if stats.sd is None:
        return (f'{head} {unit}   SD/CI not estimable (n=1) '
                f'[seed {stats.seeds[0]}]')
    bounds = f'[{fmt.format(stats.lo)}, {fmt.format(stats.hi)}]'
    return (f'{head} +- {stats.sd:.2f} {unit}   95% CI '
            f'{bounds}   (n={stats.n})')


def collect(runs: Sequence[ArmRun]) -> Tuple[
        Dict[str, Dict[int, ArmRun]], List[int]]:
    """Index the runs by arm and seed.

    把 ArmRun 按臂与 seed 建索引，并返回排序后的 seed 列表。

    Args:
        runs: loaded arms. / 已加载的臂。

    Returns:
        (arm -> seed -> ArmRun, sorted seeds). / （索引, 排序 seed 列表）。
    """
    table: Dict[str, Dict[int, ArmRun]] = {a: {} for a in ARM_KEYS}
    for run in runs:
        table[run.arm][run.seed] = run
    seeds = sorted({r.seed for r in runs})
    return table, seeds


def arm_stats(table: Dict[str, Dict[int, ArmRun]],
              metric: str) -> Dict[str, Stats]:
    """Across-seed Stats of one metric for every present arm.

    每个存在的臂在某项指标上的跨 seed 汇总。

    Args:
        table: arm -> seed -> ArmRun. / 臂 -> seed -> 记录。
        metric: metric name. / 指标名。

    Returns:
        arm key -> Stats. / 臂名 -> Stats。
    """
    out: Dict[str, Stats] = {}
    for arm in ARM_KEYS:
        seeds = sorted(table[arm])
        if not seeds:
            continue
        values = [metric_value(table[arm][s].error, metric) for s in seeds]
        out[arm] = summarize(values, seeds)
    return out


def contrast_stats(table: Dict[str, Dict[int, ArmRun]],
                   coeffs: Dict[str, float],
                   metric: str) -> Optional[Stats]:
    """Per-seed contrast, then aggregated across seeds.

    先在每个 seed 内部求对比量，再跨 seed 汇总 —— 不是拿汇总后的均值相减。
    两者在 seed 不平衡时结果不同，而审稿人要的是前者：对比量本身的
    run-to-run 分布。

    Args:
        table: arm -> seed -> ArmRun. / 臂 -> seed -> 记录。
        coeffs: arm key -> +-1 coefficient. / 臂名 -> ±1 系数。
        metric: metric name. / 指标名。

    Returns:
        Stats over the seeds where EVERY needed arm is present, or None.
        在所有相关臂都齐备的 seed 上汇总；一个都没有则返回 None。
    """
    usable = sorted(set.intersection(
        *[set(table[arm]) for arm in coeffs]))
    if not usable:
        return None
    values = []
    for seed in usable:
        values.append(sum(
            coeffs[arm] * metric_value(table[arm][seed].error, metric)
            for arm in coeffs))
    return summarize(values, usable)


def print_report(table: Dict[str, Dict[int, ArmRun]], seeds: List[int],
                 runs: Sequence[ArmRun], n_boot: int, seed_rng: int) -> None:
    """Print the whole console report.

    打印完整的控制台报告。

    Args:
        table: arm -> seed -> ArmRun. / 臂 -> seed -> 记录。
        seeds: sorted seed list. / 排序后的 seed 列表。
        runs: loaded arms, for the inventory block. / 用于清单块。
        n_boot: paired-bootstrap resamples, 0 = skip. / 配对 bootstrap 次数。
        seed_rng: rng seed for the bootstrap. / bootstrap 随机种子。
    """
    basis = runs[0].index.size
    print('=' * 78)
    print('MULTI-SEED 2x2 SUMMARY  (Reviewer 2, comment 5b)')
    print('=' * 78)
    print(f'evaluation basis : {basis:,} templates per arm')
    print(f'seeds            : {", ".join(str(s) for s in seeds)}  '
          f'(n={len(seeds)})')
    print('arms present     :')
    for arm in ARM_KEYS:
        got = sorted(table[arm])
        state = ', '.join(str(s) for s in got) if got else 'MISSING'
        print(f'  {ARM_LABEL[arm]:>5}  seeds: {state}')
    missing = [ARM_LABEL[a] for a in ARM_KEYS if not table[a]]
    if missing:
        print(f'  NOTE: {", ".join(missing)} absent — the contrasts that '
              'need them are skipped.')

    # --- what one seed can and cannot support --------------------------
    if len(seeds) < 2:
        print()
        print('-' * 78)
        print('SINGLE SEED — WHAT THIS CAN AND CANNOT SUPPORT')
        print('-' * 78)
        print('  CAN   : point estimates per arm and per contrast, and the '
              'test-set\n          (paired-bootstrap) interval for this one '
              'pair of trained nets.')
        print('  CANNOT: any statement about run-to-run variance. SD and the '
              'across-seed\n          CI are undefined at n=1, so Reviewer '
              '2 comment 5(b) is NOT yet\n          answered. Re-run '
              'Scripts/train/train_2x2.sh with further seeds.')

    # --- per arm -------------------------------------------------------
    print()
    print('-' * 78)
    print('PER ARM — across-seed mean, SD, t-based 95% CI')
    print('-' * 78)
    for metric, unit, higher_better in METRICS:
        direction = 'higher is better' if higher_better else 'lower is better'
        print(f'  {metric} [{unit}, {direction}]')
        stats_by_arm = arm_stats(table, metric)
        for arm in ARM_KEYS:
            if arm not in stats_by_arm:
                continue
            print(f'    {ARM_LABEL[arm]:>5} : '
                  f'{format_stats(stats_by_arm[arm], unit)}')
        print()

    # --- per contrast --------------------------------------------------
    print('-' * 78)
    print('PER CONTRAST — computed PER SEED, then aggregated')
    print('-' * 78)
    print('  (differencing the pooled arm means is NOT the same thing '
          'whenever the\n   seeds are unbalanced; the per-seed form is what '
          'the reviewer asked for)')
    rng = np.random.default_rng(seed_rng)
    for name, coeffs in CONTRASTS:
        print()
        print(f'  {name}')
        needed = [ARM_LABEL[a] for a in coeffs if not table[a]]
        if needed:
            print(f'    SKIPPED — no results for {", ".join(needed)}.')
            continue
        for metric, unit, _ in METRICS:
            stats = contrast_stats(table, coeffs, metric)
            if stats is None:
                print(f'    {metric:>7} : SKIPPED — no seed has every arm.')
                continue
            print(f'    {metric:>7} : {format_stats(stats, unit, True)}')
            per_seed = '  '.join(
                f'{s}:{v:+.2f}' for s, v in zip(stats.seeds, stats.values))
            print(f'              per seed  {per_seed}')
            print(power_line(stats, unit))
        if n_boot:
            print('    paired bootstrap within each seed '
                  f'(B={n_boot}, i.i.d. over templates):')
            for seed in seeds:
                if any(seed not in table[a] for a in coeffs):
                    continue
                per_arm = {a: table[a][seed] for a in coeffs}
                cells = []
                for metric, unit, _ in METRICS:
                    got = paired_bootstrap(per_arm, coeffs, metric,
                                           n_boot, rng)
                    if got is None:
                        cells.append(f'{metric}=n/a')
                        continue
                    point, lo, hi = got
                    cells.append(f'{metric} {point:+.2f} '
                                 f'[{lo:+.2f},{hi:+.2f}] {unit}')
                print(f'      seed {seed}: ' + '   '.join(cells))

    # --- the statement the response letter must not blur ---------------
    print()
    print('-' * 78)
    print('THE TWO INTERVALS ARE DIFFERENT QUANTITIES — DO NOT CONFLATE')
    print('-' * 78)
    print('  ACROSS-SEED CI (above, "95% CI" on each row)')
    print('    covers TRAINING-RUN variance: random initialization of the '
          'SAR trunk,\n    data order, and nondeterministic kernels. It is '
          'the quantity Reviewer 2\n    comment 5(b) asks for, and it is the '
          'one to quote in Table III and in\n    any claim about whether an '
          'arm beats another.')
    if n_boot:
        print('  PAIRED BOOTSTRAP CI (above, per seed)')
        print('    covers TEST-SET sampling for ONE fixed pair of trained '
              'networks. It says\n    nothing about what a re-run would do. '
              'It is also OPTIMISTIC here: the\n    20k eval set is 7,740 '
              'held-out pairs x a 289-offset grid, so templates are\n'
              '    correlated within a pair and within a SEN1-2 scene; an '
              'i.i.d. bootstrap\n    therefore UNDERSTATES the interval — '
              'this is how the submitted paper\n    reached P(delta>0)=1.00. '
              'Use a scene-level cluster bootstrap for this axis.')
    else:
        print('  PAIRED BOOTSTRAP CI')
        print('    not computed (pass --bootstrap B). It would cover '
              'TEST-SET sampling for\n    one fixed pair of trained '
              'networks — a different quantity, and one that\n    an i.i.d. '
              'template bootstrap understates because the 20k eval set is\n'
              '    7,740 pairs x a 289-offset grid.')
    print('  Neither interval is a substitute for the other, and they must '
          'not be\n  pooled, averaged, or reported as one number.')


def latex_fragment(table: Dict[str, Dict[int, ArmRun]],
                   seeds: List[int]) -> str:
    """Build the paste-ready LaTeX table fragment.

    生成可直接粘进论文的 LaTeX 表格片段（Table III 的多 seed 版本）。

    Args:
        table: arm -> seed -> ArmRun. / 臂 -> seed -> 记录。
        seeds: sorted seed list. / 排序后的 seed 列表。

    Returns:
        The LaTeX source. / LaTeX 源码字符串。
    """
    def cell(stats: Optional[Stats], signed: bool) -> str:
        if stats is None:
            return '--'
        fmt = '{:+.2f}' if signed else '{:.2f}'
        head = fmt.format(stats.mean)
        if stats.sd is None:
            return f'${head}$'
        body = f'${head} \\pm {stats.sd:.2f}$'
        if signed:
            body += (f' \\, [{stats.lo:+.2f}, {stats.hi:+.2f}]')
        return body

    seed_text = ', '.join(str(s) for s in seeds)
    lines = [
        '% Multi-seed 2x2 (Reviewer 2, comment 5b).',
        '% Auto-generated by Source/Tools/multiseed_summary.py -- do not',
        '% hand-edit; re-run the tool instead.',
        f'% seeds: {seed_text}',
        '\\begin{tabular}{lccc}',
        '\\toprule',
        'Arm & CMR(1) (\\%) & CMR(3) (\\%) & MEE (px) \\\\',
        '\\midrule',
    ]
    stats_cache = {m: arm_stats(table, m) for m, _, _ in METRICS}
    for arm in ARM_KEYS:
        if arm not in stats_cache['CMR(1)']:
            continue
        cells = ' & '.join(cell(stats_cache[m].get(arm), False)
                           for m, _, _ in METRICS)
        n_arm = stats_cache['CMR(1)'][arm].n
        lines.append(f'{ARM_LABEL[arm]} ($n={n_arm}$) & {cells} \\\\')
    lines += [
        '\\midrule',
        '\\multicolumn{4}{l}{\\emph{Contrasts: per-seed difference, mean '
        '$\\pm$ SD, 95\\% CI}} \\\\',
    ]
    for name, coeffs in CONTRASTS:
        stats = {m: contrast_stats(table, coeffs, m) for m, _, _ in METRICS}
        if all(v is None for v in stats.values()):
            continue
        cells = ' & '.join(cell(stats[m], True) for m, _, _ in METRICS)
        label = name.replace('-', '$-$')
        lines.append(f'{label} & {cells} \\\\')
    lines += [
        '\\bottomrule',
        '\\end{tabular}',
        '% The intervals above are ACROSS-SEED (training-run variance).',
        '% The paired/cluster bootstrap covers test-set sampling and is a',
        '% different quantity; report it separately, never merged.',
    ]
    return '\n'.join(lines) + '\n'


def write_csv(path: str, table: Dict[str, Dict[int, ArmRun]]) -> None:
    """Write the machine-readable summary CSV.

    写出可被后续脚本/图表直接消费的汇总 CSV。每行含 n、均值、SD、CI，
    以及分号分隔的逐 seed 原始值，便于复核与画图。

    Args:
        path: output path. / 输出路径。
        table: arm -> seed -> ArmRun. / 臂 -> seed -> 记录。
    """
    header = ['kind', 'name', 'metric', 'unit', 'n', 'mean', 'sd',
              'ci_lo', 'ci_hi', 'ci_half_width', 'seeds', 'per_seed_values']

    def row(kind: str, name: str, metric: str, unit: str,
            stats: Stats) -> List[str]:
        def num(value: Optional[float]) -> str:
            return '' if value is None else f'{value:.6f}'
        return [kind, name, metric, unit, str(stats.n), num(stats.mean),
                num(stats.sd), num(stats.lo), num(stats.hi), num(stats.half),
                ';'.join(str(s) for s in stats.seeds),
                ';'.join(f'{v:.6f}' for v in stats.values)]

    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for metric, unit, _ in METRICS:
            for arm, stats in arm_stats(table, metric).items():
                writer.writerow(row('arm', ARM_LABEL[arm], metric, unit,
                                    stats))
        for name, coeffs in CONTRASTS:
            for metric, unit, _ in METRICS:
                stats = contrast_stats(table, coeffs, metric)
                if stats is not None:
                    writer.writerow(row('contrast', name, metric, unit,
                                        stats))


def main() -> int:
    """Parse args, load the arms, guard the basis, print and emit.

    解析参数、加载各臂、校验评测基准、打印报告并写出 LaTeX / CSV。

    Returns:
        Process exit code. / 进程退出码。
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('specs', nargs='*',
                        help='prediction CSVs, optionally arm:seed:path / '
                             '预测 CSV，可加 arm:seed: 前缀')
    parser.add_argument('--root', default=None,
                        help='scan this directory for DebugResult_*/ '
                             '/ 扫描该目录下的 DebugResult_*/')
    parser.add_argument('--latex', default=None,
                        help='also write the LaTeX fragment here / '
                             '把 LaTeX 片段另存到该路径')
    parser.add_argument('--csv', default=None,
                        help='also write the summary CSV here / '
                             '把汇总 CSV 另存到该路径')
    parser.add_argument('--bootstrap', type=int, default=0,
                        help='paired-bootstrap resamples per seed, 0=off / '
                             '每个 seed 的配对 bootstrap 次数，0 表示不做')
    parser.add_argument('--seed', type=int, default=2024,
                        help='rng seed for the bootstrap / bootstrap 随机种子')
    parser.add_argument('--basis-tol', type=int, default=0,
                        help='row-count slack between arms, 0=exact / '
                             '各臂行数容差，0 表示必须完全一致')
    parser.add_argument('--allow-basis-mismatch', action='store_true',
                        help='downgrade the mixed-basis refusal to a warning '
                             '/ 把混基准的拒绝降级为警告')
    parser.add_argument('--no-latex-stdout', action='store_true',
                        help='do not echo the LaTeX fragment / '
                             '不在终端回显 LaTeX 片段')
    args = parser.parse_args()

    specs: List[str] = list(args.specs)
    if args.root:
        specs.extend(discover_root(args.root))
    if not specs:
        parser.error('nothing to aggregate: give CSV paths or --root DIR')

    runs = build_runs(specs)
    if not runs:
        print('ERROR: no usable prediction CSVs.', file=sys.stderr)
        return 2
    check_evaluation_basis(runs, args.basis_tol, args.allow_basis_mismatch)

    table, seeds = collect(runs)
    print_report(table, seeds, runs, args.bootstrap, args.seed)

    fragment = latex_fragment(table, seeds)
    if not args.no_latex_stdout:
        print()
        print('-' * 78)
        print('LATEX FRAGMENT (paste into the paper)')
        print('-' * 78)
        print(fragment, end='')
    if args.latex:
        with open(args.latex, 'w') as handle:
            handle.write(fragment)
        print(f'\nwrote LaTeX fragment: {args.latex}')
    if args.csv:
        write_csv(args.csv, table)
        print(f'wrote summary CSV    : {args.csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
