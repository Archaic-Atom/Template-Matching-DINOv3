#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired CMR bootstrap at three cluster levels: template / pair / scene.

三层配对 bootstrap:同一批模板上比较两臂(或多臂两两)的 CMR(k),分别按
模板 (i.i.d.)、留出影像对、SEN1-2 场景重采样。20k 评测集由 7,740 对留出
影像 x 289 偏移网格采样而来,同一对(乃至同一场景)的模板高度相关;按模板
i.i.d. 重采样会低估区间——投稿版 P(dCMR>0)=1.00 就是这么来的。

Cluster keys are derived from the prediction CSV's own ``rgb_name`` column:
聚类键全部来自预测 CSV 自带的 ``rgb_name`` 列:

    pair   ROIs1158_spring_s2_120_p1125      one optical reference = one pair
    scene  ROIs1158_spring_s2_120            the SEN1-2 scene it was cut from

A row whose ``rgb_name`` does not parse is a hard error, never a catch-all
bucket: silently pooling unparsable names would restore exactly the i.i.d.
behaviour this script exists to escape.
``rgb_name`` 解析失败直接报错,绝不塞进「其它」桶——静默兜底等于退回 i.i.d.,
正是本脚本要避免的。

CSV cols (JackFramework eval output / JF 评测输出列):
    index,sar_name,rgb_name,gt_x,gt_y,pred_x,pred_y,score

Usage / 用法:
    python Source/Tools/cluster_bootstrap.py \
        S1/Tr:DebugResult_x2_s1tr_s2024_20k/test_tmdataset_xxx.csv \
        R/Tr:DebugResult_x2_rtr_s2024_20k/test_tmdataset_xxx.csv \
        --B 10000 --seed 2024 --out cluster_ci_s2024.csv
"""

import argparse
import csv
import math
import os
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

LEVELS: Tuple[str, ...] = ("template", "pair", "scene")
DEFAULT_KS: Tuple[int, ...] = (1, 3)
REQUIRED_COLS: Tuple[str, ...] = (
    "index", "sar_name", "rgb_name", "gt_x", "gt_y", "pred_x", "pred_y",
)
BOOT_CHUNK: int = 256
CI_PCTS: Tuple[float, float] = (2.5, 97.5)

# ROIs<id>_<season>_s<sensor>_<scene>_p<patch>, anchored at both ends so a
# renamed or rewritten CSV fails loudly instead of half-parsing.
# 两端都锚定,改名/重写过的 CSV 会直接报错而不是半解析。
SEN12_NAME_RE = re.compile(
    r"^ROIs(?P<roi>\d+)"
    r"_(?P<season>[A-Za-z]+)"
    r"_s(?P<sensor>\d+)"
    r"_(?P<scene>\d+)"
    r"_p(?P<patch>\d+)$"
)
# SAR crops carry the sampled offset as a "<dx>_<dy>_" prefix.
# SAR 裁片名前面带 "<dx>_<dy>_" 偏移前缀。
SAR_OFFSET_PREFIX_RE = re.compile(r"^-?\d+_-?\d+_(?=ROIs)")
# The SAR crop carries its sampled offset in the filename, and this project
# uses BOTH conventions: a prefix (`20_44_ROIs..._p415`) in the small debug
# exports, and a suffix (`ROIs..._p457_20_36`) in every 20k eval export.
# Handling only the prefix made the rgb/sar cross-check fire on all 20,000
# rows of every real eval — a warning that always fires is a warning nobody
# reads, and it masks the genuine mismatch it exists to catch.
# SAR 裁片把采样偏移写进文件名,本项目两种约定都在用:小调试导出是前缀
# (`20_44_ROIs..._p415`),而所有 20k 评测导出是后缀(`ROIs..._p457_20_36`)。
# 只认前缀会导致交叉校验在每一次真实评测的两万行上全部报警——每次必响的
# 警告等于没有警告,还会盖住它本来要抓的真不一致。
SAR_OFFSET_SUFFIX_RE = re.compile(r"_-?\d+_-?\d+$")


class ArmPredictions(NamedTuple):
    """One evaluated arm, keyed by the eval manifest's template index.

    一臂的评测结果,按评测清单的模板 index 索引。

    Attributes:
        label: Short arm name used in the report. 报告里的短臂名。
        path: Source CSV path. 来源 CSV 路径。
        errors: index -> L2 localization error in pixels. index -> L2 误差(px)。
        rgb: index -> optical reference filename. index -> 光学参考图文件名。
        sar: index -> SAR template filename. index -> SAR 模板文件名。
    """

    label: str
    path: str
    errors: Dict[int, float]
    rgb: Dict[int, str]
    sar: Dict[int, str]


def strip_extension(raw: str) -> str:
    """Reduce a possibly path-qualified filename to its bare stem.

    把可能带路径的文件名化简为不含扩展名的主干。

    Args:
        raw: Raw cell value, e.g. ``ROIs1158_spring_s2_120_p1125.png``.
            原始单元格值。

    Returns:
        The basename without its extension. 去掉目录与扩展名后的名字。
    """
    return os.path.splitext(os.path.basename(raw.strip()))[0]


def parse_sen12_name(raw: str, column: str, row_no: int) -> Dict[str, str]:
    """Parse a SEN1-2 patch filename into its named fields.

    把 SEN1-2 裁片文件名解析成命名字段。

    Args:
        raw: Cell value from the prediction CSV. 预测 CSV 的单元格值。
        column: Column the value came from, for the error message.
            值所在列名,用于报错信息。
        row_no: 1-based data row number, for the error message.
            以 1 起算的数据行号,用于报错信息。

    Returns:
        Dict with keys ``roi``, ``season``, ``sensor``, ``scene``, ``patch``.
        含 roi/season/sensor/scene/patch 五个键的字典。

    Raises:
        ValueError: The name does not match the SEN1-2 pattern. Cluster keys
            must never fall back to a catch-all bucket, so this is fatal.
            名字不符合 SEN1-2 模式。聚类键不允许兜底,因此直接致命报错。
    """
    name = strip_extension(raw)
    if column == "sar_name":
        name = SAR_OFFSET_PREFIX_RE.sub("", name)
        name = SAR_OFFSET_SUFFIX_RE.sub("", name)
    match = SEN12_NAME_RE.match(name)
    if match is None:
        raise ValueError(
            f"row {row_no}: cannot parse {column}={raw!r} as a SEN1-2 patch "
            f"name (expected ROIs<id>_<season>_s<n>_<scene>_p<patch>). "
            f"Refusing to bucket it — that would silently restore the i.i.d. "
            f"bootstrap. If this CSV is not SEN1-2, rerun with "
            f"--levels template,pair and say so in the letter."
        )
    return match.groupdict()


def scene_key_of(fields: Dict[str, str]) -> str:
    """Build the scene cluster key from parsed name fields.

    由解析出的字段拼出场景聚类键。

    Args:
        fields: Output of :func:`parse_sen12_name`. parse_sen12_name 的输出。

    Returns:
        ``ROIs<id>_<season>_s<sensor>_<scene>``, the patch id dropped.
        丢掉 patch 号后的场景前缀。
    """
    return (f"ROIs{fields['roi']}_{fields['season']}"
            f"_s{fields['sensor']}_{fields['scene']}")


def load_arm(label: str, path: str) -> ArmPredictions:
    """Read one prediction CSV into per-template L2 errors and filenames.

    读一份预测 CSV,取逐模板 L2 误差与文件名。

    Args:
        label: Short arm name. 短臂名。
        path: Path to ``test_*.csv`` (or ``.tmp_test_*_rank0.csv``).
            test_*.csv(或 .tmp_test_*_rank0.csv)路径。

    Returns:
        The populated :class:`ArmPredictions`. 填好的 ArmPredictions。

    Raises:
        ValueError: A required column is missing, or a template index repeats
            (which would silently corrupt the pairing across arms).
            缺列,或 index 重复(会静默破坏跨臂配对)。
    """
    errors: Dict[int, float] = {}
    rgb: Dict[int, str] = {}
    sar: Dict[int, str] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        present = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLS if c not in present]
        if missing:
            raise ValueError(
                f"{path}: missing column(s) {missing}; found "
                f"{reader.fieldnames}"
            )
        for row_no, row in enumerate(reader, start=1):
            index = int(row["index"])
            if index in errors:
                raise ValueError(
                    f"{path}: template index {index} appears twice "
                    f"(row {row_no}) — arms cannot be paired on it."
                )
            errors[index] = math.hypot(
                float(row["pred_x"]) - float(row["gt_x"]),
                float(row["pred_y"]) - float(row["gt_y"]),
            )
            rgb[index] = row["rgb_name"]
            sar[index] = row["sar_name"]
    if not errors:
        raise ValueError(f"{path}: no data rows.")
    return ArmPredictions(label, path, errors, rgb, sar)


def align_arms(arms: Sequence[ArmPredictions]) -> List[int]:
    """Intersect the arms' template indices and check they describe one set.

    取各臂 index 的交集,并核对它们描述的是同一批模板。

    Args:
        arms: Loaded arms, at least two. 已载入的各臂,至少两个。

    Returns:
        Sorted list of common template indices. 排序后的公共 index 列表。

    Raises:
        ValueError: No common templates, or the same index names different
            files in different arms (i.e. the CSVs are not the same eval set).
            无公共模板,或同一 index 在不同臂里对应不同文件(不是同一评测集)。
    """
    common = set(arms[0].errors)
    for arm in arms[1:]:
        common &= set(arm.errors)
    if not common:
        raise ValueError("arms share no template index — different eval sets?")
    indices = sorted(common)
    head = arms[0]
    for arm in arms[1:]:
        for index in indices:
            if arm.rgb[index] != head.rgb[index]:
                raise ValueError(
                    f"template {index}: {head.label} has rgb_name "
                    f"{head.rgb[index]!r} but {arm.label} has "
                    f"{arm.rgb[index]!r} — the arms are not paired."
                )
    return indices


def crosscheck_sar_names(arm: ArmPredictions, indices: Sequence[int]) -> int:
    """Count rows whose SAR name disagrees with the optical name it pairs to.

    统计 SAR 名与配对光学名不一致的行数。

    A mismatch means the cluster key derived from ``rgb_name`` no longer
    describes the row's actual image pair — worth a loud warning, because
    these CSVs have been rewritten by hand before.
    不一致意味着由 ``rgb_name`` 推出的聚类键不再描述该行真实的影像对——
    值得大声警告,因为这些 CSV 以前被手工改写过。

    Args:
        arm: The arm to check. 待检查的臂。
        indices: Template indices to check. 待检查的模板 index。

    Returns:
        Number of mismatching rows. 不一致的行数。
    """
    bad = 0
    for row_no, index in enumerate(indices, start=1):
        # This is a warning-level check, so an unparsable name counts as a
        # mismatch instead of aborting — the scene level is where a parse
        # failure is fatal, and --levels template,pair must stay usable.
        # 这是警告级检查,解析失败计作不一致而非中止——致命的是 scene 层级,
        # --levels template,pair 这个退路必须仍然可用。
        try:
            rgb_fields = parse_sen12_name(arm.rgb[index], "rgb_name", row_no)
            sar_fields = parse_sen12_name(arm.sar[index], "sar_name", row_no)
            same = all(rgb_fields[key] == sar_fields[key]
                       for key in ("roi", "season", "scene", "patch"))
        except ValueError:
            same = False
        if not same:
            if bad == 0:
                print(
                    f"WARN [{arm.label}]: rgb/sar name mismatch, first at "
                    f"index {index}: {arm.rgb[index]} vs {arm.sar[index]}",
                    file=sys.stderr,
                )
            bad += 1
    return bad


def cluster_keys_for_level(level: str, rgb_names: Sequence[str]) -> List[str]:
    """Map each aligned template to its cluster key at one level.

    把每个对齐后的模板映射到该层级的聚类键。

    Args:
        level: One of ``template`` / ``pair`` / ``scene``.
            取 template / pair / scene 之一。
        rgb_names: Per-template optical reference filenames, in aligned order.
            按对齐顺序排列的逐模板光学参考图文件名。

    Returns:
        Cluster key per template. 逐模板的聚类键。

    Raises:
        ValueError: Unknown level, or an unparsable name at scene level.
            未知层级,或 scene 层级下有名字解析失败。
    """
    if level == "template":
        return [f"#{i}" for i in range(len(rgb_names))]
    if level == "pair":
        return [strip_extension(name) for name in rgb_names]
    if level == "scene":
        return [
            scene_key_of(parse_sen12_name(name, "rgb_name", row_no))
            for row_no, name in enumerate(rgb_names, start=1)
        ]
    raise ValueError(f"unknown cluster level: {level!r}")


def encode_clusters(keys: Sequence[str]) -> Tuple[np.ndarray, int]:
    """Turn cluster keys into contiguous ids, in first-appearance order.

    把聚类键编码成连续 id,按首次出现顺序。

    First-appearance ordering is deliberate: with singleton clusters the ids
    become ``arange(n)``, so the cluster machinery consumes the RNG in exactly
    the same order as the plain i.i.d. bootstrap and the two must agree
    bit-for-bit (see :func:`iid_reference_bootstrap`).
    刻意用首次出现顺序:全是单元素簇时 id 就是 arange(n),聚类路径与朴素
    i.i.d. bootstrap 消耗随机数的顺序完全一致,两者必须逐位相等。

    Args:
        keys: Cluster key per template. 逐模板聚类键。

    Returns:
        Tuple of (int64 cluster id array, number of clusters).
        (int64 簇 id 数组, 簇数)。
    """
    lookup: Dict[str, int] = {}
    ids = np.empty(len(keys), dtype=np.int64)
    for position, key in enumerate(keys):
        cluster_id = lookup.get(key)
        if cluster_id is None:
            cluster_id = len(lookup)
            lookup[key] = cluster_id
        ids[position] = cluster_id
    return ids, len(lookup)


def cluster_bootstrap(
    hits: np.ndarray,
    cluster_ids: np.ndarray,
    n_clusters: int,
    n_boot: int,
    seed: int,
    chunk: int = BOOT_CHUNK,
) -> np.ndarray:
    """Resample whole clusters with replacement and recompute every rate.

    整簇有放回重采样,重算每个系列的比率。

    One draw of ``n_clusters`` clusters is shared by every series, which is
    what makes the comparison PAIRED: both arms are read through the same
    resampled templates, so the shared per-template difficulty cancels.
    每次重采样抽一组簇、所有系列共用,这就是「配对」的含义:两臂读的是同一
    批重采样模板,模板难度的共同方差被消掉。

    Args:
        hits: ``(n_series, n_templates)`` array of 0/1 hit indicators.
            (系列数, 模板数) 的 0/1 命中矩阵。
        cluster_ids: Per-template cluster id from :func:`encode_clusters`.
            逐模板簇 id。
        n_clusters: Number of distinct clusters. 簇数。
        n_boot: Bootstrap replicates. bootstrap 重复次数。
        seed: RNG seed. 随机种子。
        chunk: Replicates per vectorized block, to bound memory.
            每块的重复次数,用于限制内存。

    Returns:
        ``(n_series, n_boot)`` array of bootstrap rates, in percent.
        (系列数, 重复次数) 的 bootstrap 比率(百分数)。
    """
    n_series = hits.shape[0]
    sums = np.empty((n_clusters, n_series), dtype=np.float64)
    for series in range(n_series):
        sums[:, series] = np.bincount(
            cluster_ids, weights=hits[series], minlength=n_clusters
        )
    sizes = np.bincount(cluster_ids, minlength=n_clusters).astype(np.float64)

    rng = np.random.default_rng(seed)
    out = np.empty((n_series, n_boot), dtype=np.float64)
    counts = np.empty((chunk, n_clusters), dtype=np.float64)
    done = 0
    while done < n_boot:
        block = min(chunk, n_boot - done)
        for row in range(block):
            draw = rng.integers(0, n_clusters, n_clusters)
            counts[row] = np.bincount(draw, minlength=n_clusters)
        numer = counts[:block] @ sums          # (block, n_series)
        denom = counts[:block] @ sizes         # (block,)
        out[:, done:done + block] = (100.0 * numer / denom[:, None]).T
        done += block
    return out


def iid_reference_bootstrap(
    hits: np.ndarray,
    n_boot: int,
    seed: int,
    chunk: int = BOOT_CHUNK,
) -> np.ndarray:
    """Plain i.i.d. paired bootstrap, written independently as a cross-check.

    朴素 i.i.d. 配对 bootstrap,独立实现,用作交叉校验。

    This is the estimator the submitted paper used (resample templates, take
    the mean hit rate). It exists here only so that the cluster machinery can
    be checked against it under singleton clusters; it is not reported.
    这就是投稿版用的估计量(重采样模板取均值)。放在这里只为在单元素簇下校验
    聚类路径,不参与报告。

    Args:
        hits: ``(n_series, n_templates)`` 0/1 hit indicators. 0/1 命中矩阵。
        n_boot: Bootstrap replicates. 重复次数。
        seed: RNG seed. 随机种子。
        chunk: Replicates per vectorized block. 每块重复次数。

    Returns:
        ``(n_series, n_boot)`` bootstrap rates in percent. 百分数比率矩阵。
    """
    n_series, n_templates = hits.shape
    out = np.empty((n_series, n_boot), dtype=np.float64)
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        block = min(chunk, n_boot - done)
        for row in range(block):
            draw = rng.integers(0, n_templates, n_templates)
            out[:, done + row] = 100.0 * hits[:, draw].sum(axis=1) / n_templates
        done += block
    return out


def summarize_delta(delta: np.ndarray) -> Dict[str, float]:
    """Reduce a bootstrap delta distribution to the numbers the letter quotes.

    把 bootstrap 差值分布压缩成回复信要引用的几个数。

    Args:
        delta: ``(n_boot,)`` bootstrap replicates of the CMR gap, in points.
            (重复次数,) 的 CMR 差值 bootstrap 样本(百分点)。

    Returns:
        Dict with ``ci_lo``, ``ci_hi``, ``p_gt0``, ``se``. 含四个键的字典。
    """
    ci_lo, ci_hi = np.percentile(delta, CI_PCTS)
    return {
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_gt0": float(np.mean(delta > 0.0)),
        "se": float(np.std(delta, ddof=1)),
    }


def run_analysis(
    arms: Sequence[ArmPredictions],
    indices: Sequence[int],
    ks: Sequence[int],
    levels: Sequence[str],
    n_boot: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]], np.ndarray]:
    """Run the paired cluster bootstrap at every requested level.

    在每个请求的层级上跑配对聚类 bootstrap。

    Args:
        arms: Aligned arms. 已对齐的各臂。
        indices: Common template indices, sorted. 排序后的公共 index。
        ks: CMR thresholds in pixels. CMR 阈值(px)。
        levels: Cluster levels to run. 要跑的聚类层级。
        n_boot: Bootstrap replicates. 重复次数。
        seed: RNG seed; every level uses the SAME seed so that levels differ
            only through their clustering, not through the random draws.
            随机种子;各层级用同一个种子,层级间只因聚类不同而不同。

    Returns:
        Tuple of (per (pair, level, k) result records, per-level cluster
        geometry, the ``(n_series, n_templates)`` hit matrix).
        (逐 (对比, 层级, k) 的结果记录, 逐层级的簇统计, 命中矩阵)。
    """
    n_arms = len(arms)
    n_templates = len(indices)
    rgb_names = [arms[0].rgb[i] for i in indices]
    errors = np.array(
        [[arm.errors[i] for i in indices] for arm in arms], dtype=np.float64
    )
    # Series layout: series (a, j) = arm a at threshold ks[j].
    # 系列排布:series (a, j) = 第 a 臂在阈值 ks[j] 上。
    hits = np.empty((n_arms * len(ks), n_templates), dtype=np.float64)
    for a in range(n_arms):
        for j, k in enumerate(ks):
            hits[a * len(ks) + j] = (errors[a] <= k).astype(np.float64)
    point = hits.mean(axis=1) * 100.0

    geometry: Dict[str, Dict[str, float]] = {}
    boot_by_level: Dict[str, np.ndarray] = {}
    for level in levels:
        keys = cluster_keys_for_level(level, rgb_names)
        cluster_ids, n_clusters = encode_clusters(keys)
        sizes = np.bincount(cluster_ids, minlength=n_clusters)
        geometry[level] = {
            "n_clusters": float(n_clusters),
            "mean_size": float(sizes.mean()),
            "max_size": float(sizes.max()),
        }
        boot_by_level[level] = cluster_bootstrap(
            hits, cluster_ids, n_clusters, n_boot, seed
        )

    records: List[Dict[str, object]] = []
    for a in range(n_arms):
        for b in range(a + 1, n_arms):
            for j, k in enumerate(ks):
                sa, sb = a * len(ks) + j, b * len(ks) + j
                base_se = None
                for level in levels:
                    boot = boot_by_level[level]
                    stats = summarize_delta(boot[sa] - boot[sb])
                    if level == "template":
                        base_se = stats["se"]
                    deff = (
                        (stats["se"] / base_se) ** 2
                        if base_se not in (None, 0.0) else float("nan")
                    )
                    a_ci = np.percentile(boot[sa], CI_PCTS)
                    b_ci = np.percentile(boot[sb], CI_PCTS)
                    records.append({
                        "arm_a": arms[a].label,
                        "arm_b": arms[b].label,
                        "k_px": k,
                        "level": level,
                        "n_templates": n_templates,
                        "n_clusters": int(geometry[level]["n_clusters"]),
                        "mean_cluster_size": geometry[level]["mean_size"],
                        "max_cluster_size": int(geometry[level]["max_size"]),
                        "cmr_a": point[sa],
                        "cmr_b": point[sb],
                        "delta": point[sa] - point[sb],
                        "ci_lo": stats["ci_lo"],
                        "ci_hi": stats["ci_hi"],
                        "p_delta_gt0": stats["p_gt0"],
                        "se_delta": stats["se"],
                        "design_effect": deff,
                        "cmr_a_ci_lo": float(a_ci[0]),
                        "cmr_a_ci_hi": float(a_ci[1]),
                        "cmr_b_ci_lo": float(b_ci[0]),
                        "cmr_b_ci_hi": float(b_ci[1]),
                        "n_boot": n_boot,
                        "seed": seed,
                    })
    return records, geometry, hits


def run_self_test(
    hits: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[bool, float, float]:
    """Check the cluster estimator collapses to the i.i.d. one at size 1.

    校验:所有簇大小为 1 时,聚类估计量必须退化成 i.i.d. 估计量。

    Every template is made its own cluster and the cluster machinery is run
    against an independently written plain bootstrap. With first-appearance
    cluster ids the two consume the RNG identically and sum the same integers,
    so agreement should be exact; the Monte-Carlo tolerance is reported as the
    fallback yardstick.
    每个模板自成一簇,跑聚类路径并与独立实现的朴素 bootstrap 对比。首次出现
    顺序的簇 id 让两者随机数消耗一致、求和的整数也一致,应当精确相等;同时
    报告蒙特卡洛容差作为兜底判据。

    Args:
        hits: ``(n_series, n_templates)`` 0/1 hit indicators. 0/1 命中矩阵。
        n_boot: Bootstrap replicates. 重复次数。
        seed: RNG seed. 随机种子。

    Returns:
        Tuple of (passed, max absolute discrepancy in points, tolerance).
        (是否通过, 最大绝对偏差(百分点), 容差)。
    """
    n_templates = hits.shape[1]
    singleton_ids = np.arange(n_templates, dtype=np.int64)
    clustered = cluster_bootstrap(
        hits, singleton_ids, n_templates, n_boot, seed
    )
    reference = iid_reference_bootstrap(hits, n_boot, seed)
    max_abs = float(np.max(np.abs(clustered - reference)))
    # Fallback yardstick: 4 Monte-Carlo standard errors of a bootstrap mean.
    # 兜底判据:bootstrap 均值的 4 倍蒙特卡洛标准误。
    tol = 4.0 * float(np.max(np.std(reference, axis=1))) / math.sqrt(n_boot)
    return max_abs <= max(tol, 1e-9), max_abs, tol


def print_report(
    arms: Sequence[ArmPredictions],
    records: Sequence[Dict[str, object]],
    geometry: Dict[str, Dict[str, float]],
    levels: Sequence[str],
    ks: Sequence[int],
    n_templates: int,
    n_boot: int,
    seed: int,
    self_test: Optional[Tuple[bool, float, float]],
) -> None:
    """Print the paste-ready block the response letter quotes verbatim.

    打印可直接粘进回复信的汇总块。

    Args:
        arms: Loaded arms. 已载入的各臂。
        records: Output of :func:`run_analysis`. run_analysis 的输出。
        geometry: Per-level cluster geometry. 逐层级簇统计。
        levels: Levels that were run. 已跑的层级。
        ks: CMR thresholds. CMR 阈值。
        n_templates: Paired template count. 配对模板数。
        n_boot: Bootstrap replicates. 重复次数。
        seed: RNG seed. 随机种子。
        self_test: ``(passed, max_abs, tol)`` or None if skipped.
            自检结果,跳过时为 None。
    """
    bar = "=" * 78
    print("")
    print(bar)
    print("PASTE-READY SUMMARY — paired cluster bootstrap of CMR(k)")
    print(bar)
    for arm in arms:
        print(f"  arm {arm.label:<8} n_rows={len(arm.errors):<7} {arm.path}")
    print(f"  paired on {n_templates} common templates; "
          f"B={n_boot} replicates; seed={seed}; 95% percentile intervals.")
    print("  Cluster levels (all three resample the SAME templates, "
          "paired across arms):")
    for level in levels:
        geo = geometry[level]
        print(f"    {level:<9} clusters={int(geo['n_clusters']):>7}   "
              f"mean size={geo['mean_size']:>8.2f}   "
              f"max size={int(geo['max_size']):>6}")
    print("  template = i.i.d. (what the submitted paper did, kept for "
          "comparability)")
    print("  pair     = one held-out optical/SAR pair, all its grid offsets "
          "resampled together")
    print("  scene    = one SEN1-2 scene, all its pairs resampled together "
          "— QUOTE THIS ONE")

    pairs: List[Tuple[str, str]] = []
    for rec in records:
        key = (str(rec["arm_a"]), str(rec["arm_b"]))
        if key not in pairs:
            pairs.append(key)

    for arm_a, arm_b in pairs:
        print("")
        print(f"--- {arm_a} vs {arm_b}   (Delta = {arm_a} - {arm_b}) ---")
        for k in ks:
            rows = [r for r in records
                    if r["arm_a"] == arm_a and r["arm_b"] == arm_b
                    and r["k_px"] == k]
            head = rows[0]
            print(f"  CMR({k}):  {arm_a} = {head['cmr_a']:.2f} %   "
                  f"{arm_b} = {head['cmr_b']:.2f} %   "
                  f"Delta = {head['delta']:+.2f} pt")
            print(f"    {'level':<9} {'clusters':>9} {'mean size':>10} "
                  f"{'95% CI (pt)':>20} {'P(D>0)':>8} {'deff':>7}")
            for rec in rows:
                ci = f"[{rec['ci_lo']:+.2f}, {rec['ci_hi']:+.2f}]"
                deff = rec["design_effect"]
                deff_txt = "  n/a" if isinstance(deff, float) and math.isnan(
                    deff) else f"{deff:.2f}"
                print(f"    {str(rec['level']):<9} "
                      f"{rec['n_clusters']:>9} "
                      f"{rec['mean_cluster_size']:>10.2f} "
                      f"{ci:>20} "
                      f"{rec['p_delta_gt0']:>8.3f} "
                      f"{deff_txt:>7}")
            scene = [r for r in rows if r["level"] == "scene"]
            if scene and scene[0]["n_clusters"] < 20:
                print(f"    NOTE: only {scene[0]['n_clusters']} scenes — a "
                      f"percentile CI from so few clusters is itself noisy; "
                      f"say so in the letter.")

    print("")
    if self_test is None:
        print("  self-test: SKIPPED (--no-self-test).")
    else:
        passed, max_abs, tol = self_test
        verdict = "PASS" if passed else "FAIL"
        print(f"  self-test [{verdict}]: with every cluster of size 1 the "
              f"cluster estimator reproduces the plain i.i.d. bootstrap; "
              f"max |diff| = {max_abs:.3e} pt (tolerance {tol:.3e} pt).")
        if not passed:
            print("  The resampling is WRONG — do not quote these intervals.")
    print(bar)
    print("")


def write_csv(path: str, records: Sequence[Dict[str, object]]) -> None:
    """Write one row per (contrast, level, threshold) for the figure scripts.

    每个 (对比, 层级, 阈值) 写一行,供绘图脚本读取。

    Args:
        path: Output CSV path. 输出 CSV 路径。
        records: Output of :func:`run_analysis`. run_analysis 的输出。
    """
    fields = list(records[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {path}  ({len(records)} rows)")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build and parse the command line.

    构建并解析命令行。

    Args:
        argv: Argument list, or None for ``sys.argv[1:]``.
            参数列表,None 表示用 sys.argv[1:]。

    Returns:
        Parsed namespace. 解析后的命名空间。
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "specs", nargs="+", metavar="LABEL:CSV",
        help="two or more arms, e.g. S1/Tr:DebugResult_x/test_a.csv "
             "/ 两臂或更多,形如 标签:CSV",
    )
    parser.add_argument("--B", type=int, default=10000,
                        help="bootstrap replicates / bootstrap 重复次数")
    parser.add_argument("--seed", type=int, default=2024,
                        help="rng seed; output is deterministic / 随机种子")
    parser.add_argument("--ks", default="1,3",
                        help="CMR thresholds in px / CMR 阈值(px)")
    parser.add_argument("--levels", default=",".join(LEVELS),
                        help="cluster levels to run / 要跑的聚类层级")
    parser.add_argument("--out", default=None,
                        help="optional CSV output path / 可选 CSV 输出路径")
    parser.add_argument("--no-self-test", action="store_true",
                        help="skip the singleton-cluster sanity check "
                             "/ 跳过单元素簇自检")
    parser.add_argument("--no-name-crosscheck", action="store_true",
                        help="skip the rgb/sar filename agreement check "
                             "/ 跳过光学/SAR 文件名一致性检查")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Load the arms, run the three-level paired bootstrap, print the block.

    载入各臂、跑三层配对 bootstrap、打印汇总块。

    Args:
        argv: Argument list, or None for ``sys.argv[1:]``.
            参数列表,None 表示用 sys.argv[1:]。

    Returns:
        Process exit code: 0 on success, 1 if the self-test failed, 2 on a
        usage or data error. 进程退出码:成功 0,自检失败 1,用法/数据错误 2。
    """
    try:
        return _run(parse_args(argv))
    except (ValueError, OSError) as exc:
        # Loud, single-line, no traceback: every one of these is a data or
        # usage problem the caller must fix, not a bug to debug.
        # 单行大声报错、不打 traceback:这些都是调用方要修的数据/用法问题。
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    """Execute one analysis for already-parsed arguments.

    对已解析的参数执行一次分析。

    Args:
        args: Parsed command line. 解析后的命令行参数。

    Returns:
        Process exit code. 进程退出码。

    Raises:
        ValueError: Bad arguments or unusable prediction CSVs.
            参数错误或预测 CSV 不可用。
    """
    if len(args.specs) < 2:
        print("ERROR: need at least two LABEL:CSV arms to compare.",
              file=sys.stderr)
        return 2

    ks = tuple(int(tok) for tok in args.ks.split(",") if tok.strip())
    levels = tuple(tok.strip() for tok in args.levels.split(",")
                   if tok.strip())
    for level in levels:
        if level not in LEVELS:
            print(f"ERROR: unknown level {level!r}; pick from {LEVELS}.",
                  file=sys.stderr)
            return 2

    arms: List[ArmPredictions] = []
    for spec in args.specs:
        if ":" not in spec:
            print(f"ERROR: bad spec {spec!r}; expected LABEL:CSV.",
                  file=sys.stderr)
            return 2
        label, path = spec.split(":", 1)
        arms.append(load_arm(label, path))

    indices = align_arms(arms)
    for arm in arms:
        extra = len(arm.errors) - len(indices)
        if extra:
            print(f"WARN [{arm.label}]: {extra} template(s) dropped, not "
                  f"present in every arm.", file=sys.stderr)
    if not args.no_name_crosscheck:
        bad = crosscheck_sar_names(arms[0], indices)
        if bad:
            print(f"WARN: {bad}/{len(indices)} rows have disagreeing rgb/sar "
                  f"names; the pair and scene keys may not describe the real "
                  f"image pairs. Investigate before quoting the intervals.",
                  file=sys.stderr)

    records, geometry, hits = run_analysis(
        arms, indices, ks, levels, args.B, args.seed
    )
    self_test = None
    if not args.no_self_test:
        self_test = run_self_test(hits, args.B, args.seed)

    print_report(arms, records, geometry, levels, ks, len(indices),
                 args.B, args.seed, self_test)
    if args.out:
        write_csv(args.out, records)

    return 0 if (self_test is None or self_test[0]) else 1


if __name__ == "__main__":
    sys.exit(main())
