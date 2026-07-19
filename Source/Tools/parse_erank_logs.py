# -*- coding: utf-8 -*-
"""Extract per-iteration erank (l4/l5) series from Stage-1 F training logs.

从三个 Stage-1 训练日志里提取逐迭代 erank 序列,降采样后写成一个 CSV,
给论文图 fig:erank_traj 当数据源。Run on the remote (CPU only).

Output columns: run, epoch, it, total, l4, l5  (it = within-epoch iteration).
Always keeps iteration 1 of each epoch (the instantaneous-ish init readings).
"""
import csv
import re

LOGS = [
    ('run1_lam1', 'train_stage1_256_F_v1_rank22.log'),
    ('run3_lam25', 'train_stage1_256_F_v2_lambda25_rank7.log'),
    ('run4_final', 'train_stage1_256_F.log'),
]
PAT = re.compile(
    r'(\d+)/(\d+) .*?e: (\d+),.*?l4: ([0-9.]+), l5: ([0-9.]+)')
SUBSAMPLE = 25  # keep every Nth line / 每 N 行保留一行

rows = []
for run, path in LOGS:
    try:
        raw = open(path, 'r', errors='replace').read().replace('\r', '\n')
    except FileNotFoundError:
        print(f'missing: {path}')
        continue
    # Strip ANSI color codes — the JF progress bar wraps fields in them,
    # which silently breaks the field regex (2026-06-12 lesson: rows=0).
    # 剥离 ANSI 颜色码:JF 进度条把字段包在颜色码里,不剥正则一行都配不上。
    raw = re.sub(r'\x1b\[[0-9;]*m', '', raw)
    n = 0
    for line in raw.split('\n'):
        m = PAT.search(line)
        if not m:
            continue
        it, total, ep, l4, l5 = (int(m.group(1)), int(m.group(2)),
                                 int(m.group(3)), float(m.group(4)),
                                 float(m.group(5)))
        n += 1
        # keep epoch-start lines exactly + subsample the rest
        # 每个 epoch 的第一条精确保留,其余降采样。
        if it <= 2 or n % SUBSAMPLE == 0:
            rows.append((run, ep, it, total, l4, l5))

with open('erank_traj.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run', 'epoch', 'it', 'total', 'l4', 'l5'])
    w.writerows(rows)
print(f'wrote erank_traj.csv rows={len(rows)}')
