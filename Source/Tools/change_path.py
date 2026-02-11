#!/usr/bin/env python3
"""
Utility script to rewrite dataset CSV entries to a new root path.
Usage:
    python Datasets/change_path.py old.csv new.csv [--old-root ... --new-root ...]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Default roots reflect the paths shared in the request. Override with CLI flags when needed.
DEFAULT_OLD_ROOT = "/data1/xiongwenxuan/Datasets/SEN_Datasets"
DEFAULT_NEW_ROOT = "/data1/datasets/Template_matching/SEN_Datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite CSV file paths from the old SEN_Datasets root to the new "
            "Template_matching root while validating that every referenced file exists."
        )
    )
    parser.add_argument("source_csv", help="Path to the original CSV file.")
    parser.add_argument("output_csv", help="Where to save the updated CSV rows.")
    parser.add_argument(
        "--old-root",
        default=DEFAULT_OLD_ROOT,
        help=f"Old dataset root to replace (default: {DEFAULT_OLD_ROOT}).",
    )
    parser.add_argument(
        "--new-root",
        default=DEFAULT_NEW_ROOT,
        help=f"New dataset root to use (default: {DEFAULT_NEW_ROOT}).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Write the output even if some remapped files are missing. "
            "Missing entries are still reported."
        ),
    )
    return parser.parse_args()


def normalize_root(root: str) -> str:
    root = root.strip()
    if not root:
        raise ValueError("Dataset root cannot be empty.")
    return root.rstrip("/")


def should_treat_as_path(value: str) -> bool:
    value = (value or "").strip()
    return bool(value) and value.startswith("/")


def rewrite_path(value: str, old_root: str, new_root: str) -> str:
    raw = value.strip()
    if raw.startswith(old_root):
        return new_root + raw[len(old_root) :]
    return raw


def convert_rows(
    rows: List[Dict[str, str]], old_root: str, new_root: str
) -> Tuple[List[Dict[str, str]], List[str], int]:
    converted: List[Dict[str, str]] = []
    missing: List[str] = []
    replacements = 0

    for row in rows:
        new_row: Dict[str, str] = {}
        for key, value in row.items():
            if value is None:
                new_row[key] = ""
                continue

            trimmed = value.strip()
            if should_treat_as_path(trimmed):
                updated = rewrite_path(trimmed, old_root, new_root)
                if updated != trimmed:
                    replacements += 1
                if not Path(updated).exists():
                    missing.append(updated)
                new_row[key] = updated
            else:
                new_row[key] = trimmed
        converted.append(new_row)

    return converted, missing, replacements


def main() -> int:
    args = parse_args()
    source_path = Path(args.source_csv).expanduser().resolve()
    output_path = Path(args.output_csv).expanduser().resolve()

    if not source_path.is_file():
        print(f"[ERROR] Cannot find source CSV: {source_path}", file=sys.stderr)
        return 1

    if source_path == output_path:
        print("[ERROR] Source and destination paths must differ.", file=sys.stderr)
        return 1

    try:
        old_root = normalize_root(args.old_root)
        new_root = normalize_root(args.new_root)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    with source_path.open("r", encoding="utf-8", newline="") as csv_in:
        reader = csv.DictReader(csv_in)
        if not reader.fieldnames:
            print("[ERROR] CSV file is missing a header row.", file=sys.stderr)
            return 1
        fieldnames = reader.fieldnames
        rows = [row for row in reader]

    converted_rows, missing_paths, replacements = convert_rows(rows, old_root, new_root)

    if missing_paths and not args.allow_missing:
        print(
            "[ERROR] The following files are missing after remapping "
            f"({len(missing_paths)} total)."
        )
        for path in missing_paths[:20]:
            print(f"  - {path}")
        if len(missing_paths) > 20:
            print("  ...")
        print("Nothing was written. Fix the missing files or pass --allow-missing.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(converted_rows)

    print(
        f"[OK] Processed {len(converted_rows)} rows. "
        f"Updated {replacements} path entries. "
        f"Missing paths: {len(missing_paths)}."
    )
    if missing_paths and args.allow_missing:
        print("Missing files were allowed; review the log above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    # python change_path.py /data1/datasets/Template_matching/SEN_Datasets/image_test.csv  ./image_test.csv