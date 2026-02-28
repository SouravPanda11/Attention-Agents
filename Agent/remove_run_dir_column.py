#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def has_utf8_bom(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(3) == b"\xef\xbb\xbf"


def load_csv(path: Path) -> tuple[list[list[str]], csv.Dialect, bool]:
    bom = has_utf8_bom(path)
    encoding = "utf-8-sig" if bom else "utf-8"

    with path.open("r", encoding=encoding, newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(f, dialect=dialect))

    return rows, dialect, bom


def write_csv(path: Path, rows: list[list[str]], dialect: csv.Dialect, bom: bool) -> None:
    encoding = "utf-8-sig" if bom else "utf-8"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding=encoding, newline="") as f:
        writer = csv.writer(f, dialect=dialect)
        writer.writerows(rows)
    tmp_path.replace(path)


def remove_run_dir_from_file(path: Path, dry_run: bool) -> bool:
    rows, dialect, bom = load_csv(path)
    if not rows:
        return False

    header = rows[0]
    if "run_dir" not in header:
        return False

    idx = header.index("run_dir")
    updated_rows: list[list[str]] = []
    for row in rows:
        if idx < len(row):
            updated_rows.append(row[:idx] + row[idx + 1 :])
        else:
            updated_rows.append(row)

    if not dry_run:
        write_csv(path, updated_rows, dialect, bom)
    return True


def collect_csv_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove the 'run_dir' column from CSV files recursively."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="Agent/runs",
        help="Root directory to search (default: Agent/runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be updated without writing changes.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    csv_files = collect_csv_files(root)
    changed: list[Path] = []
    for csv_path in csv_files:
        try:
            if remove_run_dir_from_file(csv_path, dry_run=args.dry_run):
                changed.append(csv_path)
        except Exception as exc:
            print(f"[ERROR] {csv_path}: {exc}")

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {len(changed)} file(s).")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
