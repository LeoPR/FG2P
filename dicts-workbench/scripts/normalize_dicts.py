#!/usr/bin/env python3
"""Apply sequential normalization rules from TSV files."""

from __future__ import annotations

import argparse
from pathlib import Path


def load_groups(rule_files: list[Path]) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}

    for rule_file in rule_files:
        for lineno, raw_line in enumerate(rule_file.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            row = raw_line.split("\t")
            if len(row) < 3:
                raise ValueError(f"{rule_file}:{lineno}: expected at least 3 columns")

            group, action, value1 = row[:3]
            value2 = row[3] if len(row) > 3 else ""

            group = group.strip()
            action = action.strip().lower()
            rules = groups.setdefault(group, {"regex": []})

            if action == "src":
                rules["src"] = (rule_file.parent / value1).resolve()
            elif action == "dst":
                rules["dst"] = (rule_file.parent / value1).resolve()
            elif action == "regex":
                rules["regex"].append((value1, value2))
            else:
                raise ValueError(f"{rule_file}:{lineno}: unknown action '{action}'")

    return groups


def apply_group(name: str, rules: dict[str, object]) -> None:
    src = rules.get("src")
    dst = rules.get("dst")
    regex_rules = rules.get("regex", [])

    if not src:
        raise ValueError(f"group '{name}' missing src")
    if not dst:
        raise ValueError(f"group '{name}' missing dst")

    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"group '{name}' source not found: {src}")

    import re

    dst.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    changed_lines = 0
    substitutions = 0

    with src.open("r", encoding="utf-8", newline="") as fin, dst.open("w", encoding="utf-8", newline="") as fout:
        for raw_line in fin:
            total_lines += 1
            line = raw_line.rstrip("\n")
            if "\t" not in line:
                fout.write(raw_line)
                continue

            word, text = line.split("\t", 1)
            normalized = text

            for pattern, replacement in regex_rules:
                normalized, count = re.subn(pattern, replacement, normalized)
                substitutions += count

            changed_lines += normalized != text
            fout.write(f"{word}\t{normalized}\n")

    print(
        f"[{name}] src={src} dst={dst} lines={total_lines} "
        f"changed_lines={changed_lines} substitutions={substitutions}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply grouped normalization rules from TSV files")
    parser.add_argument(
        "--rules",
        nargs="+",
        default=["dicts-workbench/rules/*.rules.tsv"],
        help="Rule file(s) or glob(s). Default: dicts-workbench/rules/*.rules.tsv",
    )
    parser.add_argument("--group", default="", help="Optional group filter (example: pt-br)")
    args = parser.parse_args()

    # parents[2] = project root (script is at dicts-workbench/scripts/normalize_dicts.py)
    repo_root = Path(__file__).resolve().parents[2]
    rule_files = [path for pattern in args.rules for path in sorted(repo_root.glob(pattern))]
    if not rule_files:
        raise FileNotFoundError("No rule files found. Check --rules pattern.")

    groups = load_groups(rule_files)
    selected = [args.group] if args.group else sorted(groups)

    for group in selected:
        if group not in groups:
            raise ValueError(f"group '{group}' not found in provided rule files")
        apply_group(group, groups[group])


if __name__ == "__main__":
    main()