#!/usr/bin/env python3
"""Apply sequential normalization rules from TSV files.

Actions:
    src      source file (relative to rule file)
    dst      destination file (relative to rule file)
    regex    regex substitution (value1=pattern, value2=replacement)
    append   append manual TSV (already in final format, no regex)
    nfc      normalize IPA to NFC if "true"
    tag      BCP 47 tag for this group (e.g. pt-BR)
    mode     "full" (default) or "overlay" (only changed lines)

Escape sequences in regex value1/value2:
    \\s  -> literal space  (avoids invisible trailing whitespace in TSV)
    \\t  -> literal tab
    \\n  -> literal newline

Regex backreferences (\\1, \\2, etc.) are preserved for re.sub().
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


_ESCAPE_MAP = {"\\s": " ", "\\t": "\t", "\\n": "\n"}


def _expand_escapes(value: str) -> str:
    """Expand \\s, \\t, \\n in rule values while preserving regex backrefs."""
    for seq, char in _ESCAPE_MAP.items():
        value = value.replace(seq, char)
    return value


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
            rules["_rule_file"] = str(rule_file)

            if action == "src":
                rules["src"] = (rule_file.parent / value1).resolve()
            elif action == "dst":
                rules["dst"] = (rule_file.parent / value1).resolve()
            elif action == "regex":
                rules["regex"].append((_expand_escapes(value1), _expand_escapes(value2)))
            elif action == "append":
                rules.setdefault("append", []).append((rule_file.parent / value1).resolve())
            elif action == "nfc":
                rules["nfc"] = value1.strip().lower() == "true"
            elif action == "tag":
                rules["tag"] = value1.strip()
            elif action == "mode":
                mode = value1.strip().lower()
                if mode not in ("full", "overlay"):
                    raise ValueError(f"{rule_file}:{lineno}: mode must be 'full' or 'overlay', got '{mode}'")
                rules["mode"] = mode
            else:
                raise ValueError(f"{rule_file}:{lineno}: unknown action '{action}'")

    return groups


def _resolve_order(groups: dict[str, dict], selected: list[str]) -> list[str]:
    """Topological sort: if group A's src points to group B's dst, run B first."""
    dst_map: dict[str, str] = {}
    for name in selected:
        dst = groups[name].get("dst")
        if dst:
            dst_map[str(Path(dst).resolve())] = name

    order: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        src = groups[name].get("src")
        if src:
            dep = dst_map.get(str(Path(src).resolve()))
            if dep and dep != name:
                visit(dep)
        order.append(name)

    for name in selected:
        visit(name)
    return order


def apply_group(name: str, rules: dict[str, object]) -> dict:
    """Process one group. Returns metadata dict for manifest."""
    import re
    import unicodedata

    src = rules.get("src")
    dst = rules.get("dst")
    regex_rules = rules.get("regex", [])
    append_files = rules.get("append", [])
    apply_nfc = rules.get("nfc", False)
    mode = rules.get("mode", "full")

    if not src:
        raise ValueError(f"group '{name}' missing src")
    if not dst:
        raise ValueError(f"group '{name}' missing dst")

    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"group '{name}' source not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    changed_lines = 0
    written_lines = 0
    substitutions = 0
    appended_lines = 0

    with src.open("r", encoding="utf-8", newline="") as fin, dst.open("w", encoding="utf-8", newline="") as fout:
        for raw_line in fin:
            total_lines += 1
            line = raw_line.rstrip("\r\n")
            if "\t" not in line:
                if mode == "full":
                    fout.write(raw_line)
                    written_lines += 1
                continue

            word, text = line.split("\t", 1)
            normalized = text

            for pattern, replacement in regex_rules:
                normalized, count = re.subn(pattern, replacement, normalized)
                substitutions += count

            if apply_nfc:
                normalized = unicodedata.normalize("NFC", normalized)

            was_changed = normalized != text
            changed_lines += was_changed

            if mode == "full" or was_changed:
                fout.write(f"{word}\t{normalized}\n")
                written_lines += 1

        # Append manual files (already in final format, no regex applied)
        for append_path in append_files:
            if not append_path.exists():
                print(f"  [warn] append file not found: {append_path}")
                continue
            for raw_line in Path(append_path).read_text(encoding="utf-8").splitlines():
                if not raw_line.strip() or raw_line.startswith("#"):
                    continue
                if "\t" not in raw_line:
                    continue
                fout.write(f"{raw_line}\n")
                appended_lines += 1
                written_lines += 1

    parts = [
        f"[{name}] src={src.name} dst={dst.name} lines={total_lines}",
        f"changed={changed_lines} subs={substitutions}",
        f"written={written_lines}",
    ]
    if mode == "overlay":
        parts.append("mode=overlay")
    if appended_lines:
        parts.append(f"appended={appended_lines}")
    print(" ".join(parts))

    return {
        "group": name,
        "tag": rules.get("tag", ""),
        "src": str(src),
        "dst": str(dst),
        "mode": mode,
        "nfc": apply_nfc,
        "total_lines": total_lines,
        "written_lines": written_lines,
        "changed_lines": changed_lines,
        "appended_lines": appended_lines,
        "rule_file": rules.get("_rule_file", ""),
    }


def write_manifest(entries: list[dict], manifest_path: Path, repo_root: Path) -> None:
    """Write dicts/manifest.yaml with metadata for each processed group."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Auto-generated by normalize_dicts.py — do not edit manually.",
        f"# Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "#",
        "# Each entry maps a generated dictionary file to its BCP 47 tag",
        "# and processing metadata. The dataset reader uses this file to",
        "# discover available languages, their tags, and file locations.",
        "",
        "entries:",
    ]

    for e in entries:
        dst = Path(e["dst"])
        try:
            rel = dst.relative_to(repo_root)
        except ValueError:
            rel = dst
        tag = e.get("tag") or e["group"]

        lines.append(f"  - group: {e['group']}")
        lines.append(f"    tag: \"{tag}\"")
        lines.append(f"    path: \"{str(rel).replace(chr(92), '/')}\"")
        lines.append(f"    mode: {e['mode']}")
        lines.append(f"    nfc: {str(e['nfc']).lower()}")
        lines.append(f"    lines: {e['written_lines']}")
        if e.get("rule_file"):
            try:
                rf = Path(e["rule_file"]).relative_to(repo_root)
            except ValueError:
                rf = e["rule_file"]
            lines.append(f"    rule_file: \"{str(rf).replace(chr(92), '/')}\"")

        lines.append("")

    manifest_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[manifest] {manifest_path} ({len(entries)} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply grouped normalization rules from TSV files")
    parser.add_argument(
        "--rules",
        nargs="+",
        default=["dicts-workbench/rules/*.rules.tsv"],
        help="Rule file(s) or glob(s). Default: dicts-workbench/rules/*.rules.tsv",
    )
    parser.add_argument("--group", default="", help="Optional group filter (example: pt-br)")
    parser.add_argument(
        "--manifest",
        default="dicts/manifest.yaml",
        help="Path for generated manifest. Default: dicts/manifest.yaml",
    )
    args = parser.parse_args()

    # parents[2] = project root (script is at dicts-workbench/scripts/normalize_dicts.py)
    repo_root = Path(__file__).resolve().parents[2]
    rule_files = [path for pattern in args.rules for path in sorted(repo_root.glob(pattern))]
    if not rule_files:
        raise FileNotFoundError("No rule files found. Check --rules pattern.")

    groups = load_groups(rule_files)
    selected = [args.group] if args.group else sorted(groups)

    for name in selected:
        if name not in groups:
            raise ValueError(f"group '{name}' not found in provided rule files")

    # Resolve execution order (topological sort by dependency)
    ordered = _resolve_order(groups, selected)

    # Process groups in dependency order
    manifest_entries = []
    for name in ordered:
        meta = apply_group(name, groups[name])
        manifest_entries.append(meta)

    # Write manifest
    manifest_path = repo_root / args.manifest
    write_manifest(manifest_entries, manifest_path, repo_root)


if __name__ == "__main__":
    main()
