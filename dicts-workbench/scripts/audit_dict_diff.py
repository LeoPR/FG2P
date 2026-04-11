#!/usr/bin/env python3
"""Audit dictionary TSV differences by category and recurring IPA patterns."""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path


def read_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in raw_line:
            continue
        word, ipa = raw_line.split("\t", 1)
        entries[word] = ipa
    return entries


def read_source_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in raw_line:
            continue
        word, ipa = raw_line.split("\t", 1)
        entries[word] = ipa.strip("/")
    return entries


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def build_report(left: dict[str, str], right: dict[str, str]) -> tuple[list[dict[str, str]], Counter[tuple[str, str]]]:
    rows: list[dict[str, str]] = []
    replacements: Counter[tuple[str, str]] = Counter()

    for word in sorted(set(left) | set(right)):
        left_ipa = left.get(word, "")
        right_ipa = right.get(word, "")

        if word not in right:
            rows.append({"word": word, "category": "only_in_left", "left_ipa": left_ipa, "right_ipa": ""})
            continue

        if word not in left:
            rows.append({"word": word, "category": "only_in_right", "left_ipa": "", "right_ipa": right_ipa})
            continue

        if left_ipa == right_ipa:
            continue

        left_nfc = nfc(left_ipa)
        right_nfc = nfc(right_ipa)

        if left_nfc == right_nfc:
            rows.append(
                {
                    "word": word,
                    "category": "unicode_equivalent",
                    "left_ipa": left_ipa,
                    "right_ipa": right_ipa,
                }
            )
            continue

        rows.append(
            {"word": word, "category": "real_content", "left_ipa": left_nfc, "right_ipa": right_nfc}
        )

        left_tokens = left_nfc.split()
        right_tokens = right_nfc.split()
        if len(left_tokens) == len(right_tokens):
            for left_token, right_token in zip(left_tokens, right_tokens):
                if left_token != right_token:
                    replacements[(left_token, right_token)] += 1

    return rows, replacements


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["word", "category", "left_ipa", "right_ipa"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], replacements: Counter[tuple[str, str]], examples: int) -> None:
    category_counts = Counter(row["category"] for row in rows)
    print("Summary")
    for category in ["only_in_left", "only_in_right", "unicode_equivalent", "real_content"]:
        print(f"- {category}: {category_counts.get(category, 0)}")

    print("\nExamples")
    for category in ["only_in_left", "only_in_right", "unicode_equivalent", "real_content"]:
        print(f"- {category}:")
        shown = 0
        for row in rows:
            if row["category"] != category:
                continue
            print(f"  {row['word']} | {row['left_ipa']} | {row['right_ipa']}")
            shown += 1
            if shown == examples:
                break
        if shown == 0:
            print("  <none>")

    print("\nTop token replacements")
    for (left_token, right_token), count in replacements.most_common(15):
        print(f"- {left_token} -> {right_token}: {count}")
    if not replacements:
        print("- <none>")


def classify_unicode_tilde_forms(entries: dict[str, str]) -> tuple[int, int]:
    text = "\n".join(entries.values())
    combining = len(re.findall(r"[aeiouAEIOU]\u0303", text))
    precomposed = sum(text.count(ch) for ch in ("ã", "ẽ", "ĩ", "õ", "ũ", "Ã", "Ẽ", "Ĩ", "Õ", "Ũ"))
    return combining, precomposed


def build_source_comparison(
    source: dict[str, str],
    output: dict[str, str],
    canonical: dict[str, str],
) -> tuple[Counter[str], Counter[tuple[str, str]]]:
    counters: Counter[str] = Counter()
    replacement_counter: Counter[tuple[str, str]] = Counter()

    common = set(source) & set(output) & set(canonical)
    for word in common:
        src = source[word]
        out = output[word]
        can = canonical[word]

        src_nfc = nfc(src)
        out_nfc = nfc(out)
        can_nfc = nfc(can)

        if src_nfc == out_nfc:
            counters["source_to_output_no_change"] += 1
        else:
            counters["source_to_output_changed"] += 1

        if out_nfc == can_nfc:
            counters["output_matches_canonical_nfc"] += 1
            continue

        counters["output_differs_from_canonical_nfc"] += 1

        out_tokens = out_nfc.split()
        can_tokens = can_nfc.split()
        if len(out_tokens) == len(can_tokens):
            for out_token, can_token in zip(out_tokens, can_tokens):
                if out_token != can_token:
                    replacement_counter[(out_token, can_token)] += 1
        else:
            counters["token_length_mismatch"] += 1

    counters["common_source_output_canonical"] = len(common)
    counters["source_only"] = len(set(source) - set(output))
    counters["canonical_only"] = len(set(canonical) - set(source))
    return counters, replacement_counter


def print_source_summary(
    source: dict[str, str],
    output: dict[str, str],
    canonical: dict[str, str],
) -> None:
    counters, replacements = build_source_comparison(source, output, canonical)
    src_combining, src_precomposed = classify_unicode_tilde_forms(source)
    out_combining, out_precomposed = classify_unicode_tilde_forms(output)
    can_combining, can_precomposed = classify_unicode_tilde_forms(canonical)

    print("\nSource -> Output -> Canonical")
    print(f"- common_source_output_canonical: {counters['common_source_output_canonical']}")
    print(f"- source_to_output_changed: {counters['source_to_output_changed']}")
    print(f"- source_to_output_no_change: {counters['source_to_output_no_change']}")
    print(f"- output_matches_canonical_nfc: {counters['output_matches_canonical_nfc']}")
    print(f"- output_differs_from_canonical_nfc: {counters['output_differs_from_canonical_nfc']}")
    print(f"- token_length_mismatch: {counters['token_length_mismatch']}")
    print(f"- source_only: {counters['source_only']}")
    print(f"- canonical_only: {counters['canonical_only']}")

    print("\nNasalized Vowel Representation")
    print(f"- source combining (a/e/i/o/u + U+0303): {src_combining}")
    print(f"- source precomposed (ã/ẽ/ĩ/õ/ũ): {src_precomposed}")
    print(f"- output combining (a/e/i/o/u + U+0303): {out_combining}")
    print(f"- output precomposed (ã/ẽ/ĩ/õ/ũ): {out_precomposed}")
    print(f"- canonical combining (a/e/i/o/u + U+0303): {can_combining}")
    print(f"- canonical precomposed (ã/ẽ/ĩ/õ/ũ): {can_precomposed}")

    print("\nTop token replacements (output -> canonical, NFC)")
    for (out_token, can_token), count in replacements.most_common(15):
        print(f"- {out_token} -> {can_token}: {count}")
    if not replacements:
        print("- <none>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TSV dictionary differences by category")
    parser.add_argument("left", help="Left TSV file")
    parser.add_argument("right", help="Right TSV file")
    parser.add_argument("--source", default="", help="Optional raw source TSV (for triadic audit)")
    parser.add_argument("--report", default="", help="Optional TSV output path")
    parser.add_argument("--examples", type=int, default=5, help="Examples per category to print")
    args = parser.parse_args()

    left_path = Path(args.left).resolve()
    right_path = Path(args.right).resolve()

    rows, replacements = build_report(read_entries(left_path), read_entries(right_path))
    print(f"Left:  {left_path}")
    print(f"Right: {right_path}\n")
    print_summary(rows, replacements, args.examples)

    if args.source:
        source_path = Path(args.source).resolve()
        print(f"\nSource: {source_path}")
        source_entries = read_source_entries(source_path)
        print_source_summary(source_entries, read_entries(left_path), read_entries(right_path))

    if args.report:
        report_path = Path(args.report).resolve()
        write_report(report_path, rows)
        print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()