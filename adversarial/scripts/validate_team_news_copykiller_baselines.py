#!/usr/bin/env python3
"""Validate the human/G0 CopyKiller export against its frozen sources."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator
import zipfile


CORE_PROPERTY_FIELDS = (
    "author",
    "title",
    "subject",
    "comments",
    "keywords",
    "last_modified_by",
    "category",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_pairs(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in iter_jsonl(path):
        doc_id = raw.get("doc_id") or raw.get("id")
        if not doc_id or doc_id in rows:
            raise RuntimeError(f"missing or duplicate doc_id in {path}: {doc_id}")
        rows[doc_id] = raw
    return rows


def normalized(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--selection-metadata", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    from docx import Document

    pairs = load_pairs(args.pairs)
    selected = json.loads(args.selection_metadata.read_text(encoding="utf-8"))[
        "selected_doc_ids"
    ]
    with (args.export / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))

    counts = Counter(row["variant"] for row in manifest)
    if counts != Counter({"human": len(selected), "g0": len(selected)}):
        raise RuntimeError(f"unexpected variant counts: {dict(counts)}")

    manifest_ids = {
        variant: [row["pair_id"] for row in manifest if row["variant"] == variant]
        for variant in ("human", "g0")
    }
    if manifest_ids["human"] != selected or manifest_ids["g0"] != selected:
        raise RuntimeError("manifest ID order differs from the frozen G1/G2 selection")
    if any(pairs[doc_id].get("split") != "test" for doc_id in selected):
        raise RuntimeError("non-test ID found in selected batch")

    expected_paths: set[Path] = set()
    text_fields = {"human": "human_text", "g0": "ai_text"}
    expected_prefixes = {"human": "H", "g0": "X"}
    for index, row in enumerate(manifest):
        variant = row["variant"]
        path = args.export / f"upload_{variant}" / row["batch"] / row["file"]
        if path in expected_paths or not path.is_file():
            raise RuntimeError(f"duplicate or missing DOCX: {path}")
        expected_paths.add(path)
        if not row["file"].startswith(expected_prefixes[variant]):
            raise RuntimeError(f"wrong filename prefix: {path}")

        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"corrupt DOCX member {bad_member}: {path}")

        document = Document(path)
        actual_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        expected_text = normalized(pairs[row["pair_id"]][text_fields[variant]])
        if actual_text != expected_text:
            raise RuntimeError(f"DOCX text mismatch: {path}")
        if int(row["chars"]) != len(expected_text):
            raise RuntimeError(f"manifest char count mismatch: {path}")

        properties = document.core_properties
        leaked = {
            field: getattr(properties, field)
            for field in CORE_PROPERTY_FIELDS
            if getattr(properties, field)
        }
        if leaked:
            raise RuntimeError(f"non-empty label-sensitive metadata {leaked}: {path}")

    actual_paths = set(args.export.glob("upload_*/*/*.docx"))
    if actual_paths != expected_paths:
        extras = sorted(str(path) for path in actual_paths - expected_paths)
        raise RuntimeError(f"unmanifested DOCX files: {extras[:5]}")

    metadata = json.loads((args.export / "export_metadata.json").read_text(encoding="utf-8"))
    if metadata["selected_doc_ids"] != selected:
        raise RuntimeError("export metadata selection differs from G1/G2 selection")
    if metadata["sources"]["pairs"]["sha256"] != sha256(args.pairs):
        raise RuntimeError("pair source SHA256 mismatch")
    if metadata["sources"]["selection_metadata"]["sha256"] != sha256(
        args.selection_metadata
    ):
        raise RuntimeError("selection metadata SHA256 mismatch")

    split_counts = Counter(raw.get("split") for raw in pairs.values())
    report = {
        "status": "passed",
        "pair_source_rows": len(pairs),
        "pair_source_splits": dict(sorted(split_counts.items())),
        "manifest_rows": len(manifest),
        "variant_counts": dict(sorted(counts.items())),
        "paired_ids": len(selected),
        "same_as_g1_g2_selection": True,
        "test_split_only": True,
        "docx_files": len(actual_paths),
        "text_exact": True,
        "zip_integrity": True,
        "label_sensitive_metadata_blank": True,
        "pair_source_sha256": sha256(args.pairs),
        "selection_metadata_sha256": sha256(args.selection_metadata),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
