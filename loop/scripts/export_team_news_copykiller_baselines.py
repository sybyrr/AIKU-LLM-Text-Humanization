#!/usr/bin/env python3
"""Export human and original-AI (G0/x_ai) texts for the existing CK sample.

The selected document IDs are read from the already frozen G1/G2
``export_metadata.json``. No resampling is performed. Human and G0 documents
are written to separate upload trees so variants are never mixed in one
CopyKiller inspection. The manifest and metadata files must not be uploaded.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def load_pairs(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(iter_jsonl(path), 1):
        doc_id = raw.get("doc_id") or raw.get("id")
        if not doc_id:
            raise ValueError(f"{path}:{line_number}: missing doc_id|id")
        if doc_id in rows:
            raise ValueError(f"{path}:{line_number}: duplicate doc_id={doc_id}")
        if not raw.get("human_text") or not raw.get("ai_text"):
            raise ValueError(f"{path}:{line_number}: missing human_text or ai_text")
        rows[doc_id] = {
            **raw,
            "doc_id": doc_id,
            "human_text": raw["human_text"],
            "ai_text": raw["ai_text"],
        }
    if not rows:
        raise ValueError(f"empty pair dataset: {path}")
    return rows


def write_docx(path: Path, text: str) -> None:
    from docx import Document

    document = Document()
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for paragraph in normalized.split("\n"):
        document.add_paragraph(paragraph)
    properties = document.core_properties
    properties.author = properties.title = properties.subject = properties.comments = ""
    properties.keywords = properties.last_modified_by = properties.category = ""
    document.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--selection-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=350)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    pairs = load_pairs(args.pairs)
    selection = json.loads(args.selection_metadata.read_text(encoding="utf-8"))
    selected_ids = selection.get("selected_doc_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise ValueError("selection metadata has no selected_doc_ids")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selection metadata contains duplicate document IDs")

    missing = [doc_id for doc_id in selected_ids if doc_id not in pairs]
    if missing:
        raise ValueError(f"pair dataset is missing {len(missing)} selected IDs: {missing[:5]}")
    non_test = [doc_id for doc_id in selected_ids if pairs[doc_id].get("split") != "test"]
    if non_test:
        raise ValueError(f"selected IDs outside test split: {non_test[:5]}")

    if args.out.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {args.out}; use --force to replace")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    variants = (
        ("human", "H", "human_text", "human_reference", "human"),
        ("g0", "X", "ai_text", "team_news_original_ai", "ai"),
    )
    manifest: list[dict[str, Any]] = []
    for variant, prefix, text_field, model, label in variants:
        for index, doc_id in enumerate(selected_ids, 1):
            batch_number = (index - 1) // args.batch_size + 1
            batch_name = f"batch{batch_number:02d}"
            upload_dir = args.out / f"upload_{variant}" / batch_name
            upload_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{prefix}{index:04d}.docx"
            text = pairs[doc_id][text_field]
            write_docx(upload_dir / filename, text)
            manifest.append(
                {
                    "file": filename,
                    "batch": batch_name,
                    "pair_id": doc_id,
                    "variant": variant,
                    "model": model,
                    "label": label,
                    "chars": len(text),
                }
            )

    with (args.out / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    metadata = {
        "schema_version": 1,
        "n_pairs_available": len(pairs),
        "n_selected_per_variant": len(selected_ids),
        "batch_size": args.batch_size,
        "selected_doc_ids": selected_ids,
        "sources": {
            "pairs": {"path": str(args.pairs), "sha256": sha256(args.pairs)},
            "selection_metadata": {
                "path": str(args.selection_metadata),
                "sha256": sha256(args.selection_metadata),
            },
        },
        "variants": {
            "human": {"field": "human_text", "prefix": "H", "label": "human"},
            "g0": {"field": "ai_text", "prefix": "X", "label": "ai"},
        },
        "upload_rule": (
            "Upload exactly one upload_* batch directory per inspection; "
            "never upload manifest.csv or export_metadata.json."
        ),
    }
    (args.out / "export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Human/G0 {len(selected_ids)} frozen IDs -> {args.out}")
    print(f"DOCX {len(manifest)} files; manifest.csv must not be uploaded")


if __name__ == "__main__":
    main()
