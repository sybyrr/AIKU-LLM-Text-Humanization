#!/usr/bin/env python3
"""Validate a paired team-news CopyKiller DOCX export end to end."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
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


def read_outputs(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    indexed = {row["doc_id"]: row["text"] for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"duplicate doc_id in {path}")
    return indexed


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--export",
        type=Path,
        help="Export directory to validate (default: <root>/export)",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    from docx import Document

    export = args.export if args.export else args.root / "export"
    sources = {
        "g1": args.root / "source" / "g1_gen_test.jsonl.gz",
        "g2": args.root / "source" / "g2_gen_test.jsonl.gz",
    }
    outputs = {variant: read_outputs(path) for variant, path in sources.items()}
    with (export / "manifest.csv").open(encoding="utf-8", newline="") as f:
        manifest = list(csv.DictReader(f))

    counts = Counter(row["variant"] for row in manifest)
    if set(counts) != set(sources) or counts["g1"] != counts["g2"]:
        raise RuntimeError(f"unpaired variant counts: {dict(counts)}")
    pair_ids = {
        variant: {row["pair_id"] for row in manifest if row["variant"] == variant}
        for variant in sources
    }
    if pair_ids["g1"] != pair_ids["g2"] or len(pair_ids["g1"]) != counts["g1"]:
        raise RuntimeError("G1/G2 manifest IDs are not a one-to-one paired set")

    actual_files = set(export.glob("upload_*/*/*.docx"))
    if len(actual_files) != len(manifest):
        raise RuntimeError(
            f"DOCX count {len(actual_files)} != manifest count {len(manifest)}"
        )

    expected_files = set()
    for row in manifest:
        variant = row["variant"]
        path = export / f"upload_{variant}" / row["batch"] / row["file"]
        if path in expected_files:
            raise RuntimeError(f"duplicate manifest path: {path}")
        expected_files.add(path)
        if not path.is_file():
            raise RuntimeError(f"missing DOCX: {path}")
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"corrupt DOCX member {bad_member}: {path}")

        document = Document(path)
        actual_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        expected_text = normalized(outputs[variant][row["pair_id"]])
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

    if actual_files != expected_files:
        extras = sorted(str(path) for path in actual_files - expected_files)
        raise RuntimeError(f"unmanifested DOCX files: {extras[:5]}")

    metadata = json.loads((export / "export_metadata.json").read_text(encoding="utf-8"))
    source_hashes = {variant: sha256(path) for variant, path in sources.items()}
    for variant, digest in source_hashes.items():
        if metadata["sources"][variant]["sha256"] != digest:
            raise RuntimeError(f"source SHA256 mismatch: {variant}")

    report = {
        "status": "passed",
        "manifest_rows": len(manifest),
        "variant_counts": dict(sorted(counts.items())),
        "paired_ids": len(pair_ids["g1"]),
        "docx_files": len(actual_files),
        "text_exact": True,
        "zip_integrity": True,
        "label_sensitive_metadata_blank": True,
        "source_sha256": source_hashes,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
