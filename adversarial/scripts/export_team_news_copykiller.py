#!/usr/bin/env python3
"""Export paired G1/G2 test outputs as isolated CopyKiller DOCX batches.

The same document IDs are sampled for both generators. G1 and G2 are placed in
separate upload trees so CopyKiller's plagiarism comparison cannot compare the
two versions of one article inside a single inspection. The manifest must never
be uploaded because it contains the variant and document-ID mapping.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil


def read_jsonl_gz(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def index_outputs(path, label):
    rows = read_jsonl_gz(path)
    indexed = {r["doc_id"]: r["text"] for r in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"{label}: duplicate doc_id")
    if any(not text.strip() for text in indexed.values()):
        raise RuntimeError(f"{label}: empty output")
    return indexed


def write_docx(path, text):
    from docx import Document

    doc = Document()
    for paragraph in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        doc.add_paragraph(paragraph)
    props = doc.core_properties
    props.author = props.title = props.subject = props.comments = ""
    props.keywords = props.last_modified_by = props.category = ""
    doc.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g1", type=Path, required=True)
    ap.add_argument("--g2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=350)
    ap.add_argument("--batch-size", type=int, default=350)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.limit < 1 or args.batch_size < 1:
        ap.error("--limit and --batch-size must be positive")

    g1 = index_outputs(args.g1, "G1")
    g2 = index_outputs(args.g2, "G2")
    if set(g1) != set(g2):
        raise RuntimeError("G1/G2 test document IDs do not match")
    ids = sorted(g1)
    if args.limit > len(ids):
        ap.error(f"--limit {args.limit} exceeds available test outputs {len(ids)}")
    selected = random.Random(args.seed).sample(ids, args.limit)

    if args.out.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {args.out}; use --force to replace")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    variants = (("g1", "A", g1), ("g2", "B", g2))
    manifest = []
    for variant, prefix, outputs in variants:
        for i, doc_id in enumerate(selected, 1):
            batch = (i - 1) // args.batch_size + 1
            batch_name = f"batch{batch:02d}"
            upload_dir = args.out / f"upload_{variant}" / batch_name
            upload_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{prefix}{i:04d}.docx"
            text = outputs[doc_id]
            write_docx(upload_dir / filename, text)
            manifest.append({
                "file": filename,
                "batch": batch_name,
                "pair_id": doc_id,
                "variant": variant,
                "model": f"team_news_{variant}",
                "label": "ai",
                "chars": len(text),
            })

    with (args.out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    metadata = {
        "seed": args.seed,
        "n_available": len(ids),
        "n_selected_per_variant": len(selected),
        "batch_size": args.batch_size,
        "selected_doc_ids": selected,
        "sources": {
            "g1": {"path": str(args.g1), "sha256": sha256(args.g1)},
            "g2": {"path": str(args.g2), "sha256": sha256(args.g2)},
        },
        "upload_rule": "Upload one batch directory at a time; never upload manifest.csv.",
    }
    (args.out / "export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"G1/G2 {len(selected)} paired IDs -> {args.out}")
    print(f"DOCX {len(manifest)} files; manifest.csv must not be uploaded")


if __name__ == "__main__":
    main()
