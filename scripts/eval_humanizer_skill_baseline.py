#!/usr/bin/env python3
"""Humanizer-skill 출력을 기존 도메인 평가축으로 채점합니다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline.evaluate import (  # noqa: E402
    load_test_rows,
    roberta_scores,
    scrn_scores,
    semantic_cosines,
    strip_dialogue_markers,
)
from pipeline.metrics_redundancy import redundancy  # noqa: E402


CONDITION = "humanizer_skill_v1.6.0"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 오류: {path}:{number}: {exc}") from exc


def select_from_manifest(rows: list[dict], manifest: Path, domain: str,
                         expected: int) -> list[dict]:
    ids = [
        str(row["source_id"]) for row in read_jsonl(manifest)
        if row.get("domain") == domain
    ]
    if len(ids) != expected or len(set(ids)) != expected:
        raise RuntimeError(
            f"{domain}: manifest는 고유 ID {expected}개여야 하지만 {len(ids)}개입니다."
        )
    mapping = {str(row["id"]): row for row in rows}
    missing = [doc_id for doc_id in ids if doc_id not in mapping]
    if missing:
        raise RuntimeError(f"{domain}: pair에 manifest ID가 없습니다: {missing[:5]}")
    return [mapping[doc_id] for doc_id in ids]


def load_generations(paths: list[Path]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    errors: dict[str, object] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("cond") != CONDITION:
                continue
            doc_id = str(row.get("doc_id"))
            if row.get("error") or not row.get("text"):
                errors[doc_id] = row.get("error", "빈 출력")
                continue
            outputs[doc_id] = row["text"]
            errors.pop(doc_id, None)
    if errors:
        raise RuntimeError(f"복구되지 않은 생성 오류 {len(errors)}건: {list(errors.items())[:3]}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--generated", type=Path, nargs="+", required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--scrn", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dialogue", action="store_true")
    args = parser.parse_args()

    rows = select_from_manifest(
        load_test_rows(args.pairs), args.manifest, args.domain, args.n
    )
    generations = load_generations(args.generated)
    expected_ids = [f"{args.domain}:{row['id']}" for row in rows]
    missing = [doc_id for doc_id in expected_ids if doc_id not in generations]
    if missing:
        raise RuntimeError(f"{args.domain}: 출력 누락 {len(missing)}건: {missing[:5]}")

    sets = {
        "human": [row["human_text"] for row in rows],
        "x_ai": [row["ai_text"] for row in rows],
        "Humanizer-skill": [generations[doc_id] for doc_id in expected_ids],
    }
    labels = list(sets)
    joined = [text for label in labels for text in sets[label]]
    raw_scores = {
        "D": roberta_scores(args.detector, joined, args.device, args.batch_size),
        "SCRN": scrn_scores(args.scrn, joined, args.device, args.batch_size),
    }
    size = len(rows)
    scores = {
        detector: {
            label: values[index * size:(index + 1) * size]
            for index, label in enumerate(labels)
        }
        for detector, values in raw_scores.items()
    }

    reference = sets["x_ai"]
    candidate = sets["Humanizer-skill"]
    if args.dialogue:
        reference = [strip_dialogue_markers(text) for text in reference]
        candidate = [strip_dialogue_markers(text) for text in candidate]
    similarities = semantic_cosines(
        reference, {"Humanizer-skill": candidate}, args.device, args.batch_size
    )["Humanizer-skill"]
    repetitions = {
        label: [redundancy(text, args.dialogue) for text in texts]
        for label, texts in sets.items()
    }

    print(f"[{args.domain}] Humanizer-skill n={size}")
    print(f"{'집단':<18}{'D ASR%':>10}{'D P(AI)':>11}{'SCRN ASR%':>12}"
          f"{'SCRN P(AI)':>13}{'의미cos':>10}{'급성붕괴':>10}")
    for label in labels:
        summary = []
        for detector in ("D", "SCRN"):
            threshold = float(np.percentile(scores[detector]["human"], 95))
            values = np.asarray(scores[detector][label])
            summary.extend((100 * float(np.mean(values <= threshold)), float(np.mean(values))))
        cosine = 1.0 if label != "Humanizer-skill" else float(np.mean(similarities))
        acute = 100 * float(np.mean([item["acute"] for item in repetitions[label]]))
        print(f"{label:<18}{summary[0]:>9.1f}%{summary[1]:>11.3f}"
              f"{summary[2]:>11.1f}%{summary[3]:>13.3f}{cosine:>10.3f}{acute:>9.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for label in labels:
            for index, text in enumerate(sets[label]):
                record = {
                    "domain": args.domain,
                    "dec": label,
                    "doc_id": str(rows[index]["id"]),
                    "text": text,
                    "p_D": scores["D"][label][index],
                    "p_SCRN": scores["SCRN"][label][index],
                    "cos": similarities[index] if label == "Humanizer-skill" else 1.0,
                    **{
                        f"rep_{name}": repetitions[label][index][name]
                        for name in ("reuse", "approx", "acute")
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
