#!/usr/bin/env python3
"""여러 평가 JSONL의 텍스트를 중복 제거해 zero-shot 탐지기로 분산 채점합니다.

입력 JSONL에는 최소한 ``text`` 필드가 있어야 합니다. ``prepare``는 텍스트의
SHA-256을 키로 사용하고 글자 수 기준으로 shard 부하를 맞춥니다. ``score``는 결과를
한 행씩 append하므로 같은 명령을 다시 실행하면 완료 지점부터 이어집니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


DEFAULT_OBSERVER = "Qwen/Qwen3-1.7B-Base"
DEFAULT_PERFORMER = "Qwen/Qwen3-1.7B"
DEFAULT_SCORER = "LGAI-EXAONE/EXAONE-4.0-1.2B"


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 오류: {path}:{number}: {exc}") from exc


def input_dir(work_dir: Path) -> Path:
    return work_dir / "inputs"


def score_dir(work_dir: Path) -> Path:
    return work_dir / "scores"


def prepare(paths: list[Path], work_dir: Path, shards: int) -> None:
    if shards < 1:
        raise ValueError("shards는 1 이상이어야 합니다.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("입력 파일이 없습니다: " + ", ".join(missing))
    if not paths:
        raise ValueError("--input으로 JSONL 파일을 하나 이상 지정해야 합니다.")

    unique: dict[str, dict[str, str]] = {}
    references: Counter[str] = Counter()
    for path in paths:
        for row in read_jsonl(path):
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}: 비어 있거나 문자열이 아닌 text가 있습니다.")
            key = text_key(text)
            if key in unique and unique[key]["text"] != text:
                raise RuntimeError("SHA-256 충돌이 발견되었습니다.")
            unique.setdefault(key, {"key": key, "text": text})
            references[str(path)] += 1

    inputs = input_dir(work_dir)
    inputs.mkdir(parents=True, exist_ok=True)
    buckets: list[list[dict[str, str]]] = [[] for _ in range(shards)]
    loads = [0] * shards
    for item in sorted(unique.values(), key=lambda row: (-len(row["text"]), row["key"])):
        shard = min(range(shards), key=lambda index: (loads[index], index))
        buckets[shard].append(item)
        loads[shard] += len(item["text"])
    for shard, rows in enumerate(buckets):
        with (inputs / f"shard_{shard}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "shards": shards,
        "unique_texts": len(unique),
        "input_files": [str(path) for path in paths],
        "references": dict(references),
        "shard_counts": [len(rows) for rows in buckets],
        "shard_chars": loads,
        "binoculars": {
            "observer": DEFAULT_OBSERVER,
            "performer": DEFAULT_PERFORMER,
            "max_length": 1024,
            "score_direction": "higher_is_ai (score=-B)",
        },
        "fastdetectgpt": {
            "scorer": DEFAULT_SCORER,
            "max_length": 2048,
            "score_direction": "higher_is_ai",
        },
    }
    (inputs / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["key"]) for row in read_jsonl(path)}


def score(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    inputs = input_dir(args.work_dir)
    scores = score_dir(args.work_dir)
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    if not 0 <= args.shard < manifest["shards"]:
        raise ValueError(f"shard 범위는 0..{manifest['shards'] - 1}입니다.")
    source = inputs / f"shard_{args.shard}.jsonl"
    scores.mkdir(parents=True, exist_ok=True)
    output = scores / f"{args.detector}_shard_{args.shard}.jsonl"
    done = scores / f"{args.detector}_shard_{args.shard}.done"
    already = completed_keys(output)
    pending = [row for row in read_jsonl(source) if row["key"] not in already]
    print(
        f"[{args.detector} shard={args.shard}] 전체 {len(already) + len(pending):,} · "
        f"완료 {len(already):,} · 남음 {len(pending):,}", flush=True
    )
    if not pending:
        done.write_text("ok\n", encoding="utf-8")
        return
    done.unlink(missing_ok=True)

    if args.detector == "binoculars":
        from detect_binoculars import binoculars

        tokenizer = AutoTokenizer.from_pretrained(args.observer, trust_remote_code=True)
        performer_tokenizer = AutoTokenizer.from_pretrained(
            args.performer, trust_remote_code=True
        )
        if tokenizer.get_vocab() != performer_tokenizer.get_vocab():
            raise RuntimeError("Binoculars observer와 performer의 토크나이저가 다릅니다.")
        observer = AutoModelForCausalLM.from_pretrained(
            args.observer, torch_dtype=torch.float16, trust_remote_code=True
        ).to(args.device).eval()
        performer = AutoModelForCausalLM.from_pretrained(
            args.performer, torch_dtype=torch.float16, trust_remote_code=True
        ).to(args.device).eval()

        def evaluate(text: str):
            result = binoculars(
                observer, performer, tokenizer, text, args.device, max_len=args.max_length
            )
            if result is None:
                return None, None, None
            bino, token_count = result
            return -float(bino), float(bino), int(token_count)

    else:
        from detect_fastdetectgpt import curvature

        tokenizer = AutoTokenizer.from_pretrained(args.scorer, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.scorer, torch_dtype=torch.float16, trust_remote_code=True
        ).to(args.device).eval()

        def evaluate(text: str):
            result = curvature(
                model, tokenizer, text, args.device, max_len=args.max_length
            )
            if result is None:
                return None, None, None
            value, token_count = result
            return float(value), None, int(token_count)

    with output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(pending, 1):
            value, raw_bino, token_count = evaluate(row["text"])
            record = {"key": row["key"], "score": value, "n_tok": token_count}
            if raw_bino is not None:
                record["bino"] = raw_bino
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 100 == 0 or index == len(pending):
                print(
                    f"[{args.detector} shard={args.shard}] {len(already) + index:,}/"
                    f"{len(already) + len(pending):,}", flush=True
                )
    done.write_text("ok\n", encoding="utf-8")


def status(work_dir: Path) -> None:
    inputs = input_dir(work_dir)
    scores = score_dir(work_dir)
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    report = {"expected": manifest["unique_texts"], "detectors": {}}
    for detector in ("binoculars", "fastdetectgpt"):
        counts = []
        for shard in range(manifest["shards"]):
            path = scores / f"{detector}_shard_{shard}.jsonl"
            counts.append(sum(1 for _ in path.open()) if path.exists() else 0)
        report["detectors"][detector] = {
            "scored": sum(counts),
            "shards": counts,
            "done_shards": sum(
                (scores / f"{detector}_shard_{shard}.done").exists()
                for shard in range(manifest["shards"])
            ),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, nargs="+", required=True)
    prepare_parser.add_argument("--shards", type=int, default=6)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument(
        "--detector", choices=("binoculars", "fastdetectgpt"), required=True
    )
    score_parser.add_argument("--shard", type=int, required=True)
    score_parser.add_argument("--device", default="cuda:0")
    score_parser.add_argument("--observer", default=DEFAULT_OBSERVER)
    score_parser.add_argument("--performer", default=DEFAULT_PERFORMER)
    score_parser.add_argument("--scorer", default=DEFAULT_SCORER)
    score_parser.add_argument("--max-length", type=int)
    subparsers.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.input, args.work_dir, args.shards)
    elif args.command == "score":
        if args.max_length is None:
            args.max_length = 1024 if args.detector == "binoculars" else 2048
        score(args)
    else:
        status(args.work_dir)


if __name__ == "__main__":
    main()
