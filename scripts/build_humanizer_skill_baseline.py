#!/usr/bin/env python3
"""Humanizer-skill 프롬프트 기준선의 고정 held-out 입력을 구성합니다."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = REPO_ROOT / "baselines" / "humanizer_skill_batch_v1.6.0.txt"
CONDITION = "humanizer_skill_v1.6.0"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 오류: {path}:{number}: {exc}") from exc


def load_test(path: Path) -> list[dict]:
    rows = [row for row in read_jsonl(path) if row.get("split") == "test"]
    if not rows:
        raise ValueError(f"test 행이 없습니다: {path}")
    required = {"id", "ai_text"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path} test 행 {index} 필드 누락: {sorted(missing)}")
    return rows


def select_like_original_eval(rows: list[dict], size: int) -> list[dict]:
    """실험 당시 eval_domain_full.py와 동일한 층화·seed 규칙입니다."""
    if size <= 0 or size >= len(rows):
        return list(rows)
    if not rows[0].get("generator"):
        return rows[:size]
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[str(row["generator"])].append(row)
    picked = []
    for name in sorted(groups):
        count = max(1, round(size * len(groups[name]) / len(rows)))
        picked.extend(groups[name][:count])
    random.Random(42).shuffle(picked)
    return picked[:size]


def select_for_comparison(domain: str, rows: list[dict], size: int,
                          reference_dir: Path | None) -> list[dict]:
    if reference_dir is None:
        return select_like_original_eval(rows, size)
    reference = reference_dir / f"{domain}_to_{domain}.jsonl"
    if not reference.exists():
        return select_like_original_eval(rows, size)
    reference_ids = [
        str(row["doc_id"]) for row in read_jsonl(reference) if row.get("dec") == "human"
    ]
    mapping = {str(row["id"]): row for row in rows}
    if size == len(reference_ids):
        missing = [doc_id for doc_id in reference_ids if doc_id not in mapping]
        if missing:
            raise RuntimeError(f"{domain}: 기존 비교 결과의 ID가 pair에 없습니다: {missing[:5]}")
        return [mapping[doc_id] for doc_id in reference_ids]
    selected = select_like_original_eval(rows, size)
    outside = [str(row["id"]) for row in selected if str(row["id"]) not in set(reference_ids)]
    if outside:
        raise RuntimeError(f"{domain}: n={size} 표본이 기존 비교 표본 밖입니다: {outside[:5]}")
    return selected


def generator_engine(value: object) -> str:
    name = str(value).lower()
    if "qwen" in name:
        return "qwen"
    if "exaone" in name:
        return "exaone"
    raise ValueError(f"지원하지 않는 generator입니다: {value!r}")


def write_stage(domains: dict[str, dict], output_root: Path, prompt: str, size: int,
                qwen_shards: int, reference_dir: Path | None) -> None:
    stage = output_root / f"n{size}"
    stage.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict]] = {
        **{f"qwen_{index}": [] for index in range(qwen_shards)},
        "exaone_0": [],
    }
    manifest = []
    for domain, domain_config in domains.items():
        rows = load_test(resolve(domain_config["pairs"]))
        for row in select_for_comparison(domain, rows, size, reference_dir):
            engine = generator_engine(row.get("generator"))
            if engine == "qwen":
                stable = f"{domain}:{row['id']}".encode("utf-8")
                shard = int(hashlib.sha256(stable).hexdigest()[:8], 16) % qwen_shards
                bucket = f"qwen_{shard}"
            else:
                bucket = "exaone_0"
            source_id = str(row["id"])
            item = {
                "doc_id": f"{domain}:{source_id}",
                "source_id": source_id,
                "domain": domain,
                "generator": row.get("generator"),
                "engine": engine,
                "cond": CONDITION,
                "system": prompt,
                "prompt": "[교정할 원문]\n\n" + row["ai_text"],
                "target_char": len(row["ai_text"]),
                "n_char": len(row["ai_text"]),
            }
            buckets[bucket].append(item)
            manifest.append({
                key: item[key] for key in
                ("doc_id", "source_id", "domain", "generator", "engine", "cond")
            })
    for name, rows in buckets.items():
        with (stage / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (stage / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = " · ".join(f"{name}={len(rows)}" for name, rows in buckets.items())
    print(f"n={size}: {len(manifest)}편 · {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=(20, 100))
    parser.add_argument("--qwen-shards", type=int, default=5)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.out_dir or resolve(config.get(
        "prompt_output_dir", "dataset/humanizer_skill_baseline/prompts"
    ))
    reference_value = config.get("reference_dir")
    reference = resolve(reference_value) if reference_value else None
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    for size in args.sizes:
        write_stage(config["domains"], output, prompt, size, args.qwen_shards, reference)


if __name__ == "__main__":
    main()
