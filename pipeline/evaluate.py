#!/usr/bin/env python3
"""Evaluate one domain with the canonical plain-beam protocol.

Primary metric: raw attack success rate (ASR) at the target-domain human
95th-percentile detector threshold.  Acute collapse and semantic cosine are
reported separately; collapse is not folded into ASR.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS))

from pipeline.metrics_redundancy import acute_collapse, redundancy


def load_test_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL: {path}:{number}: {exc}") from exc
            if row.get("split") == "test":
                rows.append(row)
    if not rows:
        raise ValueError(f"no test rows in {path}")
    required = {"id", "human_text", "ai_text"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"missing pair fields at test row {index}: {', '.join(sorted(missing))}"
            )
    return rows


def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Select exactly ``size`` rows while preserving generator proportions.

    Pair files are often stored in generator-sized blocks.  Taking the first N
    rows silently creates a one-generator evaluation, so allocation uses the
    largest-remainder method and then applies a deterministic shuffle.
    """
    if size <= 0 or size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    if not any(row.get("generator") for row in rows):
        return rng.sample(rows, size)
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[str(row.get("generator") or "unknown")].append(row)
    exact = {name: size * len(group) / len(rows) for name, group in groups.items()}
    quota = {name: math.floor(value) for name, value in exact.items()}
    remaining = size - sum(quota.values())
    order = sorted(groups, key=lambda name: (-(exact[name] - quota[name]), name))
    for name in order[:remaining]:
        quota[name] += 1
    picked = [row for name in sorted(groups) for row in rng.sample(groups[name], quota[name])]
    rng.shuffle(picked)
    assert len(picked) == size
    return picked


DIALOGUE_MARKER = re.compile(r"^\s*[AB]\s*[:：]+\s*", re.MULTILINE)


def strip_dialogue_markers(text: str) -> str:
    """Remove fixed speaker labels for semantic comparison only."""
    return DIALOGUE_MARKER.sub("", text or "")


def load_style_model(checkpoint: Path, device: str):
    import torch
    from stage2_sft import StyleBART

    state = torch.load(checkpoint, map_location=device)["model"]
    fusion = state.get("fusion.weight")
    if fusion is None or fusion.shape[1] != 2 * fusion.shape[0]:
        raise ValueError(f"non-canonical StyleBART fusion in {checkpoint}")
    model = StyleBART().to(device)
    model.load_state_dict(state)
    return model.eval()


def generate_outputs(model, tokenizer, rows, device: str, batch_size: int,
                     input_max_length: int, output_max_length: int, num_beams: int) -> list[str]:
    import torch
    from transformers.modeling_outputs import BaseModelOutput

    order = sorted(range(len(rows)), key=lambda index: len(rows[index]["ai_text"]))
    outputs: list[str | None] = [None] * len(rows)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            encoded = tokenizer(
                [rows[index]["ai_text"] for index in indices],
                return_tensors="pt", padding=True, truncation=True,
                max_length=input_max_length,
            ).to(device)
            content = model.encode(encoded.input_ids, encoded.attention_mask)
            fused = model.fuse(content, model.hsr)
            generated = model.bart.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                attention_mask=encoded.attention_mask,
                num_beams=num_beams,
                max_length=output_max_length,
            )
            for local, index in enumerate(indices):
                outputs[index] = tokenizer.decode(generated[local], skip_special_tokens=True)
    assert all(output is not None for output in outputs)
    return [str(output) for output in outputs]


def roberta_scores(path: Path, texts: list[str], device: str, batch_size: int) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start:start + batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            ).to(device)
            scores.extend(torch.softmax(model(**batch).logits, -1)[:, 1].cpu().tolist())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return scores


def scrn_scores(path: Path, texts: list[str], device: str, batch_size: int) -> list[float]:
    import torch
    from transformers import AutoTokenizer
    from scrn_train import BACKBONE, SCRN

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    saved = torch.load(path, map_location=device)
    model = SCRN().to(device)
    model.load_state_dict(saved["model"] if "model" in saved else saved)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start:start + batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            ).to(device)
            hidden = model.enc(**batch).last_hidden_state[:, 0]
            scores.extend(torch.softmax(model.clf(model.enc_s(hidden)), -1)[:, 1].cpu().tolist())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return scores


def semantic_cosines(reference: list[str], candidates: dict[str, list[str]],
                     device: str, batch_size: int) -> dict[str, list[float]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = "jhgan/ko-sroberta-multitask"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    def embed(texts: list[str]):
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = tokenizer(
                    texts[start:start + batch_size], return_tensors="pt",
                    padding=True, truncation=True, max_length=256,
                ).to(device)
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(chunks)

    ref = embed(reference)
    result = {name: (ref * embed(texts)).sum(-1).tolist() for name, texts in candidates.items()}
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--scrn", type=Path)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--dpo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=0, help="0 evaluates the full test split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--input-max-length", type=int, default=512)
    parser.add_argument("--output-max-length", type=int, default=1024)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--scoring-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dialogue", action="store_true",
                        help="strip A:/B: labels for semantic cosine only")
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from stage2_sft import MODEL as BART_MODEL

    rows = stratified_sample(load_test_rows(args.pairs), args.n, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
    sets = {
        "human": [row["human_text"] for row in rows],
        "x_ai": [row["ai_text"] for row in rows],
    }
    for label, checkpoint in (("SFT", args.sft), ("DPO", args.dpo)):
        model = load_style_model(checkpoint, args.device)
        sets[label] = generate_outputs(
            model, tokenizer, rows, args.device, args.generation_batch_size,
            args.input_max_length, args.output_max_length, args.num_beams,
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    labels = list(sets)
    joined = [text for label in labels for text in sets[label]]
    detector_scores = {"D": roberta_scores(args.detector, joined, args.device, args.scoring_batch_size)}
    if args.scrn:
        detector_scores["SCRN"] = scrn_scores(args.scrn, joined, args.device, args.scoring_batch_size)
    size = len(rows)
    split_scores = {
        detector: {label: scores[i * size:(i + 1) * size] for i, label in enumerate(labels)}
        for detector, scores in detector_scores.items()
    }
    semantic_sets = {
        label: ([strip_dialogue_markers(text) for text in texts] if args.dialogue else texts)
        for label, texts in sets.items()
    }
    cosine = semantic_cosines(
        semantic_sets["x_ai"],
        {label: semantic_sets[label] for label in labels if label != "x_ai"},
        args.device, args.scoring_batch_size,
    )
    cosine["x_ai"] = [1.0] * size
    repetitions = {
        label: [redundancy(text, args.dialogue) for text in sets[label]]
        for label in labels
    }
    collapse = {label: [item["acute"] for item in repetitions[label]] for label in labels}

    summary = {
        "schema_version": 1,
        "domain": args.domain,
        "test_size": size,
        "generator_counts": dict(sorted(collections.Counter(
            str(row.get("generator") or "unknown") for row in rows
        ).items())),
        "protocol": {
            "decoding": "plain_beam",
            "num_beams": args.num_beams,
            "input_max_length": args.input_max_length,
            "output_max_length": args.output_max_length,
            "asr_threshold": "human_p95",
            "collapse": "whitespace-free character 6-gram repeated more than 20 times",
        },
        "detectors": {},
        "metrics": {
            label: {
                "collapse_percent": 100 * float(np.mean(collapse[label])),
                "semantic_cosine": float(np.mean(cosine[label])),
                "sentence_reuse_per_1000_pairs": float(np.mean(
                    [item["reuse"] for item in repetitions[label]]
                )),
                "approximate_overlap_per_1000_pairs": float(np.mean(
                    [item["approx"] for item in repetitions[label]]
                )),
                "raw_asr_percent": {},
            }
            for label in labels
        },
    }

    print(f"[{args.domain}] test n={size} · plain beam{args.num_beams}")
    for detector, groups in split_scores.items():
        threshold = float(np.percentile(groups["human"], 95))
        human_mean = float(np.mean(groups["human"]))
        x_ai_mean = float(np.mean(groups["x_ai"]))
        gap = x_ai_mean - human_mean
        validity = "valid" if gap > 0.4 else ("weak" if gap > 0.15 else "invalid")
        summary["detectors"][detector] = {
            "threshold": threshold,
            "human_mean_p_ai": human_mean,
            "x_ai_mean_p_ai": x_ai_mean,
            "separation_gap": gap,
            "validity": validity,
        }
        print(f"{detector} threshold={threshold:.6f} gap={gap:.3f} ({validity})")
        for label in labels:
            asr = 100 * float(np.mean(np.asarray(groups[label]) <= threshold))
            summary["metrics"][label]["raw_asr_percent"][detector] = asr
            print(
                f"  {label:<5} raw_ASR={asr:5.1f}% "
                f"collapse={summary['metrics'][label]['collapse_percent']:5.1f}% "
                f"cos={summary['metrics'][label]['semantic_cosine']:.3f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for label in labels:
            for index, text in enumerate(sets[label]):
                row = {
                    "domain": args.domain,
                    "dec": label,
                    "doc_id": rows[index]["id"],
                    "generator": rows[index].get("generator"),
                    "text": text,
                    **{f"p_{detector}": split_scores[detector][label][index]
                       for detector in split_scores},
                    "cos": cosine[label][index],
                    "rep_reuse": repetitions[label][index]["reuse"],
                    "rep_approx": repetitions[label][index]["approx"],
                    "rep_acute": collapse[label][index],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.out)

    summary_path = args.out.with_suffix(".summary.json")
    summary_temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_temporary.replace(summary_path)
    print(f"saved: {args.out} · {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
