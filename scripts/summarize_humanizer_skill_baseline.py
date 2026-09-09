#!/usr/bin/env python3
"""Humanizer-skill 기준선과 같은 문서의 SFT·DPO 결과를 집계합니다."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


DEFAULT_DOMAINS = ("news", "essay", "persona", "written", "petition512B", "wiki512B")
METHODS = ("x_ai", "SFT", "DPO", "Humanizer-skill")
CONTAMINATION = re.compile(
    r"(^|\n)\s*(#{1,6}\s*)?(분석 결과|자연도 등급|주요 변경 사항|의미 보존 검증)", re.I
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(rows: list[dict], label: str, ids: set[str]) -> list[dict]:
    mapping = {str(row["doc_id"]): row for row in rows if row["dec"] == label}
    missing = sorted(ids - set(mapping))
    if missing:
        raise RuntimeError(f"{label} 결과 누락 {len(missing)}건: {missing[:5]}")
    return [mapping[doc_id] for doc_id in sorted(ids)]


def metrics(rows: list[dict], thresholds: dict[str, float]) -> dict[str, float]:
    result = {}
    for detector in ("D", "SCRN"):
        scores = np.asarray([row[f"p_{detector}"] for row in rows], dtype=float)
        result[f"{detector}_asr"] = 100 * float(np.mean(scores <= thresholds[detector]))
        result[f"{detector}_pai"] = float(np.mean(scores))
    result["cos"] = float(np.mean([row["cos"] for row in rows]))
    result["acute"] = 100 * float(np.mean([row["rep_acute"] for row in rows]))
    result["reuse"] = float(np.mean([row["rep_reuse"] for row in rows]))
    result["approx"] = float(np.mean([row["rep_approx"] for row in rows]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--transfer-dir", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--gate-json", type=Path)
    args = parser.parse_args()

    all_metrics: dict[str, dict] = {}
    validity: dict[str, dict] = {}
    gate_report: dict = {"n": args.n, "passed": True, "domains": {}}
    for domain in args.domains:
        external = read_jsonl(args.eval_dir / f"{domain}.jsonl")
        external_by = {
            label: [row for row in external if row["dec"] == label]
            for label in ("human", "x_ai", "Humanizer-skill")
        }
        ids = {str(row["doc_id"]) for row in external_by["human"]}
        if len(ids) != args.n:
            raise RuntimeError(f"{domain}: 기대 {args.n}편, 실제 {len(ids)}편")

        diagonal = read_jsonl(args.transfer_dir / f"{domain}_to_{domain}.jsonl")
        diagonal_by = {
            label: select(diagonal, label, ids)
            for label in ("human", "x_ai", "SFT", "DPO")
        }
        thresholds = {
            detector: float(np.percentile(
                [row[f"p_{detector}"] for row in external_by["human"]], 95
            ))
            for detector in ("D", "SCRN")
        }
        validity[domain] = {}
        for detector in ("D", "SCRN"):
            human_mean = float(np.mean([
                row[f"p_{detector}"] for row in external_by["human"]
            ]))
            x_ai_mean = float(np.mean([
                row[f"p_{detector}"] for row in external_by["x_ai"]
            ]))
            gap = x_ai_mean - human_mean
            validity[domain][detector] = {"gap": gap, "valid": gap > 0.15}

        all_metrics[domain] = {
            "x_ai": metrics(diagonal_by["x_ai"], thresholds),
            "SFT": metrics(diagonal_by["SFT"], thresholds),
            "DPO": metrics(diagonal_by["DPO"], thresholds),
            "Humanizer-skill": metrics(external_by["Humanizer-skill"], thresholds),
        }
        originals = {
            str(row["doc_id"]): row["text"] for row in external_by["x_ai"]
        }
        rewritten = external_by["Humanizer-skill"]
        ratios = [
            len(row["text"]) / max(1, len(originals[str(row["doc_id"])]))
            for row in rewritten
        ]
        contaminated = sum(bool(CONTAMINATION.search(row["text"])) for row in rewritten)
        skill = all_metrics[domain]["Humanizer-skill"]
        reasons = []
        if not 0.65 <= float(np.median(ratios)) <= 1.35:
            reasons.append("중앙 길이비 범위 이탈")
        if contaminated > max(1, round(0.05 * args.n)):
            reasons.append("분석/설명 출력 5% 초과")
        if skill["cos"] < 0.85:
            reasons.append("의미 cosine 0.85 미만")
        if skill["acute"] > 5.0:
            reasons.append("급성붕괴 5% 초과")
        gate_report["domains"][domain] = {
            "count": len(rewritten),
            "median_length_ratio": float(np.median(ratios)),
            "contaminated": contaminated,
            "cos": skill["cos"],
            "acute_pct": skill["acute"],
            "passed": not reasons,
            "reasons": reasons,
        }
        gate_report["passed"] = gate_report["passed"] and not reasons

    lines = [
        f"# Humanizer-skill 프롬프팅 기준선 (n={args.n}/도메인)",
        "",
        "동일한 held-out 문서에서 각 `x_ai`의 원 생성기와 같은 계열 모델을 사용했습니다. "
        "Qwen 생성문은 Qwen3-8B, EXAONE 생성문은 EXAONE-3.5-7.8B로 재작성했습니다.",
        "",
        "| 도메인 | 방법 | D ASR | SCRN ASR | 의미 cos | 급성붕괴 | 재사용 | 근사반복 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain in args.domains:
        for method in METHODS:
            item = all_metrics[domain][method]
            scrn = f"{item['SCRN_asr']:.1f}%"
            if not validity[domain]["SCRN"]["valid"]:
                scrn += "†"
            lines.append(
                f"| {domain} | {method} | {item['D_asr']:.1f}% | {scrn} | "
                f"{item['cos']:.3f} | {item['acute']:.1f}% | "
                f"{item['reuse']:.2f} | {item['approx']:.2f} |"
            )

    lines.extend([
        "", "## 도메인 매크로 평균", "",
        "| 방법 | D ASR | 유효 SCRN ASR | 의미 cos | 급성붕괴 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for method in METHODS:
        values = [all_metrics[domain][method] for domain in args.domains]
        valid_scrn = [
            all_metrics[domain][method]["SCRN_asr"] for domain in args.domains
            if validity[domain]["SCRN"]["valid"]
        ]
        scrn_mean = float(np.mean(valid_scrn)) if valid_scrn else float("nan")
        lines.append(
            f"| {method} | {np.mean([item['D_asr'] for item in values]):.1f}% | "
            f"{scrn_mean:.1f}% | {np.mean([item['cos'] for item in values]):.3f} | "
            f"{np.mean([item['acute'] for item in values]):.1f}% |"
        )
    lines.extend([
        "",
        "† SCRN의 인간–x_ai 평균 점수 차이가 0.15 이하인 도메인은 무효로 "
        "표시했으며 매크로 평균에서 제외했습니다.",
        "", "## 품질 게이트", "",
        f"전체 판정: **{'통과' if gate_report['passed'] else '실패'}**", "",
        "| 도메인 | 길이비 중앙값 | 설명 혼입 | 의미 cos | 급성붕괴 | 판정 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for domain in args.domains:
        item = gate_report["domains"][domain]
        lines.append(
            f"| {domain} | {item['median_length_ratio']:.2f} | {item['contaminated']} | "
            f"{item['cos']:.3f} | {item['acute_pct']:.1f}% | "
            f"{'통과' if item['passed'] else '실패'} |"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.gate_json:
        args.gate_json.parent.mkdir(parents=True, exist_ok=True)
        args.gate_json.write_text(
            json.dumps(gate_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"→ {args.out}")
    print("GATE", "PASS" if gate_report["passed"] else "FAIL")
    if args.gate and not gate_report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
