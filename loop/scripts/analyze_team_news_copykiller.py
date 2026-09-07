#!/usr/bin/env python3
"""팀 뉴스 G1/G2 카피킬러 결과를 매니페스트와 조인해 paired 분석한다.

카피킬러 결과확인서 표에서는 긴 문서명이 발급번호/AI작성률 행과 다른 줄로
밀릴 수 있다. 따라서 인접 줄의 고아 문서명을 복구한 뒤, PDF와 manifest의
파일명 집합이 완전히 같을 때만 결과를 기록한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pdfplumber


ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d{11})\s+(?:([AB]\d{4}\.docx)\s+)?(\d+)%\s*$"
)
LONE_FILE_RE = re.compile(r"^\s*([AB]\d{4}\.docx)\s*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_pdf(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return per-file rows and first-page report metadata."""
    with pdfplumber.open(path) as pdf:
        page_texts = [page.extract_text(layout=True) or "" for page in pdf.pages]

    lines = "\n".join(page_texts).splitlines()
    parsed: dict[str, dict[str, Any]] = {}
    used_orphan_lines: set[int] = set()

    for index, line in enumerate(lines):
        match = ROW_RE.match(line)
        if not match:
            continue
        row_number, issue_number, filename, ai_pct = match.groups()
        if filename is None:
            candidates: list[tuple[int, int, str]] = []
            for nearby in (index - 1, index + 1, index - 2, index + 2):
                if not 0 <= nearby < len(lines) or nearby in used_orphan_lines:
                    continue
                orphan = LONE_FILE_RE.match(lines[nearby])
                if orphan:
                    candidates.append((abs(nearby - index), nearby, orphan.group(1)))
            if not candidates:
                raise ValueError(
                    f"{path.name}: 표 {row_number}번 행의 분리된 문서명을 복구하지 못했습니다"
                )
            _, orphan_index, filename = min(candidates)
            used_orphan_lines.add(orphan_index)

        if filename in parsed:
            raise ValueError(f"{path.name}: 중복 문서명 {filename}")
        parsed[filename] = {
            "report_row": int(row_number),
            "document_issue_number": issue_number,
            "ck_ai_pct": int(ai_pct),
        }

    first_page = page_texts[0]

    def required(pattern: str, label: str) -> str:
        match = re.search(pattern, first_page)
        if not match:
            raise ValueError(f"{path.name}: {label} 메타데이터를 읽지 못했습니다")
        return match.group(1).strip()

    metadata = {
        "source_pdf": str(path.resolve()),
        "source_pdf_sha256": sha256(path),
        "page_count": len(page_texts),
        "inspection_number": required(r"검사번호\s+(\d{11})", "검사번호"),
        "inspection_name": required(r"검사명\s+(.+?)\s+발급일자", "검사명"),
        "displayed_mean_ai_pct": int(
            required(r"평균 AI작성률\s+(\d+)%", "평균 AI작성률")
        ),
        "displayed_max_ai_pct": int(
            required(r"최고 AI작성률\s+(\d+)%", "최고 AI작성률")
        ),
        "registered_documents": int(
            required(r"등록문서 수\s+(\d+)", "등록문서 수")
        ),
        "inspected_documents": int(
            required(r"검사문서 수\s+(\d+)", "검사문서 수")
        ),
    }
    return parsed, metadata


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return [100 * (center - half), 100 * (center + half)]


def exact_two_sided_binomial(successes: int, total: int) -> float:
    """Two-sided exact p-value for H0: p=0.5."""
    if total == 0:
        return 1.0
    tail = min(successes, total - successes)
    probability = sum(math.comb(total, i) for i in range(tail + 1)) / (2**total)
    return min(1.0, 2 * probability)


def bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, samples: int, chunk_size: int = 2000
) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def summarize_scores(values: np.ndarray, tau: int) -> dict[str, Any]:
    detected = int(np.count_nonzero(values >= tau))
    total = int(len(values))
    return {
        "n": total,
        "mean_ai_pct": float(values.mean()),
        "median_ai_pct": float(np.median(values)),
        "q1_ai_pct": float(np.percentile(values, 25)),
        "q3_ai_pct": float(np.percentile(values, 75)),
        "max_ai_pct": int(values.max()),
        "zero_count": int(np.count_nonzero(values == 0)),
        "zero_rate_pct": float(100 * np.mean(values == 0)),
        "detected_ai_count": detected,
        "detected_ai_rate_pct": 100 * detected / total,
        "detected_ai_rate_wilson_95ci_pct": wilson_interval(detected, total),
        "asr_count": total - detected,
        "asr_pct": 100 * (total - detected) / total,
        "asr_wilson_95ci_pct": wilson_interval(total - detected, total),
    }


def format_ci(values: list[float]) -> str:
    return f"{values[0]:.2f}–{values[1]:.2f}"


def write_report(path: Path, result: dict[str, Any]) -> None:
    g1 = result["variants"]["g1"]
    g2 = result["variants"]["g2"]
    paired = result["paired"]
    tau = result["tau_ai_pct"]
    lines = [
        "# 팀 뉴스 G1/G2 카피킬러 paired 결과",
        "",
        f"> 표본: 동일 test ID {result['n_pairs']}쌍 · 판정 임계값: AI작성률 ≥ {tau}%",
        "",
        "## 핵심 결과",
        "",
        "| 지표 | G1 | G2 | G2 − G1 |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| 평균 AI작성률 | {g1['mean_ai_pct']:.3f}% | {g2['mean_ai_pct']:.3f}% | "
            f"{paired['mean_delta_g2_minus_g1_pct']:+.3f}%p |"
        ),
        (
            f"| AI 탐지율 (≥{tau}%) | {g1['detected_ai_rate_pct']:.2f}% "
            f"({g1['detected_ai_count']}/{g1['n']}) | {g2['detected_ai_rate_pct']:.2f}% "
            f"({g2['detected_ai_count']}/{g2['n']}) | "
            f"{paired['detected_rate_delta_g2_minus_g1_pct']:+.2f}%p |"
        ),
        (
            f"| ASR (<{tau}%) | {g1['asr_pct']:.2f}% ({g1['asr_count']}/{g1['n']}) | "
            f"{g2['asr_pct']:.2f}% ({g2['asr_count']}/{g2['n']}) | "
            f"{paired['asr_delta_g2_minus_g1_pct']:+.2f}%p |"
        ),
        f"| 최고 AI작성률 | {g1['max_ai_pct']}% | {g2['max_ai_pct']}% | — |",
        "",
        (
            f"평균 AI작성률의 paired 차이(G2−G1)는 "
            f"**{paired['mean_delta_g2_minus_g1_pct']:+.3f}%p**이며, "
            f"bootstrap 95% CI는 **{format_ci(paired['mean_delta_bootstrap_95ci_pct'])}%p**다."
        ),
        (
            f"개별 쌍 기준으로 G2 점수가 낮아진 문서는 {paired['decreased_count']}건, "
            f"높아진 문서는 {paired['increased_count']}건, 같은 문서는 {paired['tied_count']}건이다 "
            f"(sign test p={paired['sign_test_two_sided_p']:.6g})."
        ),
        "",
        "## 50% 임계값 paired 전이",
        "",
        "| G1 판정 → G2 판정 | 건수 |",
        "| --- | ---: |",
        f"| 사람 → 사람 | {paired['threshold_transitions']['human_to_human']} |",
        f"| 사람 → AI | {paired['threshold_transitions']['human_to_ai']} |",
        f"| AI → 사람 | {paired['threshold_transitions']['ai_to_human']} |",
        f"| AI → AI | {paired['threshold_transitions']['ai_to_ai']} |",
        "",
        (
            f"불일치 쌍에 대한 exact McNemar p-value는 "
            f"**{paired['mcnemar_exact_two_sided_p']:.6g}**다. 탐지율 차이의 paired bootstrap "
            f"95% CI는 **{format_ci(paired['detected_rate_delta_bootstrap_95ci_pct'])}%p**다."
        ),
        "",
        "## 무결성 확인",
        "",
        f"- G1 PDF: {result['reports']['g1']['inspection_number']} · "
        f"{result['reports']['g1']['parsed_documents']}건 파싱 · "
        f"SHA256 `{result['reports']['g1']['source_pdf_sha256']}`",
        f"- G2 PDF: {result['reports']['g2']['inspection_number']} · "
        f"{result['reports']['g2']['parsed_documents']}건 파싱 · "
        f"SHA256 `{result['reports']['g2']['source_pdf_sha256']}`",
        "- PDF 파일명 집합과 export manifest 파일명 집합이 G1/G2 각각 완전히 일치한다.",
        "- G1과 G2는 동일한 350개 `pair_id`로 정렬해 비교했다.",
        "",
        "## 해석 경계",
        "",
        "- 이 결과는 카피킬러 한 탐지기, 보도자료 문서 유형, 현재 검사 시점에 대한 외부 탐지 결과다.",
        "- PDF가 표시하는 평균은 정수 반올림값이므로, 통계에는 350개 행의 개별 정수 점수 평균을 사용했다.",
        "- 새로 재학습한 D2 평가는 아니며, 로컬 D1/팀 D2 수치와 별도 증거로 유지한다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-pdf", type=Path, required=True)
    parser.add_argument("--g2-pdf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tau", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 <= args.tau <= 100:
        raise ValueError("--tau must be between 0 and 100")

    manifest_rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    by_variant = {
        variant: {row["file"]: row for row in manifest_rows if row["variant"] == variant}
        for variant in ("g1", "g2")
    }
    if any(len(rows) != 350 for rows in by_variant.values()):
        raise ValueError(
            f"manifest variant counts must be 350 each, got "
            f"g1={len(by_variant['g1'])}, g2={len(by_variant['g2'])}"
        )

    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for variant, pdf_path in (("g1", args.g1_pdf), ("g2", args.g2_pdf)):
        parsed[variant], reports[variant] = extract_pdf(pdf_path)
        pdf_files = set(parsed[variant])
        manifest_files = set(by_variant[variant])
        if pdf_files != manifest_files:
            missing = sorted(manifest_files - pdf_files)
            extra = sorted(pdf_files - manifest_files)
            raise ValueError(
                f"{variant}: PDF/manifest mismatch; missing={missing[:10]}, extra={extra[:10]}"
            )
        reports[variant]["parsed_documents"] = len(parsed[variant])
        values = [row["ck_ai_pct"] for row in parsed[variant].values()]
        if reports[variant]["registered_documents"] != len(values):
            raise ValueError(f"{variant}: registered-document count mismatch")
        if reports[variant]["inspected_documents"] != len(values):
            raise ValueError(f"{variant}: inspected-document count mismatch")
        if reports[variant]["displayed_max_ai_pct"] != max(values):
            raise ValueError(f"{variant}: displayed maximum does not match parsed rows")

    g1_pair_ids = {row["pair_id"] for row in by_variant["g1"].values()}
    g2_pair_ids = {row["pair_id"] for row in by_variant["g2"].values()}
    if g1_pair_ids != g2_pair_ids or len(g1_pair_ids) != 350:
        raise ValueError("G1/G2 pair_id sets are not the same 350 IDs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preserved_names = {
        "g1": "g1_copykiller_result_370276476.pdf",
        "g2": "g2_copykiller_result_370277546.pdf",
    }
    for variant, source in (("g1", args.g1_pdf), ("g2", args.g2_pdf)):
        destination = args.out_dir / preserved_names[variant]
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        if sha256(destination) != reports[variant]["source_pdf_sha256"]:
            raise ValueError(f"{variant}: preserved PDF checksum mismatch")
        reports[variant]["preserved_pdf"] = str(destination.resolve())

    paired_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for pair_id in sorted(g1_pair_ids):
        pair: dict[str, Any] = {"pair_id": pair_id}
        for variant in ("g1", "g2"):
            manifest_row = next(
                row for row in by_variant[variant].values() if row["pair_id"] == pair_id
            )
            score_row = parsed[variant][manifest_row["file"]]
            combined = {
                **manifest_row,
                **score_row,
                "variant": variant,
                "report_inspection_number": reports[variant]["inspection_number"],
                "source_pdf_sha256": reports[variant]["source_pdf_sha256"],
            }
            long_rows.append(combined)
            pair[f"{variant}_file"] = manifest_row["file"]
            pair[f"{variant}_chars"] = int(manifest_row["chars"])
            pair[f"{variant}_ck_ai_pct"] = score_row["ck_ai_pct"]
        pair["delta_g2_minus_g1_pct"] = pair["g2_ck_ai_pct"] - pair["g1_ck_ai_pct"]
        paired_rows.append(pair)

    g1_values = np.asarray([row["g1_ck_ai_pct"] for row in paired_rows], dtype=np.float64)
    g2_values = np.asarray([row["g2_ck_ai_pct"] for row in paired_rows], dtype=np.float64)
    deltas = g2_values - g1_values
    g1_detected = g1_values >= args.tau
    g2_detected = g2_values >= args.tau
    detected_delta = 100.0 * (g2_detected.astype(float) - g1_detected.astype(float))

    decreased = int(np.count_nonzero(deltas < 0))
    increased = int(np.count_nonzero(deltas > 0))
    ties = int(np.count_nonzero(deltas == 0))
    human_to_ai = int(np.count_nonzero(~g1_detected & g2_detected))
    ai_to_human = int(np.count_nonzero(g1_detected & ~g2_detected))

    result = {
        "schema_version": 1,
        "n_pairs": len(paired_rows),
        "tau_ai_pct": args.tau,
        "manifest": str(args.manifest.resolve()),
        "reports": reports,
        "variants": {
            "g1": summarize_scores(g1_values, args.tau),
            "g2": summarize_scores(g2_values, args.tau),
        },
        "paired": {
            "mean_delta_g2_minus_g1_pct": float(deltas.mean()),
            "median_delta_g2_minus_g1_pct": float(np.median(deltas)),
            "mean_delta_bootstrap_95ci_pct": bootstrap_mean_ci(
                deltas,
                seed=args.seed,
                samples=args.bootstrap_samples,
            ),
            "decreased_count": decreased,
            "increased_count": increased,
            "tied_count": ties,
            "sign_test_two_sided_p": exact_two_sided_binomial(
                decreased, decreased + increased
            ),
            "threshold_transitions": {
                "human_to_human": int(np.count_nonzero(~g1_detected & ~g2_detected)),
                "human_to_ai": human_to_ai,
                "ai_to_human": ai_to_human,
                "ai_to_ai": int(np.count_nonzero(g1_detected & g2_detected)),
            },
            "detected_rate_delta_g2_minus_g1_pct": float(detected_delta.mean()),
            "detected_rate_delta_bootstrap_95ci_pct": bootstrap_mean_ci(
                detected_delta,
                seed=args.seed + 1,
                samples=args.bootstrap_samples,
            ),
            "asr_delta_g2_minus_g1_pct": float(-detected_delta.mean()),
            "mcnemar_exact_two_sided_p": exact_two_sided_binomial(
                human_to_ai, human_to_ai + ai_to_human
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
        },
    }

    with (args.out_dir / "copykiller_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in long_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (args.out_dir / "paired_scores.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    (args.out_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.out_dir / "report.md", result)

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir.resolve()),
                "g1_parsed": len(parsed["g1"]),
                "g2_parsed": len(parsed["g2"]),
                "g1_mean": result["variants"]["g1"]["mean_ai_pct"],
                "g2_mean": result["variants"]["g2"]["mean_ai_pct"],
                "g1_asr": result["variants"]["g1"]["asr_pct"],
                "g2_asr": result["variants"]["g2"]["asr_pct"],
                "mean_delta": result["paired"]["mean_delta_g2_minus_g1_pct"],
                "mcnemar_p": result["paired"]["mcnemar_exact_two_sided_p"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
