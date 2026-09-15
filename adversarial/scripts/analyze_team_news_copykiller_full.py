#!/usr/bin/env python3
"""Analyze full-test CopyKiller results for Human, G0, G1, and G2.

All four reports must contain the same 1,985 frozen test IDs through their
variant-specific filenames. The script refuses to write results if any PDF and
manifest filename set differs. Scores are compared pairwise at CopyKiller's
50% AI-writing-rate threshold, with bootstrap intervals and exact McNemar tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader

from analyze_team_news_copykiller import (
    bootstrap_mean_ci,
    exact_two_sided_binomial,
    format_ci,
    sha256,
    summarize_scores,
)


ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d{11})\s+([ABHX]\d{4}\.docx)\s+(\d+)%\s*$"
)
VARIANT_PREFIX = {"human": "H", "g0": "X", "g1": "A", "g2": "B"}


def parse_int(value: str) -> int:
    return int(value.replace(",", ""))


def extract_pdf(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reader = PdfReader(path)
    page_texts = [
        page.extract_text(extraction_mode="layout") or "" for page in reader.pages
    ]
    parsed: dict[str, dict[str, Any]] = {}
    for line in "\n".join(page_texts).splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        row_number, issue_number, filename, ai_pct = match.groups()
        if filename in parsed:
            raise ValueError(f"{path.name}: duplicate filename {filename}")
        parsed[filename] = {
            "report_row": int(row_number),
            "document_issue_number": issue_number,
            "ck_ai_pct": int(ai_pct),
        }

    first_page = page_texts[0]

    def required(pattern: str, label: str) -> str:
        match = re.search(pattern, first_page)
        if not match:
            raise ValueError(f"{path.name}: failed to read {label}")
        return match.group(1).strip()

    metadata = {
        "source_pdf": str(path.resolve()),
        "source_pdf_sha256": sha256(path),
        "page_count": len(page_texts),
        "inspection_number": required(r"검사번호\s+(\d{11})", "inspection number"),
        "inspection_name": required(r"검사명\s+(.+?)\s+발급일자", "inspection name"),
        "displayed_mean_ai_pct": parse_int(
            required(r"평균 AI작성률\s+([\d,]+)%", "mean AI rate")
        ),
        "displayed_max_ai_pct": parse_int(
            required(r"최고 AI작성률\s+([\d,]+)%", "max AI rate")
        ),
        "registered_documents": parse_int(
            required(r"등록문서 수\s+([\d,]+)", "registered count")
        ),
        "inspected_documents": parse_int(
            required(r"검사문서 수\s+([\d,]+)", "inspected count")
        ),
    }
    return parsed, metadata


def auc_from_scores(negative: np.ndarray, positive: np.ndarray) -> float:
    values = np.concatenate([negative, positive])
    labels = np.concatenate(
        [np.zeros(len(negative), dtype=np.int8), np.ones(len(positive), dtype=np.int8)]
    )
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    positive_rank_sum = float(ranks[labels == 1].sum())
    n_positive = len(positive)
    n_negative = len(negative)
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)


def pairwise_stats(
    before: np.ndarray,
    after: np.ndarray,
    *,
    tau: int,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    delta = after - before
    before_detected = before >= tau
    after_detected = after >= tau
    detection_delta = 100.0 * (
        after_detected.astype(np.float64) - before_detected.astype(np.float64)
    )
    decreased = int(np.count_nonzero(delta < 0))
    increased = int(np.count_nonzero(delta > 0))
    below_to_detected = int(np.count_nonzero(~before_detected & after_detected))
    detected_to_below = int(np.count_nonzero(before_detected & ~after_detected))
    return {
        "mean_score_delta_after_minus_before_pct": float(delta.mean()),
        "median_score_delta_after_minus_before_pct": float(np.median(delta)),
        "mean_score_delta_bootstrap_95ci_pct": bootstrap_mean_ci(
            delta, seed=seed, samples=bootstrap_samples
        ),
        "score_decreased_count": decreased,
        "score_increased_count": increased,
        "score_tied_count": int(np.count_nonzero(delta == 0)),
        "sign_test_two_sided_p": exact_two_sided_binomial(
            decreased, decreased + increased
        ),
        "detection_rate_delta_after_minus_before_pct": float(detection_delta.mean()),
        "detection_rate_delta_bootstrap_95ci_pct": bootstrap_mean_ci(
            detection_delta, seed=seed + 100, samples=bootstrap_samples
        ),
        "asr_delta_after_minus_before_pct": float(-detection_delta.mean()),
        "threshold_transitions": {
            "below_to_below": int(np.count_nonzero(~before_detected & ~after_detected)),
            "below_to_detected": below_to_detected,
            "detected_to_below": detected_to_below,
            "detected_to_detected": int(np.count_nonzero(before_detected & after_detected)),
        },
        "mcnemar_exact_two_sided_p": exact_two_sided_binomial(
            below_to_detected, below_to_detected + detected_to_below
        ),
    }


def pilot_reproducibility(
    pilot_scores_path: Path | None,
    parsed: dict[str, dict[str, dict[str, Any]]],
    pilot_export_root: Path | None = None,
    full_export_root: Path | None = None,
) -> dict[str, Any] | None:
    if pilot_scores_path is None or not pilot_scores_path.is_file():
        return None
    pilot_rows = [
        json.loads(line)
        for line in pilot_scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    differences = []
    mismatches = []
    content_comparisons = []

    def canonical_docx_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(archive.read(name))
                digest.update(b"\0")
        return digest.hexdigest()

    for row in pilot_rows:
        variant = row["variant"]
        if variant not in ("g1", "g2"):
            continue
        full_score = parsed[variant][row["file"]]["ck_ai_pct"]
        difference = full_score - int(row["ck_ai_pct"])
        differences.append(difference)
        if pilot_export_root is not None and full_export_root is not None:
            relative = Path(f"upload_{variant}") / row["batch"] / row["file"]
            pilot_path = pilot_export_root / relative
            full_path = full_export_root / relative
            if not pilot_path.is_file() or not full_path.is_file():
                raise FileNotFoundError(f"missing pilot comparison input: {relative}")
            content_comparisons.append(
                canonical_docx_digest(pilot_path) == canonical_docx_digest(full_path)
            )
        if difference != 0:
            mismatches.append(
                {
                    "variant": variant,
                    "file": row["file"],
                    "pair_id": row["pair_id"],
                    "pilot_ck_ai_pct": int(row["ck_ai_pct"]),
                    "full_ck_ai_pct": full_score,
                    "full_minus_pilot_pct": difference,
                }
            )
    result = {
        "compared_rows": len(differences),
        "exact_match_count": sum(value == 0 for value in differences),
        "mismatch_count": sum(value != 0 for value in differences),
        "max_absolute_difference_pct": max(map(abs, differences), default=0),
        "mean_difference_full_minus_pilot_pct": float(np.mean(differences)),
        "mismatches": mismatches,
    }
    if content_comparisons:
        result["canonical_docx_content_equal_count"] = sum(content_comparisons)
        result["canonical_docx_content_compared_count"] = len(content_comparisons)
    return result


def write_report(path: Path, result: dict[str, Any]) -> None:
    variants = result["variants"]
    pairwise = result["pairwise"]
    validity = result["detector_validity"]
    pilot = result.get("pilot_reproducibility")
    lines = [
        "# 팀 뉴스 Human/G0/G1/G2 카피킬러 전수 paired 결과",
        "",
        (
            f"> 동일 test ID {result['n_pairs']:,}개 · AI 판정 임계값 "
            f"AI작성률 ≥ {result['tau_ai_pct']}%"
        ),
        "",
        "## 탐지기 유효성과 핵심 결과",
        "",
        "| 변형 | 평균 AI작성률 | AI 판정률 | ASR | 0% 문서 | 최고 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, label in (
        ("human", "사람 원문"),
        ("g0", "원본 AI(G0)"),
        ("g1", "G1"),
        ("g2", "G2"),
    ):
        row = variants[variant]
        asr = "—" if variant == "human" else f"{row['asr_pct']:.2f}%"
        lines.append(
            f"| {label} | {row['mean_ai_pct']:.3f}% | "
            f"{row['detected_ai_rate_pct']:.2f}% ({row['detected_ai_count']}/{row['n']}) | "
            f"{asr} | {row['zero_rate_pct']:.2f}% | {row['max_ai_pct']}% |"
        )
    lines += [
        "",
        (
            f"사람 뉴스 오탐률은 **{validity['human_fpr_pct']:.2f}%**, 원본 AI(G0) "
            f"탐지율은 **{validity['g0_tpr_pct']:.2f}%**다. Human 대 G0 AUC는 "
            f"**{validity['auc_human_vs_g0']:.4f}**다."
        ),
        "",
        "| 분류축 | Human 대비 AUC | balanced accuracy (50% 임계값) |",
        "| --- | ---: | ---: |",
    ]
    for variant, label in (("g0", "G0"), ("g1", "G1"), ("g2", "G2")):
        lines.append(
            f"| {label} | {validity[f'auc_human_vs_{variant}']:.4f} | "
            f"{validity[f'balanced_accuracy_human_vs_{variant}_pct']:.2f}% |"
        )

    lines += [
        "",
        "## 생성 단계별 paired 변화",
        "",
        "| 비교 | 평균 점수 변화 | ASR 변화 | McNemar p | 낮아짐/높아짐/동률 | sign-test p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("g0_to_g1", "G0 → G1"),
        ("g1_to_g2", "G1 → G2"),
        ("g0_to_g2", "G0 → G2"),
    ):
        row = pairwise[key]
        lines.append(
            f"| {label} | {row['mean_score_delta_after_minus_before_pct']:+.3f}%p "
            f"(95% CI {format_ci(row['mean_score_delta_bootstrap_95ci_pct'])}) | "
            f"{row['asr_delta_after_minus_before_pct']:+.2f}%p | "
            f"{row['mcnemar_exact_two_sided_p']:.6g} | "
            f"{row['score_decreased_count']}/{row['score_increased_count']}/"
            f"{row['score_tied_count']} | {row['sign_test_two_sided_p']:.6g} |"
        )

    g1_to_g2 = pairwise["g1_to_g2"]
    lines += [
        "",
        (
            "G1→G2는 평균 점수와 50% 임계값 전이에서는 개선이 관찰됐지만, 점수가 바뀐 "
            f"문서만의 방향성 sign test는 p={g1_to_g2['sign_test_two_sided_p']:.4f}다. "
            "따라서 추가 효과는 작고 지표 선택에 따라 증거 강도가 다르다."
        ),
    ]

    lines += [
        "",
        "## 50% 임계값 전이",
        "",
    ]
    for key, label in (
        ("g0_to_g1", "G0 → G1"),
        ("g1_to_g2", "G1 → G2"),
        ("g0_to_g2", "G0 → G2"),
    ):
        transitions = pairwise[key]["threshold_transitions"]
        lines.append(
            f"- {label}: 미탐→미탐 {transitions['below_to_below']}, "
            f"미탐→탐지 {transitions['below_to_detected']}, "
            f"탐지→미탐 {transitions['detected_to_below']}, "
            f"탐지→탐지 {transitions['detected_to_detected']}"
        )

    if pilot is not None:
        lines += [
            "",
            "## 기존 350건 검사 재현성",
            "",
            (
                f"기존 G1/G2 파일럿과 겹치는 {pilot['compared_rows']}개 점수 중 "
                f"**{pilot['exact_match_count']}개가 정확히 일치**했고, "
                f"불일치는 {pilot['mismatch_count']}개다."
            ),
        ]
        if "canonical_docx_content_equal_count" in pilot:
            lines.append(
                f"비교 입력 {pilot['canonical_docx_content_compared_count']}개는 DOCX 내부 파일 내용이 "
                f"모두 동일했다({pilot['canonical_docx_content_equal_count']}/"
                f"{pilot['canonical_docx_content_compared_count']}). 따라서 불일치는 입력 텍스트 변경이 "
                "아니라 재검사 또는 검사 묶음 차이에 따른 점수 변동으로 해석해야 한다."
            )
        if pilot["mismatch_count"]:
            lines += [
                "",
                "| 변형 | 파일 | 파일럿 | 전수 검사 | 차이 |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
            for mismatch in pilot["mismatches"]:
                lines.append(
                    f"| {mismatch['variant'].upper()} | {mismatch['file']} | "
                    f"{mismatch['pilot_ck_ai_pct']}% | {mismatch['full_ck_ai_pct']}% | "
                    f"{mismatch['full_minus_pilot_pct']:+d}%p |"
                )

    lines += [
        "",
        "## 무결성",
        "",
    ]
    for variant in ("human", "g0", "g1", "g2"):
        report = result["reports"][variant]
        lines.append(
            f"- {variant}: 검사번호 {report['inspection_number']} · "
            f"{report['parsed_documents']:,}건 · {report['page_count']}페이지 · "
            f"SHA256 `{report['source_pdf_sha256']}`"
        )
    lines += [
        "- 네 PDF의 파일명 집합은 각 manifest와 정확히 일치한다.",
        "- 네 변형은 동일한 1,985개 pair_id와 동일 순서로 조인됐다.",
        "",
        "## 해석 경계",
        "",
        "- 카피킬러 한 탐지기와 현재 검사 시점에 대한 외부 전이 결과다.",
        "- 내용 보존·사실성·자연스러움과 새 D2 평가는 별도 증거다.",
        "- PDF 표시 평균은 정수 반올림값이므로 개별 문서 점수 평균을 사용했다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for variant in ("human", "g0", "g1", "g2"):
        parser.add_argument(f"--{variant}-pdf", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--generator-manifest", type=Path, required=True)
    parser.add_argument("--selection-metadata", type=Path, required=True)
    parser.add_argument("--pilot-scores", type=Path)
    parser.add_argument("--pilot-export-root", type=Path)
    parser.add_argument("--full-export-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tau", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if (args.pilot_export_root is None) != (args.full_export_root is None):
        parser.error("--pilot-export-root and --full-export-root must be provided together")
    if not 0 <= args.tau <= 100:
        parser.error("--tau must be between 0 and 100")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")

    pdf_paths = {variant: getattr(args, f"{variant}_pdf") for variant in VARIANT_PREFIX}
    manifests = []
    for path in (args.baseline_manifest, args.generator_manifest):
        with path.open(encoding="utf-8", newline="") as handle:
            manifests.extend(csv.DictReader(handle))
    by_variant = {
        variant: {row["file"]: row for row in manifests if row["variant"] == variant}
        for variant in VARIANT_PREFIX
    }
    if any(len(rows) != 1985 for rows in by_variant.values()):
        raise ValueError(
            "manifest counts must be 1,985 each: "
            + str({variant: len(rows) for variant, rows in by_variant.items()})
        )

    selection = json.loads(args.selection_metadata.read_text(encoding="utf-8"))[
        "selected_doc_ids"
    ]
    if len(selection) != 1985 or len(set(selection)) != 1985:
        raise ValueError("selection metadata must contain 1,985 unique IDs")

    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for variant, pdf_path in pdf_paths.items():
        parsed[variant], reports[variant] = extract_pdf(pdf_path)
        expected_files = set(by_variant[variant])
        actual_files = set(parsed[variant])
        if actual_files != expected_files:
            raise ValueError(
                f"{variant}: PDF/manifest mismatch; "
                f"missing={sorted(expected_files - actual_files)[:10]}, "
                f"extra={sorted(actual_files - expected_files)[:10]}"
            )
        if any(not filename.startswith(VARIANT_PREFIX[variant]) for filename in actual_files):
            raise ValueError(f"{variant}: unexpected filename prefix")
        reports[variant]["parsed_documents"] = len(actual_files)
        values = [row["ck_ai_pct"] for row in parsed[variant].values()]
        if reports[variant]["registered_documents"] != 1985:
            raise ValueError(f"{variant}: registered count is not 1,985")
        if reports[variant]["inspected_documents"] != 1985:
            raise ValueError(f"{variant}: inspected count is not 1,985")
        if reports[variant]["displayed_max_ai_pct"] != max(values):
            raise ValueError(f"{variant}: displayed maximum differs from rows")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant, source in pdf_paths.items():
        inspection = reports[variant]["inspection_number"].lstrip("0")
        destination = args.out_dir / "pdfs" / variant / f"{variant}_full_{inspection}.pdf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(source):
            raise ValueError(f"refusing to overwrite different preserved PDF: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        if sha256(destination) != reports[variant]["source_pdf_sha256"]:
            raise ValueError(f"{variant}: preserved PDF checksum mismatch")
        reports[variant]["preserved_pdf"] = str(destination.resolve())

    by_pair = {
        variant: {row["pair_id"]: row for row in rows.values()}
        for variant, rows in by_variant.items()
    }
    if any(set(rows) != set(selection) for rows in by_pair.values()):
        raise ValueError("variant pair IDs differ from frozen selection")

    paired_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for pair_id in selection:
        pair_row: dict[str, Any] = {"pair_id": pair_id}
        for variant in ("human", "g0", "g1", "g2"):
            manifest_row = by_pair[variant][pair_id]
            score_row = parsed[variant][manifest_row["file"]]
            pair_row[f"{variant}_file"] = manifest_row["file"]
            pair_row[f"{variant}_chars"] = int(manifest_row["chars"])
            pair_row[f"{variant}_ck_ai_pct"] = score_row["ck_ai_pct"]
            long_rows.append(
                {
                    **manifest_row,
                    **score_row,
                    "report_inspection_number": reports[variant]["inspection_number"],
                    "source_pdf_sha256": reports[variant]["source_pdf_sha256"],
                }
            )
        paired_rows.append(pair_row)

    arrays = {
        variant: np.asarray(
            [row[f"{variant}_ck_ai_pct"] for row in paired_rows], dtype=np.float64
        )
        for variant in VARIANT_PREFIX
    }
    summaries = {
        variant: summarize_scores(values, args.tau) for variant, values in arrays.items()
    }
    human_fpr = summaries["human"]["detected_ai_rate_pct"]
    detector_validity = {
        "human_fpr_pct": human_fpr,
        "g0_tpr_pct": summaries["g0"]["detected_ai_rate_pct"],
    }
    for variant in ("g0", "g1", "g2"):
        detector_validity[f"auc_human_vs_{variant}"] = auc_from_scores(
            arrays["human"], arrays[variant]
        )
        detector_validity[f"balanced_accuracy_human_vs_{variant}_pct"] = (
            summaries[variant]["detected_ai_rate_pct"] + (100 - human_fpr)
        ) / 2

    result = {
        "schema_version": 1,
        "n_pairs": len(paired_rows),
        "tau_ai_pct": args.tau,
        "reports": reports,
        "variants": summaries,
        "detector_validity": detector_validity,
        "pairwise": {
            "g0_to_g1": pairwise_stats(
                arrays["g0"], arrays["g1"], tau=args.tau, seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
            ),
            "g1_to_g2": pairwise_stats(
                arrays["g1"], arrays["g2"], tau=args.tau, seed=args.seed + 1,
                bootstrap_samples=args.bootstrap_samples,
            ),
            "g0_to_g2": pairwise_stats(
                arrays["g0"], arrays["g2"], tau=args.tau, seed=args.seed + 2,
                bootstrap_samples=args.bootstrap_samples,
            ),
        },
        "pilot_reproducibility": pilot_reproducibility(
            args.pilot_scores,
            parsed,
            args.pilot_export_root,
            args.full_export_root,
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
    }

    with (args.out_dir / "copykiller_scores_full.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in long_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (args.out_dir / "paired_scores_full.csv").open(
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
                "parsed": {variant: len(rows) for variant, rows in parsed.items()},
                "human_fpr": detector_validity["human_fpr_pct"],
                "g0_tpr": detector_validity["g0_tpr_pct"],
                "g1_asr": summaries["g1"]["asr_pct"],
                "g2_asr": summaries["g2"]["asr_pct"],
                "pilot_reproducibility": result["pilot_reproducibility"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
