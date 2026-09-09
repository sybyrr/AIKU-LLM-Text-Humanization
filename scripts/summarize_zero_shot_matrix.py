#!/usr/bin/env python3
"""Binoculars·FastDetectGPT 점수를 6×6 전량과 프롬프트 기준선으로 집계합니다.

출력 JSON과 HTML에는 집계값만 기록하며 원문, 생성문, 문서 ID는 기록하지 않습니다.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean


DEFAULT_DOMAINS = ("news", "essay", "persona", "written", "petition512B", "wiki512B")
DETECTORS = ("binoculars", "fastdetectgpt")
BASELINE_METHODS = ("x_ai", "SFT", "DPO", "Humanizer-skill")

CSS = """
:root{--bg:#f3f6fb;--paper:#fff;--ink:#162033;--muted:#647087;--line:#dbe2ee;--blue:#245dcc;--navy:#10274d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Pretendard,"Noto Sans KR",system-ui,sans-serif;line-height:1.55}main{width:min(1400px,calc(100% - 32px));margin:28px auto 72px}header{padding:30px 34px;border-radius:18px;color:#fff;background:linear-gradient(125deg,var(--navy),var(--blue))}h1{margin:0 0 8px;font-size:clamp(26px,4vw,38px)}header p{margin:0;color:#dce8ff}.meta{margin-top:14px;font-size:13px;color:#eaf1ff}section{padding:22px;margin-top:16px;background:var(--paper);border:1px solid var(--line);border-radius:15px}h2{margin:0 0 5px;font-size:21px}.lead{margin:0 0 14px;color:var(--muted)}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;background:#fff;font-size:12px}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead th{background:#f7f9fc;color:#4f5a70}.matrix td{text-align:center;min-width:125px}.matrix td b{display:block;font-size:16px}.matrix td small{display:block;color:var(--muted)}.diag{outline:2px solid var(--blue);outline-offset:-2px}footer{text-align:center;color:var(--muted);font-size:12px;margin-top:18px}@media(max-width:800px){main{width:min(100% - 18px,1400px)}}
"""


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("백분위를 계산할 유효 점수가 없습니다.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def load_scores(work_dir: Path) -> dict[str, dict[str, float | None]]:
    inputs = work_dir / "inputs"
    scores_dir = work_dir / "scores"
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    result: dict[str, dict[str, float | None]] = {name: {} for name in DETECTORS}
    for detector in DETECTORS:
        for shard in range(manifest["shards"]):
            path = scores_dir / f"{detector}_shard_{shard}.jsonl"
            done = scores_dir / f"{detector}_shard_{shard}.done"
            if not done.exists():
                raise RuntimeError(f"채점 미완료: {detector} shard {shard}")
            for row in read_jsonl(path):
                result[detector][str(row["key"])] = row.get("score")
        if len(result[detector]) != manifest["unique_texts"]:
            raise RuntimeError(
                f"{detector} 점수 수 {len(result[detector])} != {manifest['unique_texts']}"
            )
    return result


def group(path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        result[str(row["dec"])].append(row)
    return dict(result)


def values(rows: list[dict], scores: dict[str, float | None]) -> list[float]:
    found = []
    for row in rows:
        score = scores.get(text_key(row["text"]))
        if score is not None:
            found.append(float(score))
    return found


def result(values_: list[float], threshold: float, expected: int) -> dict:
    if not values_:
        return {"n": 0, "expected": expected, "coverage_pct": 0.0,
                "asr": None, "tpr": None, "mean_score": None}
    return {
        "n": len(values_), "expected": expected,
        "coverage_pct": 100 * len(values_) / expected,
        "asr": 100 * fmean(value <= threshold for value in values_),
        "tpr": 100 * fmean(value > threshold for value in values_),
        "mean_score": fmean(values_),
    }


def calibration(rows: dict[str, list[dict]], scores: dict[str, float | None]) -> dict:
    human = values(rows["human"], scores)
    x_ai = values(rows["x_ai"], scores)
    threshold = percentile(human, 0.95)
    return {
        "threshold": threshold,
        "human_n": len(human), "x_ai_n": len(x_ai),
        "human_mean": fmean(human), "x_ai_mean": fmean(x_ai),
        "gap": fmean(x_ai) - fmean(human),
    }


def aggregate(cross_dir: Path, skill_dir: Path, reference_dir: Path,
              domains: tuple[str, ...], score_maps: dict) -> dict:
    output = {"domains": list(domains), "cross_domain_full": {}, "humanizer_n100": {}}
    calibrations: dict[str, dict] = {}
    for target in domains:
        reference = group(cross_dir / f"{target}_to_{target}.jsonl")
        calibrations[target] = {
            detector: calibration(reference, score_maps[detector])
            for detector in DETECTORS
        }

    for source in domains:
        for target in domains:
            rows = group(cross_dir / f"{source}_to_{target}.jsonl")
            cell = {"source": source, "target": target,
                    "calibration": calibrations[target], "methods": {}}
            for detector in DETECTORS:
                threshold = calibrations[target][detector]["threshold"]
                cell["methods"][detector] = {
                    method: result(values(rows[method], score_maps[detector]), threshold,
                                   len(rows[method]))
                    for method in ("human", "x_ai", "SFT", "DPO")
                }
            output["cross_domain_full"][f"{source}_to_{target}"] = cell

    for domain in domains:
        external = group(skill_dir / f"{domain}.jsonl")
        reference = group(reference_dir / f"{domain}_to_{domain}.jsonl")
        ids = {str(row["doc_id"]) for row in external["human"]}
        rows = {
            "human": external["human"],
            "x_ai": external["x_ai"],
            "SFT": [row for row in reference["SFT"] if str(row["doc_id"]) in ids],
            "DPO": [row for row in reference["DPO"] if str(row["doc_id"]) in ids],
            "Humanizer-skill": external["Humanizer-skill"],
        }
        domain_result = {"calibration": {}, "methods": {}}
        for detector in DETECTORS:
            cal = calibration(external, score_maps[detector])
            domain_result["calibration"][detector] = cal
            domain_result["methods"][detector] = {
                method: result(values(rows[method], score_maps[detector]), cal["threshold"],
                               len(rows[method]))
                for method in BASELINE_METHODS
            }
        output["humanizer_n100"][domain] = domain_result
    return output


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def page(title: str, subtitle: str, body: str) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>"
        f"<header><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>"
        f'<div class="meta">생성 {generated} · FPR 5% 기준</div></header>{body}'
        "<footer>집계 결과만 포함하며 샘플 본문과 문서 ID는 포함하지 않았습니다.</footer>"
        "</main></body></html>"
    )


def matrix_section(data: dict, domains: tuple[str, ...], detector: str) -> str:
    label = "Binoculars" if detector == "binoculars" else "FastDetectGPT"
    heads = "".join(f"<th>{html.escape(domain)}</th>" for domain in domains)
    rows = []
    for source in domains:
        cells = []
        for target in domains:
            methods = data[f"{source}_to_{target}"]["methods"][detector]
            css_class = "diag" if source == target else ""
            cells.append(
                f'<td class="{css_class}"><b>{pct(methods["DPO"]["asr"])}</b>'
                f'<small>SFT {pct(methods["SFT"]["asr"])} · '
                f'N {methods["DPO"]["n"]:,}/{methods["DPO"]["expected"]:,}</small></td>'
            )
        rows.append(f"<tr><th>{html.escape(source)}</th>{''.join(cells)}</tr>")
    return (
        f"<section><h2>{label} ASR</h2>"
        '<p class="lead">큰 숫자는 DPO, 작은 숫자는 SFT입니다. 행은 학습 source, 열은 target입니다.</p>'
        f'<div class="table-wrap"><table class="matrix"><thead><tr><th>source ＼ target</th>{heads}'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def render_cross(data: dict, domains: tuple[str, ...], output: Path) -> None:
    body = "".join(matrix_section(data, domains, detector) for detector in DETECTORS)
    output.write_text(page(
        "6×6 크로스도메인 전량 zero-shot 평가",
        "Binoculars와 FastDetectGPT의 target별 인간 95백분위 임계값 기준 결과입니다.",
        body,
    ), encoding="utf-8")


def render_baseline(data: dict, domains: tuple[str, ...], output: Path) -> None:
    sections = []
    for detector in DETECTORS:
        label = "Binoculars" if detector == "binoculars" else "FastDetectGPT"
        rows = []
        for domain in domains:
            for method in BASELINE_METHODS:
                item = data[domain]["methods"][detector][method]
                mean_score = (
                    "—" if item["mean_score"] is None else f'{item["mean_score"]:.4f}'
                )
                rows.append(
                    f"<tr><td>{html.escape(domain)}</td><td>{html.escape(method)}</td>"
                    f"<td>{item['n']:,}/{item['expected']:,}</td><td>{pct(item['asr'])}</td>"
                    f"<td>{mean_score}</td></tr>"
                )
        sections.append(
            f"<section><h2>{label}</h2>"
            '<p class="lead">같은 held-out 문서에서 x_ai, SFT, DPO, 프롬프트 기준선을 비교합니다.</p>'
            '<div class="table-wrap"><table><thead><tr><th>도메인</th><th>방법</th>'
            f"<th>N</th><th>ASR</th><th>평균 점수</th></tr></thead><tbody>{''.join(rows)}"
            "</tbody></table></div></section>"
        )
    output.write_text(page(
        "Humanizer-skill 기준선 zero-shot 평가",
        "도메인별 100편에서 기존 학습 모델과 프롬프트 기반 재작성을 비교한 결과입니다.",
        "".join(sections),
    ), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cross-dir", type=Path, required=True)
    parser.add_argument("--skill-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    domains = tuple(args.domains)
    score_maps = load_scores(args.work_dir)
    summary = aggregate(
        args.cross_dir, args.skill_dir, args.reference_dir, domains, score_maps
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "zero_shot_aggregate.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    render_cross(
        summary["cross_domain_full"], domains,
        args.output_dir / "cross_domain_full_zeroshot.html",
    )
    render_baseline(
        summary["humanizer_n100"], domains,
        args.output_dir / "humanizer_skill_baseline_n100_zeroshot.html",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
