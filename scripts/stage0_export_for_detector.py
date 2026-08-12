#!/usr/bin/env python3
"""외부 탐지기(카피킬러 등) 수동 업로드용 파일 내보내기.

카피킬러는 공개 API가 없어 웹 UI 다중 업로드가 유일한 경로다.
한 검사 목록에 최대 350개 · 총 200MB 까지 올라간다(공식 매뉴얼).

⚠ 확장자: 카피킬러 업로드 대화상자가 받는 것은
     hwp, hwpx, doc, docx, ppt, pptx, xls, xlsx, pdf
   **txt 는 받지 않는다**(실측 확인). 그래서 기본 출력은 docx 다.
   매뉴얼도 출처 인식 때문에 hwpx/docx 를 권장하고, PDF 는 각주 인식 문제가 있다.

주의:
  · 파일명에 특수기호가 있으면 검사가 진행되지 않는다 → 영숫자·언더바만 쓴다
  · 파일명이 결과확인서에 그대로 출력된다 → 라벨(human/ai)을 파일명에 넣으면
    채점자가 정답을 보게 된다. 그래서 파일명은 익명 일련번호로 하고,
    매핑은 manifest.csv 로만 보관한다.
  · docx 문서 속성(author/title)에도 라벨이 새지 않도록 비워 둔다.

사용:
  python3 stage0_export_for_detector.py --out pilot/export/batch1 \
      --human pilot/data/seed_abstract.jsonl --gen pilot/gen/abstract_qwen3-8b.jsonl
"""
import argparse
import collections
import csv
import json
import random
from pathlib import Path

MAX_FILES = 350  # 카피킬러 한 배치 한도


def write_docx(path, text):
    """문단 구조를 유지한 docx 를 쓴다. 문서 속성은 비워 라벨 유출을 막는다."""
    from docx import Document

    doc = Document()
    # strip 하지 않는다 — 띄어쓰기는 이 연구에서 탐지 신호로 쓰이는 요소다
    # (KatFishNet의 핵심 피처 중 하나). 원문을 글자 단위로 보존한다.
    # 다만 줄바꿈 표현만은 통일한다: 원본에 섞여 있는 CRLF 를 docx 가 LF 로 바꿔 저장하므로,
    # 정규화하지 않으면 "원본과 docx 가 다르다"는 착시가 생긴다(본문은 동일).
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for para in text.split("\n"):
        doc.add_paragraph(para)
    cp = doc.core_properties
    cp.author = ""
    cp.title = ""
    cp.comments = ""
    cp.last_modified_by = ""
    doc.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--gen", action="append", default=[], help="여러 번 지정 가능")
    ap.add_argument("--prefix", default="D")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--format", default="docx", choices=["docx", "txt"],
                    help="카피킬러는 txt 를 받지 않는다 — 기본 docx")
    ap.add_argument("--batch-size", type=int, default=MAX_FILES,
                    help="배치 폴더 하나에 담을 파일 수 (카피킬러 한도 350)")
    ap.add_argument("--exclude-checked", default="",
                    help="이미 검사한 점수 jsonl — 그 pair_id 의 인간 원문은 제외")
    ap.add_argument("--human-field", default="human_text")
    ap.add_argument("--id-field", default="pair_id")
    a = ap.parse_args()

    # 이미 검사가 끝난 인간 원문은 다시 올리지 않는다 (카피킬러 검사 건수 절약)
    checked = set()
    if a.exclude_checked and Path(a.exclude_checked).exists():
        for l in open(a.exclude_checked, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("label") == "human":
                    checked.add(r["pair_id"])
        print(f"기검사 인간 원문 {len(checked)}건 제외")

    items = []
    for l in open(a.human, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r[a.id_field] in checked:
                continue
            items.append(
                {
                    "pair_id": r[a.id_field],
                    "variant": "human",
                    "model": "human",
                    "label": "human",
                    "text": r[a.human_field],
                }
            )
    for path in a.gen:
        for l in open(path, encoding="utf-8"):
            if not l.strip():
                continue
            g = json.loads(l)
            if not g.get("text"):
                continue
            items.append(
                {
                    "pair_id": g["doc_id"],
                    "variant": g["cond"],
                    "model": g["model"],
                    "label": "ai",
                    "text": g["text"],
                }
            )

    # 업로드 순서로 정답이 드러나지 않도록 섞는다
    random.Random(a.seed).shuffle(items)

    # 같은 논문의 인간 원문과 AI 재서술은 **다른 배치**로 보낸다.
    # 카피킬러는 본래 표절 검사기라 한 검사 목록 안의 문서끼리 대조한다.
    # AI 재서술은 원문과 내용이 같으므로, 둘이 같은 배치에 있으면
    # 표절 쪽 판정이 AI작성률에 영향을 줄 여지가 있다(확인된 바는 없으나 피할 수 있는 위험).
    # 인간을 앞쪽 배치에, AI 를 뒤쪽 배치에 몰아 배치 경계를 라벨로 나눈다.
    items.sort(key=lambda x: 0 if x["label"] == "human" else 1)

    # 업로드 대상은 docs/ 안에만 둔다. manifest(정답표)는 그 바깥에 둬서
    # 파일 전체 선택으로 업로드해도 정답표가 딸려 올라가지 않게 한다.
    # 배치 폴더로 나눈다. 카피킬러는 한 검사 목록에 350개까지만 받는다.
    # 폴더 하나가 곧 업로드 1회분이므로, 폴더째 전체 선택해 올리면 된다.
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nb = -(-len(items) // a.batch_size)
    width = max(4, len(str(len(items))))
    rows = []
    for i, it in enumerate(items):
        b = i // a.batch_size + 1
        docs = out / f"batch{b:02d}"
        docs.mkdir(parents=True, exist_ok=True)
        name = f"{a.prefix}{i+1:0{width}d}.{a.format}"
        if a.format == "docx":
            write_docx(docs / name, it["text"])
        else:
            (docs / name).write_text(it["text"], encoding="utf-8")
        rows.append({"file": name, "batch": f"batch{b:02d}",
                     **{k: v for k, v in it.items() if k != "text"},
                     "chars": len(it["text"])})

    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "batch", "pair_id", "variant", "model", "label", "chars"])
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    total_mb = sum((out / r["batch"] / r["file"]).stat().st_size for r in rows) / 1e6
    print(f"\n{n:,}개 파일 · {total_mb:.1f}MB → {out}/batch01 … batch{nb:02d}")
    print(f"  배치 {nb}개 · 폴더당 최대 {a.batch_size}개 (마지막 {n - (nb-1)*a.batch_size}개)")
    print(f"  라벨 구성: " + " · ".join(
        f"{k} {v:,}" for k, v in collections.Counter(r["label"] for r in rows).most_common()))
    print(f"정답표: {out}/manifest.csv  (**업로드 금지** — 배치 폴더 바깥에 있음)")
    print(f"업로드: 폴더 하나가 검사 1회분이다. batch01 부터 순서대로 올리면 된다.")


if __name__ == "__main__":
    main()
