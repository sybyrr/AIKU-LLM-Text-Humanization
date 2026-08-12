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
    a = ap.parse_args()

    items = []
    for l in open(a.human, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            items.append(
                {
                    "pair_id": r["pair_id"],
                    "variant": "human",
                    "model": "human",
                    "label": "human",
                    "text": r["human_text"],
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

    # 업로드 대상은 docs/ 안에만 둔다. manifest(정답표)는 그 바깥에 둬서
    # 파일 전체 선택으로 업로드해도 정답표가 딸려 올라가지 않게 한다.
    out = Path(a.out)
    docs = out / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, it in enumerate(items, 1):
        name = f"{a.prefix}{i:04d}.{a.format}"
        if a.format == "docx":
            write_docx(docs / name, it["text"])
        else:
            (docs / name).write_text(it["text"], encoding="utf-8")
        rows.append({"file": name, **{k: v for k, v in it.items() if k != "text"},
                     "chars": len(it["text"])})

    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "pair_id", "variant", "model", "label", "chars"])
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    total_mb = sum((docs / r["file"]).stat().st_size for r in rows) / 1e6
    print(f"{n}개 파일 · {total_mb:.2f}MB → {docs}/  (업로드 대상)")
    print(f"정답표: {out}/manifest.csv  (업로드하지 말 것 — docs/ 바깥에 둠)")
    if n > MAX_FILES:
        print(f"⚠ 카피킬러 한 배치 한도(350개)를 넘는다 — {-(-n//MAX_FILES)}회로 나눠 올릴 것")
    else:
        print(f"카피킬러 한 배치(350개 · 200MB)에 들어간다 — 1회 업로드로 충분")


if __name__ == "__main__":
    main()
