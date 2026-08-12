#!/usr/bin/env python3
"""공공데이터포털 KCI Open API 로 한국어 논문 초록을 수집한다.

경로 확정 경위 (notes/43 · progress.md):
  · KCI 포털 자체 API 는 초록을 주지만 이 서버 IP 가 방화벽 차단돼 있어 못 쓴다.
  · data.go.kr 의 `3049042` 는 "링크형"이라 결국 kci.go.kr 로 넘어가 역시 못 쓴다.
  · `15085348` 만 apis.data.go.kr 에 직접 호스팅되고, 그 26개 상세기능 중
    **openApiM310List(KCI논문 정보 조회)** 가 전체 KCI 논문 242만 건 + KOR_ABST 를 준다.
    (openApiD267List 는 OA학술지 한정 + 초록이 문단 단위로 쪼개져 있어 부적합)

API 특성상 주의할 점:
  · 검색 조건 파라미터가 없다 — serviceKey/pageNo/recordCnt 뿐이라 전량 훑으며 걸러야 한다.
  · recordCnt 최대 100 (1000 을 주면 resultCode 30 으로 거절).
  · 레코드가 원문 등록일 오름차순이라 앞 페이지일수록 옛 논문이다 →
    ChatGPT 이전(2021년 이전) 글만 쓰는 우리에겐 유리하다. 뒤로 갈수록 버리는 비율이 는다.
  · 개발계정 트래픽 5,000회/일.

날짜: ORTE_RESI_DT(원문 등록일시)를 발행연도 대신 쓴다. 등록은 발행 이후이므로
"등록이 2021년 이전"이면 발행도 확실히 2021년 이전이다 — 보수적이라 안전하다.

사용:
  python3 collect_kci.py --out mash/kci_human_ko_v1.jsonl --target 8000
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

API = "https://apis.data.go.kr/B552540/KCIOpenApi/artiInfo/openApiM310List"
KEY_FILE = "/workspace/.API_KEY"
HANGUL = re.compile(r"[가-힣]")


def hangul_ratio(s):
    body = re.sub(r"\s", "", s)
    return len(HANGUL.findall(s)) / len(body) if body else 0.0


def fetch(key, page, cnt, retries=4):
    q = urllib.parse.urlencode({"serviceKey": key, "pageNo": page, "recordCnt": cnt})
    for i in range(retries):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                raw = r.read().decode("utf-8", "replace")
            root = ET.fromstring(raw)
            code = root.findtext(".//resultCode")
            if code != "00":
                # 30 = 파라미터 오류처럼 재시도해도 소용없는 것
                raise RuntimeError(f"resultCode={code} {root.findtext('.//resultMsg')}")
            return [{c.tag: (c.text or "") for c in item} for item in root.iter("item")]
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=8000, help="새로 모을 초록 수")
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--min-hangul", type=float, default=0.30)
    ap.add_argument("--max-year", type=int, default=2020, help="이 해까지만 (포함)")
    ap.add_argument("--journal-cap", type=int, default=100, help="학술지당 상한 (v0 과 동일)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="이미 가진 데이터 jsonl (kci_article_id 로 중복 제거)")
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--max-pages", type=int, default=4000)
    ap.add_argument("--sleep", type=float, default=0.15)
    a = ap.parse_args()

    key = Path(KEY_FILE).read_text().strip()

    seen = set()
    for p in a.exclude:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                v = r.get("kci_article_id") or r.get("source_document_id")
                if v:
                    seen.add(v)
    print(f"기존 보유 {len(seen)}편 — 중복 제거 대상", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 이어받기: 이미 쓴 것은 건너뛴다
    kept, jcount = [], Counter()
    if out.exists():
        for l in open(out, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                kept.append(r)
                seen.add(r["kci_article_id"])
                jcount[r["journal_id"]] += 1
        print(f"이어받기: 기존 출력 {len(kept)}편", flush=True)

    f = open(out, "a", encoding="utf-8")
    stats = Counter()
    page = a.start_page
    t0 = time.time()
    try:
        while len(kept) < a.target and page < a.start_page + a.max_pages:
            try:
                items = fetch(key, page, 100)
            except Exception as e:
                print(f"  페이지 {page} 실패: {e}", flush=True)
                page += 1
                continue
            if not items:
                print(f"페이지 {page}: 결과 없음 — 종료", flush=True)
                break
            for it in items:
                stats["총"] += 1
                aid = it.get("ARTI_ID", "").strip()
                abst = (it.get("KOR_ABST") or "").strip()
                dt = (it.get("ORTE_RESI_DT") or it.get("FIRS_ORTE_RESI_DT") or "").strip()
                jid = (it.get("SERE_ID") or "").strip()
                if not aid or aid in seen:
                    stats["중복/무효"] += 1
                    continue
                if not abst:
                    stats["한글초록 없음"] += 1
                    continue
                if not (a.min_chars <= len(abst) <= a.max_chars):
                    stats["길이 밖"] += 1
                    continue
                hr = hangul_ratio(abst)
                if hr < a.min_hangul:
                    stats["한글비 미달"] += 1
                    continue
                if not (dt[:4].isdigit() and int(dt[:4]) <= a.max_year):
                    stats["연도 밖"] += 1
                    continue
                if jcount[jid] >= a.journal_cap:
                    stats["학술지 상한"] += 1
                    continue
                rec = {
                    "kci_article_id": aid,
                    "original_title": (it.get("ARTI_KOR_TITL") or "").strip(),
                    "original_abstract": abst,
                    "keywords": (it.get("KOR_KEYW") or "").strip(),
                    "research_field_code": (it.get("STUD_FIEL_CD") or "").strip(),
                    "journal_id": jid,
                    "registered": dt[:8],
                    "doi": (it.get("DOI") or "").strip(),
                    "abstract_hangul_ratio": round(hr, 4),
                    "abstract_chars": len(abst),
                    "source": "data.go.kr/15085348/openApiM310List",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept.append(rec)
                seen.add(aid)
                jcount[jid] += 1
                stats["채택"] += 1
                if len(kept) >= a.target:
                    break
            if page % 25 == 0:
                f.flush()
                el = time.time() - t0
                print(f"페이지 {page} · 채택 {len(kept)}/{a.target} "
                      f"· 학술지 {len(jcount)}종 · {el/60:.1f}분", flush=True)
            page += 1
            time.sleep(a.sleep)
    finally:
        f.close()

    print(f"\n=== 수집 종료 · 마지막 페이지 {page} ===")
    for k, v in stats.most_common():
        print(f"  {k:<14}{v:>8,}")
    print(f"\n채택 {len(kept):,}편 → {out}")
    if kept:
        yrs = Counter(r["registered"][:4] for r in kept)
        print(f"  등록연도: {min(yrs)}~{max(yrs)} · 학술지 {len(jcount)}종")
        ln = sorted(r["abstract_chars"] for r in kept)
        print(f"  초록 길이 중앙값 {ln[len(ln)//2]}자")


if __name__ == "__main__":
    main()
