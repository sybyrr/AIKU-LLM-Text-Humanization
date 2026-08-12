#!/usr/bin/env python3
"""평가셋용 — OA학술지 논문의 **본문 + 한글초록**을 짝지어 수집한다.

왜 본문이 필요한가:
  평가용 AI 초록을 "인간 초록의 재서술"로 만들면 학습쌍 $x_{ai}$ 와 구성 방식이 같아져,
  humanizer 가 학습한 바로 그 분포를 평가하게 된다(ASR 과대평가).
  **본문을 주고 초록을 쓰게 하면** 과제 자체가 달라져 이 문제가 사라지고,
  연구자가 실제로 LLM 을 쓰는 방식과도 가장 가깝다.

⚠️ 페이지네이션 한계와 우회:
  D267/D268 은 totalCount 가 9,957 / 291,500 이라도 **offset 5,000 / 4,000 을 넘기면 빈 응답**을 준다.
  전량 훑기로는 본문 보유 논문을 36편밖에 못 얻는다(실측).
  대신 **`artiId` 요청변수**가 동작한다 — 논문 ID 를 주면 그 논문의 전 문단을 한 번에 준다
  (실측 158문단 54,971자). 그래서 ① 순차 훑기로 초록 보유 논문 ID 목록을 얻고
  ② 각 ID 로 본문을 직접 조회한다. 주의: 파라미터명이 `ARTIID` 면 조용히 무시되고
  기본 레코드가 돌아온다 — 반드시 `artiId` 로 써야 한다.

두 오퍼레이션을 조인한다 (둘 다 OA학술지 한정):
  · openApiD267List  OA학술지-초록 관리 — USELANG=kor 인 PARA 를 ABSSEQ 순으로 이어붙인다
  · openApiD268List  OA학술지-본문 관리 — TYPE=B101(본문 문단)만 BODYSEQ 순으로 이어붙인다
    (B2xx/B3xx 는 장·절 제목이라 제외. HTML 엔티티가 이중 이스케이프돼 있어 두 번 푼다.)

본문 발췌:
  본문 중앙값이 2만 자를 넘어 슬롯 컨텍스트(4096토큰 ≈ 6~8천 자)에 안 들어간다.
  앞부분만 자르면 서론만 남아 결과·결론이 빠지므로, **앞 절반 + 뒤 절반**을 떼어 붙인다.
  초록이 담아야 할 목적(앞)과 결과·결론(뒤)이 둘 다 들어가게 하려는 것이다.

사용:
  python3 collect_kci_bodies.py --out mash/kci_eval_bodies.jsonl --target 300
"""
import argparse
import collections
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "https://apis.data.go.kr/B552540/KCIOpenApi/artiInfo"
KEY_FILE = "/workspace/.API_KEY"
HANGUL = re.compile(r"[가-힣]")


def unesc(s):
    # API 응답의 엔티티가 이중 이스케이프돼 있다 (&amp;#8544; → &#8544; → Ⅰ)
    return html.unescape(html.unescape(s or ""))


def fetch(op, key, page, cnt, arti_id=None, retries=4):
    params = {"serviceKey": key, "pageNo": page, "recordCnt": cnt}
    if arti_id:
        params["artiId"] = arti_id          # 대소문자 주의: ARTIID 는 무시된다
    q = urllib.parse.urlencode(params)
    for i in range(retries):
        try:
            with urllib.request.urlopen(f"{BASE}/{op}?{q}", timeout=90) as r:
                raw = r.read().decode("utf-8", "replace").strip()
            # 빈 응답은 "그 논문에 본문이 없다" 또는 "offset 한계 초과"라는 **정상 결과**다.
            # 이걸 예외로 보고 재시도하면 건당 12초씩 버린다(실측: 47건에 9분).
            if not raw:
                return []
            root = ET.fromstring(raw)
            if root.findtext(".//resultCode") != "00":
                raise RuntimeError(root.findtext(".//resultMsg"))
            return [{c.tag: unesc(c.text) for c in it} for it in root.iter("item")]
        except ET.ParseError:
            return []          # 파싱 불가도 같은 취급 — 재시도해도 같은 응답이 온다
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    return []


def harvest(op, key, pages, cnt, keep, label):
    """pages 만큼 훑으며 keep(item) 이 참인 레코드를 ARTIID 별로 모은다."""
    out = collections.defaultdict(list)
    for p in range(1, pages + 1):
        try:
            items = fetch(op, key, p, cnt)
        except Exception as e:
            print(f"  {label} p{p} 실패: {e}", file=sys.stderr)
            continue
        if not items:
            break
        for it in items:
            if keep(it):
                out[it.get("ARTIID", "")].append(it)
        if p % 20 == 0:
            print(f"  {label}: {p}페이지 · 논문 {len(out)}편", flush=True)
    return out


def joined(parts, seq_key):
    return "\n".join(p.get("PARA", "") for p in sorted(parts, key=lambda x: x.get(seq_key, "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--abst-pages", type=int, default=12)
    ap.add_argument("--body-pages", type=int, default=120)
    ap.add_argument("--min-abst", type=int, default=300)
    ap.add_argument("--max-abst", type=int, default=1200)
    ap.add_argument("--min-body", type=int, default=3000)
    ap.add_argument("--excerpt", type=int, default=5000, help="본문 발췌 총 길이(앞+뒤)")
    a = ap.parse_args()

    key = Path(KEY_FILE).read_text().strip()

    print("한글초록 수집…", flush=True)
    absts = harvest("openApiD267List", key, a.abst_pages, 1000,
                    lambda it: it.get("USELANG") == "kor", "초록")
    print(f"  → 초록 보유 논문 {len(absts):,}편", flush=True)

    # 본문은 순차 훑기로는 offset 4,000 에서 막힌다 → 초록 보유 논문 ID 로 하나씩 직접 조회
    print(f"본문 조회 (논문 {len(absts):,}편 대상, artiId 직접 질의)…", flush=True)
    bodies, tried = {}, 0
    for aid in sorted(absts):
        if len(bodies) >= a.target * 2:      # 여유분까지만 — 질의 한도(5,000/일) 절약
            break
        tried += 1
        try:
            items = fetch("openApiD268List", key, 1, 1000, arti_id=aid)
        except Exception as e:
            print(f"  {aid} 실패: {e}", file=sys.stderr)
            continue
        paras = [it for it in items if it.get("TYPE") == "B101"]
        if paras:
            bodies[aid] = paras
        if tried % 100 == 0:
            print(f"  {tried}건 질의 · 본문 확보 {len(bodies)}편", flush=True)
        time.sleep(0.1)
    print(f"  → 본문 보유 논문 {len(bodies):,}편 (질의 {tried:,}회)", flush=True)

    rows, stats = [], collections.Counter()
    for aid in sorted(set(absts) & set(bodies)):
        stats["교집합"] += 1
        abst = joined(absts[aid], "ABSSEQ").strip()
        body = joined(bodies[aid], "BODYSEQ").strip()
        if not (a.min_abst <= len(abst) <= a.max_abst):
            stats["초록 길이 밖"] += 1
            continue
        if len(body) < a.min_body:
            stats["본문 짧음"] += 1
            continue
        b = re.sub(r"\s", "", abst)
        hr = len(HANGUL.findall(abst)) / len(b) if b else 0
        if hr < 0.30:
            stats["한글비 미달"] += 1
            continue
        # 앞 절반 + 뒤 절반 발췌 — 목적(앞)과 결과·결론(뒤)을 모두 담는다
        h = a.excerpt // 2
        excerpt = body if len(body) <= a.excerpt else body[:h] + "\n(…중략…)\n" + body[-h:]
        rows.append({
            "kci_article_id": aid,
            "original_abstract": abst,
            "body_excerpt": excerpt,
            "body_chars": len(body),
            "abstract_chars": len(abst),
            "abstract_hangul_ratio": round(hr, 4),
            "source": "data.go.kr/15085348/openApiD267List+D268List",
        })
        stats["채택"] += 1
        if len(rows) >= a.target:
            break

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n" + " · ".join(f"{k} {v:,}" for k, v in stats.most_common()))
    print(f"{len(rows):,}편 → {out}")
    if rows:
        bl = sorted(r["body_chars"] for r in rows)
        al = sorted(r["abstract_chars"] for r in rows)
        print(f"  본문 중앙값 {bl[len(bl)//2]:,}자 (발췌 후 최대 {a.excerpt:,}자)"
              f" · 초록 중앙값 {al[len(al)//2]}자")


if __name__ == "__main__":
    main()
