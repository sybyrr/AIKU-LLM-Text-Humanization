#!/usr/bin/env python3
"""MASH 1단계용 인간 원문 풀 추출 — 국립국어원 신문 말뭉치 2022판(2021년 기사).

러셀식 `extract_russell_pool.sql` 과 목적이 다르다. 저쪽은 제목만 주고 기사를 새로
쓰게 할 원문을 뽑았고, 이쪽은 **polish 해서 x_ai 를 만들 원문**을 뽑는다.
그래서 길이 조건과 인용 조건이 다르다.

필터 근거는 notes 에 정리돼 있고 요약하면:
  * 본문 800~1200자 — ko-BART 인코더 1024토큰의 절반(실측 최대 561토큰). x_ai 가
    원문보다 길어져도 상한 안에 들어온다. 자/토큰 최악 1.80 기준으로도 안전하다.
  * n_para >= 6 — 기사 구조가 있어야 한다. para_idx=1 은 제목이라 본문 문단은 5개 이상.
  * 인용 비율 <= 40% — 본문 800편 실측에서 p90 이 38.4%. 그보다 인용이 많은 글은
    사실상 인터뷰 전사라 polish 할 산문이 얼마 없다.
  * topic 5종 — 러셀 풀과 같은 정치/경제/사회/IT·과학/문화. 스포츠·연예·생활·미용건강 제외.

표본은 (topic, publisher) 층으로 나눠 라운드로빈으로 고른다. 시드를 쓰지 않고
파일 순서대로 결정론적으로 뽑으므로 몇 번을 돌려도 같은 결과가 나온다.

사용:
    python mash_extract_pool.py --n 10  --out ../dataset/human/mash_pilot_10.jsonl
    python mash_extract_pool.py --n 3100 --out ../dataset/human/mash_pool.jsonl
"""
import argparse, json, pathlib, re, sys, zipfile
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor

ZIP = "/shared/corpus/raw/NIKLNEWSPAPER_2022_v1.0_JSON.zip"
TOPICS = ["정치", "경제", "사회", "IT/과학", "문화"]

# 직접 인용. 여는 따옴표와 닫는 따옴표가 짝을 이루는 구간만 센다.
QUOTE = re.compile(r"[“‘\"]([^”’\"]{5,})[”’\"]")

MIN_CHAR, MAX_CHAR = 800, 1200
MIN_PARA = 6            # 제목 포함. 본문 문단 5개 이상
MAX_QUOTE_RATIO = 0.40

# (topic, publisher) 층마다 후보를 몇 편까지 들고 있을지. 층이 170개쯤 나오므로
# 이 값이 곧 표본 상한(층수 × 이 값)이 된다. --n 에 맞춰 자동으로 잡되 --per-stratum
# 으로 덮어쓸 수 있다. 전수를 다 들고 있으면 메모리만 먹고 쓸모가 없다.
PER_STRATUM = 3


def quote_ratio(body):
    qs = QUOTE.findall(body)
    return sum(len(q) for q in qs) / len(body) if body else 0.0


def scan(task):
    """json 파일 하나에서 조건을 통과한 기사를 (topic, publisher) 층별로 최대 per_stratum 개.

    per_stratum 을 전역으로 두면 워커 프로세스에 전달되는지가 start method 에 따라
    달라지므로 인자로 함께 넘긴다.
    """
    name, per_stratum = task
    doc = json.loads(zipfile.ZipFile(ZIP).read(name))
    out = defaultdict(list)
    for d in doc["document"]:
        paras = d["paragraph"]
        if len(paras) < MIN_PARA:
            continue
        meta = d["metadata"]
        topic = meta.get("topic")
        if topic not in TOPICS:
            continue
        body = "\n\n".join(p["form"] for p in paras[1:])
        if not (MIN_CHAR <= len(body) <= MAX_CHAR):
            continue
        qr = quote_ratio(body)
        if qr > MAX_QUOTE_RATIO:
            continue
        key = (topic, meta.get("publisher", ""))
        if len(out[key]) >= per_stratum:
            continue
        out[key].append({
            "doc_id": d["id"],
            "file_id": doc["id"],
            "publisher": meta.get("publisher", ""),
            "author": meta.get("author", ""),
            "date": meta.get("date", ""),
            "topic": topic,
            "original_topic": meta.get("original_topic", ""),
            "title": paras[0]["form"],          # para_idx=1 은 제목이다(본문 아님)
            "body": body,                       # para_idx>=2 를 개행 2개로 join
            "n_char": len(body),
            "n_para": len(paras) - 1,           # 본문 문단 수
            "quote_ratio": round(qr, 4),
        })
    return {k: v for k, v in out.items()}


def pick(pool, n):
    """topic 은 고르게, 같은 topic 안에서는 매체를 회전시키며 뽑는다.

    한 바퀴에 topic 당 1편씩 담고, 매체 커서와 층 내부 인덱스를 따로 둔다.
    (둘을 같은 변수로 쓰면 표본이 '층당 후보 수 × topic 수' 에서 막힌다.)
    """
    by_topic = {t: OrderedDict() for t in TOPICS}
    for (topic, pub), items in sorted(pool.items()):
        by_topic[topic][pub] = items

    cursor = {t: 0 for t in TOPICS}     # 매체 회전 위치
    used = defaultdict(int)             # (topic, pub) 에서 몇 편 썼는지
    pub_use = defaultdict(int)          # 매체별 전체 사용 횟수
    chosen = []
    while len(chosen) < n:
        added = 0
        for topic in TOPICS:
            if len(chosen) >= n:
                break
            pubs = list(by_topic[topic].keys())
            if not pubs:
                continue
            # 회전 순서를 유지하되 전체에서 덜 쓴 매체를 먼저 본다. 커서만 쓰면
            # topic 마다 0번 매체부터 시작해 표본이 앞쪽 매체 몇 곳에 쏠린다.
            order = sorted(range(len(pubs)),
                           key=lambda s: (pub_use[pubs[(cursor[topic] + s) % len(pubs)]], s))
            for step in order:
                pub = pubs[(cursor[topic] + step) % len(pubs)]
                k = used[(topic, pub)]
                if k < len(by_topic[topic][pub]):
                    chosen.append(by_topic[topic][pub][k])
                    used[(topic, pub)] = k + 1
                    pub_use[pub] += 1
                    cursor[topic] = (cursor[topic] + step + 1) % len(pubs)
                    added += 1
                    break
        if added == 0:
            break
    return chosen[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-stratum", type=int, default=0,
                    help="0 이면 --n 에 맞춰 자동 (층 170개 기준 n//20, 최소 3)")
    ap.add_argument("--exclude", default=None,
                    help="이 JSON(list of doc_id) 에 있는 기사는 후보에서 제외. 확장 시 "
                         "이미 D몫·pool 로 쓴 기사를 빼 누수·중복을 막는다.")
    args = ap.parse_args()

    per_stratum = args.per_stratum or max(PER_STRATUM, args.n // 20)

    exclude = set()
    if args.exclude:
        # 주의: used_ids 는 게이트가 "news-<doc_id>" 형식으로 저장하지만, 여기 scan 이
        # 뽑는 r["doc_id"] 는 원시 "<doc_id>"(접두어 없음)다. 접두어를 벗겨 양쪽을 맞춘다.
        # (이 정규화가 없으면 제외가 조용히 0건 매칭 → D·코어 기사가 확장에 재유입된다.)
        raw = json.load(open(args.exclude))
        exclude = {e[5:] if e.startswith("news-") else e for e in raw}
        print(f"제외 대상 {len(exclude):,}편 로드 (news- 접두어 정규화)", file=sys.stderr)

    names = [n for n in zipfile.ZipFile(ZIP).namelist() if n.endswith(".json")]
    pool = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for part in ex.map(scan, [(n, per_stratum) for n in names]):
            for k, v in part.items():
                pool[k].extend([r for r in v if r["doc_id"] not in exclude])
    print(f"조건 통과 후보 {sum(len(v) for v in pool.values()):,}편 / 층 {len(pool):,}개"
          f" (제외 {len(exclude):,} 반영)", file=sys.stderr)

    rows = pick(pool, args.n)
    if len(rows) < args.n:
        print(f"경고: {args.n}편 요청했으나 {len(rows)}편만 뽑혔습니다.", file=sys.stderr)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{out} — {len(rows)}편", file=sys.stderr)
    tp = defaultdict(int)
    for r in rows:
        tp[r["topic"]] += 1
    print("  topic:", dict(tp), file=sys.stderr)
    print(f"  매체 {len({r['publisher'] for r in rows})}곳 · "
          f"평균 {sum(r['n_char'] for r in rows)//len(rows)}자 · "
          f"평균 본문문단 {sum(r['n_para'] for r in rows)/len(rows):.1f} · "
          f"평균 인용비율 {100*sum(r['quote_ratio'] for r in rows)/len(rows):.1f}%",
          file=sys.stderr)


if __name__ == "__main__":
    main()
