#!/usr/bin/env python3
"""Stage 1d — 게이트 통과 쌍들을 합쳐 '누수 없는' 최종 데이터셋으로 조립한다.

Stage 1c(게이트)가 여러 번(코어·확장·top-up) 돌면 결과가 여러 파일로 흩어지고, 그중 확장분에
D 학습 기사가 섞여 들어갈 수 있다 — 과거 `news_dpair_final.jsonl` 의 **722편 D-누수**가 그것이다.
그 조립·정제가 지금까지 수동이라 재현 불가였다(`audit_overlap.py` 는 사후 검사만 했다). 이 스크립트가
그 단계를 코드로 못박아 **누수를 구조적으로 차단**한다:

  1) 여러 pair 파일을 순서대로 합친다(앞 파일 우선)
  2) D 학습 기사(detector-split)와 지정 제외 id 를 **전부 뺀다**  ← 누수 차단
  3) 기사 단위로 중복 제거 (1 기사 = 1 쌍; 2-arm 중복 폐기)
  4) 기사 id 해시로 train/dev/test 분할 (seed 고정 → 재실행해도 동일, 같은 기사는 한 split 에만)
  5) 누수 0·중복 0·split 분리를 assert 로 검증하고 감사 요약을 출력

pair 스키마: {id, generator, human_text, ai_text, d_human, d_ai}  (+ 조립 후 split 필드)
id 는 `<domain>-<기사id>` (예: news-NIRW2200000001.10293). detector_split.json 은 {detector:[...], pool:[...]}.

사용:
  python build_clean_dataset.py \
      --pairs core.jsonl expand.jsonl topup.jsonl \
      --exclude-detector dataset/news_track/detector_split.json \
      --split 8:1:1 --seed 42 \
      --out dataset/news_track/news_dpair_clean20k.jsonl
"""
import argparse, hashlib, json, pathlib


def norm(i):
    """도메인 접두어(news-/petition-/wiki- 등)를 떼어 원 기사 id 로 정규화한다.
    detector_split 과 pair 가 접두어 유무로 어긋나던 과거 버그(note 50)를 막는다."""
    i = str(i)
    return i.split("-", 1)[1] if ("-" in i and not i[0].isdigit()) else i


def load_ids(path):
    d = json.load(open(path))
    ids = d.get("detector", d.get("ids", [])) if isinstance(d, dict) else d
    return {norm(x) for x in ids}


def split_of(article_id, ratios, seed):
    """기사 id 해시로 결정적 분할. 같은 기사는 항상 같은 split → 재실행 안정, 누수 방지."""
    tr, dv, te = ratios
    h = int(hashlib.md5(f"{seed}:{article_id}".encode()).hexdigest(), 16) % (tr + dv + te)
    return "train" if h < tr else ("dev" if h < tr + dv else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="게이트 통과 pair jsonl (복수 지정 시 앞 파일 우선)")
    ap.add_argument("--exclude-detector", action="append", default=[],
                    help="detector_split.json — 그 'detector' 기사를 제외(누수 차단). 복수 지정 가능")
    ap.add_argument("--exclude-ids", action="append", default=[],
                    help="추가 제외 id 목록(json 배열 또는 detector_split). 복수 지정 가능")
    ap.add_argument("--split", default="8:1:1", help="train:dev:test 비율")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ratios = tuple(int(x) for x in args.split.split(":"))
    assert len(ratios) == 3, "--split 은 train:dev:test 형식이어야 한다"

    exclude = set()
    for p in args.exclude_detector + args.exclude_ids:
        exclude |= load_ids(p)
    print(f"제외 기사 {len(exclude)}편 (D 학습 + 지정)", flush=True)

    seen, rows, n_in, n_leak, n_dup = set(), [], 0, 0, 0
    for path in args.pairs:                       # 앞 파일 우선(코어 먼저, top-up 나중)
        for l in open(path):
            if not l.strip():
                continue
            n_in += 1
            r = json.loads(l)
            aid = norm(r["id"])
            if aid in exclude:                    # ← 누수 차단
                n_leak += 1
                continue
            if aid in seen:                       # 기사 단위 중복 제거(2-arm 폐기)
                n_dup += 1
                continue
            seen.add(aid)
            r["split"] = split_of(aid, ratios, args.seed)
            rows.append(r)

    # ── 안전장치: 누수·중복·split 분리 검증 (실패 시 즉시 중단) ──
    ids = [norm(r["id"]) for r in rows]
    assert not (set(ids) & exclude), "누수 검증 실패: 제외 기사가 결과에 남아 있다"
    assert len(ids) == len(set(ids)), "중복 검증 실패: 같은 기사가 두 번 있다"
    by = {"train": [], "dev": [], "test": []}
    for r in rows:
        by[r["split"]].append(r)
    tr, dv, te = (set(norm(r["id"]) for r in by[s]) for s in ("train", "dev", "test"))
    assert not (tr & te) and not (tr & dv) and not (dv & te), "split 간 기사 겹침"

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gens = {}
    for r in rows:
        gens[r.get("generator", "?")] = gens.get(r.get("generator", "?"), 0) + 1
    print(f"조립 완료 → {out}", flush=True)
    print(f"  입력 {n_in}행 → 누수 제외 {n_leak} · 중복 제외 {n_dup} · 최종 {len(rows)}쌍", flush=True)
    print(f"  split: train {len(by['train'])} / dev {len(by['dev'])} / test {len(by['test'])}", flush=True)
    print(f"  생성기: {gens}", flush=True)
    print("  ✅ 누수 0 · 중복 0 · split 분리 검증 통과", flush=True)


if __name__ == "__main__":
    main()
