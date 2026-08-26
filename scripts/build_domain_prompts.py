#!/usr/bin/env python3
"""Stage 1b — P3b 통일 프롬프트 조립 (도메인 무관 정본).

`news_build_p3b.py`(신문 전용)와 `prompts/mash/rewrite_ko.user.txt`(청원·위키가 쓴
P2 회귀본, 폐기)를 **하나로 대체**한다. 전 도메인이 글자까지 같은 프롬프트를 쓴다.

정본 결정(노트 51 §1b·§7): 원논문 MASH 는 도메인별 프롬프트 분기 없이 하나의 재작성
절차를 균일 적용한다. rewrite_ko 는 "문장 순서 유지"(= P3b 가 복사 유발로 제거한 P2 요구)를
도로 넣어 신문과 불일치했다. 여기서는 그 요구를 넣지 않는다.

입력  human_pool.jsonl : {doc_id, text|body, n_char, ...}
출력  prompts.jsonl    : {doc_id, cond, system, prompt, target_char, n_char}
                         스키마는 stage0/generate 계열과 호환.

사용:
    python build_domain_prompts.py --in  dataset/{dom}_track/human_pool.jsonl \
                                   --out dataset/{dom}_track/prompts.jsonl \
                                   --domain {dom} [--exclude 파일] [--n N]
"""
import argparse, json, pathlib

# ── P3b 통일 프롬프트 (도메인 무관) ─────────────────────────────────────────────
# "기사/초록/글" 같은 도메인 단어를 넣지 않는다. 사실 보존 목록은 장르 공통 항목만 둔다.
SYSTEM = "당신은 한국어 글을 다듬는 전문 편집자입니다."

USER = (
    "다음 한국어 글을 같은 내용, 다른 문장으로 다시 쓰세요.\n"
    "- 원문의 문장을 그대로 옮기지 말고, 어휘와 문장 구조를 새로 짜세요. "
    "문장을 합치거나 나누어도 좋습니다.\n"
    "- 사실은 하나도 바꾸지 마세요: 숫자·연도·비율·단위·통계값, 인명·기관명·지명·전문 용어, "
    "증가/감소 같은 방향과 결론을 모두 보존하세요.\n"
    "- 따옴표 안의 인용문은 요약하거나 바꾸지 말고 글자 그대로 유지하세요.\n"
    "- 요약하지 말고 분량을 원문의 90~110%로 유지하세요.\n"
    "다시 쓴 글만 출력하세요 — 설명이나 제목, 마크다운 서식은 쓰지 마세요.\n\n"
    "{text}"
)


def get_text(r):
    """human_pool 스키마 차이 흡수: 신문 풀은 'body', 청원·위키·기타는 'text'."""
    for k in ("body", "text", "human_text"):
        if r.get(k):
            return r[k]
    raise KeyError(f"본문 필드 없음 (doc_id={r.get('doc_id')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="human_pool.jsonl")
    ap.add_argument("--out", required=True, help="prompts.jsonl")
    ap.add_argument("--domain", required=True, help="cond 라벨 및 doc_id 접두어")
    ap.add_argument("--exclude", default=None, help="이 파일의 doc_id 는 제외(파일럿/누수 분리)")
    ap.add_argument("--n", type=int, default=0, help="0 이면 전량")
    args = ap.parse_args()

    seen = set()
    if args.exclude:
        seen = {json.loads(l)["doc_id"] for l in open(args.exclude) if l.strip()}
        print(f"제외 대상 {len(seen)}편", flush=True)

    cond = f"P3b-{args.domain}"
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for l in open(args.inp):
            if not l.strip():
                continue
            r = json.loads(l)
            did = r["doc_id"]
            if did in seen:
                continue
            # doc_id 에 도메인 접두어가 없으면 붙여 트랙 간 충돌 방지(stage0 조인 키와 일치).
            pid = did if str(did).startswith(f"{args.domain}-") else f"{args.domain}-{did}"
            text = get_text(r)
            nchar = r.get("n_char", len(text))
            f.write(json.dumps({
                "doc_id": pid,
                "cond": cond,
                "system": SYSTEM,
                "prompt": USER.replace("{text}", text),
                "target_char": nchar,
                "n_char": nchar,
            }, ensure_ascii=False) + "\n")
            n += 1
            if args.n and n >= args.n:
                break
    print(f"{out} — {n}편 (cond={cond})", flush=True)


if __name__ == "__main__":
    main()
