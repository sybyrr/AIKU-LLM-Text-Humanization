#!/usr/bin/env python3
"""LLM-as-judge 탐지기 — Russell et al. (ACL 2025) 프롬프트를 그대로 적용.

notes/21-러셀-ACL2025-평가방식.md 3절 "프롬프트 탐지기" 행에 해당한다.
원논문 저장소(jenna-russell/human_detectors)의 프롬프트를 한 글자도 고치지 않고 쓴다.
프롬프트 실물은 scripts/prompts/russell/ 에 보관.

변형 4종:
  zero_shot           이진 판정만
  zero_shot_cot       근거 서술 후 판정 (기본값)
  zero_shot_guide     탐지 가이드 + 이진 판정
  zero_shot_cot_guide 탐지 가이드 + 근거 서술 후 판정

한국어 주의:
  · 가이드(detection_guide.txt)는 영어 어휘 목록이 본체다("delve", "tapestry",
    says→explained 치환 등). 한국어에는 구조적 단서만 넘어오므로, 가이드 없는
    두 변형이 언어 독립적인 비교 기준이다.
  · 채점 모델이 생성에도 쓰인 모델이면 자기 출력에 유리해진다(self-detection).
    모델별로 나눠 보고하고, 대각선(채점자 = 생성자)은 따로 표시해야 한다.
  · 러셀은 평가자에게 제목을 안 보여줬다(제목이 같으면 인간/AI 짝이 드러나므로).
    우리 데이터도 본문만 넣는다 — 인간 글은 para_idx>=2, 생성물은 본문만이라 그대로 맞는다.

사용:
  python detect_llm_judge.py --variant zero_shot_cot --n-human 100 --n-per-cell 25
"""
import argparse, json, random, re, time, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

PROMPT_DIR = "/workspace/scripts/prompts/russell"
ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
DESC = re.compile(r"<description>\s*(.*?)\s*</description>", re.S | re.I)


def build_prompt(variant, text):
    tpl = open(f"{PROMPT_DIR}/classify_{variant}.txt").read()
    if "guide" in variant:
        guide = open(f"{PROMPT_DIR}/detection_guide.txt").read()
        return tpl.format(guide, text)
    return tpl.format(text)


def parse(raw):
    """<answer> 안의 라벨을 뽑는다. 태그를 안 지키면 본문 전체에서 찾는다."""
    m = ANSWER.search(raw)
    seg = m.group(1) if m else raw[-200:]
    up = seg.upper()
    ai, hu = "AI-GENERATED" in up, "HUMAN-WRITTEN" in up
    if ai and not hu:
        return "ai"
    if hu and not ai:
        return "human"
    return None  # 둘 다 있거나 둘 다 없으면 파싱 실패로 남긴다


def judge(url, prompt, max_tokens, think, timeout):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.9,
        "seed": 42,
    }
    if not think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def sample(args):
    """인간 전편 + 생성물은 (모델 x 조건) 셀마다 n-per-cell 편씩 균등 추출."""
    rnd = random.Random(args.seed)
    items = []
    humans = [json.loads(l) for l in open(args.human) if l.strip()]
    rnd.shuffle(humans)
    for r in humans[:args.n_human]:
        items.append({"doc_id": r["doc_id"], "label": "human", "model": "human",
                      "cond": None, "text": r["body"]})
    cells = defaultdict(list)
    for l in open(args.gen):
        r = json.loads(l)
        if r.get("text"):
            cells[(r["model"], r["cond"])].append(r)
    for (m, c), rows in sorted(cells.items()):
        if args.conds and c not in args.conds:
            continue
        rnd.shuffle(rows)
        for r in rows[:args.n_per_cell]:
            items.append({"doc_id": r["doc_id"], "label": "ai", "model": m,
                          "cond": c, "text": r["text"]})
    rnd.shuffle(items)          # 러셀 제시 규칙: 순서 무작위화
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="zero_shot_cot",
                    choices=["zero_shot", "zero_shot_cot", "zero_shot_guide", "zero_shot_cot_guide"])
    ap.add_argument("--judge", required=True, help="결과에 기록할 채점 모델 이름")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--human", default="/workspace/dataset/human/full_base.jsonl")
    ap.add_argument("--gen", default="/workspace/dataset/generations/full_generations.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-human", type=int, default=100)
    ap.add_argument("--n-per-cell", type=int, default=25)
    ap.add_argument("--conds", nargs="*", default=None, help="예: A E C (미지정이면 전부)")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--think", action="store_true", help="하이브리드 모델의 사고 모드 켜기")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="스모크용 상한")
    args = ap.parse_args()

    items = sample(args)
    if args.limit:
        items = items[:args.limit]
    n_h = sum(i["label"] == "human" for i in items)
    print(f"{args.judge} / {args.variant} / 사고모드 {'on' if args.think else 'off'}: "
          f"인간 {n_h} + AI {len(items)-n_h} = {len(items)}편", flush=True)

    def work(it):
        t0 = time.time()
        try:
            r = judge(args.url, build_prompt(args.variant, it["text"]),
                      args.max_tokens, args.think, args.timeout)
        except Exception as e:
            return {**{k: it[k] for k in ("doc_id", "label", "model", "cond")},
                    "pred": None, "error": str(e)[:200]}
        msg = r["choices"][0]["message"]
        raw = msg.get("content") or ""
        reason = msg.get("reasoning_content") or ""
        d = DESC.search(raw)
        return {**{k: it[k] for k in ("doc_id", "label", "model", "cond")},
                "pred": parse(raw),
                "desc": (d.group(1) if d else "")[:1200],
                "reason_char": len(reason),
                "n_char": len(it["text"]),
                "finish": r["choices"][0].get("finish_reason"),
                "tokens": r.get("usage", {}).get("completion_tokens"),
                "elapsed": round(time.time() - t0, 1)}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.slots) as ex:
        rows = list(ex.map(work, items))
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in rows if r.get("pred")]
    print(f"완료 {len(ok)}/{len(rows)} · {time.time()-t0:.0f}초 "
          f"(편당 {sum(r.get('elapsed',0) for r in rows)/max(len(rows),1):.1f}초)", flush=True)
    bad = [r for r in rows if not r.get("pred")]
    if bad:
        print(f"  판정 실패 {len(bad)}건, 예: {bad[0].get('error') or '태그 파싱 실패'}", flush=True)

    # 러셀 지표: TPR/FPR (임계값 조정 불가한 이진 판정이므로 AUROC 안 씀)
    hum = [r for r in ok if r["label"] == "human"]
    fpr = sum(r["pred"] == "ai" for r in hum) / len(hum) * 100 if hum else float("nan")
    print(f"\nFPR(인간을 AI로 오판) {fpr:.1f}%  n={len(hum)}")
    by = defaultdict(list)
    for r in ok:
        if r["label"] == "ai":
            by[r["model"]].append(r["pred"] == "ai")
    print(f"{'생성 모델':16}{'n':>5}{'TPR':>8}")
    for m in sorted(by, key=lambda m: -sum(by[m]) / len(by[m])):
        v = by[m]
        print(f"{m:16}{len(v):>5}{sum(v)/len(v)*100:>7.1f}%")


if __name__ == "__main__":
    main()
