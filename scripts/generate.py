#!/usr/bin/env python3
"""llama-server 에 프롬프트를 병렬로 던져 생성 결과를 JSONL 로 받는다.

사용:
  python generate.py --in prompts.jsonl --out gen.jsonl --model exaone-33b [--slots 4] [--resume]

instruct 모델이므로 /v1/chat/completions 를 쓴다. llama-server 가 GGUF 에 들어 있는
chat template 을 적용해준다. raw /completion 을 쓰면 템플릿이 안 붙어서 지시 이행이
나빠진다.

Qwen3 계열은 하이브리드 사고 모드가 기본 on 이라 <think> 블록이 섞여 나온다.
chat_template_kwargs.enable_thinking=false 로 끄고, 그래도 새어나오면 사후 제거한다.
"""
import argparse, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

THINK = re.compile(r"<think>.*?</think>\s*", re.S)

def chat(url, prompt, n_predict, seed, temp, no_think, system=None):
    # 통일 프롬프트(P3b)는 system/user 를 나눠 쓴다(build_domain_prompts.py 가 system 필드를 넣는다).
    # 러셀식 조건 A/C/E 는 system 이 없으므로 user 하나만 보낸다.
    messages = ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "max_tokens": n_predict,
        "temperature": temp,
        "top_p": 0.9,
        "seed": seed,
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)

def clean(text):
    """사고 블록과 흔한 군더더기 제거"""
    text = THINK.sub("", text).strip()
    # 첫 줄이 라벨뿐이면 제거한다. 러셀식은 "제목:" 이 새어나왔고,
    # 재서술 프롬프트는 "다듬은 본문:" 류의 머리말이 붙는 경우가 있다.
    lines = text.split("\n")
    if lines and re.match(
            r"^\s*(제목|Title|(다듬은|수정된|편집된|정리한)\s*(본문|글|텍스트)|Polished(\s+text)?)"
            r"\s*[:：]\s*$", lines[0]):
        text = "\n".join(lines[1:]).strip()
    elif lines and re.match(
            r"^\s*(제목|Title)\s*[:：]", lines[0]):
        text = "\n".join(lines[1:]).strip()
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="결과에 기록할 모델 이름")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-think", action="store_true", help="Qwen3 등 사고 모드 끄기")
    ap.add_argument("--retries", type=int, default=3, help="간헐적 파서 오류 재시도 횟수")
    ap.add_argument("--resume", action="store_true",
                    help="기존 --out 의 성공분(text 있는 레코드)을 건너뛰고 이어서 append")
    # max_tokens = target_char * token_ratio.
    # 실측 2.87자/토큰(EXAONE)이므로 필요 토큰은 target_char/2.87 ≈ 0.35배.
    # 0.8 로 두면 2배 이상 여유가 있으면서, 슬롯 컨텍스트(-c/-np = 4096)를 넘지 않는다.
    # 1.6 으로 두면 target_char 2400 인 건이 3840토큰을 요청해 슬롯 한도를 넘겨
    # HTTP 500 이 난다(Qwen3 토크나이저가 한국어를 더 잘게 쪼개 특히 취약).
    ap.add_argument("--token-ratio", type=float, default=0.8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    # resume: 이미 성공한 (doc_id, cond) 는 건너뛴다. 오류 레코드는 재시도 대상으로 남긴다.
    # (장시간 배치가 중간에 죽으면 전량 손실이던 것을 방지)
    mode = "w"
    if args.resume and os.path.exists(args.out):
        done = set()
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    if o.get("text"):
                        done.add((o["doc_id"], o["cond"]))
        rows = [r for r in rows if (r["doc_id"], r["cond"]) not in done]
        mode = "a"
        print(f"resume: 완료 {len(done)}건 건너뜀", flush=True)
    print(f"{args.model}: {len(rows)}건, 슬롯 {args.slots}", flush=True)

    def work(item):
        i, row = item
        t0 = time.time()
        # llama.cpp 의 chat 파서가 간헐적으로 500 을 낸다
        # ("does not match the expected peg-native format"). 파라미터 문제가 아니라
        # 샘플링 결과에 따라 뜨는 것이라, 시드를 바꿔 재시도하면 통과한다. 90건 중 1건 빈도.
        last = None
        for attempt in range(args.retries + 1):
            try:
                r = chat(args.url, row["prompt"], int(row["target_char"] * args.token_ratio),
                         args.seed + i + 1000 * attempt, args.temp, args.no_think,
                         row.get("system"))
                break
            except Exception as e:
                last = e
                time.sleep(1)
        else:
            return {"doc_id": row["doc_id"], "cond": row["cond"], "model": args.model,
                    "error": f"{args.retries + 1}회 시도 실패: {str(last)[:150]}"}
        try:                                       # 200 이지만 choices 가 없는 오류 페이로드 방어
            msg = r["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return {"doc_id": row["doc_id"], "cond": row["cond"], "model": args.model,
                    "error": f"응답 형식 오류: {str(r)[:150]}"}
        raw = msg.get("content") or ""
        # 추론 모델이 사고 모드로 돌면 content 가 비고 reasoning_content 로 빠진다.
        # 조용히 빈 문자열이 저장되면 전량 공백이 되므로 에러로 올린다.
        reason_len = len(msg.get("reasoning_content") or "")
        text = clean(raw)
        if not text:
            return {"doc_id": row["doc_id"], "cond": row["cond"], "model": args.model,
                    "error": f"빈 출력 (reasoning_content {reason_len}자, "
                             f"finish={r['choices'][0].get('finish_reason')})"}
        return {
            "doc_id": row["doc_id"], "cond": row["cond"], "model": args.model,
            "target_char": row["target_char"], "human_char": row.get("n_char"),
            "gen_char": len(text), "tokens": r.get("usage", {}).get("completion_tokens"),
            "had_think": "<think>" in raw or reason_len > 0,
            "reason_char": reason_len,
            "has_markdown": bool(re.search(r"(\*\*|^#{1,4}\s|^\s*[-*•]\s)", text, re.M)),
            "elapsed": round(time.time() - t0, 1), "text": text,
        }

    # 완료되는 즉시 디스크에 스트리밍 기록(락) → 배치가 중간에 죽어도 partial 이 남아 --resume 가능.
    import threading
    t0 = time.time()
    lock = threading.Lock()
    out = []
    with open(args.out, mode) as f:
        def run_one(item):
            o = work(item)
            with lock:
                f.write(json.dumps(o, ensure_ascii=False) + "\n"); f.flush()
                out.append(o)
            return o
        with ThreadPoolExecutor(max_workers=args.slots) as ex:
            list(ex.map(run_one, enumerate(rows)))

    ok = [o for o in out if "error" not in o]
    err = [o for o in out if "error" in o]
    el = time.time() - t0
    tok = sum(o["tokens"] or 0 for o in ok)
    print(f"완료 {len(ok)} 실패 {len(err)} | {tok}토큰 / {el:.0f}초 = {tok/max(el,1):.1f} t/s", flush=True)
    if err:
        print("  실패 예:", err[0]["error"][:120], flush=True)
    if ok:
        dev = [o["gen_char"] / o["target_char"] for o in ok]
        print(f"  길이 달성률 평균 {sum(dev)/len(dev)*100:.0f}% "
              f"(최소 {min(dev)*100:.0f}% 최대 {max(dev)*100:.0f}%)", flush=True)
        print(f"  마크다운 {sum(o['has_markdown'] for o in ok)}건 / "
              f"사고블록 {sum(o['had_think'] for o in ok)}건", flush=True)

if __name__ == "__main__":
    main()
