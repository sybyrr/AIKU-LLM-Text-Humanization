#!/usr/bin/env python3
"""조건 C 용 한 문장 요약 생성 + 검사.

러셀의 subtitle 대응물. 코퍼스에 부제가 없어서 합성한다.
길이 목표는 러셀 실측 비율(부제/기사 = 2.9%)에 맞춰 50~70자.

검사 두 가지 (둘 다 실패하면 조건 C 가 무의미해진다):
  1. 리드 베끼기 — 요약이 원문 문장을 그대로 옮기면 "산문 유출 없음" 이라는
     조건 C 의 존재 이유가 사라진다. 원문과의 최장 공통 부분문자열로 검사.
  2. 환각 — 요약에 원문에 없는 숫자가 들어가면 생성 모델이 틀린 전제로
     기사를 쓰게 된다. 숫자 토큰이 원문에 있는지 확인.
"""
import argparse, json, re, urllib.request

PROMPT = """다음 신문 기사를 한 문장으로 요약하세요. 조건:
- 50자 이상 70자 이하
- 기사에 실제로 나온 사실만 사용
- 기사 문장을 그대로 옮기지 말고 새로 쓸 것
- 요약문 한 문장만 출력하고 다른 말은 하지 말 것

기사:
{body}

한 문장 요약:"""

def lcs_len(a, b):
    """최장 공통 부분문자열 길이 (원문 베끼기 검사용)"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/workspace/dataset/human/full_base.jsonl")
    ap.add_argument("--out", default="/workspace/dataset/prompts/full_summaries.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--max-copy", type=int, default=25,
                    help="원문과 겹치는 연속 글자 수 상한")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    out = []
    for i, row in enumerate(rows):
        body = row["body"]
        payload = {"messages": [{"role": "user", "content": PROMPT.format(body=body)}],
                   "max_tokens": 400, "temperature": 0.3, "seed": 100 + i}
        if args.no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(args.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            s = json.load(r)["choices"][0]["message"]["content"]
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip().strip('"「」').split("\n")[0].strip()

        copy = lcs_len(s, body)
        nums_s = set(re.findall(r"\d+(?:\.\d+)?", s))
        nums_b = set(re.findall(r"\d+(?:\.\d+)?", body))
        hallu = sorted(nums_s - nums_b)

        rec = {"doc_id": row["doc_id"], "summary": s, "sum_char": len(s),
               "max_copy": copy, "hallucinated_nums": hallu,
               "ok": copy <= args.max_copy and not hallu}
        out.append(rec)
        flag = "OK " if rec["ok"] else "확인"
        print(f'{flag} [{len(s):3d}자 겹침{copy:3d}자{" 환각:"+",".join(hallu) if hallu else ""}] {s[:60]}')

    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    good = sum(o["ok"] for o in out)
    print(f"\n{good}/{len(out)} 통과 | 평균 {sum(o['sum_char'] for o in out)//len(out)}자 "
          f"| 최대겹침 {max(o['max_copy'] for o in out)}자")

if __name__ == "__main__":
    main()
