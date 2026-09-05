#!/usr/bin/env python3
"""Binoculars — observer/performer 두 모델의 perplexity 비로 AI 글을 가린다.

MASH 가 쓴 탐지기 5종 중 하나이고, notes/20 이 요구한 likelihood 계열 두 번째다.
Fast-DetectGPT 와 계열은 같지만 **모델 한 개의 편향에 기대지 않는다** — 지금
EXAONE 스코어러로 EXAONE 생성물을 채점하는 self-detection 문제를 다른 각도로 친다.

Hans et al. (2024) 정의:

    logPPL(s)  = -(1/L) Σ_i log q(s_i | s_<i)              performer 가 실제 토큰에 매긴 값
    X-PPL(s)   =  (1/L) Σ_i H( p(·|s_<i), q(·|s_<i) )      observer 분포 기준 교차엔트로피
                 H(p,q) = -Σ_v p(v) log q(v)
    B(s)       =  logPPL / X-PPL

  **B 가 낮을수록 기계 글이다.** 사람 글은 observer 가 예상하기 어려운 선택을 섞어서
  분자가 커진다. 방향이 Fast-DetectGPT(높을수록 AI)와 반대이므로, 비교하기 쉽게
  `score = -B` 를 함께 기록한다. 이 값은 다른 탐지기와 같은 방향(클수록 AI)이다.

  observer 와 performer 는 **토크나이저가 같아야 한다**(위치별 분포를 맞대야 하므로).
  같은 계열의 base / instruct 쌍을 쓴다.

사용:
    python detect_binoculars.py --human ../dataset/human/mash_pilot_10.jsonl \
        --gen ../dataset/generations/mash_pilot_generations.jsonl \
        --out ../dataset/scores/mash_binoculars.jsonl --device cuda:0
"""
import argparse, json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CHUNK = 128        # 시퀀스를 이만큼씩 끊어 소프트맥스한다. vocab 15만이라 한 번에 펴면 OOM.


@torch.no_grad()
def binoculars(obs, perf, tok, text, device, max_len=1024):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
    if ids.shape[1] < 32:
        return None
    obs_logits = obs(ids).logits[0, :-1]        # (T-1, V)
    perf_logits = perf(ids).logits[0, :-1]
    target = ids[0, 1:]

    n = target.shape[0]
    ce_sum = 0.0        # -Σ log q(실제 토큰)
    xent_sum = 0.0      # Σ H(p, q)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        lq = F.log_softmax(perf_logits[s:e].float(), dim=-1)
        lp = F.log_softmax(obs_logits[s:e].float(), dim=-1)
        ce_sum += -lq.gather(-1, target[s:e].unsqueeze(-1)).squeeze(-1).sum().item()
        xent_sum += -(lp.exp() * lq).sum(-1).sum().item()

    logppl = ce_sum / n
    xppl = xent_sum / n
    if xppl <= 0:
        return None
    return logppl / xppl, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observer", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--performer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--human", default="/workspace/dataset/human/mash_pilot_10.jsonl")
    ap.add_argument("--gen", default="/workspace/dataset/generations/mash_pilot_generations.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    print(f"observer  {args.observer}\nperformer {args.performer}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.observer, trust_remote_code=True)
    tok2 = AutoTokenizer.from_pretrained(args.performer, trust_remote_code=True)
    if tok.get_vocab() != tok2.get_vocab():
        raise SystemExit("두 모델의 토크나이저가 다릅니다. 같은 계열 base/instruct 쌍을 쓰세요.")
    obs = AutoModelForCausalLM.from_pretrained(
        args.observer, dtype=torch.float16, trust_remote_code=True).to(args.device).eval()
    perf = AutoModelForCausalLM.from_pretrained(
        args.performer, dtype=torch.float16, trust_remote_code=True).to(args.device).eval()

    rows = []
    for r in (json.loads(l) for l in open(args.human) if l.strip()):
        out = binoculars(obs, perf, tok, r["body"], args.device, args.max_len)
        if out:
            rows.append({"doc_id": r["doc_id"], "label": "human", "model": "human",
                         "cond": None, "bino": out[0], "score": -out[0], "n_tok": out[1]})
    print(f"인간 {len(rows)}편", flush=True)

    gens = [json.loads(l) for l in open(args.gen) if l.strip()]
    for i, g in enumerate(gens):
        if not g.get("text"):
            continue
        out = binoculars(obs, perf, tok, g["text"], args.device, args.max_len)
        if out:
            rows.append({"doc_id": g["doc_id"], "label": "ai", "model": g["model"],
                         "cond": g["cond"], "bino": out[0], "score": -out[0], "n_tok": out[1]})
        if (i + 1) % 20 == 0:
            print(f"  생성물 {i+1}/{len(gens)}", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 쌍 비교. 같은 기사의 x_human 대비 얼마나 움직였나 (길이·주제가 통제된다)
    H = {r["doc_id"]: r["bino"] for r in rows if r["label"] == "human"}
    hs = sorted(H.values())
    print(f"\n인간 {len(hs)}편 Binoculars: {hs[0]:.4f} ~ {hs[-1]:.4f} (중앙값 {hs[len(hs)//2]:.4f})")
    print("낮을수록 기계 글\n")
    by = {}
    for r in rows:
        if r["label"] == "ai":
            by.setdefault(r["model"], []).append(r)
    print(f"{'모델':<16}{'n':>4}{'평균 B':>10}{'평균 ΔB':>10}{'ΔB<0':>8}")
    for m in sorted(by, key=lambda k: sum(x["bino"] for x in by[k]) / len(by[k])):
        v = by[m]
        d = [x["bino"] - H[x["doc_id"]] for x in v if x["doc_id"] in H]
        neg = sum(1 for x in d if x < 0)
        print(f"{m:<16}{len(v):>4}{sum(x['bino'] for x in v)/len(v):>10.4f}"
              f"{sum(d)/len(d):>+10.4f}{neg:>5}/{len(d)}")


if __name__ == "__main__":
    main()
