#!/usr/bin/env python3
"""Fast-DetectGPT — 조건부 확률 곡률 기반 AI 텍스트 탐지.

notes/20-평가기준.md 15번 줄이 요구하는 "likelihood 계열" 탐지기 중 하나.
학습이 필요 없고 스코어링 모델 한 개로 단일 forward pass에 계산된다.

원리:
  같은 모델이 매긴 실제 토큰의 로그확률이, 그 위치에서 모델이 뽑았을 법한
  토큰들의 로그확률 분포에서 얼마나 위에 있는지를 잰다.
      d(x) = (log p(x) - mu) / sigma
      mu    = E_{x~p}[log p(x)]        위치별 기댓값
      sigma = sqrt(Var_{x~p}[log p(x)])
  AI 생성문은 모델이 고확률로 뽑은 토큰만 쓰므로 d 가 크고,
  인간 글은 모델 기준 "의외의" 선택이 섞여 d 가 작다.

한국어 주의:
  · 스코어링 모델은 한국어를 제대로 다루는 것이어야 한다. 영어 위주 모델을 쓰면
    인간 글도 전부 낮은 확률을 받아 변별이 안 된다.
  · 생성에 쓴 모델로 그 모델의 출력을 채점하면 자기 출력에 유리해진다(self-detection).
    별도 모델을 쓰거나, 최소한 모델별로 나눠 보고해야 한다.

사용:
  python detect_fastdetectgpt.py --scorer LGAI-EXAONE/EXAONE-4.0-1.2B \
      --human full_base.jsonl --gen full_generations.jsonl --out scores.jsonl
"""
import argparse, json, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def curvature(model, tok, text, device, max_len=2048):
    """Fast-DetectGPT 의 해석적 곡률 d(x). 값이 클수록 AI 쪽."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
    if ids.shape[1] < 32:
        return None
    logits = model(ids).logits[0, :-1]           # (T-1, V) 다음 토큰 예측
    target = ids[0, 1:]                           # (T-1,)
    logp = F.log_softmax(logits.float(), dim=-1)  # (T-1, V)

    # 실제 토큰의 로그확률 합
    actual = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)   # (T-1,)

    # 위치별로 모델 분포 하에서의 기댓값과 분산 (샘플링 없이 해석적으로)
    p = logp.exp()
    mu = (p * logp).sum(-1)                                      # E[log p]
    var = (p * logp.pow(2)).sum(-1) - mu.pow(2)                  # Var[log p]
    var = var.clamp_min(1e-8)

    # 합이 아니라 평균으로 계산해야 한다(Bao et al. 원 구현).
    # 합으로 하면 분자는 n, 분모는 sqrt(n) 에 비례해 점수가 sqrt(n) 만큼 부풀고,
    # 길이가 다른 모델끼리 비교가 불가능해진다(실측 길이-점수 상관 0.68).
    n = actual.shape[0]
    d = (actual.mean() - mu.mean()) / var.mean().sqrt()
    return d.item(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    ap.add_argument("--human", default="/workspace/dataset/human/full_base.jsonl")
    ap.add_argument("--gen", default="/workspace/dataset/generations/full_generations.jsonl")
    ap.add_argument("--out", default="/workspace/dataset/scores/detect_scores.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    print(f"스코어링 모델 로딩: {args.scorer}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.scorer, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.scorer, torch_dtype=torch.float16, trust_remote_code=True).to(args.device).eval()

    rows = []
    humans = [json.loads(l) for l in open(args.human) if l.strip()]
    for r in humans:
        out = curvature(model, tok, r["body"], args.device, args.max_len)
        if out:
            rows.append({"doc_id": r["doc_id"], "label": "human", "model": "human",
                         "cond": None, "score": out[0], "n_tok": out[1]})
    print(f"인간 {len(rows)}편", flush=True)

    gens = [json.loads(l) for l in open(args.gen) if l.strip()]
    for i, g in enumerate(gens):
        if not g.get("text"):
            continue
        out = curvature(model, tok, g["text"], args.device, args.max_len)
        if out:
            rows.append({"doc_id": g["doc_id"], "label": "ai", "model": g["model"],
                         "cond": g["cond"], "score": out[0], "n_tok": out[1]})
        if (i + 1) % 200 == 0:
            print(f"  생성물 {i+1}/{len(gens)}", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 요약: 인간 글 5% 만 오탐하는 임계값에서의 TPR (notes/20 축 1 표준)
    hs = sorted(r["score"] for r in rows if r["label"] == "human")
    if hs:
        thr = hs[max(0, int(len(hs) * 0.95) - 1)]
        print(f"\n인간 {len(hs)}편 · FPR 5% 임계값 = {thr:.3f}")
        print(f"{'모델':<16}{'조건':<6}{'n':>5}{'TPR@FPR5%':>12}{'평균 점수':>11}")
        by = {}
        for r in rows:
            if r["label"] == "ai":
                by.setdefault((r["model"], r["cond"]), []).append(r["score"])
        for k in sorted(by):
            v = by[k]
            tpr = sum(s > thr for s in v) / len(v) * 100
            print(f"{k[0]:<16}{k[1] or '-':<6}{len(v):>5}{tpr:>11.1f}%{sum(v)/len(v):>11.3f}")


if __name__ == "__main__":
    main()
