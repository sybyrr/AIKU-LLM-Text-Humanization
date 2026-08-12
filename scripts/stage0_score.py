#!/usr/bin/env python3
"""Stage 0 파일럿용 Fast-DetectGPT 채점기.

detect_fastdetectgpt.py 와 곡률 수식은 완전히 동일하다(점수 비교 가능성 유지).
달라진 점은 셋:
  · 임의 JSONL을 채점한다 (인간/생성 2파일 스키마에 묶이지 않음)
  · CPU 실행을 지원한다 (--dtype float32)
  · resume 을 지원한다 (CPU는 느려서 중단 복구가 필요)

사용:
  python3 stage0_score.py --in pilot/data/seed_abstract.jsonl \
      --text-field human_text --id-field pair_id --label human \
      --out pilot/scores/seed_abstract.jsonl --device cpu
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def curvature(model, tok, text, device, max_len=2048):
    """Fast-DetectGPT 해석적 곡률 d(x). 클수록 AI 쪽.

    detect_fastdetectgpt.py L32-56 과 동일 (mean 방식 — sum 은 길이 편향 버그).
    """
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
    if ids.shape[1] < 32:
        return None
    logits = model(ids).logits[0, :-1]
    target = ids[0, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)

    actual = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    p = logp.exp()
    mu = (p * logp).sum(-1)
    var = ((p * logp.pow(2)).sum(-1) - mu.pow(2)).clamp_min(1e-8)

    d = (actual.mean() - mu.mean()) / var.mean().sqrt()
    return d.item(), actual.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scorer", default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--id-field", default="pair_id")
    ap.add_argument("--label", required=True, help="human | ai")
    ap.add_argument("--group", default="", help="모델명·프롬프트 변형 등 집계 축")
    ap.add_argument("--group-field", default="", help="레코드에서 group 을 읽을 필드명")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16"])
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    # CPU 에서 fp16 matmul 은 극단적으로 느리거나 미지원 — 자동으로 fp32 로 간다
    dtype = {"float32": torch.float32, "float16": torch.float16}.get(
        a.dtype, torch.float32 if a.device.startswith("cpu") else torch.float16
    )

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]

    # 같은 원문이 프롬프트 변형별로 여러 번 등장하므로 (id, group) 을 키로 쓴다
    def key(rec_id, grp):
        return (str(rec_id), str(grp or ""))

    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    done.add(key(o["id"], o.get("group")))
        print(f"resume: 기존 {len(done)}건 건너뜀", flush=True)

    def grp_of(r):
        return r.get(a.group_field, "") if a.group_field else a.group

    todo = [r for r in rows if key(r[a.id_field], grp_of(r)) not in done]
    if not todo:
        print("이미 전부 채점됨")
        return

    print(f"스코어링 모델 로딩: {a.scorer} ({a.device}, {dtype})", flush=True)
    tok = AutoTokenizer.from_pretrained(a.scorer, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(a.scorer, torch_dtype=dtype, trust_remote_code=True)
        .to(a.device)
        .eval()
    )

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n_short = 0
    with open(a.out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            text = (r.get(a.text_field) or "").strip()
            if not text:
                continue
            out = curvature(model, tok, text, a.device, a.max_len)
            if out is None:  # 32토큰 미만 — 조용히 버리지 않고 세어 둔다
                n_short += 1
                continue
            group = grp_of(r)
            f.write(
                json.dumps(
                    {
                        "id": str(r[a.id_field]),
                        "label": a.label,
                        "group": group,
                        "domain": r.get("domain", ""),
                        "score": out[0],
                        "n_tok": out[1],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()  # 중단 대비
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)

    if n_short:
        print(f"⚠ 32토큰 미만으로 제외: {n_short}건", file=sys.stderr)
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
