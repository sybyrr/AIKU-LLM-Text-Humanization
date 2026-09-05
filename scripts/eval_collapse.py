#!/usr/bin/env python3
"""임의 StyleBART ckpt(concat/additive 자동판별)를 clean20k TEST split(held-out)에서
생성해 붕괴(collapse율·연속반복)·탐지 D·의미 cos 를 두 디코딩으로 측정한다.

왜 clean20k test 인가: 기존 transfer 셋(news_dpair_v2 test)은 clean20k 와 split 이 어긋나
150편 중 116편이 clean20k **train** 에 있다 → concat+20k 를 train 으로 학습 후 그 셋으로
붕괴를 재면 학습데이터 누수. clean20k test(2,044) 는 어느 SFT 학습에도 안 들어가 held-out.

붕괴율 정의 = 한 어절이 30회 초과 반복된 문서 비율(compare_repetition.py 와 동일).
사용:
  python eval_collapse.py --ckpt /workspace/models/news_sft_concat20k/style_bart.pt --tag concat20k
"""
import argparse, json, collections, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers.modeling_outputs import BaseModelOutput
from stage2_sft import StyleBART, MODEL

NT = "/workspace/dataset/news_track"; DEV = "cuda:0"
DECS = {"plain": dict(num_beams=4, max_length=1024)}   # 논문 세팅(맨 beam4). EOS 고친 뒤 정직한 평가(반복방지 크러치 제거).


def load_any(path):
    """ckpt fusion.weight shape 로 concat(768x1536)/additive(768x768) 자동판별해 로드."""
    ck = torch.load(path, map_location=DEV)["model"]
    fs = tuple(ck["fusion.weight"].shape)
    kind = "concat" if fs[1] == 2 * fs[0] else "additive"
    m = StyleBART().to(DEV)                    # 기본은 concat(Linear(2d,d))
    if kind == "additive":
        d = m.bart.config.d_model
        m.fusion = nn.Linear(d, d).to(DEV)     # additive 는 Linear(d,d) 로 교체 후 로드
    m.load_state_dict(ck); m.eval()
    return m, kind


def fuse(m, cr, style, kind):
    if kind == "concat":
        sb = style.view(1, 1, -1).expand(cr.size(0), cr.size(1), -1)
        return m.fusion(torch.cat([cr, sb], dim=-1))
    return cr + m.fusion(cr + style)           # additive


@torch.no_grad()
def gen(m, kind, tok, x, dec):
    e = tok(x, return_tensors="pt", truncation=True, max_length=512).to(DEV)
    cr = m.encode(e.input_ids, e.attention_mask)
    fused = fuse(m, cr, m.hsr, kind)
    o = m.bart.generate(encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                        attention_mask=e.attention_mask, **dec)
    return tok.decode(o[0], skip_special_tokens=True)


def loop_scan(toks):
    """가장 긴 주기적 반복구간(period 1~4) → (길이, period, 시작)."""
    best = (0, 1, 0)
    for p in range(1, 5):
        i = p
        while i < len(toks):
            if toks[i] == toks[i - p]:
                s = i - p
                while i < len(toks) and toks[i] == toks[i - p]: i += 1
                if i - s > best[0]: best = (i - s, p, s)
            else: i += 1
    return best


def collapse_stats(texts):
    coll = 0; runs = []; lens = []
    for t in texts:
        toks = t.split(); lens.append(len(toks))
        mc = max(collections.Counter(toks).values()) if toks else 0
        if mc > 30: coll += 1
        runs.append(loop_scan(toks)[0])
    n = len(texts)
    return dict(collapse=100 * coll / n, maxrun=float(np.mean(runs)), olen=float(np.mean(lens)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", default=f"{NT}/news_dpair_clean20k.jsonl")
    ap.add_argument("--split", default="test", help="held-out 평가 split (기본 test)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default=None, help="샘플 출력 jsonl (선택)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(args.pairs) if json.loads(l).get("split") == args.split][:args.n]
    xai = [r["ai_text"] for r in rows]
    print(f"[{args.tag}] eval={args.split} n={len(rows)} (held-out)", flush=True)

    m, kind = load_any(args.ckpt)
    print(f"로드: {args.ckpt}  fusion={kind}", flush=True)

    # 우리 D + ko-sroberta (테스트 스크립트 관례)
    dtok = AutoTokenizer.from_pretrained("/workspace/models/news_roberta_D")
    dm = AutoModelForSequenceClassification.from_pretrained("/workspace/models/news_roberta_D").to(DEV).eval()
    et = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")
    em = AutoModel.from_pretrained("jhgan/ko-sroberta-multitask").to(DEV).eval()

    @torch.no_grad()
    def dscore(ts):
        o = []
        for i in range(0, len(ts), 32):
            b = dtok(ts[i:i+32], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
            o.extend(torch.softmax(dm(**b).logits, -1)[:, 1].cpu().tolist())
        return o

    @torch.no_grad()
    def embed(ts):
        o = []
        for i in range(0, len(ts), 32):
            b = et(ts[i:i+32], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
            h = em(**b).last_hidden_state; mk = b.attention_mask.unsqueeze(-1).float()
            o.append(F.normalize((h*mk).sum(1)/mk.sum(1), dim=-1).cpu())
        return torch.cat(o)

    xai_emb = embed(xai)
    print("\n" + "=" * 74)
    print(f"eval_collapse · {args.tag} (fusion={kind}, clean20k {args.split} n={len(rows)})")
    print("=" * 74)
    print(f"{'디코딩':<12}{'붕괴율':>8}{'평균최대연속':>12}{'출력길이':>9}{'D평균':>8}{'사람판정':>9}{'의미cos':>9}")
    samples = {}
    for name, dec in DECS.items():
        outs = [gen(m, kind, tok, x, dec) for x in xai]
        cs = collapse_stats(outs)
        d = dscore(outs); dm_ = float(np.mean(d)); human = 100 * float(np.mean(np.array(d) < 0.5))
        cos = float((embed(outs) * xai_emb).sum(-1).mean())
        print(f"{name:<12}{cs['collapse']:>7.0f}%{cs['maxrun']:>12.1f}{cs['olen']:>9.0f}"
              f"{dm_:>8.3f}{human:>8.0f}%{cos:>9.3f}", flush=True)
        samples[name] = outs
    print("=" * 74)

    if args.out:
        with open(args.out, "w") as f:
            for name, outs in samples.items():
                for r, o in zip(rows, outs):
                    f.write(json.dumps({"doc_id": r["id"], "dec": name, "text": o}, ensure_ascii=False) + "\n")
        print(f"샘플 저장: {args.out}")


if __name__ == "__main__":
    main()
