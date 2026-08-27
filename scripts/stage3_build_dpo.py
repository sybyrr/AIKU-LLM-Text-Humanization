#!/usr/bin/env python3
"""MASH Stage 3 준비 — DPO 선호쌍 구축.

논문 방식(노트 30 §3):
  prompt   x   = x_ai
  chosen   y_w = x_human            (Stage 1 에서 확보한 실제 사람 글)
  rejected y_l = π_SFT(·|x_ai, HSR) 샘플 중 D(y_l) > τ 인 것만 (hard negative)

즉 SFT 모델이 human 스타일로 생성했는데도 탐지기(RoBERTa)가 여전히 AI 로 잡는
"틀린" 출력을 rejected 로 쓴다. 추가 데이터 수집 없이 SFT 실패 사례를 재활용.

D(=RoBERTa 게이트)로 채점하는 것이 핵심 — 게이트/보상/평가 탐지기가 같아야 한다.
τ 미만(이미 human 판정)인 샘플은 hard negative 가 아니므로 버린다.

출력: {prompt, chosen, rejected, d_rejected} jsonl → stage3_dpo.py 가 학습.

사용:
  python stage3_build_dpo.py --sft <style_bart.pt> --pairs <D_pair.jsonl> \
      --roberta <ft_model dir> --tau 0.5 --out <dpo_pairs.jsonl> \
      --n-sample 4   # x_ai 당 후보 몇 개 뽑을지
"""
import argparse, json, pathlib, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import BaseModelOutput
from stage2_sft import StyleBART, MODEL as BART_MODEL

DEV = "cuda:0"


def load_sft(path):
    ck = torch.load(path, map_location=DEV)
    m = StyleBART().to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    return m


@torch.no_grad()
def gen_human_style(sft, tok, x_ais, n_sample, maxlen=512):
    """SFT HSR 경로로 human 스타일 후보 생성 (배치). x_ais 리스트 → 입력별 후보 리스트.
    배치로 묶어 한 번에 generate → GPU 활용률↑ (건별 호출의 언배치 오버헤드 제거)."""
    enc = tok(x_ais, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(DEV)
    cr = sft.encode(enc.input_ids, enc.attention_mask)
    fused = sft.fuse(cr, sft.hsr)   # concat fuse
    out = sft.bart.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=fused),
        attention_mask=enc.attention_mask, max_length=maxlen,
        do_sample=True, top_p=0.95, temperature=1.0,
        num_return_sequences=n_sample, no_repeat_ngram_size=3)
    texts = [tok.decode(o, skip_special_tokens=True) for o in out]
    # num_return_sequences 는 입력별로 그룹화됨: 입력 i → texts[i*k:(i+1)*k]
    return [texts[i * n_sample:(i + 1) * n_sample] for i in range(len(x_ais))]


@torch.no_grad()
def d_score(rob, rtok, texts):
    """RoBERTa 게이트 P(AI). 게이트와 동일한 탐지기 — 절단(512)도 D 학습과 일치."""
    enc = rtok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
    return torch.softmax(rob(**enc).logits, -1)[:, 1].cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--roberta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.5, help="D>τ 인 것만 rejected(hard neg)")
    ap.add_argument("--n-sample", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32, help="한 번에 생성할 x_ai 개수(배치)")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BART_MODEL)
    sft = load_sft(args.sft)
    rtok = AutoTokenizer.from_pretrained(args.roberta)
    rob = AutoModelForSequenceClassification.from_pretrained(args.roberta).to(DEV).eval()

    pairs = [json.loads(l) for l in open(args.pairs) if l.strip()]
    if args.split:
        pairs = [p for p in pairs if p.get("split", "train") == args.split]

    # 재개: 이미 출력에 있는 id 는 건너뛰고 이어쓴다 (중단 시 유실 방지).
    # 주의: hard neg 없어 스킵됐던 쌍은 기록이 없어 재시도된다 — 재샘플링이라
    # 이번엔 hard neg 가 나올 수도 있고, 비용도 작으므로 그대로 둔다.
    done = set()
    if pathlib.Path(args.out).exists():
        done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
        print(f"재개: 기존 선호쌍 {len(done)}건 건너뜀", flush=True)

    # id 폴백을 인덱스로 고정(배치 후에도 안정적), done 제외하고 처리 대상만
    for idx, p in enumerate(pairs):
        p.setdefault("_idx", idx)
    todo = [p for p in pairs if p.get("id", p["_idx"]) not in done]
    print(f"처리 대상 {len(todo):,} · 배치 {args.batch} · n_sample {args.n_sample}", flush=True)

    n_out, n_no_hard = len(done), 0
    with open(args.out, "a") as f:
        for bs in range(0, len(todo), args.batch):
            batch = todo[bs:bs + args.batch]
            xais = [p["ai_text"] for p in batch]
            cand_lists = gen_human_style(sft, tok, xais, args.n_sample)   # 입력별 후보 리스트
            flat = [c for cl in cand_lists for c in cl]
            ds_flat = d_score(rob, rtok, flat)                            # 배치 채점
            k = args.n_sample
            for bi, p in enumerate(batch):
                cands = cand_lists[bi]
                ds = ds_flat[bi * k:(bi + 1) * k]
                # D>τ 인 것 중 가장 AI 같은(=가장 나쁜) 것을 hard negative 로
                hard = [(c, d) for c, d in zip(cands, ds) if d > args.tau]
                if not hard:
                    n_no_hard += 1
                    continue
                rej, drej = max(hard, key=lambda x: x[1])
                f.write(json.dumps({
                    "prompt": p["ai_text"],
                    "chosen": p["human_text"],
                    "rejected": rej,
                    "d_rejected": round(drej, 4),
                    "id": p.get("id", p["_idx"]),
                }, ensure_ascii=False) + "\n")
                n_out += 1
            f.flush()   # 재개가 정확하려면 배치 단위로 디스크에 반영
            done_n = min(bs + args.batch, len(todo))
            if (bs // args.batch) % 5 == 0 or done_n >= len(todo):
                print(f"  {done_n}/{len(todo)} · 선호쌍 {n_out} · hard neg 없음 {n_no_hard}", flush=True)

    print(f"\nDPO 선호쌍 {n_out} (hard neg 없어 스킵 {n_no_hard}) → {args.out}")
    print("hard neg 없음 = SFT 가 이미 human 으로 잘 바꾼 케이스(τ 미만). 많으면 SFT 가 좋다는 뜻.")


if __name__ == "__main__":
    main()
