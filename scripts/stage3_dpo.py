#!/usr/bin/env python3
"""MASH Stage 3 — DPO Alignment (StyleBART).

선호쌍 (prompt=x_ai, chosen=x_human, rejected=SFT의 hard neg) 으로 SFT 모델을 정렬한다.
목표: HSR 경로 출력이 x_human 쪽(덜 AI)으로, rejected 쪽에서 멀어지게.

DPO loss (Rafailov et al. 2023):
  L = -log σ( β·[ (logπ(y_w|x) - logπ_ref(y_w|x)) - (logπ(y_l|x) - logπ_ref(y_l|x)) ] )
  · π      = 학습 중인 정책 (StyleBART, HSR 경로)
  · π_ref  = 고정된 SFT 초기본 (참조)
  · logπ(y|x) = HSR 경로 디코더의 시퀀스 로그우도 (pad 제외 합)

trl.DPOTrainer 를 못 쓰는 이유: 모델이 커스텀(StyleBART, encoder_outputs 주입)이라
표준 CausalLM/Seq2Seq 인터페이스가 아니다. 그래서 loss 를 직접 구현한다.

노트 30 §3: reward r(x,y) = -C·D(y). DPO 는 이 reward 를 chosen/rejected 선호로
암묵 표현하므로, D 로 거른 rejected 를 쓰는 것으로 탐지기 신호가 학습에 들어간다.

사용:
  python stage3_dpo.py --dpo-pairs <dpo_pairs.jsonl> --sft <style_bart.pt> \
      --out-dir <ckpt> --beta 0.1 --epochs 1
"""
import argparse, json, copy, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from stage2_sft import StyleBART, MODEL as BART_MODEL

DEV = "cuda:0"


def seq_logprob(model, tok, prompts, targets, maxlen=512, norm=False):
    """HSR 경로에서 target 시퀀스의 로그우도 합 (배치, pad 제외)."""
    enc = tok(prompts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(DEV)
    # 타깃(chosen/rejected)에만 eos 를 붙인다(라벨 역할). 인코더 입력(prompts)엔 안 붙임 —
    # SFT 가 인코더에 eos 없이 학습됐으므로 일관성 유지. stage2 라벨 수정과 같은 취지.
    seqs = [tok(t, truncation=True, max_length=maxlen - 1).input_ids + [tok.eos_token_id]
            for t in targets]
    Lt = max(len(s) for s in seqs)
    lbl = torch.full((len(seqs), Lt), tok.pad_token_id, dtype=torch.long)
    for bi, s in enumerate(seqs):
        lbl[bi, :len(s)] = torch.tensor(s, dtype=torch.long)
    lbl = lbl.to(DEV)
    cr = model.encode(enc.input_ids, enc.attention_mask)
    fused = model.fuse(cr, model.hsr)                 # HSR(human) 경로 (concat fuse)
    dec_in = model.bart.prepare_decoder_input_ids_from_labels(lbl)
    logits = model.bart(attention_mask=enc.attention_mask,
                        encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                        decoder_input_ids=dec_in).logits
    logp = F.log_softmax(logits, -1)
    tok_lp = logp.gather(-1, lbl.unsqueeze(-1)).squeeze(-1)   # (B, T)
    mask = (lbl != tok.pad_token_id).float()
    s = (tok_lp * mask).sum(-1)                               # (B,) 시퀀스 로그우도 합
    return s / mask.sum(-1) if norm else s                    # norm=True → 토큰당 평균(길이정규화, SimPO식)


class DPODS(Dataset):
    def __init__(self, rows): self.r = rows
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        r = self.r[i]; return r["prompt"], r["chosen"], r["rejected"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-pairs", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)      # 논문 미명시 → 표준 DPO 기본
    ap.add_argument("--epochs", type=int, default=5)        # 논문 Appendix B
    ap.add_argument("--bs", type=int, default=2)            # per-device (논문)
    ap.add_argument("--accum", type=int, default=8)         # grad accum → 유효배치 bs×accum=16 (논문)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--length-norm", action="store_true", help="로그우도를 토큰당 평균(SimPO식 길이정규화). 켜면 --beta 를 크게(2~) 줘야 함")
    ap.add_argument("--simpo", action="store_true", help="SimPO: reference-free 길이정규화 + target margin γ (ref 항 제거)")
    ap.add_argument("--gamma", type=float, default=1.0, help="SimPO target reward margin γ")
    ap.add_argument("--rpo-alpha", type=float, default=0.0, help="RPO: chosen 토큰당 NLL 앵커 가중(0=off). pc를 절대적으로 올려 unlearning(rew_c 하락) 방지")
    ap.add_argument("--dpop", action="store_true", help="DPOP: chosen 이 ref 밑으로 떨어질 때만 hinge 페널티 (rew_c<0 직접 방지)")
    ap.add_argument("--dpop-lambda", type=float, default=5.0, help="DPOP 페널티 강도 λ")
    ap.add_argument("--eval-pairs", default=None,
                    help="D_pair.jsonl. 주면 학습 후 dev split 에서 D 점수 전/후 비교")
    ap.add_argument("--roberta", default=None, help="동결 D 디렉토리 (eval 용)")
    ap.add_argument("--seed", type=int, default=42, help="재현용 시드(torch/DataLoader shuffle)")
    args = ap.parse_args()

    import random
    torch.manual_seed(args.seed); random.seed(args.seed)

    # β ↔ length-norm 커플링 가드: length-norm 이면 로그확률이 토큰당 평균이라 β 를 크게(2~) 줘야 하고,
    # 안 켜면 시퀀스 합이라 β 가 작아야(~0.1) 한다. 어긋나면 logsigmoid 포화(gradient≈0)로 조용히 미학습된다.
    # 정본(run_domain_pipeline.sh)은 length-norm + β2.0. 기본값은 vanilla DPO(β0.1)이니 주의.
    if args.length_norm and args.beta < 1.0:
        print(f"⚠️ length-norm 인데 β={args.beta}<1.0 — 신호가 너무 약하다(정본 β=2.0).", flush=True)
    if not args.length_norm and not args.simpo and args.beta > 0.5:
        print(f"⚠️ length-norm 없이 β={args.beta}>0.5 — 시퀀스 합 로그확률이라 gradient 포화 위험. "
              "length-norm 을 켜거나 β 를 낮춰라.", flush=True)

    tok = AutoTokenizer.from_pretrained(BART_MODEL)
    ck = torch.load(args.sft, map_location=DEV)

    policy = StyleBART().to(DEV); policy.load_state_dict(ck["model"]); policy.train()
    ref = StyleBART().to(DEV); ref.load_state_dict(ck["model"]); ref.eval()
    for p in ref.parameters(): p.requires_grad_(False)

    rows = [json.loads(l) for l in open(args.dpo_pairs) if l.strip()]
    print(f"DPO 선호쌍 {len(rows):,}", flush=True)
    dl = DataLoader(DPODS(rows), batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: (list(zip(*b))))
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # grad accumulation: micro-batch bs 를 accum 번 모아 1회 갱신 → 유효배치 bs×accum.
    # 작은 배치의 노이지한 DPO 그래디언트가 정책을 degenerate 로 표류시키는 걸 완화(논문 유효16).
    step = 0; micro = 0
    for ep in range(args.epochs):
        for prompts, chosen, rejected in dl:
            prompts, chosen, rejected = list(prompts), list(chosen), list(rejected)
            nrm = args.length_norm or args.simpo   # SimPO 는 항상 길이정규화
            pc = seq_logprob(policy, tok, prompts, chosen, args.maxlen, norm=nrm)
            pr = seq_logprob(policy, tok, prompts, rejected, args.maxlen, norm=nrm)
            if args.simpo:                         # SimPO: ref 없이 정규화 로그확률 차 - margin γ
                rc = rr = None
                logits = args.beta * (pc - pr) - args.gamma
            else:
                with torch.no_grad():
                    rc = seq_logprob(ref, tok, prompts, chosen, args.maxlen, norm=nrm)
                    rr = seq_logprob(ref, tok, prompts, rejected, args.maxlen, norm=nrm)
                # DPO: chosen 은 ref 대비 올리고 rejected 는 내린다
                h = (pc - rc) - (pr - rr)
                if args.dpop:                    # DPOP: chosen 이 ref 밑(rc>pc)으로 가면 hinge 로 되돌림
                    h = h - args.dpop_lambda * torch.relu(rc - pc)
                logits = args.beta * h
            loss = -F.logsigmoid(logits).mean() / args.accum   # accum 평균 위해 스케일
            if args.rpo_alpha > 0:               # RPO: chosen 토큰당 NLL 을 더해 pc 를 절대적으로 위로 (unlearning 방지)
                pcn = pc if nrm else seq_logprob(policy, tok, prompts, chosen, args.maxlen, norm=True)
                loss = loss + args.rpo_alpha * (-pcn.mean()) / args.accum
            acc = (logits > 0).float().mean()    # chosen 이 rejected 보다 선호되는 비율
            loss.backward(); micro += 1
            if micro % args.accum == 0:           # accum 번째마다 갱신
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    if args.simpo:               # SimPO reward = β·정규화 로그확률(절대). 안 떨어져야 좋음
                        rew_c = (args.beta * pc).mean().item(); rew_r = (args.beta * pr).mean().item()
                    else:                        # DPO reward = β·(policy−ref). 음수=unlearning
                        rew_c = (args.beta * (pc - rc)).mean().item(); rew_r = (args.beta * (pr - rr)).mean().item()
                    print(f"ep{ep} step{step} loss {loss.item()*args.accum:.4f} acc {acc.item():.2f} "
                          f"margin {(pc-pr).mean().item():+.2f} rew_c {rew_c:+.3f} rew_r {rew_r:+.3f}", flush=True)

    if micro % args.accum != 0:                   # 남은 micro-batch 그래디언트 flush (에폭 꼬리 유실 방지)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step(); opt.zero_grad(); step += 1
        print(f"flush: 잔여 그래디언트 적용 (총 step {step})", flush=True)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"model": policy.state_dict(), "config": vars(args)}, f"{args.out_dir}/dpo_bart.pt")
    tok.save_pretrained(args.out_dir)
    print(f"저장: {args.out_dir}/dpo_bart.pt")

    # ── 학습 후 dev 자동 검증 ──
    # DPO 가 정말 D 를 낮췄는가 + 내용을 보존했는가를 학습 직후 한눈에.
    # "reward 는 오르는데 글이 망가지는" 모드를 여기서 잡는다.
    if not (args.eval_pairs and args.roberta):
        print("(--eval-pairs/--roberta 주면 dev 검증 리포트 출력)")
        return
    from transformers import AutoModelForSequenceClassification
    from transformers.modeling_outputs import BaseModelOutput as _BMO
    dev = [json.loads(l) for l in open(args.eval_pairs) if l.strip()]
    dev = [p for p in dev if p.get("split") == "dev"][:120]
    rtok = AutoTokenizer.from_pretrained(args.roberta)
    rob = AutoModelForSequenceClassification.from_pretrained(args.roberta).to(DEV).eval()

    @torch.no_grad()
    def gen_hsr(model, x):
        e = tok(x, return_tensors="pt", truncation=True, max_length=args.maxlen).to(DEV)
        cr = model.encode(e.input_ids, e.attention_mask)
        fused = model.fuse(cr, model.hsr)             # concat fuse
        o = model.bart.generate(encoder_outputs=_BMO(last_hidden_state=fused),
                                attention_mask=e.attention_mask, max_length=args.maxlen,
                                num_beams=4, no_repeat_ngram_size=3)
        return tok.decode(o[0], skip_special_tokens=True)

    @torch.no_grad()
    def dscore(texts):
        e = rtok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        return torch.softmax(rob(**e).logits, -1)[:, 1].cpu().tolist()

    def overlap(a, b):  # 12자 조각 겹침 — 내용 보존 대리지표(높으면 원문 베낌)
        segs = {a[i:i+12] for i in range(max(len(a) - 11, 0))}
        if not segs: return 0.0
        return sum(b.find(s) >= 0 for s in segs) / len(segs)

    policy.eval()
    d_ai, d_sft, d_dpo, ov_sft, ov_dpo, rows = [], [], [], [], [], []
    for p in dev:
        g_sft = gen_hsr(ref, p["ai_text"])      # ref = 학습 전 SFT본
        g_dpo = gen_hsr(policy, p["ai_text"])   # 학습 후
        rows.append({"x_ai": p["ai_text"], "sft": g_sft, "dpo": g_dpo, "x_human": p["human_text"]})
        ov_sft.append(overlap(p["ai_text"], g_sft)); ov_dpo.append(overlap(p["ai_text"], g_dpo))
    ds_all = dscore([p["ai_text"] for p in dev]) + \
             dscore([r["sft"] for r in rows]) + dscore([r["dpo"] for r in rows])
    n = len(dev); d_ai, d_sft, d_dpo = ds_all[:n], ds_all[n:2*n], ds_all[2*n:]
    mean = lambda x: sum(x) / len(x)
    tau = 0.5
    print(f"\n=== DPO dev 검증 (n={n}, τ={tau}) ===")
    print(f"  D(P(AI)) 평균:  x_ai {mean(d_ai):.3f} → SFT {mean(d_sft):.3f} → DPO {mean(d_dpo):.3f}")
    print(f"  human 판정율:   SFT {sum(d<tau for d in d_sft)/n:.0%} → DPO {sum(d<tau for d in d_dpo)/n:.0%}")
    print(f"  원문 겹침(보존): SFT {mean(ov_sft):.2f} → DPO {mean(ov_dpo):.2f}  (과하게 오르면 원문 복붙)")
    json.dump(rows[:12], open(f"{args.out_dir}/dpo_dev_samples.json", "w"), ensure_ascii=False, indent=2)
    print(f"  육안 검수 샘플 12건: {args.out_dir}/dpo_dev_samples.json")


if __name__ == "__main__":
    main()
