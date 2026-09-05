#!/usr/bin/env python3
"""MASH Stage 2 — Style-injection SFT (ko-BART).

MASH 의 이중 경로 구조를 ko-BART 에 얹는다. 논문 식 (4)(5):

  x_ai → 인코더 → CR (content representation, 인코더 last_hidden_state)
    경로1: CR + ASR → fusion → 디코더 → x_ai 복원   L_recon (의미 보존)
    경로2: CR + HSR → fusion → 디코더 → x_human 생성 L_trans (문체 이식)
  L_SFT = λ·L_recon + (1-λ)·L_trans

구현 결정 (논문에 코드가 없어 우리가 정한 것):
  · ASR/HSR = learnable 스타일 벡터 (d_model 차원).
  · fusion = 논문식 concat→proj: content(CR) 과 style(ASR/HSR) 을 concat 후 Linear(2d→d).
    두 경로가 같은 가중치를 쓴다(논문 명시). init 은 항등(fused≈cr)에서 시작해 스타일 기여를
    학습으로 키운다. (이전 additive 형 `cr+fusion(cr+style)` 은 content 를 오염시켜 붕괴 유발.)
  · λ = 0.5 기본. recon(의미) vs trans(문체) 균형.

장시간(clean20k 16k×수ep) 대비 epoch 별 last.pt 저장 + --resume 지원(사용자 요구: 안 끊기게).

두 경로 모두 디코더 입력은 teacher forcing 이다. 경로1 정답=x_ai, 경로2 정답=x_human.
추론 시엔 HSR 경로만 쓴다(x_ai → x_human 스타일).

사용:
  python stage2_sft.py --pairs <D_pair.jsonl> --out-dir <ckpt> --epochs 3
"""
import argparse, json, math
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration

MODEL = "gogamza/kobart-base-v2"


class StyleBART(nn.Module):
    """ko-BART + learnable ASR/HSR 스타일 벡터 + 공유 fusion."""
    def __init__(self):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(MODEL)
        d = self.bart.config.d_model
        self.asr = nn.Parameter(torch.zeros(d))   # AI style
        self.hsr = nn.Parameter(torch.zeros(d))   # human style
        nn.init.normal_(self.asr, std=0.02); nn.init.normal_(self.hsr, std=0.02)
        # 논문식 fusion: H_fused = W_p·[content ; style] + b_p — content 와 style 을
        # concat 후 projection(2d→d). 이전 구현은 cr+style 로 "더해서" 넣었는데, 그러면
        # content 표현이 오염돼 디코더 cross-attention 이 무너지고 같은 토큰을 반복했다
        # (SFT 30%·DPO 53% 붕괴). concat 은 둘을 분리 유지해 projection 이 섞는 법을 배운다.
        self.fusion = nn.Linear(2 * d, d)
        # 초기엔 항등(fused≈cr): content 절반=I, style 절반=0, bias=0. 스타일 기여는 학습으로.
        with torch.no_grad():
            self.fusion.weight.zero_(); self.fusion.weight[:, :d].copy_(torch.eye(d))
            self.fusion.bias.zero_()

    def encode(self, input_ids, attention_mask):
        return self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def fuse(self, cr, style):
        # 논문식 concat→proj. style(d,) 를 (B,S,d) 로 broadcast 해 content 와 이어붙인다.
        style_b = style.view(1, 1, -1).expand(cr.size(0), cr.size(1), -1)
        return self.fusion(torch.cat([cr, style_b], dim=-1))

    def path(self, cr, enc_mask, style, labels):
        # labels 만 넘기면 BART 가 내부에서 decoder_input_ids 를 자동 shift 한다.
        fused = self.fuse(cr, style)
        from transformers.modeling_outputs import BaseModelOutput
        out = self.bart(attention_mask=enc_mask,
                        encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                        labels=labels)
        return out.loss

    def forward(self, x_ai_ids, x_ai_mask, ai_lbl, hu_lbl, lam=0.5):
        cr = self.encode(x_ai_ids, x_ai_mask)
        l_recon = self.path(cr, x_ai_mask, self.asr, ai_lbl)   # x_ai 복원
        l_trans = self.path(cr, x_ai_mask, self.hsr, hu_lbl)   # x_human 생성
        return lam * l_recon + (1 - lam) * l_trans, l_recon, l_trans


class PairDS(Dataset):
    def __init__(self, pairs, tok, maxlen=512):
        self.p = pairs; self.tok = tok; self.maxlen = maxlen

    def __len__(self): return len(self.p)

    def __getitem__(self, i):
        r = self.p[i]
        return r["ai_text"], r["human_text"]


def collate(batch, tok, maxlen):
    ai = [b[0] for b in batch]; hu = [b[1] for b in batch]
    enc = tok(ai, padding=True, truncation=True, max_length=maxlen, return_tensors="pt")

    def labels(texts):
        # ko-BART 토크나이저는 eos(</s>) 를 자동으로 안 붙인다. 라벨 끝에 eos 를 직접 넣어야
        # 모델이 "멈추는 법"(eos 방출)을 배운다. 안 붙이면 추론 때 max_length 까지 생성하며
        # 같은 문장을 반복 채운다(길이 폭주·문장 반복). eos 를 loss 대상에 포함시켜야 함.
        seqs = [tok(t, truncation=True, max_length=maxlen - 1).input_ids + [tok.eos_token_id]
                for t in texts]
        L = max(len(s) for s in seqs)
        lbl = torch.full((len(seqs), L), -100, dtype=torch.long)   # pad 자리 = -100 (loss 제외)
        for i, s in enumerate(seqs):
            lbl[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return lbl

    return enc.input_ids, enc.attention_mask, labels(ai), labels(hu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)   # 원논문 값
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="train", help="이 split 만 학습 (train/dev/test)")
    ap.add_argument("--resume", action="store_true", help="out-dir/last.pt 있으면 이어서 학습")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    pairs = [json.loads(l) for l in open(args.pairs) if l.strip()]
    if args.split:
        pairs = [p for p in pairs if p.get("split", "train") == args.split]
    print(f"학습 쌍 {len(pairs):,} ({args.split})", flush=True)

    import os, traceback
    os.makedirs(args.out_dir, exist_ok=True)
    last_path = os.path.join(args.out_dir, "last.pt")
    done_f, fail_f = os.path.join(args.out_dir, "SFT_DONE"), os.path.join(args.out_dir, "SFT_FAIL")
    for f in (done_f, fail_f):
        if os.path.exists(f): os.remove(f)

    model = StyleBART().to(args.device)
    dl = DataLoader(PairDS(pairs, tok), batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, tok, args.maxlen))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.06)

    start_ep, step = 0, 0
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=args.device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, step = ck["epoch"] + 1, ck["step"]
        print(f"resume: epoch {start_ep} 부터 (step {step}/{total})", flush=True)

    try:
        for ep in range(start_ep, args.epochs):
            model.train()
            for batch in dl:
                xi, xm, al, hl = [t.to(args.device) for t in batch]
                loss, lr_, lt_ = model(xi, xm, al, hl, lam=args.lam)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"ep{ep} step{step}/{total} loss {loss.item():.3f} "
                          f"(recon {lr_.item():.3f} trans {lt_.item():.3f})", flush=True)
            # epoch 체크포인트 (중단 대비 resume 용)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": ep, "step": step,
                        "config": vars(args)}, last_path)
            print(f"[ckpt] epoch {ep} 저장 → {last_path}", flush=True)
    except Exception:
        with open(fail_f, "w") as f: f.write(traceback.format_exc())
        print("학습 실패 — traceback 을 SFT_FAIL 에 기록", flush=True); raise

    torch.save({"model": model.state_dict(), "config": vars(args)}, f"{args.out_dir}/style_bart.pt")
    tok.save_pretrained(args.out_dir)
    open(done_f, "w").write("ok")
    print(f"저장: {args.out_dir}/style_bart.pt · SFT_DONE", flush=True)


if __name__ == "__main__":
    main()
