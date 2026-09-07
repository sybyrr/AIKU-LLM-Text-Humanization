"""DPO — 순수 torch 수동 구현 (trl 미사용).

이유: ① TITAN Xp(sm_61) 폴백 시 torch 를 내리면 trl·peft 버전이 연쇄로 내려가는
지옥을 피한다 ② StyleParaphraser(스타일 임베딩 + fusion) 같은 커스텀 모델은
어차피 표준 트레이너에 안 맞는다. 손실 자체는 7줄이다.

  loss = -log σ( β·[(logπ(y_w|x) − logπ_ref(y_w|x)) − (logπ(y_l|x) − logπ_ref(y_l|x))] )

ref logp 는 r_build_prefs.py 가 G_{t-1}(=정책 초기값)로 미리 계산해 prefs 에 박아 둔다
— 학습 중 ref 모델을 메모리에 올릴 필요가 없다.
설정: lr 5e-6, β=0.1, 5 epoch (notes/31 § Round). y_w 는 arm 별로 다르다 (인간앵커/자기앵커).
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import io_utils
from .paraphraser import STYLE_HUMAN


class PrefDataset(Dataset):
    def __init__(self, prefs):
        self.rows = prefs

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collate(tok, para_cfg):
    def to_labels(texts):
        # ko-BART tokenizer 는 EOS 를 자동으로 붙이지 않는다. 팀 SFT/DPO와 동일하게
        # EOS를 명시해 종료 학습을 보존하고, 나머지 pad만 -100으로 가린다.
        seqs = [tok(t, truncation=True, max_length=para_cfg.max_tgt_len - 1).input_ids
                for t in texts]
        seqs = [s if s and s[-1] == tok.eos_token_id else s + [tok.eos_token_id] for s in seqs]
        width = max(len(s) for s in seqs)
        lab = torch.full((len(seqs), width), -100, dtype=torch.long)
        for i, seq in enumerate(seqs):
            lab[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return lab

    def collate(batch):
        enc = tok([b["x_ai"] for b in batch], truncation=True, max_length=para_cfg.max_src_len,
                  padding=True, return_tensors="pt")
        lab_w = to_labels([b["y_w"] for b in batch])
        lab_l = to_labels([b["y_l"] for b in batch])
        ref_w = torch.tensor([b["ref_logp_w"] for b in batch], dtype=torch.float32)
        ref_l = torch.tensor([b["ref_logp_l"] for b in batch], dtype=torch.float32)
        return enc, lab_w, lab_l, ref_w, ref_l

    return collate


def run_dpo(cfg, model, tok, prefs, out_dir, device, epochs=None, log_prefix="dpo"):
    from . import paraphraser as P

    assert prefs, "preference pair 가 0건 — r_build_prefs 산출물 확인"
    d = cfg.dpo
    epochs = epochs or d.epochs
    model.to(device)

    ds = PrefDataset(prefs)
    g = torch.Generator().manual_seed(cfg.seed)
    dl = DataLoader(ds, batch_size=d.batch_size, shuffle=True, generator=g,
                    collate_fn=make_collate(tok, cfg.paraphraser))

    optim = torch.optim.AdamW(model.parameters(), lr=d.lr)
    length_norm = bool(d.get("length_norm", False))
    dpop = bool(d.get("dpop", False))
    dpop_lambda = float(d.get("dpop_lambda", 5.0))
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, tot_acc, tot_margin, seen = 0.0, 0.0, 0.0, 0
        optim.zero_grad()
        for i, (enc, lab_w, lab_l, ref_w, ref_l) in enumerate(dl):
            style = torch.full((lab_w.size(0),), STYLE_HUMAN, dtype=torch.long, device=device)
            pol_w, pol_l = model.pair_logprobs(
                enc["input_ids"].to(device), enc["attention_mask"].to(device), style,
                lab_w.to(device), lab_l.to(device), normalize=length_norm,
            )
            ref_w, ref_l = ref_w.to(device), ref_l.to(device)
            margin = (pol_w - ref_w) - (pol_l - ref_l)
            if dpop:
                margin = margin - dpop_lambda * torch.relu(ref_w - pol_w)
            loss = -F.logsigmoid(d.beta * margin).mean()
            (loss / d.grad_accum).backward()
            if (i + 1) % d.grad_accum == 0 or (i + 1) == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), d.grad_clip)
                optim.step()
                optim.zero_grad()
            bs = lab_w.size(0)
            tot_loss += loss.item() * bs
            tot_acc += (margin > 0).float().sum().item()
            tot_margin += margin.sum().item()
            seen += bs
        rec = {"epoch": ep, "loss": round(tot_loss / seen, 4),
               "pref_acc": round(tot_acc / seen, 4), "margin": round(tot_margin / seen, 3)}
        history.append(rec)
        print(f"[{log_prefix}] {rec}", flush=True)

    P.save(model, tok, out_dir / "final", {"epochs": epochs, "n_prefs": len(prefs),
                                           "beta": d.beta, "lr": d.lr,
                                           "length_norm": length_norm, "dpop": dpop,
                                           "dpop_lambda": dpop_lambda if dpop else None})
    io_utils.write_json(out_dir / "history.json", history)
    return history
