"""Stage 2 — Style-injection SFT (1회만 실행. 라운드마다가 아니다 — notes/31).

L = λ·L_recon + (1−λ)·L_trans
  · 문서마다 두 예제를 만든다: (x_ai, s_ai → x_ai) 와 (x_ai, s_human → x_human)
  · 두 예제가 1:1 이므로 per-sample 가중치 recon=2λ, trans=2(1−λ) 를 곱한
    배치 평균이 정확히 λ 보간이 된다
  · early stopping 은 dev 의 trans CE (우리가 실제로 쓰는 경로) 기준
"""
import math
import random

import torch
from torch.utils.data import DataLoader, Dataset

from . import io_utils
from .paraphraser import STYLE_AI, STYLE_HUMAN


class SftDataset(Dataset):
    """[(kind, src, tgt, style_id)] — kind ∈ {recon, trans}"""

    def __init__(self, dpair_rows, include_recon=True):
        self.rows = []
        for r in dpair_rows:
            self.rows.append(("trans", r["ai_text"], r["human_text"], STYLE_HUMAN))
            if include_recon:
                self.rows.append(("recon", r["ai_text"], r["ai_text"], STYLE_AI))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collate(tok, para_cfg, lambda_recon):
    w_recon, w_trans = 2.0 * lambda_recon, 2.0 * (1.0 - lambda_recon)

    def collate(batch):
        srcs = [b[1] for b in batch]
        tgts = [b[2] for b in batch]
        styles = torch.tensor([b[3] for b in batch], dtype=torch.long)
        weights = torch.tensor([w_recon if b[0] == "recon" else w_trans for b in batch])
        enc = tok(srcs, truncation=True, max_length=para_cfg.max_src_len, padding=True,
                  return_tensors="pt")
        lab = tok(text_target=tgts, truncation=True, max_length=para_cfg.max_tgt_len,
                  padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        return enc, styles, lab, weights

    return collate


@torch.no_grad()
def eval_trans_ce(model, tok, dpair_rows, para_cfg, device, batch_size=8):
    """dev 문서들의 trans 경로 토큰평균 CE."""
    model.eval()
    ds = SftDataset(dpair_rows, include_recon=False)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=make_collate(tok, para_cfg, 0.5))
    total, n = 0.0, 0
    for enc, styles, lab, _w in dl:
        ce = model.label_scores(enc["input_ids"].to(device), enc["attention_mask"].to(device),
                                styles.to(device), lab.to(device), reduce="mean")
        total += ce.sum().item()
        n += len(ce)
    return total / max(1, n)


def run_sft(cfg, model, tok, train_rows, dev_rows, out_dir, device, epochs=None, log_prefix="sft"):
    """학습 후 best/(dev trans CE 최소) 와 last/ 를 out_dir 에 저장. history 반환."""
    from . import paraphraser as P

    sft = cfg.sft
    epochs = epochs or sft.epochs
    model.to(device)

    ds = SftDataset(train_rows)
    collate = make_collate(tok, cfg.paraphraser, sft.lambda_recon)
    g = torch.Generator().manual_seed(cfg.seed)
    dl = DataLoader(ds, batch_size=sft.batch_size, shuffle=True, generator=g, collate_fn=collate)

    optim = torch.optim.AdamW(model.parameters(), lr=sft.lr)
    total_steps = len(dl) * epochs
    warmup = max(1, int(total_steps * sft.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    )

    best = {"ce": math.inf, "epoch": -1}
    bad, history = 0, []
    for ep in range(1, epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for enc, styles, lab, w in dl:
            ce = model.label_scores(enc["input_ids"].to(device), enc["attention_mask"].to(device),
                                    styles.to(device), lab.to(device), reduce="mean")
            loss = (ce * w.to(device)).mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), sft.grad_clip)
            optim.step()
            sched.step()
            tot += loss.item() * len(ce)
            seen += len(ce)

        dev_ce = eval_trans_ce(model, tok, dev_rows, cfg.paraphraser, device, sft.batch_size) if dev_rows else float("nan")
        rec = {"epoch": ep, "train_loss": round(tot / max(1, seen), 4), "dev_trans_ce": round(dev_ce, 4)}
        history.append(rec)
        print(f"[{log_prefix}] {rec}", flush=True)

        P.save(model, tok, out_dir / "last", {"epoch": ep, "dev_trans_ce": dev_ce})
        if dev_rows and dev_ce < best["ce"] - 1e-4:
            best = {"ce": dev_ce, "epoch": ep}
            bad = 0
            P.save(model, tok, out_dir / "best", {"epoch": ep, "dev_trans_ce": dev_ce})
        elif dev_rows:
            bad += 1
            if bad >= sft.patience:
                print(f"[{log_prefix}] early stop — dev trans CE {sft.patience}에폭 개선 없음 "
                      f"(best {best['ce']:.4f} @ ep{best['epoch']})", flush=True)
                break
    if not dev_rows:
        P.save(model, tok, out_dir / "best", {"epoch": epochs, "note": "dev 없음 — last 와 동일"})

    history.append({"best_epoch": best["epoch"], "best_dev_trans_ce": best["ce"]})
    io_utils.write_json(out_dir / "history.json", history)
    return history
