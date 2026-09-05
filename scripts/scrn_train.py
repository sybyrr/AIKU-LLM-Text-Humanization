#!/usr/bin/env python3
"""SCRN (Siamese Calibrated Reconstruction Network) — 한국어 신문 트랙, 전이-평가 탐지기(역할 ②).

논문: "Are AI-Generated Text Detectors Robust to Adversarial Perturbations?" (ACL 2024, arXiv 2406.01179).
우리 파이프라인 D(klue-roberta-base)와 **다른 계열/더 큰** xlm-roberta-large 백본 → 강건성·도메인일반화·독립성.

구조(β-VAE 해석 — 논문 "노이즈 주입→복원"이 reparameterization trick과 동치):
  h = encoder(x)[:,0]                      # CLS pooled (1024d)
  z_s = enc_s(h); z_p = enc_p(h)           # 평균 / 로그분산 (각 dz=512)
  갈래 k=1,2: eps_k~N(0,I); z̃_k = z_s + eps_k·exp(½ z_p)
             recon_k = dec(z̃_k)→h ,  logits_k = clf(z̃_k)
손실:
  L_cls = CE(l1,y)+CE(l2,y)
  L_re  = MSE(r1,h⊥)+MSE(r2,h⊥) + β·KL_vae(z_s,z_p)     # β=0.5, KL이 z_p 퇴화 방지
  L_sc  = symKL(softmax l1, softmax l2)                  # 두 노이즈에도 같은 확신 → 강건
  L = λ1 L_cls + λ2 L_re + λ3 L_sc                        # 0.5, 0.01, 0.5 (논문)
추론: 노이즈 없이 z_s → clf → P(AI) ("클수록 AI" = Binoculars와 같은 방향).

데이터: news_dpair_clean20k.jsonl (train: human=0, ai=1 / dev로 best 선택).
안전: epoch 체크포인트+resume, 200스텝 로그, best-dev 저장, SCRN_DONE/SCRN_FAIL 센티넬.
사용: python scrn_train.py --epochs 3 --bs 8 --accum 2
"""
import argparse, json, time, random, pathlib, traceback
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

NT = "/workspace/dataset/news_track"
DEV = "cuda:0"
BACKBONE = "monologg/koelectra-base-v3-discriminator"   # ELECTRA base(110M): D(klue-roberta)와 다른 구조=독립성↑, TITAN Xp에 빠름


# ── 데이터: clean20k 의 human/ai 를 각각 라벨 0/1 텍스트로 펼침 ──
class PairTextDS(Dataset):
    def __init__(self, path, split):
        self.rows = []
        for l in open(path):
            r = json.loads(l)
            if r["split"] != split:
                continue
            self.rows.append((r["human_text"], 0))
            self.rows.append((r["ai_text"], 1))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def make_collate(tok, max_len):
    def collate(batch):
        texts = [b[0] for b in batch]; ys = torch.tensor([b[1] for b in batch])
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return enc, ys
    return collate


# ── 모델 ──
class SCRN(nn.Module):
    def __init__(self, backbone=BACKBONE, dz=512):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone)
        d = self.enc.config.hidden_size            # 1024
        self.dz = dz
        self.enc_s = nn.Linear(d, dz)              # 평균
        self.enc_p = nn.Linear(d, dz)              # 로그분산
        self.dec = nn.Sequential(nn.Linear(dz, d), nn.GELU(), nn.Linear(d, d))
        self.clf = nn.Sequential(nn.Linear(dz, dz), nn.GELU(), nn.Dropout(0.1), nn.Linear(dz, 2))

    def represent(self, enc):
        return self.enc(**enc).last_hidden_state[:, 0]     # CLS (B,d)

    def latent(self, h):
        return self.enc_s(h), self.enc_p(h)                # z_s, z_p(logvar)

    def branch(self, z_s, z_p):
        z = z_s + torch.randn_like(z_s) * torch.exp(0.5 * z_p)
        return self.clf(z), self.dec(z)                    # logits, recon

    @torch.no_grad()
    def score(self, enc):
        """추론: 노이즈 없이 z_s → P(AI)."""
        z_s, _ = self.latent(self.represent(enc))
        return torch.softmax(self.clf(z_s), -1)[:, 1]


def vae_kl(z_s, z_p):
    # KL( N(z_s, exp(z_p)) || N(0,I) ) 평균 — z_p(로그분산) 퇴화 방지
    return (-0.5 * (1 + z_p - z_s.pow(2) - z_p.exp()).sum(1)).mean()

def sym_kl(l1, l2):
    p, q = F.log_softmax(l1, -1), F.log_softmax(l2, -1)
    P, Q = p.exp(), q.exp()
    return 0.5 * ((P * (p - q)).sum(1) + (Q * (q - p)).sum(1)).mean()


def metrics(ys, ps):
    """AUROC(tie-aware rank-sum) + F1@0.5 + 인간오탐(FPR)."""
    n1 = sum(1 for y in ys if y == 1); n0 = len(ys) - n1
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    ranks = [0.0] * len(ps); j = 0
    while j < len(order):
        k = j
        while k + 1 < len(order) and ps[order[k + 1]] == ps[order[j]]:
            k += 1
        avg = (j + k) / 2 + 1
        for t in range(j, k + 1):
            ranks[order[t]] = avg
        j = k + 1
    sum_pos = sum(ranks[i] for i in range(len(ps)) if ys[i] == 1)
    auroc = (sum_pos - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")
    tp = sum(1 for p, y in zip(ps, ys) if y == 1 and p >= 0.5)
    fp = sum(1 for p, y in zip(ps, ys) if y == 0 and p >= 0.5)
    fn = sum(1 for p, y in zip(ps, ys) if y == 1 and p < 0.5)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / n0 if n0 else 0.0
    return auroc, f1, fpr


@torch.no_grad()
def eval_split(model, dl):
    model.eval(); ys, ps = [], []
    for enc, y in dl:
        enc = {k: v.to(DEV) for k, v in enc.items()}
        ps += model.score(enc).float().cpu().tolist(); ys += y.tolist()
    return metrics(ys, ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=f"{NT}/news_dpair_clean20k.jsonl")
    ap.add_argument("--out-dir", default="/workspace/models/news_scrn")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--enc-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--l1", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--l3", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    done, fail = out / "SCRN_DONE", out / "SCRN_FAIL"
    for s in (done, fail):
        if s.exists(): s.unlink()
    def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    try:
        torch.manual_seed(0); random.seed(0)
        tok = AutoTokenizer.from_pretrained(BACKBONE)
        cf = make_collate(tok, args.max_len)
        tr = PairTextDS(args.pairs, "train"); dv = PairTextDS(args.pairs, "dev")
        log(f"train {len(tr):,} texts · dev {len(dv):,} texts · backbone {BACKBONE}")
        tdl = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=cf, num_workers=4, drop_last=True)
        ddl = DataLoader(dv, batch_size=32, shuffle=False, collate_fn=cf, num_workers=2)

        model = SCRN().to(DEV)
        # base(110M)라 12GB에 여유 → gradient checkpointing 불필요(속도 우선)
        enc_params = list(model.enc.parameters())
        head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
        opt = torch.optim.AdamW([
            {"params": enc_params, "lr": args.enc_lr},
            {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
        steps = (len(tdl) // args.accum) * args.epochs
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.cuda.amp.GradScaler()

        start_ep, best = 0, -1.0
        ck = out / "last.pt"
        if ck.exists():
            st = torch.load(ck, map_location=DEV)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            sch.load_state_dict(st["sch"]); scaler.load_state_dict(st["scaler"])
            start_ep = st["epoch"] + 1; best = st.get("best", -1.0)
            log(f"resume: epoch {start_ep} 부터, best AUROC {best:.4f}")

        for ep in range(start_ep, args.epochs):
            model.train(); t0 = time.time(); run = {"cls": 0, "re": 0, "sc": 0}; opt.zero_grad()
            for i, (enc, y) in enumerate(tdl):
                enc = {k: v.to(DEV) for k, v in enc.items()}; y = y.to(DEV)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    h = model.represent(enc)
                    z_s, z_p = model.latent(h)
                    l1, r1 = model.branch(z_s, z_p)
                    l2, r2 = model.branch(z_s, z_p)
                    L_cls = F.cross_entropy(l1, y) + F.cross_entropy(l2, y)
                    ht = h.detach()
                    L_re = F.mse_loss(r1, ht) + F.mse_loss(r2, ht) + args.beta * vae_kl(z_s, z_p)
                    L_sc = sym_kl(l1, l2)
                    loss = (args.l1 * L_cls + args.l2 * L_re + args.l3 * L_sc) / args.accum
                scaler.scale(loss).backward()
                if (i + 1) % args.accum == 0:
                    scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
                run["cls"] += float(L_cls); run["re"] += float(L_re); run["sc"] += float(L_sc)
                if (i + 1) % 200 == 0:
                    sp = (i + 1) / (time.time() - t0)
                    log(f"ep{ep} {i+1}/{len(tdl)} | cls {run['cls']/200:.3f} re {run['re']/200:.3f} "
                        f"sc {run['sc']/200:.4f} | {sp:.1f} it/s")
                    run = {"cls": 0, "re": 0, "sc": 0}
            auroc, f1, fpr = eval_split(model, ddl)
            log(f"== epoch {ep} 끝 · dev AUROC {auroc:.4f} F1 {f1:.4f} 인간오탐 {fpr:.3f} ==")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sch": sch.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": ep, "best": best}, ck)
            if auroc > best:
                best = auroc
                torch.save({"model": model.state_dict(), "backbone": BACKBONE, "dz": model.dz,
                            "dev_auroc": auroc, "dev_f1": f1, "dev_fpr": fpr}, out / "scrn_best.pt")
                log(f"  ↑ best 갱신 → scrn_best.pt (AUROC {auroc:.4f})")

        json.dump({"backbone": BACKBONE, "best_dev_auroc": best}, open(out / "meta.json", "w"))
        done.touch()
        log(f"완료 · best dev AUROC {best:.4f} → {out}/scrn_best.pt")
    except Exception:
        fail.write_text(traceback.format_exc())
        log("실패:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
