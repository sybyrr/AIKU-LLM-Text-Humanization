"""대리 탐지기 — KLUE-RoBERTa(및 앵커 KoELECTRA) 학습·추론·τ 캘리브레이션·k-fold OOF.

설계 근거 (notes/31 § Stage 0):
- 라벨은 provenance (인간 0 / AI 1) — 카피킬러 점수가 아니다
- τ = 고정 인간 dev-cal 에서 FPR 5% 지점 (0.5 가 아니다)
- train split 자체의 점수는 반드시 k-fold OOF 로 뽑는다 — D₀ 가 자기 학습 데이터를
  채점하면 Stage 1 필터와 DPO hard negative 선별이 동시에 오염된다
- Pascal(sm_61) 이라 전 과정 fp32, AMP 없음
"""
import copy
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import io_utils


class TextClsDataset(Dataset):
    def __init__(self, texts, labels):
        assert len(texts) == len(labels)
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate(tokenizer, max_len):
    def collate(batch):
        texts = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return enc, labels

    return collate


def make_factory(model_name, num_labels=2, tiny=False, init_from=None):
    """(model, tokenizer) 를 만드는 팩토리.

    tiny=True 는 스모크용 — 실제 토크나이저 + 초소형 랜덤 가중치.
    init_from 은 continual arm 용 — 이전 라운드 탐지기에서 이어서 학습.
    """

    def factory():
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(init_from) if init_from else model_name)
        if tiny:
            c = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            for k, v in dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                             intermediate_size=128).items():
                setattr(c, k, v)
            if hasattr(c, "embedding_size"):  # electra 계열
                c.embedding_size = 64
            model = AutoModelForSequenceClassification.from_config(c)
        elif init_from:
            model = AutoModelForSequenceClassification.from_pretrained(str(init_from))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        return model, tok

    return factory


def _linear_schedule(optimizer, total_steps, warmup_ratio):
    warmup = max(1, int(total_steps * warmup_ratio))

    def f(step):
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_detector(
    factory,
    train_texts,
    train_labels,
    det_cfg,
    device,
    dev_texts=None,
    dev_labels=None,
    sample_weights=None,
    epochs=None,
    select_best=True,
    log_prefix="detector",
):
    """학습 후 (model, tokenizer, history) 반환.

    - dev 가 있고 select_best=True 면 에폭별 dev AUC 최고 가중치로 되돌린다.
    - OOF fold 학습은 select_best=False 로 고정 에폭을 쓴다 (fold 간 균일성).
    - sample_weights 가 있으면 WeightedRandomSampler (replay buffer 의 class-balance).
    """
    model, tok = factory()
    model.to(device)
    epochs = epochs or det_cfg.epochs
    bs, accum, max_len = det_cfg.batch_size, det_cfg.grad_accum, det_cfg.max_len

    ds = TextClsDataset(train_texts, train_labels)
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(ds), replacement=True
        )
        dl = DataLoader(ds, batch_size=bs, sampler=sampler, collate_fn=make_collate(tok, max_len))
    else:
        g = torch.Generator().manual_seed(0)
        dl = DataLoader(ds, batch_size=bs, shuffle=True, generator=g, collate_fn=make_collate(tok, max_len))

    steps_per_epoch = math.ceil(len(dl) / accum)
    optim = torch.optim.AdamW(model.parameters(), lr=det_cfg.lr, weight_decay=det_cfg.weight_decay)
    sched = _linear_schedule(optim, steps_per_epoch * epochs, det_cfg.warmup_ratio)

    history, best = [], {"auc": -1.0, "state": None, "epoch": -1}
    for ep in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        optim.zero_grad()
        for i, (enc, labels) in enumerate(dl):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            out = model(**enc, labels=labels)
            (out.loss / accum).backward()
            total += out.loss.item() * len(labels)
            seen += len(labels)
            if (i + 1) % accum == 0 or (i + 1) == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
        rec = {"epoch": ep, "train_loss": round(total / max(1, seen), 4)}

        if dev_texts is not None:
            scores = predict_scores(model, tok, dev_texts, device, max_len=max_len)
            rec["dev_auc"] = round(auc_of(dev_labels, scores), 6)
            if select_best and rec["dev_auc"] > best["auc"]:
                best = {"auc": rec["dev_auc"], "epoch": ep,
                        "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
        history.append(rec)
        print(f"[{log_prefix}] epoch {ep}/{epochs} {rec}", flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"[{log_prefix}] best epoch {best['epoch']} (dev AUC {best['auc']}) 로 복원", flush=True)
        history.append({"best_epoch": best["epoch"], "best_dev_auc": best["auc"]})
    return model, tok, history


@torch.no_grad()
def predict_scores(model, tok, texts, device, bs=32, max_len=512):
    """P(AI) 점수 배열. 입력 순서 보존."""
    model.eval()
    out = np.zeros(len(texts), dtype=np.float64)
    # 길이순 정렬로 패딩 낭비를 줄이고, 원래 순서로 되돌린다
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for s in range(0, len(order), bs):
        idx = order[s : s + bs]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        p = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        out[idx] = p
    return out


def auc_of(labels, scores):
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def calibrate_tau(human_scores, fpr_target):
    """인간 점수 분포에서 FPR ≤ target 이 되는 최소 임계값. 판정: score > τ ⇒ AI."""
    hs = np.asarray(human_scores, dtype=np.float64)
    tau = float(np.quantile(hs, 1.0 - fpr_target, method="higher"))
    return tau


def fpr_at(human_scores, tau):
    return float(np.mean(np.asarray(human_scores) > tau))


def tpr_at(ai_scores, tau):
    return float(np.mean(np.asarray(ai_scores) > tau))


def kfold_oof(pairs_train, factory, det_cfg, device, k, seed, log_prefix="oof"):
    """train split 의 pair 를 문서 단위 k-fold 로 나눠 OOF 점수를 뽑는다.

    factory: train_detector 와 같은 팩토리 — fold 마다 호출되어 사전학습 가중치에서 재출발.
    반환: {doc_id: {"human": p, "ai": p, "fold": f}}
    """
    docs = list(pairs_train)
    random.Random(seed).shuffle(docs)
    folds = [docs[i::k] for i in range(k)]
    oof = {}
    for fi, held in enumerate(folds):
        train_docs = [d for fj, f in enumerate(folds) if fj != fi for d in f]
        tr_texts, tr_labels = [], []
        for d in train_docs:
            tr_texts += [d["human_text"], d["ai_text"]]
            tr_labels += [0, 1]
        model, tok, _ = train_detector(
            factory, tr_texts, tr_labels, det_cfg, device,
            select_best=False, log_prefix=f"{log_prefix}-fold{fi}",
        )
        h_scores = predict_scores(model, tok, [d["human_text"] for d in held], device, max_len=det_cfg.max_len)
        a_scores = predict_scores(model, tok, [d["ai_text"] for d in held], device, max_len=det_cfg.max_len)
        for d, hp, ap in zip(held, h_scores, a_scores):
            oof[d["doc_id"]] = {"human": float(hp), "ai": float(ap), "fold": fi}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return oof


def save_detector(model, tok, out_dir, meta):
    out_dir = str(out_dir)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    io_utils.write_json(f"{out_dir}/meta.json", meta)


def load_detector(dir_path, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(str(dir_path))
    tok = AutoTokenizer.from_pretrained(str(dir_path))
    model.to(device)
    model.eval()
    return model, tok
