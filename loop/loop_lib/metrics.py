"""평가 지표 — notes/31 § 평가 표 구현.

  안전장치   length_auc            길이 단독 AUC (0.5 근처 유지 확인)
  하한선     linear_probe_*        TF-IDF char 2-4gram + LR (실측 test AUC 0.9932)
  인간성     style_features/…      형태소 문체 특징 + 인간 분포와의 z-거리
  진단       feature_movement      function view(내용어 품사 마스킹) 로지스틱 가중치
  의미 보존  preservation_row      수치·방향어·고유명사·복사(12자 n그램·LCS) — stage0_check_pairs.py 이식
  의미 보존  simcse_similarity     KoSimCSE cosine
  유창성     lm_perplexity         독립 한국어 LM (kogpt2) PPL

무거운 모델(simcse/ppl)은 지연 로드. kiwipiepy 부재 시 해당 지표만 None 으로 빠진다.
"""
import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np

# ── stage0_check_pairs.py 와 동일한 상수 (기준선 수치 승계) ────────
NGRAM = 12
NUM = re.compile(r"\d+(?:[.,]\d+)*\s*(?:%|퍼센트|명|건|개|년|월|일|시간|배|원|억|만|천)?")
NEG = re.compile(r"(않|못하|못한|없|아니|불가|미흡|비유의)")
DIRECTION = {
    "증가": r"증가|늘어|상승|높아|많아",
    "감소": r"감소|줄어|하락|낮아|적어",
    "유의": r"유의(?!하지|하지 않)",
    "비유의": r"유의하지 않|유의미하지 않|비유의",
    "정적": r"정\(\+\)|정적 (?:상관|영향)|양(?:\(\+\)|의 상관)",
    "부적": r"부\(-\)|부적 (?:상관|영향)|음(?:\(-\)|의 상관)",
}

_KIWI = None
_KIWI_FAILED = False


def kiwi():
    global _KIWI, _KIWI_FAILED
    if _KIWI is None and not _KIWI_FAILED:
        try:
            from kiwipiepy import Kiwi

            _KIWI = Kiwi()
        except Exception as e:
            print(f"[metrics] kiwipiepy 없음 — 형태소 기반 지표 생략 ({e})")
            _KIWI_FAILED = True
    return _KIWI


# ── 안전장치 · 하한선 ──────────────────────────────────────────
def length_auc(human_texts, ai_texts):
    from sklearn.metrics import roc_auc_score

    y = [0] * len(human_texts) + [1] * len(ai_texts)
    s = [len(t) for t in human_texts + ai_texts]
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _tfidf_lr():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=200_000,
                        sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=1.0),
    )


def linear_probe_transfer(train_h, train_a, test_h, test_a):
    """train 으로 학습해 test 를 채점 — Stage 0 하한선 확인용."""
    from sklearn.metrics import roc_auc_score

    clf = _tfidf_lr()
    clf.fit(train_h + train_a, [0] * len(train_h) + [1] * len(train_a))
    p = clf.predict_proba(test_h + test_a)[:, 1]
    return float(roc_auc_score([0] * len(test_h) + [1] * len(test_a), p))


def linear_probe_cv(h_texts, a_texts, folds=5, seed=42):
    """라운드 산출물의 선형 분리도 (같은 세트 안 CV AUC)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X = h_texts + a_texts
    y = np.array([0] * len(h_texts) + [1] * len(a_texts))
    if len(h_texts) < folds or len(a_texts) < folds:
        return float("nan")
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = _tfidf_lr()
        clf.fit([X[i] for i in tr], y[tr])
        p = clf.predict_proba([X[i] for i in te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


# ── 문체 특징 (형태소) ─────────────────────────────────────────
STYLE_FEATURES = [
    "noun_ratio", "verb_ratio", "vx_per_100m", "josa_ratio",
    "myeo_per_100m", "eoseo_per_100m", "euro_per_100m", "ttohan_per_1000c",
    "geosida_per_1000c", "sikida_per_100m", "sents_per_1000c", "mean_sent_chars",
]


def style_features(text):
    """한 문서의 문체 특징. notes/31 의 한국어 'AI 티' 지문 항목을 그대로 좇는다:
    AI쪽 명사+으로/또한/으며 ↔ 인간쪽 어서/보조용언/동사/시키."""
    k = kiwi()
    if k is None or not text.strip():
        return None
    toks = k.tokenize(text)
    n = max(1, len(toks))
    sents = [s for s in re.split(r"(?<=[.!?다])\s+", text.strip()) if s]
    n_sents = max(1, len(sents))
    chars = max(1, len(text))

    def cnt(pred):
        return sum(1 for t in toks if pred(t))

    return {
        "noun_ratio": cnt(lambda t: t.tag.startswith("NN")) / n,
        "verb_ratio": cnt(lambda t: t.tag == "VV") / n,
        "vx_per_100m": cnt(lambda t: t.tag == "VX") / n * 100,
        "josa_ratio": cnt(lambda t: t.tag.startswith("J")) / n,
        "myeo_per_100m": cnt(lambda t: t.tag in ("EC",) and t.form in ("며", "으며")) / n * 100,
        "eoseo_per_100m": cnt(lambda t: t.tag == "EC" and t.form in ("어서", "아서", "여서", "라서")) / n * 100,
        "euro_per_100m": cnt(lambda t: t.tag == "JKB" and t.form in ("으로", "로")) / n * 100,
        "ttohan_per_1000c": text.count("또한") / chars * 1000,
        "geosida_per_1000c": (text.count("것이다") + text.count("것으로")) / chars * 1000,
        "sikida_per_100m": cnt(lambda t: t.form == "시키" and t.tag.startswith("XSV")) / n * 100,
        "sents_per_1000c": n_sents / chars * 1000,
        "mean_sent_chars": chars / n_sents,
    }


def style_stats(texts):
    """특징별 (mean, std) — 인간 기준 분포."""
    rows = [f for f in (style_features(t) for t in texts) if f]
    if not rows:
        return None
    out = {}
    for k in STYLE_FEATURES:
        v = np.array([r[k] for r in rows])
        out[k] = {"mean": float(v.mean()), "std": float(v.std() + 1e-8)}
    return out


def style_distance(texts, ref_stats):
    """인간 분포 대비 z-점수 절대값 평균 (작을수록 인간에 가깝다) + 특징별 내역."""
    if ref_stats is None:
        return None, None
    rows = [f for f in (style_features(t) for t in texts) if f]
    if not rows:
        return None, None
    per = {}
    for k in STYLE_FEATURES:
        m = float(np.mean([r[k] for r in rows]))
        per[k] = {"mean": m, "z": (m - ref_stats[k]["mean"]) / ref_stats[k]["std"]}
    dist = float(np.mean([abs(v["z"]) for v in per.values()]))
    return dist, per


# ── function view + 특징 이동 추적 ─────────────────────────────
CONTENT_TAGS = ("NNG", "NNP", "NNB", "NP", "NR", "SL", "SH", "SN", "XR", "VV", "VA")


def function_view(text):
    """내용어를 품사 태그로 마스킹 — 어휘가 아니라 문법 구조만 남긴다."""
    k = kiwi()
    if k is None:
        return None
    out = []
    for t in k.tokenize(text):
        if t.tag in CONTENT_TAGS:
            out.append(t.tag)
        else:
            out.append(f"{t.form}/{t.tag}")
    return " ".join(out)


def feature_movement(h_texts, a_texts, top_k=25, seed=42):
    """function view 위 로지스틱 회귀 — 라운드별 'AI 티'가 어느 특징으로 옮겨가는지.
    반환: {"auc": …, "ai_side": [(feat, w)…], "human_side": [(feat, w)…]}"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    hv = [v for v in (function_view(t) for t in h_texts) if v]
    av = [v for v in (function_view(t) for t in a_texts) if v]
    if not hv or not av:
        return None
    X_raw = hv + av
    y = np.array([0] * len(hv) + [1] * len(av))
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"[^ ]+", ngram_range=(1, 3),
                          max_features=100_000, sublinear_tf=True)
    X = vec.fit_transform(X_raw)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    auc = float(np.mean(cross_val_score(clf, X, y, cv=3, scoring="roc_auc"))) if min(len(hv), len(av)) >= 3 else float("nan")
    clf.fit(X, y)
    names = vec.get_feature_names_out()
    w = clf.coef_[0]
    order = np.argsort(w)
    return {
        "auc_cv3": auc,
        "ai_side": [[names[i], round(float(w[i]), 3)] for i in order[::-1][:top_k]],
        "human_side": [[names[i], round(float(w[i]), 3)] for i in order[:top_k]],
    }


# ── 의미 보존 (stage0_check_pairs.py 이식) ─────────────────────
def _ngrams(s, n=NGRAM):
    s = re.sub(r"\s+", "", s)
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


def ngram_overlap_pct(src, gen):
    h = _ngrams(src)
    if not h:
        return 0.0
    return len(h & _ngrams(gen)) / len(h) * 100


def lcs_len(a, b):
    a, b = re.sub(r"\s+", "", a), re.sub(r"\s+", "", b)
    if not a or not b:
        return 0
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


def _nums(s):
    return Counter(re.sub(r"\s+", "", t) for t in NUM.findall(s))


def preservation_row(human_text, gen_text, x_ai=None):
    """생성문이 인간 원문의 사실을 지키는지 + 복사 정도. x_ai 를 주면 입력 대비 복사도 잰다."""
    hn, gn = _nums(human_text), _nums(gen_text)
    hd = {k: len(re.findall(v, human_text)) for k, v in DIRECTION.items()}
    gd = {k: len(re.findall(v, gen_text)) for k, v in DIRECTION.items()}
    row = {
        "len_ratio": round(len(gen_text) / max(1, len(human_text)) * 100, 1),
        "nums_missing": sum((hn - gn).values()),
        "nums_hallucinated": sum((gn - hn).values()),
        "dir_lost": [k for k in DIRECTION if hd[k] > 0 and gd[k] == 0],
        "neg_delta": len(NEG.findall(gen_text)) - len(NEG.findall(human_text)),
        "overlap_vs_human": round(ngram_overlap_pct(human_text, gen_text), 2),
        "lcs_vs_human": lcs_len(human_text, gen_text),
    }
    if x_ai is not None:
        row["overlap_vs_xai"] = round(ngram_overlap_pct(x_ai, gen_text), 2)
        row["lcs_vs_xai"] = lcs_len(x_ai, gen_text)
    k = kiwi()
    if k is not None:
        hp = {t.form for t in k.tokenize(human_text) if t.tag == "NNP"}
        if hp:
            gp = {t.form for t in k.tokenize(gen_text) if t.tag == "NNP"}
            row["propn_recall"] = round(len(hp & gp) / len(hp) * 100, 1)
    return row


# ── 무거운 모델 지표 (지연 로드) ───────────────────────────────
def simcse_similarity(texts_a, texts_b, model_name, device, batch_size=16, max_len=512):
    """쌍별 cosine 유사도 (KoSimCSE — [CLS])."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def embed(texts):
        out = []
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s : s + batch_size], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            h = model(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu())
        return torch.cat(out)

    ea, eb = embed(texts_a), embed(texts_b)
    sims = (ea * eb).sum(dim=-1).numpy()
    del model
    return sims


def lm_perplexity(texts, model_name, device, batch_size=8, max_len=512):
    """독립 한국어 LM 의 문서별 PPL."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    ppls = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s : s + batch_size], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            labels = ids.masked_fill(mask == 0, -100)
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
            tgt = labels[:, 1:]
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                ignore_index=-100, reduction="none").view(tgt.size())
            valid = (tgt != -100).float()
            nll = (ce * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            ppls.extend(torch.exp(nll).cpu().tolist())
    del model
    return ppls
