"""pair 데이터 로딩 — splits_v2 ⋈ human_pool ⋈ gen_full_p3b_clean.

정본 스키마 (한 행 = 한 원문):
  {doc_id, kci_article_id, split, human_text, ai_text, title, field}

- doc_id = "abstract-" + kci_article_id
- split ∈ {train, dev, test} : data/splits_v2.jsonl 이 유일한 배정 기준
- dev 는 결정적으로 cal/gate 반반으로 나눈다:
    cal  = τ 캘리브레이션 + 에폭 선택
    gate = 탐지기 붕괴 게이트 (FPR 측정) — 캘리브레이션에 쓴 표본으로 게이트를 재면
           FPR 5% 가 구성상 보장되어 게이트가 무의미해지므로 반드시 분리한다.
  분할은 md5(doc_id) 기반 — 플랫폼·세션 무관 재현.
"""
import hashlib

from . import io_utils
from .config import rp

DOC_PREFIX = "abstract-"
SPLITS = ("train", "dev", "test")
EXPECTED = {"train": 7249, "dev": 946, "test": 920, "total": 9115}


def _limit_by_split(rows, limit):
    if not limit:
        return rows
    by = {k: [] for k in SPLITS}
    for r in rows:
        if len(by[r["split"]]) < limit:
            by[r["split"]].append(r)
    return [r for k in SPLITS for r in by[k]]


def _load_paired_jsonl(path, limit=None):
    """이미 pair 형태인 외부 데이터셋을 정본 스키마로 검증해 읽는다.

    필수 필드는 ``doc_id|id, split, human_text, ai_text``다. 원본 파일을 직접
    수정하지 않고 스냅샷/어댑터 파일을 연결할 때 사용한다.
    """
    rows, seen = [], set()
    for i, raw in enumerate(io_utils.iter_jsonl(path), 1):
        doc_id = raw.get("doc_id") or raw.get("id")
        split = raw.get("split")
        if not doc_id or split not in SPLITS or not raw.get("human_text") or not raw.get("ai_text"):
            raise ValueError(
                f"{path}:{i}: 필수 pair 필드가 없거나 split 이 잘못됐다 "
                f"(doc_id|id, split={SPLITS}, human_text, ai_text)"
            )
        if doc_id in seen:
            raise ValueError(f"{path}:{i}: 중복 doc_id={doc_id}")
        seen.add(doc_id)
        rows.append({
            **raw,
            "doc_id": doc_id,
            "kci_article_id": raw.get("kci_article_id", doc_id),
            "split": split,
            "human_text": raw["human_text"],
            "ai_text": raw["ai_text"],
            "title": raw.get("title", ""),
            "field": raw.get("field", ""),
        })
    if not rows:
        raise RuntimeError(f"pair 데이터가 비었다: {path}")
    return _limit_by_split(rows, limit)


def load_pairs(cfg, limit=None):
    """전 pair 로드. limit 는 스모크용 — split 별로 앞에서 limit 개씩만 남긴다."""
    paired = cfg.data.get("paired_jsonl")
    if paired:
        return _load_paired_jsonl(rp(paired), limit=limit)

    splits = {r["doc_id"]: r for r in io_utils.iter_jsonl(rp(cfg.data.splits))}
    humans = {r["kci_article_id"]: r for r in io_utils.iter_jsonl(rp(cfg.data.human_pool))}
    gens = {}
    for r in io_utils.iter_jsonl(rp(cfg.data.gen)):
        if r.get("text"):
            gens[r["doc_id"]] = r

    rows, missing = [], []
    for doc_id, s in splits.items():
        aid = s["kci_article_id"]
        h, g = humans.get(aid), gens.get(doc_id)
        if h is None or g is None:
            missing.append(doc_id)
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "kci_article_id": aid,
                "split": s["split"],
                "human_text": h["original_abstract"],
                "ai_text": g["text"],
                "title": h.get("original_title", ""),
                "field": s.get("research_field", ""),
            }
        )
    if missing:
        raise RuntimeError(f"splits 에 있는데 원문/생성문이 없는 doc {len(missing)}건: {missing[:5]} …")

    if limit:
        rows = _limit_by_split(rows, limit)
    else:
        n = {k: sum(1 for r in rows if r["split"] == k) for k in SPLITS}
        assert len(rows) == EXPECTED["total"] and all(n[k] == EXPECTED[k] for k in SPLITS), (
            f"pair 수가 splits_v2 와 다르다: {n} (기대 {EXPECTED}) — 데이터 파일 확인"
        )
    return rows


def by_split(pairs):
    out = {k: [] for k in SPLITS}
    for r in pairs:
        out[r["split"]].append(r)
    return out


def dev_half(doc_id: str) -> str:
    """dev 를 cal/gate 로 가르는 결정적 규칙."""
    h = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    return "cal" if int(h, 16) % 2 == 0 else "gate"


def dev_halves(dev_pairs):
    cal = [r for r in dev_pairs if dev_half(r["doc_id"]) == "cal"]
    gate = [r for r in dev_pairs if dev_half(r["doc_id"]) == "gate"]
    return cal, gate


def texts_labels(pairs):
    """pair 목록 → (texts, labels, doc_ids). 라벨은 provenance: 인간 0 / AI 1."""
    texts, labels, ids = [], [], []
    for r in pairs:
        texts.append(r["human_text"])
        labels.append(0)
        ids.append(r["doc_id"])
        texts.append(r["ai_text"])
        labels.append(1)
        ids.append(r["doc_id"])
    return texts, labels, ids


def assert_no_test_docs(doc_ids, where: str, pairs=None, test_ids=None):
    """학습 산출물에 test 문서가 섞이는 사고 방지. 어디서든 학습 직전에 호출한다."""
    if test_ids is None:
        test_ids = {r["doc_id"] for r in pairs if r["split"] == "test"}
    bad = set(doc_ids) & set(test_ids)
    if bad:
        raise RuntimeError(f"{where}: test 문서 {len(bad)}건이 학습 경로에 들어왔다 — {sorted(bad)[:3]} …")
