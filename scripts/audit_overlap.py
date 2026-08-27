#!/usr/bin/env python3
"""데이터 누수 감사 재확인 — clean20k(공격학습)와 detector-split(D학습)의 기사 겹침,
clean20k train↔test 분리, 폐기본 news_dpair_final의 722 누수 대조, D 실측 품질.

읽기 전용. 사용: python audit_overlap.py
"""
import json, collections, numpy as np

NT = "/workspace/dataset/news_track"


def ids(path, split=None):
    s = set()
    for l in open(path):
        r = json.loads(l)
        if split is None or r.get("split") == split:
            s.add(r["id"])
    return s


def main():
    det = json.load(open(f"{NT}/detector_split.json"))
    D = set(det["detector"]); POOL = set(det.get("pool", []))
    clean = f"{NT}/news_dpair_clean20k.jsonl"
    c_all = ids(clean); c_tr = ids(clean, "train"); c_dev = ids(clean, "dev"); c_te = ids(clean, "test")

    print("=" * 66)
    print("데이터 누수 감사 (읽기전용 재확인)")
    print("=" * 66)
    print(f"detector-split: D학습 {len(D)}편 · pool {len(POOL)}편 · D∩pool = {len(D & POOL)}")
    print(f"clean20k: 전체 {len(c_all)} (train {len(c_tr)}/dev {len(c_dev)}/test {len(c_te)}), 고유 {len(c_all)}")
    print("-" * 66)
    print("[1] 탐지기↔공격 겹침")
    print(f"    clean20k ∩ D학습(750)      = {len(c_all & D)}   (0이어야 함)")
    print(f"    clean20k ∩ pool(2250)      = {len(c_all & POOL)}   (정상 — 공격후보 기사)")
    print("[2] 공격 학습↔평가 분리 (concat+20k 실험)")
    print(f"    train ∩ test               = {len(c_tr & c_te)}   (0이어야 held-out)")
    print(f"    train ∩ dev                = {len(c_tr & c_dev)}")
    print(f"    test  ∩ D학습              = {len(c_te & D)}   (0이어야 함)")

    # [3] 폐기본 대조
    finalp = f"{NT}/news_dpair_final.jsonl"
    try:
        f_all = ids(finalp)
        print("[3] 폐기본 news_dpair_final.jsonl (오염 — 사용금지)")
        print(f"    final ∩ D학습              = {len(f_all & D)}   (=722 누수, clean20k에서 제거됨)")
    except FileNotFoundError:
        print("[3] news_dpair_final.jsonl 없음(이미 정리)")

    # [4] D 실측 품질 (held-out pool 채점)
    try:
        rows = [json.loads(l) for l in open(f"{NT}/pool_scores.jsonl")]
        hs = np.array([r["score"] for r in rows if r.get("kind") == "human"])
        ai = np.array([r["score"] for r in rows if r.get("kind") not in ("human", None)])
        allv = np.concatenate([hs, ai]); y = np.array([0] * len(hs) + [1] * len(ai))
        order = allv.argsort(); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
        auroc = (ranks[y == 1].sum() - len(ai) * (len(ai) + 1) / 2) / (len(hs) * len(ai))
        thr95 = np.percentile(hs, 95)
        print("[4] D 실측 품질 (held-out pool, D 미학습 기사)")
        print(f"    AUROC {auroc:.4f} · FPR5%(임계 {thr95:.3f})에서 AI recall {100*np.mean(ai>=thr95):.1f}%")
        print(f"    τ=0.5: human FPR {100*np.mean(hs>=0.5):.1f}% · AI recall {100*np.mean(ai>=0.5):.1f}%")
    except FileNotFoundError:
        print("[4] pool_scores.jsonl 없음")
    print("=" * 66)


if __name__ == "__main__":
    main()
