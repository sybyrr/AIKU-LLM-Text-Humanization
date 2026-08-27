#!/usr/bin/env python3
"""BERTScore(내용 보존) — 후보(humanizer 출력) vs 기준 x_ai. 조건별 평균 F1.

notes/60: 품질지표는 최종평가(멀쩡한 humanizer 출력) 때 "회피의 게이트"로 쓴다
(붕괴·의미파괴로 낮춘 회피는 무효). BERTScore 는 토큰단위 정합이라 문장 cos 보완.
model_type=xlm-roberta-large (캐시됨, BERTScore 다국어 기본, num_layers 17).

후보 jsonl: {doc_id, text, <group>}  (예: eval_collapse --out 의 {doc_id, dec, text}
            또는 transfer_gen 의 {doc_id, text, model}).
기준: --ref-pairs 의 ai_text 를 doc_id(=id) 로 조인.
사용:
  python metrics_quality.py --cand d2/eval_samples.jsonl --group dec
  python metrics_quality.py --cand ../dataset/news_track/transfer_gen.jsonl --group model --device cpu
"""
import argparse, json, collections, statistics as st
from bert_score import score as bertscore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ref-pairs", default="/workspace/dataset/news_track/news_dpair_clean20k.jsonl")
    ap.add_argument("--group", default="model", help="후보 jsonl의 조건 필드명 (model/dec/cond)")
    ap.add_argument("--model-type", default="xlm-roberta-large")
    ap.add_argument("--num-layers", type=int, default=17)  # xlm-roberta-large 권장층
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    ref_by = {}
    for l in open(args.ref_pairs):
        r = json.loads(l)
        if "ai_text" in r:
            ref_by[r["id"]] = r["ai_text"]
        elif r.get("model") == "xai":   # transfer_gen 처럼 xai 행이 기준인 경우
            ref_by[r["doc_id"]] = r["text"]
    print(f"기준 x_ai {len(ref_by):,}건 로드", flush=True)

    cands, refs, groups = [], [], []
    miss = 0
    for l in open(args.cand):
        if not l.strip():
            continue
        r = json.loads(l)
        g = r.get(args.group, "?")
        if g == "xai" or g == "human":   # 기준/인간 행은 채점 대상 아님
            continue
        ref = ref_by.get(r["doc_id"])
        if ref is None or not r.get("text"):
            miss += 1; continue
        cands.append(r["text"]); refs.append(ref); groups.append(g)
    print(f"채점 {len(cands):,}건 (기준누락/빈 {miss}) · model_type={args.model_type} · {args.device}", flush=True)
    if not cands:
        print("채점할 후보 없음"); return

    P, R, F1 = bertscore(cands, refs, model_type=args.model_type, num_layers=args.num_layers,
                         batch_size=args.batch_size, device=args.device, verbose=True)
    f1 = F1.tolist()
    by = collections.defaultdict(list)
    for g, v in zip(groups, f1):
        by[g].append(v)

    print("\n" + "=" * 46)
    print(f"BERTScore-F1 (내용보존 vs x_ai) · 전체 {st.mean(f1):.4f}")
    print("=" * 46)
    print(f"{'조건':<14}{'n':>6}{'BERTScore-F1':>16}")
    for g in sorted(by):
        print(f"{g:<14}{len(by[g]):>6}{st.mean(by[g]):>16.4f}")
    print("=" * 46)


if __name__ == "__main__":
    main()
