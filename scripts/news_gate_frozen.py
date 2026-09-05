#!/usr/bin/env python3
"""신문 D_pair v2 — 고정 탐지기(frozen D) 게이트.

news_gate_5fold.py(5-fold OOF)는 폐기한다. fold마다 다른 모델 5개가 채점하고
버려지므로 이후 DPO 보상·최종 평가에 쓸 '그 탐지기'가 없다 — "게이트/보상/평가
전 과정에 같은 D" 원칙과 모순 (2026-08-16 회의 지적). 또 pair 전체(test 포함)로
학습한 모델이라 최종 평가에 쓰면 test 누수가 된다.

논문(3:1:1:1 의 detector-train split)과 같은 구조로 고친다:
  1. 기사 단위로 detector-split(기본 750편)을 먼저 뗀다
  2. 그 기사들의 인간 원문 + 각 arm 의 AI 재서술로 klue/roberta-base 를
     1회 학습해 동결 → --model-out (DPO 보상·최종 평가에 그대로 재사용)
  3. 나머지 풀(pair 후보)을 동결 D 로 채점: D(x_h)<τ AND D(x_ai)>=τ 만 통과
  4. 통과쌍을 기사 단위 8:1:1 로 분할해 D_pair 출력
D 학습 기사는 pair 풀에서 완전히 빠지므로 게이트·평가 어느 쪽에도 누수가 없다.

단계 산출물이 이미 있으면 그 단계는 건너뛴다(중단-재실행 안전):
  <out 디렉토리>/detector_split.json · <model-out>/config.json ·
  <out 디렉토리>/pool_scores.jsonl · --out

사용:
  python news_gate_frozen.py --pool pool.jsonl \
      --gen qwen3-8b=gen_qwen.jsonl --gen exaone-3.5-7.8b=gen_exa.jsonl \
      --det-n 750 --tau 0.5 --model-out /workspace/models/news_roberta_D \
      --out news_dpair_v2.jsonl
"""
import argparse, json, pathlib, re
from collections import Counter
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = "klue/roberta-base"; DEV = "cuda:0"
HANGUL = re.compile(r"[가-힣]")


def valid(rec, human_text):
    """깨짐 검사 — 생성 실패만 거른다 (PIPELINE.md 원칙: 의미 필터는 안 넣음)"""
    t = rec.get("text")
    if not t or "error" in rec:
        return False
    r = len(t) / max(len(human_text), 1)
    return 0.5 <= r <= 2.0 and len(HANGUL.findall(t)) / len(t) >= 0.30


class DS(Dataset):
    # max_length 512: 기사 800~1200자 ≈ 500~600토큰. 256이면 D가 앞 절반만 보고
    # 판정한다. D를 쓰는 모든 곳(게이트/DPO 보상/평가/Stage4)이 이 절단과 일치할 것.
    def __init__(s, tok, t): s.e = tok(t, padding=True, truncation=True, max_length=512, return_tensors="pt")
    def __len__(s): return len(s.e["input_ids"])
    def __getitem__(s, i): return {k: v[i] for k, v in s.e.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--gen", action="append", required=True, help="이름=경로. 복수 지정 = 복수 arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--det-n", type=int, default=750)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)  # 512토큰 학습, 12GB 안전 범위
    ap.add_argument("--frozen-only", action="store_true",
                    help="확장용: detector-split·D학습 건너뛰고 동결 D 로 --pool 전량 채점·게이트. "
                         "D 는 원래 750편으로 학습된 것을 그대로 재사용(누수 없음).")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out).parent
    mdir = pathlib.Path(args.model_out)
    tok = AutoTokenizer.from_pretrained(BASE)

    # 도메인 무관 로딩: news 풀은 pair_id/human_text, 그 외 도메인 풀은 doc_id/(body|text).
    # (오케스트레이터가 build_domain_prompts.py 와 같은 human_pool.jsonl 하나를 이 게이트에도 넘긴다.)
    pool = {(r.get("pair_id") or r["doc_id"]):
            (r.get("human_text") or r.get("body") or r.get("text"))
            for r in map(json.loads, open(args.pool))}

    arms = {}
    for spec in args.gen:
        name, path = spec.split("=", 1)
        arms[name], n_bad = {}, 0
        for g in map(json.loads, open(path)):
            if g["doc_id"] in pool and valid(g, pool[g["doc_id"]]):
                arms[name][g["doc_id"]] = g["text"]
            else:
                n_bad += 1
        print(f"arm {name}: 유효 {len(arms[name]):,} / 깨짐·누락 {n_bad}", flush=True)

    # ── 1. 기사 단위 detector-split ──
    split_f = outdir / "detector_split.json"
    if args.frozen_only:
        # 확장 모드: D 재학습 없이 --pool 을 채점한다. 단, D 가 학습한 기사(det-split)는
        # 반드시 제외한다 — 안 그러면 D 가 자기 학습 문서를 채점해 게이트를 통과시키는 누수가 난다.
        # (과거 이 가드가 없어 확장 풀에 det 750편이 섞여 722편이 누수된 사고가 있었다 → audit_overlap.py [3])
        assert (mdir / "config.json").exists(), \
            f"--frozen-only 는 기존 동결 D 가 있어야 함: {mdir}"
        assert split_f.exists(), \
            f"--frozen-only 는 detector_split.json 이 있어야 D 학습 기사를 제외한다(누수 방지): {split_f}"
        det_ids = set(json.load(open(split_f))["detector"])
        pool_ids = [i for i in sorted(pool) if i not in det_ids]
        n_excl = len(pool) - len(pool_ids)
        print(f"frozen-only: pool {len(pool_ids):,}편 채점 "
              f"(D 학습 {len(det_ids)}편 중 {n_excl}편 제외, D 재학습 없음)", flush=True)
    elif split_f.exists():
        sp = json.load(open(split_f))
        det_ids, pool_ids = set(sp["detector"]), sp["pool"]
        print(f"split 재사용: det {len(det_ids)} / pool {len(pool_ids)}", flush=True)
    else:
        ids = sorted(pool)
        np.random.RandomState(42).shuffle(ids)
        det_ids, pool_ids = set(ids[:args.det_n]), ids[args.det_n:]
        json.dump({"detector": sorted(det_ids), "pool": pool_ids}, open(split_f, "w"))
        print(f"split 생성: det {len(det_ids)} / pool {len(pool_ids)}", flush=True)

    # ── 2. D 1회 학습 → 동결 저장 (frozen-only 면 건너뜀) ──
    if not args.frozen_only and not (mdir / "config.json").exists():
        tr_t = [pool[i] for i in sorted(det_ids)]
        tr_y = [0] * len(tr_t)
        for name in sorted(arms):
            add = [arms[name][i] for i in sorted(det_ids) if i in arms[name]]
            tr_t += add; tr_y += [1] * len(add)
        n0, n1 = tr_y.count(0), tr_y.count(1)
        # arm 이 2개면 AI 가 인간의 2배 → class weight 로 보정
        w = torch.tensor([len(tr_y) / (2 * n0), len(tr_y) / (2 * n1)], dtype=torch.float).to(DEV)
        print(f"D 학습: 인간 {n0} + AI {n1} (weight {w[0]:.2f}/{w[1]:.2f})", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2).to(DEV)
        dl = DataLoader(list(zip(DS(tok, tr_t), torch.tensor(tr_y))), batch_size=args.bs, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        for ep in range(args.epochs):
            model.train(); tot = 0.0
            for x, y in dl:
                x = {k: v.to(DEV) for k, v in x.items()}
                loss = lossf(model(**x).logits, y.to(DEV))
                loss.backward(); opt.step(); opt.zero_grad(); tot += loss.item()
            print(f"  epoch {ep}: loss {tot/len(dl):.4f}", flush=True)
        mdir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(mdir); tok.save_pretrained(mdir)
        del model; torch.cuda.empty_cache()
        print(f"D 저장(동결): {mdir}", flush=True)
    else:
        print(f"D 재사용: {mdir}", flush=True)

    # ── 3. 풀 채점 (동결 D) ──
    score_f = outdir / "pool_scores.jsonl"
    if score_f.exists():
        scores = {(r["kind"], r["id"]): r["score"] for r in map(json.loads, open(score_f))}
        print(f"점수 재사용: {len(scores):,}건", flush=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(mdir).to(DEV).eval()
        keys, texts = [], []
        for i in pool_ids:
            keys.append(("human", i)); texts.append(pool[i])
            for name in sorted(arms):
                if i in arms[name]:
                    keys.append((name, i)); texts.append(arms[name][i])
        ps = []
        with torch.no_grad():
            for x in DataLoader(DS(tok, texts), batch_size=32):
                x = {k: v.to(DEV) for k, v in x.items()}
                ps.extend(torch.softmax(model(**x).logits, -1)[:, 1].cpu().numpy())
        with open(score_f, "w") as f:
            for (k, i), p in zip(keys, ps):
                f.write(json.dumps({"kind": k, "id": i, "score": round(float(p), 4)}) + "\n")
        scores = {(k, i): round(float(p), 4) for (k, i), p in zip(keys, ps)}
        print(f"채점 완료: {len(scores):,}건", flush=True)

    # ── 4. 게이트 + 기사 단위 8:1:1 ──
    passed, stat = [], {}
    for name in sorted(arms):
        n_h = n_f = n_p = n_all = 0
        for i in pool_ids:
            if ("human", i) not in scores or (name, i) not in scores:
                continue
            n_all += 1
            dh, da = scores[("human", i)], scores[(name, i)]
            hp, fl = dh < args.tau, da >= args.tau
            n_h += hp; n_f += fl
            if hp and fl:
                n_p += 1
                passed.append({"id": i, "generator": name, "human_text": pool[i],
                               "ai_text": arms[name][i], "d_human": dh, "d_ai": da})
        stat[name] = (n_all, n_h, n_f, n_p)

    ids = sorted({p["id"] for p in passed})
    np.random.RandomState(42).shuffle(ids)
    n1, n2 = int(len(ids) * 0.8), int(len(ids) * 0.9)
    sp = {d: ("train" if j < n1 else "dev" if j < n2 else "test") for j, d in enumerate(ids)}
    with open(args.out, "w") as f:
        for p in passed:
            f.write(json.dumps({**p, "split": sp[p["id"]]}, ensure_ascii=False) + "\n")

    print(f"\n게이트 τ={args.tau} · 풀 {len(pool_ids)}기사:", flush=True)
    for name, (n_all, n_h, n_f, n_p) in stat.items():
        print(f"  {name}: 유효쌍 {n_all} · 인간→인간 {n_h} ({100*n_h/max(n_all,1):.0f}%) · "
              f"flip {n_f} ({100*n_f/max(n_all,1):.0f}%) · 통과 {n_p} ({100*n_p/max(n_all,1):.0f}%)", flush=True)
    c = Counter(sp[p["id"]] for p in passed)
    print(f"D_pair v2: {len(passed):,}쌍 → {args.out}", flush=True)
    print(f"  split(기사 단위): train {c['train']} · dev {c['dev']} · test {c['test']}", flush=True)


if __name__ == "__main__":
    main()
