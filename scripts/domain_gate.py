#!/usr/bin/env python3
"""도메인 공통 D_pair — 고정 탐지기(frozen D) 게이트.

news_gate_5fold.py(5-fold OOF)는 폐기한다. fold마다 다른 모델 5개가 채점하고
버려지므로 이후 DPO 보상·최종 평가에 쓸 '그 탐지기'가 없다 — "게이트/보상/평가
전 과정에 같은 D" 원칙과 모순 (2026-08-16 회의 지적). 또 pair 전체(test 포함)로
학습한 모델이라 최종 평가에 쓰면 test 누수가 된다.

논문(3:1:1:1 의 detector-train split)과 같은 구조로 고친다:
  1. 문서 단위로 detector-split(기본 750편)을 먼저 뗀다
  2. 그 문서들의 인간 원문 + 각 arm 의 AI 재서술로 klue/roberta-base 를
     1회 학습해 동결 → --model-out (DPO 보상·최종 평가에 그대로 재사용)
  3. 나머지 풀(pair 후보)을 동결 D 로 채점: D(x_h)<τ AND D(x_ai)>=τ 만 통과
  4. 통과쌍을 문서 단위 8:1:1 로 분할해 D_pair 출력
D 학습 문서는 pair 풀에서 완전히 빠지므로 게이트·평가 어느 쪽에도 누수가 없다.

단계 산출물이 이미 있으면 그 단계는 건너뛴다(중단-재실행 안전):
  <out 디렉토리>/detector_split.json · <model-out>/config.json ·
  <out 디렉토리>/pool_scores.jsonl · --out

사용:
  python domain_gate.py --pool pool.jsonl \
      --gen qwen3-8b=gen_qwen.jsonl --gen exaone-3.5-7.8b=gen_exa.jsonl \
      --det-n 750 --tau 0.5 --model-out models/{domain}_roberta_D \
      --out dataset/{domain}_track/dpair.jsonl
"""
import argparse, json, pathlib, re
from collections import Counter
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = "klue/roberta-base"; DEV = "cuda:0"
HANGUL = re.compile(r"[가-힣]")
DIGIT = re.compile(r"\d")


def digit_density(t):
    """1,000자당 숫자 개수. 층화 축 — 에세이는 논제별로 1.0~22.8 로 20배 차이가 난다
    (사실·자료 인용형 vs 의견·경험형). 논제 5개만 test 에 들어가므로 무작위로 뽑으면
    한쪽으로 쏠려 test 가 도메인 대표성을 잃는다."""
    return len(DIGIT.findall(t)) / max(len(t), 1) * 1000


def pick_det_groups(sizes, det_n, how="large"):
    """detector-split 에 넣을 그룹(논제)을 고른다.

    논제 크기가 12~700편으로 58배 차이 나서 "편수 비율 = 논제 비율" 을 동시에
    만족시킬 수 없다. det 750편(전체 8.9%)을 채우면 큰 논제로는 2개, 작은 논제로는
    17개가 된다 — D 의 주제 다양성과 평가 풀의 주제 다양성이 정면으로 상충한다.
    어느 쪽이 나은지는 생성 후 논제별 통과율·d 점수로 판정할 문제라 전략을 인자로 뺀다.

      large  큰 논제부터 — det 논제 최소(2), 평가 풀 논제 최대(30)
      small  작은 논제부터 — det 논제 최대(17), 평가 풀 논제 최소(15)
      spread 크기 순위를 고르게 건너뛰며 — 양쪽 절충
    """
    keys = sorted(sizes, key=lambda k: -sizes[k])
    if how == "small":
        order = keys[::-1]
    elif how == "spread":
        step = max(1, len(keys) // 8)
        order = keys[::step] + [k for k in keys if k not in set(keys[::step])]
    else:
        order = keys
    det, n = [], 0
    for k in order:
        if n >= det_n:
            break
        det.append(k); n += sizes[k]
    return det, n


def split_groups(groups, ratios, stratify=None, seed=42, tries=None):
    """그룹(예: 논제·책) 단위로 train/dev/test 분할 — 그룹은 쪼개지지 않는다.

    stratify 가 없으면 크기만 맞추면 되므로 탐색 없이 탐욕 배정한다(O(G log G)).
    stratify 가 있으면 split 별 가중평균을 전체 평균에 맞추는 국소탐색을 하되,
    비용을 증분 갱신한다. 전량 재계산 방식은 그룹이 많을 때 터진다
    (문어 6,666권에서 5,300억 연산으로 10시간 이상 소요된 사고가 있었다).
    """
    names = ["train", "dev", "test"]
    keys = sorted(groups)
    size = {k: len(groups[k]) for k in keys}
    total = sum(size.values())
    target = {n: total * r for n, r in zip(names, ratios)}

    def emit(a):
        return {d: a[k] for k in keys for d in groups[k]}

    def greedy(order):
        cur = {n: 0 for n in names}; a = {}
        for k in order:
            n = max(names, key=lambda x: target[x] - cur[x])
            a[k] = n; cur[n] += size[k]
        return a, cur

    if not stratify:
        a, _ = greedy(sorted(keys, key=lambda k: -size[k]))   # 큰 그룹부터
        return emit(a)

    mean_all = sum(stratify[k] * size[k] for k in keys) / total

    def cost(cur, acc):
        c = sum(abs(cur[n] - target[n]) / max(target[n], 1) for n in names)
        for n in names:
            if cur[n]:
                c += 2.0 * abs(acc[n] / cur[n] - mean_all) / max(abs(mean_all), 1e-9)
        return c

    if tries is None:                       # 그룹이 많을수록 탐색을 줄인다
        tries = max(1, min(300, 20000 // max(len(keys), 1)))
    rng = np.random.RandomState(seed)
    best, best_c = None, float("inf")
    for t in range(tries):
        order = sorted(keys, key=lambda k: -size[k])
        if t:                               # 첫 시도는 결정론적, 이후 무작위 섞기
            order = list(keys); rng.shuffle(order)
            order.sort(key=lambda k: -size[k] + rng.rand() * size[k] * 0.3)
        a, cur = greedy(order)
        acc = {n: 0.0 for n in names}
        for k, n in a.items():
            acc[n] += stratify[k] * size[k]
        c = cost(cur, acc)
        improved = True
        while improved:                     # 증분 갱신 국소탐색: 한 패스 O(G*3)
            improved = False
            for k in keys:
                n0 = a[k]
                for n1 in names:
                    if n1 == n0:
                        continue
                    cur[n0] -= size[k]; acc[n0] -= stratify[k] * size[k]
                    cur[n1] += size[k]; acc[n1] += stratify[k] * size[k]
                    c2 = cost(cur, acc)
                    if c2 < c - 1e-9:
                        a[k] = n0 = n1; c = c2; improved = True
                    else:                   # 되돌리기
                        cur[n1] -= size[k]; acc[n1] -= stratify[k] * size[k]
                        cur[n0] += size[k]; acc[n0] += stratify[k] * size[k]
        if c < best_c:
            best_c, best = c, dict(a)
    return emit(best)


MARK = re.compile(r"^([AB])\s*[::]\s*(.*)$")


def repair_first_line(text):
    """대화 도메인 복구: 첫 줄에 표식이 없고 둘째 줄이 B 면 'A: ' 를 붙인다.

    모델이 첫 발화의 접두만 빠뜨리는 사례가 20건 중 11건(55%)이었다. 교대가 100%
    보장이라 첫 화자는 반드시 A 이므로 모호함 없이 복구된다. 복구 없이 턴 수를 세면
    45% 만 일치하고, 그 형식 불일치 자체가 D 의 지름길이 된다.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if lines and not MARK.match(lines[0]) and len(lines) > 1:
        m1 = MARK.match(lines[1])
        if m1 and m1.group(1) == "B":
            lines[0] = "A: " + lines[0].strip()
    return "\n".join(lines)


def dialogue_ok(text, n_turn):
    """표식이 붙은 줄 수가 원본 턴 수와 같고, 표식 없는 줄이 없어야 한다."""
    lines = [l for l in text.split("\n") if l.strip()]
    marked = [l for l in lines if MARK.match(l)]
    if len(marked) != len(lines):
        return False, "표식누락"
    if len(marked) != n_turn:
        return False, f"턴수 {len(marked)}≠{n_turn}"
    spk = [MARK.match(l).group(1) for l in marked]
    if any(spk[i] == spk[i + 1] for i in range(len(spk) - 1)):
        return False, "교대깨짐"
    return True, ""


def valid(rec, human_text):
    """깨짐 검사 — 생성 실패만 거른다 (PIPELINE.md 원칙: 의미 필터는 안 넣음)"""
    t = rec.get("text")
    if not t or "error" in rec:
        return False
    r = len(t) / max(len(human_text), 1)
    return 0.5 <= r <= 2.0 and len(HANGUL.findall(t)) / len(t) >= 0.30


class DS(Dataset):
    # max_length 512: 문서 800~1200자 ≈ 500~600토큰. 256이면 D가 앞 절반만 보고
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
    ap.add_argument("--split-key", default=None,
                    help="이 필드(예: prompt=논제) 단위로 분할. 미지정 시 문서 단위 무작위. "
                         "같은 그룹이 train/test 에 걸치는 문장 누수를 막는다.")
    ap.add_argument("--split-ratio", default="0.8,0.1,0.1",
                    help="train,dev,test 문서 비율. 그룹이 적으면 7:1.5:1.5 권장")
    ap.add_argument("--dialogue", action="store_true",
                    help="대화 도메인: 생성물 첫 줄 'A: ' 복구 + 턴 수/교대 검증. "
                         "풀에 n_turn 필드가 있어야 한다.")
    ap.add_argument("--det-select", default="large", choices=["large", "small", "spread"],
                    help="--split-key 사용 시 detector-split 에 넣을 그룹 선택 전략")
    ap.add_argument("--stratify", action="store_true",
                    help="--split-key 사용 시 숫자밀도로 층화(성격 쏠림 방지)")
    ap.add_argument("--frozen-only", action="store_true",
                    help="확장용: detector-split·D학습 건너뛰고 동결 D 로 --pool 전량 채점·게이트. "
                         "D 는 원래 750편으로 학습된 것을 그대로 재사용(누수 없음).")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out).parent
    mdir = pathlib.Path(args.model_out)
    tok = AutoTokenizer.from_pretrained(BASE)

    # 도메인 무관 로딩: news 풀은 pair_id/human_text, 그 외 도메인 풀은 doc_id/(body|text).
    # (오케스트레이터가 build_domain_prompts.py 와 같은 human_pool.jsonl 하나를 이 게이트에도 넘긴다.)
    pool, gkey, nturn = {}, {}, {}
    for r in map(json.loads, open(args.pool)):
        pid = r.get("pair_id") or r["doc_id"]
        pool[pid] = r.get("human_text") or r.get("body") or r.get("text")
        if args.split_key:
            if args.split_key not in r:
                raise KeyError(f"--split-key '{args.split_key}' 가 풀에 없음 (doc_id={pid})")
            gkey[pid] = r[args.split_key]
        if args.dialogue:
            if "n_turn" not in r:
                raise KeyError(f"--dialogue 는 풀에 n_turn 이 있어야 함 (doc_id={pid})")
            nturn[pid] = r["n_turn"]

    def join_id(gid):
        """생성물의 doc_id 를 풀 키에 맞춘다.

        build_domain_prompts.py 는 트랙 간 충돌을 막으려 doc_id 에 '<domain>-' 접두어를
        붙이는데(essay-ESSAY_32788), 풀은 원본 id(ESSAY_32788)를 쓴다. 접두어를 벗겨
        양쪽을 맞춘다 — 안 그러면 전량 미스매치로 유효 0 이 된다.
        """
        if gid in pool:
            return gid
        if "-" in gid:
            stripped = gid.split("-", 1)[1]
            if stripped in pool:
                return stripped
        return None

    arms = {}
    for spec in args.gen:
        name, path = spec.split("=", 1)
        arms[name], n_bad = {}, 0
        n_dlg = Counter()
        for g in map(json.loads, open(path)):
            pid = join_id(g["doc_id"])
            if not (pid and valid(g, pool[pid])):
                n_bad += 1; continue
            text = g["text"]
            if args.dialogue:
                text = repair_first_line(text)
                ok, why = dialogue_ok(text, nturn[pid])
                if not ok:
                    n_bad += 1; n_dlg[why.split()[0]] += 1; continue
            arms[name][pid] = text
        print(f"arm {name}: 유효 {len(arms[name]):,} / 깨짐·누락 {n_bad}"
              + (f" (대화검증 탈락 {dict(n_dlg)})" if args.dialogue and n_dlg else ""), flush=True)

    # ── 1. 문서 단위 detector-split ──
    split_f = outdir / "detector_split.json"
    if args.frozen_only:
        # 확장 모드: D 재학습 없이 --pool 을 채점한다. 단, D 가 학습한 문서(det-split)는
        # 반드시 제외한다 — 안 그러면 D 가 자기 학습 문서를 채점해 게이트를 통과시키는 누수가 난다.
        # (과거 이 가드가 없어 확장 풀에 det 750편이 섞여 722편이 누수된 사고가 있었다 → audit_overlap.py [3])
        assert (mdir / "config.json").exists(), \
            f"--frozen-only 는 기존 동결 D 가 있어야 함: {mdir}"
        assert split_f.exists(), \
            f"--frozen-only 는 detector_split.json 이 있어야 D 학습 문서를 제외한다(누수 방지): {split_f}"
        det_ids = set(json.load(open(split_f))["detector"])
        pool_ids = [i for i in sorted(pool) if i not in det_ids]
        n_excl = len(pool) - len(pool_ids)
        print(f"frozen-only: pool {len(pool_ids):,}편 채점 "
              f"(D 학습 {len(det_ids)}편 중 {n_excl}편 제외, D 재학습 없음)", flush=True)
    elif split_f.exists():
        sp = json.load(open(split_f))
        det_ids, pool_ids = set(sp["detector"]), sp["pool"]
        print(f"split 재사용: det {len(det_ids)} / pool {len(pool_ids)}", flush=True)
    elif args.split_key:
        # D 학습 문서도 그룹 단위로 뗀다. 안 그러면 D 가 학습 때 본 문장을 pair 풀에서
        # 다시 만나 d_human 을 낮추고 d_ai 를 높여, 그 문서들의 게이트 통과율이 부풀려진다.
        gs = {}
        for d in sorted(pool):
            gs.setdefault(gkey[d], []).append(d)
        det_g, n = pick_det_groups({k: len(v) for k, v in gs.items()},
                                   args.det_n, args.det_select)
        det_ids = {d for k in det_g for d in gs[k]}
        pool_ids = [d for d in sorted(pool) if d not in det_ids]
        ndg = len({gkey[d] for d in det_ids})
        json.dump({"detector": sorted(det_ids), "pool": pool_ids}, open(split_f, "w"))
        print(f"split 생성(그룹 '{args.split_key}' · {args.det_select}): det {len(det_ids)}편/{ndg}그룹 "
              f"· pool {len(pool_ids)}편/{len(gs)-ndg}그룹", flush=True)
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

    # ── 4. 게이트 + 문서 단위 8:1:1 ──
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
    ratios = tuple(float(x) for x in args.split_ratio.split(","))
    if args.split_key:
        groups = {}
        for d in ids:
            groups.setdefault(gkey[d], []).append(d)
        strat = ({k: sum(digit_density(pool[d]) for d in v) / len(v)
                  for k, v in groups.items()} if args.stratify else None)
        sp = split_groups(groups, ratios, strat)
        c = Counter(sp.values())
        gc = {n: len({gkey[d] for d in ids if sp[d] == n}) for n in ("train", "dev", "test")}
        print(f"분할(그룹 '{args.split_key}'"
              f"{', 숫자밀도 층화' if args.stratify else ''}): "
              + " · ".join(f"{n} {c[n]}편/{gc[n]}그룹" for n in ("train", "dev", "test")), flush=True)
    else:
        np.random.RandomState(42).shuffle(ids)
        n1, n2 = int(len(ids) * ratios[0]), int(len(ids) * (ratios[0] + ratios[1]))
        sp = {d: ("train" if j < n1 else "dev" if j < n2 else "test") for j, d in enumerate(ids)}
    with open(args.out, "w") as f:
        for p in passed:
            f.write(json.dumps({**p, "split": sp[p["id"]]}, ensure_ascii=False) + "\n")

    print(f"\n게이트 τ={args.tau} · 풀 {len(pool_ids)}문서:", flush=True)
    for name, (n_all, n_h, n_f, n_p) in stat.items():
        print(f"  {name}: 유효쌍 {n_all} · 인간→인간 {n_h} ({100*n_h/max(n_all,1):.0f}%) · "
              f"flip {n_f} ({100*n_f/max(n_all,1):.0f}%) · 통과 {n_p} ({100*n_p/max(n_all,1):.0f}%)", flush=True)
    c = Counter(sp[p["id"]] for p in passed)
    print(f"D_pair v2: {len(passed):,}쌍 → {args.out}", flush=True)
    print(f"  split(문서 단위): train {c['train']} · dev {c['dev']} · test {c['test']}", flush=True)


if __name__ == "__main__":
    main()
