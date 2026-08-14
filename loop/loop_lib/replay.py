"""탐지기 재학습용 replay buffer (notes/31 § Round 5).

replay 에 재서술문만 넣으면 탐지기가 "AI 탐지기"가 아니라 "ko-BART 출력 탐지기"로
퇴화한다 — 그래서 주 arm 은 (고정 인간 + 원본 AI + 전 라운드 재서술문) 전부를 쓰고,
latest_only arm 이 그 경고를 ablation 으로 검증한다.

class-balance: 인간 절반 / AI 절반, AI 안에서는 소스(원본, r1..rt)끼리 균등
— WeightedRandomSampler 가중치로 구현.
"""
from . import config as C
from . import io_utils


def replay_gen_path(cfg, r, arm=None):
    return C.round_dir(cfg, r, arm) / "detector" / "replay_gen.jsonl.gz"


def build_buffer(cfg, pairs_train, t, arm=None):
    """(texts, labels, weights, group_names) — 라운드 t 종료 후 D_{t+1} 학습셋."""
    mode = cfg.arm.replay
    groups = {}  # name → list[text]
    if mode == "full":
        groups["orig_ai"] = [r["ai_text"] for r in pairs_train]
        rounds = range(1, t + 1)
    elif mode == "latest_only":
        rounds = [t]
    else:
        raise ValueError(f"arm.replay={mode}")

    train_ids = {r["doc_id"] for r in pairs_train}
    for r in rounds:
        path = replay_gen_path(cfg, r, arm)
        rows = [x for x in io_utils.iter_jsonl(path) if x["doc_id"] in train_ids and x.get("text")]
        if not rows:
            raise RuntimeError(f"replay 생성물이 비었다: {path}")
        groups[f"para_r{r}"] = [x["text"] for x in rows]

    humans = [r["human_text"] for r in pairs_train]
    texts = list(humans)
    labels = [0] * len(humans)
    weights = [0.5 / len(humans)] * len(humans)

    n_g = len(groups)
    for name, g_texts in groups.items():
        texts += g_texts
        labels += [1] * len(g_texts)
        weights += [0.5 / (n_g * len(g_texts))] * len(g_texts)

    info = {"human": len(humans), **{k: len(v) for k, v in groups.items()}, "mode": mode}
    return texts, labels, weights, info
