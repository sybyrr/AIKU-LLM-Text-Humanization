#!/usr/bin/env python3
"""P4 베이스라인 — 측정된 '한국어 AI 티' 지문을 프롬프트에 그대로 명시한 재서술.

notes/31: "학습이 프롬프팅보다 낫다"는 주장에 이 베이스라인이 없으면 논문이 성립하지 않는다.
기존 조건(A/C/E)은 정보량만 바꿨을 뿐 문체를 직접 지시한 적이 없다.

지문(function view 회귀, notes/31 실측):
  AI 쪽 — 명사+으로 · 또한 · 으며 · 명사 나열
  인간 쪽 — ~어서(인과) · 보조용언 · 동사 중심 · 시키(사동)

입력 = test 920편의 x_ai (학습된 humanizer G 와 **같은 입력**으로 공정 비교).
출력 = generate.py 스키마. llama.cpp 생성기(Qwen3-8B)로 돌린 뒤 s0_probe_external.py 로 채점.

사용:
  python loop/scripts/build_p4_prompts.py --out loop/runs/probes/prompts_p4.jsonl
  # 서버에서:
  #   python scripts/generate.py --in loop/runs/probes/prompts_p4.jsonl \\
  #       --out loop/runs/probes/gen_p4.jsonl --url .../v1/chat/completions \\
  #       --model qwen3-8b --no-think --temp 0.8 --seed 42 --resume
  #   python loop/scripts/s0_probe_external.py --gen loop/runs/probes/gen_p4.jsonl \\
  #       --ref-source test --role evade --name p4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop_lib import config as C
from loop_lib import data as D
from loop_lib import io_utils as io

P4_SYSTEM = "당신은 학술 문장을 다듬는 전문 국어 편집자입니다."
P4_USER = (
    "다음 한국어 글을 같은 내용, 다른 문장으로 다시 쓰되, 아래 문체 지침을 반드시 따르세요.\n"
    "- 명사를 나열해 「~으로」로 잇지 말고, 동사를 중심으로 문장을 푸세요.\n"
    "- 「또한」·「으며」로 절을 병렬하지 말고, 인과와 흐름은 「~어서/~아서」처럼 이어지는 "
    "연결어미로 자연스럽게 이으세요.\n"
    "- 「~하였다」식 명사형 술어를 반복하지 말고, 보조용언(‘~해 보다’, ‘~하게 되다’ 등)과 "
    "사동·피동을 활용해 사람이 쓴 것처럼 쓰세요.\n"
    "- 사실은 하나도 바꾸지 마세요: 숫자·연도·비율·단위·통계값, 고유명사와 전문 용어, "
    "증가/감소와 유의/비유의 방향, 연구의 목적·방법·결과·결론을 모두 보존하세요.\n"
    "- 요약하지 말고 분량을 원문의 90~110%로 유지하세요.\n"
    "다시 쓴 글만 출력하세요 — 설명이나 제목, 마크다운 서식은 쓰지 마세요.\n\n{text}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.LOOP_DIR / "configs" / "base.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = C.load_config(a.config)
    pairs = D.by_split(D.load_pairs(cfg, limit=a.limit))[a.split]
    out = Path(a.out) if a.out else C.runs_dir(cfg) / "probes" / f"prompts_p4_{a.split}.jsonl"

    rows = [{"doc_id": r["doc_id"], "cond": "P4", "domain": r["field"],
             "target_char": len(r["ai_text"]), "n_char": len(r["ai_text"]),
             "system": P4_SYSTEM, "prompt": P4_USER.format(text=r["ai_text"])}
            for r in pairs]
    io.write_jsonl(out, rows)
    print(f"P4 프롬프트 {len(rows)}건 ({a.split}) → {out}")
    print("다음: scripts/generate.py 로 생성 → loop/scripts/s0_probe_external.py --role evade 로 채점")


if __name__ == "__main__":
    main()
