#!/bin/bash
# Stage 1b → 1c → 2 → 3 도메인 오케스트레이터 (정본). 인자: <domain> <gpu>
#
# 전제:
#   - dataset/{dom}_track/human_pool.jsonl 준비됨 (Stage 1a = 도메인별 추출, 상류에서).
#   - qwen3-8b llama-server 가 $URL 에서 서빙 중 (gpu_serve.sh / run_pilot.sh 참고).
# 정본 근거: notes/51-Stage1-3-파이프라인-정본.md
# rewrite_ko.user.txt(P2 회귀본) 대신 build_domain_prompts.py(P3b 통일)를 쓴다.
set +e
dom=$1; gpu=${2:-0}
[ -z "$dom" ] && { echo "usage: $0 <domain> <gpu>"; exit 1; }

export LD_LIBRARY_PATH=/home/dev/bt/lib:/home/dev/bt/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/workspace/scripts
PY=/workspace/.venv/bin/python
S=/workspace/scripts; DS=/workspace/dataset; M=/workspace/models
T=$DS/${dom}_track
RD=$M/${dom}_roberta_D   # 도메인 탐지기 D. D2 로 교체하려면 이 한 줄만 ${dom}_roberta_D2 로 바꾼다.
URL=${URL:-http://127.0.0.1:8081/v1/chat/completions}
log(){ echo "[$(date -d '+9 hours' +%H:%M) KST] [$dom] $*"; }

POOL=$T/human_pool.jsonl
[ -f "$POOL" ] || { log "human_pool.jsonl 없음 — Stage 1a(도메인별 추출) 먼저"; exit 1; }

# ── Stage 1b: P3b 통일 프롬프트 → x_ai 재서술 ────────────────────────────────
log "1b 프롬프트 조립 (P3b 통일)"
$PY $S/build_domain_prompts.py --in "$POOL" --out $T/prompts.jsonl --domain "$dom" || exit 1

log "1b 재서술 생성 (qwen3-8b · temp0.8 · no-think · ratio0.8)"
$PY $S/generate.py --in $T/prompts.jsonl --out $T/gen_qwen.jsonl \
  --url "$URL" --model qwen3-8b --no-think --temp 0.8 --seed 42 --token-ratio 0.8 || exit 1
# D2(3생성기) 타깃이면 exaone/kanana 서버로 각각 한 번 더 생성해 아래 게이트에 --gen 3개를 준다.

# ── Stage 1c: 게이트 (도메인 D 동결) → dpair ─────────────────────────────────
log "1c 게이트 (D 동결 학습 + d_human<τ AND d_ai≥τ)"
CUDA_VISIBLE_DEVICES=$gpu $PY $S/news_gate_frozen.py \
  --pool "$POOL" --gen qwen=$T/gen_qwen.jsonl \
  --out $T/dpair.jsonl --model-out "$RD" \
  --det-n 750 --tau 0.5 || { log "게이트 실패 — 중단"; exit 1; }
# D2 정본으로 갈 때: --det-n 4500 · --gen qwen=… --gen exaone=… --gen kanana=… · 위 RD 를 ${dom}_roberta_D2 로
np=$(wc -l < $T/dpair.jsonl 2>/dev/null); log "통과쌍 ${np:-0}"
[ "${np:-0}" -lt 50 ] && { log "통과쌍 부족 — 중단"; exit 1; }

# ── Stage 2: Style SFT (concat fusion + eos, 현행 stage2_sft.py 기본) ─────────
log "2 SFT (StyleBART concat+eos · λ0.5)"
CUDA_VISIBLE_DEVICES=$gpu $PY $S/stage2_sft.py \
  --pairs $T/dpair.jsonl --out-dir $M/${dom}_sft \
  --epochs 3 --bs 8 --lam 0.5 --split train || exit 1

# ── Stage 3: DPO (DPOP λ5 + length-norm + β2.0) ─────────────────────────────
log "3 hard-neg 채굴 (SFT 생성 중 D>τ)"
CUDA_VISIBLE_DEVICES=$gpu $PY $S/stage3_build_dpo.py \
  --sft $M/${dom}_sft/style_bart.pt \
  --pairs $T/dpair.jsonl --roberta "$RD" \
  --tau 0.5 --n-sample 4 --split train --out $T/dpo_pairs.jsonl || { log "hard-neg 채굴 실패 — 중단"; exit 1; }
nd=$(wc -l < $T/dpo_pairs.jsonl 2>/dev/null); log "hard-neg ${nd:-0}쌍"
[ "${nd:-0}" -lt 30 ] && { log "hard-neg 부족 — DPO 생략(SFT 가 이미 D 속임)"; exit 0; }

log "3 DPOP λ5 학습"
CUDA_VISIBLE_DEVICES=$gpu $PY $S/stage3_dpo.py \
  --dpo-pairs $T/dpo_pairs.jsonl \
  --sft $M/${dom}_sft/style_bart.pt \
  --out-dir $M/${dom}_dpo \
  --dpop --dpop-lambda 5.0 --length-norm --beta 2.0 \
  --epochs 1 --bs 2 --accum 8 --lr 5e-6 \
  --eval-pairs $T/dpair.jsonl --roberta "$RD" || { log "DPO 학습 실패 — 중단"; exit 1; }

log "완료 — models/${dom}_sft · models/${dom}_dpo · $T/dpair.jsonl"
