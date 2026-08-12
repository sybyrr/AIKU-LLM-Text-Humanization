#!/bin/bash
# /workspace 정리 — data/ 한 폴더에 53개가 섞여 있던 것을 역할별로 나눈다.
#
#   scripts/               실행 스크립트
#   corpus/raw             정식 승인본 (API 다운로드)
#   corpus/mirror          HF 미러 parquet
#   dataset/human          인간 원문
#   dataset/prompts        조건별 프롬프트와 재료(요약·명사)
#   dataset/generations    모델별 생성물
#   dataset/scores         탐지 점수
#   review/                검수 HTML
#   archive/pilot          파일럿 잔여물 (지우지 않음 — 조건 B 폐기 근거 등이 들어 있음)
#   logs/                  서버·변환 로그
#
# 스크립트 안의 상대 경로는 공통 변수로 바꾼다. 실행 전 반드시 생성 파이프라인이
# 끝나 있어야 한다(돌고 있으면 상대 경로가 깨진다).
set -eu
W=/workspace
D=$W/data

if pgrep -f "run_domestic.sh|run_full.sh|run_night.sh|generate.py" > /dev/null; then
  echo "생성 파이프라인이 아직 돌고 있습니다. 끝난 뒤 실행하세요."; exit 1
fi

mkdir -p $W/scripts $W/corpus/raw $W/corpus/mirror \
         $W/dataset/{human,prompts,generations,scores} $W/review $W/archive/pilot $W/logs

mv_if() { for f in "$@"; do [ -e "$f" ] && mv -f "$f" "$T/" || true; done; }

echo "── 스크립트"
T=$W/scripts; mv_if $D/*.sh $D/*.py $D/*.sql $D/*.jq
rm -rf $D/__pycache__

echo "── 인간 원문"
T=$W/dataset/human; mv_if $D/full_base.jsonl $D/russell_pool_100.jsonl $D/russell_pool_100.csv

echo "── 프롬프트와 재료"
T=$W/dataset/prompts; mv_if $D/prompts_full.jsonl $D/full_summaries.jsonl $D/full_nouns.jsonl

echo "── 생성물"
T=$W/dataset/generations
[ -d $D/gen_full ] && mv -f $D/gen_full/*.jsonl $T/ 2>/dev/null || true
mkdir -p $T/quant && [ -d $D/gen_quant ] && mv -f $D/gen_quant/*.jsonl $T/quant/ 2>/dev/null || true
mv_if $D/full_generations.jsonl

echo "── 탐지 점수"
T=$W/dataset/scores; mv_if $D/detect_scores.jsonl $D/detect_scores_buggy.jsonl

echo "── 검수 HTML"
T=$W/review; mv_if $D/*.html $D/full_cards.json

echo "── 파일럿 잔여물 보관"
T=$W/archive/pilot
mv_if $D/pilot_*.jsonl $D/pilot_*.json $D/retry*.jsonl $D/retest*.jsonl \
      $D/prompts_A2.jsonl $D/prompts_A_only.jsonl $D/prompts_A100.jsonl \
      $D/prompts_AEC.jsonl $D/sample10.json $D/russell_prompts_100.jsonl
for g in gen gen_aec gen_quant gen_full; do
  [ -d "$D/$g" ] && { mkdir -p $T/$g; mv -f $D/$g/* $T/$g/ 2>/dev/null || true; rmdir $D/$g 2>/dev/null || true; }
done

echo "── 로그"
[ -d $D/logs ] && { mv -f $D/logs/* $W/logs/ 2>/dev/null || true; rmdir $D/logs 2>/dev/null || true; }

rmdir $D 2>/dev/null && echo "── data/ 제거됨" || echo "── data/ 에 남은 것: $(ls -A $D 2>/dev/null | tr '\n' ' ')"

echo
echo "=== 결과 ==="
for d in scripts corpus dataset review archive logs models notes; do
  n=$(find $W/$d -type f 2>/dev/null | wc -l)
  printf "  %-22s %4s개  %s\n" "$d/" "$n" "$(du -sh $W/$d 2>/dev/null | cut -f1)"
done
