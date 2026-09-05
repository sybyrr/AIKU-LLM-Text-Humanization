#!/usr/bin/env python3
"""Export actual saved G2 test outputs beside input, G1, and human reference.

This reads artifacts only and does not load a model or modify the running job.
--smoke explicitly labels the eight-document preflight checkpoint as non-final.
"""
import argparse
from datetime import datetime, timezone
import html
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop_lib import config as C, data as D, io_utils as io


def quoted(text):
    return '\n'.join('> ' + html.escape(line, quote=False)
                     for line in text.strip().splitlines())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--limit', type=int, default=20)
    args = ap.parse_args()
    if args.limit < 1:
        ap.error('--limit must be positive')
    cfg = C.load_config(C.LOOP_DIR / 'configs/base.yaml',
                        C.LOOP_DIR / 'configs/arms/team_news.yaml')
    production = C.runs_dir(cfg)
    test = {r['doc_id']: r for r in D.by_split(D.load_pairs(cfg))['test']}
    root = production / 'g2_preflight' if args.smoke else production
    out = root / 'arms' / cfg.arm.name / 'round1'
    generated_path = out / 'eval/gen_test.jsonl.gz'
    if not generated_path.is_file():
        raise SystemExit('G2 test outputs are not available yet; run after evaluation generation finishes.')
    generation_rows = io.read_jsonl(generated_path)
    generated = {r['doc_id']: r['text'] for r in generation_rows}
    if len(generated) != len(generation_rows) or not set(generated) <= set(test):
        raise RuntimeError('Duplicate or non-test document IDs in G2 outputs')
    g1_rows = io.read_jsonl(production / 'arms' / cfg.arm.name / 'round0/eval/gen_test.jsonl.gz')
    g1 = {r['doc_id']: r['text'] for r in g1_rows}
    if not set(generated) <= set(g1):
        raise RuntimeError('Missing G1 comparison outputs')
    ids = sorted(generated)
    selected = random.Random(42).sample(ids, min(args.limit, len(ids)))
    label = '소량 연결 검증 모델 — 본학습 G₂ 결과 아님' if args.smoke else '본학습 G₂'
    lines = [f'# 뉴스 출력 직접 검수: {label}', '',
             f'- 생성기: `{out / "dpo/final"}`',
             f'- 원본 산출물: `{generated_path}`',
             f'- 표본: test {len(ids):,}건 중 {len(selected)}건, seed=42 무작위 추출(점수로 고르지 않음).',
             '- 아래 글은 저장된 실제 출력이다. 표시 과정에서 교정하거나 다시 생성하지 않았다.',
             '- 입력 AI와 비교해 숫자·날짜·이름·인용·부정·인과관계·결론이 유지되는지 먼저 확인한다.',
             '- 그다음 반복·문장 단절·어색한 표현과 G₁ 대비 개선 여부를 확인한다.',
             '- 인간 원문은 참고 정답이며, 입력 AI에 이미 있던 차이와 G₂가 새로 만든 차이를 구분한다.',
             '- 코퍼스 전문이 포함되어 있으므로 팀 내부에서만 검수하고 공개 저장소에 올리지 않는다.', '']
    if args.smoke:
        lines += ['**주의: 긴 train 문서 8건으로 실행 연결만 시험한 체크포인트다.**',
                  '**이 파일은 열람 형식의 예시이며 본학습 G₂의 성능 증거가 아니다.**', '']
    for i, doc_id in enumerate(selected, 1):
        row = test[doc_id]
        lines += [f'## {i}. `{doc_id}`', '']
        for title, text in [('입력 AI 글', row['ai_text']), ('기존 G₁ 출력', g1[doc_id]),
                            (label + ' 출력', generated[doc_id]), ('인간 원문 (참고)', row['human_text'])]:
            lines += [f'### {title}', '', quoted(text), '']
        lines += ['검수: 사실 보존 □ / 자연스러움 □ / G₁보다 개선 □ / 수정 필요 사항:', '', '---', '']
    destination = out / 'review/review_samples.md'
    io.write_text(destination, '\n'.join(lines))
    io.write_json(destination.parent / 'review_manifest.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'smoke': args.smoke,
        'seed': 42, 'n_available': len(ids), 'selected_doc_ids': selected,
        'generator': str(out / 'dpo/final'), 'generated_path': str(generated_path),
    })
    print(destination)


if __name__ == '__main__':
    main()
