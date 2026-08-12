-- 조건 A / E / C 프롬프트 생성 (B 폐기, evade 제외)
--
-- 공통: "당신은 {날짜}자 {매체} {섹션}의 기자입니다" 기자 시점 프레이밍.
--   기존 "다음은 …기사의 제목입니다" 대비 개선 확인됨 (EXAONE-33B 10편):
--     · 프롬프트 메타정보를 본문에 옮겨 적는 사고 소멸
--     · 첫 문장이 인용동사로 끝나는 비율 0/10 → 3/10 (인간도 3/10)
--     · 길이 달성률 80% → 91%
--   남은 차이: 첫 문장에 "2022년 6월 13일," 처럼 발행일을 박는 패턴(인간 0/10).
--   이건 러셀 evade 지침이 겨냥하는 항목이므로 베이스라인에서는 지우지 않고
--   지표로 기록한다. 지우면 이미 회피 처리된 조건이 되어 대조가 사라진다.
--
-- 조건별로 제목 외에 주는 것 (러셀 실측 부제/기사 = 2.9% 가 기준선):
--   A  없음                    34자   1.9%   0.7배
--   C  합성 요약 1문장          70자   3.8%   1.3배
--   E  명사 목록 25개          117자   6.4%   2.3배
--
-- 폐기한 조건 B(도입부 2문장): 인간 산문이 그대로 복제됐다. 측정 결과
--   LCS 평균 28.2자 / N그램 겹침 20.4% (A·C 는 9~10자 / 1% 수준),
--   Qwen3-14B 는 최악 132자 도입부를 100% 통째로 복사.
--   E 가 같은 자리(정보량)를 유출 없이 대체한다.

.mode csv

CREATE OR REPLACE MACRO head(d, p, t, kind, n) AS format(
     '당신은 {}자 {} {} 섹션의 기자입니다. 다음 {}에 해당하는 기사를 '
  || '약 {}자 분량으로 작성하세요. 일반 독자가 이해하기 쉽도록 간결하게 작성하세요. '
  || '제목은 다시 쓰지 말고 본문만 작성하고, 굵은 글씨·목록·소제목 같은 '
  || '마크다운 서식은 쓰지 마세요.', d, p, t, kind, n);

CREATE OR REPLACE TABLE prompts AS
SELECT doc_id, 'A' AS cond, target_char, n_char, title,
       head(date_ko, publisher, topic, '제목', target_char)
       || format(chr(10)||chr(10)||'제목: {}', title) AS prompt
FROM read_json('pilot_base.jsonl')
UNION ALL
SELECT b.doc_id, 'C', b.target_char, b.n_char, b.title,
       head(b.date_ko, b.publisher, b.topic, '제목과 요약', b.target_char)
       || format(chr(10)||chr(10)||'제목: {}'||chr(10)||'요약: {}', b.title, s.summary)
FROM read_json('pilot_base.jsonl') b JOIN read_json('pilot_summaries.jsonl') s USING (doc_id)
UNION ALL
SELECT b.doc_id, 'E', b.target_char, b.n_char, b.title,
       head(b.date_ko, b.publisher, b.topic, '제목과 핵심어', b.target_char)
       || format(chr(10)||chr(10)||'제목: {}'||chr(10)||'핵심어: {}', b.title, e.nouns)
FROM read_json('pilot_base.jsonl') b JOIN read_json('pilot_nouns.jsonl') e USING (doc_id);

COPY (SELECT * FROM prompts ORDER BY cond, doc_id) TO '/workspace/dataset/prompts/prompts_full.jsonl' (FORMAT JSON);
