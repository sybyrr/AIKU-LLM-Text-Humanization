-- 파일럿 10편 × 조건 3개 프롬프트 생성
--
-- 조건은 "프롬프트에 인간 원문 정보를 얼마나 주느냐"의 사다리다.
-- 러셀 실측 기준선 = 부제/기사 = 2.9% (300편 실측, 단어·글자 기준 모두 ≈2.8%)
--
--   A  제목만              평균 34자   기사 대비 1.9%   러셀 0.7배
--   B  제목 + 도입부 2문장  평균 152자  기사 대비 8.5%   러셀 2.9배
--   C  제목 + 요약 1문장    약 55자     기사 대비 3.1%   러셀 1.1배
--
-- 라벨 주의: 러셀 원문은 'Subtitle:' 인데 이 코퍼스에 부제가 없다.
--   B 는 본문 첫 문단이므로 '도입부', C 는 합성 요약이므로 '요약'으로 붙인다.
--   '부제:' 로 쓰면 모델이 한국 기사에 없는 부제를 만들어 붙일 수 있다.
--
-- 공통 지시(원논문 대비): 전문가 인용 요구 제거, 발행일 추가,
--   마크다운 금지 + 제목 재출력 금지. 상세는 extract_russell_pool.sql 주석 참고.

.mode csv

CREATE OR REPLACE TABLE base AS
SELECT doc_id, publisher, topic, date, n_char, n_para, title, body,
       round(n_char, -2)::INT AS target_char,
       format('{}년 {}월 {}일', date // 10000, (date // 100) % 100, date % 100) AS date_ko,
       -- 문단을 문장으로 쪼갠 뒤 앞 2문장 = 조건 B 컨텍스트
       array_to_string(
         string_split(replace(replace(body, chr(10)||chr(10), ' '), '. ', '.|'), '|')[1:2], ' '
       ) AS lead2
FROM read_json('sample10.json');

-- 공통 지시문. {} 는 순서대로 date_ko, publisher, topic, 제공물 이름, target_char
CREATE OR REPLACE MACRO instr(d, p, t, kind, n) AS format(
     '다음은 {}자 {}의 {} 섹션에 실린 기사의 {}입니다. 이에 해당하는 기사를 '
  || '약 {}자 분량으로 작성하세요. 일반 독자가 이해하기 쉽도록 간결하게 작성하세요. '
  || '제목은 다시 쓰지 말고 본문만 작성하고, 굵은 글씨·목록·소제목 같은 '
  || '마크다운 서식은 쓰지 마세요.', d, p, t, kind, n);

CREATE OR REPLACE TABLE prompts AS
SELECT doc_id, 'A' AS cond, publisher, topic, date, n_char, target_char, title,
       instr(date_ko, publisher, topic, '제목', target_char)
         || format('

제목: {}', title) AS prompt
FROM base
UNION ALL
SELECT doc_id, 'B', publisher, topic, date, n_char, target_char, title,
       instr(date_ko, publisher, topic, '제목과 도입부', target_char)
         || format('

제목: {}
도입부: {}', title, lead2)
FROM base
-- 조건 C 는 요약이 필요하므로 여기서는 자리만 만들고
-- summarize.py 가 채운 뒤 build_pilot_prompts_c.sql 이 조립한다
;

COPY (SELECT doc_id, title, body, n_char, target_char, publisher, topic, date_ko FROM base)
  TO 'pilot_base.jsonl' (FORMAT JSON);
COPY (SELECT * FROM prompts ORDER BY cond, doc_id) TO 'pilot_prompts_AB.jsonl' (FORMAT JSON);
