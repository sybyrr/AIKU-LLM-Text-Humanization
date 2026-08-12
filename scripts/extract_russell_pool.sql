-- 러셀 방식 인간 원문 풀 추출
-- 원본: 국립국어원 신문 말뭉치 2023 (2022년 생산 기사)
--   * 파일럿은 HF 미러 Saxo/ko-news-corpus-{1,7,8} 사용 (재배포 조건 확인 필요)
--   * 정식 승인본을 받으면 SOURCE 경로만 바꿔 그대로 재실행
--
-- 구조와 용어는 ../notes/40-코퍼스-구조-용어.md 에 정의돼 있다. 요약:
--   * ID 3층 — file_id(파일) > doc_id(기사) > sentence_id(문단)
--   * para_idx = sentence_id 의 마지막 마디. 1부터 빈틈 없이 이어짐
--       para_idx = 1  → 제목 (본문 아님)
--       para_idx = 2  → 리드 문단 (부제 아님)
--       para_idx >= 3 → 본문 문단
--   * 행의 단위는 문단이다. 컬럼명 sentence/sentence_id 는 업로더가 잘못 붙인 것
--     (본문 행의 29.1% 가 문장 2개 이상을 담고 있음)
--     → 이어붙일 때 반드시 개행 2개로 join 해야 문단 구조가 살아난다
--   * title 컬럼은 기사 제목이 아니라 파일 제목('노컷뉴스 2022년 기사'). 쓰지 말 것

.mode csv

CREATE OR REPLACE VIEW src AS
SELECT doc_id, publisher, date, topic, author,
       CAST(regexp_extract(sentence_id, '\.([0-9]+)$', 1) AS INT) AS para_idx,
       sentence
FROM read_parquet(['/workspace/corpus/mirror/shard1.parquet', '/workspace/corpus/mirror/shard7.parquet', '/workspace/corpus/mirror/shard8.parquet']);

CREATE OR REPLACE TABLE docs AS
SELECT doc_id,
       any_value(publisher)                              AS publisher,
       any_value(date)                                   AS date,
       any_value(topic)                                  AS topic,
       max(para_idx) - 1                                 AS n_para,
       max(sentence) FILTER (para_idx = 1)                AS title,
       string_agg(sentence, chr(10)||chr(10) ORDER BY para_idx) FILTER (para_idx > 1) AS body
FROM src
GROUP BY doc_id;

CREATE OR REPLACE TABLE cand AS
SELECT *, length(body) AS n_char
FROM docs
WHERE date < 20221101                                    -- ChatGPT 출시 이전
  AND topic IN ('정치', '경제', '사회', 'IT/과학', '문화')  -- 스포츠/연예/생활/미용건강 제외
  AND length(body) BETWEEN 1500 AND 2500                 -- 러셀 675~810단어 대응
  AND n_para >= 8
  AND title NOT LIKE '%.'                                -- 제목 파싱 실패분 방어
  AND length(title) BETWEEN 10 AND 60;

-- 토픽별 20편 × 5토픽 = 100편. 토픽 안에서 매체 라운드로빈으로 분산.
-- hash(doc_id) 정렬이라 재실행해도 같은 표본이 나옴 (난수 아님).
CREATE OR REPLACE TABLE pool AS
WITH rr AS (
  SELECT *, row_number() OVER (PARTITION BY topic, publisher ORDER BY hash(doc_id)) AS rn_tp
  FROM cand
), pick AS (
  SELECT *, row_number() OVER (PARTITION BY topic ORDER BY rn_tp, hash(doc_id)) AS rn_t
  FROM rr
)
SELECT doc_id, publisher, topic, date, n_para, n_char, title, body
FROM pick
WHERE rn_t <= 20
ORDER BY topic, publisher, doc_id;

COPY pool TO '/workspace/dataset/human/russell_pool_100.csv'   (HEADER, DELIMITER ',');
COPY pool TO '/workspace/dataset/human/russell_pool_100.jsonl' (FORMAT JSON);

-- 러셀 프롬프트 조립 (조건 A).
-- 프롬프트에 들어가는 슬롯은 5개뿐이고, 그중 기사를 특정하는 건 제목 하나다.
-- 나머지(매체/섹션/발행일)는 범주 정보, 목표 분량은 스칼라 1개.
-- 인간 본문(para_idx >= 2)에서 온 문자는 0자.
--
-- 원논문 대비 조정 5가지:
--   1. 부제 삭제 — 코퍼스에 부제가 없음. para_idx=2 는 리드 문단(본문)이라
--      부제 자리에 넣으면 인간 산문이 유출됨
--   2. 목표 분량을 단어수 → 글자수(100자 단위 반올림). 짝의 길이를 맞추는 게 목적.
--      미달/초과해도 재생성하지 않고 편차를 사후 측정한다
--   3. 발행일 추가 — 원논문에 없음. 2022년 기사를 2026년 모델로 생성하므로
--      시점을 안 주면 사실 오류가 나고, 그건 문체가 아닌 단서로 탐지된다.
--      단 이건 시점 오류 차단용이지 모델의 지식 사용을 막지는 못한다
--   4. '전문가 인용 포함' 지시 제거 — 원논문에 있으나 가짜 전문가 생성을 유도해
--      문체가 아닌 단서를 만든다. '일반 독자가 이해하기 쉽도록 간결하게' 는 유지
--   5. 출력 형식 지시 추가 — 원논문에 없음. 기사는 마크다운 형식이 아니라서
--      볼드/불릿이 나오면 형식 위반이 문체 단서로 오인된다. 제목 재출력도 금지
--      (평가자에게 제목을 보여주지 않으므로)
--
-- 제외한 필드: author(고유명사 유출), original_topic(태그가 기사 내용과 불일치),
--              n_para(측정 대상인 종속변수), body(정답)
CREATE OR REPLACE TABLE prompts AS
SELECT doc_id, publisher, topic, date, n_char,
       round(n_char, -2)::INT AS target_char,
       title,
       format(
         '다음은 {}년 {}월 {}일자 {}의 {} 섹션에 실린 기사의 제목입니다. '
      || '이 제목에 해당하는 기사를 약 {}자 분량으로 작성하세요. '
      || '일반 독자가 이해하기 쉽도록 간결하게 작성하세요. '
      || '제목은 다시 쓰지 말고 본문만 작성하고, 굵은 글씨·목록·소제목 같은 '
      || '마크다운 서식은 쓰지 마세요.

제목: {}',
         date // 10000, (date // 100) % 100, date % 100,
         publisher, topic, round(n_char, -2)::INT, title
       ) AS prompt
FROM pool;

COPY prompts TO 'russell_prompts_100.jsonl' (FORMAT JSON);
