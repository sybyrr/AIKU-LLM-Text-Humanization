# corpus/mirror — HF 미러 parquet 자리 (이 사본에는 없음)

원 머신에는 HF `Saxo/ko-news-corpus-{1,7,8}` 에서 받은 `shard1.parquet` · `shard7.parquet` · `shard8.parquet` 가 있었으나, 폴더 이관 때 복사되지 않았다.

- `scripts/extract_russell_pool.sql` 이 이 경로를 읽는다. 인간 풀을 재추출하려면 위 HF 데이터셋을 다시 받아 같은 이름으로 두거나, SQL 의 SOURCE 를 `corpus/raw/` 공식본으로 교체한다.
- 단, `corpus/raw/` 의 zip 은 **22-시리즈(신문 말뭉치 2022)** 이고 현재 데이터셋 doc_id 는 **23-시리즈** — 판본 불일치 문제는 [progress.md](../../progress.md) 참조.
- 이 디렉토리는 비어 있어도 지우지 말 것 (스크립트가 경로를 참조).
