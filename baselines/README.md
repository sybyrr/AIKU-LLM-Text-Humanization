# 프롬프트 기준선과 zero-shot 평가

이 디렉터리는 학습 기반 SFT·DPO humanizer와 별도로 실행한 프롬프트 기준선의 명세를
보관합니다. 같은 held-out `x_ai`를 서로 독립적인 세 경로에 입력합니다.

```text
x_ai
  ├─ 도메인별 SFT 모델       → SFT
  ├─ 도메인별 DPO 모델       → DPO
  └─ Humanizer-skill 프롬프트 → Humanizer-skill
```

Humanizer-skill 출력에 SFT나 DPO를 추가로 적용하지 않습니다. SFT와 DPO는 프롬프트
기준선의 구성요소가 아니라 비교 대상입니다.

## Humanizer-skill 조건

- 도메인: `news`, `essay`, `persona`, `written`, `petition512B`, `wiki512B`
- 표본: 도메인별 held-out 20편 파일럿 후, 같은 표본을 포함하는 100편 최종평가
- seed: 42
- 생성기 대응: Qwen 생성문은 Qwen3-8B, EXAONE 생성문은 EXAONE-3.5-7.8B로 재작성
- 생성: temperature 0.1, 출력 토큰 예산은 입력 추정 토큰 수의 0.8배
- 평가: 도메인별 D·SCRN, 의미 cosine, 급성붕괴, 문장 재사용, 근사중복
- ASR 임계값: 같은 도메인의 인간 문서 점수 95백분위

배치용 프롬프트는 [humanizer_skill_batch_v1.6.0.txt](humanizer_skill_batch_v1.6.0.txt)에
고정했습니다. 이 프롬프트는 DaleSeo의
[`korean-skills`](https://github.com/DaleSeo/korean-skills/blob/ae12ba27982ebeff03b46dc738365aaa34260d9a/skills/humanizer/SKILL.md)
저장소에 있는 `humanizer` v1.6.0을 배치 평가에 맞게 변환한 것입니다. 원본 커밋은
`ae12ba27982ebeff03b46dc738365aaa34260d9a`이며, 저작권과 MIT 허가문은
[LICENSE.humanizer](LICENSE.humanizer)에 보존했습니다.

## 파일럿 게이트

각 도메인에서 아래 조건을 모두 충족해야 100편 평가로 확장합니다.

- 출력/입력 길이비 중앙값이 0.65 이상 1.35 이하입니다.
- 분석이나 변경 설명이 섞인 출력이 5% 이하입니다.
- 입력과 출력의 평균 의미 cosine이 0.85 이상입니다.
- 급성 반복 붕괴율이 5% 이하입니다.

## 실행 흐름

저장소에는 데이터, 모델, 로컬 설정을 포함하지 않습니다. 먼저
[config.example.json](config.example.json)을 복사하고 각 경로를 로컬 환경에 맞게
수정합니다.

```bash
cp baselines/config.example.json baselines/config.local.json

python scripts/build_humanizer_skill_baseline.py \
  --config baselines/config.local.json \
  --sizes 20 100
```

생성은 기존 `scripts/generate.py`와 OpenAI-compatible llama-server를 사용합니다.
설정에 적은 Qwen·EXAONE endpoint를 먼저 실행한 뒤 전체 기준선을 다음 명령으로 실행할
수 있습니다.

```bash
python scripts/run_humanizer_skill_baseline.py \
  --config baselines/config.local.json
```

오케스트레이터는 생성 결과를 `--resume`으로 이어 쓰고, 도메인별 평가를 GPU에 나누어
실행합니다. 20편 품질 게이트가 실패하면 100편 단계로 넘어가지 않습니다. 개별 단계만
실행하려면 `eval_humanizer_skill_baseline.py`와
`summarize_humanizer_skill_baseline.py`를 직접 사용할 수 있습니다.

## Binoculars·FastDetectGPT

`scripts/zero_shot_matrix.py`는 6×6 전량 결과와 프롬프트 기준선에 등장하는 텍스트를
SHA-256으로 중복 제거한 뒤 여러 GPU shard로 나누어 채점합니다. 각 문서 점수는 매번
즉시 append되므로 프로세스가 종료되어도 같은 명령으로 이어서 실행할 수 있습니다.

실험에 사용한 설정은 다음과 같습니다.

| 탐지기 | 모델 | 최대 길이 | 점수 방향 |
| --- | --- | ---: | --- |
| Binoculars | Qwen3-1.7B-Base / Qwen3-1.7B | 1,024 | 높을수록 AI (`score=-B`) |
| FastDetectGPT | EXAONE-4.0-1.2B | 2,048 | 높을수록 AI |

평가 JSONL 경로를 명시해 입력을 준비한 뒤 shard 수와 같은 수의 GPU를 지정합니다.

```bash
python scripts/zero_shot_matrix.py \
  --work-dir logs/zero_shot \
  prepare --shards 6 --input \
  logs/transfer_current6_full/*.jsonl \
  logs/humanizer_skill_baseline/n100/*.jsonl \
  logs/transfer_current6/{news_to_news,essay_to_essay,persona_to_persona,written_to_written,petition512B_to_petition512B,wiki512B_to_wiki512B}.jsonl

bash scripts/run_zero_shot_matrix.sh logs/zero_shot 0 1 2 3 4 5
```

중간 종료 자동 복구는 별도 터미널에서 실행합니다. watchdog은 각 worker PID와 완료
마커를 확인하고, worker가 사라진 미완료 shard만 같은 GPU에서 재개합니다.

```bash
POLL_SECONDS=60 bash scripts/watch_zero_shot_matrix.sh \
  logs/zero_shot 0 1 2 3 4 5
```

12개 탐지기 shard가 모두 끝나면 집계값과 결과 전용 HTML을 생성합니다.

```bash
python scripts/summarize_zero_shot_matrix.py \
  --work-dir logs/zero_shot \
  --cross-dir logs/transfer_current6_full \
  --skill-dir logs/humanizer_skill_baseline/n100 \
  --reference-dir logs/transfer_current6 \
  --output-dir results
```

공개 결과에는 집계값만 포함하며 원문, 생성문, 문서 ID와 문서별 점수는 포함하지 않습니다.
