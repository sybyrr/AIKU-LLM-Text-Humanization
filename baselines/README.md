# Prompting baseline

SFT/DPOP와 같은 held-out AI 입력을 prompting으로 재작성하는 비교 실험입니다.
각 방법은 동일한 입력에 독립적으로 적용합니다.

```text
x_ai
  ├─ SFT generator          → SFT
  ├─ DPOP generator         → DPO
  └─ Humanizer-skill prompt → Humanizer-skill
```

## 실험 조건

| 항목 | 설정 |
| --- | --- |
| 도메인 | news, essay, persona, written, petition512B, wiki512B |
| 표본 | 도메인별 20편 파일럿 → 같은 20편을 포함한 100편 최종 평가 |
| 생성 모델 | Qwen 입력은 Qwen3-8B, EXAONE 입력은 EXAONE-3.5-7.8B |
| 생성 | temperature 0.1, top-p 0.9, 출력 토큰 예산 = 입력 글자 수 × 0.8 |
| 평가 | 도메인별 RoBERTa·SCRN, 의미 cosine, 반복 붕괴·문장 재사용·근사중복 |
| ASR | 같은 도메인의 Human 점수 P95 이하인 출력 비율 |

[고정 프롬프트](humanizer_skill_batch_v1.6.0.txt)는 DaleSeo의
[`korean-skills` humanizer v1.6.0](https://github.com/DaleSeo/korean-skills/blob/ae12ba27982ebeff03b46dc738365aaa34260d9a/skills/humanizer/SKILL.md)을
배치 평가용으로 변환한 것입니다. 원본 commit은
`ae12ba27982ebeff03b46dc738365aaa34260d9a`이며,
라이선스는 [LICENSE.humanizer](LICENSE.humanizer)를 참고하세요.

## 준비와 실행

도메인별 test pair, RoBERTa/SCRN checkpoint, SFT/DPOP reference 출력을 준비합니다.
데이터 형식과 환경 설정은 [데이터 안내](../docs/data.md)와
[실행 가이드](../docs/reproduction.md)에 있습니다.

```bash
cp baselines/config.example.json baselines/config.local.json
# pair/checkpoint/reference 경로와 llama-server endpoint 설정

python scripts/build_humanizer_skill_baseline.py \
  --config baselines/config.local.json --sizes 20 100

python scripts/run_humanizer_skill_baseline.py \
  --config baselines/config.local.json
```

Config의 상대 데이터 경로는 **저장소 루트 기준**입니다. 생성기 endpoint를 먼저 실행합니다.
오케스트레이터는 도메인마다 GPU 하나를 할당하므로, 6개 도메인을 함께 실행하려면
`evaluation.gpus`에 GPU 6개를 지정합니다. GPU 수에 맞춰 `domains`를 나누어 실행할 수도
있습니다.

### 같은 문서에서 비교

`reference_dir/{domain}_to_{domain}.jsonl`의 문서 ID를 사용해 SFT/DPOP와 같은
100편을 선택합니다. Reference 파일이 없으면 pair의 test split에서 표본을 선택합니다.
이 경우 SFT/DPOP도 생성된 manifest의 문서 ID로 평가하고, 같은 Human 기준으로
세 방법의 ASR을 계산합니다.

## 파일럿 게이트와 개별 단계

20편에서 아래 조건을 모두 통과하면 100편으로 확장합니다.

- 출력/입력 길이비 중앙값: 0.65–1.35
- 분석·변경 설명 혼입: 5% 이하
- 평균 의미 cosine: 0.85 이상
- 반복 붕괴율: 5% 이하

| 스크립트 | 역할 |
| --- | --- |
| `build_humanizer_skill_baseline.py` | 비교 문서 선정·prompt shard·manifest 생성 |
| `run_humanizer_skill_baseline.py` | 20편 생성·평가·게이트 후 100편 실행 |
| `eval_humanizer_skill_baseline.py` | 생성 결과의 RoBERTa/SCRN·cosine·반복 평가 |
| `summarize_humanizer_skill_baseline.py` | 동일 문서의 SFT/DPOP와 비교·파일럿 게이트 집계 |

집계 결과와 도표는 [`results/`](../results/)에서 확인할 수 있습니다.
