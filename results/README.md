# 실험 결과

## 결과 파일

| 실험 | 보고서·수치 | 평가 범위 |
|---|---|---|
| Detector 검증 | [JSON](metrics/detector_validation.json) | 선별 전 held-out 기사 ID 2,250개 |
| 뉴스 주 실험 | [JSON](metrics/news_main_results.json) | test 2,044편, SFT/DPOP |
| Prompting baseline | [보고서](reports/humanizer_skill_baseline_n100_20260909.html), [뉴스 JSON](metrics/news_baseline_comparison.json) | 6개 도메인 각각 같은 100편 |
| 교차 도메인 | [보고서](reports/cross_domain_current6_full_20260909.html), [JSON](metrics/cross_domain_results.json) | 6×6 조합, 각 target test 전체 |
| Adversarial alignment | [보고서](../adversarial/RESULTS.md), [JSON](metrics/adversarial_results.json) | 뉴스 test 1,985편, G1/G2 |
| 사람 평가 | [보고서](reports/human_eval_results_share_20260907.md), [JSON](metrics/human_evaluation.json) | 50개 기사 × 4개 버전, 평가자 4명 |

보고서에서 `DPO`로 표기한 모델은 길이 정규화 DPOP를 적용한 generator다.

## 그래프 생성

Matplotlib·NumPy가 설치된 환경에서 다음 명령을 실행한다.

```bash
python scripts/figures/build_detector_validation.py
python scripts/figures/build_main_results.py
python scripts/figures/build_baseline_comparison.py
python scripts/figures/build_cross_domain.py
python scripts/figures/build_adversarial_results.py
python scripts/figures/build_human_evaluation.py
```

PNG·SVG는 `results/figures/`에 저장한다. Detector 검증은 HTML 파일도 생성한다.
뉴스 주 실험과 detector 검증은 JSON을 읽고, 나머지는 보고서에서 수치를 추출해 JSON과 그림을 생성한다.

`news_tsne_original.png`는 뉴스 test 200개 기사 ID의 Human/AI/SFT/DPOP에서 추출한
RoBERTa CLS 표현 800개를 시각화한 결과다. t-SNE 계산에는
[analyze_domain.py](../scripts/analyze_domain.py)와 해당 데이터·checkpoint가 필요하다.

## 평가 기준

- 주 ASR은 전체 출력의 detector 통과율이다. 반복 붕괴 출력도 포함하며 붕괴율을 별도로 보고한다.
- 의미 cosine은 최대 256-token 임베딩으로 계산한다.
- Detector 검증 값 85/99/100%는 반올림된 분류율이다. Human은 TNR, Qwen·EXAONE은 TPR이다.
- Cross-domain 평균은 조합별 동일 가중치다. Petition·Wiki는 `petition512B`·`wiki512B` 트랙이다.
- 사람 평가의 배정 격자는 설계 예시다. Wilson 구간은 버전별 이항비율에 대한 95% 신뢰구간이며,
  기사와 평가자 간 차이를 반영한 추정 구간은 아니다.

모델과 지표의 정의는 [방법론](../docs/methodology.md)에 설명했다.
