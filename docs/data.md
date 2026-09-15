# 데이터와 실험 집합

뉴스를 중심으로 Human–AI pair를 구축하고, 6개 도메인 교차 평가와 별도 뉴스 데이터의 adversarial alignment를 수행했다.

## 1. 뉴스 Human–AI pair 구축

인간 원문은 국립국어원 **신문 말뭉치 2022판의 2021년 기사**다.
인간 기사 **24,500편**을 수집해, 선별 후 generator용 Human–AI pair **20,343쌍**을 구성했다.

[뉴스 추출기](../scripts/mash_extract_pool.py)는 정치·경제·사회·IT/과학·문화 기사를 대상으로
본문 길이 800–1,200자, 본문 5문단 이상, 직접 인용 비율 40% 이하를 적용하고 주제·매체별로 추출한다.
원본 자료를 받는 보조 코드는 [download_corpus.py](../scripts/download_corpus.py)에 있다.

각 원문을 Qwen3-8B 또는 EXAONE-3.5-7.8B로 재서술한다.
공용 [P3b 프롬프트](../scripts/build_domain_prompts.py)는 사실·수치·고유명사·인용을 유지하면서
어휘와 문장 구조를 바꾸도록 요청한다.

```text
인간 기사 수집
  → detector 학습 기사 분리
  → LLM 재서술 및 고정 detector로 후보 pair 선별
  → detector 학습 기사 제외·원문 ID 중복 제거
  → generator train / dev / test 분할
```

## 2. Detector 학습·검증 및 pair 선별

뉴스 detector는 `klue/roberta-base`를 Human/AI 이진 분류기로 fine-tuning한 모델이다.
detector 구축·검증에 사용한 3,000편의 기사 ID 중 **750편으로 학습하고, 나머지 2,250편으로 검증**했다.

| 용도 | 인간 기사 ID | 실제 분류 대상 |
|---|---:|---|
| Detector 학습 | 750 | 인간 원문 750편 + Qwen 재서술 750편 + EXAONE 재서술 750편 |
| Detector 검증 | 2,250 | 학습에 사용하지 않은 기사들의 인간 원문과 두 모델의 재서술 |

Detector 학습에 사용한 텍스트는 총 **2,250개**다.
Human/AI 라벨은 출처로 정하며, detector가 이 라벨을 얼마나 잘 맞추는지를 선별 전에 검증한다.

검증 임계값 `P(AI) ≥ 0.5`에서 보고된 올바른 분류율은 다음과 같다.

| 글의 출처 | 올바른 분류율 |
|---|---:|
| 인간 원문 (TNR) | 85% |
| Qwen 재서술 (TPR) | 99% |
| EXAONE 재서술 (TPR) | 100% |

생성 실패를 제외한 분류율을 반올림해 표시했다. [검증 결과](../results/metrics/detector_validation.json)

고정된 detector로 **전체 후보 pair**를 검사하여 다음을 동시에 만족하는 쌍을 남겼다.

- 인간 원문: `P(AI) < 0.5`
- 대응하는 AI 재서술: `P(AI) ≥ 0.5`

Detector 학습에 쓰인 750개 기사 ID는 generator 데이터에서 제외한다.
Detector 검증용 기사 중 선별 조건을 통과한 기사는 generator 데이터 구성에 사용할 수 있다.
여러 번 구축한 pair를 합칠 때의 제외·중복 제거·분할 검사는 [build_clean_dataset.py](../scripts/build_clean_dataset.py)가 담당한다.

## 3. 최종 뉴스 generator 데이터

| Split | Pair 수 |
|---|---:|
| Train | 16,266 |
| Dev | 2,033 |
| Test | 2,044 |
| **전체** | **20,343** |

최종 데이터는 **원문 1편당 AI 재서술 1개**인 고유 pair이며, Qwen과 EXAONE 비중은 약 75:25다.
같은 원문의 파생 텍스트가 서로 다른 generator split에 들어가지 않도록 기사 ID를 기준으로 분리한다.

최종 test도 위 선별 조건을 통과한 집합으로, **해당 detector가 원래 구분할 수 있었던 데이터에서의 회피 성능**을 평가한다.
이 선별 과정에 따른 분포 편향은 결과를 일반화할 때의 한계다.
선별 임계값 0.5와 평가용 Human P95 임계값의 차이는 [방법론](methodology.md#5-evaluation)에 설명했다.

## 4. 최종 교차 도메인 평가의 6개 도메인

| 보고서 ID | 글의 종류 | 원자료 출처 | Target test 수 |
|---|---|---|---:|
| `news` | 뉴스 기사 | [국립국어원 모두의 말뭉치](https://kli.korean.go.kr/corpus/), 신문 2022판 | 2,044 |
| `essay` | 학생 에세이·작문 | [AI Hub 에세이 글 평가 데이터, 545](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=545) | 1,022 |
| `persona` | 페르소나 기반 대화 | [AI Hub 페르소나 대화, 71302](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71302) | 1,117 |
| `written` | 문학·수필 등 문어 자료 | 국립국어원 문어 말뭉치 v1.2 | 1,057 |
| `petition512B` | 국민청원 | [Blue House National Petition](https://huggingface.co/datasets/dev7halo/bluehouse-national-petition) | 1,019 |
| `wiki512B` | 백과사전 문서 | [Korean Wikipedia, 20231101.ko](https://huggingface.co/datasets/wikimedia/wikipedia) | 1,287 |

국민청원과 위키피디아는 짧은 입력으로 구성한 `petition512B`, `wiki512B` 트랙을 사용했다.
전체 결과는 [교차 도메인 보고서](../results/reports/cross_domain_current6_full_20260909.html)에 있다.

행은 generator의 학습 도메인, 열은 적용한 test와 detector의 도메인이다.
각 target의 동일한 test에 6개 source generator를 적용했다.
비교 대상은 도메인별 SFT와 첫 DPOP generator다.

## 5. Adversarial alignment의 별도 뉴스 집합

[Adversarial 실험](../adversarial/README.md)은 별도의 뉴스 데이터 **19,738쌍**을 사용한다.
분할은 train **15,780**, dev **1,973**, test **1,985**다.
G1과 G2는 같은 1,985개 test ID에서 비교하지만, 앞의 2,044편 뉴스 주 평가와 데이터·모델·평가 조건이 다르다.

사람 평가는 공통 기사 ID 50개에 대해 Human/G0/G1/G2 네 버전을 사용했다.
평가자 4명이 기사마다 서로 다른 버전을 하나씩 판단해, 버전별 50건·전체 200건의 응답을 얻었다.
설계와 집계는 [사람 평가 보고서](../results/reports/human_eval_results_share_20260907.md)에 있다.

## 6. 공개 범위와 재현에 필요한 자료

**인간 원문, 생성문, 원문을 포함한 프롬프트, 검사 제출 문서, 학습 가중치는 포함하지 않는다.**
각 데이터와 기반 모델은 위 출처에서 제공하는 이용 조건에 따라 별도로 확보해야 한다.

데이터는 `dataset/{domain}_track/`, 모델은 `models/`에 배치하거나 config에서 경로를 지정한다.
실험 결과를 재현하려면 해당 실험의 pair, split, detector와 generator checkpoint가 필요하다.
데이터 준비와 실행 명령은 [재현 안내](reproduction.md)를 참고한다.
