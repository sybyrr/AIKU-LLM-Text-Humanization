# loop/ — Generator↔Detector 반복 공진화 구현

설계 정본은 [notes/31-학습-루프-설계](../notes/31-학습-루프-설계.md). 이 폴더는 그 설계의 실행 코드다.
연구 배경·판정표·arm 근거는 그 노트를 읽는다 — 여기 중복하지 않는다.

## 무엇을 학습하나

동결 생성기(Qwen3-8B, llama.cpp)가 만든 AI 초록을 **재서술기 G**(ko-BART 124M)가 인간화하고,
**대리 탐지기 D**(KLUE-RoBERTa 110M)가 학습 신호를 준다. 둘을 라운드로 잇는다.
헤드라인 수치는 루프 밖 **카피킬러 전이**(대리 탐지기 = shadow model).

## 산출물 배치

```
loop/
  configs/base.yaml        모든 고정값 (notes/31 값이 여기 있다)
  configs/arms/*.yaml       레인 0~3 오버라이드
  loop_lib/                 공통 모듈 (io·config·data·detector·paraphraser·sft·dpo·replay·metrics)
  scripts/                  실행 스크립트 (s0/s1/s2 + r_* 라운드 + run_*.sh + probe)
  smoke/                    CPU 스모크 (tiny 모델로 전 단계 관통)
  runs/                     ← 모든 산출물 (gitignore). 코드·설정만 커밋된다
    stage0/ stage1/ stage2/ arms/<arm>/round<t>/{candidates,prefs,dpo,eval,detector}/
    probes/ logs/
```

동결 앵커는 `runs/stage0/`(D₀·KoELECTRA)에 있고 라운드 산출물은 `runs/arms/` 아래라
서로 절대 덮이지 않는다.

## 서버 실행 순서

`notes/31 § 서버에서 코드 한 줄 쓰기 전에` — 먼저 환경부터. 실패하면 requirements 전체가 바뀐다.

```bash
/workspace/.venv/bin/python3 loop/check_env.py       # capability (6,1) + 실제 matmul 확인
```

착수 준비 (D₀ → D_pair → SFT). Pascal 은 fp32 전용이라 카드당 12GB 안에 든다:

```bash
CUDA_VISIBLE_DEVICES=0 bash loop/scripts/run_stage0_2.sh
```

`runs/stage2/samples_dev.md` 20건을 눈으로 확인한다 — ko-BART 가 내용을 지키며 한국어로
쓰면 GO. 깨지면 notes/31 의 Qwen3-0.6B 대안으로 간다.

라운드 (arm 하나당 카드 하나, 병렬 — 벽시계는 단일 arm과 같다):

```bash
CUDA_VISIBLE_DEVICES=0 bash loop/scripts/run_arm.sh main        1 5 &
CUDA_VISIBLE_DEVICES=1 bash loop/scripts/run_arm.sh self_anchor 1 5 &
CUDA_VISIBLE_DEVICES=2 bash loop/scripts/run_arm.sh continual   1 5 &
CUDA_VISIBLE_DEVICES=3 bash loop/scripts/run_arm.sh latest_only 1 5 &
wait
```

⚠ GPU 0~3 은 타인 사용 중일 수 있다 ([progress.md 알려진 문제](../progress.md)) — `nvidia-smi` 로
빈 카드를 확인하고 `CUDA_VISIBLE_DEVICES` 를 맞춘다. 호스트는 `gpu1` 컨테이너에 4·5 할당.

한 라운드 = 5단계. `run_round.sh` 가 순서와 게이트를 강제한다:

| 단계 | 스크립트 | 산출 |
| --- | --- | --- |
| ① 후보 생성 | `r_generate.py` | `round<t>/candidates.jsonl.gz` (원문당 n=8 샘플) |
| ② preference | `r_build_prefs.py` | `round<t>/prefs.jsonl.gz` (+ ref logp 사전계산) |
| ③ DPO | `r_dpo.py` | `round<t>/dpo/final` = G_t |
| ④ 평가 | `r_eval.py` | `round<t>/eval/{metrics.json,report.md,ck_export/}` |
| ⑤ 탐지기 재학습 | `r_retrain_detector.py` | `round<t>/detector/{model,tau.json,gate.json}` = D_{t+1} |

**평가(④)는 탐지기 재학습(⑤) 전에 돈다** — 재학습 후 지표를 재면 순환이다.
hard negative 가 부족하면 ②가 `needs_more` 를 남기고, `run_round.sh` 가 후보를 증량해(τ는
그대로) ②를 한 번 재실행한다.

## 게이트 (탐지기 붕괴)

⑤가 매 라운드 `gate.json` 을 쓴다. **dev-gate**(캘리브레이션에 안 쓴 인간 절반)에서
`FPR > 0.08` 또는 `원본 AI TPR < 0.80` 이면 붕괴로 기록하고 `run_arm.sh` 가 그 arm 을 멈춘다.
붕괴는 실패가 아니라 결과다 (notes/31 판정표: MASH Appendix D 는 1라운드 만에 99.74%→30.59%).
붕괴 궤적을 계속 보고 싶으면 `IGNORE_GATE=1`.

τ 는 dev 를 md5 로 반 가른 **dev-cal** 에서 FPR 5% 로 잡고, 게이트는 **dev-gate** 로 잰다.
같은 표본으로 캘리브레이션하고 게이트를 재면 FPR 5% 가 구성상 보장돼 게이트가 무의미해진다.

## 판정 (notes/31 판정표를 코드가 읽는 위치)

`round<t>/eval/metrics.json` 의 `detectors` 블록:
- `in_loop_dt` — **증거 아님** (순환). 참고용
- `frozen_d0`, `frozen_koelectra` — 루프에 안 들어간 동결 앵커. 여기 ASR 이 헤드라인
- `style.distance_gen` — 인간 문체와의 z-거리 (탐지기와 무관한 축)

카피킬러(주지표)는 `eval/ck_export/batch*/` 를 웹 UI 에 올려 받은 AI작성률로 별도 채점한다
(배치당 350개, `manifest.csv` 는 업로드 금지 — 정답표).

## 외부 프로브 (GPU 레인 불필요, 생성만)

라운드 착수 전 1시간짜리 진단 — 다양화 필요 여부를 실측이 결정한다 (notes/31 § 생성기 다양화):

```bash
# OOD: held-out 생성기(kanana 등)가 만든 x_ai 를 D₀ 가 잡는가
python loop/scripts/s0_probe_external.py --gen data/gen_ood.jsonl --name ood --ref-source test --role detect
# 모드B: 처음부터 쓴 AI 초록을 D₀ 가 잡는가 (재서술 편향 ⓑ)
python loop/scripts/s0_probe_external.py --gen <모드B 생성물> --name modeB --ref-source eval_bodies --role detect
# P4: 지문 명시 프롬프트가 학습 없이 인간화하는가 (학습 우위 반증 방어)
python loop/scripts/build_p4_prompts.py --out loop/runs/probes/prompts_p4.jsonl   # 생성기로 돌린 뒤
python loop/scripts/s0_probe_external.py --gen loop/runs/probes/gen_p4.jsonl --name p4 --ref-source test --role evade
```

## 로컬 스모크 (GPU 없이 코드 관통)

tiny 랜덤 모델로 전 파이프라인이 끝까지 도는지만 본다 (수렴은 서버에서). 수 분 소요:

```bash
python loop/smoke/run_smoke.py
```

## 규약

- 파이썬은 서버에서 반드시 `/workspace/.venv/bin/python3`. 러너가 없으면 `python3` 폴백.
- 모든 파일 IO 는 UTF-8 명시 (컨테이너 로케일 POSIX). 재현용 seed 는 base.yaml.
- git 으로는 코드만 왕복. `runs/` 는 gitignore — 체크포인트·생성물·점수는 올리지 않는다.
- 재실행 안전: 각 단계는 산출물이 있으면 건너뛴다. 다시 하려면 `--force`.
