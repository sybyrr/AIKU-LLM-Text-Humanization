#!/usr/bin/env python3
"""서버 프리플라이트 — 코드 한 줄 돌리기 전에 실행한다 (notes/31 § 로컬 ↔ 서버 분업).

검사 항목:
  ① torch + CUDA: TITAN Xp 는 sm_61(Pascal)이라 최신 휠에서 커널이 빠질 수 있다.
     capability 가 (6,1) 로 나오고 **실제 행렬곱이 값을 뱉어야** 진짜다.
     여기서 실패하면 cu118 계열로 내려야 하고 requirements 전체가 그 결정을 따른다.
  ② 필수 라이브러리 import + 버전
  ③ 선택한 profile의 데이터·초기 체크포인트 존재와 행수
  ④ UTF-8 왕복 (컨테이너 로케일이 POSIX 라 한글이 깨질 수 있음)

사용:
  python loop/check_env.py                                # KCI loop, 활성 GPU 환경
  python loop/check_env.py --profile team-news --team-root .
  python loop/check_env.py --cpu-only --profile none      # 로컬 코드 스모크 전
종료코드 0 = 전부 통과.
"""
import argparse
import json
import locale
import os
import sys
import tempfile
from pathlib import Path

# 서버 POSIX 로케일·로컬 cp949 양쪽에서 유니코드 출력이 죽지 않게 stdout 을 UTF-8 로 고정
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]

DATA_EXPECT = {"data/splits_v2.jsonl": 9115}
DATA_EXIST = [
    "data/human_pool.jsonl.gz",
    "data/gen_full_p3b_clean.jsonl.gz",
    "data/kci_eval_bodies.jsonl.gz",
]
DATA_OPTIONAL = {"data/prompts_ood_test.jsonl": 920}

ok_all = True


def report(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    mark = "OK " if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true", help="GPU 검사 생략 (로컬)")
    ap.add_argument(
        "--profile",
        choices=("kci", "team-news", "none"),
        default="kci",
        help="검사할 비공개 입력 묶음 (기본: kci)",
    )
    ap.add_argument(
        "--team-root",
        type=Path,
        default=ROOT,
        help="team-news profile의 dataset/ 및 models/가 있는 루트",
    )
    a = ap.parse_args()

    print(f"python  : {sys.version.split()[0]}  ({sys.executable})")
    print(f"locale  : {locale.getpreferredencoding(False)}  LANG={os.environ.get('LANG', '(비어 있음)')}")
    print(f"repo    : {ROOT}")
    print()

    # ── ① torch / CUDA ──────────────────────────────────────────
    try:
        import torch

        report("torch import", True, f"{torch.__version__} (cuda={torch.version.cuda})")
        if a.cpu_only:
            report("GPU 검사", True, "--cpu-only 로 생략")
        elif not torch.cuda.is_available():
            report("cuda available", False, "torch.cuda.is_available() == False — GPU 컨테이너인지, CUDA 휠인지 확인")
        else:
            n = torch.cuda.device_count()
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                try:
                    x = torch.randn(64, 64, device=f"cuda:{i}")
                    v = (x @ x).sum().item()
                    real = v == v  # NaN 검사
                    report(f"cuda:{i} {name}", real, f"capability={cap}, matmul={v:.1f}")
                    if cap == (6, 1):
                        print("        └ Pascal(sm_61) — fp16 은 fp32의 1/64 이므로 전 파이프라인 fp32 고정 (설계 반영됨)")
                except RuntimeError as e:
                    report(f"cuda:{i} {name}", False, f"행렬곱 실패: {e} → cu118 폴백 필요 (requirements.txt 주석)")
    except Exception as e:
        report("torch import", False, str(e))

    # ── ② 라이브러리 ────────────────────────────────────────────
    for mod, why in [
        ("transformers", "모델 전부"),
        ("yaml", "설정"),
        ("numpy", "수치"),
        ("sklearn", "선형 프로브·AUC"),
        ("tqdm", "진행 표시"),
    ]:
        try:
            m = __import__(mod)
            report(f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:
            report(f"import {mod}", False, f"{e} ({why})")
    for mod, why in [("kiwipiepy", "문체 특징·고유명사 보존 — 없으면 해당 지표 생략"),
                     ("docx", "카피킬러 export — 없으면 export 생략")]:
        try:
            m = __import__(mod)
            report(f"import {mod}", True, getattr(m, "__version__", "") + " (선택)")
        except Exception as e:
            print(f"[warn] import {mod} 실패 — {why} ({e})")

    # ── ③ 데이터·초기 체크포인트 ────────────────────────────────
    if a.profile == "kci":
        for rel, expect in DATA_EXPECT.items():
            p = ROOT / rel
            if not p.exists():
                report(rel, False, "파일 없음")
                continue
            n = sum(1 for _ in open(p, encoding="utf-8"))
            report(rel, n == expect, f"{n}행 (기대 {expect})")
        for rel in DATA_EXIST:
            p = ROOT / rel
            report(rel, p.exists(), f"{p.stat().st_size/1e6:.1f}MB" if p.exists() else "파일 없음")
        for rel, expect in DATA_OPTIONAL.items():
            p = ROOT / rel
            if not p.exists():
                print(f"[warn] {rel} 없음 — OOD 외부 프로브만 생략")
                continue
            n = sum(1 for _ in open(p, encoding="utf-8"))
            report(rel, n == expect, f"{n}행 (기대 {expect}, 선택)")
    elif a.profile == "team-news":
        team = a.team_root.resolve()
        pairs = team / "dataset/news_track/news_dpair_D2.jsonl"
        generator = team / "models/news_dpo_D2/dpo_bart.pt"
        detector = team / "models/news_roberta_D2"
        if pairs.is_file():
            n = sum(1 for line in pairs.open(encoding="utf-8") if line.strip())
            report("team-news pairs", n == 19738, f"{n}행 (기대 19738): {pairs}")
        else:
            report("team-news pairs", False, f"파일 없음: {pairs}")
        report("team-news G1", generator.is_file(), str(generator))
        detector_weights = [detector / "model.safetensors", detector / "pytorch_model.bin"]
        detector_ok = (detector / "config.json").is_file() and any(p.is_file() for p in detector_weights)
        report("team-news detector", detector_ok, str(detector))
    else:
        print("[warn] --profile none — 비공개 데이터·체크포인트 검사를 생략")

    # ── ④ UTF-8 왕복 ────────────────────────────────────────────
    try:
        probe = {"한글": "탐지기와 재서술기의 공진화", "값": 0.126}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(probe, f, ensure_ascii=False)
            tmp = f.name
        back = json.load(open(tmp, encoding="utf-8"))
        os.unlink(tmp)
        report("UTF-8 왕복", back == probe)
    except Exception as e:
        report("UTF-8 왕복", False, str(e))

    print()
    print("전부 통과 — 진행 가능" if ok_all else "실패 항목 있음 — loop/requirements.txt 주석과 notes/31 § 서버 확인 참조")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
