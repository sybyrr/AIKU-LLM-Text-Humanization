"""설정 로드 — base.yaml 위에 arm yaml 을 깊은 병합. 경로는 저장소 루트 기준으로 해석."""
import copy
from pathlib import Path

import yaml

# loop/loop_lib/config.py → 저장소 루트는 두 단계 위
REPO_ROOT = Path(__file__).resolve().parents[2]
LOOP_DIR = Path(__file__).resolve().parents[1]


class NS:
    """dict 를 점 표기로 읽는 얇은 래퍼. cfg.detector.lr 처럼 쓴다."""

    def __init__(self, d):
        self._d = d

    def __getattr__(self, k):
        try:
            v = self._d[k]
        except KeyError:
            raise AttributeError(f"설정 키 없음: {k} (있는 키: {sorted(self._d)})") from None
        return NS(v) if isinstance(v, dict) else v

    def __getitem__(self, k):
        return self._d[k]

    def __contains__(self, k):
        return k in self._d

    def get(self, k, default=None):
        v = self._d.get(k, default)
        return NS(v) if isinstance(v, dict) else v

    def to_dict(self):
        return copy.deepcopy(self._d)

    def __repr__(self):
        return f"NS({self._d!r})"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(base_path=None, *overlay_paths) -> NS:
    base_path = Path(base_path) if base_path else LOOP_DIR / "configs" / "base.yaml"
    d = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))
    for p in overlay_paths:
        if not p:
            continue
        d = _merge(d, yaml.safe_load(Path(p).read_text(encoding="utf-8")))
    return NS(d)


def rp(rel) -> Path:
    """repo-relative → 절대 경로. 이미 절대면 그대로."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def runs_dir(cfg) -> Path:
    return rp(cfg.paths.runs)


# ── 산출물 경로 규약 (스크립트들이 전부 이 함수만 쓴다) ────────────
def stage0_dir(cfg) -> Path:
    return runs_dir(cfg) / "stage0"


def stage1_dpair(cfg) -> Path:
    return runs_dir(cfg) / "stage1" / "dpair.jsonl.gz"


def stage2_dir(cfg) -> Path:
    return runs_dir(cfg) / "stage2"


def arm_dir(cfg, arm=None) -> Path:
    return runs_dir(cfg) / "arms" / (arm or cfg.arm.name)


def round_dir(cfg, t, arm=None) -> Path:
    return arm_dir(cfg, arm) / f"round{t}"


def generator_in(cfg, t, arm=None) -> Path:
    """라운드 t 가 시작할 때 쓰는 재서술기 G_{t-1} 의 경로."""
    if t <= 1:
        initial = cfg.paths.get("initial_generator")
        if initial:
            return rp(initial)
        return stage2_dir(cfg) / "best"
    return round_dir(cfg, t - 1, arm) / "dpo" / "final"


def detector_in(cfg, t, arm=None) -> Path:
    """라운드 t 의 신호원 탐지기 D_t 의 경로 (τ 파일은 같은 폴더의 tau.json)."""
    if t <= 1:
        initial = cfg.paths.get("initial_detector")
        if initial:
            return rp(initial)
        return stage0_dir(cfg) / "detector_d0"
    return round_dir(cfg, t - 1, arm) / "detector" / "model"


def tau_of(detector_dir: Path) -> Path:
    return Path(detector_dir).parent / "tau.json" if Path(detector_dir).name == "model" \
        else Path(detector_dir) / "tau.json"
