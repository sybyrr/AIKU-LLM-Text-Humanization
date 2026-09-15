"""공통 IO — 모든 파일 IO 는 UTF-8 명시 (서버 컨테이너 로케일이 POSIX 라 기본값이 깨진다)."""
import gzip
import json
import os
import random
from pathlib import Path


def _open_r(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _open_w(path, gz=None):
    """gz=None 이면 경로 확장자로 판단. 임시파일(.tmp)에 쓸 때는 최종 경로 기준으로
    gz 를 명시해야 한다 — 확장자 sniff 에 맡기면 foo.gz.tmp 가 평문으로 저장된다."""
    path = str(path)
    if gz is None:
        gz = path.endswith(".gz")
    if gz:
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8", newline="\n")


def read_jsonl(path):
    with _open_r(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def iter_jsonl(path):
    with _open_r(path) as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def write_jsonl(path, rows):
    """임시 파일에 쓴 뒤 rename — 중단돼도 반쪽짜리 파일이 남지 않는다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gz = str(path).endswith(".gz")
    tmp = path.with_name(path.name + ".tmp")
    with _open_w(tmp, gz=gz) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "at" if str(path).endswith(".gz") else "a"
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(str(path), mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(name=None):
    import torch

    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
