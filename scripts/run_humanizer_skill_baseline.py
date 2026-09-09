#!/usr/bin/env python3
"""Humanizer-skill 20편 파일럿과 100편 최종 기준선을 순서대로 실행합니다.

Qwen과 EXAONE의 OpenAI-compatible endpoint는 미리 실행되어 있어야 합니다. 생성
JSONL은 크기 단계 사이에서 공유하고 ``--resume``을 사용하므로 n=20 출력은 n=100에서
다시 생성하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value} {stamp()}\n", encoding="utf-8")
    print(value, stamp(), flush=True)


def run_checked(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, **kwargs)


def run_parallel(tasks: list[tuple[str, list[str], dict]], log_dir: Path) -> None:
    processes = []
    log_dir.mkdir(parents=True, exist_ok=True)
    for name, command, environment in tasks:
        log = (log_dir / f"{name}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, **environment},
        )
        processes.append((name, process, log))
    failures = []
    for name, process, log in processes:
        return_code = process.wait()
        log.close()
        if return_code:
            failures.append((name, return_code))
    if failures:
        raise RuntimeError(f"병렬 작업 실패: {failures}")


def generation_tasks(config: dict, size: int, prompts: Path,
                     generations: Path) -> list[tuple[str, list[str], dict]]:
    tasks = []
    generate = SCRIPTS / "generate.py"
    for engine in ("qwen", "exaone"):
        settings = config["generation"][engine]
        for shard, url in enumerate(settings["urls"]):
            name = f"{engine}_{shard}"
            command = [
                sys.executable, str(generate),
                "--in", str(prompts / f"n{size}" / f"{name}.jsonl"),
                "--out", str(generations / f"{name}.jsonl"),
                "--model", settings["model"], "--url", url,
                "--slots", str(settings.get("slots", 2)),
                "--temp", "0.1", "--token-ratio", "0.8", "--resume",
            ]
            if settings.get("no_think"):
                command.append("--no-think")
            tasks.append((f"n{size}_generate_{name}", command, {}))
    return tasks


def evaluation_tasks(config: dict, size: int, prompts: Path, generations: Path,
                     output: Path) -> list[tuple[str, list[str], dict]]:
    domains = list(config["domains"])
    gpus = config["evaluation"]["gpus"]
    if len(gpus) < len(domains):
        raise ValueError(f"도메인 {len(domains)}개에 사용할 GPU가 {len(gpus)}개뿐입니다.")
    generated = [
        generations / f"{engine}_{shard}.jsonl"
        for engine in ("qwen", "exaone")
        for shard in range(len(config["generation"][engine]["urls"]))
    ]
    tasks = []
    for index, domain in enumerate(domains):
        settings = config["domains"][domain]
        command = [
            sys.executable, str(SCRIPTS / "eval_humanizer_skill_baseline.py"),
            "--domain", domain,
            "--pairs", str(resolve(settings["pairs"])),
            "--generated", *map(str, generated),
            "--detector", str(resolve(settings["detector"])),
            "--scrn", str(resolve(settings["scrn"])),
            "--manifest", str(prompts / f"n{size}" / "manifest.jsonl"),
            "--n", str(size),
            "--out", str(output / f"n{size}" / f"{domain}.jsonl"),
            "--batch-size", str(config["evaluation"].get("batch_size", 16)),
            "--device", "cuda:0",
        ]
        if settings.get("dialogue"):
            command.append("--dialogue")
        tasks.append((
            f"n{size}_eval_{domain}", command,
            {"CUDA_VISIBLE_DEVICES": str(gpus[index])},
        ))
    return tasks


def summarize(config: dict, size: int, output: Path, gate: bool) -> int:
    command = [
        sys.executable, str(SCRIPTS / "summarize_humanizer_skill_baseline.py"),
        "--eval-dir", str(output / f"n{size}"),
        "--transfer-dir", str(resolve(config["reference_dir"])),
        "--domains", *config["domains"].keys(),
        "--n", str(size),
        "--out", str(output / f"result_n{size}.md"),
        "--gate-json", str(output / f"gate_n{size}.json"),
    ]
    if gate:
        command.append("--gate")
    return subprocess.run(command).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prompts = resolve(config["prompt_output_dir"])
    generations = resolve(config["generation_output_dir"])
    output = resolve(config["evaluation_output_dir"])
    status = output / "STATUS"
    generations.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    try:
        write_status(status, "PREPARING")
        run_checked([
            sys.executable, str(SCRIPTS / "build_humanizer_skill_baseline.py"),
            "--config", str(args.config), "--sizes", "20", "100",
            "--qwen-shards", str(len(config["generation"]["qwen"]["urls"])),
        ])
        for size in (20, 100):
            write_status(status, f"RUNNING stage=n{size}_generation")
            run_parallel(generation_tasks(config, size, prompts, generations), output / "logs")
            write_status(status, f"RUNNING stage=n{size}_evaluation")
            run_parallel(
                evaluation_tasks(config, size, prompts, generations, output), output / "logs"
            )
            return_code = summarize(config, size, output, gate=(size == 20))
            if size == 20 and return_code == 2:
                write_status(status, "PILOT_REJECTED n=20")
                return 2
            if return_code:
                raise RuntimeError(f"n={size} 집계 실패: exit {return_code}")
            if size == 20:
                write_status(status, "PILOT_PASSED n=20 advancing=n100")
        write_status(status, "DONE n20=passed n100=complete")
        return 0
    except Exception as exc:
        write_status(status, f"FAILED error={type(exc).__name__}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
