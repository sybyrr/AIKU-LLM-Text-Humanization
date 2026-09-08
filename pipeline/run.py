#!/usr/bin/env python3
"""Run the canonical Stage 1b-3 humanization pipeline from one JSON config.

This module is deliberately an orchestrator.  Training and data logic remains in
``scripts/``; this file owns path resolution, resume checks, logging, and the
exact canonical arguments passed between stages.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
EVALUATOR = Path(__file__).with_name("evaluate.py")
STAGES = ("prompts", "generate", "gate", "sft", "hard-negative", "dpo", "evaluate")


class ConfigError(ValueError):
    """Raised when a pipeline config is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class Paths:
    workspace: Path
    human_pool: Path
    prompts: Path
    pairs: Path
    dpo_pairs: Path
    detector: Path
    sft_dir: Path
    dpo_dir: Path
    evaluation: Path
    generator_outputs: dict[str, Path]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    if not config.get("domain"):
        raise ConfigError("domain is required")
    for key in ("paths", "generators", "gate", "sft", "hard_negative", "dpo"):
        if key not in config:
            raise ConfigError(f"missing config section: {key}")
    generators = config["generators"]
    if not generators:
        raise ConfigError("at least one generator is required")
    if any(not isinstance(item, dict) for item in generators):
        raise ConfigError("each generator must be an object")
    names = [item.get("name") for item in generators]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ConfigError("generator names must be non-empty and unique")
    for index, item in enumerate(generators):
        missing = {"name", "model", "url", "output"} - set(item)
        if missing:
            raise ConfigError(
                f"generator {index} is missing: {', '.join(sorted(missing))}"
            )
    if int(config["hard_negative"].get("no_repeat_ngram_size", 3)) < 0:
        raise ConfigError("hard_negative.no_repeat_ngram_size must be >= 0")
    if int(config.get("evaluation", {}).get("sample_size", 0)) < 0:
        raise ConfigError("evaluation.sample_size must be >= 0")
    return config


def resolve_path(value: str, workspace: Path, domain: str) -> Path:
    rendered = value.format(domain=domain)
    path = Path(os.path.expandvars(rendered)).expanduser()
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def resolve_paths(config: dict[str, Any], config_path: Path, override: str | None) -> Paths:
    domain = config["domain"]
    workspace_value = override if override is not None else config.get("workspace", ".")
    workspace = Path(os.path.expandvars(workspace_value)).expanduser()
    if override is not None:
        workspace = workspace.resolve()
    elif not workspace.is_absolute():
        workspace = (config_path.parent / workspace).resolve()
    else:
        workspace = workspace.resolve()
    raw = config["paths"]
    required = ("human_pool", "prompts", "pairs", "dpo_pairs", "detector", "sft", "dpo", "evaluation")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ConfigError(f"missing paths: {', '.join(missing)}")
    generator_outputs = {
        item["name"]: resolve_path(item["output"], workspace, domain)
        for item in config["generators"]
    }
    return Paths(
        workspace=workspace,
        human_pool=resolve_path(raw["human_pool"], workspace, domain),
        prompts=resolve_path(raw["prompts"], workspace, domain),
        pairs=resolve_path(raw["pairs"], workspace, domain),
        dpo_pairs=resolve_path(raw["dpo_pairs"], workspace, domain),
        detector=resolve_path(raw["detector"], workspace, domain),
        sft_dir=resolve_path(raw["sft"], workspace, domain),
        dpo_dir=resolve_path(raw["dpo"], workspace, domain),
        evaluation=resolve_path(raw["evaluation"], workspace, domain),
        generator_outputs=generator_outputs,
    )


def jsonl_keys(path: Path, fields: Iterable[str], require_text: bool = False) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    if not path.is_file():
        return keys
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"invalid JSONL: {path}:{number}: {exc}") from exc
            if require_text and not row.get("text"):
                continue
            keys.add(tuple(row.get(field) for field in fields))
    return keys


def generation_complete(prompts: Path, output: Path) -> bool:
    expected = jsonl_keys(prompts, ("doc_id", "cond"))
    completed = jsonl_keys(output, ("doc_id", "cond"), require_text=True)
    return bool(expected) and expected <= completed


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {command_text(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] $ {command_text(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def stage_slice(first: str, last: str) -> tuple[str, ...]:
    start, end = STAGES.index(first), STAGES.index(last)
    if start > end:
        raise ConfigError("--from-stage must not come after --to-stage")
    return STAGES[start:end + 1]


def build_commands(config: dict[str, Any], paths: Paths, python: str) -> dict[str, list[tuple[str, list[str]]]]:
    domain = config["domain"]
    gate = config["gate"]
    sft = config["sft"]
    hard = config["hard_negative"]
    dpo = config["dpo"]
    commands: dict[str, list[tuple[str, list[str]]]] = {}

    commands["prompts"] = [("prompts", [
        python, str(SCRIPTS / "build_domain_prompts.py"),
        "--in", str(paths.human_pool), "--out", str(paths.prompts), "--domain", domain,
    ])]

    generate_commands = []
    for item in config["generators"]:
        command = [
            python, str(SCRIPTS / "generate.py"),
            "--in", str(paths.prompts), "--out", str(paths.generator_outputs[item["name"]]),
            "--url", item["url"], "--model", item["model"],
            "--slots", str(item.get("slots", 4)),
            "--temp", str(item.get("temperature", 0.8)),
            "--seed", str(item.get("seed", 42)),
            "--token-ratio", str(item.get("token_ratio", 0.8)), "--resume",
        ]
        if item.get("no_think", False):
            command.append("--no-think")
        generate_commands.append((f"generate-{item['name']}", command))
    commands["generate"] = generate_commands

    gate_command = [
        python, str(SCRIPTS / "domain_gate.py"),
        "--pool", str(paths.human_pool), "--out", str(paths.pairs),
        "--model-out", str(paths.detector),
        "--det-n", str(gate.get("detector_documents", 750)),
        "--tau", str(gate.get("threshold", 0.5)),
        "--epochs", str(gate.get("epochs", 3)),
        "--bs", str(gate.get("batch_size", 16)),
        "--split-ratio", str(gate.get("split_ratio", "0.8,0.1,0.1")),
        "--det-select", str(gate.get("detector_group_selection", "large")),
    ]
    for item in config["generators"]:
        gate_command.extend(["--gen", f"{item['name']}={paths.generator_outputs[item['name']]}" ])
    if gate.get("split_key"):
        gate_command.extend(["--split-key", str(gate["split_key"])])
    if gate.get("stratify"):
        gate_command.append("--stratify")
    if gate.get("dialogue"):
        gate_command.append("--dialogue")
    commands["gate"] = [("gate", gate_command)]

    commands["sft"] = [("sft", [
        python, str(SCRIPTS / "stage2_sft.py"),
        "--pairs", str(paths.pairs), "--out-dir", str(paths.sft_dir),
        "--epochs", str(sft.get("epochs", 3)), "--bs", str(sft.get("batch_size", 8)),
        "--lr", str(sft.get("learning_rate", 2e-5)), "--lam", str(sft.get("lambda", 0.5)),
        "--maxlen", str(sft.get("max_length", 512)), "--split", "train", "--resume",
    ])]

    commands["hard-negative"] = [("hard-negative", [
        python, str(SCRIPTS / "stage3_build_dpo.py"),
        "--sft", str(paths.sft_dir / "style_bart.pt"),
        "--pairs", str(paths.pairs), "--roberta", str(paths.detector),
        "--out", str(paths.dpo_pairs), "--split", "train",
        "--tau", str(hard.get("threshold", 0.5)),
        "--n-sample", str(hard.get("samples_per_input", 4)),
        "--batch", str(hard.get("batch_size", 32)),
        "--no-repeat-ngram-size", str(hard.get("no_repeat_ngram_size", 3)),
    ])]

    commands["dpo"] = [("dpo", [
        python, str(SCRIPTS / "stage3_dpo.py"),
        "--dpo-pairs", str(paths.dpo_pairs), "--sft", str(paths.sft_dir / "style_bart.pt"),
        "--out-dir", str(paths.dpo_dir), "--dpop", "--length-norm",
        "--dpop-lambda", str(dpo.get("dpop_lambda", 5.0)),
        "--beta", str(dpo.get("beta", 2.0)), "--epochs", str(dpo.get("epochs", 1)),
        "--bs", str(dpo.get("batch_size", 2)),
        "--accum", str(dpo.get("gradient_accumulation", 8)),
        "--lr", str(dpo.get("learning_rate", 5e-6)),
        "--maxlen", str(dpo.get("max_length", 512)), "--seed", str(dpo.get("seed", 42)),
        "--eval-pairs", str(paths.pairs), "--roberta", str(paths.detector),
    ])]

    evaluation = config.get("evaluation", {})
    evaluate_command = [
        python, str(EVALUATOR), "--domain", domain,
        "--pairs", str(paths.pairs), "--detector", str(paths.detector),
        "--sft", str(paths.sft_dir / "style_bart.pt"),
        "--dpo", str(paths.dpo_dir / "dpo_bart.pt"), "--out", str(paths.evaluation),
        "--n", str(evaluation.get("sample_size", 0)), "--seed", str(evaluation.get("seed", 42)),
        "--num-beams", str(evaluation.get("num_beams", 4)),
        "--input-max-length", str(evaluation.get("input_max_length", 512)),
        "--output-max-length", str(evaluation.get("output_max_length", 1024)),
        "--generation-batch-size", str(evaluation.get("generation_batch_size", 8)),
        "--scoring-batch-size", str(evaluation.get("scoring_batch_size", 16)),
    ]
    if evaluation.get("scrn"):
        evaluate_command.extend(["--scrn", str(resolve_path(evaluation["scrn"], paths.workspace, domain))])
    if gate.get("dialogue"):
        evaluate_command.append("--dialogue")
    commands["evaluate"] = [("evaluate", evaluate_command)]
    return commands


def artifact_complete(stage: str, label: str, config: dict[str, Any], paths: Paths) -> bool:
    if stage == "prompts":
        return (
            line_count(paths.prompts) > 0
            and line_count(paths.prompts) == line_count(paths.human_pool)
        )
    if stage == "generate":
        name = label.removeprefix("generate-")
        return generation_complete(paths.prompts, paths.generator_outputs[name])
    if stage == "gate":
        return line_count(paths.pairs) >= int(config["gate"].get("minimum_pairs", 50)) \
            and (paths.detector / "config.json").is_file()
    if stage == "sft":
        return (paths.sft_dir / "style_bart.pt").is_file()
    if stage == "hard-negative":
        summary = paths.dpo_pairs.with_suffix(".summary.json")
        return paths.dpo_pairs.is_file() and summary.is_file()
    if stage == "dpo":
        return (paths.dpo_dir / "dpo_bart.pt").is_file()
    if stage == "evaluate":
        summary = paths.evaluation.with_suffix(".summary.json")
        return line_count(paths.evaluation) > 0 and summary.is_file()
    raise AssertionError(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", help="override config workspace")
    parser.add_argument("--gpu", default="0", help="physical GPU exposed as cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument(
        "--force", action="store_true",
        help="invoke stages even when artifacts exist; scripts may still resume/reuse their caches",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print commands only")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    paths = resolve_paths(config, config_path, args.workspace)
    selected = stage_slice(args.from_stage, args.to_stage)
    commands = build_commands(config, paths, args.python)

    needs_human_pool = any(stage in selected for stage in ("prompts", "generate", "gate"))
    if not args.dry_run and needs_human_pool and not paths.human_pool.is_file():
        raise ConfigError(f"human pool not found: {paths.human_pool}")

    log_dir = paths.workspace / "logs" / config["domain"] / "pipeline"
    status_path = log_dir / "status.json"
    status: dict[str, Any] = {
        "domain": config["domain"], "config": str(config_path),
        "workspace": str(paths.workspace), "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {}, "state": "dry-run" if args.dry_run else "running",
    }
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")

    for stage in selected:
        if stage == "evaluate" and not config.get("evaluation", {}).get("enabled", True):
            status["stages"][stage] = "disabled"
            continue
        if stage == "dpo":
            minimum = int(config["hard_negative"].get("minimum_pairs", 30))
            if not args.dry_run and line_count(paths.dpo_pairs) < minimum:
                status["stages"][stage] = f"skipped: fewer than {minimum} hard negatives"
                status["state"] = "sft-only"
                write_status(status_path, status)
                print(status["stages"][stage])
                return 0
        for label, command in commands[stage]:
            if not args.force and not args.dry_run and artifact_complete(stage, label, config, paths):
                print(f"[{label}] complete; skip")
                status["stages"][label] = "skipped-complete"
                continue
            if args.dry_run:
                print(f"[{label}] {command_text(command)}")
                status["stages"][label] = "planned"
                continue
            try:
                run_command(command, log_dir / f"{label}.log", env)
                if not artifact_complete(stage, label, config, paths):
                    raise RuntimeError(f"{label} finished without a complete artifact")
                status["stages"][label] = "complete"
                write_status(status_path, status)
            except Exception:
                status["stages"][label] = "failed"
                status["state"] = "failed"
                status["finished_at"] = datetime.now(timezone.utc).isoformat()
                write_status(status_path, status)
                raise

    status["state"] = "complete" if not args.dry_run else "dry-run"
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    if not args.dry_run:
        write_status(status_path, status)
    print(f"pipeline {status['state']}: {config['domain']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
