#!/usr/bin/env python3
"""Run the imported news G1 -> G2 pipeline, stopping after G2 evaluation.

Run inside the existing Linux GPU container. --smoke uses eight long training
documents and an isolated output tree; it never changes production artifacts.
The detector threshold and DPO objective are inherited without modification.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop_lib import config as C
from loop_lib import io_utils as io
import yaml


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    original = C.load_config(C.LOOP_DIR / 'configs/base.yaml',
                             C.LOOP_DIR / 'configs/arms/team_news.yaml')
    production = C.runs_dir(original)
    root = production / 'g2_preflight' if args.smoke else production
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'g2_pipeline.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another G2 pipeline is already running')

    detector = C.detector_in(original, 1)
    gate = io.read_json(detector.parent / 'gate.json')
    tau = io.read_json(C.tau_of(detector))['tau']
    require(gate['collapsed'] is False, 'D1 gate has collapsed')
    require(gate['gate_human_fpr'] <= original.gate.fpr_max and
            gate['gate_orig_ai_tpr'] >= original.gate.tpr_min, 'D1 gate limits failed')
    require(tau == gate['tau'], 'D1 gate and calibrated threshold disagree')
    require((detector / 'model.safetensors').is_file(), 'D1 checkpoint missing')
    generator = C.generator_in(original, 1)
    require((generator / 'style_head.pt').is_file(), 'G1 checkpoint missing')
    rows = io.read_jsonl(C.stage1_dpair(original))
    train = {r['doc_id']: r for r in rows if r['split'] == 'train'}
    held = {r['doc_id'] for r in rows if r['split'] != 'train'}
    require(len(train) == 15780, 'Unexpected train count')
    require(not set(train) & held, 'Train/held-out ID overlap')

    config = original.to_dict()
    config['paths']['runs'] = str(root)
    config['paths']['initial_generator'] = str(generator)
    config['paths']['initial_detector'] = str(detector)
    if args.smoke:
        selected = sorted(train.values(), key=lambda r: len(r['ai_text']), reverse=True)[:8]
        train = {r['doc_id']: r for r in selected}
        io.write_jsonl(root / 'stage1/dpair.jsonl.gz', selected)
        anchor = root / 'stage0'
        if not anchor.exists():
            anchor.symlink_to(production / 'stage0', target_is_directory=True)
    cfg = C.NS(config)
    out = C.round_dir(cfg, 1)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'g2_config.yaml'
    if config_path.exists():
        require(yaml.safe_load(config_path.read_text(encoding='utf-8')) == config,
                'Saved G2 configuration changed; use a separate run directory')
    else:
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    scripts = C.LOOP_DIR / 'scripts'
    hashed_paths = [Path(__file__), scripts / 'r_generate.py', scripts / 'r_build_prefs.py',
                   scripts / 'r_dpo.py', scripts / 'r_eval.py',
                   C.LOOP_DIR / 'loop_lib/dpo.py', C.LOOP_DIR / 'loop_lib/paraphraser.py']
    state = {'status': 'running', 'started_at_utc': now(), 'pid': os.getpid(),
             'smoke': args.smoke, 'train_docs': len(train), 'generator_in': str(generator),
             'detector_in': str(detector), 'tau': tau, 'round_directory': str(out),
             'config': str(config_path), 'stop_after': 'g2_evaluation', 'steps': [],
             'source_sha256': {str(p.relative_to(C.REPO_ROOT)):
                               hashlib.sha256(p.read_bytes()).hexdigest() for p in hashed_paths}}
    status_path = out / 'g2_status.json'

    def save_status():
        temp = status_path.with_suffix('.json.tmp')
        io.write_json(temp, state)
        temp.replace(status_path)

    def step(name, script, *extra):
        state['current_step'] = name
        record = {'name': name, 'started_at_utc': now()}
        state['steps'].append(record)
        save_status()
        print(f'[{now()}] START {name}', flush=True)
        start = time.monotonic()
        subprocess.run([sys.executable, '-u', str(scripts / script), '--config',
                        str(config_path), '--round', '1', '--device', 'cuda', *extra],
                       cwd=C.REPO_ROOT, check=True)
        record.update(completed_at_utc=now(), seconds=round(time.monotonic()-start, 2))
        save_status()
        print(f'[{now()}] DONE {name} ({record["seconds"]}s)', flush=True)

    save_status()
    try:
        step('generate_candidates', 'r_generate.py')
        counts = Counter()
        for candidate in io.iter_jsonl(out / 'candidates.jsonl.gz'):
            require(candidate['doc_id'] in train, 'Candidate outside train split')
            require(candidate['text'].strip(), 'Empty candidate')
            counts[candidate['doc_id']] += 1
        require(set(counts) == set(train), 'Some train documents have no candidates')
        require(all(n >= cfg.rounds.n_candidates for n in counts.values()),
                'Candidate generation incomplete; resume before proceeding')
        step('build_preferences', 'r_build_prefs.py', '--force')
        report = io.read_json(out / 'prefs_report.json')
        if report.get('needs_more'):
            step('top_up_candidates', 'r_generate.py', '--top-up')
            step('rebuild_preferences', 'r_build_prefs.py', '--force')
            report = io.read_json(out / 'prefs_report.json')
        require(report['fallback_top1'] is False, 'Smoke fallback is forbidden')
        prefs = io.read_jsonl(out / 'prefs.jsonl.gz')
        require(bool(prefs), 'No valid DPO preferences')
        for pref in prefs:
            require(pref['doc_id'] in train, 'Preference outside train split')
            row = train[pref['doc_id']]
            require(pref['x_ai'] == row['ai_text'] and pref['y_w'] == row['human_text'],
                    'Preference/anchor mismatch')
            require(pref['y_w'] != pref['y_l'] and pref['d_l'] > tau, 'Invalid hard negative')
            require(all(math.isfinite(pref[k]) for k in ('ref_logp_w', 'ref_logp_l', 'd_l')),
                    'Non-finite preference score')
        state['n_prefs'] = len(prefs)
        state['docs_covered'] = report['docs_covered']
        step('train_g2_dpo', 'r_dpo.py')
        history = io.read_json(out / 'dpo/history.json')
        require(bool(history), 'DPO history missing')
        require(all(math.isfinite(r[k]) for r in history for k in ('loss', 'pref_acc', 'margin')),
                'Non-finite DPO metrics')
        require((out / 'dpo/final/style_head.pt').is_file(), 'G2 checkpoint missing')
        extra = ['--light', '--no-ck-export']
        if args.smoke:
            extra += ['--limit', '8']
        step('evaluate_g2', 'r_eval.py', *extra)
        metrics = io.read_json(out / 'eval/metrics.json')
        require(metrics['n_test'] == (8 if args.smoke else 1985), 'Wrong evaluation size')
        state.update(status='completed', completed_at_utc=now(), current_step=None)
        save_status()
        print(f'[{now()}] G2 pipeline completed; D2 retraining was not launched', flush=True)
    except BaseException as exc:
        state.update(status='failed', failed_at_utc=now(), error=str(exc))
        save_status()
        raise


if __name__ == '__main__':
    main()
