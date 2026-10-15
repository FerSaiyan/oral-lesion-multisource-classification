#!/usr/bin/env python3
"""Run disjoint local/cloud assignments from the frozen 60-study source plan."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
import time

from run_bpex_revision_hpo import sha256
from bpex_sourcewise_dispatch import WORKERS, assignments, key

ROOT = Path(__file__).resolve().parents[2]
ARCH = {'effnet_b3': 'effnet_b3_multisource', 'effnet_b7': 'effnet_b7_imagenet',
        'vit_base': 'vit_base_imagenet', 'vit_large': 'vit_large_imagenet'}
PLAN = ROOT / 'configs/studies/bpex_sourcewise_hpo_2026-09-30.json'
RESULTS = ROOT / 'results/revision_bpex/hpo_sourcewise_corrected_v1'
STATE = ROOT / 'results/revision_bpex/sourcewise_queue'


def name(task: dict) -> str:
    return f"BPEX_revision_v3_no_{task['source']}_{task['model']}_seed{task['seed']}_{ARCH[task['model']]}"


def states(directory: Path) -> dict:
    db = directory / f'{directory.name}.db'
    if not db.exists():
        return {}
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True, timeout=30) as c:
        if c.execute('pragma integrity_check').fetchone()[0] != 'ok':
            raise RuntimeError(f'Invalid SQLite database: {db}')
        return dict(c.execute('select state,count(*) from trials group by state'))


def complete(directory: Path, plan_hash: str) -> bool:
    marker = directory / 'seedwise_completion.json'
    if not marker.exists():
        return False
    r = json.loads(marker.read_text())
    s = states(directory)
    return (r.get('plan_sha256') == plan_hash and r.get('finished_trials') == 50
            and r.get('retrain_epochs') == 50 and r.get('test_csv') is None
            and s.get('COMPLETE', 0) + s.get('PRUNED', 0) == 50
            and not s.get('RUNNING', 0)
            and not (directory / 'test_predictions.csv').exists()
            and r['retrain_checkpoint_sha256'] == sha256(directory / 'best_model_retrain.pth')
            and r['study_checkpoint_sha256'] == sha256(directory / 'best_model_study.pth')
            and r['retrain_log_sha256'] == sha256(directory / 'retrain_log.csv'))


def publish(directory: Path, transfer: Path, task: dict, plan_hash: str) -> None:
    receipt = transfer / f'{directory.name}.json'
    if receipt.exists():
        r = json.loads(receipt.read_text())
        if r['plan_sha256'] != plan_hash:
            raise ValueError('Published study has a different plan')
        return
    archive = transfer / f'{directory.name}.tar.gz'
    stage = archive.with_suffix('.gz.tmp')
    with tarfile.open(stage, 'w:gz') as out:
        for f in sorted(directory.iterdir()):
            if f.is_file() and not f.name.endswith('.tmp') and not f.name.endswith('_inprogress.pth'):
                out.add(f, arcname=f'{directory.name}/{f.name}')
    os.replace(stage, archive)
    r = {'task': task, 'study': directory.name, 'plan_sha256': plan_hash,
         'archive_sha256': sha256(archive),
         'retrain_checkpoint_sha256': sha256(directory / 'best_model_retrain.pth'),
         'test_accessed': False}
    staged = receipt.with_suffix('.json.tmp')
    staged.write_text(json.dumps(r, indent=2) + '\n')
    staged.replace(receipt)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', choices=WORKERS, required=True)
    p.add_argument('--view', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--shutdown', action='store_true')
    p.add_argument('--canonical-image-root')
    args = p.parse_args()
    plan = json.loads(PLAN.read_text())
    ph = sha256(PLAN)
    state = STATE / args.worker
    state.mkdir(parents=True, exist_ok=True)
    transfer = state / 'transfer'
    transfer.mkdir(exist_ok=True)
    with (state / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (state / 'queue-error.json').exists():
            raise RuntimeError('Inspect the recorded queue error before restarting')
        try:
            for task in plan['tasks']:
                if assignments(plan, PLAN)[key(task)] != args.worker:
                    continue
                study = name(task)
                d = RESULTS / study
                if complete(d, ph):
                    if args.worker != 'local':
                        publish(d, transfer, task, ph)
                    continue
                for attempt in range(3):
                    before = states(d)
                    finished_before = before.get('COMPLETE', 0) + before.get('PRUNED', 0)
                    command = [sys.executable, 'scripts/analysis/run_bpex_revision_hpo.py',
                               '--model', task['model'], '--seed', str(task['seed']),
                               '--excluded-source', task['source'], '--view', str(args.view),
                               '--images', str(args.images), '--execute']
                    if args.canonical_image_root:
                        command.extend(['--canonical-image-root', args.canonical_image_root])
                    if (d / f'{study}.db').exists():
                        command.append('--resume')
                    print(f'Starting {study}, states {before}', flush=True)
                    with (state / f'{study}.log').open('a') as log:
                        log.write(f'\nQueue start UTC {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}\n')
                        log.flush()
                        process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                                   stderr=subprocess.STDOUT,
                                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
                        while True:
                            try:
                                status = process.wait(timeout=60)
                                break
                            except subprocess.TimeoutExpired:
                                gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                                      '--format=csv,noheader'], capture_output=True, text=True)
                                print(f'{study} GPU: {gpu.stdout.strip()}', flush=True)
                    if status == 0 and complete(d, ph):
                        break
                    after = states(d)
                    finished = after.get('COMPLETE', 0) + after.get('PRUNED', 0)
                    if status or finished <= finished_before or attempt == 2:
                        raise RuntimeError(f'Incomplete {study}, exit={status}, finished={finished}/50')
                if args.worker != 'local':
                    publish(d, transfer, task, ph)
                print(f'Complete {study}', flush=True)
            (state / 'all-complete').write_text(ph + '\n')
        except Exception as e:
            (state / 'queue-error.json').write_text(json.dumps({'error': str(e), 'plan_sha256': ph,
                'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, indent=2) + '\n')
            raise
        finally:
            if args.shutdown:
                time.sleep(900)
                subprocess.run(['sudo', '/usr/sbin/shutdown', '-h', 'now'], check=False)


if __name__ == '__main__':
    main()
