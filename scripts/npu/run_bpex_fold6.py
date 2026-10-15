#!/usr/bin/env python3
"""Explicitly install/run the isolated BPEX GPU/NPU benchmark when the phone is attached."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import subprocess
import time

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/revision_bpex/deployment_2026-10-13_v1'
MOBILE = BASE / 'mobile'
APK = ROOT / 'scripts/npu/android_benchmark/build/outputs/apk/debug/BpexDeploymentBenchmark-debug.apk'
PACKAGE = 'org.bpex.benchmark'
REMOTE = f'/sdcard/Android/data/{PACKAGE}/files/bpex-deployment'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def npu_delegation(log):
    return {
        'full_dispatch_partition': 'Replacing 1 out of 1 node(s) with delegate (DispatchDelegate)' in log,
        'qnn_context_loaded': 'Creating new QNN context for bytecode' in log,
        'hexagon_rpc_opened': bool(re.search(r'remote_handle64_open: opened handle[^\n]*libQnnHtpV75Skel.so[^\n]*_dom=cdsp', log)),
        'cpu_graph_delegation_detected': bool(re.search(r'Replacing [1-9][0-9]* out of [0-9]+ node\(s\) with delegate \([^\n]*(?:XNNPACK|CPU)', log, re.I)),
    }


def timing_subset(rows, count):
    if not 0 <= count <= len(rows):
        raise ValueError('Timing subset count exceeds frozen test membership')
    rng = np.random.default_rng(20261013)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((row['dataset_source'], row['diagnosis_categories']), []).append(index)
    groups = [list(rng.permutation(groups[key])) for key in sorted(groups)]
    selected = []
    while len(selected) < count:
        for group in groups:
            if group and len(selected) < count:
                selected.append(int(group.pop()))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--models', nargs='+', default=['effnet_b3', 'effnet_b7', 'vit_base', 'vit_large'])
    parser.add_argument('--backends', nargs='+', choices=['GPU', 'NPU'], default=['GPU', 'NPU'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--reuse-staged', action='store_true', help='Verify existing phone fixtures and skip retransferring the bundle')
    parser.add_argument('--cooldown-seconds', type=int, default=60)
    parser.add_argument('--npu-profile', choices=['runtime-default', 'default-quiet', 'burst'], default='runtime-default')
    parser.add_argument('--wait-unthrottled', action='store_true', help='Wait up to 15 minutes for thermal status NONE before each run')
    parser.add_argument('--cool-between-passes', action='store_true', help='Cool to thermal NONE between passes, then rewarm; excluded from latency')
    parser.add_argument('--continue-on-failure', action='store_true', help='Preserve failed cases and attempt remaining cases without CPU fallback')
    parser.add_argument('--timing-count', type=int, default=1209)
    parser.add_argument('--skip-timing-models', nargs='*', default=[], help='Run only full-set fidelity for models with completed timings')
    parser.add_argument('--soak-seconds', type=int, default=90)
    parser.add_argument('--timing-only', action='store_true', help='Skip already completed prediction evaluation; requires zero soak')
    args = parser.parse_args()
    ready = json.loads((MOBILE / 'ready.json').read_text())
    with (BASE / 'payload/test.csv').open(newline='') as stream:
        test_rows = list(csv.DictReader(stream))
    indices = timing_subset(test_rows, args.timing_count)
    if len(test_rows) != 1209 or not set(args.skip_timing_models) <= set(args.models) or args.soak_seconds < 0:
        raise ValueError('Invalid compact benchmark configuration')
    if args.timing_only and (args.soak_seconds != 0 or args.skip_timing_models or args.timing_count == 0):
        raise ValueError('Timing-only requires measured images, no skipped timing models and zero soak')
    if set(args.models) - set(ready['models']):
        raise ValueError('Model is not in the verified corrected deployment bundle')
    if not APK.exists():
        raise FileNotFoundError(f'Build the isolated benchmark APK first: {APK}')
    records = {}
    for model in args.models:
        record = json.loads((MOBILE / f'{model}_ready.json').read_text())
        if sha256(MOBILE / 'models' / f'{model}_float.tflite') != record['plain_sha256'] or sha256(MOBILE / 'aot' / f'{model}_SM8650.tflite') != record['aot_sha256']:
            raise ValueError('Compiled phone artifact changed')
        records[model] = record
    print(json.dumps({'models': args.models, 'backends': args.backends, 'test_images': 1209,
                        'passes': 3, 'warmups': 30, 'soak_seconds': args.soak_seconds,
                        'timing_images': args.timing_count, 'skip_timing_models': args.skip_timing_models,
                        'prediction_evaluation': not args.timing_only,
                       'npu_profile': args.npu_profile, 'wait_unthrottled': args.wait_unthrottled,
                       'cool_between_passes': args.cool_between_passes,
                      'cleanup': 'No device or Luciano workspace cleanup; user approval required'}, indent=2), flush=True)
    if not args.execute:
        return
    adb = ['adb', '-s', args.serial]
    subprocess.run(adb + ['get-state'], check=True)
    active = subprocess.run(adb + ['shell', 'pidof', PACKAGE], capture_output=True, text=True)
    if active.stdout.strip():
        raise RuntimeError('Benchmark app is already active; do not overlap runs or reinstall during inference')
    if args.reuse_staged:
        for size in (224, 384, 512):
            folder = f'{REMOTE}/fixture/{size}'
            remote_hash = subprocess.check_output(adb + ['shell', 'sha256sum', f'{folder}/manifest.json'], text=True).split()[0]
            if remote_hash != sha256(MOBILE / 'fixture' / str(size) / 'manifest.json'):
                raise ValueError('Staged phone fixture manifest differs from the frozen bundle')
            sizes = subprocess.check_output(adb + ['shell', f"find {folder} -type f -name '*.rgb' -exec stat -c '%s' {{}} +"], text=True).split()
            if len(sizes) != 1209 or any(int(value) != size * size * 3 for value in sizes):
                raise ValueError('Staged phone inputs are missing or truncated')
        print('Verified staged fixture manifests, counts and sizes; skipping transfer', flush=True)
    else:
        storage = subprocess.check_output(adb + ['shell', 'df', '-k', '/sdcard'], text=True).splitlines()[-1].split()
        free_bytes = int(storage[-3]) * 1024
        staged_bytes = sum(path.stat().st_size for folder in ('models', 'aot', 'fixture')
                           for path in (MOBILE / folder).rglob('*') if path.is_file())
        if free_bytes < staged_bytes + 1024**3:
            raise RuntimeError('Phone needs space for the deployment bundle plus 1 GiB margin; no files were cleared')
    subprocess.run(adb + ['install', '-r', str(APK)], check=True)
    subprocess.run(adb + ['shell', 'am', 'start', '-n', f'{PACKAGE}/.BenchmarkActivity',
                          '--ez', 'prepare', 'true'], check=True)
    time.sleep(2)
    subprocess.run(adb + ['shell', 'mkdir', '-p', REMOTE], check=True)
    if not args.reuse_staged:
        for folder in ('models', 'aot', 'fixture'):
            subprocess.run(adb + ['push', str(MOBILE / folder), f'{REMOTE}/'], check=True)
    subprocess.run(adb + ['shell', 'am', 'force-stop', PACKAGE], check=True)
    output = BASE / 'phone' / time.strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True)
    (output / 'protocol.json').write_text(json.dumps({
        'timing_indices': indices, 'subset_seed': 20261013,
        'subset_method': 'Round-robin source/class strata with deterministic within-stratum random sampling',
        'frozen_test_rows': 1209, 'test_csv_sha256': sha256(BASE / 'payload/test.csv'),
        'models': args.models, 'backends': args.backends, 'npu_profile': args.npu_profile,
        'skip_timing_models': args.skip_timing_models, 'soak_seconds': args.soak_seconds,
        'prediction_protocol': 'Not requested; existing verified full-set predictions retained' if args.timing_only else 'One separate full-set pass; 20 ms idle per image; thermal pauses at MODERATE until NONE',
    }, indent=2) + '\n')
    failed = []
    for model in args.models:
        for backend in args.backends:
            profile_tag = f'_{args.npu_profile.replace("-", "_")}' if backend == 'NPU' else ''
            run_id = f'{model}_{backend.lower()}{profile_tag}_{int(time.time())}'
            job = output / run_id
            job.mkdir()
            if args.wait_unthrottled:
                deadline = time.monotonic() + 900
                while True:
                    thermal = subprocess.check_output(adb + ['shell', 'dumpsys', 'thermalservice'], text=True)
                    match = re.search(r'^Thermal Status:\s*(\d+)\s*$', thermal, re.M)
                    if not match:
                        raise RuntimeError('Unable to verify device thermal status')
                    (job / 'thermal_before.txt').write_text(thermal)
                    if int(match[1]) == 0:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Phone remained throttled for 15 minutes; no new inference was started')
                    print(f'Waiting for unthrottled start: {model} {backend}, thermal status {match[1]}', flush=True)
                    time.sleep(30)
            model_hash = records[model]['aot_sha256'] if backend == 'NPU' else records[model]['plain_sha256']
            selected = [] if model in args.skip_timing_models else indices
            subprocess.run(adb + ['shell', 'am', 'force-stop', PACKAGE], check=True)
            subprocess.run(adb + ['shell', 'am', 'start', '-n', f'{PACKAGE}/.BenchmarkActivity',
                                  '--es', 'model', model, '--es', 'backend', backend,
                                   '--es', 'run_id', run_id, '--es', 'model_sha256', model_hash,
                                   '--es', 'npu_profile', args.npu_profile,
                                    '--ez', 'cool_between_passes', str(args.cool_between_passes).lower(),
                                    '--es', 'timing_indices', ','.join(map(str, selected)) if selected else 'none',
                                    '--ez', 'prediction_evaluation', str(not args.timing_only).lower(),
                                    '--ei', 'passes', '3', '--ei', 'warmups', '30', '--ei', 'soak_seconds', str(args.soak_seconds)], check=True)
            deadline = time.monotonic() + 3600
            status = None
            while time.monotonic() < deadline:
                transfer = subprocess.run(adb + ['pull', f'{REMOTE}/results/{run_id}/status.json', str(job / 'status.json')],
                                          capture_output=True, text=True)
                if transfer.returncode == 0:
                    status = json.loads((job / 'status.json').read_text())
                    if status['status'] in ('completed', 'failed'):
                        break
                with (job / 'meminfo.txt').open('a') as stream:
                    stream.write(f'\nUNIX {time.time()}\n')
                    subprocess.run(adb + ['shell', 'dumpsys', 'meminfo', PACKAGE], stdout=stream, stderr=subprocess.STDOUT)
                time.sleep(5)
            subprocess.run(adb + ['pull', f'{REMOTE}/results/{run_id}/.', str(job)], check=True)
            pid = subprocess.run(adb + ['shell', 'pidof', PACKAGE], capture_output=True, text=True).stdout.strip()
            if pid:
                with (job / 'native_logcat.txt').open('w') as stream:
                    subprocess.run(adb + ['logcat', '-d', f'--pid={pid}'], stdout=stream, stderr=subprocess.STDOUT)
            if not status or status['status'] != 'completed':
                subprocess.run(adb + ['shell', 'am', 'force-stop', PACKAGE], check=True)
                if args.continue_on_failure:
                    failed.append(str(job))
                    print(f'Failed or timed out; preserving artifacts and continuing: {job}', flush=True)
                    continue
                raise RuntimeError(f'Phone benchmark failed or timed out; inspect {job}')
            summary = json.loads((job / 'summary.json').read_text())
            with (job / 'latency.csv').open(newline='') as stream:
                timed_rows = list(csv.DictReader(stream))
            expected_rows = [(p, index) for p in range(1, 4) for index in selected]
            if [(int(row['pass']), int(row['index'])) for row in timed_rows] != expected_rows:
                raise ValueError('Incomplete or mismatched timing phase')
            if backend == 'NPU':
                evidence = npu_delegation((job / 'native_logcat.txt').read_text())
                summary['delegation_evidence'] = evidence
                summary['full_npu_execution_verified'] = all(evidence[key] for key in ('full_dispatch_partition', 'qnn_context_loaded', 'hexagon_rpc_opened')) and not evidence['cpu_graph_delegation_detected']
                (job / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                if not summary['full_npu_execution_verified']:
                    subprocess.run(adb + ['shell', 'am', 'force-stop', PACKAGE], check=True)
                    raise RuntimeError(f'NPU hardware delegation could not be verified; inspect {job}')
            if args.timing_only:
                if summary['prediction_status'] != 'not_requested':
                    raise ValueError('Unexpected prediction evaluation in timing-only run')
                print(f'Completed timing-only {model} {backend}: mean {summary["inference"]["mean_ms"]:.3f} ms: {job}', flush=True)
                subprocess.run(adb + ['shell', 'am', 'force-stop', PACKAGE], check=True)
                continue
            actual = np.array(json.loads((job / 'logits.json').read_text()))
            reference = np.load(MOBILE / 'reference' / f'{model}_fp32_logits.npy')
            size = records[model]['test_rows']
            if (actual.shape != (size, 4) or not np.isfinite(actual).all()
                    or summary['prediction_status'] != 'completed' or summary['prediction_images'] != size):
                raise ValueError('Incomplete or non-finite full-set predictions')
            image_size = 384 if model == 'effnet_b3' else 512 if model == 'effnet_b7' else 224
            fixture = json.loads((MOBILE / 'fixture' / str(image_size) / 'manifest.json').read_text())
            labels = np.array(fixture['labels'])
            values = actual - actual.max(axis=1, keepdims=True)
            probabilities = np.exp(values) / np.exp(values).sum(axis=1, keepdims=True)
            suspicious_indices = [fixture['classes'].index(label) for label in ('cancer', 'opmd')]
            suspicious = np.isin(labels, suspicious_indices)
            screening_score = probabilities[:, suspicious_indices].sum(axis=1)
            summary.update(reference_top1_agreement=float((actual.argmax(1) == reference.argmax(1)).mean()),
                           max_absolute_logit_difference=float(abs(actual - reference).max()),
                           macro_f1=float(f1_score(labels, actual.argmax(1), average='macro')),
                           weighted_f1=float(f1_score(labels, actual.argmax(1), average='weighted')),
                           screening_f1=float(f1_score(suspicious, screening_score >= 0.5)),
                           screening_auc=float(roc_auc_score(suspicious, screening_score)),
                           checkpoint_sha256=records[model]['checkpoint_sha256'],
                           delegation_review='Inspect native_logcat.txt before describing full GPU/NPU delegation')
            (job / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            timing_text = f'mean {summary["inference"]["mean_ms"]:.3f} ms' if selected else 'timing skipped (already measured)'
            print(f'Completed {model} {backend}: {timing_text}; agreement {summary["reference_top1_agreement"]:.4%}; soak {summary.get("soak_status", "completed")}: {job}', flush=True)
            subprocess.run(adb + ['shell', 'am', 'force-stop', PACKAGE], check=True)
            time.sleep(args.cooldown_seconds)
    if failed:
        raise RuntimeError(f'Some benchmark cases remain incomplete; inspect {failed}')


if __name__ == '__main__':
    main()
