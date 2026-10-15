#!/usr/bin/env python3
"""Verify and publish the common 120-image desktop/GPU/NPU timing benchmark."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

from scripts.npu.run_bpex_fold6 import npu_delegation, timing_subset

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/revision_bpex/deployment_2026-10-13_v1'
PAPER = ROOT / 'papers/Multisource Classification'
MODELS = {'effnet_b3': 'B3', 'effnet_b7': 'B7', 'vit_base': 'ViT-Base', 'vit_large': 'ViT-Large'}
NPU_RUNS = {
    'effnet_b3': '20261013_105230/effnet_b3_npu_burst_1791899550',
    'effnet_b7': '20261013_100033/effnet_b7_npu_burst_1791896540',
    'vit_base': '20261013_100033/vit_base_npu_burst_1791896991',
    'vit_large': '20261013_100033/vit_large_npu_burst_1791897104',
}


def stats(values, recorded):
    values = np.asarray(values, dtype=float)
    if len(values) != 360 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Invalid timing observations')
    result = dict(mean_ms=float(values.mean()), sd_ms=float(values.std(ddof=1)),
                  median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)))
    if recorded['n'] != 360 or any(not np.isclose(recorded[k], v, rtol=1e-10) for k, v in result.items()):
        raise ValueError('Recorded summary differs from raw observations')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phone-run', type=Path, required=True)
    args = parser.parse_args()
    args.phone_run = args.phone_run.resolve()
    protocol = json.loads((args.phone_run / 'protocol.json').read_text())
    plan = json.loads((BASE / 'payload/deployment_plan.json').read_text())
    test_path = BASE / 'payload/test.csv'
    with test_path.open(newline='') as stream:
        test = list(csv.DictReader(stream))
    indices = timing_subset(test, 120)
    manifest_hash = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if protocol['timing_indices'] != indices or protocol['test_csv_sha256'] != manifest_hash:
        raise ValueError('Phone subset differs from the frozen common subset')
    expected = [(p, i) for p in (1, 2, 3) for i in indices]
    attachment = PAPER / 'supplement/data/deployment_runtime120_v3'
    attachment.mkdir(exist_ok=True)
    desktop, mobile, hashes = [], [], {}
    for model, display in MODELS.items():
        gpu = 0 if model.startswith('effnet') else 1
        folder = BASE / f'results_runtime120_20261015_gpu{gpu}'
        summary_path = folder / f'{model}_summary.json'
        summary = json.loads(summary_path.read_text())
        if (summary['timing_indices'] != indices or summary['test_csv_sha256'] != manifest_hash
                or summary['checkpoint_sha256'] != plan['models'][model]['checkpoint_sha256']
                or summary['timing_scope'] != 'Model execution and host retrieval of four float scores; input already uploaded; excludes softmax'):
            raise ValueError('Desktop subset/checkpoint differs')
        for precision, endpoint in summary['precisions'].items():
            source = folder / f'{model}_{precision}_latency.csv'
            with source.open(newline='') as stream:
                observations = list(csv.DictReader(stream))
            if [(int(r['pass']), int(r['index'])) for r in observations] != expected:
                raise ValueError('Desktop observation membership/order differs')
            if [r['filename'] for r in observations] != [test[i]['filename'] for _, i in expected]:
                raise ValueError('Desktop timing filenames differ')
            row = dict(model=model, precision=precision, n_images=120, n_timed=360)
            row.update(stats([r['inference_ms'] for r in observations], endpoint['inference']))
            desktop.append(row)
            with (attachment / source.name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['pass', 'index', 'inference_ms'], extrasaction='ignore')
                writer.writeheader()
                writer.writerows(observations)
            hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        for backend in ('GPU', 'NPU'):
            if backend == 'GPU':
                jobs = list(args.phone_run.glob(f'{model}_gpu_*'))
                if len(jobs) != 1:
                    raise ValueError(f'Expected one completed GPU job for {model}')
                job = jobs[0]
            else:
                job = BASE / 'phone' / NPU_RUNS[model]
            run_protocol = json.loads((job.parent / 'protocol.json').read_text())
            summary = json.loads((job / 'summary.json').read_text())
            ready = json.loads((BASE / 'mobile' / f'{model}_ready.json').read_text())
            if (json.loads((job / 'status.json').read_text())['status'] != 'completed'
                    or run_protocol['timing_indices'] != indices
                    or run_protocol['test_csv_sha256'] != manifest_hash
                    or summary['model_sha256'] != ready['plain_sha256' if backend == 'GPU' else 'aot_sha256']):
                raise ValueError('Incomplete or mismatched phone timing run')
            log = (job / 'native_logcat.txt').read_text()
            if backend == 'GPU':
                match = re.search(r'Replacing (\d+) out of (\d+) node\(s\) with delegate \(LITERT_CL\) node, yielding 1 partitions', log)
                if not match or match[1] != match[2]:
                    raise ValueError('Full GPU delegation not verified')
            else:
                evidence = npu_delegation(log)
                if (not all(evidence[k] for k in ('full_dispatch_partition', 'qnn_context_loaded', 'hexagon_rpc_opened'))
                        or evidence['cpu_graph_delegation_detected']):
                    raise ValueError('Full NPU execution not verified')
            if not re.search(r'^Thermal Status:\s*0\s*$', (job / 'thermal_before.txt').read_text(), re.M):
                raise ValueError('Phone run did not start at thermal NONE')
            source = job / 'latency.csv'
            with source.open(newline='') as stream:
                observations = list(csv.DictReader(stream))
            if [(int(r['pass']), int(r['index'])) for r in observations] != expected:
                raise ValueError('Mobile observation membership/order differs')
            row = dict(model=model, backend=backend, n_images=120, n_timed=360)
            row.update(stats([r['inference_ms'] for r in observations], summary['inference']))
            mobile.append(row)
            with (attachment / f'{model}_{backend.lower()}_latency.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['pass', 'index', 'inference_ms'], extrasaction='ignore')
                writer.writeheader()
                writer.writerows(observations)
            hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    for filename, rows in (('desktop_timing_summary.csv', desktop), ('mobile_timing_summary.csv', mobile)):
        with (attachment / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (attachment / 'verification.json').write_text(json.dumps({
        'timing_indices': indices, 'test_csv_sha256': manifest_hash,
        'images_per_endpoint': 120, 'passes': 3, 'warmups': 30,
        'desktop_endpoints': len(desktop), 'phone_endpoints': len(mobile),
        'total_verified_observations': 360 * (len(desktop) + len(mobile)),
        'phone_hardware_delegation_verified': True, 'phone_unthrottled_starts_verified': True,
        'phone_timings_reused': True,
        'timing_scope': 'Model execution and output retrieval; model loaded and input uploaded',
        'input_sha256': hashes}, indent=2) + '\n')
    archive = BASE / 'publication_archive_before_runtime_v3'
    archive.mkdir(exist_ok=True)
    for name in ('deployment_benchmark_v1', 'mobile_benchmark_v1', 'deployment_matched120_v2'):
        old = PAPER / 'supplement/data' / name
        if old.exists():
            if (archive / name).exists():
                raise FileExistsError(f'Refusing to overwrite historical archive: {name}')
            shutil.move(str(old), str(archive / name))
    old_table = PAPER / 'supplement/s_mobile_fidelity.tex'
    if old_table.exists():
        if (archive / old_table.name).exists():
            raise FileExistsError('Refusing to overwrite archived fidelity table')
        shutil.move(str(old_table), str(archive / old_table.name))
    desktop_lines = []
    for row in desktop:
        precision = 'FP32' if row['precision'] == 'fp32' else 'FP16 autocast'
        desktop_lines.append(f"{MODELS[row['model']]} & {precision} & "
            f"${row['mean_ms']:.2f}\\pm{row['sd_ms']:.2f}$ & {row['p95_ms']:.2f}\\\\")
    (PAPER / 'supplement/s_deployment_benchmark.tex').write_text('\n'.join([
        r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{RTX A4000 single-image inference. Timing covers model execution and output retrieval, with the model loaded and input uploaded. Values are mean $\pm$ SD and p95 in milliseconds over three passes of the common 120-image subset (360 observations per row), after 30 warm-up calls.}',
        r'\label{tab:s_deployment_benchmark}', r'\begin{tabular}{llrr}\toprule',
        r'Model & Runtime & Inference (ms) & p95 (ms)\\\midrule',
        *desktop_lines, r'\bottomrule', r'\end{tabular}', r'\end{table}', '']))
    mobile_lines = [f"{MODELS[r['model']]} & {r['backend']} & "
        f"${r['mean_ms']:.2f}\\pm{r['sd_ms']:.2f}$ & {r['p95_ms']:.2f}\\\\" for r in mobile]
    (PAPER / 'supplement/s_mobile_timing.tex').write_text('\n'.join([
        r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Galaxy Z Fold6 single-image inference using the same timing boundary, 120 images, warm-up count and pass count as the desktop benchmark. Values are mean $\pm$ SD and p95 in milliseconds (360 observations per row). GPU rows use LiteRT defaults; NPU rows use FP16 BURST.}',
        r'\label{tab:s_mobile_timing}', r'\begin{tabular}{llrr}\toprule',
        r'Model & Runtime & Inference (ms) & p95 (ms)\\\midrule',
        *mobile_lines, r'\bottomrule', r'\end{tabular}', r'\end{table}', '']))
    for row in mobile:
        print(f"{row['model']} {row['backend']}: {row['mean_ms']:.3f} ms")
    print(f'Verified and published {len(desktop) + len(mobile)} endpoints to {attachment}')


if __name__ == '__main__':
    main()
