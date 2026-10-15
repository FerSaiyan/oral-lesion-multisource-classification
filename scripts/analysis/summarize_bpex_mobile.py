#!/usr/bin/env python3
"""Verify completed phone observations and publish deidentified deployment tables."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from scripts.npu.run_bpex_fold6 import npu_delegation

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/revision_bpex/deployment_2026-10-13_v1'
PAPER = ROOT / 'papers/Multisource Classification'
MODELS = {'effnet_b3': 'B3', 'effnet_b7': 'B7', 'vit_base': 'ViT-Base', 'vit_large': 'ViT-Large'}
GPU_JOBS = {'effnet_b3': 'effnet_b3_gpu_1791890786', 'effnet_b7': 'effnet_b7_gpu_1791891278',
            'vit_base': 'vit_base_gpu_1791892393'}
NPU_JOBS = {'effnet_b3': 'effnet_b3_npu_burst_1791896433', 'effnet_b7': 'effnet_b7_npu_burst_1791896540',
            'vit_base': 'vit_base_npu_burst_1791896991', 'vit_large': 'vit_large_npu_burst_1791897104'}


def main():
    plan = json.loads((BASE / 'payload/deployment_plan.json').read_text())
    with (BASE / 'payload/test.csv').open(newline='') as stream:
        test = list(csv.DictReader(stream))
    labels = np.array([plan['classes'].index(row['diagnosis_categories']) for row in test])
    attachment = PAPER / 'supplement/data/mobile_benchmark_v1'
    attachment.mkdir(exist_ok=True)
    rows, fidelity = [], []
    compact = BASE / 'phone/20261013_100033'
    protocol = json.loads((compact / 'protocol.json').read_text())
    if protocol['soak_seconds'] != 0 or protocol['models'] != list(MODELS):
        raise ValueError('Unexpected final mobile protocol')
    if hashlib.sha256((BASE / 'payload/test.csv').read_bytes()).hexdigest() != protocol['test_csv_sha256']:
        raise ValueError('Frozen test manifest changed')
    for model, display in MODELS.items():
        reference = np.load(BASE / 'mobile/reference' / f'{model}_fp32_logits.npy')
        fidelity.append({'model': model, 'runtime': 'A4000 FP32',
                         'macro_f1': float(f1_score(labels, reference.argmax(1), average='macro')),
                         'weighted_f1': float(f1_score(labels, reference.argmax(1), average='weighted')),
                         'agreement': 1.0})
        for backend in ('GPU', 'NPU'):
            if backend == 'GPU' and model not in GPU_JOBS:
                continue
            job = BASE / 'phone/20261013_082626' / GPU_JOBS[model] if backend == 'GPU' else compact / NPU_JOBS[model]
            summary = json.loads((job / 'summary.json').read_text())
            ready = json.loads((BASE / 'mobile' / f'{model}_ready.json').read_text())
            expected_hash = ready['plain_sha256' if backend == 'GPU' else 'aot_sha256']
            if (json.loads((job / 'status.json').read_text())['status'] != 'completed'
                    or summary['model_sha256'] != expected_hash or summary['model'] != model):
                raise ValueError('Incomplete or mismatched mobile model')
            native = (job / 'native_logcat.txt').read_text()
            if backend == 'NPU':
                evidence = npu_delegation(native)
                if not all(evidence[k] for k in ('full_dispatch_partition', 'qnn_context_loaded', 'hexagon_rpc_opened')) or evidence['cpu_graph_delegation_detected']:
                    raise ValueError('NPU hardware execution not verified')
            else:
                import re
                match = re.search(r'Replacing (\d+) out of (\d+) node\(s\) with delegate \(LITERT_CL\) node, yielding 1 partitions', native)
                if not match or match[1] != match[2]:
                    raise ValueError('Full GPU delegation not verified')
            logits = np.array(json.loads((job / 'logits.json').read_text()))
            if logits.shape != (1209, 4) or not np.isfinite(logits).all():
                raise ValueError('Incomplete or invalid full-test predictions')
            metrics = {'model': model, 'runtime': f'Fold6 {backend}',
                       'macro_f1': float(f1_score(labels, logits.argmax(1), average='macro')),
                       'weighted_f1': float(f1_score(labels, logits.argmax(1), average='weighted')),
                       'agreement': float((logits.argmax(1) == reference.argmax(1)).mean())}
            for key in ('macro_f1', 'weighted_f1'):
                if not np.isclose(metrics[key], summary[key], rtol=1e-12):
                    raise ValueError('Reported classification metrics differ from full-test predictions')
            if not np.isclose(metrics['agreement'], summary['reference_top1_agreement'], rtol=1e-12):
                raise ValueError('Reported agreement differs from full-test predictions')
            fidelity.append(metrics)
            np.save(attachment / f'{model}_{backend.lower()}_logits.npy', logits)
            timing_job = job
            if model == 'effnet_b3' and backend == 'NPU':
                timing_job = BASE / 'phone/20261013_105230/effnet_b3_npu_burst_1791899550'
                timing_protocol = json.loads((timing_job.parent / 'protocol.json').read_text())
                timing_summary = json.loads((timing_job / 'summary.json').read_text())
                if (timing_protocol['timing_indices'] != protocol['timing_indices']
                        or timing_protocol['test_csv_sha256'] != protocol['test_csv_sha256']
                        or json.loads((timing_job / 'status.json').read_text())['status'] != 'completed'
                        or timing_summary['model_sha256'] != expected_hash
                        or timing_summary['prediction_status'] != 'not_requested'
                        or not timing_summary['full_npu_execution_verified']):
                    raise ValueError('B3 timing rerun does not match the common subset protocol')
                (attachment / 'b3_timing_protocol.json').write_text(json.dumps(timing_protocol, indent=2) + '\n')
            with (timing_job / 'latency.csv').open(newline='') as stream:
                observations = list(csv.DictReader(stream))
            selected = protocol['timing_indices'] if backend == 'NPU' else list(range(1209))
            if [(int(r['pass']), int(r['index'])) for r in observations] != [(p, i) for p in (1, 2, 3) for i in selected]:
                raise ValueError('Timing membership or observation count changed')
            row = {'model': model, 'backend': backend, 'n_images': len(selected), 'n_timed': len(observations),
                   'fidelity_images': 1209, 'agreement': metrics['agreement'], 'timing_source': str(timing_job.relative_to(BASE)),
                   'fidelity_source': str(job.relative_to(BASE)), 'model_sha256': expected_hash}
            for field, prefix in (('inference_ms', 'inference'), ('preprocessed_file_pipeline_ms', 'pipeline')):
                values = np.array([float(r[field]) for r in observations])
                if not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError('Invalid latency observation')
                stats = {'mean_ms': float(values.mean()), 'sd_ms': float(values.std(ddof=1)),
                         'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95))}
                row.update({f'{prefix}_{k}': v for k, v in stats.items()})
                recorded_summary = timing_summary if timing_job != job else summary
                recorded = recorded_summary.get('inference' if prefix == 'inference' else 'preprocessed_file_pipeline')
                if any(not np.isclose(recorded[k], v, rtol=1e-10) for k, v in stats.items()):
                    raise ValueError('Summary timing statistics differ from raw observations')
            rows.append(row)
            with (attachment / f'{model}_{backend.lower()}_latency.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
                writer.writeheader()
                writer.writerows(observations)
    for name, data in (('mobile_timing_summary.csv', rows), ('mobile_fidelity_summary.csv', fidelity)):
        with (attachment / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (attachment / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (attachment / 'verification.json').write_text(json.dumps({'verified_phone_endpoints': len(rows),
        'prediction_images_per_endpoint': 1209, 'full_graph_hardware_delegation_verified': True,
        'timing_observations': sum(row['n_timed'] for row in rows),
        'all_npu_timings_use_same_120_indices': True,
        'b3_full_set_fidelity_reused_without_repeating_predictions': True}, indent=2) + '\n')
    timing_lines = []
    for row in rows:
        timing_lines.append(f"{MODELS[row['model']]} & {row['backend']} & {row['n_timed']:,} & "
            f"${row['inference_mean_ms']:.2f}\\pm{row['inference_sd_ms']:.2f}$ & {row['inference_p95_ms']:.2f} & "
            f"${row['pipeline_mean_ms']:.2f}\\pm{row['pipeline_sd_ms']:.2f}$\\\\")
    table = '\n'.join([
        r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Galaxy Z Fold6 batch-one inference. Latency is mean $\pm$ SD in milliseconds; p95 refers to inference latency. GPU rows use LiteRT defaults and three passes over 1,209 images. All NPU rows use BURST with error-only QNN logging and three passes over the same fixed 120-image subset. The cached-RGB pipeline includes file reading, float conversion, input transfer and output readback.}',
        r'\label{tab:s_mobile_timing}', r'\begin{tabular}{llrrrr}\toprule',
        r'Model & Backend & Observations & Inference (ms) & p95 (ms) & RGB pipeline (ms)\\\midrule',
        *timing_lines, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    (PAPER / 'supplement/s_mobile_timing.tex').write_text(table)
    fidelity_lines = [f"{MODELS[row['model']]} & {row['runtime']} & {row['macro_f1']:.3f} & {row['weighted_f1']:.3f} & "
                      + ('--' if row['runtime'] == 'A4000 FP32' else f"{100 * row['agreement']:.2f}") + r'\\' for row in fidelity]
    (PAPER / 'supplement/s_mobile_fidelity.tex').write_text('\n'.join([
        r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Deployment classification metrics on all 1,209 frozen test images for the corrected seed-42 checkpoints. Agreement compares top-1 predictions with the same checkpoint evaluated in PyTorch FP32 on RTX A4000. Fold6 NPU rows use BURST; GPU rows use LiteRT defaults.}',
        r'\label{tab:s_mobile_fidelity}', r'\begin{tabular}{llrrr}\toprule',
        r'Model & Runtime & Macro-F1 & Weighted-F1 & Agreement (\%)\\\midrule',
        *fidelity_lines, r'\bottomrule', r'\end{tabular}', r'\end{table}', '']))
    for row in rows:
        print(f"{row['model']} {row['backend']}: {row['inference_mean_ms']:.3f} ms; agreement {row['agreement']:.4%}")


if __name__ == '__main__':
    main()
