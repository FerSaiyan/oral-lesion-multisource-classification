#!/usr/bin/env python3
"""Verify transferred desktop observations and summarize latency and task metrics."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/revision_bpex/deployment_2026-10-13_v1'


def main():
    plan = json.loads((BASE / 'payload/deployment_plan.json').read_text())
    with (BASE / 'payload/test.csv').open(newline='') as stream:
        test = list(csv.DictReader(stream))
    labels = np.array([plan['classes'].index(row['diagnosis_categories']) for row in test])
    suspicious = np.isin(labels, [plan['classes'].index(label) for label in ('cancer', 'opmd')])
    rows = []
    for model in plan['models']:
        summary = json.loads((BASE / 'desktop' / f'{model}_summary.json').read_text())
        if summary['checkpoint_sha256'] != plan['models'][model]['checkpoint_sha256']:
            raise ValueError('Benchmark checkpoint differs from the selected corrected model')
        for precision, endpoint in summary['precisions'].items():
            with (BASE / 'desktop' / f'{model}_{precision}_latency.csv').open(newline='') as stream:
                observations = list(csv.DictReader(stream))
            if len(observations) != 3627:
                raise ValueError('Missing timed observations')
            for repeat in range(1, 4):
                part = [row for row in observations if int(row['pass']) == repeat]
                if [row['filename'] for row in part] != [row['filename'] for row in test]:
                    raise ValueError('Timed test membership/order changed')
            for field, name in (('model_ms', 'model_only'), ('end_to_end_ms', 'end_to_end')):
                values = np.array([float(row[field]) for row in observations])
                expected = {'mean_ms': values.mean(), 'sd_ms': values.std(ddof=1),
                            'median_ms': np.median(values), 'p95_ms': np.percentile(values, 95)}
                if any(not np.isclose(endpoint[name][key], value, rtol=1e-10) for key, value in expected.items()):
                    raise ValueError('Reported timing statistics differ from the raw observations')
            logits = np.load(BASE / 'desktop' / f'{model}_{precision}_logits.npy')
            if logits.shape != (1209, 4) or not np.isfinite(logits).all():
                raise ValueError('Missing or invalid deployed predictions')
            values = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(values) / np.exp(values).sum(axis=1, keepdims=True)
            pred = probabilities.argmax(1)
            score = probabilities[:, [plan['classes'].index(label) for label in ('cancer', 'opmd')]].sum(axis=1)
            row = {'model': model, 'precision': precision, 'n_images': 1209, 'n_timed': 3627,
                   'macro_f1': float(f1_score(labels, pred, average='macro')),
                   'weighted_f1': float(f1_score(labels, pred, average='weighted')),
                   'screening_f1': float(f1_score(suspicious, score >= 0.5)),
                   'screening_auc': float(roc_auc_score(suspicious, score)),
                   'agreement': endpoint['reference_top1_agreement'],
                   'peak_allocated_mib': endpoint['peak_allocated_mib'],
                   'model_load_seconds': summary['model_load_seconds']}
            for name in ('model_only', 'end_to_end'):
                row.update({f'{name}_{key}': value for key, value in endpoint[name].items() if key != 'n'})
            rows.append(row)
    with (BASE / 'desktop_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (BASE / 'desktop_verification.json').write_text(json.dumps({'models': 4, 'precisions': 2,
                                                              'verified_observations': 4 * 2 * 3627,
                                                              'verified_test_rows_per_endpoint': 1209}, indent=2) + '\n')
    attachment = ROOT / 'papers/Multisource Classification/supplement/data/deployment_benchmark_v1'
    attachment.mkdir(parents=True, exist_ok=True)
    for source in [BASE / 'desktop_summary.csv', BASE / 'desktop_verification.json', *(BASE / 'desktop').glob('*')]:
        if source.is_file():
            if source.name.endswith('_latency.csv'):
                with source.open(newline='') as stream:
                    reader = csv.DictReader(stream)
                    fields = [field for field in reader.fieldnames if field != 'filename']
                    with (attachment / source.name).open('w', newline='') as target:
                        writer = csv.DictWriter(target, fieldnames=fields, extrasaction='ignore')
                        writer.writeheader()
                        writer.writerows(reader)
            else:
                shutil.copy2(source, attachment / source.name)
    for row in rows:
        print(f"{row['model']} {row['precision']}: model {row['model_only_mean_ms']:.2f} ms; "
              f"pipeline {row['end_to_end_mean_ms']:.2f} ms; p95 {row['end_to_end_p95_ms']:.2f} ms; "
              f"GPU {row['peak_allocated_mib']:.1f} MiB; agreement {row['agreement']:.4f}")


if __name__ == '__main__':
    main()
