#!/usr/bin/env python3
"""Choose 90%-sensitivity binary thresholds on validation; evaluate frozen test.

Thresholds use saved full-precision validation probabilities from each of the
20 fixed checkpoints, not held-out test labels. The held-out test predictions
are read only after thresholds have been determined from all validation files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from analyze_bpex_final_seeds import (
    ROOT, MODELS, CLASSES, binary_from_confusions, bootstrap_counts, cluster_map,
    confusion_from_clusters, interval, load_runs, require, sha256, write_csv,
)


def threshold_for_target(y_true: np.ndarray, scores: np.ndarray, target: float) -> float:
    positive = scores[y_true == 1]
    require(len(positive) > 0 and 0 < target <= 1, 'No validation positives or invalid target')
    # The largest score threshold attaining the target includes the k-th highest
    # positive. Ties at that score are included by the >= decision rule.
    k = int(np.ceil(target * len(positive)))
    threshold = float(np.sort(positive)[-k])
    require(np.mean(positive >= threshold) >= target - 1e-12,
            'Validation sensitivity fell below target')
    return threshold


def operating_point(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, scores >= threshold, labels=(0, 1)).ravel()
    safe = lambda a, b: float(a / b) if b else None
    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'sensitivity': safe(tp, tp + fn), 'specificity': safe(tn, tn + fp),
            'ppv': safe(tp, tp + fp), 'npv': safe(tn, tn + fn),
            'binary_f1': safe(2 * tp, 2 * tp + fp + fn)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/studies/bpex_revision_v2_final_seeds_2026-09-26.json')
    parser.add_argument('--results', type=Path, default=ROOT / 'results/revision_bpex/final_seeds_v2_corrected_global')
    parser.add_argument('--receipts', type=Path, default=ROOT / 'results/revision_bpex/cloud_final_seed_sync')
    parser.add_argument('--val-csv', type=Path, default=ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24/multisource_val_global.csv')
    parser.add_argument('--test-csv', type=Path, default=ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24/multisource_test_global.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/revision_bpex/five_seed_test_analysis_v1')
    parser.add_argument('--target', type=float, default=0.90)
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260926)
    args = parser.parse_args()
    require(0 < args.target <= 1 and args.resamples > 0, 'Invalid target/resample count')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    require(sha256(args.val_csv) == plan['source_view_sha256']['multisource_val_global.csv'],
            'Frozen validation view changed')
    reference = pd.read_csv(args.val_csv)
    require(len(reference) == 937 and reference['filename'].is_unique,
            'Frozen validation case count/uniqueness differs')
    require(set(reference.filename).isdisjoint(pd.read_csv(args.test_csv).filename),
            'Validation and test share filenames')
    reference['patient_id'] = reference['patient_id'].fillna('')
    # The cloud copy rewrites image_path and replaces source_image_path with
    # sha256(original path). Every other frozen validation column is preserved.
    expected = [column for column in reference.columns
                if column not in ('image_path', 'source_image_path')]
    hashed_sources = reference.source_image_path.map(
        lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest())
    val_y = reference.diagnosis_categories.isin(('cancer', 'opmd')).to_numpy(dtype=int)
    validated = {}
    val_rows = []
    cloud_val_sha = None
    # Complete validation-derived threshold selection before loading test rows.
    for model in MODELS:
        for seed in plan['seeds']:
            arch = plan['models'][model]['architecture']
            name = f'BPEX_revision_v2_final_{model}_seed{seed}_{arch}'
            folder = args.results / name
            metadata = json.loads((folder / 'validation_inference.json').read_text(encoding='utf-8'))
            summary = json.loads((folder / 'fixed_fit_summary.json').read_text(encoding='utf-8'))
            if cloud_val_sha is None:
                cloud_val_sha = summary['val_csv_sha256']
            require(summary['val_csv_sha256'] == cloud_val_sha,
                    f'Runs used different cloud validation manifests: {name}')
            path = folder / 'validation_predictions.csv'
            require(metadata['validation_predictions_sha256'] == sha256(path)
                    and metadata['val_csv_sha256'] == summary['val_csv_sha256']
                    and metadata['checkpoint_sha256'] == sha256(folder / 'best_model_fixed.pth')
                    and metadata['run'] == name and metadata['model'] == model and metadata['seed'] == seed,
                    f'Incomplete or mismatched validation export: {name}')
            prediction = pd.read_csv(path)
            prediction['patient_id'] = prediction['patient_id'].fillna('')
            require(len(prediction) == len(reference) and prediction[expected].equals(reference[expected])
                    and prediction.source_image_path.equals(hashed_sources)
                    and (prediction.image_path == prediction.filename.map(
                        lambda value: '/home/fertroll10/bpex-project/images/'
                                      'Multisource_dataset_for_transfer_learning/' + value)).all()
                    and (prediction.y_true == reference.diagnosis_categories).all(),
                    f'Validation case order or labels differ from frozen manifest: {name}')
            probs = prediction[[f'p_{label}' for label in CLASSES]].to_numpy(float)
            require(np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
                    and np.allclose(probs.sum(axis=1), 1, atol=2e-6)
                    and (np.asarray(CLASSES)[probs.argmax(axis=1)] == prediction.y_pred.to_numpy()).all(),
                    f'Invalid validation probabilities or labels: {name}')
            score = probs[:, CLASSES.index('cancer')] + probs[:, CLASSES.index('opmd')]
            threshold = threshold_for_target(val_y, score, args.target)
            validated[(model, seed)] = threshold
            val_rows.append({'model': model, 'seed': seed, 'threshold': threshold,
                             'validation_positives': int(val_y.sum()), 'validation_negatives': int((1-val_y).sum()),
                             **operating_point(val_y, score, threshold),
                             'validation_roc_auc': float(roc_auc_score(val_y, score)),
                             'validation_average_precision': float(average_precision_score(val_y, score)),
                             'validation_export_sha256': sha256(path)})

    frame, runs, _ = load_runs(plan, args.plan, args.results, args.receipts, args.test_csv)
    y = frame.diagnosis_categories.isin(('cancer', 'opmd')).to_numpy(dtype=int)
    cluster_idx, strata, _ = cluster_map(frame)
    counts = bootstrap_counts(strata, args.resamples, args.bootstrap_seed)
    test_rows, by_source_rows, interval_rows = [], [], []
    for model in MODELS:
        bootstrap_metrics = {k: [] for k in ('sensitivity', 'specificity', 'ppv', 'npv', 'f1')}
        for seed in plan['seeds']:
            score = (runs[model][seed][:, CLASSES.index('cancer')]
                     + runs[model][seed][:, CLASSES.index('opmd')])
            threshold = validated[(model, seed)]
            decision = (score >= threshold).astype(int)
            test_rows.append({'model': model, 'seed': seed, 'validation_threshold': threshold,
                              'test_positives': int(y.sum()), 'test_negatives': int((1-y).sum()),
                              **operating_point(y, score, threshold)})
            for source in sorted(frame.dataset_source.unique()):
                mask = (frame.dataset_source == source).to_numpy()
                by_source_rows.append({'model': model, 'seed': seed, 'source': source,
                                       'n_images': int(mask.sum()),
                                       **operating_point(y[mask], score[mask], threshold)})
            matrix_draws = confusion_from_clusters(y, decision, cluster_idx, counts, 2)
            for metric, values in binary_from_confusions(matrix_draws).items():
                bootstrap_metrics[metric].append(values)
        one = [row for row in test_rows if row['model'] == model]
        for metric, per_seed_draws in bootstrap_metrics.items():
            draws = np.nanmean(per_seed_draws, axis=0)
            low, high, n = interval(draws)
            interval_rows.append({'model': model, 'metric': metric,
                                  'seed_mean': float(np.mean([row['binary_f1' if metric == 'f1' else metric]
                                                               for row in one])),
                                  'seed_sd': float(np.std([row['binary_f1' if metric == 'f1' else metric]
                                                         for row in one], ddof=1)),
                                  'ci_low': low, 'ci_high': high, 'valid_draws': n})
    write_csv(args.output / 'validation_selected_thresholds.csv', val_rows)
    write_csv(args.output / 'validation_threshold_test_metrics.csv', test_rows)
    write_csv(args.output / 'validation_threshold_test_by_source.csv', by_source_rows)
    write_csv(args.output / 'validation_threshold_bootstrap_intervals.csv', interval_rows)
    (args.output / 'validation_threshold_manifest.json').write_text(json.dumps({
        'rule': 'largest threshold among validation positive scores yielding >= target sensitivity; >= comparison',
        'target_validation_sensitivity': args.target,
        'threshold_source': 'full-precision validation predictions matching one-time held-out inference',
        'cloud_manifest_reconciliation': 'cloud image_path = fixed cloud image root + filename; '
                                         'cloud source_image_path = sha256(local source_image_path); '
                                         'all other source columns align row by row with frozen local view',
        'cloud_validation_manifest_sha256': cloud_val_sha,
        'test_evaluation': 'apply each validation-selected threshold to the frozen test probabilities; '
                           'no threshold fitted to the held-out labels',
        'bootstrap_resamples': args.resamples, 'bootstrap_seed': args.bootstrap_seed,
        'validation_manifest_sha256': sha256(args.val_csv), 'test_manifest_sha256': sha256(args.test_csv),
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Applied {len(validated)} validation-selected thresholds to {len(frame)} frozen test images')


if __name__ == '__main__':
    main()
