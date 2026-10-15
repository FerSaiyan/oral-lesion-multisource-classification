#!/usr/bin/env python3
"""Full held-out metric analysis for per-seed tuned BPEX checkpoints.

Parity with analyze_bpex_final_seeds.py for the tuned arm: per-seed and
seed-mean point metrics, per-class and per-source metrics, global and
source confusion matrices, binary screening metrics and ROC/PR curves,
paired patient-aware bootstrap intervals, six Holm-corrected architecture
comparisons, and exploratory exact McNemar tests on seed-mean binary
decisions. Seeds are 42-46 (42 from the original per-architecture studies).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from analyze_bpex_final_seeds import (  # noqa: E402
    CLASSES,
    COMPARISONS,
    MODELS,
    SUSPICIOUS,
    binary_from_confusions,
    bootstrap_counts,
    cluster_map,
    confusion_from_clusters,
    holm,
    interval,
    metrics_from_confusions,
    point_metrics,
    point_metrics_source,
    require,
    sha256,
    weighted_auc_ap,
    write_csv,
)

TUNED = ROOT / 'results/revision_bpex/tuned_seedwise_test_v1'
ORIG_DIR = ROOT / 'results/revision_bpex/hpo_corrected_global'
SEEDWISE_DIR = ROOT / 'results/revision_bpex/hpo_seedwise_corrected_global_v1'
EXPECTED_TEST_SHA256 = '1e44ebb0b77b85813d6a30e7bac29f51eababe5f486d1a5a47b7552e530b3513'
CKPT = {'effnet_b3': 'effnet_b3_multisource', 'effnet_b7': 'effnet_b7_imagenet',
        'vit_base': 'vit_base_imagenet', 'vit_large': 'vit_large_imagenet'}
SEEDS = (42, 43, 44, 45, 46)


def source_dir(model: str, seed: int) -> tuple[Path, str]:
    if seed == 42:
        name = f'BPEX_revision_v2_global_val_{model}_{CKPT[model]}'
        return ORIG_DIR / name, name
    name = f'BPEX_revision_v2_seedwise_global_val_{model}_seed{seed}_{CKPT[model]}'
    return SEEDWISE_DIR / name, name


def retrain_best(source: Path) -> dict:
    rows = list(csv.DictReader((source / 'retrain_log.csv').open(encoding='utf-8')))
    best = max(rows, key=lambda r: float(r['val_f1']))
    return {'selected_epoch': int(best['epoch']), 'validation_macro_f1': float(best['val_f1']),
            'validation_weighted_f1': float(best['val_f1_weighted'])}


def load_runs(test_csv: Path) -> tuple[pd.DataFrame, dict[str, dict[int, np.ndarray]], list[dict]]:
    require(sha256(test_csv) == EXPECTED_TEST_SHA256, 'Frozen materialized test manifest changed')
    reference = pd.read_csv(test_csv)
    require(reference['filename'].is_unique and len(reference) == 1209,
            'Expected 1,209 unique frozen held-out filenames')
    reference['patient_id'] = reference['patient_id'].fillna('')
    probabilities: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    provenance: list[dict] = []
    for model in MODELS:
        for seed in SEEDS:
            source, name = source_dir(model, seed)
            record = json.loads((TUNED / name / 'test_eval_provenance.json').read_text(encoding='utf-8'))
            require(record['test_csv_sha256'] == EXPECTED_TEST_SHA256
                    and record['test_evaluated_once'] and record['test_rows'] == 1209,
                    f'Bad tuned provenance: {name}')
            data = pd.read_csv(TUNED / name / 'test_predictions.csv')
            require(len(data) == 1209 and list(data['filename']) == list(reference['filename']),
                    f'Prediction rows do not match frozen manifest: {name}')
            scores = data[[f'p_{label}' for label in CLASSES]].to_numpy(dtype=float)
            require(np.all(scores >= 0) and np.allclose(scores.sum(axis=1), 1.0, atol=1e-4),
                    f'Invalid tuned probabilities: {name}')
            require(list(data['y_pred']) == [CLASSES[i] for i in scores.argmax(axis=1)],
                    f'Predicted labels disagree with probabilities: {name}')
            probabilities[model][seed] = scores
            provenance.append({'model': model, 'seed': seed, 'source_study': name,
                               'checkpoint_sha256': record['checkpoint_sha256'],
                               'predictions_sha256': sha256(TUNED / name / 'test_predictions.csv'),
                               'test_evaluated_once': True, **retrain_best(source)})
    return reference, probabilities, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-csv', type=Path,
                        default=ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24/multisource_test_global.csv')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/revision_bpex/tuned_seedwise_test_analysis_v1')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260930)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame, probabilities, provenance = load_runs(args.test_csv)
    y = pd.Categorical(frame['diagnosis_categories'], categories=list(CLASSES)).codes
    require(np.all(y >= 0), 'Unknown held-out class')
    bin_y = np.isin(np.asarray(CLASSES)[y], list(SUSPICIOUS)).astype(int)
    cluster_idx, strata, audit = cluster_map(frame)
    counts = bootstrap_counts(strata, args.resamples, args.bootstrap_seed)
    weights = counts[:, cluster_idx]
    write_csv(args.output / 'cluster_audit.csv', audit)
    write_csv(args.output / 'run_provenance.csv', provenance)
    seed_rows, class_rows, source_rows, confusion_rows, binary_confusion_rows = [], [], [], [], []
    curve_rows, pr_rows, ovr_curve_rows = [], [], []
    source_boot_rows, bootstrap_rows, comparison_rows, mcnemar_rows = [], [], [], []
    boot: dict[str, dict[str, np.ndarray]] = {}
    seed_map = {(item['model'], item['seed']): item for item in provenance}
    source_values = frame['dataset_source'].to_numpy()
    for model in MODELS:
        runs = probabilities[model]
        models_boot: dict[str, list[np.ndarray]] = defaultdict(list)
        for seed in SEEDS:
            prob = runs[seed]
            metrics, matrix, classes = point_metrics(y, prob)
            seed_rows.append({'model': model, 'seed': seed, **metrics,
                              **{key: seed_map[(model, seed)][key] for key in
                                 ('validation_macro_f1', 'validation_weighted_f1', 'selected_epoch')}})
            class_rows.extend({'model': model, 'seed': seed, **row} for row in classes)
            for i, true_class in enumerate(CLASSES):
                for j, predicted_class in enumerate(CLASSES):
                    confusion_rows.append({'model': model, 'seed': seed, 'source': 'GLOBAL',
                                           'true': true_class, 'predicted': predicted_class,
                                           'count': int(matrix[i, j]),
                                           'row_fraction': float(matrix[i, j] / matrix[i].sum())
                                           if matrix[i].sum() else None})
            for source in sorted(frame['dataset_source'].unique()):
                mask = source_values == source
                sub_metrics, source_matrix, _ = point_metrics_source(y[mask], prob[mask])
                source_rows.append({'model': model, 'seed': seed, 'source': source,
                                    'n_images': int(mask.sum()), **sub_metrics})
                for i, true_class in enumerate(CLASSES):
                    for j, predicted_class in enumerate(CLASSES):
                        confusion_rows.append({'model': model, 'seed': seed, 'source': source,
                                               'true': true_class, 'predicted': predicted_class,
                                               'count': int(source_matrix[i, j]),
                                               'row_fraction': float(source_matrix[i, j] / source_matrix[i].sum())
                                               if source_matrix[i].sum() else None})
            pred = prob.argmax(axis=1)
            conf = confusion_from_clusters(y, pred, cluster_idx, counts, 4)
            for key, value in metrics_from_confusions(conf).items():
                if key != 'per_class_recall':
                    models_boot[key].append(value)
            score = prob[:, CLASSES.index('cancer')] + prob[:, CLASSES.index('opmd')]
            fpr, tpr, thresholds = roc_curve(bin_y, score)
            curve_rows.extend({'model': model, 'seed': seed, 'fpr': fpr[i], 'tpr': tpr[i],
                               'threshold': thresholds[i]} for i in range(len(fpr)))
            precision, recall, thresholds = precision_recall_curve(bin_y, score)
            pr_rows.extend({'model': model, 'seed': seed, 'precision': precision[i],
                            'recall': recall[i],
                            'threshold': thresholds[i] if i < len(thresholds) else None}
                           for i in range(len(precision)))
            for j, label in enumerate(CLASSES):
                fpr, tpr, thresholds = roc_curve(y == j, prob[:, j])
                ovr_curve_rows.extend({'model': model, 'seed': seed, 'class': label,
                                       'fpr': fpr[i], 'tpr': tpr[i], 'threshold': thresholds[i]}
                                      for i in range(len(fpr)))
            binary_pred = (score >= 0.5).astype(int)
            for source in ('GLOBAL', *sorted(frame['dataset_source'].unique())):
                mask = source_values == source if source != 'GLOBAL' else np.ones(len(y), dtype=bool)
                bc = confusion_matrix(bin_y[mask], binary_pred[mask], labels=(0, 1))
                for i, true_class in enumerate(('non_suspicious', 'suspicious')):
                    for j, predicted_class in enumerate(('non_suspicious', 'suspicious')):
                        binary_confusion_rows.append({'model': model, 'seed': seed, 'source': source,
                                                      'true': true_class, 'predicted': predicted_class,
                                                      'count': int(bc[i, j]),
                                                      'row_fraction': float(bc[i, j] / bc[i].sum())
                                                      if bc[i].sum() else None})
            binary_conf = confusion_from_clusters(bin_y, binary_pred, cluster_idx, counts, 2)
            for key, value in binary_from_confusions(binary_conf).items():
                models_boot[f'binary_{key}'].append(value)
            auc, ap = weighted_auc_ap(bin_y, score, weights)
            models_boot['binary_roc_auc'].append(auc)
            models_boot['binary_average_precision'].append(ap)
        boot[model] = {key: np.nanmean(np.stack(value, axis=0), axis=0)
                       for key, value in models_boot.items()}
        for source in sorted(frame['dataset_source'].unique()):
            mask = source_values == source
            per_seed_draws = []
            for seed in SEEDS:
                conf = confusion_from_clusters(y, runs[seed].argmax(axis=1),
                                               cluster_idx, counts, 4, mask=mask)
                mat = metrics_from_confusions(conf)
                diag = np.diagonal(conf, axis1=-2, axis2=-1)
                support, predicted = conf.sum(axis=-1), conf.sum(axis=-2)
                den = support + predicted
                class_f1 = np.divide(2 * diag, den, out=np.zeros_like(diag), where=den > 0)
                source_labels = np.unique(y[mask])
                per_seed_draws.append(class_f1[:, source_labels].mean(axis=1))
            draws = np.mean(per_seed_draws, axis=0)
            low, high, valid = interval(draws)
            source_boot_rows.append({'model': model, 'source': source,
                                     'metric': 'macro_f1_present_classes',
                                     'seed_mean': float(np.mean([
                                         point_metrics_source(y[mask], runs[s][mask])[0]['macro_f1_present_classes']
                                         for s in SEEDS])),
                                     'ci_low': low, 'ci_high': high, 'valid_draws': valid})

    seed_frame = pd.DataFrame(seed_rows)
    summary_rows = []
    scalar_metrics = [col for col in seed_frame.columns if col not in ('model', 'seed', 'selected_epoch')]
    for model in MODELS:
        one = seed_frame[seed_frame['model'] == model]
        row = {'model': model, 'n_seeds': len(one)}
        for metric in scalar_metrics:
            row[f'{metric}_mean'] = float(one[metric].mean())
            row[f'{metric}_sd'] = float(one[metric].std(ddof=1))
        summary_rows.append(row)
        for metric, draws in boot[model].items():
            low, high, valid = interval(draws)
            bootstrap_rows.append({'model': model, 'metric': metric,
                                   'seed_mean': float(one[metric].mean()) if metric in one else None,
                                   'ci_low': low, 'ci_high': high, 'valid_draws': valid})
    for a, b in COMPARISONS:
        primary = boot[a]['macro_f1'] - boot[b]['macro_f1']
        primary_finite = primary[np.isfinite(primary)]
        p = min(1.0, 2 * min((np.count_nonzero(primary_finite <= 0) + 1) /
                             (len(primary_finite) + 1),
                             (np.count_nonzero(primary_finite >= 0) + 1) /
                             (len(primary_finite) + 1)))
        row = {'model_a': a, 'model_b': b, 'macro_f1_difference': float(
            seed_frame.query('model == @a')['macro_f1'].mean() -
            seed_frame.query('model == @b')['macro_f1'].mean()),
            'macro_f1_ci_low': interval(primary)[0],
            'macro_f1_ci_high': interval(primary)[1],
            'macro_f1_bootstrap_tail_p': p,
            'macro_f1_valid_draws': len(primary_finite)}
        for metric in ('weighted_f1', 'binary_roc_auc'):
            diff = boot[a][metric] - boot[b][metric]
            row[f'{metric}_difference'] = float(seed_frame.query('model == @a')[metric].mean() -
                                                seed_frame.query('model == @b')[metric].mean())
            row[f'{metric}_ci_low'], row[f'{metric}_ci_high'], _ = interval(diff)
        comparison_rows.append(row)

        scores_a = np.mean(np.stack(list(probabilities[a].values())), axis=0)
        scores_b = np.mean(np.stack(list(probabilities[b].values())), axis=0)
        decision_a = scores_a[:, CLASSES.index('cancer')] + scores_a[:, CLASSES.index('opmd')] >= 0.5
        decision_b = scores_b[:, CLASSES.index('cancer')] + scores_b[:, CLASSES.index('opmd')] >= 0.5
        correct_a = decision_a == bin_y
        correct_b = decision_b == bin_y
        a_only = int(np.count_nonzero(correct_a & ~correct_b))
        b_only = int(np.count_nonzero(correct_b & ~correct_a))
        discordant = a_only + b_only
        mcnemar_rows.append({'model_a': a, 'model_b': b,
                             'a_only_correct': a_only, 'b_only_correct': b_only,
                             'discordant': discordant,
                             'exact_mcnemar_p': float(binomtest(a_only, discordant, 0.5).pvalue)
                             if discordant else 1.0})
    for row, adjusted in zip(comparison_rows,
                             holm([row['macro_f1_bootstrap_tail_p'] for row in comparison_rows])):
        row['macro_f1_holm_p'] = adjusted
    for row, adjusted in zip(mcnemar_rows,
                             holm([row['exact_mcnemar_p'] for row in mcnemar_rows])):
        row['exact_mcnemar_holm_p'] = adjusted

    write_csv(args.output / 'per_seed_metrics.csv', seed_rows)
    write_csv(args.output / 'model_summary.csv', summary_rows)
    write_csv(args.output / 'per_class_metrics.csv', class_rows)
    write_csv(args.output / 'per_source_metrics.csv', source_rows)
    write_csv(args.output / 'source_bootstrap_intervals.csv', source_boot_rows)
    write_csv(args.output / 'confusion_matrices.csv', confusion_rows)
    write_csv(args.output / 'binary_confusion_matrices.csv', binary_confusion_rows)
    write_csv(args.output / 'binary_roc_curves.csv', curve_rows)
    write_csv(args.output / 'binary_pr_curves.csv', pr_rows)
    write_csv(args.output / 'multiclass_ovr_roc_curves.csv', ovr_curve_rows)
    write_csv(args.output / 'bootstrap_intervals.csv', bootstrap_rows)
    write_csv(args.output / 'paired_comparisons.csv', comparison_rows)
    write_csv(args.output / 'binary_exact_mcnemar.csv', mcnemar_rows)
    summary = {
        'case_count': len(frame), 'class_counts': frame['diagnosis_categories'].value_counts().to_dict(),
        'source_counts': frame['dataset_source'].value_counts().to_dict(),
        'patient_clusters': {key: len(ids) for key, ids in strata.items()},
        'bootstrap_resamples': args.resamples, 'bootstrap_seed': args.bootstrap_seed,
        'bootstrap_design': 'shared paired draws; stratified by INTERNAL/Zenodo/Kaggle/Mendeley; '
                            'internal camera+ROMEU and Zenodo by patient; Kaggle/Mendeley by image',
        'primary_comparisons': [f'{a} - {b}' for a, b in COMPARISONS],
        'primary_inference': 'seed-mean macro F1 difference; percentile 95% interval; '
                             'two-sided empirical bootstrap-tail p with Holm correction across six comparisons',
        'exploratory_status': 'per-seed tuned checkpoints; report separately from the fixed-parameter primary analysis',
        'limitations': ['Kaggle/Mendeley patient IDs unavailable: image-level CIs may be anti-conservative',
                        'Bootstrap CIs condition on the five tuned checkpoints; no uncertainty from other seeds',
                        'The same test set must not be used for additional hyperparameter/threshold selection'],
    }
    (args.output / 'analysis_manifest.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(f'Analyzed 20 tuned fits on {len(frame)} held-out images using {args.resamples} paired resamples')


if __name__ == '__main__':
    main()
