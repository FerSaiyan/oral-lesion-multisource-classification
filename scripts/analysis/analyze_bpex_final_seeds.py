#!/usr/bin/env python3
"""Audit BPEX final predictions and compute paired, patient-aware statistics.

All five seeds share the same frozen held-out cases. Bootstrap resamples are
identical across seeds/models; internal and Zenodo photographs move together
by patient. Kaggle/Mendeley lack known patient IDs and use image-level proxies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ('benign_lesion', 'cancer', 'healthy', 'opmd')
MODELS = ('effnet_b3', 'effnet_b7', 'vit_base', 'vit_large')
SUSPICIOUS = {'cancer', 'opmd'}
COMPARISONS = (
    ('effnet_b7', 'effnet_b3'),
    ('vit_large', 'vit_base'),
    ('effnet_b7', 'vit_base'),
    ('effnet_b7', 'vit_large'),
    ('effnet_b3', 'vit_base'),
    ('effnet_b3', 'vit_large'),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_runs(plan: dict, plan_path: Path, results: Path, receipts: Path,
              test_csv: Path) -> tuple[pd.DataFrame, dict[str, dict[int, np.ndarray]], list[dict]]:
    require(sha256(test_csv) == plan['source_view_sha256']['multisource_test_global.csv'],
            'Frozen materialized test manifest changed')
    reference = pd.read_csv(test_csv)
    require(reference['filename'].is_unique and len(reference) == 1209,
            'Expected 1,209 unique frozen held-out filenames')
    reference['patient_id'] = reference['patient_id'].fillna('')
    expected = ('filename', 'dataset_source', 'patient_id', 'group_all',
                'diagnosis_categories', 'source_image_path')
    probabilities: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    provenance: list[dict] = []
    for model in MODELS:
        architecture = plan['models'][model]['architecture']
        for seed in plan['seeds']:
            name = f'BPEX_revision_v2_final_{model}_seed{seed}_{architecture}'
            directory = results / name
            receipt = json.loads((receipts / f'{name}.json').read_text(encoding='utf-8'))
            fixed = json.loads((directory / 'fixed_fit_summary.json').read_text(encoding='utf-8'))
            record = json.loads((directory / 'final_fit_provenance.json').read_text(encoding='utf-8'))
            prediction_path = directory / 'test_predictions.csv'
            require(receipt['run'] == name and receipt['seed'] == seed and receipt['model'] == model,
                    f'Mismatched transfer receipt: {name}')
            require(receipt['plan_sha256'] == record['selection_plan_sha256'] == sha256(plan_path),
                    f'Mismatched frozen plan: {name}')
            require(receipt['predictions_sha256'] == fixed['test_predictions_sha256'] ==
                    record['test_predictions_sha256'] == sha256(prediction_path),
                    f'Prediction checksum mismatch: {name}')
            require(fixed['execution_mode'] == 'fixed_single_fit' and fixed['fit_count'] == 1
                    and fixed['seed'] == seed and record['test_evaluated_once'],
                    f'Run was not one independent fixed fit: {name}')
            require(fixed['selected_epoch'] == record['validation_selected_epoch'],
                    f'Selected epoch mismatch: {name}')
            require(fixed['test_csv_sha256'] ==
                    json.loads((ROOT / 'data/processed/bpex_revision_v2_cloud_test_2026-09-26/multisource_test_global.json').read_text())['derived_sha256'],
                    f'Wrong held-out manifest: {name}')
            data = pd.read_csv(prediction_path)
            data['patient_id'] = data['patient_id'].fillna('')
            require(len(data) == len(reference) and data['filename'].is_unique,
                    f'Incomplete held-out predictions: {name}')
            columns = data[list(expected)].copy()
            columns = columns.rename(columns={'diagnosis_categories': 'diagnosis_categories'})
            require(columns.equals(reference[list(expected)]),
                    f'Cases, order, provenance or labels differ from frozen holdout: {name}')
            require((data['y_true'].to_numpy() == reference['diagnosis_categories'].to_numpy()).all(),
                    f'True labels disagree with frozen holdout: {name}')
            scores = data[[f'p_{label}' for label in CLASSES]].to_numpy(dtype=np.float64)
            require(np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()
                    and np.allclose(scores.sum(axis=1), 1, atol=2e-6),
                    f'Invalid four-class probabilities: {name}')
            require((data['y_pred'].to_numpy() == np.asarray(CLASSES)[scores.argmax(axis=1)]).all(),
                    f'Predicted labels disagree with probabilities: {name}')
            probabilities[model][seed] = scores
            provenance.append({
                'model': model, 'seed': seed, 'run': name,
                'selected_epoch': fixed['selected_epoch'],
                'validation_macro_f1': record['validation_macro_f1'],
                'validation_weighted_f1': record['validation_weighted_f1'],
                'source_optuna_trial': record['optuna_source_trial'],
                'predictions_sha256': receipt['predictions_sha256'],
            })
    return reference, probabilities, provenance


def cluster_map(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, list[int]], list[dict]]:
    strata: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, row in frame.iterrows():
        source = row['dataset_source']
        if source in ('MY_CAMERA', 'MY_ROMEU'):
            stratum = 'INTERNAL'
            patient = str(row['patient_id']).strip()
            require(patient and patient.lower() != 'nan', 'Internal patient ID missing')
            key = patient
        elif source == 'ZENODO':
            stratum = 'ZENODO'
            patient = str(row['patient_id']).strip()
            require(patient and patient.lower() != 'nan', 'Zenodo patient ID missing')
            key = patient
        else:
            require(source in ('KAGGLE', 'MENDELEY') and not row['patient_id'],
                    f'Unexpected external grouping: {source}')
            stratum = source
            key = row['filename']
        strata[stratum][key].append(int(idx))
    cluster_idx = np.full(len(frame), -1, dtype=np.int32)
    by_stratum: dict[str, list[int]] = {}
    audit: list[dict] = []
    for stratum in ('INTERNAL', 'ZENODO', 'KAGGLE', 'MENDELEY'):
        by_stratum[stratum] = []
        for key, indices in strata[stratum].items():
            cluster = len(audit)
            cluster_idx[indices] = cluster
            by_stratum[stratum].append(cluster)
            audit.append({'stratum': stratum, 'cluster': cluster,
                          'group_key': key, 'image_count': len(indices)})
    require(np.all(cluster_idx >= 0), 'Unassigned bootstrap cluster')
    return cluster_idx, by_stratum, audit


def bootstrap_counts(by_stratum: dict[str, list[int]], n_boot: int,
                     seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_boot, sum(map(len, by_stratum.values()))), dtype=np.float32)
    for cluster_ids in by_stratum.values():
        n = len(cluster_ids)
        counts[:, cluster_ids] = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    return counts


def confusion_from_clusters(y_true: np.ndarray, y_pred: np.ndarray,
                            cluster_idx: np.ndarray, counts: np.ndarray,
                            n_classes: int, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)
    cells = np.bincount(cluster_idx[mask] * n_classes * n_classes +
                        y_true[mask] * n_classes + y_pred[mask],
                        minlength=counts.shape[1] * n_classes * n_classes)
    aggregate = cells.reshape(counts.shape[1], n_classes * n_classes).astype(np.float32)
    return (counts @ aggregate).reshape(len(counts), n_classes, n_classes)


def metrics_from_confusions(conf: np.ndarray) -> dict[str, np.ndarray]:
    diag = np.diagonal(conf, axis1=-2, axis2=-1)
    support = conf.sum(axis=-1)
    predicted = conf.sum(axis=-2)
    recall = np.divide(diag, support, out=np.zeros_like(diag), where=support > 0)
    precision = np.divide(diag, predicted, out=np.zeros_like(diag), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(diag), where=(precision + recall) > 0)
    total = support.sum(axis=-1)
    return {
        'macro_f1': f1.mean(axis=-1),
        'weighted_f1': np.divide((f1 * support).sum(axis=-1), total,
                                 out=np.full_like(total, np.nan), where=total > 0),
        'balanced_accuracy': recall.mean(axis=-1),
        'accuracy': np.divide(diag.sum(axis=-1), total,
                              out=np.full_like(total, np.nan), where=total > 0),
        'per_class_recall': recall,
    }


def binary_from_confusions(conf: np.ndarray) -> dict[str, np.ndarray]:
    tn, fp, fn, tp = (conf[..., 0, 0], conf[..., 0, 1],
                      conf[..., 1, 0], conf[..., 1, 1])

    def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    return {'sensitivity': ratio(tp, tp + fn), 'specificity': ratio(tn, tn + fp),
            'ppv': ratio(tp, tp + fp), 'npv': ratio(tn, tn + fn),
            'f1': ratio(2 * tp, 2 * tp + fp + fn)}


def weighted_auc_ap(y_true: np.ndarray, score: np.ndarray, weights: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ROC-AUC and average precision with exact probability ties."""
    order = np.argsort(score, kind='stable')
    sorted_score = score[order]
    labels = y_true[order].astype(np.float32)
    w = weights[:, order]
    pos, neg = w * labels, w * (1 - labels)
    starts = np.flatnonzero(np.r_[True, sorted_score[1:] != sorted_score[:-1]])
    ends = np.r_[starts[1:] - 1, len(order) - 1]
    group_start = np.repeat(starts, ends - starts + 1)
    group_end = np.repeat(ends, ends - starts + 1)
    cum_neg = np.cumsum(neg, axis=1)
    before = cum_neg[:, group_start] - neg[:, group_start]
    tied = cum_neg[:, group_end] - before
    positives, negatives = pos.sum(axis=1), neg.sum(axis=1)
    auc = np.divide((pos * (before + 0.5 * tied)).sum(axis=1),
                     positives * negatives, out=np.full(len(w), np.nan),
                     where=(positives * negatives) > 0)

    reverse = np.argsort(-score, kind='stable')
    ranked_score = score[reverse]
    ranked_pos = weights[:, reverse] * y_true[reverse]
    ranked_weight = weights[:, reverse]
    starts = np.flatnonzero(np.r_[True, ranked_score[1:] != ranked_score[:-1]])
    ends = np.r_[starts[1:] - 1, len(reverse) - 1]
    group_end = np.repeat(ends, ends - starts + 1)
    cum_pos = np.cumsum(ranked_pos, axis=1)
    cum_weight = np.cumsum(ranked_weight, axis=1)
    precision = np.divide(cum_pos[:, group_end], cum_weight[:, group_end],
                          out=np.zeros_like(cum_pos), where=cum_weight[:, group_end] > 0)
    ap = np.divide((ranked_pos * precision).sum(axis=1), positives,
                   out=np.full(len(w), np.nan), where=positives > 0)
    return auc, ap


def point_metrics(y: np.ndarray, probs: np.ndarray) -> tuple[dict, np.ndarray, list[dict]]:
    pred = probs.argmax(axis=1)
    matrix = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=range(len(CLASSES)), zero_division=0)
    total = matrix.sum()
    per_class = []
    for i, label in enumerate(CLASSES):
        tn = total - matrix[i, :].sum() - matrix[:, i].sum() + matrix[i, i]
        fp = matrix[:, i].sum() - matrix[i, i]
        per_class.append({
            'class': label, 'support': int(support[i]), 'precision': float(precision[i]),
            'recall': float(recall[i]), 'f1': float(f1[i]),
            'specificity': float(tn / (tn + fp)) if tn + fp else None,
            'roc_auc_ovr': float(roc_auc_score(y == i, probs[:, i]))
            if 0 < (y == i).sum() < len(y) else None,
        })
    suspicious = np.isin(np.asarray(CLASSES)[y], list(SUSPICIOUS)).astype(int)
    score = probs[:, CLASSES.index('cancer')] + probs[:, CLASSES.index('opmd')]
    binary = (score >= 0.5).astype(int)
    bc = confusion_matrix(suspicious, binary, labels=(0, 1))
    tn, fp, fn, tp = bc.ravel()
    safe = lambda num, den: float(num / den) if den else None
    result = {
        'macro_f1': float(f1.mean()),
        'weighted_f1': float(f1 @ support / total),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'accuracy': float(accuracy_score(y, pred)),
        'top2_accuracy': float(np.mean([y[i] in np.argsort(probs[i])[-2:] for i in range(len(y))])),
        'ovr_roc_auc_macro': float(np.mean([x['roc_auc_ovr'] for x in per_class if x['roc_auc_ovr'] is not None])),
        'binary_sensitivity': safe(tp, tp + fn), 'binary_specificity': safe(tn, tn + fp),
        'binary_ppv': safe(tp, tp + fp), 'binary_npv': safe(tn, tn + fn),
        'binary_f1': safe(2 * tp, 2 * tp + fp + fn),
        'binary_roc_auc': float(roc_auc_score(suspicious, score)),
        'binary_average_precision': float(average_precision_score(suspicious, score)),
        'binary_tn': int(tn), 'binary_fp': int(fp), 'binary_fn': int(fn), 'binary_tp': int(tp),
    }
    return result, matrix, per_class


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    corrected = np.zeros(len(pvalues))
    previous = 0.0
    for rank, position in enumerate(order):
        previous = max(previous, min(1.0, (len(pvalues) - rank) * pvalues[position]))
        corrected[position] = previous
    return corrected.tolist()


def interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return (float('nan'), float('nan'), 0)
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), len(finite)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'Refusing empty output: {path}')
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/studies/bpex_revision_v2_final_seeds_2026-09-26.json')
    parser.add_argument('--results', type=Path, default=ROOT / 'results/revision_bpex/final_seeds_v2_corrected_global')
    parser.add_argument('--receipts', type=Path, default=ROOT / 'results/revision_bpex/cloud_final_seed_sync')
    parser.add_argument('--test-csv', type=Path, default=ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24/multisource_test_global.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/revision_bpex/five_seed_test_analysis_v1')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260926)
    args = parser.parse_args()
    require(args.resamples > 0, 'Bootstrap must contain at least one resample')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    frame, probabilities, provenance = load_runs(plan, args.plan, args.results,
                                                 args.receipts, args.test_csv)
    y = pd.Categorical(frame['diagnosis_categories'], categories=CLASSES).codes
    require(np.all(y >= 0), 'Unknown held-out class')
    bin_y = np.isin(np.asarray(CLASSES)[y], list(SUSPICIOUS)).astype(int)
    cluster_idx, strata, audit = cluster_map(frame)
    counts = bootstrap_counts(strata, args.resamples, args.bootstrap_seed)
    weights = counts[:, cluster_idx]
    write_csv(args.output / 'cluster_audit.csv', audit)
    write_csv(args.output / 'run_provenance.csv', provenance)
    seed_rows, class_rows, source_rows, confusion_rows, binary_confusion_rows = [], [], [], [], []
    ensemble_rows, curve_rows, pr_rows, ovr_curve_rows = [], [], [], []
    source_boot_rows, bootstrap_rows, comparison_rows, mcnemar_rows = [], [], [], []
    boot: dict[str, dict[str, np.ndarray]] = {}
    seed_map = {(item['model'], item['seed']): item for item in provenance}
    source_values = frame['dataset_source'].to_numpy()
    for model in MODELS:
        runs = probabilities[model]
        models_boot: dict[str, list[np.ndarray]] = defaultdict(list)
        for seed in plan['seeds']:
            prob = runs[seed]
            metrics, matrix, classes = point_metrics(y, prob)
            seed_rows.append({'model': model, 'seed': seed, **metrics,
                              **{key: seed_map[(model, seed)][key] for key in
                                 ('validation_macro_f1', 'validation_weighted_f1', 'selected_epoch')}})
            class_rows.extend({'model': model, 'seed': seed, **row} for row in classes)
            for i, true_class in enumerate(CLASSES):
                for j, predicted_class in enumerate(CLASSES):
                    confusion_rows.append({'model': model, 'seed': seed, 'source': 'GLOBAL', 'true': true_class,
                                           'predicted': predicted_class, 'count': int(matrix[i, j]),
                                           'row_fraction': float(matrix[i, j] / matrix[i].sum()) if matrix[i].sum() else None})
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
        ensemble = np.mean(np.stack([runs[seed] for seed in plan['seeds']]), axis=0)
        points, matrix, classes = point_metrics(y, ensemble)
        ensemble_rows.append({'model': model, 'n_seeds': len(plan['seeds']), **points})
        class_rows.extend({'model': model, 'seed': 'ensemble', **row} for row in classes)
        for i, true_class in enumerate(CLASSES):
            for j, predicted_class in enumerate(CLASSES):
                confusion_rows.append({'model': model, 'seed': 'ensemble', 'source': 'GLOBAL', 'true': true_class,
                                       'predicted': predicted_class, 'count': int(matrix[i, j]),
                                       'row_fraction': float(matrix[i, j] / matrix[i].sum()) if matrix[i].sum() else None})
        score = ensemble[:, CLASSES.index('cancer')] + ensemble[:, CLASSES.index('opmd')]
        for source in ('GLOBAL', *sorted(frame['dataset_source'].unique())):
            mask = source_values == source if source != 'GLOBAL' else np.ones(len(y), dtype=bool)
            bc = confusion_matrix(bin_y[mask], score[mask] >= 0.5, labels=(0, 1))
            for i, true_class in enumerate(('non_suspicious', 'suspicious')):
                for j, predicted_class in enumerate(('non_suspicious', 'suspicious')):
                    binary_confusion_rows.append({'model': model, 'seed': 'ensemble', 'source': source,
                                                  'true': true_class, 'predicted': predicted_class,
                                                  'count': int(bc[i, j]),
                                                  'row_fraction': float(bc[i, j] / bc[i].sum())
                                                  if bc[i].sum() else None})
        fpr, tpr, thresholds = roc_curve(bin_y, score)
        curve_rows.extend({'model': model, 'seed': 'ensemble', 'fpr': fpr[i], 'tpr': tpr[i],
                           'threshold': thresholds[i]} for i in range(len(fpr)))
        precision, recall, thresholds = precision_recall_curve(bin_y, score)
        pr_rows.extend({'model': model, 'seed': 'ensemble', 'precision': precision[i], 'recall': recall[i],
                        'threshold': thresholds[i] if i < len(thresholds) else None}
                       for i in range(len(precision)))
        for j, label in enumerate(CLASSES):
            fpr, tpr, thresholds = roc_curve(y == j, ensemble[:, j])
            ovr_curve_rows.extend({'model': model, 'seed': 'ensemble', 'class': label, 'fpr': fpr[i],
                                   'tpr': tpr[i], 'threshold': thresholds[i]}
                                  for i in range(len(fpr)))
        for source in sorted(frame['dataset_source'].unique()):
            mask = source_values == source
            point, source_matrix, _ = point_metrics_source(y[mask], ensemble[mask])
            source_rows.append({'model': model, 'seed': 'ensemble', 'source': source,
                                'n_images': int(mask.sum()), **point})
            for i, true_class in enumerate(CLASSES):
                for j, predicted_class in enumerate(CLASSES):
                    confusion_rows.append({'model': model, 'seed': 'ensemble', 'source': source,
                                           'true': true_class, 'predicted': predicted_class,
                                           'count': int(source_matrix[i, j]),
                                           'row_fraction': float(source_matrix[i, j] / source_matrix[i].sum())
                                           if source_matrix[i].sum() else None})
            source_labels = np.unique(y[mask])
            per_seed_draws = []
            for seed in plan['seeds']:
                conf = confusion_from_clusters(y, runs[seed].argmax(axis=1),
                                               cluster_idx, counts, 4, mask=mask)
                mat = metrics_from_confusions(conf)
                # Absent classes are excluded only for source-specific macro F1.
                diag = np.diagonal(conf, axis1=-2, axis2=-1)
                support, predicted = conf.sum(axis=-1), conf.sum(axis=-2)
                den = support + predicted
                class_f1 = np.divide(2 * diag, den, out=np.zeros_like(diag), where=den > 0)
                per_seed_draws.append(class_f1[:, source_labels].mean(axis=1))
            draws = np.mean(per_seed_draws, axis=0)
            low, high, valid = interval(draws)
            source_boot_rows.append({'model': model, 'source': source,
                                     'metric': 'macro_f1_present_classes',
                                     'seed_mean': float(np.mean([
                                         point_metrics_source(y[mask], runs[s][mask])[0]['macro_f1_present_classes']
                                         for s in plan['seeds']])),
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
    write_csv(args.output / 'ensemble_metrics.csv', ensemble_rows)
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
    np.savez_compressed(args.output / 'paired_bootstrap_draws.npz',
                        **{f'{model}_{metric}': values for model, draws in boot.items()
                           for metric, values in draws.items()})
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
        'validation_threshold': 'High-sensitivity threshold requires per-image validation probabilities; '
                                'not chosen from this held-out test set.',
        'limitations': ['Kaggle/Mendeley patient IDs unavailable: image-level CIs may be anti-conservative',
                        'The five fixed seeds estimate training variability; bootstrap CIs condition on these seeds',
                        'The same test set must not be used for additional hyperparameter/threshold selection'],
    }
    (args.output / 'analysis_manifest.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(f"Analyzed 20 fits on {len(frame)} held-out images using {args.resamples} paired resamples")


def point_metrics_source(y: np.ndarray, probs: np.ndarray) -> tuple[dict, np.ndarray, list[dict]]:
    """Source subsets can omit classes, so mark undefined binary measures."""
    pred = probs.argmax(axis=1)
    present = np.unique(y)
    full, matrix, classes = point_metrics_source_safe(y, probs)
    full['macro_f1_present_classes'] = float(f1_score(y, pred, labels=present,
                                                       average='macro', zero_division=0))
    return full, matrix, classes


def point_metrics_source_safe(y: np.ndarray, probs: np.ndarray) -> tuple[dict, np.ndarray, list[dict]]:
    pred = probs.argmax(axis=1)
    matrix = confusion_matrix(y, pred, labels=range(4))
    is_suspicious = np.isin(np.asarray(CLASSES)[y], list(SUSPICIOUS)).astype(int)
    score = probs[:, CLASSES.index('cancer')] + probs[:, CLASSES.index('opmd')]
    binary_pred = score >= 0.5
    tn, fp, fn, tp = confusion_matrix(is_suspicious, binary_pred, labels=(0, 1)).ravel()
    safe = lambda a, b: float(a / b) if b else None
    return ({'macro_f1_four_classes': float(f1_score(y, pred, labels=range(4), average='macro', zero_division=0)),
             'binary_sensitivity': safe(tp, tp + fn), 'binary_specificity': safe(tn, tn + fp),
             'binary_roc_auc': float(roc_auc_score(is_suspicious, score))
             if len(np.unique(is_suspicious)) == 2 else None,
             'binary_average_precision': float(average_precision_score(is_suspicious, score))
             if len(np.unique(is_suspicious)) == 2 else None}, matrix, [])


if __name__ == '__main__':
    main()
