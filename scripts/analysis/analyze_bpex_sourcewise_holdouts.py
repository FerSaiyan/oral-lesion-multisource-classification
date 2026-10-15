#!/usr/bin/env python3
"""Audited five-seed source-exclusion metrics from frozen prediction bundles."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_curve)

from analyze_bpex_final_seeds import (
    CLASSES, COMPARISONS, MODELS, bootstrap_counts, cluster_map,
    confusion_from_clusters, holm, interval, require, sha256, weighted_auc_ap,
)
from queue_bpex_sourcewise_hpo import PLAN, RESULTS, name, states
from run_bpex_revision_hpo import ROOT

PREDICTIONS = ROOT / 'results/revision_bpex/sourcewise_heldout_v1'
VIEW = ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24'
OUTPUT = ROOT / 'results/revision_bpex/sourcewise_heldout_analysis_v1'
SEEDS = (42, 43, 44, 45, 46)
SOURCES = ('kaggle', 'zenodo', 'mendeley')


def ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def confusion_metrics(conf, present):
    conf = np.asarray(conf, dtype=float)
    diag = np.diagonal(conf, axis1=-2, axis2=-1)
    support, predicted = conf.sum(axis=-1), conf.sum(axis=-2)
    total = support.sum(axis=-1)
    # F1 is zero when neither truth nor prediction includes a class; both definitions saved.
    f1 = np.nan_to_num(ratio(2 * diag, support + predicted), nan=0)
    recall = ratio(diag, support)
    precision = np.nan_to_num(ratio(diag, predicted), nan=0)
    tn = total[..., None] - support - predicted + diag
    specificity = ratio(tn, total[..., None] - support)
    scalar = {'macro_f1_four_classes': f1.mean(axis=-1),
              'macro_f1_present_classes': f1[..., present].mean(axis=-1),
              'weighted_f1': ratio((f1 * support).sum(axis=-1), total),
              'balanced_accuracy': np.nanmean(recall[..., present], axis=-1),
              'accuracy': ratio(diag.sum(axis=-1), total)}
    per_class = {'precision': precision, 'recall': recall, 'specificity': specificity, 'f1': f1}
    return scalar, per_class


def binary_metrics(conf):
    tn, fp, fn, tp = (conf[..., 0, 0], conf[..., 0, 1], conf[..., 1, 0], conf[..., 1, 1])
    return {'binary_sensitivity': ratio(tp, tp + fn), 'binary_specificity': ratio(tn, tn + fp),
            'binary_ppv': ratio(tp, tp + fp), 'binary_npv': ratio(tn, tn + fn),
            'binary_f1': ratio(2 * tp, 2 * tp + fp + fn),
            'binary_accuracy': ratio(tn + tp, tn + fp + fn + tp)}


def point(y, prob):
    matrix = confusion_matrix(y, prob.argmax(axis=1), labels=range(4))
    scalar, per_class = confusion_metrics(matrix, np.unique(y))
    by = np.isin(y, (1, 3)).astype(int)
    score = prob[:, 1] + prob[:, 3]
    bc = confusion_matrix(by, score >= .5, labels=(0, 1))
    scalar.update(binary_metrics(bc))
    auc, ap = weighted_auc_ap(by, score, np.ones((1, len(y))))
    scalar.update(binary_roc_auc=auc[0], binary_average_precision=ap[0],
                  binary_tn=bc[0, 0], binary_fp=bc[0, 1], binary_fn=bc[1, 0], binary_tp=bc[1, 1])
    aucs = []
    rows = []
    for i, label in enumerate(CLASSES):
        auc, ap = weighted_auc_ap((y == i).astype(int), prob[:, i], np.ones((1, len(y))))
        aucs.append(auc[0])
        rows.append({'class': label, 'support': int(matrix[i].sum()),
                     **{k: float(v[i]) for k, v in per_class.items()},
                     'roc_auc_ovr': float(auc[0]), 'average_precision_ovr': float(ap[0])
                     if 0 < np.count_nonzero(y == i) < len(y) else np.nan})
    scalar['ovr_roc_auc_macro_defined_classes'] = ratio(np.nansum(aucs), np.isfinite(aucs).sum())
    scalar['top2_accuracy'] = np.mean(np.any(np.argsort(prob, axis=1)[:, -2:] == y[:, None], axis=1))
    return {k: float(v) for k, v in scalar.items()}, matrix, bc, rows


def load_group(plan, ph, excluded, kind):
    filename = f'multisource_holdout_{excluded}.csv' if kind == 'holdout' else f'multisource_test_no_{excluded}.csv'
    csv = VIEW / filename
    freeze = json.loads((VIEW / 'view.json').read_text())['derived_csv_sha256']
    require(sha256(csv) == freeze[filename], f'Frozen evaluation manifest changed: {filename}')
    ref = pd.read_csv(csv).fillna({'patient_id': ''})
    runs, provenance = defaultdict(dict), []
    for task in plan['tasks']:
        if task['source'] != excluded:
            continue
        study = name(task)
        d, out = RESULTS / study, PREDICTIONS / study
        m = json.loads((d / 'seedwise_completion.json').read_text())
        s = states(d)
        require(m['plan_sha256'] == ph and m['finished_trials'] == 50 and m['retrain_epochs'] == 50
                and m['test_csv'] is None and s.get('COMPLETE', 0) + s.get('PRUNED', 0) == 50
                and not s.get('RUNNING', 0) and m['model'] == task['model']
                and m['seed'] == task['seed'] and m['excluded_source'] == excluded,
                f'Invalid completed HPO study: {study}')
        path = out / f'{kind}_predictions.csv'
        r = json.loads((out / f'{kind}_provenance.json').read_text())
        require(r['task'] == task and r['kind'] == kind and r['plan_sha256'] == ph
                and r['checkpoint_sha256'] == m['retrain_checkpoint_sha256']
                and r['manifest_sha256'] == freeze[filename]
                and r['predictions_sha256'] == sha256(path) and r['test_evaluated_after_selection'],
                f'Prediction provenance mismatch: {study}/{kind}')
        data = pd.read_csv(path).fillna({'patient_id': ''})
        require(len(data) == len(ref) == r['rows'], f'Missing prediction rows: {study}')
        for col in ('filename', 'diagnosis_categories', 'dataset_source', 'patient_id', 'source_image_path'):
            require(data[col].equals(ref[col]), f'Prediction cases/labels/provenance differ: {study}/{col}')
        prob = data[[f'p_{c}' for c in CLASSES]].to_numpy(dtype=float)
        require(np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all()
                and np.allclose(prob.sum(axis=1), 1, atol=2e-6), 'Invalid probabilities')
        require(np.array_equal(data.y_true, ref.diagnosis_categories)
                and np.array_equal(data.y_pred, np.asarray(CLASSES)[prob.argmax(axis=1)]),
                'Predictions do not match probabilities or truth')
        runs[task['model']][task['seed']] = prob
        log = pd.read_csv(d / 'retrain_log.csv')
        require(len(log) == 50, 'Final retrain did not contain 50 epochs')
        best = log.iloc[log.val_f1.argmax()]
        provenance.append({'excluded_source': excluded, 'kind': kind, 'model': task['model'],
                           'seed': task['seed'], 'selected_epoch': int(best.epoch),
                           'validation_macro_f1': float(best.val_f1),
                           'validation_weighted_f1': float(best.val_f1_weighted), 'study': study, **r})
    require(set(runs) == set(MODELS) and all(set(x) == set(SEEDS) for x in runs.values()),
            'Expected four architectures and five seeds in each endpoint')
    return ref, runs, provenance


def bootstrap(y, prob, cluster_idx, counts, present):
    counts = np.asarray(counts, dtype=np.float32)
    conf = confusion_from_clusters(y, prob.argmax(axis=1), cluster_idx, counts, 4)
    scalar, pc = confusion_metrics(conf, present)
    by, score = np.isin(y, (1, 3)).astype(int), prob[:, 1] + prob[:, 3]
    bc = confusion_from_clusters(by, (score >= .5).astype(int), cluster_idx, counts, 2)
    scalar.update(binary_metrics(bc))
    aucs, aps, ovr = [], [], []
    for start in range(0, len(counts), 500):
        weights = counts[start:start + 500, cluster_idx]
        auc, ap = weighted_auc_ap(by, score, weights)
        aucs.append(auc); aps.append(ap)
        # Class AUCs are undefined if the reference has no positive or negative cases.
        columns = []
        for i in range(4):
            if 0 < np.count_nonzero(y == i) < len(y):
                a, _ = weighted_auc_ap((y == i).astype(int), prob[:, i], weights)
            else:
                a = np.full(len(weights), np.nan)
            columns.append(a)
        ovr.append(np.column_stack(columns))
    scalar['binary_roc_auc'] = np.concatenate(aucs)
    scalar['binary_average_precision'] = np.concatenate(aps)
    pc['roc_auc_ovr'] = np.concatenate(ovr)
    scalar['ovr_roc_auc_macro_defined_classes'] = ratio(
        np.nansum(pc['roc_auc_ovr'], axis=1), np.isfinite(pc['roc_auc_ovr']).sum(axis=1))
    return scalar, pc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--resamples', type=int, default=10000)
    args = parser.parse_args()
    require(args.resamples > 0, 'Positive resample count required')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    ph = sha256(PLAN)
    plan = json.loads(PLAN.read_text())
    require((PREDICTIONS / 'all-evaluated').read_text().strip() == ph, 'All views must finish evaluation')
    require(sum((RESULTS / name(t) / 'seedwise_completion.json').exists() for t in plan['tasks']) == 60,
            'All 60 studies must complete before heldout analysis')
    rows, classes, matrices, binary, curves, prs, cis, class_cis = ([] for _ in range(8))
    comparisons, provenance, clusters, endpoints = [], [], [], []
    draw_archive = {}
    for excluded in SOURCES:
        for kind in ('holdout', 'in_domain'):
            ref, runs, prov = load_group(plan, ph, excluded, kind)
            provenance.extend(prov)
            y = pd.Categorical(ref.diagnosis_categories, categories=CLASSES).codes
            require(np.all(y >= 0), 'Unknown evaluation class')
            cluster_idx, strata, audit = cluster_map(ref.reset_index(drop=True))
            counts = bootstrap_counts({k: v for k, v in strata.items() if v}, args.resamples, 20261004)
            clusters.extend({'excluded_source': excluded, 'kind': kind, **r} for r in audit)
            endpoints.append({'excluded_source': excluded, 'kind': kind, 'n_images': len(y),
                              'class_counts': ref.diagnosis_categories.value_counts().to_dict(),
                              'clusters': {k: len(v) for k, v in strata.items()}})
            mean_draws = {}
            for model in MODELS:
                all_scalar, all_class = [], []
                for seed in SEEDS:
                    prob = runs[model][seed]
                    tags = {'excluded_source': excluded, 'kind': kind, 'model': model, 'seed': seed}
                    for source in ('ALL', *sorted(ref.dataset_source.unique())):
                        mask = np.ones(len(y), dtype=bool) if source == 'ALL' else ref.dataset_source.eq(source).to_numpy()
                        pts, matrix, bc, pc = point(y[mask], prob[mask])
                        tags_source = {**tags, 'evaluation_source': source}
                        rows.append({**tags_source, 'n_images': int(mask.sum()), **pts})
                        classes.extend({**tags_source, **r} for r in pc)
                        for i, truth in enumerate(CLASSES):
                            for j, pred in enumerate(CLASSES):
                                matrices.append({**tags_source, 'true': truth, 'predicted': pred,
                                                 'count': int(matrix[i, j]),
                                                 'row_fraction': float(ratio(matrix[i, j], matrix[i].sum()))})
                        for i, truth in enumerate(('non_suspicious', 'suspicious')):
                            for j, pred in enumerate(('non_suspicious', 'suspicious')):
                                binary.append({**tags_source, 'true': truth, 'predicted': pred,
                                               'count': int(bc[i, j]), 'row_fraction': float(ratio(bc[i, j], bc[i].sum()))})
                    for label, truth, score in [('screening', np.isin(y, (1, 3)), prob[:, 1] + prob[:, 3]),
                                                *[(c, y == i, prob[:, i]) for i, c in enumerate(CLASSES)]]:
                        if len(np.unique(truth)) != 2:
                            continue
                        fpr, tpr, thr = roc_curve(truth, score)
                        curves.extend({**tags, 'class': label, 'fpr': float(fpr[i]), 'tpr': float(tpr[i]),
                                       'threshold': float(thr[i])} for i in range(len(fpr)))
                        p, r, thr = precision_recall_curve(truth, score)
                        prs.extend({**tags, 'class': label, 'precision': float(p[i]), 'recall': float(r[i]),
                                    'threshold': float(thr[i]) if i < len(thr) else np.nan} for i in range(len(p)))
                    scalar, pc_draws = bootstrap(y, prob, cluster_idx, counts, np.unique(y))
                    all_scalar.append(scalar); all_class.append(pc_draws)
                    for metric, draws in scalar.items():
                        low, high, valid = interval(draws)
                        cis.append({**tags, 'metric': metric, 'ci_low': low, 'ci_high': high, 'valid_draws': valid})
                    for metric, draws in pc_draws.items():
                        for i, label in enumerate(CLASSES):
                            low, high, valid = interval(draws[:, i])
                            class_cis.append({**tags, 'class': label, 'metric': metric,
                                              'ci_low': low, 'ci_high': high, 'valid_draws': valid})
                mean_draws[model] = {k: np.mean([r[k] for r in all_scalar], axis=0) for k in all_scalar[0]}
                for metric, draws in mean_draws[model].items():
                    low, high, valid = interval(draws)
                    cis.append({'excluded_source': excluded, 'kind': kind, 'model': model,
                                'seed': 'mean', 'metric': metric, 'ci_low': low, 'ci_high': high, 'valid_draws': valid})
                    draw_archive[f'{excluded}_{kind}_{model}_{metric}'] = draws
                for metric in all_class[0]:
                    draws = np.mean([r[metric] for r in all_class], axis=0)
                    for i, label in enumerate(CLASSES):
                        low, high, valid = interval(draws[:, i])
                        class_cis.append({'excluded_source': excluded, 'kind': kind, 'model': model,
                                          'seed': 'mean', 'class': label, 'metric': metric,
                                          'ci_low': low, 'ci_high': high, 'valid_draws': valid})
                print(f'Analyzed {excluded}/{kind}/{model}, five seeds, n={len(y)}', flush=True)
            for a, b in COMPARISONS:
                for metric in ('macro_f1_present_classes', 'macro_f1_four_classes', 'weighted_f1', 'binary_roc_auc'):
                    draws = mean_draws[a][metric] - mean_draws[b][metric]
                    finite = draws[np.isfinite(draws)]
                    low, high, valid = interval(draws)
                    p = min(1., 2 * min((np.count_nonzero(finite <= 0) + 1) / (len(finite) + 1),
                                       (np.count_nonzero(finite >= 0) + 1) / (len(finite) + 1))) if valid else np.nan
                    comparisons.append({'excluded_source': excluded, 'kind': kind, 'model_a': a, 'model_b': b,
                                        'metric': metric, 'ci_low': low, 'ci_high': high,
                                        'valid_draws': valid, 'bootstrap_tail_p': p})
    frame = pd.DataFrame(rows)
    summary = []
    keys = ['excluded_source', 'kind', 'model', 'evaluation_source']
    metrics = [c for c in frame if c not in (*keys, 'seed', 'n_images')]
    for tags, group in frame.groupby(keys, sort=False):
        row = dict(zip(keys, tags)); row.update(n_seeds=len(group), n_images=int(group.n_images.iloc[0]))
        require(len(group) == 5, 'Seed summary must contain five runs')
        for metric in metrics:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_sd'] = group[metric].std(ddof=1)
        summary.append(row)
    comp = pd.DataFrame(comparisons)
    comp['holm_p'] = np.nan
    for (kind, metric), group in comp.groupby(['kind', 'metric']):
        finite = group[group.bootstrap_tail_p.notna()]
        comp.loc[finite.index, 'holm_p'] = holm(finite.bootstrap_tail_p.tolist())
        # Correct across all 18 architecture comparisons (six in each of three exclusions).
    means = pd.DataFrame(summary)
    for idx, r in comp.iterrows():
        a = means[(means.excluded_source == r.excluded_source) & (means.kind == r.kind)
                  & (means.evaluation_source == 'ALL') & (means.model == r.model_a)].iloc[0]
        b = means[(means.excluded_source == r.excluded_source) & (means.kind == r.kind)
                  & (means.evaluation_source == 'ALL') & (means.model == r.model_b)].iloc[0]
        comp.loc[idx, 'seed_mean_difference'] = a[r.metric + '_mean'] - b[r.metric + '_mean']
    for filename, data in [('per_seed_metrics', rows), ('model_summary', summary), ('per_class_metrics', classes),
                           ('confusion_matrices', matrices), ('binary_confusion_matrices', binary),
                           ('roc_curves', curves), ('pr_curves', prs), ('bootstrap_intervals', cis),
                           ('per_class_bootstrap_intervals', class_cis), ('cluster_audit', clusters)]:
        pd.DataFrame(data).to_csv(out / f'{filename}.csv', index=False)
    comp.to_csv(out / 'paired_comparisons.csv', index=False)
    pd.DataFrame(provenance).drop(columns=['task']).to_csv(out / 'run_provenance.csv', index=False)
    np.savez_compressed(out / 'paired_seed_mean_bootstrap_draws.npz', **draw_archive)
    pc = pd.DataFrame(classes)
    pc.groupby(keys + ['class']).agg({c: ['mean', 'std'] for c in
                                    ('precision', 'recall', 'specificity', 'f1', 'roc_auc_ovr')}).to_csv(out / 'per_class_summary.csv')
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': ph,
                'n_studies': 60, 'n_prediction_views': 120, 'seeds': SEEDS, 'endpoints': endpoints,
                'bootstrap_resamples': args.resamples, 'bootstrap_seed': 20261004,
                'bootstrap_design': 'shared paired draws within each endpoint; source-stratified; '
                                    'INTERNAL/Zenodo clustered by patient, Kaggle/Mendeley by image',
                'screening': 'p(cancer)+p(opmd) >= 0.5; no heldout threshold tuning',
                'macro_f1': 'Both four-class and reference-present-class macro-F1 saved; Table 2 uses present classes',
                'missing_classes': 'Recall/AUC undefined when support is absent; undefined values saved as blank/NaN',
                'comparison_multiplicity': 'Holm across 18 comparisons per metric and endpoint kind',
                'limitations': ['Exploratory seed-specific HPO, not the fixed-parameter primary analysis',
                               'CIs condition on five selected checkpoints; seed SD is separate',
                               'Kaggle/Mendeley patient identifiers unavailable; CIs may be anti-conservative',
                               'Source-exclusion in-domain splits differ from pooled split; differences do not isolate augmentation/HPO/weighting effects'],
                'output_sha256': {p.name: sha256(p) for p in out.glob('*') if p.is_file() and p.name != 'analysis_manifest.json'}}
    (out / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('ALL 120 HELDOUT VIEWS ANALYZED', flush=True)


if __name__ == '__main__':
    main()
