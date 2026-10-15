#!/usr/bin/env python3
"""Recalculate revision metrics without changing frozen predictions or old analyses."""
import hashlib
import itertools
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
RESULTS = ROOT / 'results/revision_bpex'
OUTPUT = RESULTS / 'manuscript_comparison_v1'
TABLES = PAPER / 'tables'
FIGURES = PAPER / 'figures/revision'
CLASSES = ('benign_lesion', 'cancer', 'healthy', 'opmd')
ORDER = (2, 0, 3, 1)
MODELS = ('effnet_b3', 'effnet_b7', 'vit_base', 'vit_large')
DISPLAY = dict(zip(MODELS, ('EfficientNet-B3', 'EfficientNet-B7', 'ViT-Base', 'ViT-Large')))
SEEDS = (42, 43, 44, 45, 46)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        np.asarray(values)[order] * np.arange(len(values), 0, -1)))
    return adjusted


def table(caption, label, headers, rows, landscape=False):
    lines = [r'\begingroup', r'\begin{landscape}' if landscape else '', r'\scriptsize',
             r'\setlength{\tabcolsep}{3pt}',
             r'\begin{longtable}{' + 'l' * len(headers) + '}',
             r'\caption{' + caption + r'}\label{' + label + r'}\\',
             r'\toprule', ' & '.join(headers) + r' \\', r'\midrule',
             r'\endfirsthead', r'\toprule', ' & '.join(headers) + r' \\',
             r'\midrule', r'\endhead']
    lines += [' & '.join(map(str, row)) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{longtable}', r'\end{landscape}' if landscape else '', r'\endgroup']
    return '\n'.join(lines) + '\n'


def fmt(mean, sd, scale=1):
    if not np.isfinite(mean):
        return '--'
    return f'${mean * scale:.3f}\\pm{sd * scale:.3f}$' if scale == 1 else f'${mean * scale:.1f}\\pm{sd * scale:.1f}$'


def main():
    for directory in (OUTPUT, TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    inputs = {str(Path(__file__).resolve().relative_to(ROOT)): digest(Path(__file__).resolve())}
    runs = {}
    metrics, per_class, matrices, curves, tests = [], [], [], [], []
    manuscript = PAPER / 'bpex_submission.tex'
    inputs[str(manuscript.relative_to(ROOT))] = digest(manuscript)
    # Read the historical values from the named tables, not from narrative claims.
    old = []
    text = manuscript.read_text()
    for label, endpoint, names in (
        ('tab:test_performance', 'pooled', ('accuracy', 'top2_accuracy', 'top3_accuracy', 'weighted_f1', 'binary_argmax_f1')),
        ('tab:holdout_binary', None, ('accuracy', 'top2_accuracy', 'top3_accuracy', 'binary_argmax_f1')),
    ):
        block = text.split('\\label{' + label + '}')[1].split('\\end{tabular}')[0]
        model = None
        for line in block.splitlines():
            cells = [x.strip() for x in line.split('&')]
            if cells[0] in DISPLAY.values():
                model = next(k for k, v in DISPLAY.items() if v == cells[0])
            if model is None or len(cells) != len(names) + (1 if endpoint else 2):
                continue
            source = endpoint or cells[1].lower()
            vals = [float(re.search(r'\d+\.\d+', c).group()) / 100 for c in cells[-len(names):]]
            old.append({'endpoint': source, 'model': model, **dict(zip(names, vals))})
    assert len(old) == 16 and len({(r['endpoint'], r['model']) for r in old}) == 16
    pd.DataFrame(old).to_csv(OUTPUT / 'original_manuscript_metrics.csv', index=False)

    for endpoint, analysis, predictions in (
        ('pooled', 'tuned_seedwise_test_analysis_v1', 'tuned_seedwise_test_v1'),
        ('holdout', 'sourcewise_heldout_analysis_v1', 'sourcewise_heldout_v1'),
    ):
        directory = RESULTS / analysis
        for filename in ('run_provenance.csv', 'per_seed_metrics.csv'):
            inputs[str((directory / filename).relative_to(ROOT))] = digest(directory / filename)
        provenance = pd.read_csv(directory / 'run_provenance.csv')
        previous = pd.read_csv(directory / 'per_seed_metrics.csv')
        for _, run in provenance.iterrows():
            source = 'pooled' if endpoint == 'pooled' else (
                run.excluded_source if run.kind == 'holdout' else 'in_domain_no_' + run.excluded_source)
            model, seed = run.model, int(run.seed)
            study = run.source_study if endpoint == 'pooled' else run.study
            path = RESULTS / predictions / study / ('test_predictions.csv' if endpoint == 'pooled' else run.kind + '_predictions.csv')
            assert digest(path) == run.predictions_sha256, path
            inputs[str(path.relative_to(ROOT))] = digest(path)
            data = pd.read_csv(path).fillna({'patient_id': ''})
            prob = data[['p_' + c for c in CLASSES]].to_numpy(float)
            y = np.array([CLASSES.index(c) for c in data.y_true])
            pred = prob.argmax(axis=1)
            assert np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all()
            assert np.allclose(prob.sum(axis=1), 1, atol=2e-6)
            assert np.array_equal(data.y_pred, np.asarray(CLASSES)[pred])
            assert np.array_equal(data.y_true, data.diagnosis_categories)
            key = dict(endpoint=source, model=model, seed=seed)
            if (source, model) in runs:
                ref = runs[(source, model)][SEEDS[0]][0]
                assert data[['filename', 'y_true']].equals(ref[['filename', 'y_true']])
            runs.setdefault((source, model), {})[seed] = (data, y, pred, prob)
            cm = confusion_matrix(y, pred, labels=range(4))
            support = cm.sum(axis=1)
            present = np.flatnonzero(support)
            by, score = np.isin(y, (1, 3)).astype(int), prob[:, 1] + prob[:, 3]
            bp, ba = (score >= .5).astype(int), np.isin(pred, (1, 3)).astype(int)
            bc = confusion_matrix(by, bp, labels=(0, 1))
            tn, fp, fn, tp = bc.ravel()
            def ratio(a, b):
                return a / b if b else np.nan
            values = {
                'n_images': len(y), 'accuracy': np.mean(y == pred),
                'macro_f1_four_classes': f1_score(y, pred, labels=range(4), average='macro', zero_division=0),
                'macro_f1_present_classes': f1_score(y, pred, labels=present, average='macro', zero_division=0),
                'weighted_f1': f1_score(y, pred, labels=range(4), average='weighted', zero_division=0),
                'balanced_accuracy': np.mean(cm.diagonal()[present] / support[present]),
                'binary_f1': f1_score(by, bp, zero_division=0),
                'binary_argmax_f1': f1_score(by, ba, zero_division=0),
                'binary_rule_disagreement': np.mean(bp != ba),
                'binary_accuracy': np.mean(by == bp),
                'binary_sensitivity': ratio(tp, tp + fn), 'binary_specificity': ratio(tn, tn + fp),
                'binary_ppv': ratio(tp, tp + fp), 'binary_npv': ratio(tn, tn + fn),
                'binary_roc_auc': roc_auc_score(by, score),
                'binary_average_precision': average_precision_score(by, score),
            }
            for k in (2, 3):
                values[f'top{k}_accuracy'] = np.mean(np.any(np.argsort(prob, axis=1)[:, -k:] == y[:, None], axis=1))
            prior = previous[(previous.model == model) & (previous.seed == seed)]
            if endpoint == 'holdout':
                prior = prior[(prior.excluded_source == run.excluded_source) & (prior.kind == run.kind) & (prior.evaluation_source == 'ALL')]
            assert len(prior) == 1
            for name, value in values.items():
                alias = 'macro_f1' if endpoint == 'pooled' and name == 'macro_f1_four_classes' else name
                if alias in prior.columns:
                    assert np.isclose(value, prior.iloc[0][alias], atol=1e-10), (key, name)
            metrics.append({**key, **values})
            for i, c in enumerate(CLASSES):
                diag, supp, predicted = cm[i, i], support[i], cm[:, i].sum()
                per_class.append({**key, 'class': c, 'support': supp,
                    'precision': diag / predicted if predicted else 0,
                    'recall': ratio(diag, supp), 'specificity': ratio(len(y) - supp - predicted + diag, len(y) - supp),
                    'f1': 2 * diag / (supp + predicted) if supp + predicted else 0,
                    'roc_auc': roc_auc_score(y == i, prob[:, i]) if 0 < supp < len(y) else np.nan})
            for kind, matrix in (('multiclass', cm), ('screening', bc)):
                for i, j in itertools.product(range(len(matrix)), repeat=2):
                    matrices.append({**key, 'kind': kind, 'true': i, 'predicted': j,
                                     'count': matrix[i, j], 'row_fraction': ratio(matrix[i, j], matrix[i].sum())})
            for task, truth, scores in [('screening', by, score)] + [(c, y == i, prob[:, i]) for i, c in enumerate(CLASSES) if 0 < support[i] < len(y)]:
                for kind in ('roc', 'pr'):
                    if kind == 'roc':
                        x, z, _ = roc_curve(truth, scores)
                    else:
                        z, x, _ = precision_recall_curve(truth, scores)
                    curves.extend({**key, 'task': task, 'curve': kind, 'x': a, 'y': b} for a, b in zip(x, z))
    assert len(metrics) == 140 and all(set(v) == set(SEEDS) for v in runs.values())
    frame = pd.DataFrame(metrics)
    pc = pd.DataFrame(per_class)
    cm_frame = pd.DataFrame(matrices)
    curve_frame = pd.DataFrame(curves)
    for name, df in [('per_seed_metrics', frame), ('per_class_metrics', pc), ('confusion_matrices', cm_frame), ('curves', curve_frame)]:
        df.to_csv(OUTPUT / (name + '.csv'), index=False)
    summary = frame.groupby(['endpoint', 'model']).agg(['mean', 'std'])
    summary.to_csv(OUTPUT / 'model_summary.csv')
    def cell(source, model, metric, scale=1):
        return fmt(*summary.loc[(source, model), metric], scale)

    source_rows = []
    for model, seed in itertools.product(MODELS, SEEDS):
        data, y, pred, prob = runs[('pooled', model)][seed]
        sources = data.dataset_source.replace({'MY_CAMERA': 'INTERNAL', 'MY_ROMEU': 'INTERNAL'})
        for source in ('INTERNAL', 'ZENODO', 'KAGGLE', 'MENDELEY'):
            mask = sources.to_numpy() == source
            sy, sp, sq = y[mask], pred[mask], prob[mask]
            truth, score = np.isin(sy, (1, 3)), sq[:, 1] + sq[:, 3]
            tn, fp, fn, tp = confusion_matrix(truth, score >= .5, labels=(0, 1)).ravel()
            def safe(a, b):
                return a / b if b else np.nan
            source_rows.append({'source': source, 'model': model, 'seed': seed, 'n_images': len(sy),
                'accuracy': np.mean(sy == sp), 'weighted_f1': f1_score(sy, sp, average='weighted', zero_division=0),
                'macro_f1_present_classes': f1_score(sy, sp, labels=np.unique(sy), average='macro', zero_division=0),
                'top2_accuracy': np.mean(np.any(np.argsort(sq, axis=1)[:, -2:] == sy[:, None], axis=1)),
                'top3_accuracy': np.mean(np.any(np.argsort(sq, axis=1)[:, -3:] == sy[:, None], axis=1)),
                'binary_f1': f1_score(truth, score >= .5, zero_division=0) if truth.any() else np.nan,
                'binary_sensitivity': safe(tp, tp + fn), 'binary_specificity': safe(tn, tn + fp),
                'binary_roc_auc': roc_auc_score(truth, score) if len(np.unique(truth)) == 2 else np.nan})
    source_frame = pd.DataFrame(source_rows)
    source_frame.to_csv(OUTPUT / 'pooled_source_per_seed_metrics.csv', index=False)
    source_summary = source_frame.groupby(['source', 'model']).agg(['mean', 'std'])

    comparison = []
    for row in old:
        for metric in ('accuracy', 'top2_accuracy', 'top3_accuracy', 'weighted_f1', 'binary_argmax_f1'):
            if metric in row:
                mean, sd = summary.loc[(row['endpoint'], row['model']), metric]
                comparison.append({'endpoint': row['endpoint'], 'model': row['model'], 'metric': metric,
                                   'original': row[metric], 'corrected_mean': mean, 'corrected_sd': sd,
                                   'change_percentage_points': 100 * (mean - row[metric])})
    comp = pd.DataFrame(comparison)
    comp.to_csv(OUTPUT / 'original_vs_corrected.csv', index=False)

    # Use the corrected full-source mapping to audit the legacy manifests by actual patient.
    frozen = ROOT / 'data/processed/bpex_revision_v2_frozen_2026-09-24'
    mapping_path = frozen / 'multisource_holdout_zenodo.csv'
    inputs[str(mapping_path.relative_to(ROOT))] = digest(mapping_path)
    mapping = pd.read_csv(mapping_path).set_index('filename').patient_id.to_dict()
    assert len(mapping) == 3000 and len(set(mapping.values())) == 714
    audit_rows, audit, composition_rows = [], {}, []
    for version, base in [('Original', ROOT / 'data/processed'), ('Corrected', frozen)]:
        parts = {}
        for split, suffix in [('Train', 'train'), ('Validation', 'val'), ('Test', 'test')]:
            path = base / f'multisource_{suffix}_global.csv'
            inputs[str(path.relative_to(ROOT))] = digest(path)
            data = pd.read_csv(path)
            zen = data[data.dataset_source == 'ZENODO']
            assert all(f in mapping for f in zen.filename)
            parts[split] = set(zen.filename.map(mapping))
            audit_rows.append([version, split, len(data), len(zen), len(parts[split])])
            if split == 'Test':
                for source in ('MY_CAMERA', 'MY_ROMEU', 'ZENODO', 'KAGGLE', 'MENDELEY'):
                    counts = data[data.dataset_source == source].diagnosis_categories.value_counts()
                    composition_rows.append([version, source.replace('_', ' '),
                        *[int(counts.get(CLASSES[i], 0)) for i in ORDER], int(counts.sum())])
        audit[version] = {a + '/' + b: len(parts[a] & parts[b]) for a, b in itertools.combinations(parts, 2)}
    assert all(v == 0 for v in audit['Corrected'].values())
    (OUTPUT / 'split_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    audit_tex = table('Pooled split sizes and Zenodo patient groups, using the corrected patient mapping for both versions.',
                      'tab:s_legacy_split_sizes', ['Version', 'Partition', 'All images', 'Zenodo images', 'Zenodo patients'], audit_rows)
    audit_tex += table('Zenodo patients shared between pooled partitions. Zero file overlap does not imply zero patient overlap.',
                       'tab:s_legacy_patient_overlap', ['Version', 'Train/validation', 'Train/test', 'Validation/test'],
                       [[v, *counts.values()] for v, counts in audit.items()])
    audit_tex += table('Pooled test composition by source and class in the original and corrected manifests. Changes in membership and class mix accompany the grouping correction.',
                       'tab:s_test_composition', ['Version', 'Source', 'Healthy', 'Benign', 'OPMD', 'Cancer', 'Total'], composition_rows)
    (TABLES / 'bpex_split_comparison.tex').write_text(audit_tex)

    sections = []
    sections.append(r'\section{Original submission and corrected evaluation}')
    sections.append(r'''The original Zenodo importer treated hyphenated image identifiers as separate
groups. Several photographs of the same patient could therefore enter different
pooled partitions. The corrected mapping assigns 3,000 images to 714 patients.
The audit below applies that mapping retrospectively to the original manifests.
The corrected partitions have no known internal or Zenodo patient overlap.
Kaggle and Mendeley have no reliable patient identifiers and remain file-disjoint.

The historical values are the rounded percentages in Tables 1 and 2 of the
original submission. The corrected results are means and sample SDs for seeds
42--46, each with an independent 50-trial search and 50-epoch retrain selected
by validation macro-F1. Test cases, development partitions and selected
configurations differ between versions. Changes are descriptive; they do not
estimate the isolated effect of repairing leakage. The original submission did
not report macro-F1 in these tables, so no original macro-F1 is reconstructed.

The original evaluator mapped the four-class top-1 prediction to suspicious or
non-suspicious. We retain that rule for the historical comparisons. The expanded
screening evaluation instead uses $p(\mathrm{OPMD})+p(\mathrm{cancer})\geq0.5$.
These decisions can differ and their F1 values are reported separately.''')
    sections.append(r'\input{tables/bpex_split_comparison.tex}')
    for source in ('pooled', 'kaggle', 'zenodo', 'mendeley'):
        rows = []
        for model in MODELS:
            for metric, label in [('accuracy', 'Top-1 accuracy'), ('top2_accuracy', 'Top-2 accuracy'),
                                  ('top3_accuracy', 'Top-3 accuracy'), ('weighted_f1', 'Weighted F1'),
                                  ('binary_argmax_f1', 'F1 suspicious (top-1 collapse)')]:
                match = comp[(comp.endpoint == source) & (comp.model == model) & (comp.metric == metric)]
                if match.empty:
                    continue
                r = match.iloc[0]
                delta = round(r.change_percentage_points, 1)
                rows.append([DISPLAY[model], label, f'{100*r.original:.1f}', cell(source, model, metric, 100), f'{delta:+.1f}' if delta else '0.0'])
        sections.append(table(f'Original versus corrected {source.title()} evaluation. Values are percentages; changes are percentage points. Corrected values are mean $\\pm$ sample SD across five seeds. No paired significance test is implied.',
                              'tab:s_comparison_' + source, ['Model', 'Metric', 'Original', 'Corrected', 'Change'], rows))

    sections.append(r'''Across the revised pooled runs, weighted-F1 decreased by 4.8, 5.5 and
6.0 percentage points for B3, B7 and ViT-Large, respectively, but increased by
11.0 points for ViT-Base. This pattern does not support attributing every score
change to leakage alone. Patient separation removes a route to optimistic
in-domain evaluation, while the changed test composition and independent
hyperparameter searches also affect performance.

Hold-out changes were mixed. Zenodo top-1 accuracy increased for B3, B7 and
ViT-Large, but decreased for ViT-Base. Under the original top-1-collapse rule,
Zenodo suspicious F1 increased for ViT-Base (31.5\% to 47.7\%) and decreased
for B7 and ViT-Large; B3 was nearly unchanged (40.8\% to 40.6\%). Excluding
Zenodo removes its photographs from development entirely, so correcting its
within-source grouping cannot directly repair the no-Zenodo training split.
These hold-outs assess transfer to an unseen source and still show substantial
false-negative rates.''')
    sections.append(r'\section{Repeated runs and expanded metrics}')
    sections.append(r'''All metrics below use the saved probabilities without further fitting or threshold
selection. Multiclass predictions retain all four output classes. Present-class
macro-F1 averages only classes represented in the reference labels; four-class
macro-F1 also includes absent classes. Weighted-F1 weights by reference support,
so absent classes have zero weight. Balanced accuracy averages reference-present
class recalls. Undefined absent-class recall and AUC are shown as ``--''.
Seed means average metrics, not probabilities, and are not ensemble results.''')
    rows = [[source.title(), DISPLAY[m], *[cell(source, m, k) for k in
             ('accuracy', 'top2_accuracy', 'top3_accuracy', 'weighted_f1', 'macro_f1_present_classes',
              'macro_f1_four_classes', 'balanced_accuracy')]]
            for source, m in itertools.product(('pooled', 'kaggle', 'zenodo', 'mendeley'), MODELS)]
    sections.append(table('Corrected pooled and source-holdout multiclass evaluation: mean $\\pm$ sample SD across five seeds. Top-1 is accuracy.',
                          'tab:s_pooled_metrics', ['Endpoint', 'Model', 'Top-1', 'Top-2', 'Top-3', 'W. F1', 'Macro F1 (present)', 'Macro F1 (four)', 'Balanced acc.'], rows, True))
    rows = [[source.title(), DISPLAY[m], *[cell(source, m, k) for k in
             ('binary_accuracy', 'binary_f1', 'binary_sensitivity', 'binary_specificity', 'binary_ppv',
              'binary_npv', 'binary_roc_auc', 'binary_average_precision')]]
            for source, m in itertools.product(('pooled', 'kaggle', 'zenodo', 'mendeley'), MODELS)]
    sections.append(table('Corrected pooled and source-holdout screening evaluation at summed-probability threshold 0.5: mean $\\pm$ sample SD. AP is average precision.',
                          'tab:s_screening_metrics', ['Endpoint', 'Model', 'Accuracy', 'F1 susp.', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'ROC-AUC', 'AP'], rows, True))
    rows = [[source.title(), DISPLAY[model], cell(source, model, 'binary_argmax_f1'),
             cell(source, model, 'binary_f1'), cell(source, model, 'binary_rule_disagreement', 100)]
            for source, model in itertools.product(('pooled', 'kaggle', 'zenodo', 'mendeley'), MODELS)]
    sections.append(table('Binary decision-rule comparison on the same corrected predictions. Disagreement is the percentage of images receiving different binary decisions, averaged across seeds.',
                          'tab:s_binary_rules', ['Endpoint', 'Model', 'F1 (top-1 collapse)', 'F1 (probability sum)', 'Disagreement (\%)'], rows))
    sections.append(r'''Source-specific slices of the pooled test set below contain sources that
were represented during training. They are in-domain diagnostic views, not
excluded-source hold-outs, and must not be used as evidence of improved external
transfer. The Kaggle pooled-test slice contains only healthy reference labels;
suspicious sensitivity, suspicious F1 and screening AUC are not reported there.''')
    rows = []
    for source, model in itertools.product(('INTERNAL', 'ZENODO', 'KAGGLE', 'MENDELEY'), MODELS):
        row = source_summary.loc[(source, model)]
        rows.append([source.title(), DISPLAY[model], int(row['n_images']['mean']),
            *[fmt(*row[metric]) for metric in ('accuracy', 'top2_accuracy', 'top3_accuracy', 'weighted_f1',
              'macro_f1_present_classes', 'binary_f1', 'binary_sensitivity', 'binary_specificity', 'binary_roc_auc')]])
    sections.append(table('Source-specific slices of the corrected pooled test: mean $\\pm$ sample SD. Internal combines camera and ROMEU images; screening uses threshold 0.5.',
                          'tab:s_source_metrics', ['Source', 'Model', '$n$', 'Top-1', 'Top-2', 'Top-3', 'W. F1', 'Macro F1', 'Screen F1', 'Sens.', 'Spec.', 'AUC'], rows, True))
    rows = []
    for source, model in itertools.product(('pooled', 'kaggle', 'zenodo', 'mendeley'), MODELS):
        for metric, label in [('weighted_f1', 'Weighted F1'), ('macro_f1_present_classes', 'Macro F1 (present)'), ('binary_f1', 'Screening F1')]:
            sub = frame[(frame.endpoint == source) & (frame.model == model)].set_index('seed')
            rows.append([source.title(), DISPLAY[model], label, *[f'{sub.loc[s, metric]:.3f}' for s in SEEDS], cell(source, model, metric)])
    sections.append(table('Seed-level multiclass and probability-threshold screening F1. All five seeds are retained.', 'tab:s_seed_metrics',
                          ['Endpoint', 'Model', 'Metric', *['Seed ' + str(s) for s in SEEDS], 'Mean $\pm$ SD'], rows, True))
    rows = []
    for source, model in itertools.product(('pooled', 'kaggle', 'zenodo', 'mendeley'), MODELS):
        for i in ORDER:
            sub = pc[(pc.endpoint == source) & (pc.model == model) & (pc['class'] == CLASSES[i])]
            rows.append([source.title(), DISPLAY[model], CLASSES[i].replace('_', ' '), int(sub.support.iloc[0]),
                         *[fmt(sub[k].mean(), sub[k].std()) for k in ('precision', 'recall', 'specificity', 'f1', 'roc_auc')]])
    sections.append(table('Per-class metrics across five seeds. Recall is sensitivity; AUC is one-versus-rest. Support is the number of unique reference images, not five times that number.',
                          'tab:s_per_class_metrics', ['Endpoint', 'Model', 'Class', 'Support', 'Precision', 'Recall', 'Specificity', 'F1', 'AUC'], rows, True))

    sections.append(r'\subsection{Source-exclusion in-domain endpoints}')
    sections.append(r'''The same source-exclusion checkpoints were evaluated on their separate
in-domain test sets: 952 images without Kaggle, 563 without Zenodo, and 1,145
without Mendeley. These sets differ from the corrected 1,209-image pooled test.
Neither their source composition nor their case membership is matched across
conditions, so between-condition differences are not paired component effects.
The experiments do not isolate the effect of class weighting.''')
    rows = []
    for excluded, model in itertools.product(('kaggle', 'zenodo', 'mendeley'), MODELS):
        source = 'in_domain_no_' + excluded
        rows.append([excluded.title(), DISPLAY[model], *[cell(source, model, metric) for metric in
            ('accuracy', 'top2_accuracy', 'top3_accuracy', 'weighted_f1', 'macro_f1_present_classes', 'binary_f1', 'binary_roc_auc')]])
    sections.append(table('Separate in-domain test results after source exclusion: mean $\\pm$ sample SD across five seeds. Screening F1 uses threshold 0.5.',
                          'tab:s_source_ablation', ['Excluded', 'Model', 'Top-1', 'Top-2', 'Top-3', 'W. F1', 'Macro F1', 'Screen F1', 'Screen AUC'], rows, True))

    sections.append(r'\section{Test-sample uncertainty and paired comparisons}')
    sections.append(r'''The existing 10,000-draw paired bootstrap analyses use source-stratified
resampling: internal and Zenodo photographs move together by patient; Kaggle
and Mendeley are resampled by image. Intervals condition on the five selected
checkpoints and do not capture uncertainty from other training seeds. They are
distinct from the sample SD across seeds. Image-level resampling may underestimate
uncertainty when undocumented patients have multiple photographs.''')
    rows = []
    for source in ('pooled', 'kaggle', 'zenodo', 'mendeley'):
        directory = RESULTS / ('tuned_seedwise_test_analysis_v1' if source == 'pooled' else 'sourcewise_heldout_analysis_v1')
        path = directory / 'bootstrap_intervals.csv'
        inputs[str(path.relative_to(ROOT))] = digest(path)
        ci = pd.read_csv(path)
        if source != 'pooled':
            ci = ci[(ci.excluded_source == source) & (ci.kind == 'holdout') & (ci.seed == 'mean')]
        for model in MODELS:
            for metric in ('weighted_f1', 'binary_sensitivity', 'binary_specificity', 'binary_roc_auc'):
                r = ci[(ci.model == model) & (ci.metric == metric)].iloc[0]
                rows.append([source.title(), DISPLAY[model], metric.replace('_', ' '), cell(source, model, metric), f'$[{r.ci_low:.3f},{r.ci_high:.3f}]$'])
    sections.append(table('Seed variability and conditional patient-aware 95\% bootstrap intervals.', 'tab:s_bootstrap_intervals',
                          ['Endpoint', 'Model', 'Metric', 'Mean $\pm$ SD', '95\% CI'], rows))
    path = RESULTS / 'tuned_seedwise_test_analysis_v1/paired_comparisons.csv'
    paired = pd.read_csv(path)
    inputs[str(path.relative_to(ROOT))] = digest(path)
    rows = [[DISPLAY[r.model_a], DISPLAY[r.model_b], f'{r.macro_f1_difference:+.3f}',
             f'$[{r.macro_f1_ci_low:.3f},{r.macro_f1_ci_high:.3f}]$', f'{r.macro_f1_bootstrap_tail_p:.4g}', f'{r.macro_f1_holm_p:.4g}'] for r in paired.itertuples()]
    sections.append(table('Paired seed-mean pooled macro-F1 comparisons. Differences are A minus B; Holm correction covers six architecture pairs.', 'tab:s_pairwise_tests',
                          ['Model A', 'Model B', 'Difference', '95\% CI', 'Raw $p$', 'Holm $p$'], rows))
    for seed in SEEDS:
        for a, b in itertools.combinations(MODELS, 2):
            da, ya, pa, proba = runs[('pooled', a)][seed]
            db, yb, pb, probb = runs[('pooled', b)][seed]
            assert da.filename.equals(db.filename) and np.array_equal(ya, yb)
            for task in ('multiclass', 'screening'):
                ca, cb = pa == ya, pb == yb
                if task == 'screening':
                    truth = np.isin(ya, (1, 3))
                    ca, cb = (proba[:, 1] + proba[:, 3] >= .5) == truth, (probb[:, 1] + probb[:, 3] >= .5) == truth
                ab, ba = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
                tests.append(dict(seed=seed, model_a=a, model_b=b, task=task, a_only=ab, b_only=ba,
                                  p=binomtest(ab, ab + ba, .5).pvalue if ab + ba else 1.))
    mcnemar = pd.DataFrame(tests)
    for task in ('multiclass', 'screening'):
        mask = mcnemar.task == task
        mcnemar.loc[mask, 'holm_p'] = holm(mcnemar.loc[mask, 'p'].to_numpy())
    mcnemar.to_csv(OUTPUT / 'pooled_mcnemar.csv', index=False)
    sections.append(r'''Exact McNemar tests below compare matched correct/incorrect decisions separately
for each seed. Holm correction covers all 30 pair-by-seed tests per outcome.
Photographs within a patient are correlated, violating the independent-pair
assumption of ordinary image-level McNemar tests. These results are exploratory
diagnostics; patient-aware bootstrap comparisons provide the primary inference.
Predictions are never stacked across seeds as if they were independent cases.''')
    rows = [[r.seed, DISPLAY[r.model_a], DISPLAY[r.model_b], r.task, r.a_only, r.b_only, f'{r.p:.4g}', f'{r.holm_p:.4g}'] for r in mcnemar.itertuples()]
    sections.append(table('Exploratory image-level exact McNemar tests on the pooled test set. Discordant counts are A-only and B-only correct decisions.',
                          'tab:s_mcnemar', ['Seed', 'Model A', 'Model B', 'Outcome', 'A only', 'B only', 'Raw $p$', 'Holm $p$'], rows, True))

    sections.append(r'\section{Confusion matrices and probability-derived curves}')
    sections.append(r'''Confusion-matrix cells show the mean count across five seeds and the mean
row-normalized percentage. Mean counts can be fractional; each matrix represents
one evaluation population, not five independent copies. Missing reference classes
have zero counts and undefined row percentages. Rows are true classes and columns
are predictions. Binary matrices use the summed-probability threshold of 0.5.
Seed-specific integer counts, normalized matrices, ROC/PR points and class-wise
one-versus-rest curves are retained in the accompanying CSV analysis bundle.
Curve panels show five separate seeds without confidence bands; scalar AUC
intervals are given above. PR panels include the suspicious-class prevalence baseline.''')
    for source in ('pooled', 'kaggle', 'zenodo', 'mendeley'):
        for kind in ('multiclass', 'screening'):
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
            for ax, model in zip(axes.flat, MODELS):
                sub = cm_frame[(cm_frame.endpoint == source) & (cm_frame.model == model) & (cm_frame.kind == kind)]
                order = ORDER if kind == 'multiclass' else (0, 1)
                labels = ('Healthy', 'Benign', 'OPMD', 'Cancer') if kind == 'multiclass' else ('Non-susp.', 'Suspicious')
                counts = sub.groupby(['true', 'predicted'])['count'].mean().unstack().reindex(index=order, columns=order).to_numpy()
                fractions = sub.groupby(['true', 'predicted']).row_fraction.mean().unstack().reindex(index=order, columns=order).to_numpy()
                ax.imshow(np.nan_to_num(fractions), vmin=0, vmax=1, cmap='Blues')
                for i, j in itertools.product(range(len(order)), repeat=2):
                    value = fractions[i, j]
                    label = f'{counts[i,j]:.1f}\n({100*value:.1f}%)' if np.isfinite(value) else f'{counts[i,j]:.1f}\n(NA)'
                    ax.text(j, i, label, ha='center', va='center', fontsize=8, color='white' if value > .55 else 'black')
                ax.set(xticks=range(len(order)), yticks=range(len(order)), xticklabels=labels, yticklabels=labels,
                       title=DISPLAY[model], xlabel='Predicted', ylabel='True')
            filename = source + '_' + kind + '_confusion.pdf'
            fig.savefig(FIGURES / filename)
            plt.close(fig)
            sections.append(r'\begin{figure}[p]\centering\includegraphics[width=0.94\textwidth]{figures/revision/' + filename + r'}\caption{' + source.title() + ' ' + kind + r' confusion matrices: seed-mean counts and row percentages.}\label{fig:s_cm_' + source + '_' + kind + r'}\end{figure}')
        for kind in ('roc', 'pr'):
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
            for ax, model in zip(axes.flat, MODELS):
                for seed in SEEDS:
                    sub = curve_frame[(curve_frame.endpoint == source) & (curve_frame.model == model) & (curve_frame.seed == seed) & (curve_frame.task == 'screening') & (curve_frame.curve == kind)]
                    metric = 'binary_roc_auc' if kind == 'roc' else 'binary_average_precision'
                    value = frame[(frame.endpoint == source) & (frame.model == model) & (frame.seed == seed)][metric].iloc[0]
                    ax.plot(sub.x, sub.y, label=f'{seed}: {value:.3f}', lw=1)
                data, y, _, _ = runs[(source, model)][42]
                if kind == 'roc':
                    ax.plot([0, 1], [0, 1], ':', color='gray')
                else:
                    ax.axhline(np.mean(np.isin(y, (1, 3))), ls=':', color='gray')
                ax.set(xlim=(0, 1), ylim=(0, 1), title=DISPLAY[model], xlabel='False-positive rate' if kind == 'roc' else 'Recall',
                       ylabel='Sensitivity' if kind == 'roc' else 'Precision')
                ax.legend(fontsize=8)
            filename = source + '_' + kind + '.pdf'
            fig.savefig(FIGURES / filename)
            plt.close(fig)
            sections.append(r'\begin{figure}[p]\centering\includegraphics[width=0.94\textwidth]{figures/revision/' + filename + r'}\caption{' + source.title() + ' screening ' + kind.upper() + r' curves by seed. Legend values are ' + ('ROC-AUC' if kind == 'roc' else 'average precision') + r'.}\end{figure}')
    sections.append(r'\clearpage')
    (TABLES / 'bpex_revision_evaluation.tex').write_text('\n\n'.join(sections) + '\n')
    manifest = {'input_sha256': inputs, 'seeds': SEEDS, 'prediction_files': len(metrics),
                'confusion_csv_class_order': CLASSES,
                'confusion_figure_class_order': [CLASSES[i] for i in ORDER],
                'binary_class_order': ['non_suspicious', 'suspicious'],
                'definitions': {'historical_binary': 'collapse four-class argmax', 'screening_binary': 'p_cancer + p_opmd >= 0.5'},
                'output_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (
                    list(OUTPUT.glob('*.csv')) + [OUTPUT / 'split_audit.json', TABLES / 'bpex_revision_evaluation.tex',
                    TABLES / 'bpex_split_comparison.tex'] + list(FIGURES.glob('*.pdf')))}}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Verified and recalculated 20 pooled, 60 source-holdout and 60 in-domain views; generated comparison tables and 16 figure PDFs.')
    print('Zenodo patient overlap:', audit)
    print(comp[comp.metric.isin(['accuracy', 'weighted_f1', 'binary_argmax_f1'])].to_string(index=False))


if __name__ == '__main__':
    main()
