#!/usr/bin/env python3
"""Audit final retrains and assess raw probabilities without fitting a calibrator."""
import hashlib
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import yaml

from build_revision_evidence import CLASSES, DISPLAY, MODELS, PAPER, RESULTS, ROOT, SEEDS, fmt, table

OUTPUT = RESULTS / 'calibration_diagnostics_v1'
FIGURES = PAPER / 'figures/revision'
FROZEN = ROOT / 'data/processed/bpex_revision_v2_frozen_2026-09-24'
VIEW = ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24'
CONFIGS = dict(zip(MODELS, ('effnet_imagenet_b3_ar_multisource.yaml', 'effnet_imagenet_ar.yaml',
                          'vit_base_imagenet_ar.yaml', 'vit_large_imagenet_ar.yaml')))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def reliability(scores, outcomes, bins):
    index = np.minimum((scores * bins).astype(int), bins - 1)
    count = np.bincount(index, minlength=bins)
    predicted = np.divide(np.bincount(index, weights=scores, minlength=bins), count,
                          out=np.full(bins, np.nan), where=count > 0)
    observed = np.divide(np.bincount(index, weights=outcomes, minlength=bins), count,
                         out=np.full(bins, np.nan), where=count > 0)
    ece = np.nansum(count / len(scores) * np.abs(predicted - observed))
    return float(ece), count, predicted, observed


def main():
    OUTPUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    inputs = {}
    def record(path):
        value = sha(path)
        inputs[str(path.relative_to(ROOT))] = value
        return value
    record(Path(__file__).resolve())
    record(PAPER / 'bpex_submission.tex')
    freeze_path = ROOT / 'configs/studies/bpex_revision_v2_frozen_2026-09-24.json'
    pooled_plan_path = ROOT / 'configs/studies/bpex_revision_v2_seedwise_hpo_2026-09-26.json'
    source_plan_path = ROOT / 'configs/studies/bpex_sourcewise_hpo_2026-09-30.json'
    for path in (freeze_path, pooled_plan_path, source_plan_path, VIEW / 'view.json'):
        record(path)
    frozen_hashes = json.loads(freeze_path.read_text())['csv_sha256']
    derived_hashes = json.loads((VIEW / 'view.json').read_text())['derived_csv_sha256']
    pooled_plan = json.loads(pooled_plan_path.read_text())
    source_plan = json.loads(source_plan_path.read_text())
    for model, filename in CONFIGS.items():
        path = ROOT / 'configs/studies' / filename
        assert record(path) == pooled_plan['original_yaml_sha256'][model] == source_plan['original_yaml_sha256'][model], path
        cfg = yaml.safe_load(path.read_text())
        assert cfg['study']['class_weights'] and not cfg['study']['weighted_sampler']
        assert not cfg['study'].get('normalize_sample_weights', False)
    for path in (ROOT / 'scripts/run_optuna_study.py', ROOT / 'src/oral_lesions/engine/trainer.py',
                 ROOT / 'src/oral_lesions/data/datasets.py', ROOT / 'scripts/data/build_datasets_after_clustering.py'):
        record(path)
    reference_columns = ['filename', 'dataset_source', 'patient_id', 'diagnosis_categories', 'sample_weight']
    splits = {}
    weight_rows = []
    for endpoint in ('pooled', 'kaggle', 'zenodo', 'mendeley'):
        train_file = 'multisource_train_global.csv' if endpoint == 'pooled' else f'multisource_train_no_{endpoint}.csv'
        val_file = 'multisource_val_global.csv' if endpoint == 'pooled' else f'multisource_val_no_{endpoint}.csv'
        eval_file = 'multisource_test_global.csv' if endpoint == 'pooled' else f'multisource_holdout_{endpoint}.csv'
        for filename in (train_file, val_file, eval_file):
            assert record(FROZEN / filename) == frozen_hashes[filename]
            assert record(VIEW / filename) == derived_hashes[filename]
            frozen = pd.read_csv(FROZEN / filename).fillna({'patient_id': ''})
            derived = pd.read_csv(VIEW / filename).fillna({'patient_id': ''})
            assert frozen[reference_columns].equals(derived[reference_columns]), filename
            if 'group_all' in frozen:
                assert frozen.group_all.equals(derived.group_all), filename
            splits[filename] = derived
        tr = splits[train_file]
        counts = tr.diagnosis_categories.value_counts().reindex(CLASSES).to_numpy()
        a = (1 / counts) / np.sum(1 / counts)
        source_weight = np.where(tr.dataset_source.eq('KAGGLE'), .5, 1.)
        risk_weight = tr.diagnosis_categories.map({'healthy': 1., 'benign_lesion': 1., 'opmd': 1.5, 'cancer': 2.})
        assert np.allclose(tr.sample_weight, source_weight * risk_weight)
        weight_rows.extend({'endpoint': endpoint, 'class': c, 'n_train': int(n), 'class_weight': float(w)}
                           for c, n, w in zip(CLASSES, counts, a))
    pd.DataFrame(weight_rows).to_csv(OUTPUT / 'training_class_weights.csv', index=False)
    weight_table = []
    for row in weight_rows:
        risk = {'healthy': 1., 'benign_lesion': 1., 'opmd': 1.5, 'cancer': 2.}[row['class']]
        weight_table.append(['Pooled' if row['endpoint'] == 'pooled' else 'No ' + row['endpoint'].title(),
            row['class'].replace('_', ' '), row['n_train'], f"{row['class_weight']:.6f}", f'{risk:.1f}'])
    weight_path = PAPER / 'tables/bpex_training_class_weights.tex'
    weight_path.write_text(table('Verified training counts and normalized inverse-frequency class weights for the corrected frozen training splits. The additional class multiplier is part of the sample weight, separate from the inverse-frequency weight. All four architectures and five seeds use the same counts within a split.',
        'tab:s_class_weight_values', ['Training split', 'Class', '$n_c$', '$a_c$', 'Additional multiplier $r_c$'], weight_table))

    audit_rows, metric_rows, class_rows, bin_rows = [], [], [], []
    previous_class = pd.read_csv(RESULTS / 'manuscript_comparison_v1/per_class_metrics.csv')
    record(RESULTS / 'manuscript_comparison_v1/per_class_metrics.csv')
    for endpoint in ('pooled', 'kaggle', 'zenodo', 'mendeley'):
        pooled = endpoint == 'pooled'
        analysis = RESULTS / ('tuned_seedwise_test_analysis_v1' if pooled else 'sourcewise_heldout_analysis_v1')
        record(analysis / 'run_provenance.csv')
        provenance = pd.read_csv(analysis / 'run_provenance.csv')
        if not pooled:
            provenance = provenance[(provenance.kind == 'holdout') & (provenance.excluded_source == endpoint)]
        assert len(provenance) == 20
        eval_file = 'multisource_test_global.csv' if pooled else f'multisource_holdout_{endpoint}.csv'
        reference = splits[eval_file]
        for run in provenance.sort_values(['model', 'seed']).itertuples():
            study = run.source_study if pooled else run.study
            pred_dir = RESULTS / ('tuned_seedwise_test_v1' if pooled else 'sourcewise_heldout_v1') / study
            pred_file = pred_dir / ('test_predictions.csv' if pooled else 'holdout_predictions.csv')
            receipt_path = pred_dir / ('test_eval_provenance.json' if pooled else 'holdout_provenance.json')
            record(receipt_path)
            receipt = json.loads(receipt_path.read_text())
            if pooled:
                checkpoint = Path(receipt['checkpoint'])
                assert receipt['test_csv_sha256'] == derived_hashes[eval_file] and receipt['test_evaluated_once']
                assert receipt['model'] == run.model and receipt['seed'] == run.seed
                assert checkpoint.parent.name == study
            else:
                checkpoint = RESULTS / 'hpo_sourcewise_corrected_v1' / study / 'best_model_retrain.pth'
                completion_path = checkpoint.parent / 'seedwise_completion.json'
                record(completion_path)
                completed = json.loads(completion_path.read_text())
                assert completed['model'] == run.model and completed['seed'] == run.seed and completed['excluded_source'] == endpoint
                assert completed['finished_trials'] == 50 and completed['retrain_epochs'] == 50 and completed['test_csv'] is None
                assert completed['retrain_checkpoint_sha256'] == receipt['checkpoint_sha256']
                assert completed['plan_sha256'] == receipt['plan_sha256'] == inputs[str(source_plan_path.relative_to(ROOT))]
                assert receipt['manifest_sha256'] == derived_hashes[eval_file] and receipt['test_evaluated_after_selection']
            assert checkpoint.name == 'best_model_retrain.pth' and checkpoint.is_relative_to(RESULTS)
            assert record(checkpoint) == receipt['checkpoint_sha256'] == run.checkpoint_sha256, checkpoint
            assert record(pred_file) == run.predictions_sha256
            log_path = checkpoint.parent / 'retrain_log.csv'
            record(log_path)
            log = pd.read_csv(log_path)
            assert len(log) == 50
            if pooled:
                assert int(log.loc[log.val_f1.idxmax(), 'epoch']) == run.selected_epoch
            db_path = checkpoint.parent / (study + '.db')
            with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
                states = dict(db.execute('SELECT state, COUNT(*) FROM trials GROUP BY state').fetchall())
            assert states.get('COMPLETE', 0) + states.get('PRUNED', 0) == 50 and not states.get('RUNNING', 0), study
            data = pd.read_csv(pred_file).fillna({'patient_id': ''})
            assert data[reference_columns[:-1]].equals(reference[reference_columns[:-1]])
            # The dataset exporter fills blank hold-out weights with 1.0; metrics remain unweighted.
            assert data.sample_weight.equals(reference.sample_weight.fillna(1.0))
            if 'group_all' in reference:
                assert data.group_all.equals(reference.group_all)
            assert data.y_true.equals(reference.diagnosis_categories)
            prob = data[['p_' + c for c in CLASSES]].to_numpy(float)
            y = np.array([CLASSES.index(c) for c in data.y_true])
            assert np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all()
            assert np.allclose(prob.sum(axis=1), 1, atol=2e-6)
            pred = prob.argmax(axis=1)
            assert np.array_equal(data.y_pred, np.asarray(CLASSES)[pred])
            key = {'endpoint': endpoint, 'model': run.model, 'seed': int(run.seed), 'n_images': len(y)}
            truth = np.eye(4)[y]
            confidence = prob.max(axis=1)
            suspicious = np.isin(y, (1, 3)).astype(int)
            score = np.clip(prob[:, 1] + prob[:, 3], 0, 1)
            metrics = {**key, 'multiclass_brier': np.mean(np.sum((prob - truth)**2, axis=1)),
                       'multiclass_nll': -np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-15, 1))),
                       'accuracy': np.mean(pred == y), 'mean_confidence': confidence.mean()}
            tasks = [('top1_confidence', confidence, pred == y), ('screening', score, suspicious)]
            tasks += [(c, prob[:, i], y == i) for i, c in enumerate(CLASSES)]
            for task, scores, outcomes in tasks:
                for bins in (5, 10, 20):
                    ece, count, pmean, fraction = reliability(scores, outcomes, bins)
                    metrics[task + f'_ece_{bins}'] = ece
                    if bins == 10:
                        for i in range(bins):
                            bin_rows.append({**key, 'task': task, 'bin': i, 'n_bin': int(count[i]),
                                'lower': i / bins, 'upper': (i + 1) / bins,
                                'mean_score': pmean[i], 'event_fraction': fraction[i]})
                if task != 'top1_confidence':
                    metrics[task + '_brier'] = np.mean((scores - outcomes)**2)
                    metrics[task + '_prevalence'] = np.mean(outcomes)
                    metrics[task + '_mean_score'] = np.mean(scores)
                    metrics[task + '_calibration_bias'] = np.mean(scores - outcomes)
            metrics['screening_nll'] = -np.mean(suspicious * np.log(np.clip(score, 1e-15, 1)) +
                                              (1 - suspicious) * np.log(np.clip(1 - score, 1e-15, 1)))
            metric_rows.append(metrics)
            f1 = f1_score(y, pred, labels=range(4), average=None, zero_division=0)
            for i, c in enumerate(CLASSES):
                old = previous_class[(previous_class.endpoint == endpoint) & (previous_class.model == run.model) &
                                     (previous_class.seed == run.seed) & (previous_class['class'] == c)].iloc[0]
                assert np.isclose(f1[i], old.f1, atol=1e-12)
                class_rows.append({**key, 'class': c, 'support': int(np.sum(y == i)), 'f1': f1[i]})
            audit_rows.append({**key, 'study': study, 'checkpoint': str(checkpoint.relative_to(ROOT)),
                'checkpoint_sha256': receipt['checkpoint_sha256'], 'predictions_sha256': run.predictions_sha256,
                'corrected_evaluation_manifest': eval_file, 'yaml': CONFIGS[run.model],
                'yaml_sha256': pooled_plan['original_yaml_sha256'][run.model], 'finished_trials': 50,
                'retrain_epochs': 50, 'retrain_and_freeze_verified': True})
        print(f'Verified final-retrain checkpoints, YAMLs and frozen cases: {endpoint} (20 runs)', flush=True)
    metrics = pd.DataFrame(metric_rows)
    classes = pd.DataFrame(class_rows)
    bins = pd.DataFrame(bin_rows)
    for filename, df in [('run_audit', pd.DataFrame(audit_rows)), ('per_seed_metrics', metrics),
                         ('per_class_f1', classes), ('reliability_bins', bins)]:
        df.to_csv(OUTPUT / (filename + '.csv'), index=False)
    summary = metrics.groupby(['endpoint', 'model']).agg(['mean', 'std'])
    summary.to_csv(OUTPUT / 'model_summary.csv')
    def cell(endpoint, model, metric):
        return fmt(*summary.loc[(endpoint, model), metric])
    f1_rows = []
    for endpoint in ('pooled', 'kaggle', 'zenodo', 'mendeley'):
        for model in MODELS:
            cells = []
            for c in ('healthy', 'benign_lesion', 'opmd', 'cancer'):
                group = classes[(classes.endpoint == endpoint) & (classes.model == model) & (classes['class'] == c)]
                cells.append(fmt(group.f1.mean(), group.f1.std()) if group.support.iloc[0] > 0 else '--')
            f1_rows.append([endpoint.title(), DISPLAY[model], *cells])
    f1_tex = table('Class-specific four-class top-1 F1: mean $\\pm$ sample SD across five verified final retrains. Absent reference classes are shown as --, rather than interpreted as measured class performance. F1 uses one-versus-rest TP, FP and FN in the four-class output, not a rank-based exclusion rule.',
                   'tab:s_class_f1_compact', ['Endpoint', 'Model', 'Healthy', 'Benign lesion', 'OPMD', 'Cancer'], f1_rows)
    (PAPER / 'tables/bpex_per_class_f1.tex').write_text(f1_tex)
    tex = [r'''\section{Probability calibration diagnostics}
These diagnostics use raw probabilities from the same validation-selected
final retrains as the preceding analyses. All 80 evaluation bundles were
checked against their final-retrain checkpoint hashes, the corrected frozen
evaluation cases and the original YAML hashes recorded in the study plans.
No calibrator was fitted and no probability or decision threshold was changed.

Multiclass Brier score is the mean sum of four squared probability errors
(range 0 to 2). Screening Brier score is the binary mean squared error
(range 0 to 1). Negative log-likelihood (NLL) uses natural logarithms;
probabilities inside logarithms are clipped at $10^{-15}$ for numerical
evaluation only. These proper scores measure both calibration and
discrimination; they are not pure measures of calibration.

Expected calibration error (ECE) is
$\sum_b(n_b/N)|\bar p_b-\bar y_b|$, using ten fixed equal-width bins
on $[0,1]$. Top-1 ECE compares maximum probability with four-class
correctness; screening ECE compares $p(\mathrm{OPMD})+p(\mathrm{cancer})$
with the suspicious reference label. Class-wise ECE compares each class
probability with its one-versus-rest reference indicator. ECE depends on
binning and sample size; five- and twenty-bin sensitivity diagnostics are
retained in the CSV bundle. Values below are means and sample SDs across
seeds, not estimates from pooled repeated predictions.''']
    rows = [[endpoint.title(), DISPLAY[model], *[cell(endpoint, model, k) for k in
             ('accuracy', 'mean_confidence', 'multiclass_brier', 'multiclass_nll', 'top1_confidence_ece_10',
              'screening_brier', 'screening_nll', 'screening_ece_10')]]
            for endpoint in ('pooled', 'kaggle', 'zenodo', 'mendeley') for model in MODELS]
    tex.append(table('Uncalibrated probability diagnostics across the corrected pooled test and source hold-outs. Lower Brier, NLL and ECE are better, but scores across different endpoint populations are not paired comparisons.',
                     'tab:s_calibration', ['Endpoint', 'Model', 'Top-1 acc.', 'Mean max $p$', 'Multi Brier', 'Multi NLL', 'Top-1 ECE', 'Screen Brier', 'Screen NLL', 'Screen ECE'], rows, True))
    rows = [[DISPLAY[model], c.replace('_', ' '), *[cell('pooled', model, c + suffix) for suffix in
             ('_prevalence', '_mean_score', '_calibration_bias', '_brier', '_ece_10')]]
            for model in MODELS for c in ('healthy', 'benign_lesion', 'opmd', 'cancer')]
    tex.append(table('Pooled one-versus-rest probability diagnostics. Bias is mean predicted probability minus reference prevalence. An absent reference class on a hold-out cannot validate positive-case calibration; complete endpoint bin counts are provided in the CSV bundle.',
                     'tab:s_class_calibration', ['Model', 'Class', 'Prevalence', 'Mean $p$', 'Bias', 'Brier', 'ECE'], rows, True))
    for endpoint in ('pooled', 'kaggle', 'zenodo', 'mendeley'):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for ax, model in zip(axes.flat, MODELS):
            for task, color in (('top1_confidence', 'tab:blue'), ('screening', 'tab:orange')):
                for seed in SEEDS:
                    sub = bins[(bins.endpoint == endpoint) & (bins.model == model) & (bins.seed == seed) &
                               (bins.task == task) & (bins.n_bin > 0)]
                    ax.plot(sub.mean_score, sub.event_fraction, '-o', lw=.8, ms=3, alpha=.6,
                            color=color, label=task.replace('_', ' ') if seed == 42 else None)
            ax.plot([0, 1], [0, 1], ':', color='gray')
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean raw probability in bin', ylabel='Observed reference frequency', title=DISPLAY[model])
            ax.legend(fontsize=8)
        filename = endpoint + '_reliability.pdf'
        fig.savefig(FIGURES / filename)
        plt.close(fig)
        tex.append(r'\begin{figure}[p]\centering\includegraphics[width=0.94\textwidth]{figures/revision/' + filename +
                   r'}\caption{' + endpoint.title() + r' reliability diagrams for top-1 correctness and suspicious labels, separately by seed. Ten equal-width bins; empty bins omitted. The diagonal denotes perfect calibration. No confidence bands or fitted calibrators are shown; sparse bins can be unstable.}\end{figure}')
    tex.append(r'\clearpage')
    table_path = PAPER / 'tables/bpex_calibration_diagnostics.tex'
    table_path.write_text('\n\n'.join(tex) + '\n')
    manifest = {'input_sha256': inputs, 'evaluated_runs': 80, 'seeds': SEEDS,
        'calibration_fitted': False, 'bin_counts': [5, 10, 20], 'log_clipping_epsilon': 1e-15,
        'output_sha256': {str(p.relative_to(ROOT)): sha(p) for p in list(OUTPUT.glob('*.csv')) +
                          [table_path, weight_path, PAPER / 'tables/bpex_per_class_f1.tex'] + list(FIGURES.glob('*_reliability.pdf'))}}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(summary.loc['pooled'].round(4).to_string())
    print(classes[classes.endpoint == 'pooled'].groupby(['model', 'class']).f1.agg(['mean', 'std']).round(4).to_string())


if __name__ == '__main__':
    main()
