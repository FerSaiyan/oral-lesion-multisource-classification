#!/usr/bin/env python3
"""Complete pooled per-source metrics and index the three distinct revision arms."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from analyze_bpex_final_seeds import CLASSES, MODELS, require, sha256
from analyze_bpex_sourcewise_holdouts import point, SEEDS, VIEW
from analyze_tuned_seedwise import source_dir
from run_bpex_revision_hpo import ROOT
from queue_bpex_sourcewise_hpo import PLAN, RESULTS, name as study_name
from bpex_sourcewise_dispatch import assignments, key

OUT = ROOT / 'results/revision_bpex/review_letter_metrics_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sourcewise = ROOT / 'results/revision_bpex/sourcewise_heldout_analysis_v1'
    require((sourcewise / 'report_manifest.json').is_file(), 'Finish audited source-exclusion report first')
    reference = pd.read_csv(VIEW / 'multisource_test_global.csv').fillna({'patient_id': ''})
    require(sha256(VIEW / 'multisource_test_global.csv') ==
            '1e44ebb0b77b85813d6a30e7bac29f51eababe5f486d1a5a47b7552e530b3513', 'Global test freeze changed')
    fixed_plan = json.loads((ROOT / 'configs/studies/bpex_revision_v2_final_seeds_2026-09-26.json').read_text())
    rows, pc, cm, binary, curves, provenance = [], [], [], [], [], []
    for arm in ('fixed_primary', 'tuned_exploratory'):
        for model in MODELS:
            for seed in SEEDS:
                if arm == 'fixed_primary':
                    architecture = fixed_plan['models'][model]['architecture']
                    name = f'BPEX_revision_v2_final_{model}_seed{seed}_{architecture}'
                    directory = ROOT / 'results/revision_bpex/final_seeds_v2_corrected_global' / name
                    record = json.loads((directory / 'final_fit_provenance.json').read_text())
                    digest = record['test_predictions_sha256']
                else:
                    _, name = source_dir(model, seed)
                    directory = ROOT / 'results/revision_bpex/tuned_seedwise_test_v1' / name
                    record = json.loads((directory / 'test_eval_provenance.json').read_text())
                    digest = sha256(directory / 'test_predictions.csv')
                path = directory / 'test_predictions.csv'
                require(sha256(path) == digest, 'Prediction checksum differs')
                data = pd.read_csv(path).fillna({'patient_id': ''})
                require(len(data) == 1209, 'Global test case count differs')
                for col in ('filename', 'diagnosis_categories', 'dataset_source', 'patient_id', 'source_image_path'):
                    require(data[col].equals(reference[col]), f'Frozen global cases/labels differ: {name}/{col}')
                probabilities = data[[f'p_{label}' for label in CLASSES]].to_numpy(dtype=float)
                require(np.isfinite(probabilities).all() and (probabilities >= 0).all()
                        and np.allclose(probabilities.sum(axis=1), 1, atol=2e-6), 'Invalid probabilities')
                require(np.array_equal(data.y_true, reference.diagnosis_categories)
                        and np.array_equal(data.y_pred, np.asarray(CLASSES)[probabilities.argmax(axis=1)]),
                        'Saved decisions/truth differ from probabilities/reference')
                provenance.append({'arm': arm, 'model': model, 'seed': seed, 'study': name,
                                   'predictions_sha256': digest, 'prediction_file': str(path.relative_to(ROOT))})
                for source in ('GLOBAL', *sorted(reference.dataset_source.unique())):
                    mask = np.ones(len(data), dtype=bool) if source == 'GLOBAL' else reference.dataset_source.eq(source).to_numpy()
                    y = pd.Categorical(reference.loc[mask, 'diagnosis_categories'], categories=CLASSES).codes
                    prob = probabilities[mask]
                    tags = {'arm': arm, 'model': model, 'seed': seed, 'source': source}
                    metrics, matrix, bc, per_class = point(y, prob)
                    rows.append({**tags, 'n_images': int(mask.sum()), **metrics})
                    pc.extend({**tags, **r} for r in per_class)
                    for i, truth in enumerate(CLASSES):
                        for j, pred in enumerate(CLASSES):
                            cm.append({**tags, 'true': truth, 'predicted': pred, 'count': int(matrix[i, j]),
                                       'row_fraction': matrix[i, j] / matrix[i].sum() if matrix[i].sum() else np.nan})
                    for i, truth in enumerate(('non_suspicious', 'suspicious')):
                        for j, pred in enumerate(('non_suspicious', 'suspicious')):
                            binary.append({**tags, 'true': truth, 'predicted': pred, 'count': int(bc[i, j]),
                                           'row_fraction': bc[i, j] / bc[i].sum() if bc[i].sum() else np.nan})
                    for label, truth, score in [('screening', np.isin(y, (1, 3)), prob[:, 1] + prob[:, 3]),
                                                *[(c, y == i, prob[:, i]) for i, c in enumerate(CLASSES)]]:
                        if len(np.unique(truth)) != 2:
                            continue
                        fpr, tpr, t = roc_curve(truth, score)
                        curves.extend({**tags, 'class': label, 'curve': 'ROC', 'x': float(fpr[i]), 'y': float(tpr[i]),
                                       'threshold': float(t[i])} for i in range(len(fpr)))
                        p, r, t = precision_recall_curve(truth, score)
                        curves.extend({**tags, 'class': label, 'curve': 'PR', 'x': float(r[i]), 'y': float(p[i]),
                                       'threshold': float(t[i]) if i < len(t) else np.nan} for i in range(len(p)))
                print(f'Completed pooled source metrics: {arm}/{model}/{seed}', flush=True)
    seed_frame = pd.DataFrame(rows)
    metric_cols = [c for c in seed_frame if c not in ('arm', 'model', 'seed', 'source', 'n_images')]
    summary = []
    for (arm, model, source), group in seed_frame.groupby(['arm', 'model', 'source']):
        r = {'arm': arm, 'model': model, 'source': source, 'n_seeds': len(group), 'n_images': int(group.n_images.iloc[0])}
        for metric in metric_cols:
            r[metric + '_mean'] = group[metric].mean()
            r[metric + '_sd'] = group[metric].std(ddof=1)
        summary.append(r)
    for name, data in [('pooled_per_seed_per_source_metrics', rows), ('pooled_model_summary', summary),
                       ('pooled_per_class_per_source_metrics', pc), ('pooled_confusion_matrices', cm),
                       ('pooled_binary_confusion_matrices', binary), ('pooled_roc_pr_curves', curves),
                       ('pooled_prediction_provenance', provenance)]:
        pd.DataFrame(data).to_csv(OUT / f'{name}.csv', index=False)
    pd.DataFrame(summary).query('source == "GLOBAL"').to_csv(OUT / 'table1_pooled_five_seed_metrics.csv', index=False)
    plan = json.loads(PLAN.read_text())
    owners = assignments(plan, PLAN)
    selections = []
    for task in plan['tasks']:
        name = study_name(task)
        directory = RESULTS / name
        marker = json.loads((directory / 'seedwise_completion.json').read_text())
        trials_path = directory / f'{name}_optuna_study_results.csv'
        require(sha256(trials_path) == marker['results_csv_sha256'], 'Study-results CSV changed')
        trials = pd.read_csv(trials_path)
        winner = trials[trials.state == 'COMPLETE'].sort_values(['value', 'number'], ascending=[False, True]).iloc[0]
        retrain = pd.read_csv(directory / 'retrain_log.csv')
        selected = retrain.iloc[retrain.val_f1.argmax()]
        selections.append({'excluded_source': task['source'], 'model': task['model'], 'seed': task['seed'],
                           'execution_worker': owners[key(task)], 'study': name,
                           'selected_trial': int(winner.number), 'trial_validation_macro_f1': float(winner.value),
                           'selected_retrain_epoch': int(selected.epoch),
                           'retrain_validation_macro_f1': float(selected.val_f1),
                           'retrain_validation_weighted_f1': float(selected.val_f1_weighted),
                           'complete_trials': int(trials.state.eq('COMPLETE').sum()),
                           'pruned_trials': int(trials.state.eq('PRUNED').sum()),
                           'failed_trials': int(trials.state.eq('FAIL').sum()),
                           'search_trial_duration_sum_hours': pd.to_timedelta(trials.duration).sum().total_seconds() / 3600,
                           'search_wall_span_hours': (pd.to_datetime(trials.datetime_complete).max() -
                                                      pd.to_datetime(trials.datetime_start).min()).total_seconds() / 3600,
                           'retrain_epochs': len(retrain),
                           **{c: winner[c] for c in trials if c.startswith('params_')},
                           'checkpoint_sha256': marker['retrain_checkpoint_sha256']})
    pd.DataFrame(selections).to_csv(OUT / 'sourcewise_study_selection_and_search_cost.csv', index=False)
    # Existing paired CIs and comparisons stay associated with the correct scientific arm.
    bundles = {'fixed_primary': ROOT / 'results/revision_bpex/five_seed_test_analysis_v1',
               'tuned_exploratory': ROOT / 'results/revision_bpex/tuned_seedwise_test_analysis_v1',
               'source_exclusion_exploratory': sourcewise,
               'local_compute_benchmark': ROOT / 'results/revision_bpex/inference_benchmark_2026-10-04'}
    inventory = {arm: {'directory': str(d.relative_to(ROOT)),
                       'files_sha256': {str(p.relative_to(d)): sha256(p) for p in d.rglob('*') if p.is_file()}}
                 for arm, d in bundles.items()}
    (OUT / 'bundle_manifest.json').write_text(json.dumps({'prediction_runs': len(provenance),
        'pooled_cases': 1209, 'arms': inventory,
        'files_sha256': {p.name: sha256(p) for p in OUT.glob('*.csv')}}, indent=2) + '\n')
    (OUT / 'README.md').write_text('''# BPEX response-letter metric bundle

The fixed-parameter primary arm and the per-seed HPO exploratory arm each have
20 pooled test prediction files, with seeds 42 to 46 for four architectures.
The source-exclusion exploratory arm has 60 selected retrains and 120 endpoint
prediction files (excluded-source holdout and in-domain test for every study).

`table1_pooled_five_seed_metrics.csv` contains both pooled arms with explicit
labels. The response letter features the exploratory tuned results. Fixed fits
remain separate. Means and sample SDs summarize five runs; patient-aware paired
bootstrap intervals condition on those selected checkpoints.

The `pooled_*` CSVs add complete per-source, per-class, screening and confusion
matrix measures from the already frozen probabilities. The paired global CIs,
architecture comparisons and figures are indexed in `bundle_manifest.json`.

The source-exclusion Table 2, seed-wise metrics, class measures, raw and normalized
confusion matrices, ROC/PR curves, CIs and paired comparisons are in
`../sourcewise_heldout_analysis_v1/`. Screening uses p(cancer)+p(OPMD) at 0.5.
Absent-class sensitivity and AUC are left undefined. No test-derived threshold
or best-seed selection was performed.

`sourcewise_study_selection_and_search_cost.csv` records the actual execution
worker, selected trial/configuration, validation-selected retrain epoch and trial
budgets for all 60 studies. Search spans may include retries and interruptions;
they do not include final retraining. The local synthetic-tensor computational
benchmark is in `../inference_benchmark_2026-10-04/` and excludes decoding/I/O.

Prediction-based performance metrics cannot supply independent clinician kappa,
clinically corrected-label rates, missing acquisition metadata or universal
histopathological confirmation. Those response-letter items still require
clinical records or author input.
''')
    print('Review-letter metric bundle completed', flush=True)


if __name__ == '__main__':
    main()
