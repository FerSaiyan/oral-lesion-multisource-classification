#!/usr/bin/env python3
"""Check scientific case counts, seed coverage and plotted/table data parity."""
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from analyze_bpex_sourcewise_holdouts import OUTPUT, PREDICTIONS, VIEW, SEEDS
from analyze_bpex_final_seeds import CLASSES, MODELS, require, sha256
from build_bpex_review_metrics_bundle import OUT
from queue_bpex_sourcewise_hpo import PLAN, RESULTS, name


def main():
    plan = json.loads(PLAN.read_text())
    points = pd.read_csv(OUTPUT / 'per_seed_metrics.csv')
    summary = pd.read_csv(OUTPUT / 'model_summary.csv')
    matrices = pd.read_csv(OUTPUT / 'confusion_matrices.csv')
    binary = pd.read_csv(OUTPUT / 'binary_confusion_matrices.csv')
    table = pd.read_csv(OUTPUT / 'table2_source_exclusion.csv')
    require(len(table) == 24, 'Table 2 must cover four models, three exclusions and two endpoints')
    checked = 0
    for task in plan['tasks']:
        m = json.loads((RESULTS / name(task) / 'seedwise_completion.json').read_text())
        require(m['model'] == task['model'] and m['seed'] == task['seed'] and m['excluded_source'] == task['source'],
                'Completion metadata does not identify the planned task')
        for kind in ('holdout', 'in_domain'):
            data = pd.read_csv(PREDICTIONS / name(task) / f'{kind}_predictions.csv')
            select = ((points.excluded_source == task['source']) & (points.kind == kind)
                      & (points.model == task['model']) & (points.seed == task['seed'])
                      & (points.evaluation_source == 'ALL'))
            row = points[select].iloc[0]
            require(int(row.n_images) == len(data), 'Metric case count differs from predictions')
            mask = ((matrices.excluded_source == task['source']) & (matrices.kind == kind)
                    & (matrices.model == task['model']) & (matrices.seed == task['seed'])
                    & (matrices.evaluation_source == 'ALL'))
            matrix = matrices[mask].pivot(index='true', columns='predicted', values='count').reindex(
                index=CLASSES, columns=CLASSES).to_numpy()
            require(matrix.sum() == len(data), 'Multiclass confusion matrix does not cover all cases')
            require(np.isclose(np.trace(matrix) / len(data), row.accuracy), 'Accuracy/matrix mismatch')
            support = matrix.sum(axis=1)
            for i, label in enumerate(CLASSES):
                require(support[i] == int(data.y_true.eq(label).sum()), 'Confusion row supports differ')
            subset = binary[(binary.excluded_source == task['source']) & (binary.kind == kind)
                            & (binary.model == task['model']) & (binary.seed == task['seed'])
                            & (binary.evaluation_source == 'ALL')]
            require(subset['count'].sum() == len(data), 'Binary confusion matrix does not cover all cases')
            require(row.binary_tn + row.binary_fp + row.binary_fn + row.binary_tp == len(data),
                    'Binary performance counts differ')
            checked += 1
    for (excluded, kind, model, source), group in points.groupby(['excluded_source', 'kind', 'model', 'evaluation_source']):
        require(set(group.seed) == set(SEEDS), 'Seed summary has missing/duplicated runs')
        ref = summary[(summary.excluded_source == excluded) & (summary.kind == kind)
                      & (summary.model == model) & (summary.evaluation_source == source)].iloc[0]
        for metric in ('weighted_f1', 'macro_f1_present_classes', 'binary_roc_auc'):
            require(np.isclose(ref[metric + '_mean'], group[metric].mean(), equal_nan=True), 'Mean parity failed')
            require(np.isclose(ref[metric + '_sd'], group[metric].std(ddof=1), equal_nan=True), 'Sample SD parity failed')
    pooled = pd.read_csv(OUT / 'pooled_per_seed_per_source_metrics.csv')
    for arm, old_dir in [('fixed_primary', 'five_seed_test_analysis_v1'),
                         ('tuned_exploratory', 'tuned_seedwise_test_analysis_v1')]:
        from run_bpex_revision_hpo import ROOT
        old = pd.read_csv(ROOT / 'results/revision_bpex' / old_dir / 'per_seed_metrics.csv')
        new = pooled[(pooled.arm == arm) & (pooled.source == 'GLOBAL')]
        require(len(new) == len(old) == 20, 'Expected 20 pooled runs in each arm')
        aligned = new.merge(old, on=['model', 'seed'], suffixes=('_new', '_old'), validate='one_to_one')
        for metric in ('weighted_f1', 'accuracy', 'binary_sensitivity', 'binary_specificity', 'binary_roc_auc'):
            require(np.allclose(aligned[metric + '_new'], aligned[metric + '_old'], atol=1e-10),
                    f'Existing pooled result changed: {arm}/{metric}')
    require(sha256(VIEW / 'multisource_test_global.csv') ==
            '1e44ebb0b77b85813d6a30e7bac29f51eababe5f486d1a5a47b7552e530b3513', 'Test freeze changed')
    # Refresh the final index after figure/table generation has completed.
    bundle_path = OUT / 'bundle_manifest.json'
    bundle = json.loads(bundle_path.read_text())
    from run_bpex_revision_hpo import ROOT
    for arm in bundle['arms'].values():
        directory = ROOT / arm['directory']
        arm['files_sha256'] = {str(p.relative_to(directory)): sha256(p)
                               for p in directory.rglob('*') if p.is_file()}
    bundle['files_sha256'] = {p.name: sha256(p) for p in OUT.glob('*.csv')}
    bundle_path.write_text(json.dumps(bundle, indent=2) + '\n')
    manifest = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'verified_sourcewise_views': checked,
                'pooled_prediction_runs': 40, 'seed_means_and_sample_sds_verified': True,
                'existing_primary_and_tuned_metrics_unchanged': True, 'frozen_test_checksum_unchanged': True,
                'pdf_figures': len(list((OUTPUT / 'figures').glob('*.pdf'))) + len(list((OUT / 'figures').glob('*.pdf')))}
    (OUT / 'verification.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
