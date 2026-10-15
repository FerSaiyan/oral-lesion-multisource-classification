#!/usr/bin/env python3
"""Audit conditional rank-based exclusion/referral claims on frozen predictions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from build_revision_evidence import CLASSES, DISPLAY, MODELS, PAPER, RESULTS, ROOT, SEEDS, digest, table

sys.path.insert(0, str(ROOT / 'scripts/analysis'))
from analyze_bpex_final_seeds import bootstrap_counts, cluster_map, interval

OUTPUT = RESULTS / 'rank_triage_analysis_v1'


def divide(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def metrics_from_counts(counts):
    """Cells are TN, FP, FN, TP for a flag that asserts the target is present."""
    tn, fp, fn, tp = np.moveaxis(counts, -1, 0)
    n = tn + fp + fn + tp
    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return {
        'prevalence': divide(tp + fn, n),
        'negative_coverage': divide(tn + fn, n),
        'negative_risk': divide(fn, tn + fn),
        'npv': divide(tn, tn + fn),
        'miss_rate': divide(fn, tp + fn),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'negative_likelihood_ratio': divide(1 - sensitivity, specificity),
        'positive_coverage': divide(tp + fp, n),
        'ppv': divide(tp, tp + fp),
        'f1': divide(2 * tp, 2 * tp + fp + fn),
        'accuracy': divide(tp + tn, n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261006)
    args = parser.parse_args()
    assert args.resamples > 0
    OUTPUT.mkdir(exist_ok=True)
    inputs = {str(Path(__file__).resolve().relative_to(ROOT)): digest(Path(__file__).resolve()),
              str((PAPER / 'bpex_submission.tex').relative_to(ROOT)): digest(PAPER / 'bpex_submission.tex')}
    rows, ci_rows, rank_rows, label_rows, headline_rows, audits, tie_rows = [], [], [], [], [], [], []
    names = list(CLASSES)
    for endpoint in ('pooled', 'kaggle', 'zenodo', 'mendeley'):
        pooled = endpoint == 'pooled'
        analysis = RESULTS / ('tuned_seedwise_test_analysis_v1' if pooled else 'sourcewise_heldout_analysis_v1')
        path = analysis / 'run_provenance.csv'
        inputs[str(path.relative_to(ROOT))] = digest(path)
        provenance = pd.read_csv(path)
        if not pooled:
            provenance = provenance[(provenance.excluded_source == endpoint) & (provenance.kind == 'holdout')]
        assert len(provenance) == 20
        reference, cluster_idx, draws = None, None, None
        per_model_boot = {model: {} for model in MODELS}
        for run in provenance.sort_values(['model', 'seed']).itertuples():
            model, seed = run.model, int(run.seed)
            study = run.source_study if pooled else run.study
            pred_path = RESULTS / ('tuned_seedwise_test_v1' if pooled else 'sourcewise_heldout_v1') / study / ('test_predictions.csv' if pooled else 'holdout_predictions.csv')
            assert digest(pred_path) == run.predictions_sha256
            inputs[str(pred_path.relative_to(ROOT))] = digest(pred_path)
            data = pd.read_csv(pred_path).fillna({'patient_id': ''})
            if reference is None:
                reference = data[['filename', 'y_true', 'dataset_source', 'patient_id']]
                cluster_idx, strata, audit = cluster_map(data)
                audits.extend({'endpoint': endpoint, **a} for a in audit)
                draws = bootstrap_counts({s: ids for s, ids in strata.items() if ids},
                                         args.resamples, args.bootstrap_seed)
            assert reference.equals(data[['filename', 'y_true', 'dataset_source', 'patient_id']])
            prob = data[['p_' + c for c in CLASSES]].to_numpy(float)
            y = np.array([names.index(c) for c in data.y_true])
            assert np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all()
            assert np.allclose(prob.sum(axis=1), 1, atol=2e-6)
            # Stable ordering fixes exact ties. Report boundary ties rather than hiding them.
            order = np.argsort(-prob, axis=1, kind='stable')
            ordered = np.take_along_axis(prob, order, axis=1)
            last = order[:, 3]
            top2_susp = np.all(np.isin(order[:, :2], (1, 3)), axis=1)
            top2_non = np.all(np.isin(order[:, :2], (0, 2)), axis=1)
            assert not np.any(top2_susp & top2_non)
            assert np.all(prob[top2_susp, 1] + prob[top2_susp, 3] >= .5 - 1e-10)
            by = np.isin(y, (1, 3))
            key = dict(endpoint=endpoint, model=model, seed=seed, n_images=len(y))
            tie_rows.append({**key, 'top2_boundary_ties': int(np.sum(ordered[:, 1] == ordered[:, 2])),
                             'bottom_boundary_ties': int(np.sum(ordered[:, 2] == ordered[:, 3]))})
            headline_rows.append({**key, 'cancer_prevalence': np.mean(y == 1), 'healthy_prevalence': np.mean(y == 2),
                'suspicious_prevalence': np.mean(by),
                'top2_accuracy': np.mean(np.any(order[:, :2] == y[:, None], axis=1)),
                'top3_accuracy': np.mean(last != y),
                'always_omit_cancer_top3_accuracy': np.mean(y != 1),
                'always_opmd_weighted_f1': float(divide(2 * np.mean(y == 3)**2, 1 + np.mean(y == 3))),
                'random_three_class_coverage': .75,
                'always_refer_f1': float(divide(2 * by.sum(), len(y) + by.sum())),
                'same_binary_top2_coverage': np.mean(top2_susp | top2_non),
                'same_binary_top2_selective_accuracy': float(divide(np.sum((top2_susp & by) | (top2_non & ~by)), np.sum(top2_susp | top2_non)))})
            for i, c in enumerate(CLASSES):
                ranks = (order == i).argmax(axis=1) + 1
                for rank in range(1, 5):
                    mask = ranks == rank
                    rank_rows.append({**key, 'class': c, 'rank': rank, 'n_at_rank': int(mask.sum()),
                        'true_class_at_rank': int(np.sum(mask & (y == i))),
                        'conditional_class_risk': float(divide(np.sum(mask & (y == i)), mask.sum())),
                        'mean_predicted_probability': prob[mask, i].mean() if mask.any() else np.nan})
            definitions = [(c + '_last', y == i, last != i) for i, c in enumerate(CLASSES)]
            definitions += [('suspicious_top2_pair', by, top2_susp),
                            ('non_suspicious_top2_pair', by, ~top2_non)]
            # Fixed low-probability cutoffs are descriptive diagnostics, not test-selected operating points.
            definitions += [(c + '_prob_le_' + str(t), y == i, prob[:, i] > t)
                            for c, i in (('cancer', 1), ('healthy', 2)) for t in (.01, .05, .1)]
            definitions += [('suspicious_given_cancer_last', by, last != 1),
                            ('suspicious_given_healthy_last', by, last != 2)]
            cluster_counts, rules = [], []
            for rule, truth, positive in definitions:
                code = truth.astype(int) * 2 + positive.astype(int)
                # Code order is TN, FP, FN, TP.
                cm = np.bincount(code, minlength=4)
                aggregate = np.bincount(cluster_idx * 4 + code, minlength=draws.shape[1] * 4).reshape(-1, 4)
                cluster_counts.append(aggregate)
                rules.append(rule)
                values = metrics_from_counts(cm)
                rows.append({**key, 'rule': rule, 'tn': cm[0], 'fp': cm[1], 'fn': cm[2], 'tp': cm[3],
                    'target_present': bool(truth.any()), 'negative_cases': int(cm[0] + cm[2]),
                    'positive_cases': int(cm[1] + cm[3]), **{k: float(v) for k, v in values.items()}})
                if rule in [c + '_last' for c in CLASSES] + ['suspicious_top2_pair', 'non_suspicious_top2_pair']:
                    flag = positive if rule == 'suspicious_top2_pair' else ~positive
                    for i, c in enumerate(CLASSES):
                        label_rows.append({**key, 'rule': rule, 'class': c, 'flagged_cases': int(flag.sum()),
                            'true_class_count': int(np.sum(flag & (y == i))),
                            'conditional_class_fraction': float(divide(np.sum(flag & (y == i)), flag.sum()))})
            bootstrap_cm = (draws @ np.concatenate(cluster_counts, axis=1).astype(np.float32)).reshape(args.resamples, len(rules), 4)
            for index, rule in enumerate(rules):
                for metric, values in metrics_from_counts(bootstrap_cm[:, index]).items():
                    per_model_boot[model].setdefault((rule, metric), []).append(values)
        for model in MODELS:
            for (rule, metric), boot in per_model_boot[model].items():
                assert len(boot) == 5
                low, high, valid = interval(np.mean(boot, axis=0))
                ci_rows.append(dict(endpoint=endpoint, model=model, rule=rule, metric=metric,
                                    ci_low=low, ci_high=high, valid_draws=valid))

    frame = pd.DataFrame(rows)
    intervals = pd.DataFrame(ci_rows)
    headline = pd.DataFrame(headline_rows)
    for filename, df in [('per_seed_rules', frame), ('bootstrap_intervals', intervals),
                         ('class_by_rank', pd.DataFrame(rank_rows)), ('conditional_class_counts', pd.DataFrame(label_rows)),
                         ('headline_per_seed', headline), ('cluster_audit', pd.DataFrame(audits)), ('ties', pd.DataFrame(tie_rows))]:
        df.to_csv(OUTPUT / (filename + '.csv'), index=False)
    numeric = [c for c in frame.columns if c not in ('endpoint', 'model', 'seed', 'rule', 'target_present')]
    summary = frame.groupby(['endpoint', 'model', 'rule'])[numeric].agg(['mean', 'std'])
    summary.to_csv(OUTPUT / 'rule_summary.csv')
    hsummary = headline.groupby(['endpoint', 'model']).agg(['mean', 'std'])
    hsummary.to_csv(OUTPUT / 'headline_summary.csv')
    def cell(endpoint, model, rule, metric):
        mean, sd = summary.loc[(endpoint, model, rule), metric]
        ci = intervals[(intervals.endpoint == endpoint) & (intervals.model == model) & (intervals.rule == rule) & (intervals.metric == metric)].iloc[0]
        return f'{100*mean:.1f} +/- {100*sd:.1f} [{100*ci.ci_low:.1f}, {100*ci.ci_high:.1f}]' if np.isfinite(mean) else 'NA'
    report = ['# Conditional rank-based triage audit', '',
        'All estimates are image-level reference-label outcomes from corrected, independently tuned seeds 42 to 46.',
        'They are not probabilities for a particular patient, observed referral outcomes, or confirmation against universal histopathology.',
        'Values are percentages: mean +/- sample SD [conditional 95% bootstrap interval]. Counts are averaged across seeds; cases are not stacked.',
        f'Intervals use {args.resamples:,} shared, source-stratified resamples; internal and Zenodo by patient, Kaggle and Mendeley by image.',
        'Intervals condition on the five fitted models. Bootstrap draws with an undefined conditional denominator in any seed are excluded.', '',
        '## Pooled test reference prevalence and ranking coverage', '',
        '| Model | Top-2 | Top-3 | Cancer prevalence | Always omit cancer | Always refer F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        r = hsummary.loc[('pooled', model)]
        report.append('| ' + ' | '.join([DISPLAY[model], *[f'{100*r[k]["mean"]:.1f}' for k in
            ('top2_accuracy', 'top3_accuracy', 'cancer_prevalence', 'always_omit_cancer_top3_accuracy', 'always_refer_f1')]]) + ' |')
    for endpoint in ('pooled', 'kaggle', 'zenodo', 'mendeley'):
        report += ['', '## ' + endpoint.title(), '',
                   '| Model / rule | Flagged n (mean) | Coverage | NPV / PPV | Miss rate / sensitivity |',
                   '|---|---:|---|---|---|']
        for model in MODELS:
            for rule, positive in [('cancer_last', False), ('healthy_last', False), ('suspicious_top2_pair', True), ('non_suspicious_top2_pair', False)]:
                r = summary.loc[(endpoint, model, rule)]
                count = r['positive_cases' if positive else 'negative_cases']['mean']
                metrics = ('positive_coverage', 'ppv', 'sensitivity') if positive else ('negative_coverage', 'npv', 'miss_rate')
                report.append('| ' + ' | '.join([DISPLAY[model] + ' / ' + rule, f'{count:.1f}',
                                                 *[cell(endpoint, model, rule, metric) for metric in metrics]]) + ' |')
        report += ['', 'Cancer/healthy-last NPV is P(reference is not that class | class ranked fourth).',
                   'Its miss rate is P(class ranked fourth | reference is that class), a different denominator.',
                   'Suspicious-top-two PPV is P(OPMD or cancer reference | OPMD and cancer occupy ranks 1 and 2).',
                   'Healthy has no reference support in Mendeley; apparent perfect healthy exclusion there cannot validate a healthy rule-out rule.']
    report += ['', '## Interpretation', '',
        'Overall top-3 accuracy is a weighted average over which class was omitted; it does not equal any class-specific NPV.',
        'Overall top-2 accuracy is not the PPV of the OPMD/cancer top-two event.',
        'Cancer ranked fourth does not exclude OPMD. Healthy ranked fourth includes benign reference outcomes, not just suspicious ones.',
        'A three-class list contains 75% of classes. Always omitting cancer already produces high coverage when cancer prevalence is low.',
        'Conditional predictive values depend on source prevalence. Report coverage, excluded-positive counts, miss rate and likelihood ratio beside NPV.',
        'Top-3 belongs as a secondary differential-list coverage measure, not as standalone evidence of safe exclusion.',
        'Probability cutoffs 0.01, 0.05 and 0.10 are exploratory diagnostics, not selected clinical operating points.',
        'Any exclusion threshold intended for use should be selected on validation data for a stated sensitivity target, then tested independently.']
    (OUTPUT / 'report.md').write_text('\n'.join(report) + '\n')
    def texcell(endpoint, model, rule, metric):
        mean, sd = summary.loc[(endpoint, model, rule), metric]
        ci = intervals[(intervals.endpoint == endpoint) & (intervals.model == model) & (intervals.rule == rule) & (intervals.metric == metric)].iloc[0]
        return f'${100*mean:.1f}\\pm{100*sd:.1f}$ $[{100*ci.ci_low:.1f},{100*ci.ci_high:.1f}]$' if np.isfinite(mean) else '--'
    tex = [r'''% Standalone tables; not automatically included in any manuscript.
% Values are percentages: mean +/- sample SD [conditional 95% CI].
\paragraph{Rank-based conditional evaluation.}
All estimates describe reference-labeled images from corrected seeds 42--46,
not individual-patient probabilities or observed referral outcomes. The
10,000-draw bootstrap resamples internal and Zenodo photographs by patient,
and Kaggle and Mendeley by image. Intervals condition on the five checkpoints.
NPV for class $c$ is $P(Y\ne c\mid c\mathrm{\ ranked\ fourth})$; its miss rate
is $P(c\mathrm{\ ranked\ fourth}\mid Y=c)$. These use different denominators.
Counts are seed means and can be fractional; five predictions of the same
image are not treated as five independent observations.''']
    cancer_rows, healthy_rows, referral_rows, holdout_rows = [], [], [], []
    for model in MODELS:
        c = summary.loc[('pooled', model, 'cancer_last')]
        h = summary.loc[('pooled', model, 'healthy_last')]
        cancer_rows.append([DISPLAY[model], f'{c.negative_cases["mean"]:.1f}',
            texcell('pooled', model, 'cancer_last', 'negative_coverage'),
            texcell('pooled', model, 'cancer_last', 'npv'), f'{c.fn["mean"]:.1f}',
            texcell('pooled', model, 'cancer_last', 'miss_rate'), f'{c.negative_likelihood_ratio["mean"]:.3f}'])
        healthy_rows.append([DISPLAY[model], f'{h.negative_cases["mean"]:.1f}',
            texcell('pooled', model, 'healthy_last', 'negative_coverage'),
            texcell('pooled', model, 'healthy_last', 'npv'),
            texcell('pooled', model, 'healthy_last', 'miss_rate'),
            texcell('pooled', model, 'suspicious_given_healthy_last', 'negative_risk')])
        r = summary.loc[('pooled', model, 'suspicious_top2_pair')]
        referral_rows.append([DISPLAY[model], f'{r.positive_cases["mean"]:.1f}',
            texcell('pooled', model, 'suspicious_top2_pair', 'positive_coverage'),
            texcell('pooled', model, 'suspicious_top2_pair', 'ppv'),
            texcell('pooled', model, 'suspicious_top2_pair', 'sensitivity')])
        for endpoint in ('kaggle', 'zenodo', 'mendeley'):
            r = summary.loc[(endpoint, model, 'cancer_last')]
            holdout_rows.append([endpoint.title(), DISPLAY[model], f'{r.negative_cases["mean"]:.1f}',
                texcell(endpoint, model, 'cancer_last', 'npv'), texcell(endpoint, model, 'cancer_last', 'miss_rate')])
    tex.append(table('Pooled cancer-last exclusion. There are 107 cancer-labeled images among 1,209 test images (8.9\%). Excluded cancers are the cancer reference images ranked fourth, not clinical outcomes. LR$^-$ is the negative likelihood ratio, shown as a seed mean.',
        'tab:s_rank_cancer', ['Model', 'Excluded $n$', 'Coverage', 'NPV', 'Excluded cancers', 'Cancer miss rate', 'LR$^-$'], cancer_rows, True))
    tex.append(table('Pooled healthy-last exclusion. There are 248 healthy-labeled images. NPV means a non-healthy reference label, including benign lesions. Suspicious fraction is conditional on healthy ranked fourth.',
        'tab:s_rank_healthy', ['Model', 'Excluded $n$', 'Coverage', 'Non-healthy fraction', 'Healthy miss rate', 'Suspicious fraction'], healthy_rows, True))
    tex.append(table('Pooled OPMD/cancer top-two rule. The overall suspicious reference prevalence is 54.5\%. PPV is the suspicious reference fraction among flagged images; sensitivity is the fraction of all suspicious references flagged. This is a label-based proxy, not a validated referral decision.',
        'tab:s_rank_referral', ['Model', 'Flagged $n$', 'Coverage', 'Suspicious PPV', 'Suspicious sensitivity'], referral_rows, True))
    tex.append(table('Cancer-last exclusion on source hold-outs. Predictive values depend on source composition and prevalence.',
        'tab:s_rank_holdouts', ['Hold-out', 'Model', 'Excluded $n$', 'NPV', 'Cancer miss rate'], holdout_rows, True))
    table_path = PAPER / 'tables/bpex_rank_triage_tables.tex'
    table_path.write_text('\n\n'.join(tex) + '\n')
    manifest = {'input_sha256': inputs, 'seeds': SEEDS, 'resamples': args.resamples,
        'bootstrap_seed': args.bootstrap_seed, 'class_order': CLASSES,
        'rank_tie_rule': 'stable descending probability order; explicit boundary ties recorded',
        'bootstrap_design': 'same source-stratified draws across models and seeds; internal/Zenodo patient clusters; Kaggle/Mendeley image units',
        'unit_of_analysis': 'image; image outcomes clustered by patient for uncertainty, not patient-level diagnoses',
        'output_sha256': {p.name: digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != 'manifest.json'}}
    manifest['table_sha256'] = {str(table_path.relative_to(ROOT)): digest(table_path)}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Computed class-specific rank risks and referral-rule predictive values for 80 frozen runs.')
    print('\n'.join(report[:25]))
    print(frame[frame.endpoint.eq('pooled')].groupby(['model', 'rule'])[['negative_risk', 'negative_coverage', 'npv', 'miss_rate', 'positive_coverage', 'ppv', 'sensitivity']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
