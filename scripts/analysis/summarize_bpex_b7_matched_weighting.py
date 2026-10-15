#!/usr/bin/env python3
"""Verify completed paired B7 fits and publish deidentified ablation results."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'papers/Multisource Classification'
CLASSES = ['benign_lesion', 'cancer', 'healthy', 'opmd']
NAMES = {'baseline': 'Reference', 'no_inverse_frequency': 'No inverse-frequency'}
METRICS = {
    'macro_f1': 'Macro-F1', 'weighted_f1': 'Weighted-F1', 'accuracy': 'Top-1 accuracy',
    'benign_f1': 'Benign-lesion F1', 'cancer_f1': 'Cancer F1',
    'healthy_f1': 'Healthy F1', 'opmd_f1': 'OPMD F1',
    'benign_recall': 'Benign-lesion recall', 'cancer_recall': 'Cancer recall (four-class)',
    'healthy_recall': 'Healthy recall', 'opmd_recall': 'OPMD recall (four-class)',
    'cancer_precision': 'Cancer precision (four-class)', 'screening_f1': 'Screening F1',
    'screening_sensitivity': 'Screening sensitivity', 'screening_specificity': 'Screening specificity',
    'screening_auc': 'Screening ROC-AUC',
    'cancer_screening_sensitivity': 'Screening sensitivity: cancer images',
    'opmd_screening_sensitivity': 'Screening sensitivity: OPMD images',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_table(name, caption, headers, rows):
    header = ' & '.join(headers) + r' \\'
    lines = [r'\begingroup\footnotesize', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{longtable}{' + 'l' * len(headers) + '}',
             r'\caption{' + caption + r'}\label{tab:' + name + r'}\\',
             r'\toprule', header, r'\midrule\endfirsthead',
             r'\toprule', header, r'\midrule\endhead', r'\bottomrule\endlastfoot']
    lines.extend(' & '.join(row) + r' \\' for row in rows)
    lines += [r'\end{longtable}', r'\endgroup']
    (PAPER / 'supplement' / (name + '.tex')).write_text('\n'.join(lines) + '\n')


def main():
    plan_path = ROOT / 'configs/studies/bpex_b7_matched_weighting_2026-10-13.json'
    plan = json.loads(plan_path.read_text())
    plan_hash = sha256(plan_path)
    base = ROOT / plan['results_directory']
    if (json.loads((base / 'queue_state.json').read_text()) != {'status': 'completed', 'tasks': 10}
            or plan['seeds'] != list(range(42, 47))
            or plan['conditions'] != list(NAMES) or plan['epochs'] != 50):
        raise ValueError('The full paired ablation is not complete')
    for name, expected in plan['materialized_csv_sha256'].items():
        if sha256(ROOT / plan['materialized_directory'] / name) != expected:
            raise ValueError('Materialized frozen manifest changed')
    with (ROOT / plan['materialized_directory'] / 'multisource_test_global.csv').open(newline='') as stream:
        reference = list(csv.DictReader(stream))
    if len(reference) != 1209:
        raise ValueError('Unexpected frozen test size')
    results, protocols, verification = [], {}, []
    for condition in plan['conditions']:
        for seed in plan['seeds']:
            job = base / f'{plan["experiment"]}_{condition}_seed{seed}_effnet_b7_imagenet'
            summary = json.loads((job / 'fixed_fit_summary.json').read_text())
            completion = json.loads((job / 'completion.json').read_text())
            protocol = json.loads((job / 'protocol.json').read_text())
            if (completion['plan_sha256'] != plan_hash or protocol['plan_sha256'] != plan_hash
                    or completion['epochs'] != 50 or completion['seed'] != seed
                    or completion['condition'] != condition or summary['seed'] != seed
                    or summary['execution_mode'] != 'fixed_single_fit' or summary['fit_count'] != 1
                    or summary['source_checkpoint'] is not None
                    or summary['fixed_params'] != dict(plan['shared_parameters'], num_epochs_finetune=50, patience=51)):
                raise ValueError(f'Incomplete or mismatched fit: {job}')
            for partition in ('train', 'val', 'test'):
                if summary[partition + '_csv_sha256'] != plan['materialized_csv_sha256'][f'multisource_{partition}_global.csv']:
                    raise ValueError('Fit used a different partition')
            log_path = job / 'fixed_fit_log.csv'
            with log_path.open(newline='') as stream:
                epochs = list(csv.DictReader(stream))
            selected = max(epochs, key=lambda row: (float(row['val_f1']), int(row['epoch'])))
            if (sha256(log_path) != summary['fixed_fit_log_sha256']
                    or [int(row['epoch']) for row in epochs] != list(range(1, 51))
                    or int(selected['epoch']) != summary['selected_epoch']):
                raise ValueError('Epoch budget or validation-only checkpoint selection differs')
            pred_path = job / 'test_predictions.csv'
            pred_hash = sha256(pred_path)
            if (pred_hash != summary['test_predictions_sha256']
                    or pred_hash != completion['test_predictions_sha256']
                    or summary['best_model_sha256'] != completion['checkpoint_sha256']):
                raise ValueError('Recorded prediction/checkpoint hashes differ')
            with pred_path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            identity = lambda row: (row['filename'], row['diagnosis_categories'], float(row['sample_weight']))
            if list(map(identity, rows)) != list(map(identity, reference)):
                raise ValueError('Test identities, labels, weights or order differ')
            y = np.array([row['y_true'] for row in rows])
            predicted = np.array([row['y_pred'] for row in rows])
            probabilities = np.array([[float(row['p_' + label]) for label in CLASSES] for row in rows])
            if (list(y) != [row['diagnosis_categories'] for row in reference]
                    or not np.isfinite(probabilities).all() or (probabilities < 0).any()
                    or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
                    or list(predicted) != [CLASSES[i] for i in probabilities.argmax(axis=1)]):
                raise ValueError('Invalid test labels or probabilities')
            recalls = recall_score(y, predicted, labels=CLASSES, average=None)
            class_f1 = f1_score(y, predicted, labels=CLASSES, average=None)
            precision = precision_score(y, predicted, labels=CLASSES, average=None, zero_division=0)
            cancer, opmd = y == 'cancer', y == 'opmd'
            if (int(cancer.sum()), int(opmd.sum())) != (107, 552):
                raise ValueError('Unexpected cancer/OPMD support')
            suspicious = cancer | opmd
            score = probabilities[:, 1] + probabilities[:, 3]
            decision = score >= 0.5
            metrics = {
                'macro_f1': float(f1_score(y, predicted, labels=CLASSES, average='macro')),
                'weighted_f1': float(f1_score(y, predicted, average='weighted')),
                'accuracy': float(accuracy_score(y, predicted)),
                'benign_f1': float(class_f1[0]), 'cancer_f1': float(class_f1[1]),
                'healthy_f1': float(class_f1[2]), 'opmd_f1': float(class_f1[3]),
                'benign_recall': float(recalls[0]), 'cancer_recall': float(recalls[1]),
                'healthy_recall': float(recalls[2]), 'opmd_recall': float(recalls[3]),
                'cancer_precision': float(precision[1]),
                'screening_f1': float(f1_score(suspicious, decision)),
                'screening_sensitivity': float(recall_score(suspicious, decision)),
                'screening_specificity': float(recall_score(~suspicious, ~decision)),
                'screening_auc': float(roc_auc_score(suspicious, score)),
                'cancer_screening_sensitivity': float(decision[cancer].mean()),
                'opmd_screening_sensitivity': float(decision[opmd].mean()),
            }
            if not np.isclose(metrics['macro_f1'], summary['test_f1_macro']):
                raise ValueError('Recomputed macro-F1 differs from the fit summary')
            results.append(dict(condition=condition, seed=seed, selected_epoch=summary['selected_epoch'], **metrics))
            study, model = protocol['study'].copy(), protocol['model'].copy()
            if not (study.pop('class_weights') == model.pop('class_weights') == (condition == 'baseline')):
                raise ValueError('Inverse-frequency setting differs from the assigned condition')
            study.pop('study_name_prefix')
            protocols[condition, seed] = (study, model, protocol['code_sha256'])
            verification.append({'job': str(job.relative_to(base)), 'predictions_sha256': pred_hash,
                                 'checkpoint_sha256': summary['best_model_sha256']})
    for seed in plan['seeds']:
        if protocols['baseline', seed] != protocols['no_inverse_frequency', seed]:
            raise ValueError('Paired protocols differ beyond inverse-frequency weighting')
    aggregates = []
    for metric in METRICS:
        for condition in NAMES:
            values = [row[metric] for row in results if row['condition'] == condition]
            aggregates.append({'metric': metric, 'condition': condition,
                               'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1))})
    attachment = PAPER / 'supplement/data/b7_matched_weighting_v1'
    attachment.mkdir(exist_ok=True)
    decomposition = []
    for label, metric in zip(CLASSES, ('benign_f1', 'cancer_f1', 'healthy_f1', 'opmd_f1')):
        support = sum(row['diagnosis_categories'] == label for row in reference)
        means = {row['condition']: row['mean'] for row in aggregates if row['metric'] == metric}
        contribution = (means['no_inverse_frequency'] - means['baseline']) * support / len(reference)
        decomposition.append({'class': label, 'support': support, 'test_share': support / len(reference),
                              'mean_f1': means, 'weighted_f1_change_contribution': contribution})
    total_change = sum(row['weighted_f1_change_contribution'] for row in decomposition)
    weighted_means = {row['condition']: row['mean'] for row in aggregates if row['metric'] == 'weighted_f1'}
    if not np.isclose(total_change, weighted_means['no_inverse_frequency'] - weighted_means['baseline']):
        raise ValueError('Class-specific contributions do not reproduce the weighted-F1 change')
    opmd_contribution = decomposition[3]['weighted_f1_change_contribution']
    (attachment / 'weighted_f1_decomposition.json').write_text(json.dumps({
        'classes': decomposition, 'total_weighted_f1_change': total_change,
        'opmd_fraction_of_net_gain': opmd_contribution / total_change}, indent=2) + '\n')
    for name, data in (('per_seed_metrics.csv', results), ('metric_summary.csv', aggregates)):
        with (attachment / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    differences = [dict(seed=seed, **{metric: next(r[metric] for r in results if r['condition'] == 'no_inverse_frequency' and r['seed'] == seed)
                        - next(r[metric] for r in results if r['condition'] == 'baseline' and r['seed'] == seed)
                        for metric in METRICS}) for seed in plan['seeds']]
    with (attachment / 'paired_differences.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows(differences)
    (attachment / 'protocol.json').write_text(json.dumps(plan, indent=2) + '\n')
    (attachment / 'verification.json').write_text(json.dumps({
        'plan_sha256': plan_hash, 'completed_fits': 10, 'epochs_per_fit': 50,
        'test_images': 1209, 'cancer_images': 107, 'opmd_images': 552,
        'paired_protocols_verified': True, 'screening_threshold': 0.5,
        'generator_sha256': sha256(Path(__file__)), 'fits': verification}, indent=2) + '\n')
    table_rows = []
    for metric, label in METRICS.items():
        values = [row for row in aggregates if row['metric'] == metric]
        table_rows.append([label] + [f'${100*r["mean"]:.1f}\\pm{100*r["sample_sd"]:.1f}$' for r in values])
    write_table('s_b7_weighting',
                r'Matched EfficientNet-B7 loss-weighting ablation on the frozen 1,209-image pooled test. Values are mean $\pm$ sample SD across five paired seeds, expressed as percentages. Reference retains inverse-frequency loss weights; the removal condition retains all class/source sample weights. Four-class F1 and recall use top-1 predictions. Screening uses raw $p(\mathrm{OPMD})+p(\mathrm{cancer})\geq0.5$. Cancer- and OPMD-image screening sensitivities use 107 and 552 reference images, respectively.',
                ['Metric', 'Reference', 'No inverse-frequency'], table_rows)
    seed_metrics = ['macro_f1', 'weighted_f1', 'cancer_recall', 'cancer_screening_sensitivity', 'screening_f1']
    seed_rows = [[NAMES[row['condition']], str(row['seed']), str(row['selected_epoch'])]
                 + [f'{100*row[metric]:.1f}' for metric in seed_metrics] for row in results]
    write_table('s_b7_weighting_seeds',
                r'Per-seed matched B7 ablation results, expressed as percentages. Epoch is the validation-selected checkpoint; every fit completed all 50 epochs. Cancer recall is four-class top-1 recall; cancer screening is sensitivity among cancer reference images at the summed-probability threshold of 0.5.',
                ['Condition', 'Seed', 'Epoch', 'Macro-F1', 'W. F1', 'Cancer recall', 'Cancer screening', 'Screening F1'], seed_rows)
    print('Verified all 10 matched fits; generated ablation tables and deidentified data')


if __name__ == '__main__':
    main()
