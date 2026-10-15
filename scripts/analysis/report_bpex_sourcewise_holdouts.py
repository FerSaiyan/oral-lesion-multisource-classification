#!/usr/bin/env python3
"""Create reviewer-ready tables and confusion/ROC/PR figures from audited metrics."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_bpex_sourcewise_holdouts import OUTPUT, SEEDS, SOURCES
from analyze_bpex_final_seeds import CLASSES, MODELS, require, sha256
from run_bpex_revision_hpo import ROOT

DISPLAY = {'effnet_b3': 'EfficientNet-B3', 'effnet_b7': 'EfficientNet-B7',
           'vit_base': 'ViT-Base', 'vit_large': 'ViT-Large'}


def save(fig, directory, filename):
    fig.savefig(directory / f'{filename}.pdf', bbox_inches='tight')
    fig.savefig(directory / f'{filename}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    manifest = json.loads((OUTPUT / 'analysis_manifest.json').read_text())
    for file, digest in manifest['output_sha256'].items():
        require(sha256(OUTPUT / file) == digest, f'Changed analysis output: {file}')
    figures = OUTPUT / 'figures'
    figures.mkdir(exist_ok=True)
    summary = pd.read_csv(OUTPUT / 'model_summary.csv')
    intervals = pd.read_csv(OUTPUT / 'bootstrap_intervals.csv', dtype={'seed': str})
    table = summary[summary.evaluation_source == 'ALL'].copy()
    for metric in ('macro_f1_present_classes', 'macro_f1_four_classes', 'weighted_f1', 'accuracy',
                   'balanced_accuracy', 'binary_sensitivity', 'binary_specificity', 'binary_roc_auc'):
        ci = intervals[(intervals.seed == 'mean') & (intervals.metric == metric)]
        table = table.merge(ci[['excluded_source', 'kind', 'model', 'ci_low', 'ci_high']].rename(
            columns={'ci_low': metric + '_ci_low', 'ci_high': metric + '_ci_high'}),
            on=['excluded_source', 'kind', 'model'], validate='one_to_one')
    table.to_csv(OUTPUT / 'table2_source_exclusion.csv', index=False)
    tex = [r'\begin{tabular}{llrrrr}', r'\toprule',
           r'Excluded source & Model & Macro-F1 & Weighted-F1 & Sensitivity & Specificity \\', r'\midrule']
    for _, row in table[table.kind == 'holdout'].iterrows():
        values = [f'${row[m + "_mean"]:.3f}\\pm{row[m + "_sd"]:.3f}$' for m in
                  ('macro_f1_present_classes', 'weighted_f1', 'binary_sensitivity', 'binary_specificity')]
        tex.append(' & '.join([row.excluded_source.title(), DISPLAY[row.model], *values]) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}',
            '% Mean and sample SD of five independently optimized seeds. Macro-F1 averages reference-present classes.',
            '% Four-class predictions are retained. Bootstrap CIs are supplied in table2_source_exclusion.csv.']
    (OUTPUT / 'table2_source_exclusion.tex').write_text('\n'.join(tex) + '\n')
    note = ['# Source-exclusion results for BPEX revision', '',
            'All 60 studies completed; 120 frozen endpoint prediction files were analyzed.',
            'Values below are the mean and sample standard deviation across seeds 42 to 46.',
            'Macro-F1 uses the classes present in the reference labels; four-class macro-F1 is also saved.',
            'Screening uses p(cancer) + p(OPMD) at threshold 0.5.', '',
            'Confidence intervals use 10,000 paired source-stratified bootstrap draws. Internal and Zenodo',
            'photos are clustered by patient; Kaggle and Mendeley are resampled by image because patient IDs',
            'are unavailable. The intervals condition on the five selected checkpoints.', '']
    for kind in ('holdout', 'in_domain'):
        note += [f'## {"Excluded-source holdouts" if kind == "holdout" else "In-domain test subsets"}', '',
                 '| Excluded source | Model | n | Macro-F1 | Weighted-F1 | Sensitivity | Specificity | Screening AUC |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        for _, r in table[table.kind == kind].iterrows():
            cells = [f'{r[m + "_mean"]:.3f} ± {r[m + "_sd"]:.3f}' for m in
                     ('macro_f1_present_classes', 'weighted_f1', 'binary_sensitivity', 'binary_specificity', 'binary_roc_auc')]
            note.append('| ' + ' | '.join([r.excluded_source.title(), DISPLAY[r.model], str(int(r.n_images)), *cells]) + ' |')
        note.append('')
    note += ['## Interpretation and artifacts', '',
             'Kaggle has healthy and cancer reference labels; Mendeley has benign and cancer labels.',
             'Zenodo contains all four classes. Absent-class recall and ROC-AUC are undefined.',
             'The full four-class output is retained, so a cancer-to-OPMD error remains a multiclass error',
             'even when it is positive under the screening collapse.', '',
             'These exploratory source-exclusion studies used independent per-seed HPO. They do not isolate',
             'the effect of class weighting, optimization or synthetic augmentation. Their in-domain test',
             'splits differ from the pooled corrected test set, so cross-condition differences are not paired ablation effects.', '',
             'Seed-wise metrics, per-class metrics, confusion matrices, ROC/PR curves, bootstrap intervals and',
             'Holm-adjusted architecture comparisons are in this directory. Comparisons are paired within',
             'an endpoint, with Holm correction over 18 comparisons per metric and endpoint kind.', '',
             'Clinical expert agreement, clinically corrected-label rates and a histopathology-restricted',
             'analysis require additional clinician/reference-standard data and cannot be inferred from predictions.']
    (OUTPUT / 'review_letter_metrics.md').write_text('\n'.join(note) + '\n')
    confusion = pd.read_csv(OUTPUT / 'confusion_matrices.csv')
    binary = pd.read_csv(OUTPUT / 'binary_confusion_matrices.csv')
    seeds_metrics = pd.read_csv(OUTPUT / 'per_seed_metrics.csv')
    for label, frame in [('confusion', confusion), ('binary_confusion', binary)]:
        frame.groupby(['excluded_source', 'kind', 'model', 'evaluation_source', 'true', 'predicted']).agg(
            count_mean=('count', 'mean'), count_sd=('count', 'std'),
            row_fraction_mean=('row_fraction', 'mean'), row_fraction_sd=('row_fraction', 'std')).to_csv(
                OUTPUT / f'{label}_seed_summary.csv')
    roc = pd.read_csv(OUTPUT / 'roc_curves.csv')
    pr = pd.read_csv(OUTPUT / 'pr_curves.csv')
    for excluded in SOURCES:
        for kind in ('holdout', 'in_domain'):
            for label, frame, class_order in [('multiclass', confusion, CLASSES),
                                              ('screening', binary, ('non_suspicious', 'suspicious'))]:
                sub = frame[(frame.excluded_source == excluded) & (frame.kind == kind) & (frame.evaluation_source == 'ALL')]
                for normalized in (False, True):
                    fig, axes = plt.subplots(4, 6, figsize=(19, 12), constrained_layout=True)
                    for i, model in enumerate(MODELS):
                        one = sub[sub.model == model]
                        for j, seed in enumerate((*SEEDS, 'mean')):
                            data = one if seed == 'mean' else one[one.seed == seed]
                            col = 'row_fraction' if normalized else 'count'
                            matrix = data.groupby(['true', 'predicted'])[col].mean().unstack().reindex(
                                index=class_order, columns=class_order).to_numpy()
                            ax = axes[i, j]
                            ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1 if normalized else np.nanmax(matrix))
                            for a in range(len(class_order)):
                                for b in range(len(class_order)):
                                    value = matrix[a, b]
                                    text = 'NA' if not np.isfinite(value) else (f'{value:.2f}' if normalized or seed == 'mean' else str(int(value)))
                                    ax.text(b, a, text, ha='center', va='center', fontsize=7,
                                            color='white' if np.isfinite(value) and value > np.nanmax(matrix) * .65 else 'black')
                            ax.set_xticks(range(len(class_order)), class_order, rotation=45, ha='right', fontsize=7)
                            ax.set_yticks(range(len(class_order)), class_order, fontsize=7)
                            if i == 3: ax.set_xlabel('Predicted class', fontsize=8)
                            if j == 0: ax.set_ylabel('True class', fontsize=8)
                            ax.set_title(f'{DISPLAY[model]}, {"seed mean" if seed == "mean" else seed}', fontsize=9)
                    fig.suptitle(f'No {excluded.title()}: {kind.replace("_", " ")} / {label} / '
                                 f'{"row-normalized" if normalized else "counts (seed mean is not pooled n×5)"}')
                    save(fig, figures, f'{excluded}_{kind}_{label}_{"normalized" if normalized else "counts"}')
            for label, curves, x, y in [('roc', roc, 'fpr', 'tpr'), ('pr', pr, 'recall', 'precision')]:
                fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
                sub = curves[(curves.excluded_source == excluded) & (curves.kind == kind) & (curves['class'] == 'screening')]
                for ax, model in zip(axes, MODELS):
                    for seed in SEEDS:
                        data = sub[(sub.model == model) & (sub.seed == seed)]
                        value = seeds_metrics[(seeds_metrics.excluded_source == excluded) & (seeds_metrics.kind == kind)
                                              & (seeds_metrics.model == model) & (seeds_metrics.seed == seed)
                                              & (seeds_metrics.evaluation_source == 'ALL')].iloc[0]
                        metric = 'binary_roc_auc' if label == 'roc' else 'binary_average_precision'
                        ax.plot(data[x], data[y], label=f'Seed {seed}: {value[metric]:.3f}', linewidth=1)
                    if label == 'roc': ax.plot([0, 1], [0, 1], ':', color='gray')
                    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=x, ylabel=y, title=DISPLAY[model])
                    ax.legend(fontsize=7)
                fig.suptitle(f'No {excluded.title()}: {kind.replace("_", " ")} screening {label.upper()}')
                save(fig, figures, f'{excluded}_{kind}_screening_{label}')
                fig, axes = plt.subplots(4, 4, figsize=(13, 11), constrained_layout=True)
                sub = curves[(curves.excluded_source == excluded) & (curves.kind == kind)]
                for i, model in enumerate(MODELS):
                    for j, class_label in enumerate(CLASSES):
                        ax = axes[i, j]
                        one = sub[(sub.model == model) & (sub['class'] == class_label)]
                        if one.empty:
                            ax.text(.5, .5, 'Undefined: class absent', ha='center', va='center')
                        else:
                            for seed in SEEDS:
                                data = one[one.seed == seed]
                                ax.plot(data[x], data[y], label=f'Seed {seed}', linewidth=1)
                        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=x, ylabel=y,
                               title=f'{DISPLAY[model]} / {class_label}')
                fig.suptitle(f'No {excluded.title()}: {kind.replace("_", " ")} one-vs-rest {label.upper()}')
                save(fig, figures, f'{excluded}_{kind}_multiclass_ovr_{label}')
    inventory = {str(p.relative_to(OUTPUT)): sha256(p) for p in OUTPUT.rglob('*') if p.is_file()
                 and p.name != 'report_manifest.json'}
    (OUTPUT / 'report_manifest.json').write_text(json.dumps({'files_sha256': inventory,
        'n_pdf_figures': len(list(figures.glob('*.pdf'))), 'table_rows': len(table)}, indent=2) + '\n')
    print(f'Created Table 2, metrics report and {len(list(figures.glob("*.pdf")))} figure PDFs', flush=True)


if __name__ == '__main__':
    main()
