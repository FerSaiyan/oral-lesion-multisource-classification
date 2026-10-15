#!/usr/bin/env python3
"""Render fixed/tuned pooled confusion matrices and ROC/PR without inference."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_bpex_review_metrics_bundle import OUT
from analyze_bpex_final_seeds import CLASSES, MODELS, sha256
from analyze_bpex_sourcewise_holdouts import SEEDS
from report_bpex_sourcewise_holdouts import DISPLAY, save


def main():
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    multiclass = pd.read_csv(OUT / 'pooled_confusion_matrices.csv')
    binary = pd.read_csv(OUT / 'pooled_binary_confusion_matrices.csv')
    curves = pd.read_csv(OUT / 'pooled_roc_pr_curves.csv')
    metrics = pd.read_csv(OUT / 'pooled_per_seed_per_source_metrics.csv')
    for label, frame in [('confusion', multiclass), ('binary_confusion', binary)]:
        frame.groupby(['arm', 'model', 'source', 'true', 'predicted']).agg(
            count_mean=('count', 'mean'), count_sd=('count', 'std'),
            row_fraction_mean=('row_fraction', 'mean'), row_fraction_sd=('row_fraction', 'std')).to_csv(
                OUT / f'pooled_{label}_seed_summary.csv')
    for arm in ('fixed_primary', 'tuned_exploratory'):
        title = 'Fixed-parameter primary fits' if arm == 'fixed_primary' else 'Independently optimized exploratory fits'
        for label, data, labels in [('multiclass', multiclass, CLASSES),
                                    ('screening', binary, ('non_suspicious', 'suspicious'))]:
            data = data[(data.arm == arm) & (data.source == 'GLOBAL')]
            for normalized in (False, True):
                fig, axes = plt.subplots(4, 6, figsize=(19, 12), constrained_layout=True)
                for i, model in enumerate(MODELS):
                    one = data[data.model == model]
                    for j, seed in enumerate((*SEEDS, 'mean')):
                        sub = one if seed == 'mean' else one[one.seed == seed]
                        col = 'row_fraction' if normalized else 'count'
                        matrix = sub.groupby(['true', 'predicted'])[col].mean().unstack().reindex(index=labels, columns=labels).to_numpy()
                        ax = axes[i, j]
                        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1 if normalized else np.nanmax(matrix))
                        for a in range(len(labels)):
                            for b in range(len(labels)):
                                v = matrix[a, b]
                                text = 'NA' if not np.isfinite(v) else (f'{v:.2f}' if normalized or seed == 'mean' else str(int(v)))
                                ax.text(b, a, text, ha='center', va='center', fontsize=7,
                                        color='white' if np.isfinite(v) and v > np.nanmax(matrix) * .65 else 'black')
                        ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=7)
                        ax.set_yticks(range(len(labels)), labels, fontsize=7)
                        if i == 3: ax.set_xlabel('Predicted class', fontsize=8)
                        if j == 0: ax.set_ylabel('True class', fontsize=8)
                        ax.set_title(f'{DISPLAY[model]}, {"seed mean" if seed == "mean" else seed}', fontsize=9)
                fig.suptitle(f'{title}: 1,209 frozen test cases / {label} / '
                             f'{"row-normalized" if normalized else "counts (seed mean is not pooled n×5)"}')
                save(fig, figures, f'{arm}_{label}_{"normalized" if normalized else "counts"}')
        data = curves[(curves.arm == arm) & (curves.source == 'GLOBAL')]
        for curve in ('ROC', 'PR'):
            sub = data[data.curve == curve]
            fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
            for ax, model in zip(axes, MODELS):
                for seed in SEEDS:
                    one = sub[(sub.model == model) & (sub.seed == seed) & (sub['class'] == 'screening')]
                    value = metrics[(metrics.arm == arm) & (metrics.model == model) & (metrics.seed == seed)
                                    & (metrics.source == 'GLOBAL')].iloc[0]
                    metric = 'binary_roc_auc' if curve == 'ROC' else 'binary_average_precision'
                    ax.plot(one.x, one.y, label=f'Seed {seed}: {value[metric]:.3f}', linewidth=1)
                if curve == 'ROC': ax.plot([0, 1], [0, 1], ':', color='gray')
                ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='FPR' if curve == 'ROC' else 'Recall',
                       ylabel='Sensitivity' if curve == 'ROC' else 'Precision', title=DISPLAY[model])
                ax.legend(fontsize=7)
            fig.suptitle(f'{title}: screening {curve}, threshold-free curves')
            save(fig, figures, f'{arm}_screening_{curve.lower()}')
            fig, axes = plt.subplots(4, 4, figsize=(13, 11), constrained_layout=True)
            for i, model in enumerate(MODELS):
                for j, label in enumerate(CLASSES):
                    ax = axes[i, j]
                    for seed in SEEDS:
                        one = sub[(sub.model == model) & (sub.seed == seed) & (sub['class'] == label)]
                        ax.plot(one.x, one.y, label=f'Seed {seed}', linewidth=1)
                    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='FPR' if curve == 'ROC' else 'Recall',
                           ylabel='Sensitivity' if curve == 'ROC' else 'Precision', title=f'{DISPLAY[model]} / {label}')
            fig.suptitle(f'{title}: one-vs-rest {curve}')
            save(fig, figures, f'{arm}_multiclass_ovr_{curve.lower()}')
    (OUT / 'figure_manifest.json').write_text(json.dumps({
        'n_pdf_figures': len(list(figures.glob('*.pdf'))),
        'files_sha256': {p.name: sha256(p) for p in figures.iterdir() if p.is_file()},
    }, indent=2) + '\n')
    print('Saved pooled reviewer figures for both scientific arms', flush=True)


if __name__ == '__main__':
    main()
