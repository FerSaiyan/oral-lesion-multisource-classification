#!/usr/bin/env python3
"""Render saved five-seed screening ROC and PR curves, without re-inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT / 'results/revision_bpex/five_seed_test_analysis_v1')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    per_seed = pd.read_csv(args.analysis / 'per_seed_metrics.csv')
    roc = pd.read_csv(args.analysis / 'binary_roc_curves.csv')
    pr = pd.read_csv(args.analysis / 'binary_pr_curves.csv')
    baseline = (659 / 1209)
    colors = {'effnet_b3': '#b15928', 'effnet_b7': '#ff7f00',
              'vit_base': '#1f78b4', 'vit_large': '#6a3d9a'}
    titles = {'effnet_b3': 'EfficientNet-B3', 'effnet_b7': 'EfficientNet-B7',
              'vit_base': 'ViT-Base', 'vit_large': 'ViT-Large'}
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.linspace(0, 1, 1001)
    for model, color in colors.items():
        one = per_seed[per_seed.model == model]
        roc_mean, pr_mean = [], []
        for seed in sorted(one.seed.unique()):
            subset_roc = roc[(roc.model == model) & (roc.seed.astype(str) == str(seed))]
            subset_pr = pr[(pr.model == model) & (pr.seed.astype(str) == str(seed))]
            if subset_roc.empty or subset_pr.empty:
                raise ValueError(f'Missing saved screening curve: {model} seed {seed}')
            ax_roc.plot(subset_roc.fpr, subset_roc.tpr, color=color, alpha=.15, lw=1)
            ax_pr.plot(subset_pr.recall, subset_pr.precision, color=color, alpha=.15, lw=1)
            roc_mean.append(np.interp(x, subset_roc.fpr, subset_roc.tpr))
            pr_mean.append(np.interp(x, subset_pr.recall.to_numpy()[::-1],
                                     subset_pr.precision.to_numpy()[::-1]))
        auc = one.binary_roc_auc.agg(['mean', 'std'])
        ap = one.binary_average_precision.agg(['mean', 'std'])
        ax_roc.plot(x, np.mean(roc_mean, axis=0), color=color, lw=2.1,
                    label=f"{titles[model]} AUC {auc['mean']:.3f} ± {auc['std']:.3f}")
        ax_pr.plot(x, np.mean(pr_mean, axis=0), color=color, lw=2.1,
                   label=f"{titles[model]} AP {ap['mean']:.3f} ± {ap['std']:.3f}")
    ax_roc.plot([0, 1], [0, 1], ls=':', color='gray', lw=1)
    ax_pr.axhline(baseline, ls=':', color='gray', lw=1,
                  label=f'Prevalence {baseline:.3f}')
    ax_roc.set(xlabel='False-positive rate', ylabel='Sensitivity',
               title='Suspicious vs non-suspicious: test ROC', xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.set(xlabel='Recall (sensitivity)', ylabel='Precision (PPV)',
              title='Suspicious vs non-suspicious: test PR', xlim=(0, 1), ylim=(0, 1.02))
    for ax in (ax_roc, ax_pr):
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='lower right' if ax is ax_roc else 'lower left')
    fig.suptitle('Five independent fixed-parameter fits (42–46); 1,209 held-out images', fontsize=11)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(args.analysis / f'five_seed_screening_roc_pr.{suffix}', dpi=240)
    plt.close(fig)
    print('Saved held-out five-seed screening ROC/PR figure')


if __name__ == '__main__':
    main()
