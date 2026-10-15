#!/usr/bin/env python3
"""Freeze the user-selected 60 independent source-exclusion HPO studies."""
import json
from pathlib import Path

import pandas as pd

from run_bpex_revision_hpo import ORIGINAL_CONFIGS, ROOT, sha256


def main():
    view = ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24'
    cloud = ROOT / 'data/processed/bpex_sourcewise_cloud_2026-09-30'
    cloud.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for source in ('kaggle', 'zenodo', 'mendeley'):
        manifests[source] = {}
        development = []
        for part in ('train', 'val'):
            file = f'multisource_{part}_no_{source}.csv'
            frame = pd.read_csv(view / file)
            assert source.upper() not in set(frame.dataset_source)
            development.append(frame)
            derived = frame.copy()
            derived['image_path'] = '/home/fertroll10/bpex-project/images/Multisource_dataset_for_transfer_learning/' + derived.filename
            derived['source_image_path'] = frame.source_image_path.astype(str).map(
                lambda p: __import__('hashlib').sha256(p.encode()).hexdigest())
            stage = cloud / (file + '.tmp')
            derived.to_csv(stage, index=False)
            target = cloud / file
            if target.exists() and sha256(target) != sha256(stage):
                raise ValueError(f'Existing derived manifest differs: {file}')
            stage.replace(target)
            manifests[source][file] = [sha256(view / file), sha256(target)]
        train, val = development
        assert set(train.image_path).isdisjoint(val.image_path)
        for src in ('MY_CAMERA', 'MY_ROMEU', 'ZENODO'):
            # Internal camera/ROMEU patients can span both image sources.
            sources = ['MY_CAMERA', 'MY_ROMEU'] if src.startswith('MY_') else [src]
            a = set(train[train.dataset_source.isin(sources)].patient_id.dropna())
            b = set(val[val.dataset_source.isin(sources)].patient_id.dropna())
            assert a.isdisjoint(b)
    tasks = [{'source': source, 'model': model, 'seed': seed,
              'worker': 'local' if model in ('effnet_b3', 'vit_base') else 'cloud'}
             for source in ('kaggle', 'zenodo', 'mendeley')
             for model in ORIGINAL_CONFIGS for seed in range(42, 47)]
    plan = {
        'experiment': 'corrected_source_exclusion_independent_seed_hpo',
        'author_decision': 'Four architectures, three source exclusions, five independent HPO seeds; 15-day extension granted.',
        'models': list(ORIGINAL_CONFIGS), 'seeds': list(range(42, 47)),
        'trials_per_study': 50, 'retrain_epochs': 50,
        'selection_metric': 'validation macro F1',
        'results_directory': 'results/revision_bpex/hpo_sourcewise_corrected_v1',
        'source_train_val_sha256': manifests,
        'original_yaml_sha256': {m: sha256(ROOT / 'configs/studies' / f) for m, f in ORIGINAL_CONFIGS.items()},
        'heldout_policy': 'No source holdout or in-domain test CSV in HPO/retraining. Evaluate selected checkpoints locally only after completion.',
        'tasks': tasks,
    }
    file = ROOT / 'configs/studies/bpex_sourcewise_hpo_2026-09-30.json'
    content = json.dumps(plan, indent=2, sort_keys=True) + '\n'
    if file.exists() and file.read_text() != content:
        raise FileExistsError('Existing frozen source plan differs')
    file.write_text(content)
    print(f'Frozen {len(tasks)} studies, 30 local + 30 cloud: {file}')


if __name__ == '__main__':
    main()
