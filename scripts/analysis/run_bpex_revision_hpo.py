#!/usr/bin/env python3
"""Run one corrected-split BPEX Optuna study from an original model YAML.

Print the resolved settings by default. --execute starts the original number
of trials in a new study directory; the held-out test is not accessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.exp.config import load_config


ORIGINAL_CONFIGS = {
    'effnet_b3': 'effnet_imagenet_b3_ar_multisource.yaml',
    'effnet_b7': 'effnet_imagenet_ar.yaml',
    'vit_base': 'vit_base_imagenet_ar.yaml',
    'vit_large': 'vit_large_imagenet_ar.yaml',
}
SEEDWISE_OUTPUT = 'results/revision_bpex/hpo_seedwise_corrected_global_v1'
SOURCEWISE_OUTPUT = 'results/revision_bpex/hpo_sourcewise_corrected_v1'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=ORIGINAL_CONFIGS, required=True)
    parser.add_argument('--view', type=Path, default=ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24')
    parser.add_argument('--images', type=Path, default=Path('/home/fertroll10/Documents/Struct_mouth_dataset_merged/Multisource_dataset_for_transfer_learning'))
    parser.add_argument('--execute', action='store_true', help='Actually start the Optuna study')
    parser.add_argument('--resume', action='store_true', help='Resume an interrupted study up to its original trial budget')
    parser.add_argument('--seed', type=int, choices=(42, 43, 44, 45, 46),
                        help='Exploratory independent HPO for this training seed; omitting preserves the legacy seed-42 run')
    parser.add_argument('--seedwise-plan', type=Path,
                        default=ROOT / 'configs/studies/bpex_revision_v2_seedwise_hpo_2026-09-26.json')
    parser.add_argument('--excluded-source', choices=('kaggle', 'zenodo', 'mendeley'))
    parser.add_argument('--sourcewise-plan', type=Path,
                        default=ROOT / 'configs/studies/bpex_sourcewise_hpo_2026-09-30.json')
    parser.add_argument('--canonical-image-root', help='Original frozen CSV image root, for verified path-only relocation')
    args = parser.parse_args()
    if args.excluded_source:
        if args.seed is None:
            parser.error('--excluded-source requires --seed')
        args.seedwise_plan = args.sourcewise_plan

    config_path = ROOT / 'configs/studies' / ORIGINAL_CONFIGS[args.model]
    cfg = load_config(str(config_path))
    study = dict(cfg['study'])
    if args.seed is not None:
        plan = json.loads(args.seedwise_plan.read_text(encoding='utf-8'))
        expected = plan['original_yaml_sha256'][args.model]
        if sha256(config_path) != expected:
            raise ValueError(f'Original YAML differs from frozen plan: {config_path}')
        if args.seed not in plan['seeds'] or study['n_optuna_trials'] != plan['trials_per_study']:
            raise ValueError('Seed or original trial budget differs from frozen plan')
        hashes = (plan['source_train_val_sha256'][args.excluded_source]
                  if args.excluded_source else plan['cloud_train_val_sha256'])
        for file, expected_hash in hashes.items():
            view_file = args.view / file
            accepted = expected_hash if isinstance(expected_hash, list) else [expected_hash]
            actual = sha256(view_file) if view_file.is_file() else None
            if actual not in accepted and args.canonical_image_root and args.excluded_source and actual:
                import hashlib
                import pandas as pd

                relocated = pd.read_csv(view_file)
                expected_paths = str(args.images.resolve()) + '/' + relocated.filename
                if not relocated.image_path.equals(expected_paths):
                    raise ValueError('Relocated CSV paths do not match the supplied images root')
                relocated['image_path'] = args.canonical_image_root.rstrip('/') + '/' + relocated.filename
                actual = hashlib.sha256(relocated.to_csv(index=False).encode()).hexdigest()
            if actual not in accepted:
                raise ValueError(f'Cloud corrected train/validation view differs from frozen plan: {view_file}')
        if args.excluded_source:
            import pandas as pd

            for file in hashes:
                data = pd.read_csv(args.view / file)
                if args.excluded_source.upper() in set(data['dataset_source']):
                    raise ValueError('Excluded source found in development data')
    study.update({
        'base_data_dir': str(ROOT),
        'execution_mode': 'optuna',
        'study_name_prefix': f'BPEX_revision_v2_global_val_{args.model}',
        'optuna_results_dir': str(ROOT / 'results/revision_bpex/hpo_corrected_global'),
        'images_folder': str(args.images.resolve()),
        'train_csv_path': str((args.view / 'multisource_train_global.csv').resolve()),
        'val_csv_path': str((args.view / 'multisource_val_global.csv').resolve()),
        'test_csv_path': None,
        'require_precomputed_splits': True,
        'materialized_images_only': True,
        'persist_interim_best': True,
        'mlflow': {'enabled': False},
    })
    if args.seed is not None:
        study['seed'] = args.seed
        study['study_name_prefix'] = f'BPEX_revision_v2_seedwise_global_val_{args.model}_seed{args.seed}'
        study['optuna_results_dir'] = str(ROOT / SEEDWISE_OUTPUT)
        # Transient worker failures can stop Optuna with a valid but short
        # SQLite study. Defer final retraining until 50 COMPLETE/PRUNED trials.
        study['required_finished_optuna_trials'] = plan['trials_per_study']
    if args.excluded_source:
        source = args.excluded_source
        study.update({
            'study_name_prefix': f'BPEX_revision_v3_no_{source}_{args.model}_seed{args.seed}',
            'optuna_results_dir': str(ROOT / SOURCEWISE_OUTPUT),
            'train_csv_path': str((args.view / f'multisource_train_no_{source}.csv').resolve()),
            'val_csv_path': str((args.view / f'multisource_val_no_{source}.csv').resolve()),
            'num_workers': 4,
            'dataset_provenance': 'frozen_manifest_sha256',
            'frozen_plan_sha256': sha256(args.sourcewise_plan),
        })
    model = cfg['models'][0]
    summary = {
        'original_yaml': str(config_path),
        'original_search_space': study.get('search_space', {}),
        'model': model['name'],
        'trials': study['n_optuna_trials'],
        'seed_for_search': study['seed'],
        'seedwise_plan_sha256': sha256(args.seedwise_plan) if args.seed is not None else None,
        'study_name': f"{study['study_name_prefix']}_{model['name']}",
        'train_csv_path': study['train_csv_path'],
        'val_csv_path': study['val_csv_path'],
        'test_csv_path': study['test_csv_path'],
        'images_folder': study['images_folder'],
        'optuna_results_dir': study['optuna_results_dir'],
    }
    print(json.dumps(summary, indent=2))
    if not args.execute:
        print('Dry run: add --execute to start the Optuna study.')
        return
    if not Path(study['train_csv_path']).is_file() or not Path(study['val_csv_path']).is_file():
        raise FileNotFoundError('Materialized corrected training/validation CSVs are missing')
    name = summary['study_name']
    database = Path(study['optuna_results_dir']) / name / f'{name}.db'
    completion = database.parent / 'seedwise_completion.json' if args.seed is not None else None
    if database.exists():
        if not args.resume:
            raise FileExistsError(f'Study already exists at {database}; pass --resume only after the prior process has stopped')
        import optuna

        existing = optuna.load_study(study_name=name, storage=f'sqlite:///{database}')
        # A VM-level STOP can strand a RUNNING trial. Its partial epochs are
        # not a completed search attempt; repeat it rather than underfilling
        # the 50-trial budget on the next boot.
        from optuna.trial import TrialState

        for trial in existing.trials:
            if trial.state == TrialState.RUNNING:
                existing._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
        finished = sum(
            trial.state in (TrialState.COMPLETE, TrialState.PRUNED)
            for trial in existing.trials
        )
        remaining = max(0, int(study['n_optuna_trials']) - finished)
        checkpoint = database.parent / 'best_model_retrain.pth'
        print(f'Resuming {name}: {finished} finished trials; {remaining} to reach original budget')
        if remaining == 0 and checkpoint.is_file():
            if completion is None:
                print('Original study and final retrain have already completed.')
                return
            if completion.is_file():
                marker = json.loads(completion.read_text(encoding='utf-8'))
                if (marker.get('plan_sha256') == sha256(args.seedwise_plan)
                        and marker.get('finished_trials') == plan['trials_per_study']
                        and marker.get('retrain_checkpoint_sha256') == sha256(checkpoint)):
                    print('Seed-specific study and final retrain already completed.')
                    return
            print('Full trial budget exists, but no verified final-retrain marker; '
                  'retrain once from the complete study before publishing.')
        study['n_optuna_trials'] = remaining
    from scripts.run_optuna_study import run_study

    result = run_study(study, model)
    print(json.dumps(result, indent=2, default=str))
    if completion is not None and result.get('status') != 'incomplete_study':
        import optuna
        from optuna.trial import TrialState

        finished_study = optuna.load_study(study_name=name, storage=f'sqlite:///{database}')
        finished = sum(trial.state in (TrialState.COMPLETE, TrialState.PRUNED)
                       for trial in finished_study.trials)
        retrain_log = database.parent / 'retrain_log.csv'
        checkpoint = database.parent / 'best_model_retrain.pth'
        study_checkpoint = database.parent / 'best_model_study.pth'
        if finished == plan['trials_per_study'] and all(p.is_file() for p in (
                retrain_log, checkpoint, study_checkpoint)):
            with retrain_log.open(newline='', encoding='utf-8') as stream:
                log_rows = list(csv.DictReader(stream))
            if len(log_rows) == plan['trials_per_study']:
                marker = {
                    'model': args.model, 'seed': args.seed, 'plan_sha256': sha256(args.seedwise_plan),
                    'finished_trials': finished, 'retrain_epochs': len(log_rows),
                    'retrain_checkpoint_sha256': sha256(checkpoint),
                    'study_checkpoint_sha256': sha256(study_checkpoint),
                    'retrain_log_sha256': sha256(retrain_log),
                    'results_csv_sha256': sha256(database.parent / (name + '_optuna_study_results.csv')),
                    'test_csv': None,
                    'excluded_source': args.excluded_source,
                    'runtime_manifest_sha256': {Path(study[k]).name: sha256(Path(study[k]))
                                                for k in ('train_csv_path', 'val_csv_path')},
                    'canonical_image_root': args.canonical_image_root,
                }
                staged = completion.with_suffix('.json.tmp')
                staged.write_text(json.dumps(marker, indent=2, sort_keys=True) + '\n', encoding='utf-8')
                staged.replace(completion)
                print(f'Verified complete study and final retrain: {completion}', flush=True)
            else:
                raise RuntimeError(f'Incomplete retrain log: {len(log_rows)}/50 epochs for {name}')
        else:
            raise RuntimeError(f'Optuna/retrain outputs incomplete for {name}: {finished}/50 trials')


if __name__ == '__main__':
    main()
