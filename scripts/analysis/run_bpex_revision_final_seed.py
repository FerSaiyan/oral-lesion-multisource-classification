#!/usr/bin/env python3
"""Fit one independent corrected-split BPEX seed with frozen HPO parameters.

No search is performed. One validation-selected checkpoint is trained and the
held-out test is evaluated once, after fitting, with per-image probabilities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.exp.config import load_config
from src.oral_lesions.hpo.utils import default_search_space, merge_search_space


PLAN = ROOT / 'configs/studies/bpex_revision_v2_final_seeds_2026-09-26.json'
RESULTS = ROOT / 'results/revision_bpex/final_seeds_v2_corrected_global'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(args: argparse.Namespace) -> tuple[dict, dict, dict, Path]:
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if args.model not in plan['models'] or args.seed not in plan['seeds']:
        raise ValueError('Model or seed is not in the frozen final-fit plan')
    winner = plan['models'][args.model]
    config_path = ROOT / winner['original_yaml']
    if sha256(config_path) != winner['original_yaml_sha256']:
        raise ValueError(f'Original YAML changed since the HPO winner was frozen: {config_path}')
    source = ROOT / 'results/revision_bpex/hpo_corrected_global' / winner['study_name']
    if sha256(source / f"{winner['study_name']}.db") != winner['study_database_sha256']:
        raise ValueError('Source Optuna database differs from the frozen winner plan')
    if sha256(source / f"{winner['study_name']}_optuna_study_results.csv") != winner['study_results_csv_sha256']:
        raise ValueError('Source Optuna results differ from the frozen winner plan')

    test_record = args.test_csv.with_suffix('.json')
    if not test_record.is_file():
        raise FileNotFoundError(f'Missing cloud test-view provenance: {test_record}')
    record = json.loads(test_record.read_text(encoding='utf-8'))
    if (record['source_sha256'] != plan['source_view_sha256']['multisource_test_global.csv']
            or record['derived_sha256'] != sha256(args.test_csv)
            or record['rows'] != 1209
            or record['image_root'] != str(args.images)):
        raise ValueError('Cloud test view does not match the frozen global holdout')

    cfg = load_config(str(config_path))
    model = dict(cfg['models'][0])
    if model['name'] != winner['architecture']:
        raise ValueError('Original YAML model does not match the selected Optuna architecture')
    search = merge_search_space(default_search_space(), cfg['study'].get('search_space'))
    if (search['num_epochs_finetune'].get('value') != plan['fixed_epochs']
            or search['patience'].get('value') != plan['fixed_patience']):
        raise ValueError('The original YAML no longer specifies the frozen epoch/patience settings')
    for key, value in winner['parameters'].items():
        search[key] = {'type': 'fixed', 'value': value}
    study_name_prefix = f'BPEX_revision_v2_final_{args.model}_seed{args.seed}'
    name = f"{study_name_prefix}_{model['name']}"
    study_dir = RESULTS / name
    study = dict(cfg['study'])
    study.update({
        'base_data_dir': str(ROOT), 'execution_mode': 'fixed_single_fit',
        'study_name_prefix': study_name_prefix, 'seed': args.seed,
        'search_space': search, 'optuna_results_dir': str(RESULTS),
        'images_folder': str(args.images),
        'train_csv_path': str(args.view / 'multisource_train_global.csv'),
        'val_csv_path': str(args.view / 'multisource_val_global.csv'),
        'test_csv_path': str(args.test_csv),
        'require_precomputed_splits': True, 'materialized_images_only': True,
        'mlflow': {'enabled': False},
    })
    return plan, winner, study, study_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('effnet_b3', 'effnet_b7', 'vit_base', 'vit_large'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--view', type=Path, default=ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24')
    parser.add_argument('--test-csv', type=Path, default=ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24/heldout_for_final_seeds/multisource_test_global.csv')
    parser.add_argument('--images', type=Path, default=ROOT / 'images/Multisource_dataset_for_transfer_learning')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan, winner, study, output = prepare(args)
    summary = {
        'study_name': output.name, 'model': args.model, 'seed': args.seed,
        'selected_optuna_trial': winner['best_trial'],
        'selected_optuna_validation_macro_f1': winner['best_validation_macro_f1'],
        'fixed_values': {key: value['value'] for key, value in study['search_space'].items() if value['type'] == 'fixed'},
        'train': study['train_csv_path'], 'validation': study['val_csv_path'],
        'heldout_once_after_selection': study['test_csv_path'],
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not args.execute:
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Existing final-fit artifacts must be reviewed before retry: {output}')
    from scripts.run_optuna_study import run_study

    result = run_study(study, dict(load_config(str(ROOT / winner['original_yaml']))['models'][0]))
    fixed = json.loads((output / 'fixed_fit_summary.json').read_text(encoding='utf-8'))
    if fixed['seed'] != args.seed or not fixed['test_predictions']:
        raise RuntimeError('Final fit did not complete the one-time held-out prediction step')
    with (output / 'fixed_fit_log.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    selected = max(enumerate(rows), key=lambda row: (float(row[1]['val_f1']), row[0]))[1]
    if int(selected['epoch']) != fixed['selected_epoch']:
        raise ValueError('Selected checkpoint epoch differs from its validation log')
    provenance = {
        'model': args.model, 'seed': args.seed, 'selection_plan_sha256': sha256(args.plan),
        'optuna_source_trial': winner['best_trial'],
        'optuna_source_study': winner['study_name'],
        'validation_selected_epoch': fixed['selected_epoch'],
        'validation_macro_f1': float(selected['val_f1']),
        'validation_weighted_f1': float(selected['val_f1_weighted']),
        'test_predictions_sha256': fixed['test_predictions_sha256'],
        'test_evaluated_once': True,
    }
    (output / 'final_fit_provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'result': result, 'provenance': provenance}, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
