#!/usr/bin/env python3
"""Infer frozen validation images from an already-selected BPEX final checkpoint.

This supplies per-image validation scores for sensitivity-target threshold
selection; it never opens the held-out test manifest or changes a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('effnet_b3', 'effnet_b7', 'vit_base', 'vit_large'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/studies/bpex_revision_v2_final_seeds_2026-09-26.json')
    parser.add_argument('--view', type=Path, default=ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24')
    parser.add_argument('--images', type=Path, default=ROOT / 'images/Multisource_dataset_for_transfer_learning')
    parser.add_argument('--results', type=Path, default=ROOT / 'results/revision_bpex/final_seeds_v2_corrected_global')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score
    from src.exp.config import load_config
    from src.oral_lesions.data.transforms import build_transforms
    from src.oral_lesions.data.datasets import MultiSourceDatasetCSV
    from src.oral_lesions.models.factory import create_model

    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if args.seed not in plan['seeds']:
        raise ValueError('Seed absent from final-fit plan')
    winner = plan['models'][args.model]
    config_path = ROOT / winner['original_yaml']
    if sha256(config_path) != winner['original_yaml_sha256']:
        raise ValueError('Original architecture YAML differs from the frozen plan')
    cfg = load_config(str(config_path))
    model_cfg = dict(cfg['models'][0])
    if model_cfg['name'] != winner['architecture']:
        raise ValueError('Architecture differs from the selected Optuna winner')
    name = f"BPEX_revision_v2_final_{args.model}_seed{args.seed}_{winner['architecture']}"
    folder = args.results / name
    checkpoint = folder / 'best_model_fixed.pth'
    provenance = json.loads((folder / 'final_fit_provenance.json').read_text(encoding='utf-8'))
    if provenance['selection_plan_sha256'] != sha256(args.plan) or provenance['seed'] != args.seed:
        raise ValueError('Selected checkpoint has different plan/seed provenance')
    summary = json.loads((folder / 'fixed_fit_summary.json').read_text(encoding='utf-8'))
    if sha256(checkpoint) != summary['best_model_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    val_csv = args.view / 'multisource_val_global.csv'
    if sha256(val_csv) != summary['val_csv_sha256']:
        raise ValueError('Validation CSV differs from the training split')
    output = folder / 'validation_predictions.csv'
    metadata = folder / 'validation_inference.json'
    if output.exists() or metadata.exists():
        raise FileExistsError(f'Validation inference already exists: {name}')

    train_classes = set(pd.read_csv(args.view / 'multisource_train_global.csv',
                                    usecols=['diagnosis_categories'])['diagnosis_categories'].dropna())
    val_classes = set(pd.read_csv(val_csv, usecols=['diagnosis_categories'])['diagnosis_categories'].dropna())
    classes = sorted(train_classes | val_classes)
    resize_mode = model_cfg.get('resize_mode') or cfg['study'].get('resize_mode') or 'stretch'
    _, val_transform = build_transforms(model_cfg['model_type'], model_cfg['img_size'],
                                         resize_mode=resize_mode)
    dataset = MultiSourceDatasetCSV(str(val_csv), str(args.images),
                                    transform=val_transform, label_col='diagnosis_categories', classes=classes)
    if len(dataset) != len(pd.read_csv(val_csv)):
        raise ValueError('Validation loader omitted rows')
    loader = DataLoader(dataset, batch_size=args.batch_size or cfg['study'].get('batch_size', 8),
                        shuffle=False, num_workers=args.workers if args.workers is not None
                        else cfg['study'].get('num_workers', 0), pin_memory=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_cfg['dropout_rate'] = winner['parameters']['dropout_rate']
    # Model architecture is identical, but ImageNet weights are unnecessary:
    # strict loading of the final checkpoint replaces the entire state dict.
    model_cfg['pretrained_source'] = 'none'
    model = create_model(model_cfg, num_classes=len(classes), device=device).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    probabilities = []
    selection_predictions = []
    with torch.inference_mode():
        for batch in loader:
            image = batch[0].to(device, non_blocking=True)
            # Trainer.fit_mix selects the checkpoint with AMP validation logits.
            with torch.amp.autocast(device_type=device.type, enabled=device.type != 'cpu'):
                selection_predictions.append(model(image).argmax(dim=1).cpu().numpy())
            # The one-time held-out probabilities use full precision in
            # run_optuna_study.evaluate_selected_checkpoint.
            probabilities.append(torch.softmax(model(image), dim=1).cpu().numpy())
    if getattr(dataset, '_bad_indices', None):
        raise ValueError('Unreadable validation images were substituted by loader')
    scores = np.concatenate(probabilities)
    if len(scores) != len(dataset):
        raise RuntimeError('Validation probability count differs from dataset')
    frame = dataset.data.drop(columns=['resolved_image_path'], errors='ignore').copy()
    frame['y_true'] = frame['diagnosis_categories'].astype(str)
    frame['y_pred'] = [classes[i] for i in scores.argmax(axis=1)]
    for i, label in enumerate(classes):
        frame[f'p_{label}'] = scores[:, i]
    selection_pred = [classes[i] for i in np.concatenate(selection_predictions)]
    selection_macro = float(f1_score(frame['y_true'], selection_pred, labels=classes,
                                     average='macro', zero_division=0))
    selection_weighted = float(f1_score(frame['y_true'], selection_pred, labels=classes,
                                        average='weighted', zero_division=0))
    macro = float(f1_score(frame['y_true'], frame['y_pred'], labels=classes,
                           average='macro', zero_division=0))
    weighted = float(f1_score(frame['y_true'], frame['y_pred'], labels=classes,
                              average='weighted', zero_division=0))
    if (abs(selection_macro - provenance['validation_macro_f1']) > 1e-6
            or abs(selection_weighted - provenance['validation_weighted_f1']) > 1e-6):
        raise ValueError(f'Validation prediction does not reproduce checkpoint log: '
                         f'mixed-precision macro {selection_macro} vs {provenance["validation_macro_f1"]}; '
                         f'weighted {selection_weighted} vs {provenance["validation_weighted_f1"]}; '
                         f'full-precision macro {macro}, weighted {weighted}')
    frame.to_csv(output, index=False)
    record = {'model': args.model, 'seed': args.seed, 'run': name,
              'checkpoint_sha256': sha256(checkpoint), 'val_csv_sha256': sha256(val_csv),
              'validation_predictions_sha256': sha256(output), 'rows': len(frame),
              'selection_amp_macro_f1': selection_macro,
              'selection_amp_weighted_f1': selection_weighted,
              'full_precision_macro_f1': macro, 'full_precision_weighted_f1': weighted,
              'batch_size': loader.batch_size, 'resize_mode': resize_mode}
    metadata.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'{name}: {len(frame)} validation probabilities; '
          f'selection AMP macro F1={selection_macro:.4f}; '
          f'full-precision macro F1={macro:.4f}', flush=True)


if __name__ == '__main__':
    main()
