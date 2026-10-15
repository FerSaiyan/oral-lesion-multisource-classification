#!/usr/bin/env python3
"""Evaluate completed source-exclusion retrains locally, once per held-out view."""
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_bpex_revision_hpo import ORIGINAL_CONFIGS, ROOT, sha256
from queue_bpex_sourcewise_hpo import PLAN, RESULTS, complete, name
from src.exp.config import load_config
from src.oral_lesions.data import MultiSourceDatasetCSV, build_transforms
from src.oral_lesions.models.factory import create_model

OUTPUT = ROOT / 'results/revision_bpex/sourcewise_heldout_v1'
VIEW = ROOT / 'data/processed/bpex_revision_v2_materialized_2026-09-24'
IMAGES = '/home/fertroll10/Documents/Struct_mouth_dataset_merged/Multisource_dataset_for_transfer_learning'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ph = sha256(PLAN)
    plan = json.loads(PLAN.read_text())
    freeze = json.loads((VIEW / 'view.json').read_text())['derived_csv_sha256']
    if not all(complete(RESULTS / name(t), ph) for t in plan['tasks']):
        raise RuntimeError('All 60 source-exclusion studies must finish before final evaluation')
    device = torch.device('cuda:0')
    try:
        for task in plan['tasks']:
            study = name(task)
            source = RESULTS / study
            out = OUTPUT / study
            out.mkdir(exist_ok=True)
            ck = source / 'best_model_retrain.pth'
            cfg = load_config(str(ROOT / 'configs/studies' / ORIGINAL_CONFIGS[task['model']]))
            mc = dict(cfg['models'][0])
            frame = pd.read_csv(source / f'{study}_optuna_study_results.csv')
            winner = frame[frame.state == 'COMPLETE'].sort_values('value', ascending=False).iloc[0]
            mc.update({'dropout_rate': float(winner.params_dropout_rate), 'pretrained_source': 'none'})
            _, transform = build_transforms(mc['model_type'], mc['img_size'], resize_mode='preserve')
            train = pd.read_csv(VIEW / f"multisource_train_no_{task['source']}.csv")
            val = pd.read_csv(VIEW / f"multisource_val_no_{task['source']}.csv")
            classes = sorted(set(train.diagnosis_categories) | set(val.diagnosis_categories))
            net = None
            for kind, file in (
                ('holdout', f"multisource_holdout_{task['source']}.csv"),
                ('in_domain', f"multisource_test_no_{task['source']}.csv"),
            ):
                csv = VIEW / file
                assert sha256(csv) == freeze[file]
                record_path = out / f'{kind}_provenance.json'
                dest = out / f'{kind}_predictions.csv'
                if record_path.exists():
                    r = json.loads(record_path.read_text())
                    assert r['checkpoint_sha256'] == sha256(ck)
                    assert r['predictions_sha256'] == sha256(dest)
                    continue
                if dest.exists():
                    raise FileExistsError(f'Partial evaluation requires review: {dest}')
                if net is None:
                    net = create_model(mc, num_classes=len(classes), device=device).to(device)
                    net.load_state_dict(torch.load(ck, map_location=device, weights_only=True), strict=True)
                    net.eval()
                ds = MultiSourceDatasetCSV(str(csv), IMAGES, transform=transform,
                                           label_col='diagnosis_categories', classes=classes)
                assert len(ds) == len(pd.read_csv(csv))
                loader = DataLoader(ds, batch_size=cfg['study']['batch_size'], num_workers=4,
                                    shuffle=False, pin_memory=True)
                scores = []
                with torch.inference_mode():
                    for batch in loader:
                        scores.append(torch.softmax(net(batch[0].to(device)), dim=1).cpu())
                assert not getattr(ds, '_bad_indices', None)
                prob = torch.cat(scores).numpy()
                data = ds.data.drop(columns=['resolved_image_path'], errors='ignore').copy()
                data['y_true'] = data.diagnosis_categories
                data['y_pred'] = [classes[i] for i in prob.argmax(axis=1)]
                for i, c in enumerate(classes):
                    data[f'p_{c}'] = prob[:, i]
                data.to_csv(dest, index=False)
                r = {'task': task, 'kind': kind, 'plan_sha256': ph,
                     'checkpoint_sha256': sha256(ck), 'manifest_sha256': sha256(csv),
                     'predictions_sha256': sha256(dest), 'rows': len(data),
                     'test_evaluated_after_selection': True, 'precision': 'float32'}
                record_path.write_text(json.dumps(r, indent=2) + '\n')
                print(f'Evaluated {study} {kind}, n={len(data)}', flush=True)
            del net
            torch.cuda.empty_cache()
        (OUTPUT / 'all-evaluated').write_text(ph + '\n')
    except Exception as e:
        (OUTPUT / 'evaluation-error.json').write_text(json.dumps({'error': str(e)}, indent=2))
        raise


if __name__ == '__main__':
    main()
