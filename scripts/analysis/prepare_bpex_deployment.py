#!/usr/bin/env python3
"""Stage corrected checkpoints, frozen test images and train-only calibration inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

import albumentations as A
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_bpex_b7_matched_weighting import prepare, sha256
from src.exp.config import load_config
from src.oral_lesions.data.transforms import _resize_block

OUT = ROOT / 'results/revision_bpex/deployment_2026-10-13_v1'
REMOTE = '/tmp/opencode/bpex-deployment-20261013-v1'
CONFIGS = {'effnet_b3': 'effnet_imagenet_b3_ar_multisource.yaml',
           'effnet_b7': 'effnet_imagenet_ar.yaml', 'vit_base': 'vit_base_imagenet_ar.yaml',
           'vit_large': 'vit_large_imagenet_ar.yaml'}


def archive_payload(payload):
    plan = json.loads((payload / 'deployment_plan.json').read_text())
    for model in plan['models'].values():
        if sha256(payload / model['checkpoint']) != model['checkpoint_sha256'] or sha256(payload / model['reference_predictions']) != model['reference_sha256']:
            raise ValueError('Staged checkpoint/reference checksum mismatch')
    for name, digest in plan['image_sha256'].items():
        if sha256(payload / 'images' / name) != digest:
            raise ValueError('Staged image checksum mismatch')
    for name, digest in plan['input_sha256'].items():
        if sha256(payload / 'inputs' / name) != digest:
            raise ValueError('Staged NPU input checksum mismatch')
    staging = OUT / 'luciano_payload.tar.tmp'
    with tarfile.open(staging, 'w') as archive:
        for source in sorted(payload.rglob('*')):
            relative = source.relative_to(payload)
            if source.is_file() and relative.parts[0] != 'inputs':
                archive.add(source, arcname=str(relative))
    staging.replace(OUT / 'luciano_payload.tar')
    (OUT / 'archive.json').write_text(json.dumps({'archive_sha256': sha256(OUT / 'luciano_payload.tar'),
                                                'plan_sha256': sha256(payload / 'deployment_plan.json')}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-only', action='store_true', help='Verify a staged payload and finish an interrupted archive')
    parser.add_argument('--resume', action='store_true', help='Verify and reuse this experiment\'s partially staged files')
    args = parser.parse_args()
    if Path(sys.prefix).name != 'LocalGPT_llama2':
        raise RuntimeError('Use LocalGPT_llama2')
    frozen_plan, study, _, _ = prepare()
    if args.archive_only:
        archive_payload(OUT / 'payload')
        print(f'Verified and archived {OUT}', flush=True)
        return
    if OUT.exists() and not args.resume:
        raise FileExistsError(f'Refusing to replace an existing deployment bundle: {OUT}')
    OUT.mkdir(exist_ok=args.resume)
    payload = OUT / 'payload'
    for name in ('weights', 'images', 'reference', 'inputs'):
        (payload / name).mkdir(parents=True, exist_ok=args.resume)
    with Path(study['test_csv_path']).open(newline='') as stream:
        test = list(csv.DictReader(stream))
    with Path(study['train_csv_path']).open(newline='') as stream:
        train = list(csv.DictReader(stream))
    classes = sorted({row['diagnosis_categories'] for row in train})
    rng = np.random.default_rng(20261013)
    groups = {}
    for index, row in enumerate(train):
        groups.setdefault((row['dataset_source'], row['diagnosis_categories']), []).append(index)
    calibration = [int(rng.choice(indices)) for indices in groups.values()]
    remaining = np.array([index for index in range(len(train)) if index not in calibration])
    calibration.extend(rng.choice(remaining, 256 - len(calibration), replace=False).tolist())
    calibration.sort()
    with (ROOT / 'results/revision_bpex/tuned_seedwise_test_analysis_v1/run_provenance.csv').open(newline='') as stream:
        provenance = {row['model']: row for row in csv.DictReader(stream) if int(row['seed']) == 42}
    models = {}
    for model, config in CONFIGS.items():
        source = provenance[model]
        checkpoint = ROOT / 'results/revision_bpex/hpo_corrected_global' / source['source_study'] / 'best_model_retrain.pth'
        predictions = ROOT / 'results/revision_bpex/tuned_seedwise_test_v1' / source['source_study'] / 'test_predictions.csv'
        if sha256(checkpoint) != source['checkpoint_sha256'] or sha256(predictions) != source['predictions_sha256']:
            raise ValueError(f'Corrected seed-42 provenance mismatch: {model}')
        config_path = ROOT / 'configs/studies' / config
        model_config = dict(load_config(str(config_path))['models'][0])
        for source_path, target in ((checkpoint, payload / 'weights' / f'{model}.pth'),
                                    (predictions, payload / 'reference' / f'{model}_predictions.csv')):
            if target.exists():
                if sha256(target) != sha256(source_path):
                    raise ValueError('Partial staging has a changed checkpoint/reference')
            else:
                shutil.copy2(source_path, target)
        models[model] = {'seed': 42, 'model_config': model_config,
                         'checkpoint': f'weights/{model}.pth', 'checkpoint_sha256': sha256(checkpoint),
                         'reference_predictions': f'reference/{model}_predictions.csv',
                         'reference_sha256': sha256(predictions), 'source_study': source['source_study'],
                         'selected_epoch': int(source['selected_epoch']), 'original_config_sha256': sha256(config_path)}
    relocated = []
    image_hashes = {}
    for row in test:
        if Path(row['filename']).name != row['filename']:
            raise ValueError('Unsafe test filename')
        source = Path(row['image_path'])
        target = payload / 'images' / row['filename']
        if target.exists():
            if sha256(target) != sha256(source):
                raise ValueError('Partial staging has a changed image')
        else:
            shutil.copy2(source, target)
        image_hashes[row['filename']] = sha256(source)
        relocated.append({**row, 'image_path': f'images/{row["filename"]}'})
    with (payload / 'test.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test[0]))
        writer.writeheader()
        writer.writerows(relocated)
    for size in (224, 384, 512):
        resize = A.Compose(_resize_block(size, 'preserve'))
        for split, rows in (('test', test), ('train', [train[index] for index in calibration])):
            path = payload / 'inputs' / f'{split}_{size}.npz'
            if path.exists():
                with np.load(path) as cached:
                    if (cached['filenames'].tolist() != [row['filename'] for row in rows]
                            or cached['classes'].tolist() != classes
                            or cached['images'].shape != (len(rows), size, size, 3)):
                        raise ValueError('Partial input cache has different membership or shape')
                print(f'Verified cached {split}: {len(rows)} images at {size}px', flush=True)
                continue
            images = []
            for row in rows:
                image = cv2.imread(row['image_path'])
                if image is None:
                    raise ValueError(f'Unreadable frozen {split} image')
                images.append(resize(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))['image'])
            np.savez(path, images=np.stack(images), labels=np.array([classes.index(row['diagnosis_categories']) for row in rows]),
                     classes=np.array(classes), filenames=np.array([row['filename'] for row in rows]),
                     sources=np.array([row['dataset_source'] for row in rows]))
            print(f'Cached {split}: {len(rows)} images at {size}px', flush=True)
    plan = {'experiment': 'bpex_deployment_2026-10-13_v1', 'remote_root': REMOTE,
            'cleanup_requires_user_approval': True, 'models': models, 'classes': classes,
            'test_csv': 'test.csv', 'test_rows': 1209,
            'frozen_csv_sha256': frozen_plan['frozen_csv_sha256'],
            'materialized_csv_sha256': frozen_plan['materialized_csv_sha256'],
            'image_sha256': image_hashes,
            'input_sha256': {path.name: sha256(path) for path in (payload / 'inputs').glob('*.npz')},
            'calibration_rows': 256, 'calibration_source': 'Frozen training partition only',
            'calibration_indices': calibration,
            'calibration_filenames': [train[index]['filename'] for index in calibration],
            'desktop_protocol': {'warmups': 30, 'passes': 3, 'batch_size': 1,
                                 'precisions': ['fp32', 'amp_fp16'], 'tf32': False,
                                 'end_to_end': 'Local image decode through CPU output probabilities; excludes model load, camera, UI and network'},
            'phone_protocol': {'warmups': 30, 'passes': 3, 'batch_size': 1,
                               'backends': ['GPU', 'NPU'], 'soak_seconds': 90}}
    (payload / 'deployment_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    files = [ROOT / 'src/__init__.py', ROOT / 'src/models.py', ROOT / 'src/utils.py']
    files += list((ROOT / 'src/oral_lesions').rglob('*.py'))
    files += [ROOT / 'scripts/analysis/benchmark_bpex_deployment.py']
    for source in files:
        target = payload / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Desktop workers need no calibration arrays; those stay local for export.
    archive_payload(payload)
    print(f'Prepared {OUT}; {len(test)} exact test images; 256 train-only calibration images', flush=True)


if __name__ == '__main__':
    main()
