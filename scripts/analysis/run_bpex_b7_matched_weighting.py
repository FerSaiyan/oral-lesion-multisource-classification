#!/usr/bin/env python3
"""Run paired full-budget B7 fits on unchanged frozen BPEX partitions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / 'configs/studies/bpex_b7_matched_weighting_2026-10-13.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare():
    from src.exp.config import load_config

    plan = json.loads(PLAN.read_text())
    parameters = json.loads((ROOT / plan['parameter_plan']).read_text())['models'][plan['model']]
    if parameters['parameters'] != plan['shared_parameters']:
        raise ValueError('Shared parameters differ from the frozen validation-selected winner')
    yaml_path = ROOT / parameters['original_yaml']
    if sha256(yaml_path) != parameters['original_yaml_sha256']:
        raise ValueError('Original B7 configuration changed')
    config = load_config(str(yaml_path))
    frozen = json.loads((ROOT / plan['freeze_record']).read_text())
    view = ROOT / plan['materialized_directory']
    materialization = json.loads((view / 'view.json').read_text())
    image_root = Path(materialization['image_root'])
    groups, filenames, train_counts = [], [], None
    for name, expected in plan['partition_rows'].items():
        original = ROOT / plan['frozen_directory'] / name
        derived = view / name
        if not (sha256(original) == frozen['csv_sha256'][name] == plan['frozen_csv_sha256'][name]):
            raise ValueError(f'Frozen manifest checksum mismatch: {name}')
        if not (sha256(derived) == materialization['derived_csv_sha256'][name] == plan['materialized_csv_sha256'][name]):
            raise ValueError(f'Materialized manifest checksum mismatch: {name}')
        with original.open(newline='') as stream:
            original_rows = list(csv.DictReader(stream))
        with derived.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected or len(original_rows) != expected:
            raise ValueError(f'Unexpected partition size: {name}')
        for source, row in zip(original_rows, rows):
            restored = dict(row)
            restored['image_path'] = restored.pop('source_image_path')
            if restored != source:
                raise ValueError(f'Materialization changed frozen row data or ordering: {name}')
            if Path(row['image_path']).resolve() != (image_root / row['filename']).resolve() or not Path(row['image_path']).is_file():
                raise ValueError(f'Unresolved or substituted materialized image: {name}')
        filenames.append({row['filename'] for row in rows})
        groups.append({('INTERNAL' if row['dataset_source'] in ('MY_CAMERA', 'MY_ROMEU') else 'ZENODO', row['group_all'])
                       for row in rows if row['dataset_source'] in ('MY_CAMERA', 'MY_ROMEU', 'ZENODO')})
        if name == 'multisource_train_global.csv':
            train_counts = dict(Counter(row['diagnosis_categories'] for row in rows))
            for row in rows:
                expected_weight = {'opmd': 1.5, 'cancer': 2.0}.get(row['diagnosis_categories'], 1.0)
                expected_weight *= 0.5 if row['dataset_source'] == 'KAGGLE' else 1.0
                if float(row['sample_weight']) != expected_weight:
                    raise ValueError('Frozen sample weights differ from the documented class/source factors')
    for a in range(3):
        for b in range(a):
            if filenames[a] & filenames[b] or groups[a] & groups[b]:
                raise ValueError('Frozen partitions contain image or known-patient overlap')
    study = dict(config['study'])
    study.update({
        'base_data_dir': str(ROOT), 'execution_mode': 'fixed_single_fit',
        'optuna_results_dir': str(ROOT / plan['results_directory']),
        'images_folder': str(image_root), 'require_precomputed_splits': True,
        'materialized_images_only': True, 'dataset_provenance': 'frozen_manifest_sha256',
        'frozen_plan_sha256': sha256(PLAN), 'weighted_sampler': False,
        'normalize_sample_weights': False, 'device': plan['device'],
        'batch_size': plan['batch_size'], 'num_workers': plan['num_workers'],
        'mlflow': {'enabled': False}, 'persist_interim_best': True,
        'pseudolabel': {'enabled': False},
        'train_csv_path': str(view / 'multisource_train_global.csv'),
        'val_csv_path': str(view / 'multisource_val_global.csv'),
        'test_csv_path': str(view / 'multisource_test_global.csv'),
        'search_space': {key: {'type': 'fixed', 'value': value} for key, value in
                         {**plan['shared_parameters'], 'num_epochs_finetune': plan['epochs'], 'patience': plan['patience']}.items()},
    })
    if plan['patience'] <= plan['epochs'] or plan['epochs'] != 50:
        raise ValueError('The matched fits must run all 50 epochs without early stopping')
    return plan, study, dict(config['models'][0]), train_counts


def run_task(condition, seed, plan, study, model, counts):
    import torch
    from scripts.run_optuna_study import run_study
    from src.oral_lesions.engine.trainer import Trainer

    study.update(seed=seed, class_weights=condition == 'baseline',
                 study_name_prefix=f"{plan['experiment']}_{condition}_seed{seed}")
    model['class_weights'] = condition == 'baseline'
    output = ROOT / plan['results_directory'] / f"{study['study_name_prefix']}_{model['name']}"
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite a fit: {output}')
    output.mkdir(parents=True)
    record = {'plan_sha256': sha256(PLAN), 'study': study, 'model': model,
              'training_class_counts': counts, 'python': sys.executable,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
              'code_sha256': {name: sha256(ROOT / name) for name in
                             ('scripts/run_optuna_study.py', 'src/oral_lesions/engine/trainer.py',
                              'scripts/analysis/run_bpex_b7_matched_weighting.py')}}
    (output / 'protocol.json').write_text(json.dumps(record, indent=2) + '\n')
    original_fit = Trainer.fit_mix

    def timed_fit(trainer, **kwargs):
        if kwargs['test_loader'] is not None or trainer.normalize_sample_weights:
            raise ValueError('Unexpected test loader or sample-weight normalization')
        weight = trainer.criterion.weight
        if condition == 'baseline':
            inverse = torch.tensor([1 / counts[label] for label in sorted(counts)], device=weight.device)
            if not torch.allclose(weight, inverse / inverse.sum()):
                raise ValueError('Baseline criterion does not use frozen inverse-frequency weights')
        elif weight is not None:
            raise ValueError('The control still has inverse-frequency criterion weights')
        start_hook, end_hook = kwargs['on_epoch_start'], kwargs['on_epoch_end']
        epoch_started = 0.0
        timing = output / 'epoch_timing.csv'
        with timing.open('w', newline='') as stream:
            csv.writer(stream).writerow(['epoch', 'phase', 'wall_seconds', 'gpu_peak_mib', 'unix_time'])

        def start(epoch, network):
            nonlocal epoch_started
            start_hook(epoch, network)
            if any(parameter.requires_grad for parameter in network.enet.parameters()) != (epoch >= 15):
                raise ValueError('Backbone freeze/unfreeze schedule differs from the matched protocol')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            epoch_started = time.perf_counter()

        def end(epoch, row):
            end_hook(epoch, row)
            torch.cuda.synchronize()
            phase = 'head_warmup' if epoch < study['finetune']['head_warmup_epochs'] else 'full_backbone'
            seconds = time.perf_counter() - epoch_started
            with timing.open('a', newline='') as stream:
                csv.writer(stream).writerow([epoch + 1, phase, seconds, torch.cuda.max_memory_allocated() / 2**20, time.time()])
            print(f'TIMING epoch={epoch + 1} phase={phase} seconds={seconds:.2f}', flush=True)

        kwargs.update(on_epoch_start=start, on_epoch_end=end)
        return original_fit(trainer, **kwargs)

    Trainer.fit_mix = timed_fit
    started = time.time()
    run_study(study, model)
    with (output / 'fixed_fit_log.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((output / 'fixed_fit_summary.json').read_text())
    if [int(row['epoch']) for row in rows] != list(range(1, 51)) or not summary['test_predictions']:
        raise RuntimeError('Fit did not complete exactly 50 epochs and post-selection evaluation')
    selected = max(enumerate(rows), key=lambda pair: (float(pair[1]['val_f1']), pair[0]))[1]
    if int(selected['epoch']) != summary['selected_epoch']:
        raise ValueError('Checkpoint selection differs from the validation-only log')
    for name, key in (('multisource_train_global.csv', 'train_csv_sha256'),
                      ('multisource_val_global.csv', 'val_csv_sha256'),
                      ('multisource_test_global.csv', 'test_csv_sha256')):
        if summary[key] != plan['materialized_csv_sha256'][name]:
            raise ValueError('A manifest changed during fitting')
    (output / 'completion.json').write_text(json.dumps({
        'plan_sha256': sha256(PLAN), 'condition': condition, 'seed': seed,
        'epochs': 50, 'wall_seconds': time.time() - started,
        'test_predictions_sha256': sha256(summary['test_predictions']),
        'checkpoint_sha256': summary['best_model_sha256'],
    }, indent=2) + '\n')


def benchmark(plan, study, model, counts):
    """Time discarded full-backbone train/validation batches before queue launch."""
    import statistics
    import torch
    from torch.utils.data import DataLoader
    from src.utils import set_seed
    from src.oral_lesions.models.factory import create_model
    from src.oral_lesions.data.datasets import MultiSourceDatasetCSV
    from src.oral_lesions.data.transforms import build_transforms

    set_seed(42)
    device = torch.device(study['device'])
    model['dropout_rate'] = plan['shared_parameters']['dropout_rate']
    network = create_model(model, num_classes=4, device=device).to(device).train()
    train_transform, val_transform = build_transforms('effnet', model['img_size'], resize_mode=study['resize_mode'])
    classes = sorted(counts)
    train_data = MultiSourceDatasetCSV(study['train_csv_path'], study['images_folder'], transform=train_transform, classes=classes)
    val_data = MultiSourceDatasetCSV(study['val_csv_path'], study['images_folder'], transform=val_transform, classes=classes)
    loaders = [DataLoader(data, batch_size=plan['batch_size'], num_workers=plan['num_workers'], pin_memory=True)
               for data in (train_data, val_data)]
    weights = torch.tensor([1 / counts[label] for label in classes], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights / weights.sum(), reduction='none')
    optimizer = torch.optim.Adam(network.parameters(), lr=plan['shared_parameters']['finetune_lr'])
    scaler = torch.amp.GradScaler('cuda')
    samples = {'train': [], 'validation': []}
    torch.cuda.reset_peak_memory_stats()
    for phase, loader in zip(samples, loaders):
        network.train(phase == 'train')
        iterator = iter(loader)
        for step in range(15):
            torch.cuda.synchronize()
            started = time.perf_counter()
            images, labels, sample_weights = next(iterator)[:3]
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            sample_weights = sample_weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(phase == 'train'), torch.amp.autocast('cuda'):
                loss = (criterion(network(images), labels) * sample_weights).mean()
            if phase == 'train':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            torch.cuda.synchronize()
            if step >= 3:
                samples[phase].append(time.perf_counter() - started)
        del iterator
    if not optimizer.state or not torch.isfinite(loss):
        raise RuntimeError('Full-backbone GPU benchmark did not produce a finite optimizer step')
    report = {'purpose': 'Discarded train-only preflight, not a scientific fit or test evaluation',
              'plan_sha256': sha256(PLAN), 'gpu': torch.cuda.get_device_name(0),
              'batch_size': plan['batch_size'], 'full_backbone': True,
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
              'batch_seconds': samples,
              'median_train_batch_seconds': statistics.median(samples['train']),
              'median_validation_batch_seconds': statistics.median(samples['validation']),
              'train_batches_per_epoch': len(loaders[0]), 'validation_batches_per_epoch': len(loaders[1])}
    report['projected_full_epoch_seconds'] = (report['median_train_batch_seconds'] * len(loaders[0])
                                             + report['median_validation_batch_seconds'] * len(loaders[1]))
    output = ROOT / plan['results_directory']
    output.mkdir(exist_ok=True)
    with (output / 'gpu_preflight.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--task', help='One condition:seed; otherwise run the ten-task queue')
    parser.add_argument('--benchmark', action='store_true', help='Time discarded full-backbone batches, without launching fits')
    args = parser.parse_args()
    if Path(sys.prefix).name != 'LocalGPT_llama2':
        raise RuntimeError('Use the LocalGPT_llama2 environment')
    # Prevent an inherited custom checkpoint from replacing ImageNet initialization.
    for key in ('CUSTOM_EFFNET_PATH', 'CUSTOM_MODEL_PATH', 'PRETRAINED_MODELS_DIR'):
        os.environ.pop(key, None)
    plan, study, model, counts = prepare()
    print(json.dumps({'experiment': plan['experiment'], 'plan_sha256': sha256(PLAN),
                      'verified_frozen_rows': plan['partition_rows'], 'training_class_counts': counts,
                      'shared_parameters': plan['shared_parameters'], 'tasks': 10}, indent=2), flush=True)
    if args.benchmark:
        benchmark(plan, study, model, counts)
        return
    if not args.execute:
        return
    if args.task:
        condition, seed = args.task.split(':')
        if condition not in plan['conditions'] or int(seed) not in plan['seeds']:
            raise ValueError('Task is outside the frozen ablation plan')
        run_task(condition, int(seed), plan, study, model, counts)
        return
    output = ROOT / plan['results_directory']
    output.mkdir(exist_ok=True)
    with (output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (output / 'launch.json').write_text(json.dumps({'pid': os.getpid(), 'started_unix': time.time(),
                                                     'plan_sha256': sha256(PLAN)}, indent=2) + '\n')
        for seed in plan['seeds']:
            for condition in plan['conditions']:
                task = f'{condition}:{seed}'
                directory = output / f"{plan['experiment']}_{condition}_seed{seed}_{model['name']}"
                marker = directory / 'completion.json'
                if marker.exists():
                    completed = json.loads(marker.read_text())
                    if completed['plan_sha256'] != sha256(PLAN):
                        raise ValueError('Completed task belongs to a different plan')
                    continue
                state = {'task': task, 'status': 'running', 'started_unix': time.time()}
                (output / 'queue_state.json').write_text(json.dumps(state, indent=2) + '\n')
                command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--execute', '--task', task]
                with (output / f'{condition}_seed{seed}.log').open('x') as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        gpu = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu',
                                              '--format=csv,noheader'], capture_output=True, text=True, check=True)
                        with (output / 'gpu_monitor.csv').open('a') as stream:
                            stream.write(f'{task},{gpu.stdout.strip()}\n')
                        time.sleep(30)
                state.update(status='completed' if process.returncode == 0 and marker.exists() else 'failed',
                             finished_unix=time.time(), returncode=process.returncode)
                (output / 'queue_state.json').write_text(json.dumps(state, indent=2) + '\n')
                if state['status'] != 'completed':
                    raise RuntimeError(f'Queue stopped after failed task {task}; inspect its log')
                print(json.dumps(state), flush=True)
        (output / 'queue_state.json').write_text(json.dumps({'status': 'completed', 'tasks': 10}, indent=2) + '\n')


if __name__ == '__main__':
    main()
