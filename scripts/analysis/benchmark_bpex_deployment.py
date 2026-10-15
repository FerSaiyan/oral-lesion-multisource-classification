#!/usr/bin/env python3
"""Real-checkpoint, real-image deployment benchmark with per-image timed observations."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models import enetv2, ViTFineTuner
from src.oral_lesions.data.transforms import build_transforms


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def statistics(values):
    values = np.asarray(values)
    return {'n': len(values), 'mean_ms': float(values.mean()), 'sd_ms': float(values.std(ddof=1)),
            'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--timing-protocol', type=Path, help='Phone protocol defining the shared timing indices')
    parser.add_argument('--output', type=Path, help='Separate output directory for a new benchmark')
    args = parser.parse_args()
    if Path(sys.prefix).name != 'LocalGPT_llama2':
        raise RuntimeError('Use LocalGPT_llama2, not the base environment')
    plan = json.loads((ROOT / 'deployment_plan.json').read_text())
    output = args.output or ROOT / 'results'
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with (ROOT / plan['test_csv']).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != plan['test_rows'] or len({row['filename'] for row in rows}) != len(rows):
        raise ValueError('Frozen test membership differs from the deployment plan')
    for row in rows:
        if sha256(ROOT / row['image_path']) != plan['image_sha256'][row['filename']]:
            raise ValueError('Transferred test image bytes changed')
    indices = list(range(len(rows)))
    if args.timing_protocol:
        protocol = json.loads(args.timing_protocol.read_text())
        indices = protocol['timing_indices']
        if (protocol['test_csv_sha256'] != sha256(ROOT / plan['test_csv'])
                or len(indices) != 120 or len(set(indices)) != 120
                or any(type(index) is not int or not 0 <= index < len(rows) for index in indices)):
            raise ValueError('Timing protocol does not match the frozen test manifest')
    for name in args.models:
        specification = plan['models'][name]
        checkpoint = ROOT / specification['checkpoint']
        reference_path = ROOT / specification['reference_predictions']
        if sha256(checkpoint) != specification['checkpoint_sha256'] or sha256(reference_path) != specification['reference_sha256']:
            raise ValueError('Transferred checkpoint or reference changed')
        with reference_path.open(newline='') as stream:
            reference = list(csv.DictReader(stream))
        if [row['filename'] for row in rows] != [row['filename'] for row in reference]:
            raise ValueError('Reference probabilities do not match frozen test ordering')
        reference_probabilities = np.array([[float(row[f'p_{label}']) for label in plan['classes']] for row in reference])
        reference_probabilities = reference_probabilities[indices]
        if (output / f'{name}_summary.json').exists():
            raise FileExistsError('Refusing to overwrite a completed benchmark')
        config = specification['model_config']
        _, transform = build_transforms(config['model_type'], config['img_size'], resize_mode='preserve')

        def preprocess(row):
            image = cv2.imread(str(ROOT / row['image_path']))
            if image is None:
                raise ValueError('Failed to decode a frozen test image')
            return transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))['image'].unsqueeze(0)

        started = time.perf_counter()
        network = (enetv2(config['enet_backbone'], out_dim=4, load_pretrained_geffnet=False)
                   if config['model_type'] == 'effnet' else ViTFineTuner(config['vit_model_name'], out_dim=4, timm_imagenet_pretrained=False))
        network.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        network = network.to(device).eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        summary = {'model': name, 'checkpoint_sha256': specification['checkpoint_sha256'],
                   'plan_sha256': sha256(ROOT / 'deployment_plan.json'), 'seed': 42,
                   'parameters': sum(parameter.numel() for parameter in network.parameters()),
                   'model_load_seconds': load_seconds, 'model_load_os_cache_controlled': False,
                   'gpu': torch.cuda.get_device_name(args.gpu), 'gpu_index': args.gpu,
                   'python': platform.python_version(), 'torch_cuda': torch.version.cuda,
                   'packages': {package: importlib.metadata.version(package) for package in
                                ('torch', 'timm', 'geffnet', 'albumentations', 'numpy')},
                   'protocol': {key: value for key, value in plan['desktop_protocol'].items() if key != 'end_to_end'},
                   'timing_scope': 'Model execution and host retrieval of four float scores; input already uploaded; excludes softmax',
                   'timing_indices': indices,
                   'test_csv_sha256': sha256(ROOT / plan['test_csv']), 'precisions': {}}
        for precision in plan['desktop_protocol']['precisions']:
            amp = precision == 'amp_fp16'
            warm_input = preprocess(rows[0]).to(device)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                for _ in range(plan['desktop_protocol']['warmups']):
                    network(warm_input).float().cpu().numpy()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            observations, first_logits = [], []
            with torch.inference_mode():
                for repeat in range(plan['desktop_protocol']['passes']):
                    for index in indices:
                        row = rows[index]
                        image = preprocess(row).to(device, non_blocking=True)
                        torch.cuda.synchronize()
                        forward_start = time.perf_counter()
                        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                            logits = network(image)
                        scores = logits.float().cpu().numpy()[0]
                        inference_ms = (time.perf_counter() - forward_start) * 1000
                        if not np.isfinite(scores).all():
                            raise ValueError('Non-finite deployed predictions')
                        observations.append({'pass': repeat + 1, 'index': index, 'filename': row['filename'],
                                             'inference_ms': inference_ms})
                        if repeat == 0:
                            first_logits.append(scores)
                    print(f'{name} {precision} pass {repeat + 1}/3 complete', flush=True)
            logits_array = np.stack(first_logits)
            np.save(output / f'{name}_{precision}_logits.npy', logits_array)
            values = logits_array - logits_array.max(axis=1, keepdims=True)
            probabilities = np.exp(values) / np.exp(values).sum(axis=1, keepdims=True)
            agreement = float((probabilities.argmax(1) == reference_probabilities.argmax(1)).mean())
            with (output / f'{name}_{precision}_latency.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
                writer.writeheader()
                writer.writerows(observations)
            summary['precisions'][precision] = {
                'inference': statistics([item['inference_ms'] for item in observations]),
                'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                'reference_top1_agreement': agreement,
                'max_probability_difference': float(abs(probabilities - reference_probabilities).max()),
                'test_images': len(indices), 'timed_observations': len(observations)}
            if precision == 'fp32' and agreement < 0.99:
                raise ValueError(f'FP32 deployment differs unexpectedly from the selected checkpoint: {agreement}')
            print(json.dumps(summary['precisions'][precision]), flush=True)
        (output / f'{name}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        del network, warm_input, image, logits
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
