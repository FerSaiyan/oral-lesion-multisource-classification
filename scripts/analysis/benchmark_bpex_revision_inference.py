#!/usr/bin/env python3
"""Hardware-specific synthetic-tensor forward timing for reviewer R1.8."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from run_bpex_revision_hpo import ROOT, ORIGINAL_CONFIGS, sha256
from src.exp.config import load_config
from src.oral_lesions.models.factory import create_model

OUT = ROOT / 'results/revision_bpex/inference_benchmark_2026-10-04'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(20261004)
    rows = []
    for model, file in ORIGINAL_CONFIGS.items():
        cfg = load_config(str(ROOT / 'configs/studies' / file))
        mc = dict(cfg['models'][0])
        mc['pretrained_source'] = 'none'
        for k in ('custom_model_path', 'custom_effnet_path', 'custom_backbone_path'):
            mc.pop(k, None)
        net = create_model(mc, num_classes=4, device=torch.device('cuda')).cuda().eval()
        parameters = sum(p.numel() for p in net.parameters())
        for batch in (1, int(cfg['study']['batch_size'])):
            tensor = torch.randn(batch, 3, mc['img_size'], mc['img_size'], device='cuda')
            durations = []
            with torch.inference_mode():
                for _ in range(5): net(tensor)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(30):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    net(tensor)
                    torch.cuda.synchronize()
                    durations.append((time.perf_counter() - start) * 1000)
            rows.append({'model': model, 'parameters': parameters, 'image_size': mc['img_size'],
                         'batch_size': batch, 'precision': 'float32', 'warmup_iterations': 5,
                         'measured_iterations': 30, 'latency_batch_ms_mean': float(np.mean(durations)),
                         'latency_batch_ms_sd': float(np.std(durations, ddof=1)),
                         'latency_batch_ms_median': float(np.median(durations)),
                         'throughput_images_per_second': batch * 1000 / float(np.mean(durations)),
                         'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
                         'config_sha256': sha256(ROOT / 'configs/studies' / file)})
            print(rows[-1], flush=True)
            del tensor
        del net
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT / 'inference_metrics.csv', index=False)
    packages = {}
    for package in ('torch', 'torchvision', 'timm', 'geffnet', 'numpy', 'pandas', 'optuna', 'albumentations'):
        packages[package] = importlib.metadata.version(package)
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                        '--format=csv,noheader'], text=True).strip()
    (OUT / 'benchmark_manifest.json').write_text(json.dumps({
        'gpu': hardware, 'python': platform.python_version(), 'torch_cuda': torch.version.cuda,
        'packages': packages, 'cpu': subprocess.check_output(['lscpu'], text=True),
        'protocol': 'eval mode, random weights and synthetic RGB tensors already resident on GPU; '
                    'FP32, no autocast, 5 warmups and 30 synchronized wall-clock measurements',
        'excludes': ['Image decoding/preprocessing', 'Disk or network transfer', 'Model loading'],
        'interpretation': 'Local RTX 4060 Ti compute benchmark; not original-search hardware or end-to-end clinical latency',
        'metrics_sha256': sha256(OUT / 'inference_metrics.csv'),
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
