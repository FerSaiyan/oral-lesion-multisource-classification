#!/usr/bin/env python3
"""Export corrected BPEX seed-42 models and prepare a persistent-runtime phone fixture."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'results/revision_bpex/deployment_2026-10-13_v1'
PAYLOAD = EXPERIMENT / 'payload'
WORK = Path.home() / 'nvme_tmp/oral-lesions-npu/work/bpex_corrected_20261013_v1'
EXPORT_PYTHON = Path.home() / 'nvme_tmp/oral-lesions-npu/env/bin/python'
QNN_PYTHON = Path.home() / 'nvme_tmp/npu-litert230/env/bin/python'
QAIRT = Path.home() / 'nvme_tmp/npu-litert230/qairt/2.50.0.260828'
ARCHS = {'effnet_b3': 'b3', 'effnet_b7': 'b7', 'vit_base': 'vit_b', 'vit_large': 'vit_l'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if Path(sys.prefix).name != 'LocalGPT_llama2':
        raise RuntimeError('Queue controller requires LocalGPT_llama2; export uses its dedicated existing venv')
    plan = json.loads((PAYLOAD / 'deployment_plan.json').read_text())
    WORK.mkdir()
    persistent = EXPERIMENT / 'mobile'
    persistent.mkdir()
    for folder in ('models', 'aot', 'reports', 'fixture', 'reference'):
        (persistent / folder).mkdir()
    for model, arch in ARCHS.items():
        specification = plan['models'][model]
        checkpoint = PAYLOAD / specification['checkpoint']
        if sha256(checkpoint) != specification['checkpoint_sha256']:
            raise ValueError('Corrected checkpoint checksum mismatch')
        reference = persistent / 'reference' / f'{model}_fp32_logits.npy'
        remote = f"{os.environ['BPEX_REMOTE_HOST']}:{plan['remote_root']}/results/{model}_fp32_logits.npy"
        # Desktop FP32 reference rows are saved after its three timed passes.
        for attempt in range(180):
            copied = subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', remote, str(reference)],
                                    capture_output=True, text=True)
            if copied.returncode == 0:
                break
            time.sleep(20)
        else:
            raise RuntimeError(f'Desktop reference not available for {model}; inspect remote benchmark logs')
        name = f'{model}_bpex_corrected_seed42'
        export_dir = WORK / model / 'models'
        export_dir.mkdir(parents=True)
        export = [str(EXPORT_PYTHON), '-B', '-u', str(ROOT / 'scripts/npu/export_npu_tflite.py'),
                  '--arch', arch, '--ckpt', str(checkpoint), '--name', name,
                  '--inputs', str(PAYLOAD / 'inputs'), '--out', str(export_dir),
                  '--variants', 'fp32', '--threads', '6', '--reference-logits', str(reference)]
        if arch.startswith('vit'):
            export.extend(['--attention', 'rank4'])
        with (persistent / f'{model}_export.log').open('x') as log:
            subprocess.run(export, stdout=log, stderr=subprocess.STDOUT, check=True)
        plain = export_dir / f'{name}_fp32.tflite'
        evaluation = WORK / model / 'evaluation'
        compile_command = [str(QNN_PYTHON), '-B', '-u', str(ROOT / 'scripts/npu/npu_aot_eval.py'), str(plain),
                           '--test', str(PAYLOAD / 'inputs' / f"test_{specification['model_config']['img_size']}.npz"),
                           '--torch-logits', str(reference), '--qairt-root', str(QAIRT),
                           '--cpu-count', '24', '--emu-count', '0', '--out', str(evaluation)]
        with (persistent / f'{model}_compile.log').open('x') as log:
            subprocess.run(compile_command, stdout=log, stderr=subprocess.STDOUT, check=True)
        report_path = evaluation / f'{plain.stem}_npu_eval.json'
        report = json.loads(report_path.read_text())
        if (report['contexts'] != 1 or any(stats['offloaded'] != stats['total'] or stats['partitions'] != 1
                                          for stats in report['partitionStats'])):
            raise ValueError('Corrected NPU compilation did not fully offload to one partition')
        shutil.copy2(plain, persistent / 'models' / f'{model}_float.tflite')
        shutil.copy2(report['aotModel'], persistent / 'aot' / f'{model}_SM8650.tflite')
        shutil.copy2(report_path, persistent / 'reports' / f'{model}_compile.json')
        shutil.copy2(export_dir / f'{name}_export.json', persistent / 'reports' / f'{model}_export.json')
        size = specification['model_config']['img_size']
        fixture = persistent / 'fixture' / str(size)
        if not fixture.exists():
            fixture.mkdir()
            with np.load(PAYLOAD / 'inputs' / f'test_{size}.npz') as inputs:
                for index, image in enumerate(inputs['images']):
                    image.tofile(fixture / f'{index:04d}.rgb')
                (fixture / 'manifest.json').write_text(json.dumps({
                    'n': 1209, 'size': size, 'classes': plan['classes'],
                    'labels': inputs['labels'].tolist(), 'filenames': inputs['filenames'].tolist(),
                    'input_cache_sha256': plan['input_sha256'][f'test_{size}.npz']}, indent=2) + '\n')
        (persistent / f'{model}_ready.json').write_text(json.dumps({
            'checkpoint_sha256': specification['checkpoint_sha256'],
            'plain_sha256': sha256(persistent / 'models' / f'{model}_float.tflite'),
            'aot_sha256': sha256(persistent / 'aot' / f'{model}_SM8650.tflite'),
            'test_rows': 1209, 'cpu_export_checks': 24, 'phone_tested': False}, indent=2) + '\n')
        print(f'READY {model}: plain GPU graph, SM8650 NPU context and frozen-input fixture', flush=True)
    (persistent / 'ready.json').write_text(json.dumps({'models': list(ARCHS), 'phone_tested': False,
                                                      'cleanup_requires_user_approval': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()
