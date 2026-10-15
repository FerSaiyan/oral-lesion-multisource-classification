"""Read-only integrity checks for the completed desktop/mobile deployment bundle."""
import csv
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/revision_bpex/deployment_2026-10-13_v1'


class DeploymentTests(unittest.TestCase):
    def test_publication_contains_timing_only_deployment_evidence(self):
        paper = ROOT / 'papers/Multisource Classification'
        supplement = (paper / 'bpex_supplementary.tex').read_text()
        deployment = supplement.split(r'\section{Trained-model deployment benchmarks}')[1].split(r'\section{Matched EfficientNet')[0]
        for term in ('agreement', 'fidelity', 'file-to-probability', 'calibration'):
            self.assertNotIn(term, deployment.lower())
        self.assertFalse((paper / 'supplement/s_mobile_fidelity.tex').exists())
        for name in ('deployment_benchmark_v1', 'mobile_benchmark_v1', 'deployment_matched120_v2'):
            self.assertFalse((paper / 'supplement/data' / name).exists())
            self.assertTrue((BASE / 'publication_archive_before_runtime_v3' / name).is_dir())

    def test_matched_timing_endpoints_share_exact_subset(self):
        bundle = ROOT / 'papers/Multisource Classification/supplement/data/deployment_runtime120_v3'
        verification = json.loads((bundle / 'verification.json').read_text())
        indices = verification['timing_indices']
        self.assertEqual(len(indices), 120)
        self.assertEqual(len(set(indices)), 120)
        self.assertEqual(verification['total_verified_observations'], 5760)
        expected = [(p, i) for p in (1, 2, 3) for i in indices]
        files = list(bundle.glob('*_latency.csv'))
        self.assertEqual(len(files), 16)
        for path in files:
            with path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ['pass', 'index', 'inference_ms'])
                observations = list(reader)
            self.assertEqual([(int(r['pass']), int(r['index'])) for r in observations], expected)

    def test_frozen_test_and_train_only_calibration(self):
        plan = json.loads((BASE / 'payload/deployment_plan.json').read_text())
        with (BASE / 'payload/test.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1209)
        self.assertFalse({row['filename'] for row in rows} & set(plan['calibration_filenames']))
        self.assertEqual(len(set(plan['calibration_indices'])), 256)
        self.assertTrue(plan['cleanup_requires_user_approval'])
        for size in (224, 384, 512):
            with np.load(BASE / 'payload/inputs' / f'test_{size}.npz') as data:
                self.assertEqual(data['filenames'].tolist(), [row['filename'] for row in rows])
                self.assertEqual(data['classes'].tolist(), plan['classes'])

    def test_all_desktop_endpoints_complete(self):
        verification = json.loads((BASE / 'desktop_verification.json').read_text())
        self.assertEqual(verification['verified_observations'], 29016)
        for model in ('effnet_b3', 'effnet_b7', 'vit_base', 'vit_large'):
            summary = json.loads((BASE / 'desktop' / f'{model}_summary.json').read_text())
            self.assertEqual(summary['seed'], 42)
            self.assertEqual(set(summary['precisions']), {'fp32', 'amp_fp16'})
            for endpoint in summary['precisions'].values():
                self.assertEqual(endpoint['timed_observations'], 3627)
                self.assertEqual(endpoint['test_images'], 1209)
                self.assertGreaterEqual(endpoint['reference_top1_agreement'], 0.997)

    def test_all_corrected_npu_graphs_fully_offloaded(self):
        ready = json.loads((BASE / 'mobile/ready.json').read_text())
        self.assertEqual(len(ready['models']), 4)
        self.assertFalse(ready['phone_tested'])
        for model in ready['models']:
            report = json.loads((BASE / 'mobile/reports' / f'{model}_compile.json').read_text())
            self.assertEqual(report['contexts'], 1)
            self.assertEqual(report['context']['socModel'], 57)
            self.assertEqual(report['context']['dspArch'], 75)
            for stats in report['partitionStats']:
                self.assertEqual(stats['partitions'], 1)
                self.assertEqual(stats['offloaded'], stats['total'])
            self.assertEqual(report['litertCpu']['n'], 24)
            self.assertEqual(report['litertCpu']['agreeTorchTop1'], 1.0)

    def test_apk_contains_isolated_runtime(self):
        apk = ROOT / 'scripts/npu/android_benchmark/build/outputs/apk/debug/BpexDeploymentBenchmark-debug.apk'
        with zipfile.ZipFile(apk) as archive:
            names = set(archive.namelist())
        for library in ('libLiteRt.so', 'liblitert_jni.so', 'libLiteRtDispatch_Qualcomm.so',
                        'libQnnHtp.so', 'libQnnHtpV75Stub.so'):
            self.assertIn(f'lib/arm64-v8a/{library}', names)


if __name__ == '__main__':
    unittest.main()
