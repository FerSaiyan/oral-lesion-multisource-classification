"""Read-only checks for the matched BPEX training protocol."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('run_bpex_b7_matched_weighting.py')
spec = importlib.util.spec_from_file_location('matched_b7', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MatchedProtocolTests(unittest.TestCase):
    def test_frozen_rows_and_weight_counts(self):
        plan, study, model, counts = runner.prepare()
        self.assertEqual(sum(counts.values()), 3228)
        self.assertEqual(counts, {'cancer': 262, 'opmd': 1197, 'benign_lesion': 927, 'healthy': 842})
        self.assertEqual(plan['partition_rows']['multisource_test_global.csv'], 1209)
        self.assertTrue(study['require_precomputed_splits'])
        self.assertTrue(study['materialized_images_only'])
        self.assertFalse(study['weighted_sampler'])
        self.assertFalse(study['normalize_sample_weights'])
        self.assertEqual(model['pretrained_source'], 'imagenet')
        self.assertEqual(model['img_size'], 512)
        self.assertEqual(study['finetune']['head_warmup_epochs'], 15)

    def test_stopping_cannot_preempt_full_budget(self):
        plan, study, _, _ = runner.prepare()
        self.assertEqual(study['execution_mode'], 'fixed_single_fit')
        self.assertEqual(study['search_space']['num_epochs_finetune']['value'], 50)
        self.assertGreater(study['search_space']['patience']['value'], 50)
        self.assertEqual(plan['conditions'], ['baseline', 'no_inverse_frequency'])
        self.assertEqual(plan['seeds'], [42, 43, 44, 45, 46])

    def test_changed_configuration_is_rejected(self):
        with patch.object(runner, 'sha256', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                runner.prepare()

    def test_changed_frozen_manifest_is_rejected(self):
        original = runner.sha256

        def altered(path):
            if Path(path).parent.name == 'bpex_revision_v2_frozen_2026-09-24':
                return 'changed'
            return original(path)

        with patch.object(runner, 'sha256', side_effect=altered):
            with self.assertRaisesRegex(ValueError, 'Frozen manifest checksum mismatch'):
                runner.prepare()

    def test_changed_materialized_manifest_is_rejected(self):
        original = runner.sha256

        def altered(path):
            if Path(path).parent.name == 'bpex_revision_v2_materialized_2026-09-24':
                return 'changed'
            return original(path)

        with patch.object(runner, 'sha256', side_effect=altered):
            with self.assertRaisesRegex(ValueError, 'Materialized manifest checksum mismatch'):
                runner.prepare()


if __name__ == '__main__':
    unittest.main()
