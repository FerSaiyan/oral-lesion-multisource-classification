"""Check that CPU initialization is not confused with actual graph fallback."""
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'npu/run_bpex_fold6.py'
SPEC = importlib.util.spec_from_file_location('fold6', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DelegationTests(unittest.TestCase):
    def test_subset_is_reproducible_stratified_and_unique(self):
        rows = [{'dataset_source': source, 'diagnosis_categories': label}
                for source in ('internal', 'external') for label in ('cancer', 'healthy') for _ in range(10)]
        selected = MODULE.timing_subset(rows, 12)
        self.assertEqual(selected, MODULE.timing_subset(rows, 12))
        self.assertEqual(len(set(selected)), 12)
        self.assertEqual({(rows[i]['dataset_source'], rows[i]['diagnosis_categories']) for i in selected},
                         {('internal', 'cancer'), ('internal', 'healthy'), ('external', 'cancer'), ('external', 'healthy')})

    def test_subset_handles_empty_and_full_selection(self):
        rows = [{'dataset_source': 'internal', 'diagnosis_categories': 'cancer'}] * 3
        self.assertEqual(MODULE.timing_subset(rows, 0), [])
        self.assertEqual(set(MODULE.timing_subset(rows, 3)), {0, 1, 2})
        with self.assertRaises(ValueError):
            MODULE.timing_subset(rows, 4)

    def test_full_npu_evidence(self):
        log = '\n'.join([
            'Replacing 1 out of 1 node(s) with delegate (DispatchDelegate)',
            'Creating new QNN context for bytecode 0x123',
            'remote_handle64_open: opened handle 0x123 for file:///libQnnHtpV75Skel.so?invoke&_dom=cdsp',
            'Created TensorFlow Lite XNNPACK delegate for CPU.',
        ])
        evidence = MODULE.npu_delegation(log)
        self.assertTrue(evidence['full_dispatch_partition'])
        self.assertTrue(evidence['qnn_context_loaded'])
        self.assertTrue(evidence['hexagon_rpc_opened'])
        self.assertFalse(evidence['cpu_graph_delegation_detected'])

    def test_cpu_graph_fallback(self):
        evidence = MODULE.npu_delegation('Replacing 4 out of 4 node(s) with delegate (XNNPACK)')
        self.assertTrue(evidence['cpu_graph_delegation_detected'])
        self.assertFalse(evidence['full_dispatch_partition'])

    def test_registration_alone_is_not_execution(self):
        evidence = MODULE.npu_delegation('NPU accelerator registered.\nCreated TensorFlow Lite XNNPACK delegate for CPU.')
        self.assertFalse(any(evidence.values()))


if __name__ == '__main__':
    unittest.main()
