#!/usr/bin/env python3
"""Summarize matched-fit progress and estimate runtime from observed batches/epochs."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch-seconds', type=int, default=0)
    args = parser.parse_args()
    plan = json.loads((ROOT / 'configs/studies/bpex_b7_matched_weighting_2026-10-13.json').read_text())
    directory = ROOT / plan['results_directory']
    deadline = time.monotonic() + args.watch_seconds
    while True:
        preflight = json.loads((directory / 'gpu_preflight.json').read_text())
        state = json.loads((directory / 'queue_state.json').read_text())
        timings = []
        completed = 0
        for seed in plan['seeds']:
            for condition in plan['conditions']:
                output = directory / f"{plan['experiment']}_{condition}_seed{seed}_effnet_b7_imagenet"
                completed += (output / 'completion.json').exists()
                timing = output / 'epoch_timing.csv'
                if timing.exists():
                    with timing.open(newline='') as stream:
                        timings.extend(csv.DictReader(stream))
        head = [float(row['wall_seconds']) for row in timings if row['phase'] == 'head_warmup']
        full = [float(row['wall_seconds']) for row in timings if row['phase'] == 'full_backbone']
        full_seconds = statistics.median(full) if full else preflight['projected_full_epoch_seconds']
        report = {'measured_unix': time.time(), 'queue': state, 'completed_fits': completed,
                  'completed_epochs': len(timings), 'observed_head_epochs': len(head),
                  'observed_full_epochs': len(full),
                  'median_head_epoch_seconds': statistics.median(head) if head else None,
                  'full_epoch_seconds': full_seconds,
                  'full_epoch_timing_basis': 'actual full epochs' if full else '15 real-data preflight batches per phase, first 3 discarded',
                  'fit_budget': 10, 'epoch_budget': 500,
                  'assumed_startup_checkpoint_and_test_overhead_seconds_per_fit': 120,
                  'estimate_type': 'Scheduling estimate, not a statistical confidence interval'}
        if head:
            head_seconds = statistics.median(head)
            total = 10 * (15 * head_seconds + 35 * full_seconds + 120)
            remaining = (150 - len(head)) * head_seconds + (350 - len(full)) * full_seconds + (10 - completed) * 120
            report.update(estimated_total_hours=total / 3600,
                          estimated_remaining_hours=remaining / 3600,
                          scheduling_range_total_hours=[total * 0.9 / 3600, total * 1.2 / 3600])
        gpu_log = directory / 'gpu_monitor.csv'
        if gpu_log.exists():
            lines = gpu_log.read_text().splitlines()
            report['recent_gpu_samples'] = lines[-4:]
        staging = directory / 'runtime_estimate.json.tmp'
        staging.write_text(json.dumps(report, indent=2) + '\n')
        staging.replace(directory / 'runtime_estimate.json')
        print(json.dumps(report), flush=True)
        if time.monotonic() >= deadline:
            break
        time.sleep(min(30, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
