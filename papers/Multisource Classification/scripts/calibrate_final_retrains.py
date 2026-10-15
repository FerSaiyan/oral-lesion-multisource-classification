#!/usr/bin/env python3
"""Validation-only GPU temperature scaling for the 80 corrected tuned retrains."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

from analyze_calibration import CONFIGS, FROZEN, VIEW, reliability, sha
from build_revision_evidence import CLASSES, DISPLAY, MODELS, PAPER, RESULTS, ROOT, SEEDS, fmt, table

sys.path.insert(0, str(ROOT))
from src.oral_lesions.data import build_transforms
from src.oral_lesions.models.factory import create_model
from scripts.analysis.analyze_bpex_final_seeds import cluster_map, bootstrap_counts, interval

OUTPUT = RESULTS / 'temperature_calibration_v1'


class StrictImages(Dataset):
    def __init__(self, frame, transform):
        self.frame, self.transform = frame, transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = cv2.imread(str(row.image_path))
        if image is None:
            raise RuntimeError(f'Unreadable frozen validation image: {row.image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        return image, CLASSES.index(row.diagnosis_categories), index


def gpu_status():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                             '--format=csv,noheader'], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def apply_temperature(prob, temperature, device):
    assert np.isfinite(prob).all() and (prob > 0).all() and temperature > 0
    value = torch.as_tensor(prob, dtype=torch.float64, device=device)
    return torch.softmax(value.log() / temperature, dim=1).cpu().numpy()


def score_metrics(prob, y):
    pred = prob.argmax(axis=1)
    truth = np.eye(4)[y]
    confidence = prob.max(axis=1)
    by = np.isin(y, (1, 3)).astype(int)
    score = np.clip(prob[:, 1] + prob[:, 3], 0, 1)
    tn, fp, fn, tp = confusion_matrix(by, score >= .5, labels=(0, 1)).ravel()
    def ratio(a, b):
        return a / b if b else np.nan
    result = {
        'accuracy': np.mean(pred == y),
        'weighted_f1': f1_score(y, pred, average='weighted', zero_division=0),
        'macro_f1_present': f1_score(y, pred, labels=np.unique(y), average='macro', zero_division=0),
        'multiclass_brier': np.mean(np.sum((prob - truth)**2, axis=1)),
        'multiclass_nll': -np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-15, 1))),
        'top1_ece': reliability(confidence, pred == y, 10)[0],
        'mean_confidence': confidence.mean(),
        'screening_brier': np.mean((score - by)**2),
        'screening_nll': -np.mean(by * np.log(np.clip(score, 1e-15, 1)) + (1 - by) * np.log(np.clip(1 - score, 1e-15, 1))),
        'screening_ece': reliability(score, by, 10)[0],
        'screening_f1': ratio(2 * tp, 2 * tp + fp + fn),
        'sensitivity': ratio(tp, tp + fn), 'specificity': ratio(tn, tn + fp),
        'ppv': ratio(tp, tp + fp), 'npv': ratio(tn, tn + fn),
        'screening_auc': roc_auc_score(by, score),
        'average_precision': average_precision_score(by, score),
        'screening_tn': tn, 'screening_fp': fp, 'screening_fn': fn, 'screening_tp': tp,
    }
    for i, c in enumerate(CLASSES):
        result[c + '_f1'] = f1_score(y == i, pred == i, zero_division=0)
        result[c + '_mean_probability'] = np.mean(prob[:, i])
        result[c + '_prevalence'] = np.mean(y == i)
        result[c + '_brier'] = np.mean((prob[:, i] - (y == i))**2)
        result[c + '_ece'] = reliability(prob[:, i], y == i, 10)[0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('validation', 'fit', 'evaluate', 'all'), default='all')
    parser.add_argument('--endpoint', choices=('pooled', 'kaggle', 'zenodo', 'mendeley'))
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'A CUDA GPU is required for this requested calibration run.'
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUTPUT.mkdir(exist_ok=True)
    audit_path = RESULTS / 'calibration_diagnostics_v1/run_audit.csv'
    audit = pd.read_csv(audit_path)
    assert len(audit) == 80 and audit.retrain_and_freeze_verified.all()
    protocol = {
        'method': 'one positive scalar temperature per final retrain',
        'fit_objective': 'unweighted multiclass NLL on the matching corrected validation set',
        'optimizer': 'CUDA float64 LBFGS on log temperature; initial T=1, max_iter=100, strong_wolfe',
        'evaluation_threshold': .5, 'temperature_selection_on_test': False,
        'validation_reused_for_hpo_and_epoch_selection': True,
        'interpretation': 'exploratory post-hoc calibration; validation is not an untouched calibration cohort',
        'class_order': CLASSES, 'seeds': SEEDS, 'n_models': 80,
        'audit_sha256': sha(audit_path),
        'development_policy': 'source exclusion retained; no test/holdout labels used in calibration fitting',
        'inference_precision': 'FP32, no AMP or TF32',
        'heldout_application': 'softmax(log(saved positive raw probability)/T), CUDA float64',
    }
    protocol_path = OUTPUT / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == json.loads(json.dumps(protocol))
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    protocol_hash = sha(protocol_path)
    derived_hashes = json.loads((VIEW / 'view.json').read_text())['derived_csv_sha256']
    frozen_hashes = json.loads((ROOT / 'configs/studies/bpex_revision_v2_frozen_2026-09-24.json').read_text())['csv_sha256']
    jobs = audit.sort_values(['endpoint', 'model', 'seed'])
    if args.endpoint:
        jobs = jobs[jobs.endpoint == args.endpoint]
    if args.phase in ('validation', 'all'):
        for run in jobs.itertuples():
            began = time.monotonic()
            directory = OUTPUT / run.study
            directory.mkdir(exist_ok=True)
            checkpoint = ROOT / run.checkpoint
            assert checkpoint.name == 'best_model_retrain.pth' and sha(checkpoint) == run.checkpoint_sha256
            cfg_path = ROOT / 'configs/studies' / CONFIGS[run.model]
            assert sha(cfg_path) == run.yaml_sha256
            cfg = yaml.safe_load(cfg_path.read_text())
            train_file = 'multisource_train_global.csv' if run.endpoint == 'pooled' else f'multisource_train_no_{run.endpoint}.csv'
            val_file = 'multisource_val_global.csv' if run.endpoint == 'pooled' else f'multisource_val_no_{run.endpoint}.csv'
            for filename in (train_file, val_file):
                assert sha(VIEW / filename) == derived_hashes[filename]
                assert sha(FROZEN / filename) == frozen_hashes[filename]
            train = pd.read_csv(VIEW / train_file).fillna({'patient_id': ''})
            frame = pd.read_csv(VIEW / val_file).fillna({'patient_id': ''})
            frozen = pd.read_csv(FROZEN / val_file).fillna({'patient_id': ''})
            identity = ['filename', 'dataset_source', 'patient_id', 'diagnosis_categories']
            assert frame[identity].equals(frozen[identity])
            assert set(frame.filename).isdisjoint(train.filename)
            for sources in (('MY_CAMERA', 'MY_ROMEU'), ('ZENODO',)):
                a = set(train.loc[train.dataset_source.isin(sources), 'patient_id'])
                b = set(frame.loc[frame.dataset_source.isin(sources), 'patient_id'])
                assert '' not in a | b and a.isdisjoint(b)
            if run.endpoint != 'pooled':
                assert run.endpoint.upper() not in set(train.dataset_source) | set(frame.dataset_source)
            assert sorted(set(train.diagnosis_categories) | set(frame.diagnosis_categories)) == list(CLASSES)
            destination = directory / 'validation_logits.npz'
            receipt_path = directory / 'validation_inference.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt['checkpoint_sha256'] == run.checkpoint_sha256 and receipt['yaml_sha256'] == run.yaml_sha256
                assert receipt['validation_manifest_sha256'] == derived_hashes[val_file]
                assert sha(destination) == receipt['logits_sha256']
                print(f'Validated cached logits: {run.endpoint} {run.model} seed {run.seed}', flush=True)
                continue
            assert not destination.exists(), 'Unreceipted validation output requires review.'
            mc = dict(cfg['models'][0])
            mc['pretrained_source'] = 'none'
            study_results = pd.read_csv(checkpoint.parent / f'{run.study}_optuna_study_results.csv')
            winner = study_results[study_results.state == 'COMPLETE'].sort_values('value', ascending=False).iloc[0]
            mc['dropout_rate'] = float(winner.params_dropout_rate)
            resize_mode = mc.get('resize_mode', cfg['study']['resize_mode'])
            assert resize_mode == 'preserve'
            _, transform = build_transforms(mc['model_type'], int(mc['img_size']), resize_mode=resize_mode)
            dataset = StrictImages(frame, transform)
            loader = DataLoader(dataset, batch_size=int(cfg['study']['batch_size']), num_workers=4,
                                shuffle=False, pin_memory=True)
            torch.cuda.reset_peak_memory_stats()
            model = create_model(mc, num_classes=4, device=device).to(device)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
            model.eval()
            logits, labels, indices = [], [], []
            with torch.inference_mode():
                for batch_index, (images, target, index) in enumerate(loader):
                    logits.append(model(images.to(device, non_blocking=True)).cpu())
                    labels.append(target)
                    indices.append(index)
                    if batch_index % 40 == 0:
                        print(f'Validation {run.endpoint} {run.model} seed {run.seed}, batch {batch_index}; GPU {gpu_status()}', flush=True)
            z = torch.cat(logits).numpy()
            y = torch.cat(labels).numpy()
            assert z.shape == (len(frame), 4) and np.isfinite(z).all()
            assert np.array_equal(torch.cat(indices).numpy(), np.arange(len(frame)))
            assert np.array_equal(y, np.array([CLASSES.index(c) for c in frame.diagnosis_categories]))
            with destination.open('wb') as stream:
                np.savez_compressed(stream, logits=z, y=y, filenames=frame.filename.to_numpy(dtype=str))
            receipt = {'endpoint': run.endpoint, 'model': run.model, 'seed': int(run.seed),
                'checkpoint_sha256': run.checkpoint_sha256, 'yaml_sha256': run.yaml_sha256,
                'validation_manifest_sha256': derived_hashes[val_file], 'logits_sha256': sha(destination),
                'rows': len(frame), 'img_size': int(mc['img_size']), 'resize_mode': resize_mode,
                'class_order': CLASSES, 'precision': 'FP32', 'device': torch.cuda.get_device_name(0),
                'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.monotonic() - began,
                'strict_image_loading': True, 'protocol_sha256': protocol_hash}
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
            print(f'Finished validation: {run.endpoint} {run.model} seed {run.seed}, n={len(frame)}, {receipt["seconds"]:.1f}s', flush=True)
            del model, loader, dataset, logits, labels, indices
            gc.collect()
            torch.cuda.empty_cache()
    if args.phase in ('fit', 'all'):
        for run in jobs.itertuples():
            directory = OUTPUT / run.study
            logits_path = directory / 'validation_logits.npz'
            receipt = json.loads((directory / 'validation_inference.json').read_text())
            assert sha(logits_path) == receipt['logits_sha256'] and receipt['checkpoint_sha256'] == run.checkpoint_sha256
            parameter_path = directory / 'temperature.json'
            if parameter_path.exists():
                existing = json.loads(parameter_path.read_text())
                assert existing['validation_logits_sha256'] == receipt['logits_sha256'] and existing['protocol_sha256'] == protocol_hash
                continue
            values = np.load(logits_path)
            z = torch.tensor(values['logits'], dtype=torch.float64, device=device)
            y = torch.tensor(values['y'], dtype=torch.long, device=device)
            log_t = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
            optimizer = torch.optim.LBFGS([log_t], lr=1., max_iter=100, tolerance_grad=1e-10,
                                         tolerance_change=1e-12, line_search_fn='strong_wolfe')
            before = float(F.cross_entropy(z, y))
            def closure():
                optimizer.zero_grad()
                loss = F.cross_entropy(z / log_t.exp(), y)
                loss.backward()
                return loss
            optimizer.step(closure)
            temperature = float(log_t.detach().exp())
            after = float(F.cross_entropy(z / temperature, y))
            assert np.isfinite(temperature) and temperature > 0 and after <= before + 1e-9
            params = {'endpoint': run.endpoint, 'model': run.model, 'seed': int(run.seed),
                      'temperature': temperature, 'validation_nll_raw': before, 'validation_nll_calibrated': after,
                      'validation_rows': int(len(y)), 'validation_logits_sha256': receipt['logits_sha256'],
                      'checkpoint_sha256': run.checkpoint_sha256, 'protocol_sha256': protocol_hash,
                      'fit_device': torch.cuda.get_device_name(0), 'fit_labels': 'corrected validation only'}
            parameter_path.write_text(json.dumps(params, indent=2) + '\n')
            print(f'Fit T={temperature:.4f}: {run.endpoint} {run.model} seed {run.seed}; validation NLL {before:.4f}->{after:.4f}', flush=True)
            del z, y, optimizer, log_t
            torch.cuda.empty_cache()
    if args.phase not in ('evaluate', 'all'):
        return
    # Freeze every temperature before opening evaluation labels or computing held-out metrics.
    assert all((OUTPUT / r.study / 'temperature.json').is_file() for r in audit.itertuples()), 'All 80 temperatures must be fitted first.'
    temperatures = [json.loads((OUTPUT / r.study / 'temperature.json').read_text()) for r in audit.itertuples()]
    pd.DataFrame(temperatures).to_csv(OUTPUT / 'temperatures.csv', index=False)
    parameter_hashes = {r.study: sha(OUTPUT / r.study / 'temperature.json') for r in audit.itertuples()}
    rows, class_rows, matrix_rows, bin_rows, provenance = [], [], [], [], []
    bootstrap_cache, paired_draws = {}, {}
    for run, params in zip(audit.itertuples(), temperatures):
        assert params['checkpoint_sha256'] == run.checkpoint_sha256 and params['protocol_sha256'] == protocol_hash
        directory = OUTPUT / run.study
        pooled = run.endpoint == 'pooled'
        original = RESULTS / ('tuned_seedwise_test_v1' if pooled else 'sourcewise_heldout_v1') / run.study / ('test_predictions.csv' if pooled else 'holdout_predictions.csv')
        assert sha(original) == run.predictions_sha256
        frame = pd.read_csv(original).fillna({'patient_id': ''})
        reference = pd.read_csv(VIEW / run.corrected_evaluation_manifest).fillna({'patient_id': ''})
        assert sha(VIEW / run.corrected_evaluation_manifest) == derived_hashes[run.corrected_evaluation_manifest]
        assert frame[['filename', 'diagnosis_categories', 'dataset_source', 'patient_id']].equals(reference[['filename', 'diagnosis_categories', 'dataset_source', 'patient_id']])
        raw = frame[['p_' + c for c in CLASSES]].to_numpy(float)
        calibrated = apply_temperature(raw, params['temperature'], device)
        assert np.allclose(calibrated.sum(axis=1), 1, atol=1e-12)
        assert np.array_equal(np.argsort(raw, axis=1), np.argsort(calibrated, axis=1)), 'Temperature must preserve every four-class ranking.'
        y = np.array([CLASSES.index(c) for c in frame.y_true])
        key = {'endpoint': run.endpoint, 'model': run.model, 'seed': int(run.seed), 'n_images': len(y)}
        if run.endpoint not in bootstrap_cache:
            ci_index, strata, _ = cluster_map(frame)
            ci_counts = bootstrap_counts({s: ids for s, ids in strata.items() if ids}, 10000, 20261007)
            bootstrap_cache[run.endpoint] = (ci_index, ci_counts)
        ci_index, ci_counts = bootstrap_cache[run.endpoint]
        truth = np.eye(4)[y]
        binary_y = np.isin(y, (1, 3))
        changes = np.column_stack((
            -np.log(np.clip(calibrated[np.arange(len(y)), y], 1e-15, 1)) + np.log(np.clip(raw[np.arange(len(y)), y], 1e-15, 1)),
            np.sum((calibrated - truth)**2 - (raw - truth)**2, axis=1),
            (calibrated[:, 1] + calibrated[:, 3] - binary_y)**2 - (raw[:, 1] + raw[:, 3] - binary_y)**2))
        aggregates = np.column_stack([np.bincount(ci_index, minlength=ci_counts.shape[1]),
            *[np.bincount(ci_index, weights=changes[:, i], minlength=ci_counts.shape[1]) for i in range(3)]])
        resampled = ci_counts @ aggregates
        delta_draws = resampled[:, 1:] / resampled[:, :1]
        paired_draws.setdefault((run.endpoint, run.model), []).append(delta_draws)
        for kind, prob in (('raw', raw), ('calibrated', calibrated)):
            metrics = score_metrics(prob, y)
            rows.append({**key, 'kind': kind, **metrics})
            for c in CLASSES:
                class_rows.append({**key, 'kind': kind, 'class': c, 'support': int(np.sum(y == CLASSES.index(c))), 'f1': metrics[c + '_f1']})
            score = prob[:, 1] + prob[:, 3]
            cm = confusion_matrix(np.isin(y, (1, 3)), score >= .5, labels=(0, 1))
            for i in range(2):
                for j in range(2):
                    matrix_rows.append({**key, 'kind': kind, 'true': i, 'predicted': j, 'count': int(cm[i, j])})
            for task, scores, outcomes in [('top1', prob.max(axis=1), prob.argmax(axis=1) == y), ('screening', score, np.isin(y, (1, 3)))]:
                _, counts, means, events = reliability(scores, outcomes, 10)
                bin_rows.extend({**key, 'kind': kind, 'task': task, 'bin': b, 'n_bin': int(counts[b]),
                                 'mean_score': means[b], 'event_fraction': events[b]} for b in range(10))
        for i, c in enumerate(CLASSES):
            frame['p_' + c] = calibrated[:, i]
        frame['y_pred'] = np.asarray(CLASSES)[calibrated.argmax(axis=1)]
        dest = directory / 'calibrated_evaluation_predictions.csv'
        frame.to_csv(dest, index=False)
        record = {**key, 'source_predictions_sha256': run.predictions_sha256,
                  'checkpoint_sha256': run.checkpoint_sha256, 'temperature_sha256': parameter_hashes[run.study],
                  'temperature': params['temperature'], 'calibrated_predictions_sha256': sha(dest),
                  'four_class_rankings_unchanged': True, 'protocol_sha256': protocol_hash}
        (directory / 'calibrated_evaluation_provenance.json').write_text(json.dumps(record, indent=2) + '\n')
        provenance.append(record)
    for filename, values in [('per_seed_metrics', rows), ('per_class_f1', class_rows),
                             ('binary_confusion_matrices', matrix_rows), ('reliability_bins', bin_rows), ('evaluation_provenance', provenance)]:
        pd.DataFrame(values).to_csv(OUTPUT / (filename + '.csv'), index=False)
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(['endpoint', 'model', 'kind']).agg(['mean', 'std'])
    summary.to_csv(OUTPUT / 'model_summary.csv')
    interval_rows = []
    for (endpoint, model), draws in paired_draws.items():
        assert len(draws) == 5
        mean_draws = np.mean(draws, axis=0)
        for i, metric in enumerate(('multiclass_nll', 'multiclass_brier', 'screening_brier')):
            lo, hi, valid = interval(mean_draws[:, i])
            delta = summary.loc[(endpoint, model, 'calibrated'), (metric, 'mean')] - summary.loc[(endpoint, model, 'raw'), (metric, 'mean')]
            interval_rows.append({'endpoint': endpoint, 'model': model, 'metric': metric,
                                  'difference': delta, 'ci_low': lo, 'ci_high': hi, 'valid_draws': valid})
    pd.DataFrame(interval_rows).to_csv(OUTPUT / 'paired_probability_intervals.csv', index=False)
    sections = [r'''\section{Validation-only temperature scaling}
One positive temperature was fitted separately for each of the 80 verified
final retrains using unweighted four-class negative log-likelihood on its
matching corrected validation partition. The model weights, original YAMLs
and frozen splits were unchanged. Validation logits were generated locally
on the RTX 4060 Ti in FP32, without AMP or TF32, using the original
deterministic preprocessing and strict image loading. Temperature fitting
used CUDA float64 optimization. No test or excluded-source label was used
to choose a temperature, method or operating threshold.

For raw scores $z$, calibrated probabilities are
$\mathrm{softmax}(z/T)$. For the saved, strictly positive evaluation
probabilities, the equivalent transform
$\mathrm{softmax}(\log p/T)$ avoids repeating test-image inference. All
temperatures were frozen before calculating evaluation metrics. The same
validation data had already been used for hyperparameter and epoch
selection; these are exploratory post-hoc results, not calibration on a
new untouched cohort. Source-exclusion calibration uses only its own
in-domain validation data, not the excluded source.

Every four-class ranking was checked to remain unchanged. Therefore top-1
accuracy, class-specific F1, top-k inclusion and rank-omission counts are
unchanged. Screening uses the calibrated OPMD-plus-cancer probability at
the unchanged threshold 0.5; its decisions and metrics can change.
Values below are means and sample SDs across seeds, not ensemble results.
Improvement on an excluded source is not guaranteed.''']
    for group, caption, names, headers in [
        ('probability', 'Raw versus validation-temperature-scaled probability quality. NLL uses natural logarithms, multiclass Brier sums four squared errors, and ECE uses ten equal-width bins.',
         ('multiclass_nll', 'multiclass_brier', 'top1_ece', 'screening_nll', 'screening_brier', 'screening_ece'),
         ('Multi NLL', 'Multi Brier', 'Top-1 ECE', 'Screen NLL', 'Screen Brier', 'Screen ECE')),
        ('screening', 'Screening performance after temperature scaling at the fixed threshold 0.5. Raw and calibrated rows use identical frozen evaluation cases.',
         ('screening_f1', 'sensitivity', 'specificity', 'ppv', 'npv', 'screening_auc', 'average_precision'),
         ('F1', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'ROC-AUC', 'AP')),
    ]:
        values = [[endpoint.title(), DISPLAY[model], kind.title(),
                   *[fmt(*summary.loc[(endpoint, model, kind), metric]) for metric in names]]
                  for endpoint in ('pooled', 'kaggle', 'zenodo', 'mendeley') for model in MODELS for kind in ('raw', 'calibrated')]
        sections.append(table(caption, 'tab:s_temperature_' + group, ['Endpoint', 'Model', 'Probabilities', *headers], values, True))
    parameters = pd.DataFrame(temperatures)
    ts = parameters.groupby(['endpoint', 'model']).agg({'temperature': ['mean', 'std'], 'validation_nll_raw': 'mean', 'validation_nll_calibrated': 'mean'})
    values = [[endpoint.title(), DISPLAY[model], fmt(*ts.loc[(endpoint, model), 'temperature']),
               f'{ts.loc[(endpoint, model), ("validation_nll_raw", "mean")]:.3f}',
               f'{ts.loc[(endpoint, model), ("validation_nll_calibrated", "mean")]:.3f}']
              for endpoint in ('pooled', 'kaggle', 'zenodo', 'mendeley') for model in MODELS]
    sections.append(table('Fitted temperatures and resubstitution validation NLL. These validation reductions describe the fitting objective, not independent generalization.',
                          'tab:s_temperatures', ['Endpoint', 'Model', '$T$ mean $\pm$ SD', 'Raw validation NLL', 'Scaled validation NLL'], values))
    values = [[DISPLAY[r['model']], r['metric'].replace('_', ' '), f"{r['difference']:+.3f}",
               f"$[{r['ci_low']:.3f},{r['ci_high']:.3f}]$"] for r in interval_rows if r['endpoint'] == 'pooled']
    sections.append(table('Paired pooled-test differences in proper probability scores (temperature-scaled minus raw). Intervals use 10,000 shared source-stratified draws, internal and Zenodo photographs clustered by patient and Kaggle/Mendeley by image. They condition on the models and fitted temperatures, without resampling calibration fitting. Negative values favor scaling; they are not causal estimates of loss-weighting effects.',
                          'tab:s_temperature_intervals', ['Model', 'Metric', 'Difference', '95\% CI'], values))
    values = [[DISPLAY[model], c.replace('_', ' '),
               f'{summary.loc[("pooled", model, "raw"), (c + "_prevalence", "mean")]:.3f}',
               *[fmt(*summary.loc[('pooled', model, kind), c + suffix])
                 for suffix in ('_mean_probability', '_ece', '_brier') for kind in ('raw', 'calibrated')]]
              for model in MODELS for c in ('healthy', 'benign_lesion', 'opmd', 'cancer')]
    sections.append(table('Pooled class-specific probability diagnostics before and after scalar temperature scaling. Lower overall top-1 ECE does not imply perfect calibration for every class. Class-specific mean probability, ECE and Brier are one-versus-rest measures.',
                          'tab:s_temperature_classwise', ['Model', 'Class', 'Prevalence', 'Mean $p$ raw', 'Mean $p$ scaled', 'ECE raw', 'ECE scaled', 'Brier raw', 'Brier scaled'], values, True))
    bins_frame = pd.DataFrame(bin_rows)
    for task in ('top1', 'screening'):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for ax, model in zip(axes.flat, MODELS):
            for kind, color in (('raw', 'tab:orange'), ('calibrated', 'tab:blue')):
                for seed in SEEDS:
                    sub = bins_frame[(bins_frame.endpoint == 'pooled') & (bins_frame.model == model) &
                                     (bins_frame.kind == kind) & (bins_frame.task == task) &
                                     (bins_frame.seed == seed) & (bins_frame.n_bin > 0)]
                    ax.plot(sub.mean_score, sub.event_fraction, '-o', color=color, lw=.8, ms=3, alpha=.6,
                            label=kind if seed == 42 else None)
            ax.plot([0, 1], [0, 1], ':', color='gray')
            ax.set(xlim=(0, 1), ylim=(0, 1), title=DISPLAY[model], xlabel='Mean probability in bin',
                   ylabel='Observed reference frequency')
            ax.legend(fontsize=8)
        filename = 'temperature_pooled_' + task + '_reliability.pdf'
        fig.savefig(PAPER / 'figures/revision' / filename)
        plt.close(fig)
        sections.append(r'\begin{figure}[p]\centering\includegraphics[width=0.94\textwidth]{figures/revision/' + filename +
                        r'}\caption{Pooled ' + task + r' reliability before and after validation-only temperature scaling. Five separate seed curves per condition; ten fixed equal-width bins, empty bins omitted. No confidence bands are shown.}\end{figure}')
    sections.append(r'\clearpage')
    table_path = PAPER / 'tables/bpex_temperature_calibration.tex'
    table_path.write_text('\n\n'.join(sections) + '\n')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': protocol_hash,
                'script_sha256': sha(Path(__file__).resolve()), 'parameter_sha256': parameter_hashes,
                'evaluated_models': 80, 'fit_device': torch.cuda.get_device_name(0),
                'all_four_class_rankings_unchanged': True,
                'validation_receipt_sha256': {r.study: sha(OUTPUT / r.study / 'validation_inference.json')
                                             for r in audit.itertuples()},
                'table_sha256': sha(table_path),
                'figure_sha256': {p.name: sha(p) for p in (PAPER / 'figures/revision').glob('temperature_pooled_*_reliability.pdf')},
                'output_sha256': {p.name: sha(p) for p in OUTPUT.glob('*.csv')}}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(summary.loc['pooled'][['multiclass_nll', 'multiclass_brier', 'top1_ece', 'screening_brier', 'screening_f1', 'sensitivity', 'specificity']].round(4).to_string(), flush=True)


if __name__ == '__main__':
    main()
