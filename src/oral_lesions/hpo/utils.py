from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

import optuna
import pandas as pd


def default_search_space() -> Dict[str, Dict[str, Any]]:
    return {
        'finetune_lr':         {'type': 'loguniform', 'low': 1e-6, 'high': 1e-3},
        # Hardcode these for reproducibility; keep as 'fixed' for clarity
        'num_epochs_finetune': {'type': 'fixed',      'value': 2},
        'patience':            {'type': 'fixed',      'value': 5},
        'multiplier':          {'type': 'int',        'low': 1,    'high': 15},
        'warmup_epochs':       {'type': 'int',        'low': 1,    'high': 5},
        'dropout_rate':        {'type': 'float',      'low': 0.1,  'high': 0.7},
    }


def merge_search_space(base: Dict[str, Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not override:
        return base
    merged = {k: dict(v) for k, v in base.items()}
    for k, v in override.items():
        if isinstance(v, dict):
            merged[k] = {**merged.get(k, {}), **v}
        elif isinstance(v, (list, tuple)) and len(v) == 2:
            if k in merged:
                merged[k]['low'], merged[k]['high'] = float(v[0]), float(v[1])
            else:
                merged[k] = {'type': 'float', 'low': float(v[0]), 'high': float(v[1])}
        else:
            merged[k] = v
    return merged


def suggest_from_space(trial: optuna.Trial, key: str, spec: Dict[str, Any]):
    """Suggest a value from a unified search-space spec.
    Honors provided spec, including fixed overrides for keys like
    'num_epochs_finetune' and 'patience'.
    """
    t = spec.get('type')
    low, high = spec.get('low'), spec.get('high')
    if t in ('loguniform', 'logfloat'):
        # Use modern API to avoid deprecation warnings
        return trial.suggest_float(key, float(low), float(high), log=True)
    if t == 'fixed':
        return spec.get('value')
    if t == 'float':
        step = spec.get('step')
        if step is None:
            return trial.suggest_float(key, float(low), float(high))
        return trial.suggest_float(key, float(low), float(high), step=float(step))
    if t == 'int':
        return trial.suggest_int(key, int(low), int(high))
    if t == 'categorical':
        return trial.suggest_categorical(key, spec.get('choices', []))
    raise ValueError(f"Unknown search space type for {key}: {t}")


def load_top_params_from_csv(results_csv: str, top_n: int) -> List[Dict[str, Any]]:
    df = pd.read_csv(results_csv)
    if 'state' in df.columns:
        df = df[df['state'] == 'COMPLETE']
    if 'value' not in df.columns:
        raise ValueError("Results CSV missing 'value' column.")
    df = df.sort_values('value', ascending=False).head(top_n)
    param_cols = [c for c in df.columns if c.startswith('params_')]
    seeds: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        p = {}
        for c in param_cols:
            key = c.replace('params_', '', 1)
            p[key] = row[c]
        seeds.append(p)
    return seeds


def _perturb_value(key: str, val: Any, spec: Dict[str, Any], jitter_fraction: float) -> Any:
    low, high = spec.get('low'), spec.get('high')
    t = spec.get('type')
    if t in ('loguniform', 'logfloat'):
        import math, random
        span_log10 = math.log10(float(high)) - math.log10(float(low))
        delta = jitter_fraction * span_log10
        factor = 10 ** (random.uniform(-delta, delta))
        new = float(val) * factor
        return max(float(low), min(float(high), new))
    if t == 'float':
        import random
        span = float(high) - float(low)
        new = float(val) + random.uniform(-jitter_fraction, jitter_fraction) * span
        return max(float(low), min(float(high), new))
    if t == 'int':
        import random
        span = int(high) - int(low)
        shift = int(round(random.uniform(-jitter_fraction, jitter_fraction) * span))
        new = int(val) + shift
        return max(int(low), min(int(high), new))
    # categorical -> unchanged
    return val


def enqueue_warmstart_trials(study: optuna.Study, seeds: List[Dict[str, Any]], search_space: Dict[str, Dict[str, Any]],
                             enqueue_exact: bool, perturbations_per_seed: int, jitter_fraction: float):
    if enqueue_exact:
        for p in seeds:
            study.enqueue_trial(params=p)
    if perturbations_per_seed > 0:
        for p in seeds:
            for _ in range(perturbations_per_seed):
                pert = dict(p)
                for k, spec in search_space.items():
                    if k in pert:
                        pert[k] = _perturb_value(k, pert[k], spec, jitter_fraction)
                study.enqueue_trial(params=pert)
