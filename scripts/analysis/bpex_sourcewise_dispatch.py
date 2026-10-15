"""Validate execution assignments without changing the frozen scientific plan."""
import json
from pathlib import Path

from run_bpex_revision_hpo import ROOT, sha256

LEGACY_DISPATCH = ROOT / 'configs/studies/bpex_sourcewise_dispatch_2026-10-01.json'
DISPATCH = ROOT / 'configs/studies/bpex_sourcewise_dispatch_2026-10-02.json'
WORKERS = ('local', 'cloud', 'luciano_gpu0', 'luciano_gpu1', 'photobiopc')


def key(task):
    return f"{task['source']}/{task['model']}/{task['seed']}"


def dispatch_path():
    return DISPATCH if DISPATCH.exists() else LEGACY_DISPATCH


def assignments(plan, plan_path, path=None):
    if path is None:
        path = dispatch_path()
    if not path.exists():
        return {key(t): t['worker'] for t in plan['tasks']}
    dispatch = json.loads(path.read_text())
    if dispatch['plan_sha256'] != sha256(plan_path):
        raise ValueError('Dispatch refers to a different scientific plan')
    owners = dispatch['assignments']
    if set(owners) != {key(t) for t in plan['tasks']} or not set(owners.values()) <= set(WORKERS):
        raise ValueError('Dispatch must assign every frozen study exactly once')
    return owners
