import os
import json
from typing import Any, Dict, Optional


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a config file from JSON or YAML.
    Requires PyYAML for .yaml/.yml files.
    """
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Loading YAML requires PyYAML. Install it or provide a .json config. Error: {e}"
        )
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def resolve_device(pref: Optional[str] = None):
    import torch
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
    except Exception:
        pass
    return torch.device('cpu')


def coalesce_path(cfg_value: Optional[str], env_key: Optional[str], default: Optional[str] = None,
                  must_exist: bool = False) -> Optional[str]:
    """
    Choose a path in order of precedence:
      1) cfg_value (from config) if provided
      2) Environment variable (env_key) if set
      3) default
    If must_exist is True, return the first that exists on this machine.
    """
    candidates = []
    if cfg_value:
        candidates.append(cfg_value)
    if env_key:
        v = os.environ.get(env_key)
        if v:
            candidates.append(os.path.expanduser(v))
    if default:
        candidates.append(default)
    if not candidates:
        return None
    if must_exist:
        for c in candidates:
            if c and os.path.exists(c):
                return c
    return candidates[0]
