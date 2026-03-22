import random
import numpy as np
import torch
import os
from typing import Optional


def set_seed(seed_value):
    """Set random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # For full determinism, consider these, but they can impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Seeds set with SEED = {seed_value}")


# --- Lightweight .env loader (avoid external deps) ---
def _parse_env_line(line: str):
    # ignore comments and empty lines
    s = line.strip()
    if not s or s.startswith('#'):
        return None, None
    # allow KEY="value with spaces" and KEY=value
    if '=' not in s:
        return None, None
    k, v = s.split('=', 1)
    k = k.strip()
    v = v.strip().strip('"').strip("'")
    return k, v


def load_dotenv(dotenv_path: Optional[str] = None) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ if not already set."""
    # Default to project root's .env (src/.. -> repo)
    if dotenv_path is None:
        this_dir = os.path.abspath(os.path.dirname(__file__))
        repo_root = os.path.abspath(os.path.join(this_dir, '..'))
        dotenv_path = os.path.join(repo_root, '.env')
    try:
        if os.path.exists(dotenv_path):
            with open(dotenv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    k, v = _parse_env_line(line)
                    if k and (k not in os.environ or os.environ[k] == ''):
                        os.environ[k] = v
    except Exception:
        # non-fatal
        pass


def getenv_path(key: str, default: Optional[str] = None, expanduser: bool = True) -> Optional[str]:
    """Return an environment path with optional user expansion. Returns default if not set."""
    v = os.environ.get(key, default)
    if v is None:
        return None
    return os.path.expanduser(v) if expanduser else v


if __name__ == '__main__':
    # Example usage:
    set_seed(42)
    print("Seed set. This module provides utility functions.")
