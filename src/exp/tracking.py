import os
from contextlib import contextmanager
from typing import Any, Dict, Optional


def canonicalize_tracking_uri(uri: Optional[str], default_root: str) -> Optional[str]:
    """Return a canonical MLflow tracking URI.
    - If uri is None: use file:<default_root>/mlruns
    - If uri starts with file: and path is relative: convert to absolute under default_root
    - Otherwise: return unchanged
    """
    if uri is None:
        return f"file:{os.path.join(default_root, 'mlruns')}"
    if uri.startswith('file:'):
        rest = uri[5:]
        # Handle file:///abs/path and file:/abs/path as already-absolute
        if rest.startswith('/'):
            return uri
        # Relative path (e.g., file:./mlruns or file:mlruns)
        abs_path = os.path.abspath(os.path.join(default_root, rest))
        return f"file:{abs_path}"
    return uri


@contextmanager
def mlflow_run(enabled: bool = True,
               tracking_uri: Optional[str] = None,
               experiment_name: Optional[str] = None,
               run_name: Optional[str] = None,
               tags: Optional[Dict[str, Any]] = None,
               params: Optional[Dict[str, Any]] = None,
               nested: bool = False):
    """
    Lightweight MLflow context manager that is a no-op if mlflow is unavailable
    or disabled. Avoids hard dependency on mlflow for environments without it.
    Correctly handles exceptions without yielding during error unwinding.
    """
    if not enabled:
        yield None
        return
    try:
        import mlflow
    except Exception:
        yield None
        return

    # Attempt to start an MLflow run; if it fails, degrade to no-op
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        run_ctx = mlflow.start_run(run_name=run_name, nested=nested)
    except Exception:
        yield None
        return

    with run_ctx as run:
        if tags:
            mlflow.set_tags(tags)
        if params:
            flat_params = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                           for k, v in params.items()}
            mlflow.log_params(flat_params)
        yield run
