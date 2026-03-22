import os
import subprocess
from typing import Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # Parsed only if available


def _norm_abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def get_git_info(project_root: str) -> Dict[str, str]:
    """Return basic Git info for the repo at project_root.
    Keys: git_commit, git_branch, git_dirty ("true"/"false").
    """
    info = {"git_commit": "", "git_branch": "", "git_dirty": ""}
    try:
        def _run(args: List[str]) -> str:
            return subprocess.check_output(args, cwd=project_root).decode().strip()

        commit = _run(["git", "rev-parse", "HEAD"]) or ""
        try:
            branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or ""
        except Exception:
            branch = ""
        try:
            status = _run(["git", "status", "--porcelain"]) or ""
            dirty = "true" if status else "false"
        except Exception:
            dirty = ""
        info.update({"git_commit": commit, "git_branch": branch, "git_dirty": dirty})
    except Exception:
        pass
    return info


def _iter_dvc_files(project_root: str):
    for root, _dirs, files in os.walk(project_root):
        for f in files:
            if f.endswith(".dvc"):
                yield os.path.join(root, f)


def _parse_dvc_outs(dvc_file: str) -> List[dict]:
    outs: List[dict] = []
    if yaml is None:
        return outs
    try:
        with open(dvc_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for section in ("outs", "outs_no_cache"):
            arr = data.get(section) or []
            if isinstance(arr, list):
                for o in arr:
                    if isinstance(o, dict):
                        outs.append(o)
    except Exception:
        pass
    return outs


def _iter_lock_outs(project_root: str):
    """Yield dicts of {path, md5} from dvc.lock if present."""
    outs = []
    if yaml is None:
        return outs
    lock_path = os.path.join(project_root, "dvc.lock")
    if not os.path.exists(lock_path):
        return outs
    try:
        with open(lock_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        stages = data.get("stages") or {}
        for _name, st in (stages.items() if isinstance(stages, dict) else []):
            for sec in ("outs", "outs_no_cache", "deps"):
                arr = st.get(sec) or []
                if not isinstance(arr, list):
                    continue
                for o in arr:
                    if not isinstance(o, dict):
                        continue
                    p = o.get("path")
                    if not isinstance(p, str):
                        continue
                    md5 = o.get("md5") or (o.get("hash") if isinstance(o.get("hash"), str) else None)
                    outs.append({"path": p, "md5": md5})
    except Exception:
        pass
    return outs


def read_dvc_md5_for_path(project_root: str, target_path: str) -> Optional[str]:
    """Return the DVC md5 for a given tracked path.
    Searches .dvc files (dvc add) and dvc.lock (stages/import-url).
    """
    t_abs = _norm_abs(target_path)
    # .dvc entries
    for dvcf in _iter_dvc_files(project_root):
        dvc_dir = os.path.dirname(dvcf)
        for out in _parse_dvc_outs(dvcf):
            p = out.get("path")
            if not isinstance(p, str):
                continue
            cand = p if os.path.isabs(p) else os.path.join(dvc_dir, p)
            if _norm_abs(cand) == t_abs:
                md5 = out.get("md5")
                if isinstance(md5, str) and md5:
                    return md5
    # dvc.lock entries
    for out in _iter_lock_outs(project_root):
        p = out.get("path")
        if not isinstance(p, str):
            continue
        cand = p if os.path.isabs(p) else os.path.join(project_root, p)
        if _norm_abs(cand) == t_abs:
            md5 = out.get("md5")
            if isinstance(md5, str) and md5:
                return md5
    return None


def collect_dvc_versions(project_root: str, named_paths: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Collect DVC md5s for given named paths.
    named_paths maps a logical name to a file/dir path (or None to skip).
    Returns a flat dict of tags like dvc_<name>_md5.
    """
    tags: Dict[str, str] = {}
    for name, path in named_paths.items():
        if not path:
            continue
        md5 = read_dvc_md5_for_path(project_root, path)
        if md5:
            tags[f"dvc_{name}_md5"] = md5
    return tags
