# src/oral_lesions/pl/selection.py
import os
import pandas as pd
from typing import Dict, Tuple

def _get_class_mapping_from_train(train_csv_path: str) -> Tuple[dict, dict]:
    """
    Deterministic mapping using alphabetical order of label strings from train CSV.
    Returns (idx_to_label, label_to_idx).
    """
    df = pd.read_csv(train_csv_path).dropna(subset=['diagnosis_categories'])
    classes_sorted = sorted(df['diagnosis_categories'].unique())
    idx_to_label = {i: c for i, c in enumerate(classes_sorted)}
    label_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    return idx_to_label, label_to_idx

def _count_labeled_per_class(train_csv_path: str) -> Dict[str, int]:
    df = pd.read_csv(train_csv_path).dropna(subset=['diagnosis_categories'])
    return df['diagnosis_categories'].value_counts().to_dict()

def build_pseudolabel_csv(
    train_csv_path: str,
    unlabeled_infer_path: str,
    strategy: str,
    out_infer_path: str,
    pl_high_thr: float,
    pl_density_min_p: float,
    pl_density_factor: float,
) -> Dict[str, int]:
    """
    Read teacher inference CSV (filename, pred_idx, p_max) and write a pseudo-label CSV:
      filename, diagnosis_categories
    Returns per-class acceptance counts.
    Strategies:
      - "PL-H": global threshold, then per-class cap to min class count (balances).
      - "PL-D": per-class top-N with min p constraint; N = factor * (# labeled per class).
    """
    idx_to_label, _ = _get_class_mapping_from_train(train_csv_path)
    df_inf = pd.read_csv(unlabeled_infer_path)
    if df_inf.empty:
        pd.DataFrame(columns=['filename', 'diagnosis_categories']).to_csv(out_infer_path, index=False)
        return {}

    df_inf['pred_label'] = df_inf['pred_idx'].map(idx_to_label)

    if strategy == "PL-H":
        df_thr = df_inf[df_inf['p_max'] >= float(pl_high_thr)].copy()
        if df_thr.empty:
            accepted = df_thr.head(0)
        else:
            per_class = df_thr.groupby('pred_label', as_index=False).size().rename(columns={'size': 'n'})
            min_n = per_class['n'].min()
            accepted = (df_thr.sort_values('p_max', ascending=False)
                             .groupby('pred_label', group_keys=False)
                             .head(min_n))
    elif strategy == "PL-D":
        labeled_counts = _count_labeled_per_class(train_csv_path)
        parts = []
        for label, count_l in labeled_counts.items():
            N = max(1, int(float(pl_density_factor) * int(count_l)))
            df_k = df_inf[(df_inf['pred_label'] == label) & (df_inf['p_max'] >= float(pl_density_min_p))]
            parts.append(df_k.sort_values('p_max', ascending=False).head(N))
        accepted = pd.concat(parts, axis=0) if parts else df_inf.head(0)
    else:
        raise ValueError(f"Unknown PL strategy: {strategy}")

    if accepted.empty:
        print(f"[PL] No pseudo-labels accepted with strategy={strategy}.")
        pd.DataFrame(columns=['filename', 'diagnosis_categories']).to_csv(out_infer_path, index=False)
        return {}

    df_pl = accepted[['filename', 'pred_label']].rename(columns={'pred_label': 'diagnosis_categories'})
    os.makedirs(os.path.dirname(out_infer_path), exist_ok=True)
    df_pl.to_csv(out_infer_path, index=False)
    counts = df_pl['diagnosis_categories'].value_counts().to_dict()
    print(f"[PL] Wrote pseudo-label CSV: {out_infer_path} | per-class accepted: {counts}")
    return counts
