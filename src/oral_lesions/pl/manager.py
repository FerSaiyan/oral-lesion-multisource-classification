import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .teacher import run_teacher_inference_if_needed
from ...utils import getenv_path
from .selection import build_pseudolabel_csv
from ..data.datasets import OralLesionsDatasetCSV
import pandas as pd


def build_pl_loader_from_cfg(
    pl_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    train_csv: str,
    images_folder_unlabeled: str,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    img_size: int,
    pseudo_transform,
    pl_variant: str,
    pl_high_thr: float,
    pl_density_min_p: float,
    pl_density_factor: float,
    cache_dir: Optional[str] = None,
) -> Tuple[Optional[DataLoader], Dict[str, int]]:
    """
    Orchestrates teacher inference + selection + pseudo dataset/loader build.
    Returns (train_loader_pl or None, stats_dict).
    """
    if pl_variant == "none":
        return None, {}

    unlabeled_csv = pl_cfg.get('unlabeled_csv_path') or getenv_path('PL_UNLABELED_CSV')
    unlabeled_images = images_folder_unlabeled or pl_cfg.get('unlabeled_images_folder') or getenv_path('PL_UNLABELED_IMAGES')
    teacher_ckpt = pl_cfg.get('teacher_ckpt_path') or getenv_path('PL_TEACHER_CKPT')
    if not (unlabeled_csv and os.path.exists(unlabeled_csv) and unlabeled_images and os.path.exists(unlabeled_images) and teacher_ckpt and os.path.exists(teacher_ckpt)):
        print("[PL] Missing unlabeled/teacher paths. Skipping PL.")
        return None, {}

    cache_dir = cache_dir or pl_cfg.get('cache_dir') or getenv_path('PL_CACHE_DIR') or os.path.join(os.getcwd(), 'pl_cache')
    os.makedirs(cache_dir, exist_ok=True)

    teacher_temp = float(pl_cfg.get('teacher_temperature', 1.0))
    inf_csv = os.path.join(cache_dir, f"unlabeled_infer_{model_cfg.get('name','model')}_T{teacher_temp:.2f}.csv")

    run_teacher_inference_if_needed(
        teacher_ckpt_path=teacher_ckpt,
        unlabeled_csv_path=unlabeled_csv,
        unlabeled_images_folder=unlabeled_images,
        device=device,
        out_infer_path=inf_csv,
        num_classes=num_classes,
        backbone_name=model_cfg.get('teacher_backbone', model_cfg.get('enet_backbone', 'efficientnet-b7')),
        load_geffnet_pretrained_in_model=bool(model_cfg.get('teacher_imagenet_pretrained', True)),
        temperature=teacher_temp,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
    )

    pseudo_csv = os.path.join(cache_dir, f"pseudo_{pl_variant}_thrH{pl_high_thr:.2f}_minP{pl_density_min_p:.2f}_fac{pl_density_factor:.2f}.csv")
    stats = build_pseudolabel_csv(
        train_csv_path=train_csv,
        unlabeled_infer_path=inf_csv,
        strategy=pl_variant,
        out_infer_path=pseudo_csv,
        pl_high_thr=pl_high_thr,
        pl_density_min_p=pl_density_min_p,
        pl_density_factor=pl_density_factor,
    )

    if not (os.path.exists(pseudo_csv) and os.path.getsize(pseudo_csv) > 0):
        print("[PL] Pseudo-label CSV missing or empty.")
        return None, stats

    # Align pseudo-label mapping with training classes to keep indices consistent
    try:
        train_classes = sorted(pd.read_csv(train_csv).dropna(subset=['diagnosis_categories'])['diagnosis_categories'].astype(str).unique())
    except Exception:
        train_classes = None

    ds_pl = OralLesionsDatasetCSV(pseudo_csv, unlabeled_images, transform=pseudo_transform, classes=train_classes)
    if len(ds_pl) == 0:
        print("[PL] Pseudo dataset empty after CSV filtering.")
        return None, stats

    dl_pl = DataLoader(ds_pl, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    return dl_pl, stats
