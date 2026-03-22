import os
import sys
import time
import json
import traceback
from typing import Any, Dict, List, Optional
import tempfile
import shutil
import gc

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as _MPDataLoaderIter
import multiprocessing.queues as _mp_queues
import torch.utils.data._utils.pin_memory as _pin_memory_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna
try:
    from optuna.integration import MLflowCallback  # Optuna → MLflow integration
except Exception:
    MLflowCallback = None  # optional
try:
    import mlflow  # Optional runtime dep
    mlflow_import_error = None
except Exception as e:
    mlflow = None
    mlflow_import_error = e

_orig_mp_del = getattr(_MPDataLoaderIter, "__del__", None)
if _orig_mp_del is not None:
    def _safe_mp_del(self):
        try:
            _orig_mp_del(self)
        except AssertionError as exc:
            if "can only test a child process" not in str(exc):
                raise
    _MPDataLoaderIter.__del__ = _safe_mp_del

_orig_queue_feed = getattr(_mp_queues.Queue, "_feed", None)
if _orig_queue_feed is not None:
    def _safe_queue_feed(*args, **kwargs):
        try:
            _orig_queue_feed(*args, **kwargs)
        except BrokenPipeError:
            # Happens when the main process closes the pipe while
            # DataLoader workers are still feeding; safe to ignore.
            pass
    _mp_queues.Queue._feed = _safe_queue_feed

_orig_pin_memory_loop = getattr(_pin_memory_utils, "_pin_memory_loop", None)
if _orig_pin_memory_loop is not None:
    def _safe_pin_memory_loop(*args, **kwargs):
        try:
            _orig_pin_memory_loop(*args, **kwargs)
        except (ConnectionResetError, BrokenPipeError):
            # Can happen at shutdown when worker processes have already exited.
            # Safe to ignore for our training loop.
            pass
    _pin_memory_utils._pin_memory_loop = _safe_pin_memory_loop

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import set_seed, load_dotenv, getenv_path
from src.oral_lesions.data import (
    OralLesionsDatasetCSV,
    MultiSourceDatasetCSV,
    build_transforms,
    build_pseudo_augs,
    build_dataloaders,
    ensure_splits,
    infer_dataset_variant,
)
from src.oral_lesions.models.factory import create_model
from src.oral_lesions.pl.manager import build_pl_loader_from_cfg
from src.schedulers import GradualWarmupScheduler
from src.oral_lesions.engine.trainer import Trainer
from src.exp.dvc_utils import get_git_info, collect_dvc_versions
from src.oral_lesions.engine.callbacks import CheckpointSaver, EarlyStopping
from src.exp.config import load_config, resolve_device, coalesce_path
from src.exp.tracking import canonicalize_tracking_uri
from src.oral_lesions.hpo.utils import (
    default_search_space,
    merge_search_space,
    suggest_from_space,
    load_top_params_from_csv,
    enqueue_warmstart_trials,
)


def _resolve_finetune_strategy(study_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve finetuning strategy configuration.
    Returns a dict with keys:
      - strategy: 'full', 'head_warmup', or 'gradual_unfreeze'
      - head_warmup_epochs: int
      - gradual_schedule: list[dict] or None

    Defaults to 'full' (current behavior) if not configured.
    """
    # Strategy can be defined globally or per-model; model-level overrides global.
    model_ft = model_cfg.get('finetune') or {}
    global_ft = study_cfg.get('finetune') or {}

    strategy = model_ft.get('strategy', global_ft.get('strategy', 'full'))
    head_warmup_epochs = int(model_ft.get('head_warmup_epochs', global_ft.get('head_warmup_epochs', 10)))
    gradual_schedule = model_ft.get('gradual_schedule', global_ft.get('gradual_schedule'))

    return {
        'strategy': strategy,
        'head_warmup_epochs': head_warmup_epochs,
        'gradual_schedule': gradual_schedule,
    }


def _set_backbone_trainable_effnet(model: nn.Module, train_backbone: bool) -> None:
    """
    Freeze/unfreeze EfficientNet backbone while always keeping head trainable.
    Assumes `enetv2` structure from src.models: `model.enet` as backbone,
    `model.myfc` and `model.dropout` as head.
    """
    # Backbone (feature extractor)
    if hasattr(model, "enet"):
        for p in model.enet.parameters():
            p.requires_grad = train_backbone
    # Head (classifier + dropout) should always remain trainable
    for attr in ("myfc", "dropout"):
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters():
                p.requires_grad = True


def _set_backbone_trainable_vit(model: nn.Module, train_backbone: bool, gradual_schedule: Optional[List[Dict[str, Any]]] = None, epoch_idx: Optional[int] = None) -> None:
    """
    Freeze/unfreeze ViT backbone while always keeping head trainable.

    If `gradual_schedule` is provided and epoch_idx is not None, it should be a list
    of dicts like:
        [{"until_epoch": 10, "unfreeze_last_n_blocks": 0},
         {"until_epoch": 20, "unfreeze_last_n_blocks": 2},
         {"until_epoch": 1e9, "unfreeze_last_n_blocks": -1}]

    where -1 means unfreeze all blocks.
    """
    vit = getattr(model, "vit_model", None)
    if vit is None:
        return

    # Default behavior: toggle all backbone parameters
    def _toggle_all(backbone_trainable: bool) -> None:
        for p in vit.parameters():
            p.requires_grad = backbone_trainable

    # Apply gradual schedule if provided
    if gradual_schedule is not None and epoch_idx is not None and hasattr(vit, "blocks"):
        # Find the first schedule entry whose until_epoch is >= current epoch_idx+1
        current_epoch = int(epoch_idx) + 1
        selected = None
        for s in gradual_schedule:
            try:
                until = int(s.get("until_epoch"))
            except Exception:
                continue
            if current_epoch <= until:
                selected = s
                break
        if selected is not None:
            n_blocks = getattr(vit, "blocks", None)
            if n_blocks is not None:
                unfreeze_last = int(selected.get("unfreeze_last_n_blocks", -1))
                # First freeze all blocks
                for p in vit.blocks.parameters():
                    p.requires_grad = False
                if unfreeze_last != 0:
                    if unfreeze_last < 0 or unfreeze_last >= len(vit.blocks):
                        # Unfreeze all blocks
                        for p in vit.blocks.parameters():
                            p.requires_grad = True
                    else:
                        # Unfreeze only last N blocks
                        for p in vit.blocks[-unfreeze_last:].parameters():
                            p.requires_grad = True
            else:
                # Fallback to all backbone if blocks is not available
                _toggle_all(True)
        else:
            # No schedule entry matched; fallback to all-trainable
            _toggle_all(True)
    else:
        _toggle_all(train_backbone)

    # Head and dropout should always be trainable
    if hasattr(vit, "head"):
        for p in vit.head.parameters():
            p.requires_grad = True
    if hasattr(model, "dropout"):
        for p in model.dropout.parameters():
            p.requires_grad = True



# Helper functions relocated to src.exp.config, src.exp.tracking, and src.oral_lesions.data.utils to keep this script lean.


def run_study(study_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Global
    seed = int(study_cfg.get('seed', 42))
    set_seed(seed)
    device = resolve_device(study_cfg.get('device'))
    # Results dir: env OPTUNA_RESULTS_DIR > cfg > repo default.
    # If a relative path is provided (common in YAML), resolve it from PROJECT_ROOT
    # instead of the current working directory so that notebook runs and CLI runs
    # share the same results location.
    results_dir = coalesce_path(
        study_cfg.get('optuna_results_dir'),
        env_key='OPTUNA_RESULTS_DIR',
        default=os.path.join(PROJECT_ROOT, 'results', 'optuna_studies')
    )
    if results_dir is not None and not os.path.isabs(results_dir):
        results_dir = os.path.join(PROJECT_ROOT, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Data setup
    # Base dataset dir: env DATASET_DIR > cfg > repo/data
    base_dir = coalesce_path(
        study_cfg.get('base_data_dir'),
        env_key='DATASET_DIR',
        default=os.path.join(PROJECT_ROOT, 'data'),
        must_exist=False
    )
    # Images folder: env IMAGES_FOLDER > cfg > ${base_dir}/Dataset_for_transfer_learning
    images_folder = coalesce_path(
        study_cfg.get('images_folder'),
        env_key='IMAGES_FOLDER',
        default=os.path.join(base_dir, 'Dataset_for_transfer_learning') if base_dir else None,
        must_exist=False
    )
    # CSVs (env overrides if set)
    labels_csv_default = os.path.join(base_dir, 'transfer_learning_labels.csv') if base_dir else None
    train_csv_default = os.path.join(base_dir, 'train_transfer.csv') if base_dir else None
    val_csv_default = os.path.join(base_dir, 'val_transfer.csv') if base_dir else None
    labels_csv_path = coalesce_path(study_cfg.get('labels_csv_path'), env_key='LABELS_CSV', default=labels_csv_default)
    train_csv = coalesce_path(study_cfg.get('train_csv_path'), env_key='TRAIN_CSV', default=train_csv_default)
    val_csv = coalesce_path(study_cfg.get('val_csv_path'), env_key='VAL_CSV', default=val_csv_default)
    # Optional explicit test split (e.g., multisource_test.csv)
    test_csv = coalesce_path(study_cfg.get('test_csv_path'), env_key='TEST_CSV', default=None, must_exist=False)

    # If True, require precomputed SGKF (or equivalent) splits; do not fallback.
    require_precomputed = bool(study_cfg.get('require_precomputed_splits', False))
    ensure_splits(train_csv, val_csv, labels_csv_path, seed, require_precomputed=require_precomputed)
    print(f"Data config → images_folder='{images_folder}', train_csv='{train_csv}', val_csv='{val_csv}', test_csv='{test_csv}'")

    # Derived naming
    study_name = model_cfg.get('study_name') or f"{study_cfg['study_name_prefix']}_{model_cfg['name']}"
    study_output_dir = os.path.join(results_dir, study_name)
    os.makedirs(study_output_dir, exist_ok=True)
    # Temporary directory for model artifacts (avoid cluttering results/)
    tmp_dir = tempfile.mkdtemp(prefix=f"{study_name}_")
    storage_uri = f"sqlite:///{os.path.join(study_output_dir, study_name)}.db"

    # Determine num_classes and base transforms.
    # If using the multisource CSVs produced by clustering_wrangling.ipynb,
    # diagnosis_categories will already contain the 4-class coarse labels.
    tmp_ds = OralLesionsDatasetCSV(train_csv, images_folder)
    num_classes = tmp_ds.num_classes
    del tmp_ds

    model_type = model_cfg['model_type']  # effnet | vit
    default_size = 512 if model_type == 'effnet' else 224
    img_size = int(model_cfg.get('img_size') or study_cfg.get('img_size') or default_size)
    # Aspect ratio policy (transform-level)
    resize_mode = (model_cfg.get('resize_mode')
                   or study_cfg.get('resize_mode')
                   or 'stretch')
    train_t, val_t = build_transforms(model_type, img_size, resize_mode=resize_mode)
    pseudo_t = build_pseudo_augs(img_size, resize_mode=resize_mode)

    # Pseudolabel settings (env-aware)
    pl_cfg = dict(study_cfg.get('pseudolabel') or {})
    # Fill from env if not provided
    pl_cfg.setdefault('unlabeled_csv_path', getenv_path('PL_UNLABELED_CSV'))
    pl_cfg.setdefault('unlabeled_images_folder', getenv_path('PL_UNLABELED_IMAGES'))
    pl_cfg.setdefault('teacher_ckpt_path', getenv_path('PL_TEACHER_CKPT'))
    pl_cfg.setdefault('cache_dir', getenv_path('PL_CACHE_DIR'))
    pl_enabled = bool(pl_cfg.get('enabled', False))
    
    # Imbalance handling flags (global; can be overridden per-model if desired)
    class_weights_enabled = bool(study_cfg.get('class_weights', False) or model_cfg.get('class_weights', False))
    sampler_enabled       = bool(study_cfg.get('weighted_sampler', False) or model_cfg.get('weighted_sampler', False))

    # Search space (supports override/widen)
    base_space = default_search_space()
    # Global overrides under study.search_space
    base_space = merge_search_space(base_space, study_cfg.get('search_space'))
    # Per-model overrides under model_cfg.search_space
    search_space = merge_search_space(base_space, model_cfg.get('search_space'))

    # Helper to build supervised train/val/test loaders, optionally with class weights and/or WeightedRandomSampler.
    # If the training CSV includes a 'sample_weight' column (e.g., multisource CSV),
    # those weights are passed through MultiSourceDatasetCSV and combined with class weights.
    #
    # Split semantics (for HPO and evaluation):
    #   - train_csv → used for supervised training (and pseudo-label training when enabled).
    #   - val_csv   → optimization/early-stopping split; macro F1 on this split is used
    #                 as the Optuna objective and for checkpoint selection.
    #   - test_csv  → optional held-out evaluation split; metrics on this split are logged
    #                 but are not used for early stopping or hyperparameter optimization.
    from collections import Counter
    use_sample_weights = False

    def _make_supervised_loaders(batch_size: int, num_workers: int):
        """
        Return (dl_tr, dl_va, dl_te, num_classes_local, class_weights_tensor_or_None).
        dl_te may be None if no explicit test split is configured.
        """
        nonlocal num_classes, use_sample_weights, test_csv

        # Check whether CSV has explicit per-sample weights (e.g., multisource CSV)
        try:
            has_sample_weight = 'sample_weight' in pd.read_csv(train_csv, nrows=1).columns
        except Exception:
            has_sample_weight = False

        use_sample_weights = bool(has_sample_weight)
        test_csv_exists = bool(test_csv and os.path.exists(test_csv))

        if not (class_weights_enabled or sampler_enabled) and not use_sample_weights and not test_csv_exists:
            # Simple path: no explicit weights, no sampler, no test split → use standard builder
            dl_tr, dl_va, num_classes_local = build_dataloaders(
                train_csv,
                val_csv,
                images_folder,
                model_type,
                batch_size,
                num_workers,
                img_size,
                resize_mode=resize_mode,
            )
            num_classes = num_classes_local
            return dl_tr, dl_va, None, num_classes_local, None

        # Manual path with explicit class mapping and optional weighting (and optional test split)
        tr_df = pd.read_csv(train_csv).dropna(subset=['diagnosis_categories', 'filename'])
        va_df = pd.read_csv(val_csv).dropna(subset=['diagnosis_categories', 'filename'])
        tr_labels = tr_df['diagnosis_categories'].astype(str)
        va_labels = va_df['diagnosis_categories'].astype(str)

        te_df = None
        if test_csv_exists:
            te_df = pd.read_csv(test_csv).dropna(subset=['diagnosis_categories', 'filename'])
            te_labels = te_df['diagnosis_categories'].astype(str)
            global_classes = sorted(
                set(tr_labels.unique()).union(va_labels.unique()).union(te_labels.unique())
            )
        else:
            global_classes = sorted(set(tr_labels.unique()).union(va_labels.unique()))
        if not global_classes:
            raise ValueError("No classes found in training/validation CSVs for imbalance handling.")
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(global_classes)}

        # Class weights from inverse frequency
        from collections import Counter as _Counter
        counts = _Counter(tr_labels.tolist())
        freqs = torch.tensor([counts[c] for c in global_classes], dtype=torch.float)
        inv_freqs = 1.0 / freqs
        class_weights_tensor = inv_freqs / inv_freqs.sum()

        # Choose dataset class:
        # - If 'sample_weight' present, use MultiSourceDatasetCSV so each sample returns its weight.
        # - Otherwise, fall back to OralLesionsDatasetCSV (class-balanced sampler only).
        if has_sample_weight:
            ds_tr = MultiSourceDatasetCSV(train_csv, images_folder, transform=train_t, classes=global_classes)
            ds_va = MultiSourceDatasetCSV(val_csv, images_folder, transform=val_t, classes=global_classes)
            ds_te = MultiSourceDatasetCSV(test_csv, images_folder, transform=val_t, classes=global_classes) if test_csv_exists else None
        else:
            ds_tr = OralLesionsDatasetCSV(train_csv, images_folder, transform=train_t, classes=global_classes)
            ds_va = OralLesionsDatasetCSV(val_csv, images_folder, transform=val_t, classes=global_classes)
            ds_te = OralLesionsDatasetCSV(test_csv, images_folder, transform=val_t, classes=global_classes) if test_csv_exists else None

        if len(ds_tr) == 0 or len(ds_va) == 0 or (ds_te is not None and len(ds_te) == 0):
            raise ValueError("One of the datasets is empty. Check CSV paths and image files.")

        # Build training loader
        if sampler_enabled:
            # WeightedRandomSampler based on class frequencies (ignores sample_weight column)
            sample_weights = [class_weights_tensor[class_to_idx[y]] for y in tr_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler,
                               num_workers=num_workers, pin_memory=True, drop_last=False)
        else:
            # Standard shuffle; MultiSourceDatasetCSV will still expose per-sample weights
            dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=False)

        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

        dl_te = None
        if ds_te is not None:
            dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
        num_classes = len(global_classes)
        return dl_tr, dl_va, dl_te, num_classes, class_weights_tensor

    # Static tags: Git + DVC dataset versions
    git_tags = get_git_info(PROJECT_ROOT)
    dvc_tags = collect_dvc_versions(
        PROJECT_ROOT,
        {
            "images_folder": images_folder,
            "train_csv": train_csv,
            "val_csv": val_csv,
            "test_csv": test_csv,
            "labels_csv": labels_csv_path,
        },
    )

    # Dataset variant tag (preprocessing-level); explicit cfg wins, else infer from folder name
    dataset_variant = (study_cfg.get('dataset_variant') or infer_dataset_variant(images_folder))

    # Finetuning strategy (defaults to "full" to preserve existing behavior)
    finetune_cfg = _resolve_finetune_strategy(study_cfg, model_cfg)

    def objective(trial: optuna.Trial, global_tracker: dict):
        set_seed(seed)

        # Hyperparameters (from search_space)
        finetune_lr   = suggest_from_space(trial, 'finetune_lr', search_space['finetune_lr'])
        num_epochs    = suggest_from_space(trial, 'num_epochs_finetune', search_space['num_epochs_finetune'])
        patience      = suggest_from_space(trial, 'patience', search_space['patience'])
        multiplier    = suggest_from_space(trial, 'multiplier', search_space['multiplier'])
        warmup_epochs = suggest_from_space(trial, 'warmup_epochs', search_space['warmup_epochs'])
        dropout_rate  = suggest_from_space(trial, 'dropout_rate', search_space['dropout_rate'])
        batch_size = int(study_cfg.get('batch_size', 8))
        num_workers = int(study_cfg.get('num_workers', 0))

        # Optional PL knobs
        if pl_enabled:
            pl_variant = trial.suggest_categorical("pl_variant", ["none", "PL-H", "PL-D"])  # allow disabling per trial
        else:
            pl_variant = "none"
        pl_high_thr = trial.suggest_float("pl_high_thr", 0.85, 0.95, step=0.01) if pl_variant == "PL-H" else float(pl_cfg.get('high_thr', 0.90))
        pl_density_min_p = trial.suggest_float("pl_density_min_p", 0.75, 0.90, step=0.01) if pl_variant == "PL-D" else float(pl_cfg.get('density_min_p', 0.80))
        pl_density_factor = trial.suggest_float("pl_density_factor", 0.25, 1.0, step=0.05) if pl_variant == "PL-D" else float(pl_cfg.get('density_factor', 0.5))

        # Log trial params to MLflow (if active)
        try:
            if 'mlflow' in globals() and mlflow is not None and mlflow.active_run() is not None:
                mlflow.log_params({
                    'finetune_lr': float(finetune_lr),
                    'num_epochs_finetune': int(num_epochs),
                    'patience': int(patience) if patience is not None else -1,
                    'multiplier': float(multiplier),
                    'warmup_epochs': int(warmup_epochs),
                    'dropout_rate': float(dropout_rate),
                    'batch_size': int(batch_size),
                    'num_workers': int(num_workers),
                    'model_type': model_type,
                    'img_size': int(img_size),
                    'resize_mode': str(resize_mode),
                })
        except Exception:
            pass

        # Datasets / loaders
        try:
            dl_tr, dl_va, dl_te, num_classes, class_weights_tensor = _make_supervised_loaders(batch_size, num_workers)

            dl_pl, pl_stats = None, {}
            if pl_variant != "none":
                dl_pl, pl_stats = build_pl_loader_from_cfg(
                    pl_cfg=pl_cfg,
                    model_cfg=model_cfg,
                    train_csv=train_csv,
                    images_folder_unlabeled=pl_cfg.get('unlabeled_images_folder'),
                    num_classes=num_classes,
                    device=device,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    img_size=img_size,
                    pseudo_transform=build_pseudo_augs(img_size, resize_mode=resize_mode),
                    pl_variant=pl_variant,
                    pl_high_thr=pl_high_thr,
                    pl_density_min_p=pl_density_min_p,
                    pl_density_factor=pl_density_factor,
                    cache_dir=os.path.join(study_output_dir, 'pl_cache'),
                )

            print(f"  Dataloaders: Sup={len(dl_tr)} | PL={0 if dl_pl is None else len(dl_pl)} | Val={len(dl_va)} | Test={0 if dl_te is None else len(dl_te)}")
        except Exception as e:
            print(f"ERROR building loaders: {e}")
            traceback.print_exc()
            return 0.0

        # Model
        model_build_cfg = dict(model_cfg)
        model_build_cfg['dropout_rate'] = dropout_rate
        try:
            model = create_model(model_build_cfg, num_classes=num_classes, device=device)
            model = model.to(device)
            # Default behavior: keep all parameters trainable; callbacks
            # may temporarily freeze subsets depending on finetune_cfg.
            for p in model.parameters():
                p.requires_grad = True
        except Exception as e:
            print(f"ERROR building model: {e}")
            traceback.print_exc()
            return 0.0

        # Optimizer, schedulers, scaler, trainer
        optimizer = optim.Adam(model.parameters(), lr=finetune_lr)
        sched_cos = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        sched_warm = GradualWarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmup_epochs, after_scheduler=sched_cos)
        scaler = GradScaler('cuda') if device.type == 'cuda' else GradScaler(enabled=False)

        trial_ckpt = os.path.join(tmp_dir, f"trial_{trial.number}_best.pth")
        # Only use EarlyStopping; avoid per-trial checkpoint saving.
        callbacks = []
        if patience is not None:
            callbacks.append(EarlyStopping(patience=int(patience)))

        # Criterion (optionally class-weighted and/or sample-weighted).
        # Use reduction='none' so Trainer can apply per-sample weights when available.
        if class_weights_enabled and class_weights_tensor is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device), reduction='none')
        else:
            criterion = nn.CrossEntropyLoss(reduction='none')

        trainer = Trainer(device=device, criterion=criterion, optimizer=optimizer, scheduler=sched_warm, scaler=scaler, callbacks=callbacks)

        def on_epoch_start(epoch_idx: int, model_ref: nn.Module):
            """
            Per-epoch hook to apply finetuning strategy:
              - full: no-op (all params trainable)
              - head_warmup: freeze backbone for first `head_warmup_epochs` epochs
              - gradual_unfreeze: apply schedule (primarily for ViT)
            """
            strategy = finetune_cfg.get('strategy', 'full')
            head_warmup_epochs = int(finetune_cfg.get('head_warmup_epochs', 10))
            gradual_schedule = finetune_cfg.get('gradual_schedule')

            if strategy == 'full':
                return

            if model_type == 'effnet':
                if strategy == 'head_warmup':
                    train_backbone = (epoch_idx >= head_warmup_epochs)
                    _set_backbone_trainable_effnet(model_ref, train_backbone=train_backbone)
                elif strategy == 'gradual_unfreeze':
                    # For EffNet, treat gradual_unfreeze same as head_warmup unless
                    # a more detailed scheme is added later.
                    train_backbone = (epoch_idx >= head_warmup_epochs)
                    _set_backbone_trainable_effnet(model_ref, train_backbone=train_backbone)
            elif model_type == 'vit':
                if strategy == 'head_warmup':
                    train_backbone = (epoch_idx >= head_warmup_epochs)
                    _set_backbone_trainable_vit(model_ref, train_backbone=train_backbone)
                elif strategy == 'gradual_unfreeze':
                    _set_backbone_trainable_vit(
                        model_ref,
                        train_backbone=(epoch_idx >= head_warmup_epochs),
                        gradual_schedule=gradual_schedule,
                        epoch_idx=epoch_idx,
                    )

        def on_epoch_end(epoch_idx: int, log_row: Dict[str, Any]):
            # Optimization metric: macro F1 on the validation split.
            f1 = float(log_row['val_f1'])
            # Optuna reporting + pruning
            trial.report(f1, epoch_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            # Track best within this trial (no per-trial saving)
            if f1 >= getattr(on_epoch_end, 'best_f1', -1):
                on_epoch_end.best_f1 = f1
            # Track global best across trials
            if f1 > global_tracker['best_f1']:
                global_tracker['best_f1'] = f1
                global_tracker['trial_number'] = trial.number
                global_tracker['epoch'] = int(log_row['epoch'])
                torch.save(model.state_dict(), global_tracker['model_save_path'])
            # Optional: per-epoch MLflow logging (works with MLflowCallback run context)
            try:
                if 'mlflow' in globals() and mlflow is not None and mlflow.active_run() is not None:
                    step = int(log_row['epoch'])
                    mlflow.log_metric('val_f1', f1, step=step)
                    mlflow.log_metric('val_loss', float(log_row['val_loss']), step=step)
                    mlflow.log_metric('train_loss', float(log_row['train_loss']), step=step)
                    mlflow.log_metric('val_f1_weighted', float(log_row.get('val_f1_weighted', 0.0)), step=step)
                    mlflow.log_metric('val_acc', float(log_row.get('val_acc', 0.0)), step=step)
                    if 'test_loss' in log_row:
                        mlflow.log_metric('test_loss', float(log_row.get('test_loss', 0.0)), step=step)
                        mlflow.log_metric('test_f1', float(log_row.get('test_f1', 0.0)), step=step)
                        mlflow.log_metric('test_f1_weighted', float(log_row.get('test_f1_weighted', 0.0)), step=step)
                        mlflow.log_metric('test_acc', float(log_row.get('test_acc', 0.0)), step=step)
                    mlflow.log_metric('lr', float(log_row['lr']), step=step)
                    if 'lambda_u' in log_row:
                        mlflow.log_metric('lambda_u', float(log_row['lambda_u']), step=step)
            except Exception:
                pass

        # Run training
        try:
            best_f1 = trainer.fit_mix(
                model=model,
                train_loader_sup=dl_tr,
                val_loader=dl_va,
                epochs=int(num_epochs),
                log_csv_path=os.path.join(study_output_dir, f"trial_{trial.number}_log.csv"),
                ckpt_path=trial_ckpt,
                train_loader_pl=dl_pl,
                lambda_ramp_epochs=20,
                test_loader=dl_te,
                on_epoch_start=on_epoch_start,
                on_epoch_end=on_epoch_end,
            )
        except optuna.exceptions.TrialPruned:
            print(f"    -> Trial {trial.number} pruned.")
            raise
        except StopIteration:
            # Early stopping triggered via callback
            best_f1 = getattr(on_epoch_end, 'best_f1', 0.0)
        # Do not persist per-trial checkpoints; only global best is maintained and overwritten.

        best_val_f1 = float(getattr(on_epoch_end, 'best_f1', 0.0))

        # Explicitly release GPU-heavy objects between trials to avoid
        # gradual CUDA memory growth when running many Optuna trials.
        try:
            del dl_tr, dl_va, dl_te, dl_pl
        except Exception:
            pass
        try:
            del model, optimizer, sched_cos, sched_warm, scaler, trainer
        except Exception:
            pass
        if device.type == 'cuda':
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        return best_val_f1

    # Prepare study and run
    print(f"\n==> Starting study for model: {model_cfg['name']} ({model_type})")
    study_name = model_cfg.get('study_name') or f"{study_cfg['study_name_prefix']}_{model_cfg['name']}"
    storage_path_uri = storage_uri
    global_best_tracker = {
        'best_f1': 0.0,
        'trial_number': -1,
        'epoch': -1,
        'model_save_path': os.path.join(tmp_dir, 'best_model_across_all_trials.pth')
    }
    n_trials = int(study_cfg.get('n_optuna_trials', 20))
    timeout = study_cfg.get('optuna_timeout')

    try:
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            storage=storage_path_uri,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
        )
        study.set_user_attr('model_type', model_type)
        study.set_user_attr('model_name', model_cfg['name'])

        # MLflow integration (optional)
        optuna_callbacks = []
        mlflow_cfg = dict(study_cfg.get('mlflow') or {})
        # Env override for tracking URI only (experiment comes from YAML)
        if getenv_path('MLFLOW_TRACKING_URI'):
            mlflow_cfg['tracking_uri'] = getenv_path('MLFLOW_TRACKING_URI')

        # Canonicalize tracking URI and pre-create experiment if possible
        mlflow_enabled = bool(mlflow_cfg.get('enabled', False) and (mlflow is not None))
        tracking_uri = canonicalize_tracking_uri(mlflow_cfg.get('tracking_uri'), PROJECT_ROOT) if mlflow_enabled else None
        exp_name = (mlflow_cfg.get('experiment_name') or study_name) if mlflow_enabled else None

        if mlflow_enabled:
            try:
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                if exp_name:
                    mlflow.set_experiment(exp_name)
                print(f"MLflow active → URI: '{mlflow.get_tracking_uri()}', Experiment: '{exp_name}'")
            except Exception as e:
                print(f"MLflow setup warning: {e}")
                mlflow_enabled = False
        else:
            reasons = []
            if not (mlflow_cfg.get('enabled', False)):
                reasons.append("mlflow.enabled=false")
            if mlflow is None:
                msg = "mlflow import failed"
                if 'mlflow_import_error' in globals() and mlflow_import_error is not None:
                    msg += f": {mlflow_import_error}"
                reasons.append(msg)
            if reasons:
                print(f"MLflow inactive ({', '.join(reasons)}). Skipping MLflow logging for this study.")

        # We will manage per-trial MLflow runs ourselves for richer logging
        mlflow_cb = None
        # Warmstart enqueue from previous study results
        warm_cfg = model_cfg.get('warmstart') or study_cfg.get('warmstart')
        if warm_cfg and warm_cfg.get('from_results_csv'):
            try:
                seeds = load_top_params_from_csv(warm_cfg['from_results_csv'], int(warm_cfg.get('top_n', 5)))
                enqueue_warmstart_trials(
                    study,
                    seeds,
                    search_space,
                    enqueue_exact=bool(warm_cfg.get('enqueue_exact', True)),
                    perturbations_per_seed=int(warm_cfg.get('perturbations_per_seed', 0)),
                    jitter_fraction=float(warm_cfg.get('jitter_fraction', 0.2)),
                )
                print(f"Enqueued {len(seeds)} seed trials and perturbations for warmstart.")
            except Exception as e:
                print(f"Warmstart enqueue failed: {e}")
        # Wrap objective to optionally create a manual MLflow run per trial when callback is unavailable
        def wrapped_objective(t: optuna.Trial):
            if mlflow_enabled:
                run_name = f"trial_{t.number}"
                try:
                    # Ensure the experiment matches YAML for each trial
                    if exp_name:
                        try:
                            mlflow.set_experiment(exp_name)
                        except Exception:
                            pass
                    with mlflow.start_run(run_name=run_name, nested=True):
                        # Helpful tags for filtering in UI
                        try:
                            tags = {
                                'study_name': study_name,
                                'model_name': model_cfg['name'],
                                'model_type': model_type,
                                'images_folder': images_folder,
                                'device': str(device),
                                'num_classes': str(num_classes),
                                'dataset_variant': dataset_variant,
                                'resize_mode': str(resize_mode),
                            }
                            # Attach Git + DVC tags
                            tags.update(git_tags)
                            tags.update(dvc_tags)
                            mlflow.set_tags(tags)
                        except Exception:
                            pass
                        result = objective(t, global_best_tracker)
                        try:
                            mlflow.log_metric('value', float(result))
                        except Exception:
                            pass
                        return result
                except Exception:
                    # Fallback: run without MLflow
                    return objective(t, global_best_tracker)
            else:
                return objective(t, global_best_tracker)

        # If n_trials <= 0, skip running new Optuna trials and reuse any
        # existing trials stored in the SQLite backend (useful for pure
        # retraining/evaluation passes from notebooks).
        if n_trials is not None and int(n_trials) > 0:
            study.optimize(
                wrapped_objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=optuna_callbacks,
                gc_after_trial=True,
            )
    except KeyboardInterrupt:
        print("Study interrupted by user.")
    except Exception as e:
        print(f"Error during study: {e}")
        traceback.print_exc()

    # Final results
    results_csv = os.path.join(study_output_dir, f"{study_name}_optuna_study_results.csv")
    try:
        df = study.trials_dataframe(multi_index=False)
        df.to_csv(results_csv, index=False)
        print(f"Saved study results: {results_csv}")
    except Exception:
        pass

    # Final retraining for reproducibility
    print("\n-- Final retraining with best hyperparameters --")
    # Try Optuna's best_trial API first
    bt = None
    bt_params = {}
    bt_number = None
    try:
        bt = study.best_trial
        if bt is not None:
            bt_params = dict(bt.params)
            bt_number = int(bt.number)
    except Exception:
        bt = None

    # Fallback: derive best params from the study CSV if needed
    if not bt_params:
        try:
            df_best = pd.read_csv(results_csv)
            if 'state' in df_best.columns:
                df_best = df_best[df_best['state'] == 'COMPLETE']
            if not df_best.empty and 'value' in df_best.columns:
                row = df_best.sort_values('value', ascending=False).iloc[0]
                bt_number = int(row['number']) if 'number' in row else None
                for c in df_best.columns:
                    if c.startswith('params_'):
                        key = c.replace('params_', '', 1)
                        bt_params[key] = row[c]
        except Exception:
            bt_params = {}

    if not bt_params:
        print("No best trial parameters found; skipping retrain.")
        return {
            'model_name': model_cfg['name'],
            'model_type': model_type,
            'study_name': study_name,
            'study_dir': study_output_dir,
            'results_csv': results_csv,
            'best_f1': None,
            'best_trial': None,
            'best_params': None,
            'best_model_path': None,
        }

    # Build loaders (reuse imbalance strategy)
    batch_size_rt = int(study_cfg.get('batch_size', 8))
    num_workers_rt = int(study_cfg.get('num_workers', 0))
    dl_tr, dl_va, dl_te, num_classes_rt, class_weights_tensor_rt = _make_supervised_loaders(batch_size_rt, num_workers_rt)

    # Rebuild PL (optional) using the same manager used in trials
    train_loader_pl = None
    if pl_enabled and bt_params.get('pl_variant', 'none') != 'none':
        try:
            dl_pl, _ = build_pl_loader_from_cfg(
                pl_cfg=pl_cfg,
                model_cfg=model_cfg,
                train_csv=train_csv,
                images_folder_unlabeled=pl_cfg.get('unlabeled_images_folder'),
                num_classes=num_classes,
                device=device,
                batch_size=int(study_cfg.get('batch_size', 8)),
                num_workers=int(study_cfg.get('num_workers', 0)),
                img_size=img_size,
                pseudo_transform=pseudo_t,
                pl_variant=bt_params['pl_variant'],
                pl_high_thr=float(bt_params.get('pl_high_thr', pl_cfg.get('high_thr', 0.90))),
                pl_density_min_p=float(bt_params.get('pl_density_min_p', pl_cfg.get('density_min_p', 0.80))),
                pl_density_factor=float(bt_params.get('pl_density_factor', pl_cfg.get('density_factor', 0.5))),
                cache_dir=os.path.join(study_output_dir, 'pl_cache'),
            )
            train_loader_pl = dl_pl
        except Exception:
            train_loader_pl = None

    # Model + optimizer (reuse finetune strategy for retraining)
    retrain_cfg = dict(model_cfg)
    retrain_cfg['dropout_rate'] = float(bt_params.get('dropout_rate', retrain_cfg.get('dropout_rate', 0.5)))
    model_rt = create_model(retrain_cfg, num_classes=num_classes, device=device).to(device)
    # Some fixed params may not be in bt.params if not suggested
    lr_rt = float(bt_params.get('finetune_lr', 1e-4))
    mult_rt = float(bt_params.get('multiplier', 1))
    warmup_rt = int(bt_params.get('warmup_epochs', 1))
    # If 'num_epochs_finetune' was specified as fixed in the YAML search_space,
    # it will not appear in bt.params because we don't call trial.suggest_* for fixed.
    # Fallback to the fixed value from the merged search_space, else study/model cfg, else default.
    num_epochs_spec = (search_space.get('num_epochs_finetune') or {}) if isinstance(search_space, dict) else {}
    fixed_epochs = int(num_epochs_spec.get('value')) if (isinstance(num_epochs_spec, dict) and num_epochs_spec.get('type') == 'fixed' and num_epochs_spec.get('value') is not None) else None
    num_epochs_rt = int(bt_params.get('num_epochs_finetune', fixed_epochs or model_cfg.get('num_epochs_finetune') or study_cfg.get('num_epochs_finetune') or 50))
    optimizer_rt = optim.Adam(model_rt.parameters(), lr=lr_rt)
    sched_cos_rt = CosineAnnealingWarmRestarts(optimizer_rt, T_0=5, T_mult=2)
    sched_warm_rt = GradualWarmupScheduler(optimizer_rt, multiplier=mult_rt, total_epoch=warmup_rt, after_scheduler=sched_cos_rt)
    scaler_rt = GradScaler('cuda') if device.type == 'cuda' else GradScaler(enabled=False)

    if class_weights_enabled and class_weights_tensor_rt is not None:
        crit = nn.CrossEntropyLoss(weight=class_weights_tensor_rt.to(device), reduction='none')
    else:
        crit = nn.CrossEntropyLoss(reduction='none')
    best_f1_rt = 0.0
    best_retrain_path = os.path.join(tmp_dir, "best_model_retrain.pth")

    # Use the same Trainer API for retraining to keep behavior consistent
    trainer_rt = Trainer(device=device, criterion=crit, optimizer=optimizer_rt, scheduler=sched_warm_rt, scaler=scaler_rt, callbacks=[])

    # Reuse finetuning strategy resolved earlier
    finetune_cfg_rt = _resolve_finetune_strategy(study_cfg, model_cfg)

    def on_epoch_start_rt(epoch_idx: int, model_ref: nn.Module):
        strategy = finetune_cfg_rt.get('strategy', 'full')
        head_warmup_epochs = int(finetune_cfg_rt.get('head_warmup_epochs', 10))
        gradual_schedule = finetune_cfg_rt.get('gradual_schedule')

        if strategy == 'full':
            return

        if model_type == 'effnet':
            if strategy == 'head_warmup':
                train_backbone = (epoch_idx >= head_warmup_epochs)
                _set_backbone_trainable_effnet(model_ref, train_backbone=train_backbone)
            elif strategy == 'gradual_unfreeze':
                train_backbone = (epoch_idx >= head_warmup_epochs)
                _set_backbone_trainable_effnet(model_ref, train_backbone=train_backbone)
        elif model_type == 'vit':
            if strategy == 'head_warmup':
                train_backbone = (epoch_idx >= head_warmup_epochs)
                _set_backbone_trainable_vit(model_ref, train_backbone=train_backbone)
            elif strategy == 'gradual_unfreeze':
                _set_backbone_trainable_vit(
                    model_ref,
                    train_backbone=(epoch_idx >= head_warmup_epochs),
                    gradual_schedule=gradual_schedule,
                    epoch_idx=epoch_idx,
                )

    def on_epoch_end_rt(epoch_idx: int, log_row: Dict[str, Any]):
        nonlocal best_f1_rt
        # Use validation macro F1 for retrain checkpoint selection.
        f1 = float(log_row['val_f1'])
        if f1 >= best_f1_rt:
            best_f1_rt = f1
            try:
                torch.save(model_rt.state_dict(), best_retrain_path)
            except Exception:
                pass
        # Optional MLflow per-epoch logging
        try:
            if 'mlflow' in globals() and mlflow is not None and mlflow.active_run() is not None:
                step = int(log_row['epoch'])
                mlflow.log_metric('retrain_val_f1', f1, step=step)
                mlflow.log_metric('retrain_val_loss', float(log_row['val_loss']), step=step)
                mlflow.log_metric('retrain_train_loss', float(log_row['train_loss']), step=step)
                mlflow.log_metric('retrain_lr', float(log_row['lr']), step=step)
                if 'lambda_u' in log_row:
                    mlflow.log_metric('retrain_lambda_u', float(log_row['lambda_u']), step=step)
        except Exception:
            pass

    try:
        _ = trainer_rt.fit_mix(
            model=model_rt,
            train_loader_sup=dl_tr,
            val_loader=dl_va,
            epochs=int(num_epochs_rt),
            log_csv_path=os.path.join(study_output_dir, f"retrain_log.csv"),
            ckpt_path=best_retrain_path,
            train_loader_pl=train_loader_pl,
            lambda_ramp_epochs=20,
            test_loader=dl_te,
            on_epoch_start=on_epoch_start_rt,
            on_epoch_end=on_epoch_end_rt,
        )
    except Exception:
        pass

    print(f"Best F1 (study): {global_best_tracker['best_f1']:.4f}")
    print(f"Best F1 (retrain): {best_f1_rt:.4f}")
    if os.path.exists(best_retrain_path):
        print(f"Best retrain model saved to: {best_retrain_path}")

    # Persist best models inside the study directory so they remain available
    # after temporary directories are cleaned up.
    persistent_best_study_model = os.path.join(study_output_dir, "best_model_study.pth")
    persistent_best_retrain_model = os.path.join(study_output_dir, "best_model_retrain.pth")
    try:
        if os.path.exists(global_best_tracker['model_save_path']):
            shutil.copy2(global_best_tracker['model_save_path'], persistent_best_study_model)
            print(f"Copied best study model to: {persistent_best_study_model}")
        if os.path.exists(best_retrain_path):
            shutil.copy2(best_retrain_path, persistent_best_retrain_model)
            print(f"Copied best retrain model to: {persistent_best_retrain_model}")
    except Exception as e:
        print(f"Warning: failed to persist best model(s) to study directory: {e}")

    # Return summary for aggregation
    summary = {
        'model_name': model_cfg['name'],
        'model_type': model_type,
        'study_name': study_name,
        'study_dir': study_output_dir,
        'results_csv': results_csv,
        'best_f1': float(global_best_tracker['best_f1']),
        'best_trial': bt_number if (bt_number is not None) else (int(study.best_trial.number) if hasattr(study, 'best_trial') else None),
        'best_params': json.dumps(bt_params if bt_params else getattr(study.best_trial, 'params', {})),
        # Persist the path to the best retrain model inside the study directory (if available)
        'best_model_path': persistent_best_retrain_model if os.path.exists(persistent_best_retrain_model) else None,
    }
    # MLflow study-level logging (best across all trials)
    mlflow_cfg = study_cfg.get('mlflow') or {}
    if mlflow_cfg.get('enabled', False) and mlflow is not None:
        try:
            tracking_uri = canonicalize_tracking_uri(mlflow_cfg.get('tracking_uri'), PROJECT_ROOT)
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            exp_name = mlflow_cfg.get('experiment_name') or study_name
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=f"{study_name}_summary"):
                mlflow.set_tags({
                    'study_name': study_name,
                    'model_name': model_cfg['name'],
                    'model_type': model_type,
                    'train_csv': os.path.basename(train_csv),
                    'val_csv': os.path.basename(val_csv),
                    'images_folder': images_folder,
                    'device': str(device),
                    'num_classes': str(num_classes),
                    'pl_enabled': str(bool(study_cfg.get('pseudolabel', {}).get('enabled', False))),
                    'dataset_variant': dataset_variant,
                    'resize_mode': str(resize_mode),
                })
                # Also record Git + DVC dataset versions at study summary level
                try:
                    mlflow.set_tags({**git_tags, **dvc_tags})
                except Exception:
                    pass
                # Log the best validation F1 achieved in the study.
                mlflow.log_metric('best_val_f1_study', float(global_best_tracker['best_f1']))
                # Save best model across all trials as artifact
                if os.path.exists(global_best_tracker['model_save_path']):
                    mlflow.log_artifact(global_best_tracker['model_save_path'], artifact_path='artifacts')
                # Also log the best trial's own checkpoint (if exists)
                try:
                    best_trial_ckpt = os.path.join(tmp_dir, f"trial_{global_best_tracker['trial_number']}_best.pth")
                    if os.path.exists(best_trial_ckpt):
                        mlflow.log_artifact(best_trial_ckpt, artifact_path='artifacts')
                except Exception:
                    pass
                # Log the retrain checkpoint
                try:
                    if os.path.exists(best_retrain_path):
                        mlflow.log_artifact(best_retrain_path, artifact_path='artifacts')
                except Exception:
                    pass
                # Save results CSV and SQLite DB
                if os.path.exists(results_csv):
                    mlflow.log_artifact(results_csv, artifact_path='artifacts')
                db_path = os.path.join(study_output_dir, f"{study_name}.db")
                if os.path.exists(db_path):
                    mlflow.log_artifact(db_path, artifact_path='artifacts')
                # Log best trial params as JSON
                try:
                    mlflow.log_dict(getattr(study.best_trial, 'params', {}), artifact_file='artifacts/best_trial_params.json')
                except Exception:
                    pass
            # Clean up temporary checkpoints if MLflow artifacts were stored
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to log MLflow study summary: {e}")
    else:
        # If MLflow is disabled, keep tmp_dir for user inspection; still print its location
        print(f"Artifacts stored temporarily at: {tmp_dir}")
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Unified Optuna study for EffNet + ViT (YAML/JSON-driven)')
    ap.add_argument('--config', '-c', required=True, help='Path to YAML/JSON study config')
    ap.add_argument('--list', action='store_true', help='List models defined in the config and exit')
    args = ap.parse_args()

    # Load .env (if present) before reading/normalizing config
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    cfg = load_config(args.config)

    # Resolve defaults
    global_cfg = cfg.get('study', {})
    global_cfg.setdefault('seed', 42)
    global_cfg.setdefault('optuna_results_dir', os.environ.get('OPTUNA_RESULTS_DIR', os.path.join(PROJECT_ROOT, 'results', 'optuna_studies')))
    global_cfg.setdefault('study_name_prefix', 'UnifiedStudy')
    global_cfg.setdefault('base_data_dir', os.environ.get('DATASET_DIR', os.path.join(PROJECT_ROOT, 'data')))

    models: List[Dict[str, Any]] = cfg.get('models', [])
    if not models:
        raise ValueError('No models defined in the config under key: models')

    if args.list:
        print('Models in config:')
        for m in models:
            print(f"- {m.get('name')} ({m.get('model_type')})")
        return

    results: List[Dict[str, Any]] = []
    for mcfg in models:
        try:
            summary = run_study(global_cfg, mcfg)
            results.append(summary)
        except Exception as e:
            print(f"Study for model '{mcfg.get('name')}' failed: {e}")
            traceback.print_exc()

    # Aggregate results
    try:
        agg_path = os.path.join(global_cfg['optuna_results_dir'], f"{global_cfg['study_name_prefix']}_aggregate_results.csv")
        pd.DataFrame(results).to_csv(agg_path, index=False)
        print(f"Aggregated results saved to: {agg_path}")
    except Exception as e:
        print(f"Failed to save aggregated results: {e}")


if __name__ == '__main__':
    main()
