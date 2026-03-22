import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader


def _repo_root(start: Optional[Path] = None) -> Path:
    """
    Find the repository root by searching upwards for a folder containing
    both 'src' and 'configs' (or a 'dvc.yaml' file).
    """
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
        if (parent / "dvc.yaml").exists():
            return parent
    return Path.cwd().resolve()


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_dotenv
from src.exp.config import load_config
from src.oral_lesions.data import MultiSourceDatasetCSV, build_transforms
from src.oral_lesions.models.factory import create_model


def _build_classes(csv_paths: Sequence[Path], label_col: str = "coarse_label") -> List[str]:
    labels = set()
    for p in csv_paths:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if label_col not in df.columns:
            continue
        labels |= set(df[label_col].dropna().astype(str).unique())
    classes = sorted(labels)
    if not classes:
        raise ValueError(f"No classes found in CSVs: {[str(p) for p in csv_paths]}")
    return classes


def _load_model_from_ckpt(cfg: Dict, ckpt_path: Path, num_classes: int) -> Tuple[torch.nn.Module, torch.device]:
    model_cfg = cfg["models"][0]
    model_type = model_cfg.get("model_type", "effnet")
    mcfg: Dict = {"model_type": model_type, "dropout_rate": model_cfg.get("dropout_rate", 0.5)}

    if model_type == "effnet":
        mcfg["enet_backbone"] = model_cfg.get("enet_backbone", "tf_efficientnet-b3_ns")
        mcfg["pretrained_source"] = model_cfg.get("pretrained_source", "imagenet")
    elif model_type == "vit":
        # Ensure we instantiate the same ViT variant that was used during training
        # so that checkpoint shapes (e.g., embed_dim 1024 for vit_large) match.
        mcfg["vit_model_name"] = model_cfg.get("vit_model_name", "vit_base_patch16_224")
        mcfg["pretrained_source"] = model_cfg.get("pretrained_source", "imagenet_timm")

    # Avoid env-driven custom weights
    for k in ("CUSTOM_EFFNET_PATH", "CUSTOM_VIT_PATH", "CUSTOM_VIT_BACKBONE_PATH", "PRETRAINED_MODELS_DIR"):
        os.environ.pop(k, None)
    for k in ("custom_model_path", "custom_effnet_path", "custom_vit_path"):
        mcfg.pop(k, None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(mcfg, num_classes=num_classes, device=device).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_state[k[len("model.") :]] = v
            else:
                new_state[k] = v
        state = new_state

    model.load_state_dict(state, strict=False)
    model.eval()
    return model, device


def _build_loader(
    csv_path: Path,
    images_folder: Path,
    classes: List[str],
    model_type: str,
    img_size: int,
    resize_mode: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    _, val_t = build_transforms(model_type, img_size, resize_mode=resize_mode)
    ds = MultiSourceDatasetCSV(str(csv_path), str(images_folder), transform=val_t, classes=classes)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def _eval_split(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    classes: List[str],
) -> Dict:
    y_true: List[int] = []
    y_pred: List[int] = []
    top2_correct = 0
    top3_correct = 0
    total = 0

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                xb, yb = batch[0], batch[1]
            else:
                raise ValueError("Expected batch to be (images, labels, ..) for evaluation.")

            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            topk = torch.topk(logits, k=min(3, num_classes), dim=1).indices
            yb_exp = yb.view(-1, 1)
            bs = yb.size(0)
            total += bs

            if num_classes >= 2:
                in_top2 = topk[:, :2].eq(yb_exp).any(dim=1)
                top2_correct += in_top2.sum().item()
            if num_classes >= 3:
                in_top3 = topk[:, :3].eq(yb_exp).any(dim=1)
                top3_correct += in_top3.sum().item()

    elapsed = time.perf_counter() - start_time
    avg_sec_per_image = elapsed / total if total > 0 else None

    # Raw confusion matrix in the native class-index order used during training.
    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Reorder matrix for display so that classes appear by clinical severity:
    # healthy → benign_lesion → opmd → cancer. Any missing or extra classes are
    # appended afterward in their original order.
    severity_order = ["healthy", "benign_lesion", "opmd", "cancer"]
    idx_by_name = {str(name).lower(): idx for idx, name in enumerate(classes)}
    ordered_indices: List[int] = []
    for name in severity_order:
        idx = idx_by_name.get(name)
        if idx is not None and idx not in ordered_indices:
            ordered_indices.append(idx)
    for idx in range(len(classes)):
        if idx not in ordered_indices:
            ordered_indices.append(idx)

    if ordered_indices and ordered_indices != list(range(len(classes))):
        cm = cm_raw[np.ix_(ordered_indices, ordered_indices)]
        classes_ordered = [classes[i] for i in ordered_indices]
    else:
        cm = cm_raw
        classes_ordered = list(classes)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    top2_acc = top2_correct / total if num_classes >= 2 and total > 0 else None
    top3_acc = top3_correct / total if num_classes >= 3 and total > 0 else None

    # Binary collapse: healthy/benign_lesion → 0 (non-suspicious),
    # opmd/cancer → 1 (suspicious).
    binary_map: Dict[int, int] = {}
    for idx, name in enumerate(classes):
        name_l = str(name).lower()
        if name_l in ("healthy", "benign_lesion"):
            binary_map[idx] = 0
        elif name_l in ("opmd", "cancer"):
            binary_map[idx] = 1

    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []
    if binary_map:
        for t, p in zip(y_true, y_pred):
            if t in binary_map and p in binary_map:
                y_true_bin.append(binary_map[t])
                y_pred_bin.append(binary_map[p])
    f1_binary = None
    if y_true_bin:
        from sklearn.metrics import f1_score as _f1_score_binary

        f1_binary = _f1_score_binary(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

    return {
        "cm": cm,
        "acc": acc,  # Top-1 accuracy
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "top2_acc": top2_acc,
        "top3_acc": top3_acc,
        "f1_binary": f1_binary,
        "y_true": y_true,
        "y_pred": y_pred,
        "classes_ordered": classes_ordered,
        "avg_sec_per_image": avg_sec_per_image,
    }


def _plot_confusion(cm, class_names: List[str], title: str, save_path: Optional[Path] = None) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(c) for c in class_names])
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(class_names)), max(5, 0.9 * len(class_names))))
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.tight_layout()
    plt.show()


def _format_topk(res: Dict) -> str:
    """
    Format top-1 / top-2 / top-3 accuracies for printing.
    """
    parts = [f"Top1: {res['acc']:.4f}"]
    if res.get("top2_acc") is not None:
        parts.append(f"Top2: {res['top2_acc']:.4f}")
    if res.get("top3_acc") is not None:
        parts.append(f"Top3: {res['top3_acc']:.4f}")
    return " | ".join(parts)


def main(base_config_path: Optional[str] = None) -> None:
    """
    Batch confusion-matrix evaluation for multisource studies, mirroring the
    training batch runner in notebooks/experiments/run_optuna_studies.ipynb.

    For each dataset variant:
      - Finds the corresponding Optuna study directory and best_model_retrain.pth.
      - Evaluates on its own val/test CSVs.
      - Additionally evaluates heldout CSVs:
          * multisource_no_kaggle  → Kaggle holdout
          * multisource_no_zenodo  → Zenodo holdout
          * multisource_no_mendeley → Mendeley holdout
    """

    repo = REPO_ROOT
    load_dotenv(repo / ".env")

    # Allow overriding the base study config (which controls which Optuna
    # study/results are evaluated) from the caller. Defaults to the original
    # multisource config for backwards compatibility.
    if base_config_path is not None:
        base_config = Path(base_config_path)
        if not base_config.is_absolute():
            base_config = repo / base_config
    else:
        base_config = repo / "configs" / "studies" / "effnet_imagenet_b3_ar_multisource.yaml"

    # Evaluation output root: results/evals/<config_stem> for confusion
    # matrices, plus a shared metrics log under results/evals/.
    eval_root = repo / "results" / "evals"
    eval_root.mkdir(parents=True, exist_ok=True)
    config_id = base_config.stem
    config_eval_dir = eval_root / config_id
    config_eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = eval_root / "batch_confusion_metrics.txt"

    cfg = load_config(str(base_config))
    base_study_cfg = dict(cfg.get("study", {}))
    model_cfg = cfg["models"][0]

    # Candidate results roots: config value, env override, and common defaults.
    results_roots: List[Path] = []
    cfg_root = Path(base_study_cfg.get("optuna_results_dir", "results/optuna_studies_multisource"))
    if not cfg_root.is_absolute():
        cfg_root = repo / cfg_root
    results_roots.append(cfg_root)

    env_root_str = os.environ.get("OPTUNA_RESULTS_DIR")
    if env_root_str:
        env_root = Path(env_root_str)
        if not env_root.is_absolute():
            env_root = repo / env_root
        if env_root not in results_roots:
            results_roots.append(env_root)

    default_root = repo / "results" / "optuna_studies_multisource"
    if default_root not in results_roots:
        results_roots.append(default_root)


    base_transfer_dir = Path(
        os.environ.get("MULTISOURCE_BASE_DIR", str(repo / "data"))
    )

    multisource_images = Path(
        os.environ.get(
            "MULTISOURCE_IMAGES",
            str(base_transfer_dir / "Multisource_dataset_for_transfer_learning"),
        )
    )
    multi_no_kaggle_images = Path(
        os.environ.get(
            "MULTISOURCE_NO_KAGGLE_IMAGES",
            str(base_transfer_dir / "Multisource_no_kaggle_dataset_for_transfer_learning"),
        )
    )
    multi_no_zenodo_images = Path(
        os.environ.get(
            "MULTISOURCE_NO_ZENODO_IMAGES",
            str(base_transfer_dir / "Multisource_no_zenodo_dataset_for_transfer_learning"),
        )
    )
    multi_no_mendeley_images = Path(
        os.environ.get(
            "MULTISOURCE_NO_MENDELEY_IMAGES",
            str(base_transfer_dir / "Multisource_no_mendeley_dataset_for_transfer_learning"),
        )
    )

    kaggle_holdout_images = Path(
        os.environ.get(
            "KAGGLE_HOLDOUT_IMAGES",
            str(base_transfer_dir / "Kaggle_holdout_dataset_for_transfer_learning"),
        )
    )
    zenodo_holdout_images = Path(
        os.environ.get(
            "ZENODO_HOLDOUT_IMAGES",
            str(base_transfer_dir / "Zenodo_holdout_dataset_for_transfer_learning"),
        )
    )
    mendeley_holdout_images = Path(
        os.environ.get(
            "MENDELEY_HOLDOUT_IMAGES",
            str(base_transfer_dir / "Mendeley_holdout_dataset_for_transfer_learning"),
        )
    )

    dataset_specs: List[Dict] = [
        {
            "key": "multisource_internal_val",
            "images_folder": multisource_images,
            "train_csv": repo / "data/processed/multisource_train.csv",
            "val_csv": repo / "data/processed/multisource_val.csv",
            "test_csv": repo / "data/processed/multisource_test.csv",
            "extra_holdouts": {},
        },
        {
            "key": "multisource_global_val",
            "images_folder": multisource_images,
            "train_csv": repo / "data/processed/multisource_train_global.csv",
            "val_csv": repo / "data/processed/multisource_val_global.csv",
            "test_csv": repo / "data/processed/multisource_test_global.csv",
            "extra_holdouts": {},
        },
        {
            "key": "multisource_no_kaggle",
            "images_folder": multi_no_kaggle_images,
            "train_csv": repo / "data/processed/multisource_train_no_kaggle.csv",
            "val_csv": repo / "data/processed/multisource_val_no_kaggle.csv",
            "test_csv": repo / "data/processed/multisource_test_no_kaggle.csv",
            "extra_holdouts": {
                "kaggle_holdout": {
                    "csv": repo / "data/processed/multisource_holdout_kaggle.csv",
                    "images_folder": kaggle_holdout_images,
                },
            },
        },
        {
            "key": "multisource_no_zenodo",
            "images_folder": multi_no_zenodo_images,
            "train_csv": repo / "data/processed/multisource_train_no_zenodo.csv",
            "val_csv": repo / "data/processed/multisource_val_no_zenodo.csv",
            "test_csv": repo / "data/processed/multisource_test_no_zenodo.csv",
            "extra_holdouts": {
                "zenodo_holdout": {
                    "csv": repo / "data/processed/multisource_holdout_zenodo.csv",
                    "images_folder": zenodo_holdout_images,
                },
            },
        },
        {
            "key": "multisource_no_mendeley",
            "images_folder": multi_no_mendeley_images,
            "train_csv": repo / "data/processed/multisource_train_no_mendeley.csv",
            "val_csv": repo / "data/processed/multisource_val_no_mendeley.csv",
            "test_csv": repo / "data/processed/multisource_test_no_mendeley.csv",
            "extra_holdouts": {
                "mendeley_holdout": {
                    "csv": repo / "data/processed/multisource_holdout_mendeley.csv",
                    "images_folder": mendeley_holdout_images,
                },
            },
        },
    ]

    only_keys_env = os.environ.get("BATCH_EVAL_ONLY_KEYS")
    only_keys: Optional[set] = None
    if only_keys_env:
        only_keys = {k.strip() for k in only_keys_env.split(",") if k.strip()}
    if only_keys is not None:
        dataset_specs = [d for d in dataset_specs if d["key"] in only_keys]

    model_type = model_cfg.get("model_type", "effnet")
    default_size = 384 if model_type == "effnet" else 224
    img_size = int(model_cfg.get("img_size", default_size))
    resize_mode = base_study_cfg.get("resize_mode", "preserve")

    base_prefix = base_study_cfg.get("study_name_prefix", "EffNet_b3_multisource")
    base_exp_name = (base_study_cfg.get("mlflow") or {}).get("experiment_name", base_prefix)

    any_found = False

    # Accumulate human-readable metrics lines for this config so we can append
    # them to a shared log file with clear section boundaries.
    metrics_lines: List[str] = []
    metrics_lines.append("")
    metrics_lines.append("=" * 80)
    metrics_lines.append(f"CONFIG: {config_id}  ({base_config})")
    metrics_lines.append("=" * 80)

    for spec in dataset_specs:
        key = spec["key"]
        images_folder = Path(spec["images_folder"])
        train_csv = Path(spec["train_csv"])
        val_csv = Path(spec["val_csv"])
        test_csv = Path(spec["test_csv"])
        extra_holdouts = spec.get("extra_holdouts") or {}

        study_cfg = dict(base_study_cfg)
        study_cfg["study_name_prefix"] = f"{base_prefix}_{key}"
        mlflow_cfg = dict(study_cfg.get("mlflow") or {})
        mlflow_cfg["experiment_name"] = f"{base_exp_name}_{key}"
        study_cfg["mlflow"] = mlflow_cfg

        study_name = f"{study_cfg['study_name_prefix']}_{model_cfg['name']}"

        # Try to locate the study directory under any candidate results root.
        study_dir: Optional[Path] = None
        for root in results_roots:
            cand = root / study_name
            if cand.exists():
                study_dir = cand
                break
        # Fallback: search by pattern (key + model name) if exact name not found.
        if study_dir is None:
            for root in results_roots:
                for d in root.glob(f"*{key}*{model_cfg['name']}*"):
                    if d.is_dir():
                        study_dir = d
                        break
                if study_dir is not None:
                    break

        header = f"\n=== Evaluating dataset variant: {key} ==="
        print(header)
        metrics_lines.append(header)
        if study_dir is None:
            print("  [warn] No study directory found for this variant in any of:")
            for r in results_roots:
                print("         -", r)
                metrics_lines.append(f"  [warn] No study directory under: {r}")
            continue

        print("Study dir:", study_dir)
        metrics_lines.append(f"Study dir: {study_dir}")

        # Determine which checkpoints are available (retrain and/or study).
        ckpt_specs: List[Tuple[str, Path]] = []
        retrain_ckpt = study_dir / "best_model_retrain.pth"
        study_ckpt = study_dir / "best_model_study.pth"
        if retrain_ckpt.exists():
            ckpt_specs.append(("retrain", retrain_ckpt))
        if study_ckpt.exists():
            ckpt_specs.append(("study", study_ckpt))
        if not ckpt_specs:
            print("  [warn] No best_model_retrain.pth or best_model_study.pth found; skipping.")
            metrics_lines.append("  [warn] No best_model_retrain.pth or best_model_study.pth found; skipping.")
            continue

        any_found = True

        # Build global class list across this variant's splits + heldouts.
        class_csvs: List[Path] = [train_csv, val_csv, test_csv]
        for h in extra_holdouts.values():
            class_csvs.append(Path(h["csv"]))
        classes = _build_classes(class_csvs)
        num_classes = len(classes)

        batch_size = int(study_cfg.get("batch_size", 32))
        num_workers = int(study_cfg.get("num_workers", 0))

        # Train / val / test loaders shared across checkpoints.
        dl_train = _build_loader(
            train_csv,
            images_folder,
            classes,
            model_type,
            img_size,
            resize_mode,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        dl_val = _build_loader(
            val_csv,
            images_folder,
            classes,
            model_type,
            img_size,
            resize_mode,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        dl_test = None
        if test_csv.exists():
            dl_test = _build_loader(
                test_csv,
                images_folder,
                classes,
                model_type,
                img_size,
                resize_mode,
                batch_size=batch_size,
                num_workers=num_workers,
            )

        for ckpt_label, ckpt_path in ckpt_specs:
            ckpt_header = f"--- Checkpoint: {ckpt_label} ({ckpt_path.name}) ---"
            print(ckpt_header)
            metrics_lines.append(ckpt_header)

            model, device = _load_model_from_ckpt(cfg, ckpt_path, num_classes=num_classes)

            ckpt_eval_dir = config_eval_dir / ckpt_label
            ckpt_eval_dir.mkdir(parents=True, exist_ok=True)

            res_train = _eval_split(dl_train, model, device, num_classes, classes)
            res_val = _eval_split(dl_val, model, device, num_classes, classes)

            train_f1_bin = "n/a" if res_train["f1_binary"] is None else f"{res_train['f1_binary']:.4f}"
            train_topk = _format_topk(res_train)
            train_time = (
                "n/a"
                if res_train["avg_sec_per_image"] is None
                else f"{res_train['avg_sec_per_image']*1000.0:.2f} ms/img"
            )
            train_line = (
                f"[{ckpt_label} | {key} / train] {train_topk} | "
                f"F1 macro: {res_train['f1_macro']:.4f} | "
                f"F1 weighted: {res_train['f1_weighted']:.4f} | "
                f"F1 binary (suspicious): {train_f1_bin} | "
                f"avg time: {train_time}"
            )
            print(train_line)
            metrics_lines.append(train_line)

            val_f1_bin = "n/a" if res_val["f1_binary"] is None else f"{res_val['f1_binary']:.4f}"
            val_topk = _format_topk(res_val)
            val_time = (
                "n/a"
                if res_val["avg_sec_per_image"] is None
                else f"{res_val['avg_sec_per_image']*1000.0:.2f} ms/img"
            )
            val_line = (
                f"[{ckpt_label} | {key} / val] {val_topk} | "
                f"F1 macro: {res_val['f1_macro']:.4f} | "
                f"F1 weighted: {res_val['f1_weighted']:.4f} | "
                f"F1 binary (suspicious): {val_f1_bin} | "
                f"avg time: {val_time}"
            )
            print(val_line)
            metrics_lines.append(val_line)
            _plot_confusion(
                res_val["cm"],
                res_val.get("classes_ordered", classes),
                f"{key} – Confusion (val, {ckpt_label})",
                save_path=ckpt_eval_dir / f"{key}_val_confusion.png",
            )

            if dl_test is not None:
                res_test = _eval_split(dl_test, model, device, num_classes, classes)
                test_f1_bin = "n/a" if res_test["f1_binary"] is None else f"{res_test['f1_binary']:.4f}"
                test_topk = _format_topk(res_test)
                test_time = (
                    "n/a"
                    if res_test["avg_sec_per_image"] is None
                    else f"{res_test['avg_sec_per_image']*1000.0:.2f} ms/img"
                )
                test_line = (
                    f"[{ckpt_label} | {key} / test] {test_topk} | "
                    f"F1 macro: {res_test['f1_macro']:.4f} | "
                    f"F1 weighted: {res_test['f1_weighted']:.4f} | "
                    f"F1 binary (suspicious): {test_f1_bin} | "
                    f"avg time: {test_time}"
                )
                print(test_line)
                metrics_lines.append(test_line)
                _plot_confusion(
                    res_test["cm"],
                    res_test.get("classes_ordered", classes),
                    f"{key} – Confusion (test, {ckpt_label})",
                    save_path=ckpt_eval_dir / f"{key}_test_confusion.png",
                )

            # Extra heldout splits for this variant.
            for holdout_name, holdout_spec in extra_holdouts.items():
                holdout_csv = Path(holdout_spec["csv"])
                holdout_imgs = Path(holdout_spec["images_folder"])
                if not holdout_csv.exists():
                    print(f"  [warn] Holdout CSV for {holdout_name} not found: {holdout_csv}")
                    continue
                if not holdout_imgs.exists():
                    print(f"  [warn] Holdout images folder for {holdout_name} not found: {holdout_imgs}")
                    continue

                dl_holdout = _build_loader(
                    holdout_csv,
                    holdout_imgs,
                    classes,
                    model_type,
                    img_size,
                    resize_mode,
                    batch_size=int(study_cfg.get("batch_size", 32)),
                    num_workers=num_workers,
                )
                res_holdout = _eval_split(dl_holdout, model, device, num_classes, classes)
                holdout_f1_bin = "n/a" if res_holdout["f1_binary"] is None else f"{res_holdout['f1_binary']:.4f}"
                holdout_topk = _format_topk(res_holdout)
                holdout_time = (
                    "n/a"
                    if res_holdout["avg_sec_per_image"] is None
                    else f"{res_holdout['avg_sec_per_image']*1000.0:.2f} ms/img"
                )
                holdout_line = (
                    f"[{ckpt_label} | {key} / {holdout_name}] {holdout_topk} | "
                    f"F1 macro: {res_holdout['f1_macro']:.4f} | "
                    f"F1 weighted: {res_holdout['f1_weighted']:.4f} | "
                    f"F1 binary (suspicious): {holdout_f1_bin} | "
                    f"avg time: {holdout_time}"
                )
                print(holdout_line)
                metrics_lines.append(holdout_line)
                _plot_confusion(
                    res_holdout["cm"],
                    res_holdout.get("classes_ordered", classes),
                    f"{key} – Confusion ({holdout_name}, {ckpt_label})",
                    save_path=ckpt_eval_dir / f"{key}_{holdout_name}_confusion.png",
                )

    if not any_found:
        print(
            "\n[info] No matching study checkpoints were found for any dataset variant.\n"
            "       Ensure that multisource Optuna studies have been run and that their\n"
            "       results directories are present under one of the following roots:\n"
        )
        for r in results_roots:
            print("       -", r)
            metrics_lines.append(f"       - {r}")
    # Append metrics for this config to the shared log file.
    with metrics_log_path.open("a", encoding="utf-8") as f:
        for line in metrics_lines:
            f.write(line.rstrip() + "\n")


if __name__ == "__main__":
    main()
