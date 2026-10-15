#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""make_gradcam_explainability_panel.py

Generate a paper-ready explainability panel from precomputed Grad-CAM masks.

Typical use:

  python scripts/make_gradcam_explainability_panel.py \
    --csv "/abs/path/to/multisource_train_balanced.csv" \
    --mask-dir "/abs/path/to/data/artifacts/roi_masks/multisource_train_balanced" \
    --n 4 \
    --out "figures/gradcam_panel.png"

Optional: also run classifier inference to display predicted class, top-k
probabilities, and whether each example is correct/incorrect:

  python scripts/make_gradcam_explainability_panel.py \
    --csv "/abs/path/to/multisource_train_balanced.csv" \
    --mask-dir "/abs/path/to/data/artifacts/roi_masks/multisource_train_balanced" \
    --model-config "/abs/path/to/configs/studies/uni_effnet_imagenet_ar.yaml" \
    --checkpoint "/abs/path/to/best_model_retrain.pth" \
    --only-correct \
    --n 4

Notes
-----
- This script does NOT compute Grad-CAM. It only visualizes precomputed masks.
- Grad-CAM masks are expected to be single-channel images with the same
  filename as the source image (as produced by
  scripts/synthetic_data/build_roi_masks_gradcam.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, cast

import numpy as np
import pandas as pd
from PIL import Image


def _project_root() -> Path:
    # <repo>/papers/Multisource Classification/scripts/
    return Path(__file__).resolve().parents[3]


def _ensure_repo_on_path() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build an explainability panel from Grad-CAM masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_argument_group("Inputs")
    src.add_argument("--csv", type=str, default=None, help="CSV with image paths and labels.")
    src.add_argument(
        "--images",
        type=str,
        nargs="*",
        default=None,
        help="Explicit image paths (overrides --csv sampling).",
    )
    src.add_argument("--image-column", type=str, default="image_path", help="CSV column containing image paths.")
    src.add_argument("--label-column", type=str, default="coarse_label", help="CSV column containing true label.")
    src.add_argument("--source-column", type=str, default="dataset_source", help="CSV column indicating dataset/source.")
    src.add_argument(
        "--mask-name-column",
        type=str,
        default="filename",
        help=(
            "CSV column containing the mask filename in --mask-dir. "
            "If missing, the script falls back to using the image basename."
        ),
    )
    src.add_argument(
        "--mask-dir",
        type=str,
        required=True,
        help="Directory containing Grad-CAM masks (same basenames as images).",
    )

    sel = p.add_argument_group("Selection")
    sel.add_argument("--n", type=int, default=4, help="Number of samples to show.")
    sel.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    sel.add_argument(
        "--external-only",
        action="store_true",
        help="Keep only external datasets (exclude values listed in --internal-sources).",
    )
    sel.add_argument(
        "--internal-sources",
        type=str,
        nargs="*",
        default=["MY_CAMERA", "MY_ROMEU"],
        help="Source names considered internal when using --external-only.",
    )
    sel.add_argument(
        "--include-sources",
        type=str,
        nargs="*",
        default=None,
        help="Only keep these source values from --source-column.",
    )
    sel.add_argument(
        "--exclude-sources",
        type=str,
        nargs="*",
        default=None,
        help="Exclude these source values from --source-column.",
    )
    sel.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Restrict sampling to these true labels (e.g. cancer opmd).",
    )
    sel.add_argument("--only-correct", action="store_true", help="Keep only correctly classified examples.")
    sel.add_argument("--only-incorrect", action="store_true", help="Keep only misclassified examples.")
    sel.add_argument(
        "--balanced-correct-incorrect",
        type=int,
        default=0,
        help="Force a balanced panel with K correct + K incorrect samples.",
    )
    sel.add_argument(
        "--balanced-order",
        type=str,
        default="correct-first",
        choices=["correct-first", "incorrect-first", "shuffle"],
        help="Row order when --balanced-correct-incorrect is used.",
    )
    sel.add_argument(
        "--required-true-labels",
        type=str,
        nargs="*",
        default=None,
        help="Require these true labels to appear at least once in the final panel.",
    )
    sel.add_argument(
        "--require-all-classes",
        action="store_true",
        help="Shortcut for --required-true-labels with all labels in --label-order.",
    )
    sel.add_argument(
        "--max-scan",
        type=int,
        default=2000,
        help="Max number of candidates to scan when using --only-correct/--only-incorrect with live inference.",
    )

    viz = p.add_argument_group("Visualization")
    viz.add_argument(
        "--display-size",
        type=int,
        default=384,
        help="Resize+pad images/masks to this square size for a uniform panel.",
    )
    viz.add_argument("--alpha", type=float, default=0.4, help="Overlay strength for the red Grad-CAM mask.")
    viz.add_argument("--topk", type=int, default=3, help="Top-k probabilities to display when inference is enabled.")
    viz.add_argument(
        "--prob-format",
        type=str,
        default="adaptive",
        choices=["adaptive", "fixed"],
        help="Probability label formatting style for bar annotations.",
    )
    viz.add_argument(
        "--prob-decimals",
        type=int,
        default=3,
        help="Decimals used when --prob-format=fixed.",
    )
    viz.add_argument(
        "--out",
        type=str,
        default="figures/gradcam_panel.png",
        help="Output figure path (relative to this folder is ok).",
    )
    viz.add_argument("--dpi", type=int, default=300, help="Output DPI.")
    viz.add_argument(
        "--no-prob-bars",
        action="store_true",
        help="Hide right-side probability bar charts to create a narrower panel.",
    )
    viz.add_argument(
        "--probs-in-title",
        action="store_true",
        help="When probabilities are available, append compact per-class percentages under Pred in the overlay title.",
    )
    viz.add_argument(
        "--figure-title",
        type=str,
        default="",
        help="Optional title shown above the full panel.",
    )

    inf = p.add_argument_group("Optional classifier inference")
    inf.add_argument(
        "--predictions-csv",
        type=str,
        default=None,
        help=(
            "Optional CSV with model predictions/probabilities. Useful when the "
            "inference dependencies (e.g., geffnet) are not available in the current env."
        ),
    )
    inf.add_argument(
        "--pred-key-column",
        type=str,
        default="filename",
        help="Join key column in --predictions-csv (usually 'filename' or 'image_path').",
    )
    inf.add_argument(
        "--pred-label-column",
        type=str,
        default="y_pred",
        help="Predicted label column in --predictions-csv.",
    )
    inf.add_argument(
        "--pred-prob-prefix",
        type=str,
        default="p_",
        help="Probability column prefix (expects columns like p_cancer, p_opmd, ...).",
    )
    inf.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Study YAML (with keys {study, models}) used to create the classifier.",
    )
    inf.add_argument("--checkpoint", type=str, default=None, help="Classifier checkpoint (.pth state_dict).")
    inf.add_argument(
        "--label-order",
        type=str,
        nargs="*",
        # Default matches training-time `sorted(labels)` in this repo.
        default=["benign_lesion", "cancer", "healthy", "opmd"],
        help="Class label order matching the model output indices.",
    )
    inf.add_argument(
        "--model-img-size",
        type=int,
        default=384,
        help="Model input size (square) used for inference preprocessing.",
    )
    inf.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. cuda:0 or cpu). Defaults to cuda if available.",
    )

    args = p.parse_args()
    if args.only_correct and args.only_incorrect:
        p.error("--only-correct and --only-incorrect are mutually exclusive")
    if args.balanced_correct_incorrect > 0 and (args.only_correct or args.only_incorrect):
        p.error("--balanced-correct-incorrect cannot be combined with --only-correct/--only-incorrect")
    if args.require_all_classes and args.required_true_labels:
        p.error("Use either --require-all-classes or --required-true-labels, not both")
    return args


def _resize_pad_pil(img: Image.Image, size: int, resample: int) -> Image.Image:
    """Aspect-ratio preserving resize + black pad to square."""
    if size <= 0:
        return img
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Invalid image size")
    scale = float(size) / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), resample=resample)

    canvas = Image.new(img.mode, (size, size), color=0)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas.paste(resized, (x0, y0))
    return canvas


def _overlay_red(img_rgb_01: np.ndarray, mask_01: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay a grayscale mask as a red hue on a [H,W,3] float image in [0,1]."""
    if img_rgb_01.ndim != 3 or img_rgb_01.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {img_rgb_01.shape}")
    if mask_01.ndim != 2:
        raise ValueError(f"Expected mask [H,W], got {mask_01.shape}")
    m = np.clip(mask_01, 0.0, 1.0)
    m3 = np.stack([m, m, m], axis=-1)
    color = np.zeros_like(img_rgb_01)
    color[..., 0] = 1.0
    out = img_rgb_01 * (1.0 - alpha * m3) + color * (alpha * m3)
    return np.clip(out, 0.0, 1.0)


def _load_image_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_mask_l(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def _load_dataset(csv_path: Path, image_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if image_col not in df.columns:
        raise KeyError(f"Missing column '{image_col}' in {csv_path}")
    if label_col not in df.columns:
        # label column is optional for purely visual panels
        df[label_col] = None
    return cast(pd.DataFrame, df)


def _filter_has_files(
    df: pd.DataFrame,
    image_col: str,
    mask_dir: Path,
    mask_name_col: str,
) -> pd.DataFrame:
    def _ok(row) -> bool:
        try:
            img = Path(str(row[image_col]))
        except Exception:
            return False
        if not img.is_file():
            return False

        mask_name = None
        if mask_name_col in getattr(row, "index", []) and pd.notna(row[mask_name_col]):
            mask_name = str(row[mask_name_col])
        if not mask_name:
            mask_name = img.name

        return (mask_dir / mask_name).is_file()

    m = df.apply(_ok, axis=1).astype(bool)
    return df.loc[m.values].reset_index(drop=True)


def _choose_samples(
    df: pd.DataFrame,
    n: int,
    seed: int,
    labels: Optional[Sequence[str]],
    label_col: str,
) -> pd.DataFrame:
    out = cast(pd.DataFrame, df)
    if labels is not None and len(labels) > 0:
        out = out[out[label_col].isin(labels)]
    out = out.reset_index(drop=True)
    if len(out) == 0:
        return cast(pd.DataFrame, out)
    if n >= len(out):
        return cast(pd.DataFrame, out)
    return cast(pd.DataFrame, out.sample(n=n, random_state=seed))


def _maybe_build_classifier(args: argparse.Namespace):
    if args.predictions_csv:
        # Prefer explicit prediction inputs over live inference.
        return None
    if not args.model_config and not args.checkpoint:
        return None
    if not args.model_config or not args.checkpoint:
        raise ValueError("To enable inference, both --model-config and --checkpoint are required")

    _ensure_repo_on_path()
    import torch

    try:
        from src.oral_lesions.models.factory import create_model
    except ModuleNotFoundError as e:
        print(f"[warn] Inference disabled (missing dependency): {e}")
        return None

    # Load study YAML
    import yaml

    cfg_path = Path(args.model_config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models = cfg.get("models", []) or []
    if not models:
        raise ValueError(f"No 'models' section found in {cfg_path}")
    model_cfg = models[0]

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    m = create_model(model_cfg, num_classes=len(args.label_order), device=device_t)
    ckpt_path = Path(args.checkpoint)
    sd = torch.load(ckpt_path, map_location=device_t)
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.to(device_t)
    m.eval()
    return m, device_t


def _preprocess_for_model(img: Image.Image, size: int) -> "np.ndarray":
    """Return a normalized CHW float32 tensor as numpy (no torch dependency here)."""
    img = _resize_pad_pil(img, size=size, resample=Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def _predict(model_bundle, img_rgb: Image.Image, label_order: Sequence[str], model_img_size: int):
    if model_bundle is None:
        return None
    model, device = model_bundle

    import torch

    x = _preprocess_for_model(img_rgb, size=model_img_size)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(xt)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = label_order[pred_idx]
    return pred_label, probs


def _fmt_prob(v: float, mode: str = "adaptive", decimals: int = 3) -> str:
    v = float(v)
    if mode == "fixed":
        return f"{v:.{max(0, int(decimals))}f}"
    # adaptive: avoid visual "1.00 / 0.00" artifacts for extreme probabilities
    if v >= 0.9995:
        return ">0.999"
    if v <= 0.0005:
        return "<0.001"
    return f"{v:.3f}"


def _short_label(name: str) -> str:
    m = {
        "healthy": "H",
        "benign_lesion": "B",
        "opmd": "O",
        "cancer": "C",
    }
    return m.get(str(name), str(name)[:1].upper())


def _sample_balanced_with_coverage(
    df_ok: pd.DataFrame,
    df_bad: pd.DataFrame,
    k: int,
    seed: int,
    label_col: str,
    required_labels: Optional[Sequence[str]] = None,
    max_tries: int = 4000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df_ok) < k or len(df_bad) < k:
        raise RuntimeError(
            f"Not enough samples for balanced selection: requested {k}+{k}, "
            f"available correct={len(df_ok)}, incorrect={len(df_bad)}"
        )

    required = {str(x) for x in (required_labels or [])}
    if not required:
        s_ok = cast(pd.DataFrame, df_ok.sample(n=k, random_state=seed).reset_index(drop=True))
        s_bad = cast(pd.DataFrame, df_bad.sample(n=k, random_state=seed + 1).reset_index(drop=True))
        return s_ok, s_bad

    all_labels = set(df_ok[label_col].astype(str).tolist()) | set(df_bad[label_col].astype(str).tolist())
    missing = sorted(required - all_labels)
    if missing:
        raise RuntimeError(
            f"Required labels are not available under current filters: {missing}. "
            f"Available: {sorted(all_labels)}"
        )

    rng = np.random.default_rng(seed)
    ok_idx_all = np.arange(len(df_ok))
    bad_idx_all = np.arange(len(df_bad))
    for _ in range(max_tries):
        ok_idx = rng.choice(ok_idx_all, size=k, replace=False)
        bad_idx = rng.choice(bad_idx_all, size=k, replace=False)
        s_ok = cast(pd.DataFrame, df_ok.iloc[ok_idx].reset_index(drop=True))
        s_bad = cast(pd.DataFrame, df_bad.iloc[bad_idx].reset_index(drop=True))
        labels = set(s_ok[label_col].astype(str).tolist()) | set(s_bad[label_col].astype(str).tolist())
        if required.issubset(labels):
            return s_ok, s_bad

    raise RuntimeError(
        f"Could not satisfy required label coverage after {max_tries} attempts. "
        f"Try a different seed, relax filters, or increase --max-scan."
    )


def main() -> int:
    args = _parse_args()

    # Normalize a few argparse values to concrete types.
    label_order: List[str] = list(args.label_order or [])
    if not label_order:
        raise ValueError("--label-order is empty; provide class names in output-index order")
    required_true_labels: List[str] = []
    if args.require_all_classes:
        required_true_labels = list(label_order)
    elif args.required_true_labels:
        required_true_labels = [str(x) for x in args.required_true_labels]

    mask_dir = Path(args.mask_dir)
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask dir not found: {mask_dir}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build candidate dataframe
    if args.images:
        df = pd.DataFrame({args.image_column: args.images})
        if args.csv:
            # Try to attach labels from CSV if present
            ref = _load_dataset(Path(args.csv), args.image_column, args.label_column)
            keep_cols = [args.image_column, args.label_column]
            if args.mask_name_column in ref.columns:
                keep_cols.append(args.mask_name_column)
            if args.source_column in ref.columns:
                keep_cols.append(args.source_column)
            ref = ref[keep_cols].copy()
            df = df.merge(ref, on=args.image_column, how="left")
        else:
            df[args.label_column] = None
    else:
        if not args.csv:
            raise ValueError("Provide either --images or --csv")
        df = _load_dataset(Path(args.csv), args.image_column, args.label_column)

    df = _filter_has_files(df, args.image_column, mask_dir, args.mask_name_column)
    if args.labels:
        df = df[df[args.label_column].isin(args.labels)].reset_index(drop=True)

    # Optional source filtering (e.g., external-only).
    if (
        args.external_only
        or (args.include_sources is not None and len(args.include_sources) > 0)
        or (args.exclude_sources is not None and len(args.exclude_sources) > 0)
    ):
        if args.source_column not in df.columns:
            raise KeyError(
                f"Source filtering requested but column '{args.source_column}' is missing. "
                f"Available columns: {list(df.columns)}"
            )

        src_series = cast(pd.Series, df[args.source_column].astype(str))
        if args.external_only:
            internal_list = [str(x) for x in (args.internal_sources or [])]
            keep_mask = ~src_series.isin(internal_list)
            df = cast(pd.DataFrame, df.loc[keep_mask.values].reset_index(drop=True))
            src_series = cast(pd.Series, df[args.source_column].astype(str))
        if args.include_sources is not None and len(args.include_sources) > 0:
            keep_list = [str(x) for x in args.include_sources]
            keep_mask = src_series.isin(keep_list)
            df = cast(pd.DataFrame, df.loc[keep_mask.values].reset_index(drop=True))
            src_series = cast(pd.Series, df[args.source_column].astype(str))
        if args.exclude_sources is not None and len(args.exclude_sources) > 0:
            drop_list = [str(x) for x in args.exclude_sources]
            keep_mask = ~src_series.isin(drop_list)
            df = cast(pd.DataFrame, df.loc[keep_mask.values].reset_index(drop=True))

    df = cast(pd.DataFrame, df)

    if len(df) == 0:
        raise RuntimeError("No candidates found with both image and mask present")

    # Merge predictions, if provided.
    if args.predictions_csv:
        pred_path = Path(args.predictions_csv)
        pred_df = pd.read_csv(pred_path)
        if args.pred_key_column not in pred_df.columns:
            raise KeyError(f"Missing '{args.pred_key_column}' in predictions CSV: {pred_path}")
        cols = [args.pred_key_column]
        if args.pred_label_column in pred_df.columns:
            cols.append(args.pred_label_column)
        # Optional probability columns
        for c in [f"{args.pred_prob_prefix}{lab}" for lab in label_order]:
            if c in pred_df.columns:
                cols.append(c)
        pred_df = pred_df[cols].copy()

        if args.pred_key_column not in df.columns:
            raise KeyError(
                f"Join key '{args.pred_key_column}' not found in main CSV. "
                f"Available columns: {list(df.columns)}"
            )
        df = df.merge(pred_df, on=args.pred_key_column, how="left")
        df = cast(pd.DataFrame, df)

    # Ensure inference config uses the normalized label order.
    args.label_order = label_order
    model_bundle = _maybe_build_classifier(args)

    # Select samples
    if int(args.balanced_correct_incorrect) > 0:
        k = int(args.balanced_correct_incorrect)
        if args.predictions_csv:
            if args.pred_label_column not in df.columns:
                raise KeyError(
                    f"--balanced-correct-incorrect requires '{args.pred_label_column}' in merged predictions"
                )
            pred_series = df[args.pred_label_column].astype(str)
            true_series = df[args.label_column].astype(str)
            ok_mask = pred_series == true_series
            df_ok = cast(pd.DataFrame, df.loc[ok_mask.values].reset_index(drop=True))
            df_bad = cast(pd.DataFrame, df.loc[(~ok_mask).values].reset_index(drop=True))

            s_ok, s_bad = _sample_balanced_with_coverage(
                df_ok,
                df_bad,
                k=k,
                seed=args.seed,
                label_col=args.label_column,
                required_labels=required_true_labels,
            )
            if args.balanced_order == "correct-first":
                samples = cast(pd.DataFrame, pd.concat([s_ok, s_bad], ignore_index=True))
            elif args.balanced_order == "incorrect-first":
                samples = cast(pd.DataFrame, pd.concat([s_bad, s_ok], ignore_index=True))
            else:
                samples = cast(pd.DataFrame, pd.concat([s_ok, s_bad], ignore_index=True))
                samples = cast(pd.DataFrame, samples.sample(frac=1.0, random_state=args.seed).reset_index(drop=True))
        else:
            if model_bundle is None:
                raise ValueError(
                    "--balanced-correct-incorrect requires either --predictions-csv "
                    "or a working inference setup (--model-config/--checkpoint + deps)."
                )
            df_shuf = cast(pd.DataFrame, df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True))
            picked_ok = []
            picked_bad = []
            for j, (_, row) in enumerate(df_shuf.iterrows()):
                if j >= int(args.max_scan):
                    break
                img_path = Path(str(row[args.image_column]))
                img = _load_image_rgb(img_path)
                pred = _predict(model_bundle, img, label_order, args.model_img_size)
                if pred is None:
                    continue
                pred_label, _probs = pred
                true_label = row.get(args.label_column, None)
                if true_label is None or (isinstance(true_label, float) and np.isnan(true_label)):
                    continue
                is_correct = str(true_label) == str(pred_label)
                if is_correct:
                    picked_ok.append(row)
                if not is_correct:
                    picked_bad.append(row)
                # Early stop once we have enough buffer for random coverage attempts.
                if len(picked_ok) >= max(k * 5, k + 8) and len(picked_bad) >= max(k * 5, k + 8):
                    break

            if len(picked_ok) < k or len(picked_bad) < k:
                raise RuntimeError(
                    f"Not enough samples for balanced selection after scanning {min(len(df_shuf), int(args.max_scan))} rows: "
                    f"correct={len(picked_ok)}, incorrect={len(picked_bad)}, requested={k} each"
                )

            s_ok_pool = cast(pd.DataFrame, pd.DataFrame(picked_ok))
            s_bad_pool = cast(pd.DataFrame, pd.DataFrame(picked_bad))
            s_ok, s_bad = _sample_balanced_with_coverage(
                s_ok_pool,
                s_bad_pool,
                k=k,
                seed=args.seed,
                label_col=args.label_column,
                required_labels=required_true_labels,
            )
            if args.balanced_order == "correct-first":
                samples = cast(pd.DataFrame, pd.concat([s_ok, s_bad], ignore_index=True))
            elif args.balanced_order == "incorrect-first":
                samples = cast(pd.DataFrame, pd.concat([s_bad, s_ok], ignore_index=True))
            else:
                samples = cast(pd.DataFrame, pd.concat([s_ok, s_bad], ignore_index=True))
                samples = cast(pd.DataFrame, samples.sample(frac=1.0, random_state=args.seed).reset_index(drop=True))

    elif args.only_correct or args.only_incorrect:
        if args.predictions_csv:
            if args.pred_label_column not in df.columns:
                raise KeyError(
                    f"--only-correct/--only-incorrect require '{args.pred_label_column}' in merged predictions"
                )
            pred_series = df[args.pred_label_column]
            true_series = df[args.label_column]
            ok = pred_series.astype(str) == true_series.astype(str)
            df = df[ok].reset_index(drop=True) if args.only_correct else df[~ok].reset_index(drop=True)
            df = cast(pd.DataFrame, df)
            if len(df) == 0:
                raise RuntimeError("No samples matched the requested correct/incorrect filter")
            samples = _choose_samples(df, n=args.n, seed=args.seed, labels=None, label_col=args.label_column)
        else:
            if model_bundle is None:
                raise ValueError(
                    "--only-correct/--only-incorrect require either --predictions-csv "
                    "or a working inference setup (--model-config/--checkpoint + deps)."
                )
            # Live inference: scan a shuffled subset and pick matches.
            df_shuf = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
            picked = []
            for j, (_, row) in enumerate(df_shuf.iterrows()):
                if j >= int(args.max_scan):
                    break
                img_path = Path(str(row[args.image_column]))
                img = _load_image_rgb(img_path)
                pred = _predict(model_bundle, img, label_order, args.model_img_size)
                if pred is None:
                    continue
                pred_label, _probs = pred
                true_label = row.get(args.label_column, None)
                if true_label is None or (isinstance(true_label, float) and np.isnan(true_label)):
                    continue
                is_correct = str(true_label) == str(pred_label)
                if args.only_correct and not is_correct:
                    continue
                if args.only_incorrect and is_correct:
                    continue
                picked.append(row)
                if len(picked) >= args.n:
                    break
            if not picked:
                raise RuntimeError("No samples matched the requested correct/incorrect filter")
            samples = cast(pd.DataFrame, pd.DataFrame(picked))
    else:
        samples = _choose_samples(df, n=args.n, seed=args.seed, labels=None, label_col=args.label_column)

    samples = cast(pd.DataFrame, samples)

    # Print a compact manifest of selected samples for traceability.
    manifest_cols = [c for c in [args.source_column, args.mask_name_column, args.image_column, args.label_column] if c in samples.columns]
    print("Selected samples:")
    if manifest_cols:
        for i, (_, r) in enumerate(samples[manifest_cols].iterrows(), start=1):
            src = str(r[args.source_column]) if args.source_column in manifest_cols else ""
            mname = str(r[args.mask_name_column]) if args.mask_name_column in manifest_cols else Path(str(r[args.image_column])).name
            lab = str(r[args.label_column]) if args.label_column in manifest_cols else ""
            print(f"  {i:02d}. source={src:<10} label={lab:<13} file={mname}")
    else:
        print("  (no manifest columns available)")

    # ---- Plot ----
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    n_rows = len(samples)
    prob_cols = [f"{args.pred_prob_prefix}{lab}" for lab in label_order]
    has_pred_probs = bool(args.predictions_csv) and all(c in samples.columns for c in prob_cols)
    show_prob_bars = ((model_bundle is not None) or has_pred_probs) and (not args.no_prob_bars)
    n_cols = 3 if show_prob_bars else 2
    fig_w = 11 if show_prob_bars else 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, 3.6 * n_rows))
    fig = cast(Figure, fig)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = Path(str(row[args.image_column]))
        mask_name = None
        if args.mask_name_column in row.index:
            mask_val = row.get(args.mask_name_column, None)
            if mask_val is not None and not (isinstance(mask_val, float) and np.isnan(mask_val)):
                mask_name = str(mask_val)
        if not mask_name:
            mask_name = img_path.name
        mask_path = mask_dir / mask_name
        true_label = row.get(args.label_column, None)
        if isinstance(true_label, float) and np.isnan(true_label):
            true_label = None

        img = _load_image_rgb(img_path)
        mask = _load_mask_l(mask_path)

        if args.display_size and args.display_size > 0:
            img_v = _resize_pad_pil(img, size=args.display_size, resample=Image.BILINEAR)
            mask_v = _resize_pad_pil(mask, size=args.display_size, resample=Image.NEAREST)
        else:
            img_v = img
            mask_v = mask

        base = np.array(img_v, dtype=np.float32) / 255.0
        cam = np.array(mask_v, dtype=np.float32) / 255.0
        overlay = _overlay_red(base, cam, alpha=float(args.alpha))

        pred_label = None
        probs = None
        if args.predictions_csv:
            pred_val = row.get(args.pred_label_column, None)
            if pred_val is not None and not (isinstance(pred_val, float) and np.isnan(pred_val)):
                pred_label = str(pred_val)
            prob_cols = [f"{args.pred_prob_prefix}{lab}" for lab in label_order]
            if show_prob_bars and all(c in row.index for c in prob_cols):
                try:
                    probs = np.array([float(row[c]) for c in prob_cols], dtype=np.float32)
                except Exception:
                    probs = None
        elif model_bundle is not None:
            pred = _predict(model_bundle, img, label_order, args.model_img_size)
            if pred is not None:
                pred_label, probs = pred

        # Column 0: original
        axes[i, 0].imshow(base)
        title = "Original"
        if true_label is not None:
            title += f"\nTrue: {true_label}"
        axes[i, 0].set_title(title, fontsize=10)
        axes[i, 0].axis("off")

        # Column 1: Grad-CAM overlay
        axes[i, 1].imshow(overlay)
        title = "Grad-CAM (red)"
        if pred_label is not None:
            if true_label is None:
                title += f"\nPred: {pred_label}"
            else:
                ok = str(true_label) == str(pred_label)
                title += f"\nPred: {pred_label} ({'correct' if ok else 'incorrect'})"
            if args.probs_in_title and probs is not None:
                probs_arr = cast(np.ndarray, probs)
                items = []
                for j, lab in enumerate(label_order):
                    items.append(f"{_short_label(lab)} {_fmt_prob(float(probs_arr[j]), mode=args.prob_format, decimals=args.prob_decimals)}")
                title += "\n" + " | ".join(items)
        axes[i, 1].set_title(title, fontsize=10)
        axes[i, 1].axis("off")

        # Column 2: prob bars
        if show_prob_bars:
            ax = axes[i, 2]
            assert probs is not None
            probs = cast(np.ndarray, probs)
            topk = int(max(1, args.topk))
            idxs = np.argsort(-probs)[:topk]
            labels = [label_order[j] for j in idxs][::-1]
            vals = [float(probs[j]) for j in idxs][::-1]
            ax.barh(labels, vals, color="#666666")
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("p", fontsize=9)
            ax.set_title("Top probabilities", fontsize=10)
            for y, v in enumerate(vals):
                ax.text(
                    min(v + 0.02, 1.02),
                    y,
                    _fmt_prob(v, mode=args.prob_format, decimals=args.prob_decimals),
                    va="center",
                    fontsize=9,
                )
            ax.grid(axis="x", alpha=0.2)

    if args.figure_title:
        fig.suptitle(args.figure_title, fontsize=14)
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    else:
        plt.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
