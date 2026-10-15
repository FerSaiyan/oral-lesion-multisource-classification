#!/usr/bin/env python3
"""Cache letterboxed uint8 RGB images for NPU export, calibration and evaluation.

Uses the project's own validation resize (``_resize_block``: LongestMaxSize +
PadIfNeeded with zeros for ``resize_mode: preserve``) and the same cv2 loading as
``MultiSourceDatasetCSV``. Normalisation is left out on purpose: the exported
NPU graphs take float RGB in [0, 1] (NHWC) and normalise in-graph.

Run in the project env (LocalGPT_llama2):
    python scripts/npu/prepare_npu_inputs.py --size 384 --out ~/nvme_tmp/oral-lesions-npu/work/inputs
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.oral_lesions.data.transforms import _resize_block  # noqa: E402

import albumentations as A  # noqa: E402

SPLITS = {
    "test": REPO / "data/processed/multisource_test_global.csv",
    "train": REPO / "data/processed/multisource_train_global.csv",
}


def load_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cv2.imread returned None: {path}")
    if image.ndim < 3 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resize-mode", default="preserve")
    parser.add_argument("--calib-count", type=int, default=256, help="stratified train images for calibration")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    resize = A.Compose(_resize_block(args.size, args.resize_mode))

    classes = sorted(set(pd.read_csv(SPLITS["train"])["coarse_label"].dropna().astype(str))
                     | set(pd.read_csv(SPLITS["test"])["coarse_label"].dropna().astype(str)))
    for split, csv in SPLITS.items():
        df = pd.read_csv(csv).dropna(subset=["filename", "coarse_label"])
        df = df[df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
        if split == "train":
            # Stratified by class and source so calibration sees every domain.
            groups = df.groupby(["coarse_label", "dataset_source"], group_keys=False)
            frac = min(1.0, args.calib_count / len(df))
            df = groups.apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=0))
            df = df.sample(frac=1.0, random_state=0).head(args.calib_count).reset_index(drop=True)
        images = np.stack([resize(image=load_rgb(p))["image"] for p in df["image_path"]])
        labels = np.array([classes.index(str(c)) for c in df["coarse_label"]], np.int64)
        path = args.out / f"{split}_{args.size}.npz"
        np.savez(path, images=images, labels=labels, classes=np.array(classes),
                 filenames=df["filename"].to_numpy(str), sources=df["dataset_source"].to_numpy(str))
        print(f"{path}: {images.shape} {images.dtype}, classes {classes}, counts {np.bincount(labels).tolist()}")


if __name__ == "__main__":
    main()
