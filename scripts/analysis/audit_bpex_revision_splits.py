#!/usr/bin/env python3
"""Audit the separately generated BPEX revision-v2 split manifests.

The audit verifies row preservation, duplicate paths, exact-file separation, and
known-patient separation for the internal and Zenodo sources. Kaggle and
Mendeley are checked only at the file level because reliable patient identifiers
are unavailable for those releases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


CLASSES = ("healthy", "benign_lesion", "opmd", "cancer")
SOURCES = ("MY_CAMERA", "MY_ROMEU", "ZENODO", "KAGGLE", "MENDELEY")
INTERNAL_SOURCES = {"MY_CAMERA", "MY_ROMEU"}
DIGIT_RE = re.compile(r"(?:PH|PA|PE|RG)?(\d+)", re.IGNORECASE)


VARIANTS = {
    "internal_validation": {
        "train": "multisource_train.csv",
        "val": "multisource_val.csv",
        "test": "multisource_test.csv",
    },
    "global_validation": {
        "train": "multisource_train_global.csv",
        "val": "multisource_val_global.csv",
        "test": "multisource_test_global.csv",
    },
    "no_kaggle": {
        "train": "multisource_train_no_kaggle.csv",
        "val": "multisource_val_no_kaggle.csv",
        "test": "multisource_test_no_kaggle.csv",
        "holdout": "multisource_holdout_kaggle.csv",
    },
    "no_zenodo": {
        "train": "multisource_train_no_zenodo.csv",
        "val": "multisource_val_no_zenodo.csv",
        "test": "multisource_test_no_zenodo.csv",
        "holdout": "multisource_holdout_zenodo.csv",
    },
    "no_mendeley": {
        "train": "multisource_train_no_mendeley.csv",
        "val": "multisource_val_no_mendeley.csv",
        "test": "multisource_test_no_mendeley.csv",
        "holdout": "multisource_holdout_mendeley.csv",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _known_group(row: pd.Series) -> str | None:
    source = str(row.get("dataset_source", ""))
    patient = row.get("patient_id")
    if pd.isna(patient):
        return None
    patient_text = str(patient)
    if source in INTERNAL_SOURCES:
        match = DIGIT_RE.search(patient_text)
        return f"INTERNAL:{match.group(1).zfill(6)}" if match else None
    if source == "ZENODO":
        return f"ZENODO:{patient_text}"
    return None


def _counts(df: pd.DataFrame) -> dict[str, Any]:
    labels = df["coarse_label"].astype(str)
    sources = df["dataset_source"].astype(str)
    by_class = labels.value_counts().reindex(CLASSES, fill_value=0)
    by_source = sources.value_counts().reindex(SOURCES, fill_value=0)
    cross_tab = pd.crosstab(sources, labels).reindex(
        index=SOURCES, columns=CLASSES, fill_value=0
    )
    return {
        "rows": int(len(df)),
        "by_class": {name: int(value) for name, value in by_class.items()},
        "by_source": {name: int(value) for name, value in by_source.items()},
        "source_by_class": {
            source: {label: int(cross_tab.loc[source, label]) for label in CLASSES}
            for source in SOURCES
        },
    }


def _pairwise_overlaps(
    frames: dict[str, pd.DataFrame], value_column: str
) -> dict[str, int]:
    values = {
        name: set(frame[value_column].dropna().astype(str))
        for name, frame in frames.items()
    }
    return {
        f"{left}__{right}": len(values[left] & values[right])
        for left, right in combinations(values, 2)
    }


def _patient_overlaps(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, int], dict[str, int]]:
    groups: dict[str, set[str]] = {}
    for name, frame in frames.items():
        group_values = frame.apply(_known_group, axis=1).dropna().astype(str)
        groups[name] = set(group_values)
    overlaps = {
        f"{left}__{right}": len(groups[left] & groups[right])
        for left, right in combinations(groups, 2)
    }
    counts = {name: len(values) for name, values in groups.items()}
    return overlaps, counts


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image_path", "dataset_source", "coarse_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def audit(root: Path) -> dict[str, Any]:
    base_path = root / "test_dataset_multisource.csv"
    base = _read(base_path)
    base_paths = set(base["image_path"].astype(str))
    base_groups = base.apply(_known_group, axis=1).dropna().astype(str)
    duplicate_filename_mask = base["filename"].duplicated(keep=False)
    duplicate_filename_rows = base[duplicate_filename_mask].copy()
    duplicate_filename_values = sorted(
        duplicate_filename_rows["filename"].astype(str).unique()
    )
    conflicting_duplicate_group_count = 0
    conflicting_duplicate_label_count = 0
    for _, rows in duplicate_filename_rows.groupby("filename", dropna=False):
        known_groups = rows.apply(_known_group, axis=1).dropna().astype(str).unique()
        labels = rows["coarse_label"].dropna().astype(str).unique()
        conflicting_duplicate_group_count += int(len(known_groups) > 1)
        conflicting_duplicate_label_count += int(len(labels) > 1)

    report: dict[str, Any] = {
        "root": str(root.resolve()),
        "base": {
            **_counts(base),
            "duplicate_image_paths": int(base["image_path"].duplicated().sum()),
            "duplicate_filenames": int(base["filename"].duplicated().sum()),
            "duplicate_filename_values": duplicate_filename_values,
            "duplicate_filename_groups_with_conflicting_known_group": (
                conflicting_duplicate_group_count
            ),
            "duplicate_filename_groups_with_conflicting_label": (
                conflicting_duplicate_label_count
            ),
            "known_patient_groups": int(base_groups.nunique()),
            "internal_patient_groups": int(
                base_groups[base_groups.str.startswith("INTERNAL:")].nunique()
            ),
            "zenodo_patient_groups": int(
                base_groups[base_groups.str.startswith("ZENODO:")].nunique()
            ),
        },
        "internal_only": {},
        "variants": {},
        "csv_sha256": {},
        "checks": {},
    }

    internal_base = base[base["dataset_source"].isin(INTERNAL_SOURCES)].copy()
    internal_frames: dict[str, pd.DataFrame] = {}
    internal_manifest_rows: dict[str, int] = {}
    internal_filename_multiplicity_matches: dict[str, bool] = {}
    for split, filename in (
        ("train", "new_test_train.csv"),
        ("test", "new_test_test.csv"),
    ):
        manifest = pd.read_csv(root / filename)
        manifest_names = manifest["filename"].astype(str)
        unknown = set(manifest_names) - set(internal_base["filename"].astype(str))
        if unknown:
            raise ValueError(f"{filename} contains {len(unknown)} unknown filenames")
        selected = internal_base[
            internal_base["filename"].astype(str).isin(set(manifest_names))
        ].copy()
        internal_frames[split] = selected
        internal_manifest_rows[split] = int(len(manifest))
        internal_filename_multiplicity_matches[split] = (
            manifest_names.value_counts().sort_index().to_dict()
            == selected["filename"].astype(str).value_counts().sort_index().to_dict()
        )

    internal_file_overlap = _pairwise_overlaps(internal_frames, "image_path")
    internal_patient_overlap, internal_patient_counts = _patient_overlaps(internal_frames)
    internal_union = set().union(
        *(set(frame["image_path"].astype(str)) for frame in internal_frames.values())
    )
    report["internal_only"] = {
        "splits": {name: _counts(frame) for name, frame in internal_frames.items()},
        "file_overlaps": internal_file_overlap,
        "known_patient_overlaps": internal_patient_overlap,
        "known_patient_groups_by_split": internal_patient_counts,
        "manifest_rows": internal_manifest_rows,
        "filename_multiplicity_matches_base": internal_filename_multiplicity_matches,
        "union_matches_internal_base": internal_union
        == set(internal_base["image_path"].astype(str)),
    }

    for variant, filenames in VARIANTS.items():
        frames = {name: _read(root / filename) for name, filename in filenames.items()}
        file_overlaps = _pairwise_overlaps(frames, "image_path")
        patient_overlaps, patient_counts = _patient_overlaps(frames)
        duplicate_paths = {
            name: int(frame["image_path"].duplicated().sum())
            for name, frame in frames.items()
        }

        development_names = [name for name in ("train", "val", "test") if name in frames]
        development_union = set().union(
            *(
                set(frames[name]["image_path"].astype(str))
                for name in development_names
            )
        )
        if variant.startswith("no_"):
            excluded_source = variant.removeprefix("no_").upper()
            expected_development = set(
                base.loc[base["dataset_source"] != excluded_source, "image_path"].astype(str)
            )
        else:
            expected_development = base_paths

        report["variants"][variant] = {
            "splits": {name: _counts(frame) for name, frame in frames.items()},
            "duplicate_image_paths": duplicate_paths,
            "file_overlaps": file_overlaps,
            "known_patient_overlaps": patient_overlaps,
            "known_patient_groups_by_split": patient_counts,
            "development_union_matches_expected_base": development_union
            == expected_development,
        }

    for path in sorted(root.glob("*.csv")):
        report["csv_sha256"][path.name] = _sha256(path)

    all_file_overlaps = [
        value
        for variant in report["variants"].values()
        for value in variant["file_overlaps"].values()
    ]
    all_patient_overlaps = [
        value
        for variant in report["variants"].values()
        for value in variant["known_patient_overlaps"].values()
    ]
    report["checks"] = {
        "base_has_no_duplicate_paths": report["base"]["duplicate_image_paths"] == 0,
        "base_duplicate_filenames_are_nonconflicting": (
            report["base"]["duplicate_filename_groups_with_conflicting_known_group"]
            == 0
            and report["base"]["duplicate_filename_groups_with_conflicting_label"]
            == 0
        ),
        "all_manifests_have_no_duplicate_paths": all(
            value == 0
            for variant in report["variants"].values()
            for value in variant["duplicate_image_paths"].values()
        ),
        "all_variant_partitions_are_file_disjoint": all(
            value == 0 for value in all_file_overlaps
        ),
        "all_known_patient_partitions_are_disjoint": all(
            value == 0 for value in all_patient_overlaps
        ),
        "all_development_unions_match_expected_base": all(
            variant["development_union_matches_expected_base"]
            for variant in report["variants"].values()
        ),
        "internal_only_is_file_disjoint": all(
            value == 0 for value in internal_file_overlap.values()
        ),
        "internal_only_is_patient_disjoint": all(
            value == 0 for value in internal_patient_overlap.values()
        ),
        "internal_only_union_matches_base": report["internal_only"][
            "union_matches_internal_base"
        ],
        "internal_only_filename_multiplicity_matches_base": all(
            report["internal_only"]["filename_multiplicity_matches_base"].values()
        ),
    }
    report["checks"]["all_passed"] = all(report["checks"].values())
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    base = report["base"]
    lines = [
        "# BPEX revision-v2 split audit",
        "",
        f"- Base manifest: **{base['rows']:,} images**",
        f"- Known internal patient groups: **{base['internal_patient_groups']:,}**",
        f"- Known Zenodo patient groups: **{base['zenodo_patient_groups']:,}**",
        f"- Duplicate image paths: **{base['duplicate_image_paths']:,}**",
        f"- Duplicate filename values: **{base['duplicate_filenames']:,}** "
        "(checked separately from path identity)",
        f"- All checks passed: **{report['checks']['all_passed']}**",
        "",
        "## Base source/class counts",
        "",
        "| Source | Healthy | Benign lesion | OPMD | Cancer | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCES:
        counts = base["source_by_class"][source]
        lines.append(
            f"| {source} | {counts['healthy']:,} | {counts['benign_lesion']:,} | "
            f"{counts['opmd']:,} | {counts['cancer']:,} | {base['by_source'][source]:,} |"
        )
    class_counts = base["by_class"]
    lines.append(
        f"| **Total** | **{class_counts['healthy']:,}** | "
        f"**{class_counts['benign_lesion']:,}** | **{class_counts['opmd']:,}** | "
        f"**{class_counts['cancer']:,}** | **{base['rows']:,}** |"
    )
    lines.extend(
        [
            "",
            "## Split sizes",
            "",
            "| Variant | Train | Validation | Test | Holdout |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, variant in report["variants"].items():
        splits = variant["splits"]
        lines.append(
            f"| {name} | {splits['train']['rows']:,} | {splits['val']['rows']:,} | "
            f"{splits['test']['rows']:,} | {splits.get('holdout', {}).get('rows', 0):,} |"
        )
    lines.extend(["", "## Boolean checks", ""])
    for name, passed in report["checks"].items():
        lines.append(f"- `{name}`: **{passed}**")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/processed/bpex_revision_v2"),
        help="Directory containing the revision-v2 CSV manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision_bpex/split_audit"),
        help="Directory for summary.json and summary.md.",
    )
    args = parser.parse_args(argv)

    report = audit(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(report, args.output_dir / "summary.md")
    print(json.dumps(report["checks"], indent=2, sort_keys=True))
    return 0 if report["checks"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
