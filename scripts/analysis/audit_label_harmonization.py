#!/usr/bin/env python3
"""Audit Qwen3 four-class diagnosis mapping against deterministic phrase rules.

This script is intentionally dependency-free.  It reads the diagnosis-review
SQLite database, applies an auditable, phrase-only rule set, and reports
agreement at occurrence, unique-phrase, patient, and modeled-image levels.

The rules are a reproducibility/sensitivity baseline, not a clinical gold
standard.  Expert adjudication remains necessary, particularly for phrases
that are non-specific, contain competing diagnoses, or require clinical or
histopathological context.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


CLASSES = ("healthy", "benign_lesion", "opmd", "cancer")
SEVERITY = {name: rank for rank, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    category: str
    pattern: re.Pattern[str]


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", text or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _rule(rule_id: str, category: str, pattern: str) -> Rule:
    return Rule(rule_id, category, re.compile(pattern))


# Explicit precedence is clinically conservative: cancer > OPMD > benign >
# healthy.  A negated-atypia phrase is handled before the OPMD rules.
RULES: tuple[Rule, ...] = (
    _rule(
        "C01_frank_malignancy",
        "cancer",
        r"\b(carcinoma|adenocarcinoma|linfoma|melanoma|sarcoma)\b|"
        r"\bcec\b|carcinoma\s+espino\s*celular|carcinoma\s+epidermoide",
    ),
    _rule(
        "O01_dysplasia_or_atypia",
        "opmd",
        r"\b(displasia|desplasia)\b|\batipias?\b|\batipica\b",
    ),
    _rule(
        "O02_leukoplakia_erythroplakia",
        "opmd",
        r"\b(leuco|leuko)plasia\b|\beritroplasia\b|\blvp\b",
    ),
    _rule(
        "O03_actinic_cheilitis",
        "opmd",
        r"\b(queilite|quelite|quielite)\s+actinica\b|"
        r"\bles(ao|oes)\s+actinica\b|\bqa\b",
    ),
    _rule(
        "O04_lichen_lichenoid",
        "opmd",
        r"\b(liquen|lichen)\s+plano\b|\bliquenoide\b|\bmucosite\s+liquenoide\b|"
        r"\blpo\b|\bllo\b",
    ),
    _rule(
        "O05_explicit_potentially_malignant",
        "opmd",
        r"\bpotencialmente\s+maligna\b|\bdoenca\s+potencialmente\s+maligna\b",
    ),
    _rule(
        "O06_atypical_proliferative_lesion",
        "opmd",
        r"\blesao\s+escamosa\s+proliferativa\b|"
        r"\blesao\s+glandular\s+proliferativa\s+atipica\b",
    ),
    _rule(
        "O07_nicotinic_stomatitis",
        "opmd",
        r"\b(estomatite|dermatite)\s+nicotinica\b",
    ),
    _rule(
        "H01_explicit_absence_of_lesion",
        "healthy",
        r"\bsem\s+(sinais\s+de\s+)?les(ao|oes)\b|\bmucosa\s+normal\b|"
        r"\bexame\s+normal\b|\bremissao\s+completa\b",
    ),
    _rule("H02_normal_variant_torus", "healthy", r"\btorus\b"),
    _rule(
        "B01_reactive_hyperplasia",
        "benign_lesion",
        r"\bhiperplasia\b|\bhiperplasica\b|\bhfi\b|\bhf\b|\bfibroma\b|"
        r"\bfibrolipomatosa\b",
    ),
    _rule(
        "B02_friction_or_trauma",
        "benign_lesion",
        r"\b(hiper)?(queratose|ceratose)\s+friccional\b|\btrauma\b|"
        r"\btraumatica\b|\bmordida\b|\bmordiscada\b|\blesao\s+reativa\b",
    ),
    _rule(
        "B03_infectious",
        "benign_lesion",
        r"\bcandid|\bherpes\b|\bhersim\b|\bmicose\b|\bleishmaniose\b|"
        r"\bmononucleose\b|\bactinomyces\b",
    ),
    _rule(
        "B04_cyst_or_salivary",
        "benign_lesion",
        r"\bcisto\b|\bcistica\b|\bmucocele\b|\bsialocisto\b|\bceratocisto\b",
    ),
    _rule(
        "B05_vascular",
        "benign_lesion",
        r"\bhemangioma\b|\bvascular\b|\bvarizes?\b",
    ),
    _rule(
        "B06_benign_tumor_or_pigmented",
        "benign_lesion",
        r"\blipoma\b|\bpapiloma\b|\bmelanoacantoma\b|\bmelanotica\b|"
        r"\bmelanocitica\b|\bnevo\b|\btatuagem\b",
    ),
    _rule(
        "B07_inflammatory_or_immune",
        "benign_lesion",
        r"\bgranuloma\b|\bgranulomatosa\b|\binflamacao\b|\bestomatite\s+protetica\b|"
        r"\bpenfigo\b|\bpenfigoide\b|\bpsoriase\b|\bglossite\b|"
        r"\bqueilite\s+(angular|glandular)\b",
    ),
    _rule(
        "B08_other_specific_benign",
        "benign_lesion",
        r"\bexostose\b|\bosteoradionecrose\b|\bfistula\b|"
        r"\blesao\s+fibro\s+ossea\b|\beritema\s+migratorio\s+benigno\b",
    ),
)


def classify_phrase(text: str) -> tuple[str, str]:
    value = normalize(text)
    # "Sem atipias" is evidence against the otherwise broad atypia rule.
    value_for_opmd = re.sub(r"\bsem\s+atipias?\b", "", value)
    matches: list[Rule] = []
    for rule in RULES:
        candidate = value_for_opmd if rule.category == "opmd" else value
        if rule.pattern.search(candidate):
            matches.append(rule)
    if not matches:
        return "unknown", "UNMAPPED"
    winner = max(matches, key=lambda item: SEVERITY[item.category])
    return winner.category, winner.rule_id


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(left == right for left, right in pairs) / total
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(left_counts[c] * right_counts[c] for c in CLASSES) / (total * total)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def comparison_summary(rows: Iterable[tuple[str, str]]) -> dict[str, object]:
    materialized = list(rows)
    mapped = [(a, b) for a, b in materialized if a in CLASSES and b in CLASSES]
    return {
        "n_total": len(materialized),
        "n_mapped_by_rules": len(mapped),
        "rule_coverage": len(mapped) / len(materialized) if materialized else None,
        "n_agree": sum(a == b for a, b in mapped),
        "agreement_on_mapped": (
            sum(a == b for a, b in mapped) / len(mapped) if mapped else None
        ),
        "cohen_kappa_on_mapped": cohen_kappa(mapped),
    }


def patient_labels(rows: list[dict[str, object]]) -> tuple[dict[str, str], dict[str, str]]:
    by_patient: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_patient[str(row["rg"])].append(row)

    llm_labels: dict[str, str] = {}
    rule_labels: dict[str, str] = {}
    for rg, patient_rows in by_patient.items():
        # Reproduce the current pipeline's confidence/count ranking.  The final
        # alphabetical tie-break makes an otherwise implicit tie deterministic.
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in patient_rows:
            grouped[str(row["llm_category"])].append(float(row["confidence"]))
        ranked = sorted(
            grouped.items(),
            key=lambda item: (
                -sum(item[1]) / len(item[1]),
                -len(item[1]),
                -max(item[1]),
                item[0],
            ),
        )
        llm_labels[rg] = ranked[0][0]

        mapped = [str(row["rule_category"]) for row in patient_rows if row["rule_category"] in CLASSES]
        rule_labels[rg] = max(mapped, key=lambda label: SEVERITY[label]) if mapped else "unknown"
    return llm_labels, rule_labels


def read_modeled_images(paths: Sequence[Path]) -> list[dict[str, str]]:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                source = row.get("dataset_source", "")
                if source not in {"MY_CAMERA", "MY_ROMEU"}:
                    continue
                key = row.get("image_path") or row.get("filename") or ""
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/app/romeu_unknown_phrases.sqlite"))
    parser.add_argument("--run-id", type=int, default=39)
    parser.add_argument(
        "--manifests",
        type=Path,
        nargs="*",
        default=[
            Path("data/processed/multisource_train.csv"),
            Path("data/processed/multisource_val.csv"),
            Path("data/processed/multisource_test.csv"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision_bpex/label_harmonization"),
    )
    args = parser.parse_args()

    connection = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    query = """
        SELECT c.rg, ph.id AS phrase_id, ph.phrase AS phrase_text,
               pp.category AS llm_category, pp.confidence
        FROM patient_review_choice AS c
        JOIN phrase AS ph ON ph.id = c.phrase_id
        JOIN phrase_prediction AS pp
          ON pp.run_id = ? AND pp.rg = c.rg AND pp.phrase_id = c.phrase_id
        WHERE pp.status = 'ok'
        ORDER BY c.rg, ph.id
    """
    raw_rows = [dict(row) for row in connection.execute(query, (args.run_id,))]
    connection.close()
    if not raw_rows:
        raise SystemExit(f"No predictions found for run_id={args.run_id}")

    for row in raw_rows:
        category, rule_id = classify_phrase(str(row["phrase_text"]))
        row["rule_category"] = category
        row["rule_id"] = rule_id

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Phrase audit contains no patient identifier.  One row is retained per
    # phrase ID and LLM category because the same phrase can be context-dependent.
    unique_rows: dict[tuple[int, str], dict[str, object]] = {}
    for row in raw_rows:
        key = (int(row["phrase_id"]), str(row["llm_category"]))
        unique_rows[key] = {
            "phrase_id": row["phrase_id"],
            "phrase_text": row["phrase_text"],
            "llm_category": row["llm_category"],
            "rule_category": row["rule_category"],
            "rule_id": row["rule_id"],
        }
    phrase_audit = sorted(unique_rows.values(), key=lambda row: (str(row["llm_category"]), str(row["phrase_text"])))
    with (args.output_dir / "phrase_mapping_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phrase_audit[0]))
        writer.writeheader()
        writer.writerows(phrase_audit)

    llm_patient, rule_patient = patient_labels(raw_rows)
    patient_pairs = [(rule_patient[rg], llm_patient[rg]) for rg in sorted(llm_patient)]

    image_pairs: list[tuple[str, str]] = []
    image_unknown = 0
    for row in read_modeled_images(args.manifests):
        rg = str(row.get("patient_id", "")).upper()
        if rg not in rule_patient:
            image_unknown += 1
            continue
        image_pairs.append((rule_patient[rg], str(row.get("coarse_label") or row.get("diagnosis_categories"))))

    occurrence_pairs = [(str(row["rule_category"]), str(row["llm_category"])) for row in raw_rows]
    phrase_pairs = [(str(row["rule_category"]), str(row["llm_category"])) for row in phrase_audit]
    summary = {
        "purpose": "Sensitivity baseline only; deterministic rules are not a clinical reference standard.",
        "database": str(args.db),
        "llm_run_id": args.run_id,
        "class_order": list(CLASSES),
        "precedence": "cancer > opmd > benign_lesion > healthy; patient rule label uses maximum mapped severity",
        "occurrence_level": comparison_summary(occurrence_pairs),
        "unique_phrase_id_category_level": comparison_summary(phrase_pairs),
        "patient_level": comparison_summary(patient_pairs),
        "modeled_image_level": {
            **comparison_summary(image_pairs),
            "n_images_without_rule_patient_label": image_unknown,
        },
        "counts": {
            "selected_phrase_occurrences": len(raw_rows),
            "unique_phrase_texts": len({str(row["phrase_text"]) for row in raw_rows}),
            "unique_phrase_ids": len({int(row["phrase_id"]) for row in raw_rows}),
            "patients": len(llm_patient),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
