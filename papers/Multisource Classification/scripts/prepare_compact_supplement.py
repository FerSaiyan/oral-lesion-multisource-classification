#!/usr/bin/env python3
"""Prepare a compact presentation of existing evidence, without refitting models."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
OUT = PAPER / "supplement"
RESULTS = ROOT / "results/revision_bpex"


def tex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "_": r"\_", "#": r"\#", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(c, c) for c in str(value))


def table(label, caption, headers, rows, widths=None):
    rows = [list(map(str, row)) for row in rows]
    # Compare architectures only within the same evaluation population/class.
    highlights = {
        "tab:s_comparison_pooled": ((), {1: 1, 2: 1}),
        "tab:s_pooled_metrics": ((0,), dict.fromkeys(range(2, 8), 1)),
        "tab:s_screening_metrics": ((0,), dict.fromkeys(range(2, 9), 1)),
        "tab:s_per_class_metrics": ((0, 2), dict.fromkeys(range(4, 9), 1)),
        "tab:s_source_metrics": ((0,), dict.fromkeys(range(3, 9), 1)),
        "tab:s_source_ablation": ((0,), dict.fromkeys(range(2, 7), 1)),
        "tab:s_temperature_probability": ((0,), dict.fromkeys(range(2, 7), -1)),
        "tab:s_temperature_screening": ((0,), dict.fromkeys(range(2, 8), 1)),
        "tab:s_temperature_classwise": ((1,), {4: -1, 5: -1}),
        "tab:s_rank_cancer": ((), {2: 1, 4: -1}),
        "tab:s_rank_holdouts": ((0,), {3: 1, 4: -1}),
    }
    if label in highlights:
        group_columns, directions = highlights[label]
        entries, best = [], {}
        for i, row in enumerate(rows):
            group = tuple(row[j] for j in group_columns)
            for column, direction in directions.items():
                # Raw/scaled means are ranked separately; SDs and CIs are not ranked.
                for part, segment in enumerate(re.finditer(r"[^/]+", row[column])):
                    number = re.search(r"[+-]?\d+(?:\.\d+)?", segment.group())
                    if number is None:
                        continue
                    start, end = (segment.start() + position for position in number.span())
                    key = (group, column, part)
                    value = direction * float(number.group())
                    best[key] = max(best.get(key, value), value)
                    entries.append((i, column, start, end, key, value))
        for i, column, start, end, key, value in reversed(entries):
            if value == best[key]:
                cell = rows[i][column]
                command = r"\mathbf" if cell[:start].count("$") % 2 else r"\textbf"
                rows[i][column] = cell[:start] + command + "{" + cell[start:end] + "}" + cell[end:]
        caption += " Best means in bold (ties included)."
    columns = widths or "l" * len(headers)
    header = " & ".join(headers) + r" \\"
    lines = [r"\begingroup", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{" + columns + "}",
             r"\caption{" + caption + r"}\label{" + label + r"}\\",
             r"\toprule", header, r"\midrule", r"\endfirsthead",
             r"\multicolumn{" + str(len(headers)) + r"}{l}{\textit{Continued}}\\",
             r"\toprule", header, r"\midrule", r"\endhead",
             r"\midrule", r"\multicolumn{" + str(len(headers)) +
             r"}{r}{\textit{Continued on next page}}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    for row in rows:
        cells = [str(x).replace("EfficientNet-", "").replace("ViT-Base", "ViT-B")
                 .replace("ViT-Large", "ViT-L").replace("benign lesion", "Benign")
                  .replace("healthy", "Healthy").replace("opmd", "OPMD") for x in row]
        if label in ("tab:s_pooled_metrics", "tab:s_screening_metrics", "tab:s_per_class_metrics",
                     "tab:s_bootstrap_intervals", "tab:s_temperature_probability",
                     "tab:s_temperature_screening", "tab:s_rank_holdouts"):
            if cells[0] in ("Kaggle", "Zenodo", "Mendeley"):
                cells[0] += " holdout"
        elif label == "tab:s_source_ablation":
            cells[0] = "No " + cells[0]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\end{longtable}", r"\endgroup"]
    (OUT / (label.split(":")[-1] + ".tex")).write_text("\n".join(lines) + "\n")


def existing(filename, label):
    text = (PAPER / "tables" / filename).read_text()
    blocks = re.findall(r"\\begin\{longtable\}.*?\\end\{longtable\}", text, re.S)
    block, = [b for b in blocks if r"\label{" + label + "}" in b]
    body = block.split(r"\endhead", 1)[1].split(r"\bottomrule", 1)[0]
    rows = []
    for line in body.splitlines():
        if " & " in line:
            rows.append([cell.strip() for cell in line.strip().removesuffix(r"\\").split(" & ")])
    return rows


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    models = ("effnet_b3", "effnet_b7", "vit_base", "vit_large")
    names = ("EfficientNet-B3", "EfficientNet-B7", "ViT-Base", "ViT-Large")
    cm = pd.read_csv(RESULTS / "manuscript_comparison_v1/confusion_matrices.csv")
    curves = pd.read_csv(RESULTS / "manuscript_comparison_v1/curves.csv")
    colors = plt.get_cmap("tab10").colors[:5]
    for endpoint in ("pooled", "kaggle", "zenodo", "mendeley"):
        fig, axes = plt.subplots(2, 4, figsize=(7, 4.2),
                                 gridspec_kw={"width_ratios": [2, 1, 2, 1]},
                                 layout="constrained")
        for k, (model, name) in enumerate(zip(models, names)):
            for kind, column in (("multiclass", 2 * (k % 2)), ("screening", 2 * (k % 2) + 1)):
                ax = axes[k // 2, column]
                sub = cm[(cm.endpoint == endpoint) & (cm.model == model) & (cm.kind == kind)]
                assert set(sub.seed) == set(range(42, 47))
                order = (2, 0, 3, 1) if kind == "multiclass" else (0, 1)
                labels = ("H", "B", "O", "C") if kind == "multiclass" else ("NS", "S")
                counts = sub.groupby(["true", "predicted"])["count"].mean().unstack().reindex(index=order, columns=order).to_numpy()
                fractions = sub.groupby(["true", "predicted"]).row_fraction.mean().unstack().reindex(index=order, columns=order).to_numpy()
                ax.imshow(np.nan_to_num(fractions), vmin=0, vmax=1, cmap="Blues")
                for i in range(len(order)):
                    for j in range(len(order)):
                        value = fractions[i, j]
                        label = f"{counts[i,j]:.1f}\n{100*value:.1f}%" if np.isfinite(value) else f"{counts[i,j]:.1f}\nNA"
                        ax.text(j, i, label, ha="center", va="center", fontsize=7,
                                color="white" if value > 0.55 else "black")
                ax.set(xticks=range(len(order)), yticks=range(len(order)),
                       xticklabels=labels, yticklabels=labels,
                       xlabel="Predicted", ylabel="True" if kind == "multiclass" else "",
                       title=name if kind == "multiclass" else "Screening")
                ax.tick_params(length=0)
        fig.savefig(OUT / f"{endpoint}_confusion.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(7, 3.5), layout="constrained")
        for column, (model, name) in enumerate(zip(models, names)):
            for row, kind in enumerate(("roc", "pr")):
                ax = axes[row, column]
                for seed, color in zip(range(42, 47), colors):
                    sub = curves[(curves.endpoint == endpoint) & (curves.model == model)
                                 & (curves.seed == seed) & (curves.task == "screening") & (curves.curve == kind)]
                    assert not sub.empty
                    ax.plot(sub.x, sub.y, color=color, lw=0.8)
                if kind == "roc":
                    ax.plot([0, 1], [0, 1], ":", color="gray", lw=0.8)
                else:
                    sub = cm[(cm.endpoint == endpoint) & (cm.model == model)
                             & (cm.seed == 42) & (cm.kind == "screening")]
                    ax.axhline(sub[sub.true == 1]["count"].sum() / sub["count"].sum(),
                               ls=":", color="gray", lw=0.8)
                ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0, 0.5, 1), yticks=(0, 0.5, 1),
                       xlabel="False-positive rate" if row == 0 else "Recall",
                       ylabel=("Sensitivity" if row == 0 else "Precision") if column == 0 else "",
                       title=name if row == 0 else "")
        fig.legend([Line2D([], [], color=c) for c in colors], [str(s) for s in range(42, 47)],
                   loc="outside lower center", ncol=5, frameon=False, fontsize=7, title="Seed", title_fontsize=7)
        fig.savefig(OUT / f"{endpoint}_curves.pdf")
        plt.close(fig)

    bins = pd.read_csv(RESULTS / "temperature_calibration_v1/reliability_bins.csv")
    fig, axes = plt.subplots(2, 4, figsize=(7, 3.8), layout="constrained")
    for column, (model, name) in enumerate(zip(models, names)):
        for row, task in enumerate(("top1", "screening")):
            ax = axes[row, column]
            for kind, color in (("raw", "#2874a6"), ("calibrated", "#c46a18")):
                for seed in range(42, 47):
                    sub = bins[(bins.endpoint == "pooled") & (bins.model == model)
                               & (bins.seed == seed) & (bins.task == task)
                               & (bins.kind == kind) & (bins.n_bin > 0)]
                    assert not sub.empty
                    ax.plot(sub.mean_score, sub.event_fraction, color=color, lw=0.8, alpha=0.7)
            ax.plot([0, 1], [0, 1], ":", color="gray")
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0, 0.5, 1), yticks=(0, 0.5, 1),
                   xlabel="Mean score", ylabel=("Top-1 correctness" if row == 0 else "Suspicious fraction") if column == 0 else "",
                   title=name if row == 0 else "")
    fig.legend([Line2D([], [], color=c) for c in ("#2874a6", "#c46a18")],
               ["Raw", "Temperature-scaled"], loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(OUT / "pooled_reliability.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    archive = PAPER / "archive/supplement_before_condense_2026-10-07"
    archive.mkdir(exist_ok=True)
    for source in (PAPER / "bpex_supplementary.tex", PAPER / "pdf/bpex_supplementary.pdf"):
        target = archive / source.name
        if not target.exists():
            shutil.copy2(source, target)

    # Extract only the saved source cell, never execute notebook code or its outputs.
    notebook = ROOT / "notebooks/data wrangling/romeu_scraping_wrangling.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    cell, = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"
             and "run_name='qwen3_coarse4_patient_final_only'" in "".join(c["source"])]
    prompt_node, = [n for n in ast.parse(cell).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "COARSE_SYSTEM_PROMPT" for t in n.targets)]
    system = ast.literal_eval(prompt_node.value)
    (OUT / "qwen_system_prompt.txt").write_text(system, encoding="utf-8")

    backend = ROOT / "src/inference/phrase_classifier.py"
    tree = ast.parse(backend.read_text())
    function, = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "classify_via_local_server"]
    user_nodes = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "user_prompt" for t in n.targets)]
    user_node = max(user_nodes, key=lambda n: n.lineno)
    values = {"CATEGORIES": ["healthy", "benign_lesion", "opmd", "cancer"],
              "phrase_text": "{phrase_text}", "context": "{context}"}
    user = "".join(str(ast.literal_eval(n)) if isinstance(n, ast.Constant)
                   else str(values[n.value.id]) for n in user_node.value.values)
    (OUT / "qwen_user_template.txt").write_text(user, encoding="utf-8")
    (OUT / "qwen_chat_template.txt").write_text(
        "<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        "<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant\n")

    rules = ROOT / "scripts/analysis/audit_label_harmonization.py"
    source = rules.read_text()
    start = source.index('CLASSES = (')
    end = source.index('\n\ndef cohen_kappa')
    classifier = ("from __future__ import annotations\nimport re\nimport unicodedata\n"
                  "from dataclasses import dataclass\n\n" + source[start:end] + "\n")
    (OUT / "deterministic_classifier.py").write_text(classifier, encoding="utf-8")

    inputs = [notebook, backend, rules, Path(__file__).resolve()]
    inputs.extend(PAPER / "tables" / name for name in (
        "bpex_revision_evaluation.tex", "bpex_training_class_weights.tex",
        "bpex_temperature_calibration.tex", "bpex_rank_triage_tables.tex"))
    bundles = {
        "label_harmonization": ("phrase_mapping_audit.csv", "summary.json"),
        "manuscript_comparison_v1": ("per_seed_metrics.csv", "per_class_metrics.csv",
            "model_summary.csv", "pooled_source_per_seed_metrics.csv", "pooled_mcnemar.csv",
            "original_vs_corrected.csv", "split_audit.json", "confusion_matrices.csv", "curves.csv"),
        "rank_triage_analysis_v1": ("headline_summary.csv", "per_seed_rules.csv",
            "bootstrap_intervals.csv", "conditional_class_counts.csv"),
        "tuned_seedwise_test_analysis_v1": ("bootstrap_intervals.csv", "paired_comparisons.csv"),
        "sourcewise_heldout_analysis_v1": ("bootstrap_intervals.csv", "paired_comparisons.csv",
            "per_class_bootstrap_intervals.csv"),
        "calibration_diagnostics_v1": ("training_class_weights.csv", "per_seed_metrics.csv"),
        "inference_benchmark_2026-10-04": ("inference_metrics.csv", "benchmark_manifest.json"),
        "temperature_calibration_v1": ("model_summary.csv", "per_seed_metrics.csv",
            "paired_probability_intervals.csv", "temperatures.csv", "reliability_bins.csv"),
    }
    for folder, names in bundles.items():
        target = OUT / "data" / folder
        target.mkdir(exist_ok=True)
        for name in names:
            path = RESULTS / folder / name
            inputs.append(path)
            shutil.copy2(path, target / name)

    revision = "bpex_revision_evaluation.tex"
    rows = existing(revision, "tab:s_pooled_metrics")
    table("tab:s_pooled_metrics", "Four-class classification: mean $\\pm$ sample SD across seeds 42--46. Each source holdout is evaluated by a model trained and selected without that source.",
          ["Evaluation set", "Model", "Top-1", "Top-2", "Top-3", "W. F1", "Macro F1", "Bal. acc."],
          [[*r[:7], r[8]] for r in rows])
    table("tab:s_screening_metrics", "Raw-probability screening at threshold 0.5: mean $\\pm$ sample SD. Each source holdout uses the model trained and selected without that source.",
          ["Evaluation set", "Model", "F1", "Sens.", "Spec.", "PPV", "NPV", "AUC", "AP"],
          [[*r[:2], *r[3:]] for r in existing(revision, "tab:s_screening_metrics")])
    table("tab:s_per_class_metrics", "One-versus-rest class metrics: mean $\\pm$ sample SD. Classes with no reference images are omitted; all four prediction classes are retained.",
          ["Evaluation set", "Model", "Class", "$n$", "Precision", "Recall", "Specificity", "F1", "AUC"],
          [r for r in existing(revision, "tab:s_per_class_metrics") if r[3] != "0"])
    table("tab:s_source_metrics", "Source-specific slices evaluated by the pooled model, trained on all sources. These are not source holdouts. Values are mean $\\pm$ sample SD. Only Kaggle normal-folder images entered the pooled data, so the Kaggle slice contains healthy references only; -- marks metrics that are undefined without suspicious references, and Spec. is the proportion of these images classified as non-suspicious. Kaggle cancer-folder images form the Kaggle holdout (Table~\\ref{tab:s_screening_metrics}).",
          ["Pooled-test source", "Model", "$n$", "W. F1", "Macro F1", "Screen F1", "Sens.", "Spec.", "AUC"],
          [[*r[:3], *r[6:]] for r in existing(revision, "tab:s_source_metrics")])
    table("tab:s_source_ablation", "Remaining-source test performance: mean $\\pm$ sample SD. The named source is absent from training, validation and this test; its external holdout is reported separately. Case membership differs between conditions.",
          ["Training regime", "Model", "Top-1", "W. F1", "Macro F1", "Screen F1", "AUC"],
          [[*r[:3], *r[5:]] for r in existing(revision, "tab:s_source_ablation")])
    table("tab:s_bootstrap_intervals", "Conditional 95\\% bootstrap intervals for the seed-mean endpoints. Across-seed SD is reported in Tables~\\ref{tab:s_pooled_metrics} and~\\ref{tab:s_screening_metrics}.",
          ["Evaluation set", "Model", "W. F1", "Sensitivity", "Specificity", "ROC-AUC"],
          [[group[0][0], group[0][1], *(r[4] for r in group)]
           for rows in [existing(revision, "tab:s_bootstrap_intervals")]
           for group in [rows[i:i+4] for i in range(0, len(rows), 4)]])
    table("tab:s_pairwise_tests", "Paired pooled macro-F1 comparisons (A minus B); Holm correction over six pairs.",
          ["Model A", "Model B", "Difference", "95\\% CI", "Raw $p$", "Holm $p$"],
          existing(revision, "tab:s_pairwise_tests"))
    table("tab:s_comparison_pooled", "Pooled weighted-F1 before and after correction (percent). Corrected values are mean $\\pm$ SD; populations and search protocols differ.",
          ["Model", "Original", "Corrected", "Change (pp)"],
          [[r[0], *r[2:]] for r in existing(revision, "tab:s_comparison_pooled") if r[1] == "Weighted F1"])
    table("tab:s_class_weight_values", "Training class counts and inverse-frequency weights. Each cell gives $n_c$ / $a_c$; counts are shared across architectures and seeds within a split.",
          ["Split", "Healthy", "Benign", "OPMD", "Cancer"],
          [[group[0][0], *(group[j][2] + " / " + group[j][3] for j in (2, 0, 3, 1))]
           for rows in [existing("bpex_training_class_weights.tex", "tab:s_class_weight_values")]
           for group in [rows[i:i+4] for i in range(0, len(rows), 4)]])

    temperature = "bpex_temperature_calibration.tex"
    table("tab:s_temperature_probability", "Probability quality before and after validation-only temperature scaling. Cells give raw / scaled seed means. NLL uses natural logarithms; multiclass Brier sums four squared errors; ECE uses ten equal-width bins.",
          ["Evaluation set", "Model", "NLL", "Brier", "Top-1 ECE", "Screen Brier", "Screen ECE"],
          [[raw[0], raw[1], *(raw[j].split(r"\pm")[0].strip("$") + " / " +
             scaled[j].split(r"\pm")[0].strip("$") for j in (3, 4, 5, 7, 8))]
           for rows in [existing(temperature, "tab:s_temperature_probability")]
           for raw, scaled in zip(rows[::2], rows[1::2])])
    table("tab:s_temperature_screening", "Screening after scaling at threshold 0.5. Cells give raw / scaled seed means on identical evaluation cases; SD and complete metrics are in the data attachment.",
          ["Evaluation set", "Model", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "ROC-AUC"],
          [[raw[0], raw[1], *(raw[j].split(r"\pm")[0].strip("$") + " / " +
             scaled[j].split(r"\pm")[0].strip("$") for j in range(3, 9))]
           for rows in [existing(temperature, "tab:s_temperature_screening")]
           for raw, scaled in zip(rows[::2], rows[1::2])])
    table("tab:s_temperature_intervals", "Paired pooled probability-score differences (scaled minus raw) with conditional 95\\% intervals. Negative values favor scaling.",
          ["Model", "Metric", "Difference", "95\\% CI"],
          [[r[0], r[1].replace("multiclass nll", "Multiclass NLL")
            .replace("multiclass brier", "Multiclass Brier")
            .replace("screening brier", "Screening Brier"), *r[2:]]
           for r in existing(temperature, "tab:s_temperature_intervals")])
    table("tab:s_temperature_classwise", "Pooled class-specific calibration: raw / scaled seed means. Top-1 ECE improvement does not establish calibration for every class.",
          ["Model", "Class", "Prevalence", "Mean $p$", "ECE", "Brier"],
          [[*r[:3], *(r[j].split(r"\pm")[0].strip("$") + " / " +
             r[j+1].split(r"\pm")[0].strip("$") for j in (3, 5, 7))]
           for r in existing(temperature, "tab:s_temperature_classwise")])
    rank = "bpex_rank_triage_tables.tex"
    table("tab:s_rank_cancer", "Cancer ranked fourth in the pooled test (107 cancer references among 1,209 images). Percentages are seed mean $\\pm$ SD [conditional 95\\% CI].",
          ["Model", "Coverage", "NPV", "Omitted cancer $n$", "Cancer miss rate"],
          [[r[0], *r[2:6]] for r in existing(rank, "tab:s_rank_cancer")])
    table("tab:s_rank_holdouts", "Cancer ranked fourth on external source holdouts, using models trained and selected without the respective source. Percentages are seed mean $\\pm$ SD [conditional 95\\% CI].",
          ["Evaluation set", "Model", "Omitted subgroup $n$", "NPV", "Cancer miss rate"],
          existing(rank, "tab:s_rank_holdouts"))
    with (RESULTS / "label_harmonization/phrase_mapping_audit.csv").open() as handle:
        phrases = list(csv.DictReader(handle))
    table("tab:s_mapping_disagreements", "Distinct phrase/category disagreements among covered mappings. Six distinct pairs account for seven selected occurrences; these are method disagreements, without independent clinician adjudication.",
          ["Diagnosis phrase", "Qwen3", "Regex", "Rule"],
          [[tex(r["phrase_text"]), tex(r["llm_category"]), tex(r["rule_category"]), tex(r["rule_id"].split("_")[0])]
           for r in phrases if r["rule_category"] != "unknown" and r["rule_category"] != r["llm_category"]],
          widths=r"p{7.0cm}lll")

    figures()
    protected = [PAPER / "review1_answer_humanized.tex", PAPER / "pdf/review1_answer_humanized.pdf",
                 PAPER / "bpex_submission.tex"]
    manifest = {
        "purpose": "Presentation-only selection from existing evidence; no new statistical fitting.",
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "protected_sha256": {str(p.relative_to(PAPER)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "classifier_sha256": hashlib.sha256(classifier.encode()).hexdigest(),
        "output_sha256": {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "provenance.json"},
    }
    (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared compact tables, exact prompt/classifier snapshots and aggregate data in supplement/.")


if __name__ == "__main__":
    main()
