# Multi-dataset oral lesion classification

Code for the manuscript *Multi-dataset Deep Learning for Multiclass
Classification of Oral Lesions* (under review). The models classify intraoral
photographs as healthy, benign lesion, oral potentially malignant disorder
(OPMD) or cancer. They were trained on one internal clinical sample and three
public datasets (Zenodo, Kaggle and Mendeley) and evaluated on a pooled test
set and on holdouts of each public source.

Trained weights: <https://huggingface.co/Fertroll10/oral-lesions-multisource-classification>

## What is not included

The internal clinical images, their labels, the split manifests and the
diagnosis-phrase database contain patient information and are not shared.
The manifests also list internal images, so the pooled splits cannot be
rebuilt from this repository alone. The public datasets must be obtained from
their providers:

- Zenodo: Piyarathne et al., [doi:10.5281/zenodo.10664056](https://doi.org/10.5281/zenodo.10664056)
- Mendeley: Oral Images Dataset, version 2, [doi:10.17632/mhjyrn35p4.2](https://doi.org/10.17632/mhjyrn35p4.2)
- Kaggle: [Oral Cancer Images for Classification](https://www.kaggle.com/datasets/muhammadatef/oral-cancer-images-for-classification)

## Repository layout

| Part of the paper | Code |
|---|---|
| Training and Optuna search (EfficientNet-B3/B7, ViT-Base/Large) | `scripts/run_optuna_study.py`, `src/`, `configs/studies/*.yaml` |
| Patient-aware splits, corrected Zenodo patient parser (`_zenodo_patient_mapper`), source holdouts | `scripts/data/`, `scripts/analysis/{freeze,audit}_bpex_revision_splits.py`, `scripts/analysis/prepare_bpex_materialized_view.py` |
| Five-seed searches and 50-epoch retrains | `scripts/analysis/run_bpex_revision_hpo.py`, `scripts/analysis/run_bpex_revision_final_seed.py`, `configs/studies/bpex_revision_v2_*.json` |
| Source-exclusion experiments | `scripts/analysis/*bpex_sourcewise*.py`, `configs/studies/bpex_sourcewise_hpo_2026-09-30.json` |
| Metrics, bootstrap intervals, paired tests, ROC/PR curves | `scripts/analysis/analyze_bpex_final_seeds.py`, `analyze_tuned_seedwise.py`, `build_bpex_review_metrics_bundle.py`, `plot_bpex_*.py` |
| Supplement tables, calibration, ranked-list analysis, Grad-CAM panel | `papers/Multisource Classification/scripts/` |
| Matched EfficientNet-B7 loss-weighting ablation (Supplementary S8) | `scripts/analysis/*bpex_b7_matched_weighting.py`, `configs/studies/bpex_b7_matched_weighting_2026-10-13.json` |
| Desktop inference benchmark (Supplementary S7) | `scripts/analysis/*bpex_deployment*.py`, `benchmark_bpex_revision_inference.py`, `summarize_bpex_matched_timing.py` |
| Phone GPU/NPU benchmark on the Galaxy Z Fold6 (Supplementary S7) | `scripts/npu/`, including the Android app in `scripts/npu/android_benchmark/` |
| Diagnosis-phrase mapping with local Qwen3 and the deterministic comparison | `src/inference/phrase_classifier.py`, `scripts/analysis/audit_label_harmonization.py`, `docs/llm_phrase_classification_protocol.md`, `papers/Multisource Classification/supplement/` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The original runs used Python 3.10, PyTorch 2.7 and timm 1.0.15. Several
revision scripts stop unless they run inside a conda environment named
`LocalGPT_llama2`; edit that check if you use a different environment.

## Training on your own splits

`scripts/run_optuna_study.py` runs an Optuna search followed by a retrain of
the best configuration. It reads CSV manifests with at least `filename`
(relative to the image folder) and `diagnosis_categories` (one of
`benign_lesion`, `cancer`, `healthy`, `opmd`). Optional columns include
`dataset_source`, `patient_id` and `sample_weight`.

Copy one of the four study configs and set `images_folder`, `train_csv_path`
and `val_csv_path` to your data. Values in the config take precedence over the
`IMAGES_FOLDER`, `TRAIN_CSV` and `VAL_CSV` environment variables.

```bash
cp configs/studies/effnet_imagenet_b3_ar_multisource.yaml my_b3.yaml
# edit the path fields in my_b3.yaml
python scripts/run_optuna_study.py --config my_b3.yaml
```

The four YAML files in `configs/studies/` are byte-identical to the ones used
in the paper, so their SHA-256 hashes match those recorded in the frozen plans.
Their path fields point to the original machines.

## Reproducing the revision runs

The revision runners (`run_bpex_revision_hpo.py`, `run_bpex_revision_final_seed.py`,
`run_bpex_b7_matched_weighting.py`) check every manifest against the SHA-256
hashes in `configs/studies/bpex_revision_v2_frozen_2026-09-24.json` before
training. Because those manifests include internal images, the runners can only
be executed with the original data. They are published to document the exact
protocol: five seeds (42 to 46), a 50-trial search per seed, a 50-epoch retrain,
checkpoint selection by validation macro-F1, and one evaluation of each selected
checkpoint on the frozen test set or source holdout. The supplement scripts
under `papers/` also read private result files and are included for the same
reason.

## Licence

Code: Apache 2.0 (`LICENSE`). Model weights: CC BY-NC 4.0
(`MODEL_WEIGHTS_LICENSE.md`). The terms of the original datasets also apply.

## Citation

The manuscript is under review. A citation will be added here once it is
published.
