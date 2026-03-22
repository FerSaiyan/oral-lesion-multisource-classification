# Multisource Oral Lesion Classification (Oral Oncology)

This repository contains the training and evaluation code used for multiclass oral lesion classification experiments (healthy, benign lesion, OPMD, cancer).

GitHub: <https://github.com/FerSaiyan/oral-lesion-multisource-classification>

## What is included

- Optuna-based training pipeline for EfficientNet and ViT (`run_optuna_study.py`)
- Batch confusion-matrix and hold-out evaluation script (`batch_confusion_eval_multisource.py`)
- Phrase-level clinician free-text classification pipeline (`scripts/inference/run_phrase_classifier.py`, `src/inference/phrase_classifier.py`)
- Redacted reproducible notebook for LLM phrase classification (`notebooks/public/llm_phrase_classification_demo.ipynb`)
- Core model, dataset, training, and experiment utilities under `src/`
- Study configuration files under `configs/studies/`

## What is intentionally excluded

- Internal clinical dataset and any patient-identifiable data
- Scraping/mining code and internal-system connectors
- Web application code and diffusion/synthetic-data pipelines (separate projects)

## Data access and restrictions

The internal dataset used in the manuscript is not publicly shareable.
Some external datasets may require permission from their data owners.
To reproduce training/evaluation, obtain authorized copies of the datasets directly from their original providers.

Expected split files (not included here) are referenced in:

- `data/processed/multisource_train.csv`
- `data/processed/multisource_val.csv`
- `data/processed/multisource_test.csv`

## Reproducibility (CSV schema and splits)

The training/evaluation scripts are file-driven. Reproducing results requires providing CSV files with the expected columns.

### 1) Supervised train/val/test CSVs

Minimum required columns:

- `filename`: image file name relative to `IMAGES_FOLDER`
- `diagnosis_categories`: final class label used for training/evaluation

Common optional columns (used in multisource experiments and reporting):

- `coarse_label`
- `coarse_label_id`
- `fine_label`
- `sample_weight` (if present, per-sample loss weighting is enabled)
- `dataset_source`
- `patient_id`
- `session_id`
- `group_id` / `group_all`
- `image_path` (informational; loader uses `filename` + `IMAGES_FOLDER`)

### 2) Pseudolabeling unlabeled CSV

Minimum required column:

- `filename`

### 3) Split semantics used in this repository

- `train_csv`: supervised training split
- `val_csv`: validation split used for Optuna objective and checkpoint selection
- `test_csv`: optional held-out split for reporting only (not used for optimization)

## LLM free-text protocol

The exact local LLM protocol (prompt template, decoding parameters, server/CLI setup, JSON schema, and fallback behavior) is documented in:

- `docs/llm_phrase_classification_protocol.md`

This is intentionally published outside private notebooks so the method remains reproducible without exposing restricted data-curation code.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Minimal run examples

Train (EffNet B3 multisource):

```bash
python run_optuna_study.py --config configs/studies/effnet_imagenet_b3_ar_multisource.yaml
```

Train (ViT large):

```bash
python run_optuna_study.py --config configs/studies/vit_large_imagenet_ar.yaml
```

Batch confusion/evaluation:

```bash
python batch_confusion_eval_multisource.py --config configs/studies/effnet_imagenet_b3_ar_multisource.yaml
```

Phrase classification (local llama.cpp server):

```bash
export LOCAL_LLM_SERVER_URL=http://127.0.0.1:8080/completion
python scripts/inference/run_phrase_classifier.py \
  --db data/app/romeu_unknown_phrases.sqlite \
  --model local_llm \
  --run-name qwen_local_v1 \
  --prompt-version v1
```

## Recommended environment variables

Set these to point to your authorized local copies:

- `IMAGES_FOLDER`
- `TRAIN_CSV`
- `VAL_CSV`
- `TEST_CSV`
- `OPTUNA_RESULTS_DIR`

For hold-out evaluation script:

- `MULTISOURCE_BASE_DIR`
- `MULTISOURCE_IMAGES`
- `MULTISOURCE_NO_KAGGLE_IMAGES`
- `MULTISOURCE_NO_ZENODO_IMAGES`
- `MULTISOURCE_NO_MENDELEY_IMAGES`

## Model checkpoints

Trained model weights are released separately on Hugging Face:

- `<https://huggingface.co/<org-or-user>/<model-collection>>`

Model-weight license details: see `MODEL_WEIGHTS_LICENSE.md`.

## Citation

Manuscript-ready availability snippets and BibTeX templates are provided in `MANUSCRIPT_SNIPPETS.md`.

## License

Code in this repository is licensed under Apache License 2.0 (`LICENSE`).
