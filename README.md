# Multisource Oral Lesion Classification (Oral Oncology)

This repository contains the training and evaluation code used for multiclass oral lesion classification experiments (healthy, benign lesion, OPMD, cancer).

GitHub: <https://github.com/FerSaiyan/oral-lesion-multisource-classification>

## What is included

- Optuna-based training pipeline for EfficientNet and ViT (`run_optuna_study.py`)
- Batch confusion-matrix and hold-out evaluation script (`batch_confusion_eval_multisource.py`)
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
