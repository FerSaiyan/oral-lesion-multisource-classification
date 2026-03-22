# Model Card Template

## Model Details

- **Model name**: `<model-name>`
- **Task**: Multiclass oral lesion classification (healthy, benign lesion, OPMD, cancer)
- **Framework**: PyTorch
- **Code repository**: `<https://github.com/<user-or-org>/<repo>>`
- **License (weights)**: `cc-by-nc-4.0`

## Intended Use

This model is intended for research use and clinical decision-support prototyping only. It is not intended to replace specialist diagnosis.

## Training Data

The model was trained on a harmonized multi-source oral-image dataset including internal and public sources. Internal clinical data are not publicly released.

## Evaluation

Report the exact split protocol used in the manuscript (patient/group-aware train/validation/test and hold-out external datasets).

## Limitations

- Performance can degrade under domain shift (new devices, centers, protocols, populations).
- Labels can include noise due to harmonization and coarse category mapping.
- Clinical use requires specialist oversight and confirmation workflows.

## Ethical Considerations

This model should be used only in settings compliant with local ethics approvals, data governance, and patient privacy regulations.
