# Manuscript Snippets

## Code and model availability section

```latex
\section*{Code and model availability}
Code used for training and evaluation is publicly available at \url{https://github.com/FerSaiyan/oral-lesion-multisource-classification} (release v1.0.0) \cite{CodeRepo2026}. Trained model checkpoints are available at \url{https://huggingface.co/<user-or-org>/<model-collection>} \cite{ModelRepo2026}.
```

## Data availability section

```latex
\section*{Data availability}
The internal clinical dataset contains sensitive patient data and cannot be made publicly available. Public datasets used in this study are available from their original providers; access conditions follow each provider's policy (including permission-based access where applicable). We provide code and configuration files to reproduce training and evaluation on authorized copies of the data.
```

## BibTeX entries

```bibtex
@misc{CodeRepo2026,
  author       = {Frazatto, Akio Kenzo Tezuka and Motta, Ana Carolina Fragoso and Ribeiro, Ana Elisa Rodrigues Alves and Tin\'os, Renato and Bachmann, Luciano},
  title        = {Multisource Oral Lesion Classification: Training and Evaluation Code},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/FerSaiyan/oral-lesion-multisource-classification},
  note         = {Accessed 22 March 2026}
}

@misc{ModelRepo2026,
  author       = {Frazatto, Akio Kenzo Tezuka and Motta, Ana Carolina Fragoso and Ribeiro, Ana Elisa Rodrigues Alves and Tin\'os, Renato and Bachmann, Luciano},
  title        = {Trained Multiclass Oral Lesion Models},
  year         = {2026},
  howpublished = {Hugging Face model repository},
  url          = {https://huggingface.co/<user-or-org>/<model-collection>},
  note         = {Accessed 22 March 2026}
}
```
