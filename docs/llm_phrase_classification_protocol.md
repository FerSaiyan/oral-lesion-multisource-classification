# Diagnosis-phrase mapping protocol (local Qwen3)

Internal image labels come from diagnosis phrases written by dentists in
Portuguese. The phrases were first reviewed and selected manually. A local
language model then mapped each selected phrase to one of the four study
classes: `healthy`, `benign_lesion`, `opmd` or `cancer`. Phrases that could not
be mapped unambiguously were labelled `unknown` and excluded. No patient data
left the institutional hardware.

## Model and runtime

- Model: Qwen3-30B-A3B-Thinking-2507, Unsloth dynamic 3-bit GGUF quantisation
  (`Qwen3-30B-A3B-Thinking-2507-UD-IQ3_XXS.gguf`).
- Runtime: llama.cpp `llama-server`, all layers on the GPU (`-ngl 99`),
  4,096-token context (`-c 4096`).
- Generation: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0,
  up to 4,096 predicted tokens (`n_predict`).

```bash
llama-server -m /path/to/Qwen3-30B-A3B-Thinking-2507-UD-IQ3_XXS.gguf \
  -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
export LOCAL_LLM_SERVER_URL=http://127.0.0.1:8080/completion
```

## Prompt

The exact texts used for the article are in
`papers/Multisource Classification/supplement/`:

- `qwen_system_prompt.txt`: the system prompt with class definitions, examples
  and decision rules (for example, carcinoma in situ or invasive carcinoma is
  always `cancer`, even when an OPMD term is also present).
- `qwen_user_template.txt`: the non-streaming user message, with placeholders
  for the phrase and its context.
- `qwen_chat_template.txt`: the ChatML wrapper sent to `llama-server`.

The model reasons first and must end with a single JSON object containing
`category`, `confidence` and `rationale`. The user template's example output
uses `malignant`, which is not one of the four categories; the system prompt
defines the valid ones.

The request code is in `src/inference/phrase_classifier.py`, and
`scripts/inference/run_phrase_classifier.py` is a command-line wrapper. Both
read phrases from a local SQLite database that is not part of this
repository. The module's default `CATEGORIES` list is an earlier
five-category scheme. For the article (prompt version `v2_coarse4`) the list
was set to the four study classes and `qwen_system_prompt.txt` was passed as
`system_prompt`.

## Deterministic comparison

To check the mapping, an accent-normalised regular-expression classifier was
written independently of the model:
`papers/Multisource Classification/supplement/deterministic_classifier.py`
(extracted from `scripts/analysis/audit_label_harmonization.py`). It covered
249 of 259 phrase occurrences and agreed with Qwen3 on 242 of them (97.2%,
Cohen's kappa 0.953). At image level the two agreed on 1,388 of 1,450 covered
images (95.7%, kappa 0.923). Agreement with independent clinicians was not
measured.
