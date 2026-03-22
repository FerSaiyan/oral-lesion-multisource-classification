# Local LLM Phrase Classification Protocol (llama.cpp + Qwen)

This document captures the exact protocol used to classify clinician free-text diagnosis phrases into coarse categories.

## Why this is separate from private notebooks

The original research notebook contains private/internal data handling and session curation logic that cannot be shared publicly.
To keep the method reproducible, the LLM inference logic is published as standalone code:

- `scripts/inference/run_phrase_classifier.py`
- `src/inference/phrase_classifier.py`

This allows full method transparency without exposing restricted data-processing code.

## Categories

The classifier outputs exactly one of:

- `malignant`
- `potentially_malignant`
- `infectious`
- `reactive_inflammatory`
- `other`

## Prompt format

The assistant is prompted using a chat-template style string:

```text
<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_prompt}
<|im_end|>
<|im_start|>assistant
```

Default system prompt:

```text
You are Qwen3-30B-A3B-Thinking, a helpful reasoning assistant for clinical oral medicine. Think step by step, but FINAL output must be a single JSON object with keys: category, confidence, rationale.
```

User prompt (non-stream mode):

```text
Você é um classificador clínico de diagnósticos de Estomatologia.
Atribua UMA categoria dentre as opções abaixo, com base na frase e no contexto (pt-BR).

Categorias: ['malignant', 'potentially_malignant', 'infectious', 'reactive_inflammatory', 'other']
Frase: {phrase_text}
Contexto: {context}

Responda APENAS com um JSON único, sem texto adicional, por exemplo:
{"category": "malignant", "confidence": 0.92, "rationale": "…"}
```

## Decoding and runtime parameters

### llama-server payload

```json
{
  "n_predict": 4096,
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "presence_penalty": 1.0,
  "stream": false
}
```

### llama-cli flags (fallback when server is not used)

- `-n 4096`
- `-ngl 99`
- `-c 4096`
- `--temp 0.6`
- `--top-p 0.95`
- `--top-k 20`
- `--presence-penalty 1.0`

## Expected model output

A single JSON object with:

- `category` (enum above)
- `confidence` (float in `[0, 1]`)
- `rationale` (short explanation)

If parsing fails or the model invocation fails, the pipeline falls back to deterministic rule-based classification and records a fallback rationale.

## Environment variables

- `LOCAL_LLM_SERVER_URL` (example: `http://127.0.0.1:8080/completion`)
- `LOCAL_LLM_BIN` (path to `llama-cli`, only needed if server URL is not set)
- `LOCAL_LLM_MODEL` (path to GGUF model, only needed if server URL is not set)

Optional OpenAI/Azure backend variables are also supported in the same module.

## Example usage

Start `llama-server` (example):

```bash
llama-server -m /path/to/Qwen3-30B-A3B-Thinking.gguf -ngl 99 -c 4096 --port 8080
```

Run phrase classification:

```bash
export LOCAL_LLM_SERVER_URL=http://127.0.0.1:8080/completion
python scripts/inference/run_phrase_classifier.py \
  --db data/app/romeu_unknown_phrases.sqlite \
  --model local_llm \
  --run-name qwen_local_v1 \
  --prompt-version v1
```

For Qwen2.5 family models, keep the same protocol and replace only the model file/path and run name.
