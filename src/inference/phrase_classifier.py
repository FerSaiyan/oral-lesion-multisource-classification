from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from pathlib import Path

try:
    # Notebook/terminal-friendly progress bar
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None


# === Inference schema ===

def ensure_inference_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS model_run (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            params_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS phrase_prediction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES model_run(id) ON DELETE CASCADE,
            rg TEXT NOT NULL,
            document_id INTEGER NULL REFERENCES document(id) ON DELETE CASCADE,
            phrase_id INTEGER NOT NULL REFERENCES phrase(id) ON DELETE CASCADE,
            phrase_text TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            rationale TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ok',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, phrase_id, document_id)
        );
        CREATE INDEX IF NOT EXISTS idx_pred_run ON phrase_prediction(run_id);
        CREATE INDEX IF NOT EXISTS idx_pred_phrase ON phrase_prediction(phrase_id);
        """
    )
    conn.commit()


# === Classification categories ===

CATEGORIES = [
    "malignant",
    "potentially_malignant",
    "infectious",
    "reactive_inflammatory",
    "other",
]


def _norm(text: str) -> str:
    import unicodedata

    t = text or ""
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Very small rule-based baseline so you can run offline now and swap with LLM later
_RULES: List[Tuple[str, str]] = [
    (r"carcin(oma|o) escamo|espinocelular|sarcoma|melanom", "malignant"),
    (r"displasia|eritroplasia|leucoplasia", "potentially_malignant"),
    (r"candidiase|candidose|herpes|monilia|candida", "infectious"),
    (r"hiperplasia fibrosa|fibroma|granuloma|queilite|ulcera traumat", "reactive_inflammatory"),
    (r"hemangioma|linfangioma|mucocele|mucocela|ameloblastoma|papiloma", "other"),
]


def classify_by_rules(phrase_text: str, context: str = "") -> Tuple[str, float, str]:
    p = _norm(phrase_text)
    # apply rules by priority
    for pat, cat in _RULES:
        if re.search(pat, p):
            conf = 0.85 if cat in ("malignant", "potentially_malignant") else 0.8
            rationale = (
                f"Classified as {cat.replace('_',' ')} because the phrase matches pattern '{pat}'. "
                f"Phrase: '{phrase_text}'."
            )
            return cat, conf, rationale
    # fallback to other
    return (
        "other",
        0.6,
        "Defaulted to 'other' because no category-specific keywords matched the phrase.",
    )


# Placeholder for OpenAI structured reasoning integration
def _make_openai_client():
    """Instantiate OpenAI client (supports vanilla and Azure via env vars).

    Env:
      - OPENAI_API_KEY (vanilla)
      - OPENAI_BASE_URL (optional, for proxies/self-hosted endpoints)
      - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION (Azure)
    """
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is not installed. pip install openai>=1.50.0") from exc

    import os
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_ep = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    if azure_key and azure_ep:
        return OpenAI(azure_endpoint=azure_ep, api_key=azure_key, api_version=azure_ver)

    base_url = os.getenv("OPENAI_BASE_URL")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required for OpenAI backend")
    if base_url:
        return OpenAI(api_key=key, base_url=base_url)
    return OpenAI(api_key=key)


def classify_via_openai(phrase_text: str, context: str, model: str, system_prompt: str) -> Tuple[str, float, str]:
    client = _make_openai_client()
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "category": {"type": "string", "enum": CATEGORIES},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string", "minLength": 4, "maxLength": 800},
        },
        "required": ["category", "confidence", "rationale"],
    }
    user = (
        "Você é um classificador clínico. Atribua UMA categoria às 5 possíveis, "
        "com base na frase e no contexto (pt-BR).\n\n"
        f"Categorias: {CATEGORIES}\n"
        f"Frase: {phrase_text}\n"
        f"Contexto: {context}\n"
        "Responda no schema com 'category', 'confidence' (0–1) e 'rationale' (2-3 frases curtas)."
    )
    try:
        resp = client.responses.create(
            model=model,
            reasoning={"effort": "medium"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user}]},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "PhraseClassification", "schema": schema, "strict": True},
            },
        )
        # Robust parsing for Responses API
        data_txt = None
        try:
            data_txt = resp.output[0].content[0].text
        except Exception:
            pass
        if not data_txt:
            # Try message-like extraction
            data_txt = getattr(resp, "output_text", None) or json.dumps(getattr(resp, "output", {}))
        data = json.loads(data_txt)
        cat = data.get("category")
        conf = float(data.get("confidence", 0.0))
        rat = data.get("rationale", "")
        if cat not in CATEGORIES:
            raise ValueError("Invalid category from model")
        if not (0.0 <= conf <= 1.0):
            conf = 0.5
        return cat, conf, rat
    except Exception as e:
        # Fallback to rules with status note in rationale
        cat, conf, rat = classify_by_rules(phrase_text, context)
        rat = f"[fallback] OpenAI error: {type(e).__name__}: {e}. Using rules. " + rat
        return cat, conf, rat


def _parse_local_llm_output(
    raw: str,
    *,
    elapsed_sec: float,
    stderr_txt: str = "",
    tokens_per_second: Optional[float] = None,
    phrase_text: str = "",
    context: str = "",
) -> Tuple[str, float, str, Dict]:
    """Parse local LLM text output into (category, confidence, rationale, stats)."""
    # Derive tokens/s from stderr if not given
    if tokens_per_second is None and stderr_txt:
        for line in stderr_txt.splitlines():
            for pat in (
                r"([\d.]+)\s+tokens per second",
                r"([\d.]+)\s+tokens/s",
                r"([\d.]+)\s+tok/s",
            ):
                m = re.search(pat, line)
                if m:
                    try:
                        tokens_per_second = float(m.group(1))
                        break
                    except Exception:
                        tokens_per_second = None
            if tokens_per_second is not None:
                break

    # Extract the JSON object at the end of the output (after any reasoning text)
    start = raw.rfind("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        cat, conf, rat = classify_by_rules(phrase_text, context)
        rat = (
            f"[fallback] local LLM output had no JSON object. "
            f"Using rules. Raw: {raw[:400]!r}."
        ) + (" " + rat if rat else "")
        return cat, conf, rat, {"tokens_per_second": tokens_per_second, "elapsed_sec": elapsed_sec}

    try:
        data = json.loads(raw[start : end + 1])
    except Exception as exc:
        cat, conf, rat = classify_by_rules(phrase_text, context)
        rat = (
            f"[fallback] failed to parse local LLM JSON: {type(exc).__name__}: {exc}. "
            "Using rules."
        ) + (" " + rat if rat else "")
        return cat, conf, rat, {"tokens_per_second": tokens_per_second, "elapsed_sec": elapsed_sec}

    cat_raw = str(data.get("category", "")).strip().lower().replace(" ", "_")
    if cat_raw not in CATEGORIES:
        for c in CATEGORIES:
            if cat_raw == c or cat_raw.replace("_", "") == c.replace("_", ""):
                cat_raw = c
                break
    if cat_raw not in CATEGORIES:
        cat_raw = "other"

    try:
        conf_val = float(data.get("confidence", 0.7))
    except Exception:
        conf_val = 0.7
    if not (0.0 <= conf_val <= 1.0):
        conf_val = max(0.0, min(1.0, conf_val))

    rat_text = str(data.get("rationale", "")).strip()
    if not rat_text:
        rat_text = "Model did not provide a rationale."

    return cat_raw, conf_val, rat_text, {"tokens_per_second": tokens_per_second, "elapsed_sec": elapsed_sec}


def classify_via_local_server(
    phrase_text: str,
    context: str,
    system_prompt: Optional[str] = None,
    debug_stream: bool = False,
) -> Tuple[str, float, str, Dict]:
    """Classify phrase using a running llama-server instance.

    Requires:
      - LOCAL_LLM_SERVER_URL pointing to the completion endpoint
        (e.g., http://localhost:8080/completion)
      - requests package installed
    """
    if requests is None:
        raise RuntimeError("requests is required for LOCAL_LLM_SERVER_URL backend (pip install requests).")

    server_url = os.getenv("LOCAL_LLM_SERVER_URL", "").strip()
    if not server_url:
        raise RuntimeError("LOCAL_LLM_SERVER_URL is not set for llama-server backend.")

    if system_prompt is None:
        system_prompt = (
            "You are Qwen3-30B-A3B-Thinking, a helpful reasoning assistant for "
            "clinical oral medicine. Think step by step, but FINAL output must be a "
            "single JSON object with keys: category, confidence, rationale."
        )

    if debug_stream:
        user_prompt = (
            "Você é um classificador clínico de diagnósticos de Estomatologia.\n"
            "Atribua UMA categoria dentre as opções abaixo, com base na frase e no contexto (pt-BR).\n\n"
            f"Categorias: {CATEGORIES}\n"
            f"Frase: {phrase_text}\n"
            f"Contexto: {context}\n\n"
            "Primeiro, pense passo a passo em voz alta explicando seu raciocínio.\n"
            "EM SEGUIDA, na ÚLTIMA linha, retorne APENAS um JSON único, sem texto adicional, por exemplo:\n"
            '{"category": "malignant", "confidence": 0.92, "rationale": "…"}\n'
        )
    else:
        user_prompt = (
            "Você é um classificador clínico de diagnósticos de Estomatologia.\n"
            "Atribua UMA categoria dentre as opções abaixo, com base na frase e no contexto (pt-BR).\n\n"
            f"Categorias: {CATEGORIES}\n"
            f"Frase: {phrase_text}\n"
            f"Contexto: {context}\n\n"
            "Responda APENAS com um JSON único, sem texto adicional, por exemplo:\n"
            '{"category": "malignant", "confidence": 0.92, "rationale": "…"}\n'
        )

    prompt = (
        "<|im_start|>system\n"
        f"{system_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    payload = {
        "prompt": prompt,
        "n_predict": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.0,
        # Let server decide default stop tokens; prompt format should be enough.
        "stream": bool(debug_stream),
    }

    start_ts = time.perf_counter()
    try:
        if debug_stream:
            resp = requests.post(server_url, json=payload, stream=True, timeout=180)
            resp.raise_for_status()
            # Force UTF-8 to avoid mojibake when server omits charset
            resp.encoding = "utf-8"
            parts: List[str] = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # Handle possible SSE-style "data: {...}"
                if line.startswith("data:"):
                    line = line[5:].strip()
                    if not line:
                        continue
                try:
                    obj = json.loads(line)
                    delta = obj.get("content") or obj.get("completion") or obj.get("response") or ""
                    if delta:
                        parts.append(delta)
                        print(delta, end="", flush=True)
                except json.JSONDecodeError:
                    parts.append(line)
                    print(line, end="", flush=True)
            raw = "".join(parts)
            tokens_per_second = None
            stderr_txt = ""
        else:
            resp = requests.post(server_url, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            raw = (
                data.get("content")
                or data.get("completion")
                or data.get("response")
                or ""
            )
            # Try to read tokens/s from common fields
            tokens_per_second = None
            for key in ("tokens_per_second", "tokens/sec", "tps"):
                v = data.get(key)
                if isinstance(v, (int, float)):
                    tokens_per_second = float(v)
                    break
            if tokens_per_second is None:
                metrics = data.get("metrics") or {}
                v = metrics.get("tokens_per_second")
                if isinstance(v, (int, float)):
                    tokens_per_second = float(v)
            stderr_txt = ""
    except Exception as exc:
        elapsed = time.perf_counter() - start_ts
        cat, conf, rat = classify_by_rules(phrase_text, context)
        rat = (
            f"[fallback] local LLM server error: {type(exc).__name__}: {exc}. "
            "Using rules."
        ) + (" " + rat if rat else "")
        return cat, conf, rat, {"tokens_per_second": None, "elapsed_sec": elapsed}

    elapsed = time.perf_counter() - start_ts
    return _parse_local_llm_output(
        raw,
        elapsed_sec=elapsed,
        stderr_txt=stderr_txt,
        tokens_per_second=tokens_per_second,
        phrase_text=phrase_text,
        context=context,
    )


def classify_via_local_llm(
    phrase_text: str,
    context: str,
    system_prompt: Optional[str] = None,
    debug_stream: bool = False,
) -> Tuple[str, float, str, Dict]:
    """Classify phrase using a local llama.cpp + Qwen3 Thinking model.

    Prefers a running llama-server if LOCAL_LLM_SERVER_URL is set; otherwise
    falls back to calling `llama-cli` directly.

    The model is expected to return a JSON object with:
      - category: one of CATEGORIES
      - confidence: float in [0, 1]
      - rationale: short explanation
    """
    # If a llama-server is configured, use that first to avoid reloads.
    if os.getenv("LOCAL_LLM_SERVER_URL"):
        return classify_via_local_server(
            phrase_text=phrase_text,
            context=context,
            system_prompt=system_prompt,
            debug_stream=debug_stream,
        )

    bin_path = os.getenv("LOCAL_LLM_BIN", "llama-cli")
    model_path = os.getenv("LOCAL_LLM_MODEL", "")
    if not os.path.isfile(bin_path):
        raise RuntimeError(
            "LOCAL LLM binary not found. Set LOCAL_LLM_BIN to your llama-cli executable path. "
            f"Current value: {bin_path}"
        )
    if not model_path:
        raise RuntimeError("LOCAL_LLM_MODEL is not set. Please provide a GGUF model path.")
    if not os.path.isfile(model_path):
        raise RuntimeError(f"LOCAL LLM model not found at {model_path}. Set LOCAL_LLM_MODEL to override.")

    if system_prompt is None:
        system_prompt = (
            "You are Qwen3-30B-A3B-Thinking, a helpful reasoning assistant for "
            "clinical oral medicine. Think step by step, but FINAL output must be a "
            "single JSON object with keys: category, confidence, rationale."
        )

    if debug_stream:
        user_prompt = (
            "Você é um classificador clínico de diagnósticos de Estomatologia.\n"
            "Atribua UMA categoria dentre as opções abaixo, com base na frase e no contexto (pt-BR).\n\n"
            f"Categorias: {CATEGORIES}\n"
            f"Frase: {phrase_text}\n"
            f"Contexto: {context}\n\n"
            "Primeiro, pense passo a passo em voz alta explicando seu raciocínio.\n"
            "EM SEGUIDA, na ÚLTIMA linha, retorne APENAS um JSON único, sem texto adicional, por exemplo:\n"
            '{"category": "malignant", "confidence": 0.92, "rationale": "…"}\n'
        )
    else:
        user_prompt = (
            "Você é um classificador clínico de diagnósticos de Estomatologia.\n"
            "Atribua UMA categoria dentre as opções abaixo, com base na frase e no contexto (pt-BR).\n\n"
            f"Categorias: {CATEGORIES}\n"
            f"Frase: {phrase_text}\n"
            f"Contexto: {context}\n\n"
            "Responda APENAS com um JSON único, sem texto adicional, por exemplo:\n"
            '{"category": "malignant", "confidence": 0.92, "rationale": "…"}\n'
        )

    prompt = (
        "<|im_start|>system\n"
        f"{system_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    cmd = [
        bin_path,
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        "4096",
        "-ngl",
        "99",
        "-c",
        "4096",
        "--temp",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--presence-penalty",
        "1.0",
    ]

    start_ts = time.perf_counter()
    try:
        if debug_stream:
            # Stream tokens/reasoning to stdout in real time.
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            stdout_chunks: List[str] = []
            timeout_sec = 180.0
            if proc.stdout is None:
                raise RuntimeError("Failed to attach to local LLM stdout.")
            while True:
                ch = proc.stdout.read(1)
                if ch == "":
                    break
                stdout_chunks.append(ch)
                print(ch, end="", flush=True)
                if (time.perf_counter() - start_ts) > timeout_sec:
                    proc.kill()
                    raise TimeoutError("local LLM timed out after 180 seconds.")
            proc.wait()
            raw = "".join(stdout_chunks)
            stderr_txt = proc.stderr.read() if proc.stderr is not None else ""
        else:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
                stdin=subprocess.DEVNULL,
            )
            raw = (proc.stdout or "").strip()
            stderr_txt = proc.stderr or ""
    except Exception as exc:
        elapsed = time.perf_counter() - start_ts
        cat, conf, rat = classify_by_rules(phrase_text, context)
        rat = f"[fallback] local LLM invocation error: {type(exc).__name__}: {exc}. Using rules. " + rat
        return cat, conf, rat, {"tokens_per_second": None, "elapsed_sec": elapsed}

    elapsed = time.perf_counter() - start_ts

    if proc.returncode != 0:
        cat, conf, rat = classify_by_rules(phrase_text, context)
        rat = (
            f"[fallback] local LLM returned non-zero exit ({proc.returncode}): "
            f"{stderr_txt.strip()[:400]}. Using rules. " + rat
        )
        return cat, conf, rat, {"tokens_per_second": None, "elapsed_sec": elapsed}

    return _parse_local_llm_output(
        raw,
        elapsed_sec=elapsed,
        stderr_txt=stderr_txt,
        phrase_text=phrase_text,
        context=context,
    )


@dataclass
class RunConfig:
    name: str = "baseline_rules"
    model: str = "rules"
    prompt_version: str = "v1"
    params: Dict = None


def _insert_run(conn: sqlite3.Connection, cfg: RunConfig) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO model_run(name, model, prompt_version, params_json) VALUES (?,?,?,?)",
        (cfg.name, cfg.model, cfg.prompt_version, json.dumps(cfg.params or {})),
    )
    return cur.lastrowid


def _select_candidates(conn: sqlite3.Connection, run_id: int, limit: Optional[int] = None) -> List[Tuple[int, str, int, int, str]]:
    """
    Return rows: (document_id, rg, phrase_id, phrase_id_again, phrase_text)
    phrase_id is repeated for convenient unpacking.
    document_id may be NULL if no linked document exists for that (rg, phrase_id).
    """
    sql = (
        """
        WITH final_phrases AS (
            SELECT rg, phrase_id FROM patient_biopsy_review_choice
            UNION
            SELECT rg, phrase_id FROM patient_review_choice
        ),
        chosen_docs AS (
            SELECT
                fp.rg,
                fp.phrase_id,
                (
                    SELECT d.id
                    FROM doc_phrase dp
                    JOIN document d ON d.id = dp.document_id
                    WHERE dp.phrase_id = fp.phrase_id
                      AND d.rg = fp.rg
                      AND IFNULL(d.text,'') <> ''
                    ORDER BY d.id DESC
                    LIMIT 1
                ) AS document_id
            FROM final_phrases fp
        )
        SELECT
            cd.document_id,
            cd.rg,
            p.id AS phrase_id,
            p.id AS phrase_id2,
            p.phrase
        FROM chosen_docs cd
        JOIN phrase p ON p.id = cd.phrase_id
        LEFT JOIN phrase_prediction pp
            ON pp.run_id = ?
           AND pp.phrase_id = p.id
           AND (
                (pp.document_id IS NULL AND cd.document_id IS NULL)
                OR pp.document_id = cd.document_id
           )
        WHERE pp.id IS NULL
        ORDER BY cd.document_id IS NULL, cd.document_id ASC, cd.rg ASC, p.id ASC
        """
    )
    if limit is not None:
        sql += " LIMIT ?"
        rows = conn.execute(sql, (run_id, int(limit))).fetchall()
    else:
        rows = conn.execute(sql, (run_id,)).fetchall()
    return rows


def _select_candidates_patient_review(
    conn: sqlite3.Connection, run_id: int, limit: Optional[int] = None
) -> List[Tuple[int, str, int, int, str]]:
    """
    Return rows restricted to patient-level final diagnoses
    from patient_review_choice only.
    """
    sql = (
        """
        WITH final_phrases AS (
            SELECT rg, phrase_id FROM patient_review_choice
        ),
        chosen_docs AS (
            SELECT
                fp.rg,
                fp.phrase_id,
                (
                    SELECT d.id
                    FROM doc_phrase dp
                    JOIN document d ON d.id = dp.document_id
                    WHERE dp.phrase_id = fp.phrase_id
                      AND d.rg = fp.rg
                      AND IFNULL(d.text,'') <> ''
                    ORDER BY d.id DESC
                    LIMIT 1
                ) AS document_id
            FROM final_phrases fp
        )
        SELECT
            cd.document_id,
            cd.rg,
            p.id AS phrase_id,
            p.id AS phrase_id2,
            p.phrase
        FROM chosen_docs cd
        JOIN phrase p ON p.id = cd.phrase_id
        LEFT JOIN phrase_prediction pp
            ON pp.run_id = ?
           AND pp.phrase_id = p.id
           AND (
                (pp.document_id IS NULL AND cd.document_id IS NULL)
                OR pp.document_id = cd.document_id
           )
        WHERE pp.id IS NULL
        ORDER BY cd.document_id IS NULL, cd.document_id ASC, cd.rg ASC, p.id ASC
        """
    )
    if limit is not None:
        sql += " LIMIT ?"
        rows = conn.execute(sql, (run_id, int(limit))).fetchall()
    else:
        rows = conn.execute(sql, (run_id,)).fetchall()
    return rows


def _run_inference_with_selector(
    selector,
    *,
    db_path: str,
    run_name: str,
    model: str,
    prompt_version: str,
    limit: Optional[int],
    log_every: int,
    verbose: bool,
    debug_stream: bool,
    system_prompt: Optional[str],
) -> Tuple[int, int]:
    path = Path(db_path).expanduser()
    conn = sqlite3.connect(str(path))
    ensure_inference_schema(conn)
    run_id = _insert_run(conn, RunConfig(run_name, model, prompt_version, params={"type": model}))
    conn.commit()

    rows = selector(conn, run_id=run_id, limit=limit)
    total = len(rows)

    pbar = None
    if verbose and tqdm is not None and total > 0:
        pbar = tqdm(total=total, desc=f"Inference ({model})", unit="phrase")

    inserted = 0
    for (document_id, rg, phrase_id, _pid2, phrase_text) in rows:
        # Fetch a short context excerpt (first 400 chars of document text)
        if document_id is not None:
            ctx_row = conn.execute("SELECT text FROM document WHERE id = ?", (document_id,)).fetchone()
            context = (ctx_row[0] if ctx_row and ctx_row[0] else "")[:400]
        else:
            # No document available: let the model decide based only on the final diagnosis phrase
            context = ""
        stats: Dict = {}
        if model == "rules":
            cat, conf, rationale = classify_by_rules(phrase_text, context)
        elif model in {"local_llm", "local_qwen3"}:
            if debug_stream:
                print(
                    f"\n[llm-debug] rg={rg} phrase_id={phrase_id} "
                    f"phrase={phrase_text[:80]!r}"
                )
            cat, conf, rationale, stats = classify_via_local_llm(
                phrase_text,
                context,
                system_prompt=system_prompt,
                debug_stream=debug_stream,
            )
        else:
            # System prompt kept concise; allow override via system_prompt argument
            sys_prompt = (
                system_prompt
                or "Você é um classificador de diagnósticos de Estomatologia. "
                "Atribua uma categoria única e explique brevemente (sem dados sensíveis)."
            )
            cat, conf, rationale = classify_via_openai(phrase_text, context, model=model, system_prompt=sys_prompt)
        conn.execute(
            """
            INSERT INTO phrase_prediction(run_id, rg, document_id, phrase_id, phrase_text, category, confidence, rationale, status)
            VALUES (?,?,?,?,?,?,?,?, 'ok')
            """,
            (run_id, rg, document_id, phrase_id, phrase_text, cat, float(conf), rationale),
        )
        inserted += 1
        if pbar is not None:
            pbar.update(1)
        if verbose and (inserted % max(1, log_every) == 0 or inserted == total):
            # Short log line with timing stats (if available) and a rationale snippet
            tokps = stats.get("tokens_per_second") if stats else None
            rat_snip = rationale.replace("\n", " ")
            if len(rat_snip) > 120:
                rat_snip = rat_snip[:117] + "..."
            base_msg = f"[inference] {inserted}/{total} rg={rg} phrase_id={phrase_id} cat={cat} conf={float(conf):.2f}"
            if tokps:
                base_msg += f" tokens/s={tokps:.1f}"
            print(base_msg)
            if rat_snip:
                print(f"  rationale: {rat_snip}")
        if inserted % max(1, log_every) == 0:
            conn.commit()
    conn.commit()
    if pbar is not None:
        pbar.close()
    conn.close()
    return inserted, total


def run_inference(
    db_path: str = "data/app/romeu_unknown_phrases.sqlite",
    run_name: str = "baseline",
    model: str = "rules",
    prompt_version: str = "v1",
    limit: Optional[int] = None,
    log_every: int = 50,
    verbose: bool = False,
    debug_stream: bool = False,
    system_prompt: Optional[str] = None,
) -> Tuple[int, int]:
    """Run phrase classification for all final phrases (patient + biopsy).

    Returns (inserted, total_candidates_scanned).
    """
    return _run_inference_with_selector(
        _select_candidates,
        db_path=db_path,
        run_name=run_name,
        model=model,
        prompt_version=prompt_version,
        limit=limit,
        log_every=log_every,
        verbose=verbose,
        debug_stream=debug_stream,
        system_prompt=system_prompt,
    )


def run_inference_for_final_phrases(
    db_path: str = "data/app/romeu_unknown_phrases.sqlite",
    run_name: str = "baseline_patient_final",
    model: str = "rules",
    prompt_version: str = "v1",
    limit: Optional[int] = None,
    log_every: int = 50,
    verbose: bool = False,
    debug_stream: bool = False,
    system_prompt: Optional[str] = None,
) -> Tuple[int, int]:
    """Run phrase classification only for patient-level final diagnoses.

    Uses phrases from patient_review_choice (not biopsy or Prof. Carol tables).
    """
    return _run_inference_with_selector(
        _select_candidates_patient_review,
        db_path=db_path,
        run_name=run_name,
        model=model,
        prompt_version=prompt_version,
        limit=limit,
        log_every=log_every,
        verbose=verbose,
        debug_stream=debug_stream,
        system_prompt=system_prompt,
    )


# Backward-compatible alias
def run_rule_inference(**kwargs):
    return run_inference(model="rules", **kwargs)


def run_inference_batched(
    db_path: str = "data/app/romeu_unknown_phrases.sqlite",
    run_name: str = "baseline_batched",
    model: str = "rules",
    prompt_version: str = "v1",
    limit: Optional[int] = None,
    log_every: int = 50,
    verbose: bool = False,
    debug_stream: bool = False,
    system_prompt: Optional[str] = None,
    num_workers: int = 2,
) -> Tuple[int, int]:
    """Run phrase classification with simple multi-threaded batching.

    Uses a thread pool to send multiple LLM requests in parallel (mainly useful
    for local_llm / llama-server). DB writes remain single-threaded.
    """
    path = Path(db_path).expanduser()
    conn = sqlite3.connect(str(path))
    ensure_inference_schema(conn)
    run_id = _insert_run(conn, RunConfig(run_name, model, prompt_version, params={"type": model, "batched": True}))
    conn.commit()

    rows = _select_candidates(conn, run_id=run_id, limit=limit)
    total = len(rows)
    if total == 0:
        conn.close()
        return 0, 0

    # Precompute contexts to keep DB access single-threaded.
    tasks: List[Tuple[int, str, int, str, str]] = []
    for (document_id, rg, phrase_id, _pid2, phrase_text) in rows:
        if document_id is not None:
            ctx_row = conn.execute("SELECT text FROM document WHERE id = ?", (document_id,)).fetchone()
            context = (ctx_row[0] if ctx_row and ctx_row[0] else "")[:400]
        else:
            context = ""
        tasks.append((document_id, rg, phrase_id, phrase_text, context))

    if num_workers < 1:
        num_workers = 1

    pbar = None
    if verbose and tqdm is not None and total > 0:
        pbar = tqdm(total=total, desc=f"Inference-batched ({model}, {num_workers} workers)", unit="phrase")

    def _worker(task: Tuple[int, str, int, str, str]):
        document_id, rg, phrase_id, phrase_text, context = task
        stats: Dict = {}
        if model == "rules":
            cat, conf, rationale = classify_by_rules(phrase_text, context)
        elif model in {"local_llm", "local_qwen3"}:
            cat, conf, rationale, stats = classify_via_local_llm(
                phrase_text,
                context,
                system_prompt=system_prompt,
                debug_stream=debug_stream,
            )
        else:
            sys_prompt = (
                system_prompt
                or "Você é um classificador de diagnósticos de Estomatologia. "
                "Atribua uma categoria única e explique brevemente (sem dados sensíveis)."
            )
            cat, conf, rationale = classify_via_openai(phrase_text, context, model=model, system_prompt=sys_prompt)
        return document_id, rg, phrase_id, phrase_text, cat, float(conf), rationale, stats

    inserted = 0
    results: List[Tuple[int, str, int, str, str, float, str, Dict]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        future_to_task = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(future_to_task):
            document_id, rg, phrase_id, phrase_text, cat, conf, rationale, stats = fut.result()
            results.append((document_id, rg, phrase_id, phrase_text, cat, conf, rationale, stats))
            inserted += 1
            if pbar is not None:
                pbar.update(1)
            if verbose and (inserted % max(1, log_every) == 0 or inserted == total):
                tokps = stats.get("tokens_per_second") if stats else None
                rat_snip = rationale.replace("\n", " ")
                if len(rat_snip) > 120:
                    rat_snip = rat_snip[:117] + "..."
                base_msg = f"[inference-batched] {inserted}/{total} rg={rg} phrase_id={phrase_id} cat={cat} conf={float(conf):.2f}"
                if tokps:
                    base_msg += f" tokens/s={tokps:.1f}"
                print(base_msg)
                if rat_snip:
                    print(f"  rationale: {rat_snip}")

    # Insert predictions sequentially
    for document_id, rg, phrase_id, phrase_text, cat, conf, rationale, _stats in results:
        conn.execute(
            """
            INSERT INTO phrase_prediction(run_id, rg, document_id, phrase_id, phrase_text, category, confidence, rationale, status)
            VALUES (?,?,?,?,?,?,?,?, 'ok')
            """,
            (run_id, rg, document_id, phrase_id, phrase_text, cat, float(conf), rationale),
        )
    conn.commit()
    if pbar is not None:
        pbar.close()
    conn.close()
    return inserted, total
