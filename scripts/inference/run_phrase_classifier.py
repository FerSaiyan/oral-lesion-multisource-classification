#!/usr/bin/env python3
"""Run phrase classification (rules, local LLM, or OpenAI) and store predictions in the DB.

Usage examples:
  python scripts/inference/run_phrase_classifier.py \
      --db data/app/romeu_unknown_phrases.sqlite \
      --model rules --run-name baseline_rules --limit 1000

  # With OpenAI (set OPENAI_API_KEY or Azure envs first):
  python scripts/inference/run_phrase_classifier.py \
      --db data/app/romeu_unknown_phrases.sqlite \
      --model gpt-4o-mini --run-name gpt4o_pass1 --limit 500

  # With local llama.cpp + Qwen3 Thinking (configure LOCAL_LLM_BIN / LOCAL_LLM_MODEL if needed):
  python scripts/inference/run_phrase_classifier.py \
      --db data/app/romeu_unknown_phrases.sqlite \
      --model local_llm --run-name qwen3_local --limit 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="Path to SQLite DB (romeu_unknown_phrases.sqlite)")
    p.add_argument(
        "--model",
        default="rules",
        help="'rules', 'local_llm' (llama.cpp + Qwen3), or an OpenAI model name (e.g., gpt-4o-mini)",
    )
    p.add_argument("--run-name", default="baseline", help="Name for this inference run")
    p.add_argument("--prompt-version", default="v1", help="Prompt/version label")
    p.add_argument("--limit", type=int, default=None, help="Max candidates to process (default: all)")
    p.add_argument("--log-every", type=int, default=50, help="Commit/print progress every N rows")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Ensure repo root on path
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.inference.phrase_classifier import run_inference

    print(f"[inference] DB={args.db} model={args.model} run={args.run_name} limit={args.limit}")
    inserted, total = run_inference(
        db_path=args.db,
        run_name=args.run_name,
        model=args.model,
        prompt_version=args.prompt_version,
        limit=args.limit,
        log_every=args.log_every,
    )
    print(f"[inference] inserted={inserted} / scanned={total}")


if __name__ == "__main__":
    main()
