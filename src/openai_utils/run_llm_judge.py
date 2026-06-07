"""Generic LLM judge runner via the OpenAI Batch API.

Reads a directory of prompt .txt files, submits them to the Batch API,
polls until done, and writes parsed JSON responses to an output JSONL file.

Each prompt file:
  - filename stem → custom_id  (e.g. layer13_joy_pc1)
  - entire file content → user message

Output JSONL (one line per prompt):
  {"custom_id": "<stem>", "response_data": { ...parsed JSON... }}

Typical usage (exp3 axis interpretation):

    uv run python scripts/run_llm_judge.py \\
        --prompts-dir analysis/local_axis_interpretation/llm_prompts \\
        --out-file    analysis/local_axis_interpretation/llm_judge_results.jsonl

Cost estimate only (no upload):

    uv run python scripts/run_llm_judge.py \\
        --prompts-dir analysis/local_axis_interpretation/llm_prompts \\
        --out-file    analysis/local_axis_interpretation/llm_judge_results.jsonl \\
        --prepare-only

Resume after interruption (poll existing batch IDs):

    uv run python scripts/run_llm_judge.py \\
        --out-file analysis/local_axis_interpretation/llm_judge_results.jsonl \\
        --collect-only

Parse from already-downloaded raw output files:

    uv run python scripts/run_llm_judge.py \\
        --from-results-dir /path/to/raw_outputs \\
        --out-file analysis/local_axis_interpretation/llm_judge_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai_utils.openai_batch import (
    collect_results,
    collect_results_from_local_dir,
    count_prompt_tokens,
    load_batch_ids,
    poll_until_done,
    save_batch_ids,
    split_into_batches_by_token,
    upload_and_submit,
    write_batch_input_file,
    write_output_jsonl,
)
from utils.prompts import LLM_JUDGE_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT: str = LLM_JUDGE_SYSTEM_PROMPT


# Helpers

def load_prompts(
    prompts_dir: Path,
    glob: str = "*.txt",
    skip_ids: set[str] | None = None,
) -> list[tuple[str, list[dict[str, str]]]]:
    """Load prompt .txt files as (custom_id, [user_message]) pairs."""
    items: list[tuple[str, list[dict[str, str]]]] = []
    prompt_files = sorted(prompts_dir.glob(glob))
    if not prompt_files:
        raise FileNotFoundError(f"No files matching {glob!r} in {prompts_dir}")
    skip_ids = skip_ids or set()
    skipped = 0
    for path in prompt_files:
        cid = path.stem
        if cid in skip_ids:
            skipped += 1
            continue
        items.append((cid, [{"role": "user", "content": path.read_text(encoding="utf-8").strip()}]))
    logger.info("Loaded %d prompt(s) (skipped %d already-processed)", len(items), skipped)
    return items


def load_already_processed_ids(out_file: Path) -> set[str]:
    """Return set of custom_ids already present in the output JSONL (for resuming)."""
    if not out_file.exists():
        return set()
    ids: set[str] = set()
    with open(out_file) as f:
        for line in f:
            try:
                ids.add(json.loads(line.strip())["custom_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    logger.info("Found %d already-processed IDs in %s", len(ids), out_file)
    return ids


def append_output_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Append rows to a JSONL file (safe for incremental writes)."""
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Appended %d rows → %s", len(rows), path)


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--prompts-dir", type=Path, default=None,
        help="Directory containing prompt .txt files (one file per request).",
    )
    p.add_argument(
        "--prompts-glob", default="*.txt",
        help="Glob pattern for prompt files inside --prompts-dir (default: *.txt).",
    )
    p.add_argument(
        "--system-prompt-file", type=Path, default=None,
        help=(
            "Path to a text file containing a custom system prompt. "
            "If omitted, a generic affective-science system prompt is used."
        ),
    )
    p.add_argument(
        "--out-file", type=Path, default=None,
        help="Output JSONL file for parsed LLM responses.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help=(
            "Output directory. If --out-file is not given, the output file will be "
            "placed here as 'llm_judge_results.jsonl'."
        ),
    )
    p.add_argument(
        "--model", default="gpt-4.1-mini",
        help="OpenAI model identifier (default: gpt-4.1-mini).",
    )
    p.add_argument(
        "--est-output-tokens", type=int, default=512,
        help="Estimated output tokens per request for batch splitting (default: 512).",
    )
    p.add_argument(
        "--max-tokens-per-batch", type=int, default=20_000_000,
        help="Max token budget per batch job (default: 20,000,000).",
    )
    p.add_argument(
        "--poll-interval", type=int, default=60,
        help="Seconds between status-poll requests (default: 60).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip custom_ids already present in --out-file (continue interrupted run).",
    )
    p.add_argument(
        "--prepare-only", action="store_true",
        help="Write batch input JSONL files and print cost estimate; do not upload.",
    )
    p.add_argument(
        "--collect-only", action="store_true",
        help=(
            "Skip preparation/submission; load batch IDs from <out_file>_batch_ids.txt "
            "and go straight to polling & collection."
        ),
    )
    p.add_argument(
        "--from-results-dir", type=Path, default=None, metavar="DIR",
        help=(
            "Parse all *.jsonl files in DIR as already-downloaded Batch API output "
            "and write results without contacting the OpenAI API."
        ),
    )
    return p.parse_args()


# Main

def main() -> None:
    load_dotenv()
    args = parse_args()

    # --- Resolve output file ---
    if args.out_file is not None:
        out_file: Path = args.out_file
    elif args.out_dir is not None:
        out_file = args.out_dir / "llm_judge_results.jsonl"
    elif args.prompts_dir is not None:
        out_file = args.prompts_dir.parent / "llm_judge_results.jsonl"
    else:
        raise ValueError("Provide at least one of --out-file, --out-dir, or --prompts-dir.")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    batch_ids_path = out_file.with_name(out_file.stem + "_batch_ids.txt")
    batch_inputs_dir = out_file.parent / (out_file.stem + "_batch_inputs")

    # --- --from-results-dir: parse pre-downloaded output files ---
    if args.from_results_dir is not None:
        rows = collect_results_from_local_dir(args.from_results_dir)
        if rows:
            write_output_jsonl(rows, out_file)
        else:
            logger.warning("No rows parsed.")
        return

    # --- --collect-only: poll existing batch IDs ---
    if args.collect_only:
        if not batch_ids_path.exists():
            raise FileNotFoundError(
                f"--collect-only requires {batch_ids_path}; run without that flag first."
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        import openai
        client = openai.OpenAI(api_key=api_key)
        batch_ids = load_batch_ids(batch_ids_path)
        final = poll_until_done(client, batch_ids, args.poll_interval)
        rows = collect_results(client, final)
        if rows:
            append_output_jsonl(rows, out_file)
        return

    # --- Load prompts ---
    if args.prompts_dir is None:
        raise ValueError("--prompts-dir is required unless --collect-only or --from-results-dir is set.")

    skip_ids = load_already_processed_ids(out_file) if args.resume else set()
    items = load_prompts(args.prompts_dir, glob=args.prompts_glob, skip_ids=skip_ids)
    if not items:
        logger.info("No prompts to process.")
        return

    # --- Add system prompt ---
    system_content = (
        args.system_prompt_file.read_text(encoding="utf-8").strip()
        if args.system_prompt_file else DEFAULT_SYSTEM_PROMPT
    )
    items_with_sys: list[tuple[str, list[dict[str, str]]]] = [
        (cid, [{"role": "system", "content": system_content}] + msgs)
        for cid, msgs in items
    ]

    # --- Split into batches ---
    batches = split_into_batches_by_token(
        items_with_sys,
        model=args.model,
        est_output_tokens=args.est_output_tokens,
        max_tokens=args.max_tokens_per_batch,
    )
    logger.info(
        "Split %d prompts into %d batch(es) for model %s.",
        len(items), len(batches), args.model,
    )

    # --- Cost estimate ---
    sample = items_with_sys[:min(200, len(items_with_sys))]
    avg_tok = sum(count_prompt_tokens(msgs, args.model) for _, msgs in sample) / len(sample)
    total_prompt = round(avg_tok * len(items))
    total_out = len(items) * args.est_output_tokens
    cost_in  = total_prompt / 1e6 * 0.075
    cost_out = total_out    / 1e6 * 0.150
    logger.info("Cost estimate (Batch API 50%% discount):")
    logger.info(
        "  %d prompts  avg %.0f tok/req  Input $%.4f  Output $%.4f  Total $%.4f",
        len(items), avg_tok, cost_in, cost_out, cost_in + cost_out,
    )

    if args.prepare_only:
        batch_inputs_dir.mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(batches):
            write_batch_input_file(
                batch, batch_inputs_dir / f"batch_input_{i:04d}.jsonl", args.model,
            )
        logger.info("--prepare-only: done.")
        return

    # --- Submit, poll, collect ---
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment or .env file.")
    import openai
    client = openai.OpenAI(api_key=api_key)

    batch_inputs_dir.mkdir(parents=True, exist_ok=True)
    submitted_ids: list[str] = []
    task_name = out_file.stem
    for i, batch in enumerate(batches):
        p = batch_inputs_dir / f"batch_input_{i:04d}.jsonl"
        write_batch_input_file(batch, p, args.model)
        bid = upload_and_submit(
            client, p, i, len(batches),
            description=f"{task_name} {i + 1}/{len(batches)}",
        )
        submitted_ids.append(bid)
        save_batch_ids(submitted_ids, batch_ids_path)

    final = poll_until_done(client, submitted_ids, args.poll_interval)
    rows = collect_results(client, final)
    logger.info("Total rows collected: %d", len(rows))
    if rows:
        append_output_jsonl(rows, out_file)
    else:
        logger.warning("No rows collected — check batch logs.")


if __name__ == "__main__":
    main()
