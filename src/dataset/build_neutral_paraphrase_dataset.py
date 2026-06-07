"""Build a neutral-paraphrase dataset via the OpenAI Batch API.

For each unique (source_id, base_text) the model rewrites the utterance as a
semantically equivalent but affectively neutral paraphrase — removing any
emotional colouring while preserving meaning.

Output: dataset/neutral_paraphrases/neutral_paraphrases.jsonl
  Each row = one source with fields:
    source_id, base_text, neutral_paraphrase, meaning_preserved_score

The resulting activations extracted by extract_base_text_residual_stream.py
can then be subtracted from emotion-intensity activations to produce clean
CAA steering vectors.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv

from utils.constants import (
    DEFAULT_MAX_TOKENS_PER_BATCH,
    MAX_REQUESTS_PER_BATCH,
    POLL_INTERVAL_SEC,
)
from utils.prompts import NEUTRAL_PARAPHRASE_SCHEMA, NEUTRAL_PARAPHRASE_SYSTEM_PROMPT
from openai_utils.openai_batch import (
    count_prompt_tokens as _count_msg_tokens,
    load_batch_ids,
    poll_until_done,
    save_batch_ids,
    upload_and_submit,
    write_output_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Estimated output tokens per request (~50 tok rewrite + JSON overhead).
EST_OUTPUT_TOKENS_PER_REQ = 80

SYSTEM_PROMPT: str = NEUTRAL_PARAPHRASE_SYSTEM_PROMPT

# Candidate loading

def load_candidates(jsonl_path: Path) -> list[tuple[str, str]]:
    """Return ordered list of (source_id, base_text) from emotion_rewrites.jsonl.

    One entry per unique source_id (insertion order of the first occurrence).
    """
    seen: dict[str, str] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["source_id"]
            if sid not in seen:
                seen[sid] = rec["base_text"]
    candidates = list(seen.items())
    logger.info("Loaded %d unique source texts from %s", len(candidates), jsonl_path)
    return candidates


# Message construction

def _build_messages(base_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f'Utterance: "{base_text}"'},
    ]


# Token counting

def count_prompt_tokens(base_text: str, model: str) -> int:
    return _count_msg_tokens(_build_messages(base_text), model)


# Batch splitting

def split_into_batches(
    candidates: list[tuple[str, str]],
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_BATCH,
    max_requests: int = MAX_REQUESTS_PER_BATCH,
) -> list[list[tuple[str, str]]]:
    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_tokens = 0

    for sid, text in candidates:
        task_tokens = count_prompt_tokens(text, model) + EST_OUTPUT_TOKENS_PER_REQ
        overflow_req = len(current) >= max_requests
        overflow_tok = current_tokens + task_tokens > max_tokens
        if current and (overflow_req or overflow_tok):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append((sid, text))
        current_tokens += task_tokens

    if current:
        batches.append(current)
    return batches


# JSONL batch line

def _make_batch_line(source_id: str, base_text: str, model: str) -> dict[str, Any]:
    return {
        "custom_id": source_id,
        "method":    "POST",
        "url":       "/v1/chat/completions",
        "body": {
            "model":    model,
            "messages": _build_messages(base_text),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name":   "NeutralParaphraseOutput",
                    "schema": NEUTRAL_PARAPHRASE_SCHEMA,
                    "strict": True,
                },
            },
        },
    }


def write_batch_input_file(
    candidates: list[tuple[str, str]],
    path: Path,
    model: str,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for sid, text in candidates:
            fh.write(json.dumps(_make_batch_line(sid, text, model), ensure_ascii=False) + "\n")


# Parse output lines

def _parse_output_lines(
    lines: list[str],
    source_label: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """Parse JSONL output lines from Batch API. Returns (rows, n_ok, n_err)."""
    rows: list[dict[str, Any]] = []
    n_ok = n_err = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        source_id = obj.get("custom_id", "")
        resp = obj.get("response")
        if resp is None or resp.get("status_code") != 200:
            err = obj.get("error") or (resp or {}).get("body")
            logger.warning("Failed request  custom_id=%r  err=%s", source_id, err)
            n_err += 1
            continue
        raw_json = resp["body"]["choices"][0]["message"]["content"]
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning(
                "JSON decode error  custom_id=%r  raw=%r", source_id, raw_json[:200]
            )
            n_err += 1
            continue
        rows.append({
            "source_id":               source_id,
            "base_text":               data.get("base_text", ""),
            "neutral_paraphrase":      data.get("neutral_paraphrase", ""),
            "meaning_preserved_score": data.get("meaning_preserved_score"),
        })
        n_ok += 1
    return rows, n_ok, n_err


def collect_results_from_local_dir(
    local_dir: Path,
    glob: str = "*.jsonl",
) -> list[dict[str, Any]]:
    """Parse all JSONL files matching *glob* in *local_dir*; return flat row list."""
    all_rows: list[dict[str, Any]] = []
    output_files = sorted(local_dir.glob(glob))
    if not output_files:
        logger.warning("No files matching %r found in %s", glob, local_dir)
        return all_rows
    for path in output_files:
        rows, n_ok, n_err = _parse_output_lines(
            path.read_text(encoding="utf-8").splitlines(),
            source_label=path.name,
        )
        logger.info("  local %s  parsed_ok=%d  errors=%d", path.name, n_ok, n_err)
        all_rows.extend(rows)
    logger.info("Local results total: %d rows from %d file(s)", len(all_rows), len(output_files))
    return all_rows


def collect_results(
    client: openai.OpenAI,
    final_batches: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bid, b in final_batches.items():
        if b.status != "completed":
            logger.warning("Batch %s ended with status=%s — skipping.", bid, b.status)
            continue
        if not b.output_file_id:
            logger.warning("Batch %s has no output_file_id.", bid)
            continue
        logger.info("Downloading  batch=%s  file=%s …", bid, b.output_file_id)
        text = client.files.content(b.output_file_id).text
        batch_rows, n_ok, n_err = _parse_output_lines(
            text.splitlines(), source_label=bid
        )
        rows.extend(batch_rows)
        logger.info("  batch=%s  parsed_ok=%d  errors=%d", bid, n_ok, n_err)
    return rows


# Estimate / report

def print_estimate(
    candidates: list[tuple[str, str]],
    batches: list[list[tuple[str, str]]],
    model: str,
) -> None:
    import random
    sample = candidates if len(candidates) <= 500 else random.sample(candidates, 500)
    total_prompt = sum(count_prompt_tokens(t, model) for _, t in sample)
    avg = total_prompt / len(sample)
    total_prompt_est = round(avg * len(candidates))
    total_out = len(candidates) * EST_OUTPUT_TOKENS_PER_REQ
    cost_in  = total_prompt_est / 1e6 * 0.075
    cost_out = total_out        / 1e6 * 0.150
    n_batches = len(batches)

    logger.info("")
    logger.info("=" * 66)
    logger.info("Batch API estimate  (model: %s)", model)
    logger.info("-" * 66)
    logger.info("  Total requests           : %10d", len(candidates))
    logger.info("  Number of batch jobs     : %10d  (serialised)", n_batches)
    logger.info("  Avg prompt tokens / req  : %10.0f", avg)
    logger.info("  Est. prompt tokens total : %10.1f M", total_prompt_est / 1e6)
    logger.info("  Est. output tokens total : %10.1f M", total_out / 1e6)
    logger.info("-" * 66)
    logger.info("  Cost estimate (Batch API 50 %% discount)")
    logger.info("    Input  ($0.075 / 1M)   : $%9.2f", cost_in)
    logger.info("    Output ($0.150 / 1M)   : $%9.2f", cost_out)
    logger.info("    Total                  : $%9.2f", cost_in + cost_out)
    logger.info("-" * 66)
    logger.info("  SLA per batch            : 24 h")
    logger.info("=" * 66)
    logger.info("")


# CLI

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a neutral-paraphrase dataset via the OpenAI Batch API.\n"
            "Output: dataset/neutral_paraphrases/neutral_paraphrases.jsonl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-jsonl", type=Path,
        default=Path("dataset/emotion_rewrites/emotion_rewrites.jsonl"),
        help=(
            "emotion_rewrites.jsonl produced by build_emotion_rewrite_dataset.py "
            "(used to read unique source_id / base_text pairs). "
            "Default: dataset/emotion_rewrites/emotion_rewrites.jsonl"
        ),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("dataset/neutral_paraphrases"),
        help="Output directory (default: dataset/neutral_paraphrases).",
    )
    p.add_argument(
        "--model", default="gpt-4.1-mini",
        help="OpenAI model identifier (default: gpt-4.1-mini).",
    )
    p.add_argument(
        "--max-tokens-per-batch", type=int, default=DEFAULT_MAX_TOKENS_PER_BATCH,
        help="Maximum token budget per batch job (default: 20,000,000).",
    )
    p.add_argument(
        "--poll-interval", type=int, default=POLL_INTERVAL_SEC,
        help="Seconds between status-poll requests (default: 60).",
    )
    p.add_argument(
        "--prepare-only", action="store_true",
        help="Write batch JSONL files and print estimate; exit without uploading.",
    )
    p.add_argument(
        "--collect-only", action="store_true",
        help=(
            "Skip preparation / submission; load batch IDs from batch_ids.txt "
            "and go straight to polling & collection."
        ),
    )
    p.add_argument(
        "--from-results-dir", type=Path, default=None, metavar="DIR",
        help=(
            "Parse all *.jsonl files in DIR as already-downloaded Batch API outputs "
            "and write the final dataset without contacting the OpenAI API."
        ),
    )
    p.add_argument(
        "--n-samples", type=int, default=None,
        help="Debug: cap total candidates to first N utterances.",
    )
    return p.parse_args()


# Main

def main() -> None:
    load_dotenv()
    args = _parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_ids_path = output_dir / "batch_ids.txt"
    out_jsonl_path = output_dir / "neutral_paraphrases.jsonl"

    if args.from_results_dir is not None:
        results_dir: Path = args.from_results_dir
        if not results_dir.is_dir():
            raise NotADirectoryError(
                f"--from-results-dir is not a directory: {results_dir}"
            )
        logger.info("Loading pre-downloaded results from %s …", results_dir)
        rows = collect_results_from_local_dir(results_dir)
        logger.info("Total rows collected: %d", len(rows))
        if rows:
            write_output_jsonl(rows, out_jsonl_path)
        else:
            logger.warning("No rows parsed — check the result files.")
        return

    candidates = load_candidates(args.input_jsonl)
    if args.n_samples is not None:
        candidates = candidates[: args.n_samples]
        logger.info("Capped to %d candidates (--n-samples).", len(candidates))

    batches = split_into_batches(
        candidates,
        model=args.model,
        max_tokens=args.max_tokens_per_batch,
    )
    logger.info("Will submit %d batch job(s).", len(batches))
    print_estimate(candidates, batches, args.model)

    if args.prepare_only:
        for i, batch in enumerate(batches):
            p = output_dir / f"batch_input_{i:04d}.jsonl"
            write_batch_input_file(batch, p, args.model)
            logger.info("Batch input written → %s  (%d requests)", p, len(batch))
        logger.info("--prepare-only: done.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Set it in your environment or .env file."
        )
    client = openai.OpenAI(api_key=api_key)

    if args.collect_only:
        if not batch_ids_path.exists():
            raise FileNotFoundError(
                f"--collect-only requires {batch_ids_path}; run without that flag first."
            )
        batch_ids = load_batch_ids(batch_ids_path)
        final = poll_until_done(client, batch_ids, args.poll_interval)
        rows = collect_results(client, final)
        if rows:
            write_output_jsonl(rows, out_jsonl_path)
        return

    submitted_ids: list[str] = []
    for i, batch in enumerate(batches):
        p = output_dir / f"batch_input_{i:04d}.jsonl"
        write_batch_input_file(batch, p, args.model)
        bid = upload_and_submit(
            client, p, i, len(batches),
            f"Neutral paraphrase {i + 1}/{len(batches)}",
        )
        submitted_ids.append(bid)
        save_batch_ids(submitted_ids, batch_ids_path)

    final = poll_until_done(client, submitted_ids, args.poll_interval)
    rows  = collect_results(client, final)

    logger.info("Total rows collected: %d", len(rows))
    if rows:
        write_output_jsonl(rows, out_jsonl_path)
    else:
        logger.warning("No rows collected — check batch logs.")


if __name__ == "__main__":
    main()
