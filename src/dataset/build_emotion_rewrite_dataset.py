from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv

# Sibling-script import for candidate loading (src/dataset/ on path)
sys.path.insert(0, str(Path(__file__).parent))
from build_affective_rewrite_dataset import load_all_candidates  # noqa: E402

from utils.constants import (
    BATCH_ENDPOINT,
    DEFAULT_MAX_TOKENS_PER_BATCH,
    INTENSITY_LEVELS,
    MAX_REQUESTS_PER_BATCH,
    PLUTCHIK_EMOTIONS,
    POLL_INTERVAL_SEC,
)
from utils.prompts import EMOTION_REWRITE_SCHEMA, EMOTION_REWRITE_SYSTEM_PROMPT
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

# Local alias used in schema and CSV field definitions.
EMOTION_CATEGORIES: list[str] = PLUTCHIK_EMOTIONS

# Estimated output tokens per request (8 rewrites × ~30 tok each + JSON overhead).
EST_OUTPUT_TOKENS_PER_REQ = 280

SYSTEM_PROMPT: str = EMOTION_REWRITE_SYSTEM_PROMPT

CSV_FIELDNAMES: list[str] = [
    "source_dataset",
    "source_split",
    "source_id",
    "intensity_level",
    "base_text",
    "meaning_preserved_score",
    *[f"{e}_rewrite" for e in EMOTION_CATEGORIES],
]

# Task = (source_id, source_dataset, source_split, utterance, intensity_level)
Task = tuple[str, str, str, str, str]


# Message construction

def _build_messages(base_text: str, intensity_level: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Intensity level: {intensity_level}\n\n"
                f'Utterance: "{base_text}"'
            ),
        },
    ]


# Token counting

def count_prompt_tokens(base_text: str, intensity_level: str, model: str) -> int:
    """Count prompt tokens for a single (utterance, intensity_level) request."""
    return _count_msg_tokens(_build_messages(base_text, intensity_level), model)


def estimate_dataset_tokens(tasks: list[Task], model: str) -> tuple[int, float]:
    """Return (total_prompt_tokens, avg_per_request) via sampling."""
    sample = tasks if len(tasks) <= 500 else random.sample(tasks, 500)
    sampled = sum(count_prompt_tokens(t[3], t[4], model) for t in sample)
    avg = sampled / len(sample)
    return round(avg * len(tasks)), avg


# Task generation

def generate_tasks(
    candidates: list[tuple[str, str, str, str]],
    intensity_levels: list[str] = INTENSITY_LEVELS,
) -> list[Task]:
    """Expand (source_id, utterance, dataset, split) × intensity_levels → flat task list."""
    tasks: list[Task] = []
    for sid, utt, ds, sp in candidates:
        for level in intensity_levels:
            tasks.append((sid, ds, sp, utt, level))
    return tasks


# Batch splitting

def split_into_batches(
    tasks: list[Task],
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_BATCH,
    max_requests: int = MAX_REQUESTS_PER_BATCH,
) -> list[list[Task]]:
    """Partition tasks into slices respecting both the token and request limits."""
    batches: list[list[Task]] = []
    current: list[Task] = []
    current_tokens = 0

    for task in tasks:
        sid, ds, sp, text, level = task
        task_tokens = count_prompt_tokens(text, level, model) + EST_OUTPUT_TOKENS_PER_REQ
        overflow_req = len(current) >= max_requests
        overflow_tok = current_tokens + task_tokens > max_tokens
        if current and (overflow_req or overflow_tok):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(task)
        current_tokens += task_tokens

    if current:
        batches.append(current)
    return batches


# JSONL writing

def _make_batch_line(task: Task, model: str) -> dict[str, Any]:
    sid, ds, sp, text, level = task
    return {
        "custom_id": f"{sid}__intensity_{level}",
        "method":    "POST",
        "url":       BATCH_ENDPOINT,
        "body": {
            "model":    model,
            "messages": _build_messages(text, level),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name":   "EmotionRewriteOutput",
                    "schema": EMOTION_REWRITE_SCHEMA,
                    "strict": True,
                },
            },
        },
    }


def write_batch_input_file(tasks: list[Task], path: Path, model: str) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(_make_batch_line(task, model), ensure_ascii=False) + "\n")


# Result parsing

def _parse_output_lines(
    lines: list[str],
    candidates_map: dict[str, tuple[str, str, str]],
    source_label: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """Parse JSONL output lines into row dicts. Returns (rows, n_ok, n_err)."""
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
        custom_id = obj.get("custom_id", "")
        if "__intensity_" not in custom_id:
            logger.warning("Unexpected custom_id format: %r  (source=%s)", custom_id, source_label)
            continue
        sid, intensity_level = custom_id.rsplit("__intensity_", 1)
        resp = obj.get("response")
        if resp is None or resp.get("status_code") != 200:
            err = obj.get("error") or (resp or {}).get("body")
            logger.warning("Failed request  custom_id=%r  err=%s", custom_id, err)
            n_err += 1
            continue
        raw_json = resp["body"]["choices"][0]["message"]["content"]
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning(
                "JSON decode error  custom_id=%r  raw=%r", custom_id, raw_json[:200]
            )
            n_err += 1
            continue
        ds, sp, _ = candidates_map.get(sid, ("", "", ""))
        row: dict[str, Any] = {
            "source_dataset":          ds,
            "source_split":            sp,
            "source_id":               sid,
            "intensity_level":         intensity_level,
            "base_text":               data.get("base_text", ""),
            "meaning_preserved_score": data.get("meaning_preserved_score"),
            **{
                f"{e}_rewrite": data.get(f"{e}_rewrite", "")
                for e in EMOTION_CATEGORIES
            },
        }
        rows.append(row)
        n_ok += 1
    return rows, n_ok, n_err


def collect_results_from_local_dir(
    local_dir: Path,
    candidates_map: dict[str, tuple[str, str, str]],
    glob: str = "*_output.jsonl",
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
            candidates_map,
            source_label=path.name,
        )
        logger.info("  local %s  parsed_ok=%d  errors=%d", path.name, n_ok, n_err)
        all_rows.extend(rows)
    logger.info("Local results total: %d rows from %d file(s)", len(all_rows), len(output_files))
    return all_rows


def load_completed_ids_from_local_dir(local_dir: Path) -> set[tuple[str, str]]:
    """Return set of (source_id, intensity_level) that succeeded in local output files."""
    done: set[tuple[str, str]] = set()
    for path in sorted(local_dir.glob("*_output.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = obj.get("custom_id", "")
            if "__intensity_" not in custom_id:
                continue
            resp = obj.get("response")
            if resp and resp.get("status_code") == 200:
                sid, level = custom_id.rsplit("__intensity_", 1)
                done.add((sid, level))
    logger.info(
        "Completed IDs loaded from %s: %d (source_id, intensity_level) pairs",
        local_dir, len(done),
    )
    return done


def collect_results(
    client: openai.OpenAI,
    final_batches: dict[str, Any],
    candidates_map: dict[str, tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Download all output files; return list of flat row dicts."""
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
            text.splitlines(), candidates_map, source_label=bid
        )
        rows.extend(batch_rows)
        logger.info("  batch=%s  parsed_ok=%d  errors=%d", bid, n_ok, n_err)
    return rows


# Output writing

def write_output_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV  → %s  (%d rows)", path, len(rows))


# Estimation / reporting

def print_estimate(
    tasks: list[Task],
    batches: list[list[Task]],
    model: str,
    total_prompt_tokens: int,
    avg_per_req: float,
) -> None:
    n_batches = len(batches)
    total_out_tokens = len(tasks) * 300
    cost_in  = total_prompt_tokens / 1e6 * 0.075
    cost_out = total_out_tokens    / 1e6 * 0.150

    logger.info("")
    logger.info("=" * 66)
    logger.info("Batch API estimate  (model: %s)", model)
    logger.info("-" * 66)
    logger.info("  Total tasks              : %10d", len(tasks))
    logger.info(
        "  Tasks breakdown          :  %d utterances × %d intensity level(s)",
        len(tasks) // len(set(t[4] for t in tasks)) if tasks else 0,
        len(set(t[4] for t in tasks)),
    )
    logger.info("  Number of batch jobs     : %10d  (serialised)", n_batches)
    logger.info(
        "  Requests per batch       :  %s",
        "  ".join(str(len(b)) for b in batches[:8])
        + ("  …" if n_batches > 8 else ""),
    )
    logger.info("  Avg prompt tokens / req  : %10.0f", avg_per_req)
    logger.info("  Est. prompt tokens total : %10.1f M", total_prompt_tokens / 1e6)
    logger.info("  Est. output tokens total : %10.1f M", total_out_tokens    / 1e6)
    logger.info("-" * 66)
    logger.info("  Cost estimate (Batch API 50 %% discount)")
    logger.info("    Input  ($0.075 / 1M)   : $%9.2f", cost_in)
    logger.info("    Output ($0.150 / 1M)   : $%9.2f", cost_out)
    logger.info("    Total                  : $%9.2f", cost_in + cost_out)
    logger.info("-" * 66)
    logger.info("  Tier-2 queue limit       : 20,000,000 tokens (serial)")
    logger.info("  SLA per batch            : 24 h")
    logger.info(
        "  Est. total wall-clock    : %d min – %d h  (%d batches × 5–60 min)",
        n_batches * 5, (n_batches * 60) // 60, n_batches,
    )
    logger.info("=" * 66)
    logger.info("")


# CLI

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build emotion-category contrastive rewrites via OpenAI Batch API.\n"
            "Generates joy/anger/fear/sadness/neutral rewrites at matched intensity\n"
            "to enable computing clean emotion-direction vectors for CAA.\n\n"
            "Output: dataset/emotion_rewrites/emotion_rewrites.{csv,jsonl}\n"
            "  Each row = one (source_id, intensity_level) with 5 emotion rewrites."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("dataset/emotion_rewrites"),
        help="Directory for output files (default: dataset/emotion_rewrites).",
    )
    p.add_argument(
        "--model", default="gpt-4.1-mini",
        help="OpenAI model identifier (default: gpt-4.1-mini).",
    )
    p.add_argument(
        "--intensity-levels", nargs="+", default=INTENSITY_LEVELS,
        choices=INTENSITY_LEVELS, metavar="LEVEL",
        help="Intensity levels to generate (default: low medium high).",
    )
    p.add_argument(
        "--max-tokens-per-batch", type=int, default=DEFAULT_MAX_TOKENS_PER_BATCH,
        help=(
            "Maximum prompt tokens per batch job "
            "(default: 20,000,000 = Tier-2 queue limit). "
            "Adjust for your account tier."
        ),
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
        "--resume-from", type=Path, default=None, metavar="DIR",
        help=(
            "Directory containing already-downloaded *_output.jsonl files. "
            "Rows found there are loaded as-is and their (source_id, intensity_level) "
            "pairs are excluded from the new batch submission."
        ),
    )
    p.add_argument(
        "--from-results-dir", type=Path, default=None, metavar="DIR",
        help=(
            "Parse all *.jsonl files in DIR as already-downloaded Batch API outputs "
            "and write the final dataset without contacting the OpenAI API. "
            "Example: --from-results-dir dataset/emotion_rewrite_batch_results"
        ),
    )
    p.add_argument(
        "--n-samples", type=int, default=None,
        help="Debug: cap total candidates to first N utterances.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for candidate sampling (default: 42).",
    )
    return p.parse_args()


# Main

def main() -> None:
    load_dotenv()
    args = _parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_ids_path = output_dir / "batch_ids.txt"
    out_csv_path   = output_dir / "emotion_rewrites.csv"
    out_jsonl_path = output_dir / "emotion_rewrites.jsonl"

    candidates = load_all_candidates(seed=args.seed, n_samples=args.n_samples)
    logger.info("Candidates loaded: %d", len(candidates))

    candidates_map: dict[str, tuple[str, str, str]] = {
        sid: (ds, sp, utt) for sid, utt, ds, sp in candidates
    }

    # --from-results-dir: build dataset from pre-downloaded result files
    if args.from_results_dir is not None:
        results_dir: Path = args.from_results_dir
        if not results_dir.is_dir():
            raise NotADirectoryError(f"--from-results-dir path is not a directory: {results_dir}")
        logger.info("Loading pre-downloaded results from %s …", results_dir)
        rows = collect_results_from_local_dir(results_dir, candidates_map, glob="*.jsonl")
        logger.info("Total rows collected: %d", len(rows))
        if rows:
            write_output_csv(rows, out_csv_path)
            write_output_jsonl(rows, out_jsonl_path)
        else:
            logger.warning("No rows parsed — check the result files.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = openai.OpenAI(api_key=api_key)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    local_rows: list[dict[str, Any]] = []
    completed_ids: set[tuple[str, str]] = set()
    if args.resume_from is not None:
        if not args.resume_from.is_dir():
            raise NotADirectoryError(f"--resume-from path is not a directory: {args.resume_from}")
        logger.info("Loading local results from %s …", args.resume_from)
        local_rows = collect_results_from_local_dir(args.resume_from, candidates_map)
        completed_ids = {(r["source_id"], r["intensity_level"]) for r in local_rows}
        logger.info("Skipping %d already-completed (source_id, intensity_level) pairs.", len(completed_ids))

    if not args.collect_only:
        tasks = generate_tasks(candidates, intensity_levels=args.intensity_levels)
        if completed_ids:
            before = len(tasks)
            tasks = [t for t in tasks if (t[0], t[4]) not in completed_ids]
            logger.info(
                "Tasks after skipping completed: %d → %d (skipped %d)",
                before, len(tasks), before - len(tasks),
            )
        logger.info(
            "Total tasks (%d utterances × %d intensity levels): %d",
            len(candidates), len(args.intensity_levels), len(tasks),
        )

        if not tasks:
            logger.info("All tasks already completed in --resume-from directory.")
            all_final_batches = {}
        else:
            logger.info("Estimating token counts …")
            total_prompt_tokens, avg_per_req = estimate_dataset_tokens(tasks, args.model)

            batches = split_into_batches(
                tasks, model=args.model, max_tokens=args.max_tokens_per_batch,
            )

            print_estimate(tasks, batches, args.model, total_prompt_tokens, avg_per_req)

            jsonl_paths: list[Path] = []
            for i, batch_tasks in enumerate(batches):
                p = output_dir / f"batch_input_{i:04d}.jsonl"
                write_batch_input_file(batch_tasks, p, args.model)
                jsonl_paths.append(p)
                logger.info(
                    "Wrote %s  (%d requests, %.2f MB)",
                    p.name, len(batch_tasks), p.stat().st_size / 1e6,
                )

            if args.prepare_only:
                logger.info("--prepare-only: done.  JSONL files written; nothing uploaded.")
                return

            submitted_ids: list[str] = []
            all_final_batches = {}

            for i, p in enumerate(jsonl_paths):
                bid = upload_and_submit(
                    client, p, i, len(batches),
                    f"Emotion rewrite {i + 1}/{len(batches)}",
                )
                submitted_ids.append(bid)
                save_batch_ids(submitted_ids, batch_ids_path)

                logger.info(
                    "Waiting for batch %d/%d to finish before submitting next …",
                    i + 1, len(batches),
                )
                final = poll_until_done(client, [bid], args.poll_interval)
                all_final_batches.update(final)

    else:
        if not batch_ids_path.exists():
            raise FileNotFoundError(
                f"{batch_ids_path} not found.  "
                "Run without --collect-only first to submit batches."
            )
        submitted_ids = load_batch_ids(batch_ids_path)
        all_final_batches = poll_until_done(client, submitted_ids, args.poll_interval)

    rows = collect_results(client, all_final_batches, candidates_map)
    if local_rows:
        logger.info(
            "Merging %d local rows + %d new API rows …", len(local_rows), len(rows)
        )
        rows = local_rows + rows
    logger.info("Total rows collected: %d", len(rows))

    if rows:
        write_output_csv(rows, out_csv_path)
        write_output_jsonl(rows, out_jsonl_path)
        logger.info("CSV  → %s", out_csv_path.resolve())
        logger.info("JSONL→ %s", out_jsonl_path.resolve())
    else:
        logger.warning(
            "No fully-annotated rows to write yet.  "
            "Check batch status with: openai batches list"
        )


if __name__ == "__main__":
    main()
