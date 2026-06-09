"""Generic OpenAI Batch API helpers.

Reusable across any task that submits many chat-completion requests via the
Batch API (50% cost discount, 24-hour SLA).

Typical usage
-------------
    from openai_utils.openai_batch import (
        make_batch_line,
        split_into_batches_by_token,
        upload_and_submit,
        poll_until_done,
        collect_results,
        collect_results_from_local_dir,
        save_batch_ids,
        load_batch_ids,
        write_output_jsonl,
        count_prompt_tokens,
    )
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import openai
import tiktoken
from dotenv import load_dotenv

from utils.constants import (
    BATCH_ENDPOINT,
    COMPLETION_WINDOW,
    DEFAULT_MAX_TOKENS_PER_BATCH,
    MAX_REQUESTS_PER_BATCH,
    POLL_INTERVAL_SEC,
    TERMINAL_STATUSES,
)

logger = logging.getLogger(__name__)

_enc: Optional[tiktoken.Encoding] = None


def _get_encoder(model: str) -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        try:
            _enc = tiktoken.encoding_for_model(model)
        except KeyError:
            _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def count_prompt_tokens(messages: list[dict[str, str]], model: str) -> int:
    """Estimate token count for a list of chat messages."""
    enc = _get_encoder(model)
    total = 3  # reply priming
    for msg in messages:
        total += 4
        total += len(enc.encode(msg["content"]))
    return total


def make_batch_line(
    custom_id: str,
    messages: list[dict[str, str]],
    model: str,
    *,
    response_format: Optional[dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Build one JSONL request line for the Batch API.

    Parameters
    ----------
    custom_id       : Unique identifier returned in the output (e.g. file stem).
    messages        : Chat messages list (system + user).
    model           : OpenAI model identifier.
    response_format : Optional response_format dict (e.g. json_schema).
    max_tokens      : Optional max_tokens for the completion.
    """
    body: dict[str, Any] = {"model": model, "messages": messages}
    if response_format is not None:
        body["response_format"] = response_format
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return {"custom_id": custom_id, "method": "POST", "url": BATCH_ENDPOINT, "body": body}


def split_into_batches_by_token(
    items: list[tuple[str, list[dict[str, str]]]],
    model: str,
    est_output_tokens: int = 512,
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_BATCH,
    max_requests: int = MAX_REQUESTS_PER_BATCH,
) -> list[list[tuple[str, list[dict[str, str]]]]]:
    """Split (custom_id, messages) pairs into batches respecting token and request limits.

    Parameters
    ----------
    items             : List of (custom_id, messages) pairs.
    model             : Model for token counting.
    est_output_tokens : Estimated output tokens per request.
    max_tokens        : Max total tokens per batch job.
    max_requests      : Max requests per batch job.
    """
    batches: list[list[tuple[str, list[dict[str, str]]]]] = []
    current: list[tuple[str, list[dict[str, str]]]] = []
    current_tokens = 0
    for custom_id, messages in items:
        task_tokens = count_prompt_tokens(messages, model) + est_output_tokens
        if current and (len(current) >= max_requests or current_tokens + task_tokens > max_tokens):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append((custom_id, messages))
        current_tokens += task_tokens
    if current:
        batches.append(current)
    return batches


def write_batch_input_file(
    items: list[tuple[str, list[dict[str, str]]]],
    path: Path,
    model: str,
    *,
    response_format: Optional[dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
) -> None:
    """Write a batch input JSONL file from (custom_id, messages) pairs."""
    with path.open("w", encoding="utf-8") as fh:
        for custom_id, messages in items:
            line = make_batch_line(
                custom_id, messages, model,
                response_format=response_format,
                max_tokens=max_tokens,
            )
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    logger.info("Batch input written → %s  (%d requests)", path, len(items))


def upload_and_submit(
    client: openai.OpenAI,
    jsonl_path: Path,
    batch_index: int,
    n_batches: int,
    description: str = "",
    *,
    queue_retry_interval: int = 120,
) -> str:
    """Upload a batch input file and submit a batch job. Returns batch_id."""
    jsonl_path = Path(jsonl_path)
    size_mb = jsonl_path.stat().st_size / 1e6
    logger.info(
        "Uploading batch %d/%d: %s (%.2f MB) …",
        batch_index + 1, n_batches, jsonl_path.name, size_mb,
    )
    with jsonl_path.open("rb") as fh:
        file_obj = client.files.create(file=fh, purpose="batch")
    logger.info("  file_id=%s", file_obj.id)
    meta_desc = description or f"batch {batch_index + 1}/{n_batches}"
    while True:
        try:
            batch = client.batches.create(
                input_file_id=file_obj.id,
                endpoint=BATCH_ENDPOINT,
                completion_window=COMPLETION_WINDOW,
                metadata={"description": meta_desc},
            )
            logger.info("  batch_id=%s  status=%s", batch.id, batch.status)
            return batch.id
        except openai.BadRequestError as exc:
            if "Enqueued token limit" in str(exc):
                logger.warning(
                    "Enqueued token limit reached; retrying in %ds …",
                    queue_retry_interval,
                )
                time.sleep(queue_retry_interval)
            else:
                raise


def poll_until_done(
    client: openai.OpenAI,
    batch_ids: list[str],
    poll_interval: int = POLL_INTERVAL_SEC,
) -> dict[str, Any]:
    """Poll batch jobs until all reach a terminal status. Returns {batch_id: batch}."""
    remaining = set(batch_ids)
    final: dict[str, Any] = {}
    logger.info("Polling %d batch(es) every %ds …", len(remaining), poll_interval)
    while remaining:
        done_this_round: set[str] = set()
        for bid in sorted(remaining):
            b = client.batches.retrieve(bid)
            rc = b.request_counts
            logger.info(
                "  %s  status=%-12s  completed=%d  failed=%d  total=%d",
                bid, b.status,
                rc.completed if rc else 0,
                rc.failed    if rc else 0,
                rc.total     if rc else 0,
            )
            if b.status in TERMINAL_STATUSES:
                final[bid] = b
                done_this_round.add(bid)
        remaining -= done_this_round
        if remaining:
            time.sleep(poll_interval)
    return final


def parse_output_lines(
    lines: list[str],
    source_label: str = "",
) -> tuple[list[dict[str, Any]], int, int]:
    """Parse raw Batch API JSONL output lines.

    Returns
    -------
    rows   : List of dicts: {"custom_id": str, "response_data": dict | str}
    n_ok   : Number of successfully parsed responses.
    n_err  : Number of failed / unparseable responses.
    """
    rows: list[dict[str, Any]] = []
    n_ok = n_err = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            n_err += 1
            continue
        custom_id = obj.get("custom_id", "")
        resp = obj.get("response")
        if resp is None or resp.get("status_code") != 200:
            err = obj.get("error") or (resp or {}).get("body")
            logger.warning("Failed request  custom_id=%r  err=%s", custom_id, err)
            n_err += 1
            continue
        raw_content = resp["body"]["choices"][0]["message"]["content"]
        try:
            response_data = json.loads(raw_content)
        except json.JSONDecodeError:
            # Strip markdown code fences (e.g. ```json ... ```) and retry
            stripped = raw_content.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[-1]
                stripped = stripped.rsplit("```", 1)[0].strip()
            try:
                response_data = json.loads(stripped)
            except json.JSONDecodeError:
                response_data = raw_content
                logger.warning(
                    "JSON decode error  custom_id=%r  raw=%r", custom_id, raw_content[:200]
                )
        rows.append({"custom_id": custom_id, "response_data": response_data})
        n_ok += 1
    return rows, n_ok, n_err


def collect_results(
    client: openai.OpenAI,
    final_batches: dict[str, Any],
) -> list[dict[str, Any]]:
    """Download and parse results from completed batch jobs."""
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
        batch_rows, n_ok, n_err = parse_output_lines(text.splitlines(), source_label=bid)
        rows.extend(batch_rows)
        logger.info("  batch=%s  parsed_ok=%d  errors=%d", bid, n_ok, n_err)
    return rows


def collect_results_from_local_dir(
    local_dir: Path,
    glob: str = "*.jsonl",
) -> list[dict[str, Any]]:
    """Parse pre-downloaded Batch API output files in a local directory."""
    all_rows: list[dict[str, Any]] = []
    output_files = sorted(local_dir.glob(glob))
    if not output_files:
        logger.warning("No files matching %r found in %s", glob, local_dir)
        return all_rows
    for path in output_files:
        rows, n_ok, n_err = parse_output_lines(
            path.read_text(encoding="utf-8").splitlines(),
            source_label=path.name,
        )
        logger.info("  local %s  parsed_ok=%d  errors=%d", path.name, n_ok, n_err)
        all_rows.extend(rows)
    logger.info("Total: %d rows from %d file(s)", len(all_rows), len(output_files))
    return all_rows


def save_batch_ids(ids: list[str], path: Path) -> None:
    """Save batch IDs to a text file for resuming later."""
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    logger.info("Batch IDs saved → %s", path)


def load_batch_ids(path: Path) -> list[str]:
    """Load batch IDs from a text file."""
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    logger.info("Loaded %d batch ID(s) from %s", len(ids), path)
    return ids


def write_output_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write parsed result rows to a JSONL file (overwrite)."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("JSONL → %s  (%d rows)", path, len(rows))


def _resolve_api_key(api_key: str | None = None) -> str:
    load_dotenv()
    resolved = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    return resolved


def make_client(api_key: str | None = None) -> openai.OpenAI:
    """Create an OpenAI client, resolving the API key from the environment if not given."""
    return openai.OpenAI(api_key=_resolve_api_key(api_key))


def summarise_jsonl(jsonl_path: str | Path) -> dict[str, Any]:
    """Return a summary dict (request count, size, first custom_id) for a JSONL file."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(path)
    line_count = 0
    first_custom_id = None
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            line_count += 1
            if first_custom_id is None:
                try:
                    first_custom_id = json.loads(line).get("custom_id")
                except json.JSONDecodeError:
                    first_custom_id = None
    return {
        "path":            str(path),
        "requests":        line_count,
        "size_bytes":      path.stat().st_size,
        "first_custom_id": first_custom_id,
    }


def retrieve_batch(batch_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    """Fetch the current status of a batch job."""
    client = make_client(api_key)
    return client.batches.retrieve(batch_id).model_dump(mode="json")


def download_batch_output(
    batch_id: str,
    output_path: str | Path | None = None,
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Download the output JSONL for a completed batch job.

    Raises ``RuntimeError`` if the batch is not yet in a terminal state or
    has no output file.
    """
    client = make_client(api_key)
    batch = client.batches.retrieve(batch_id)
    if batch.status not in TERMINAL_STATUSES:
        raise RuntimeError(f"Batch {batch_id} is not finished yet (status={batch.status}).")
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch_id} has no output_file_id (status={batch.status}).")
    content = client.files.content(batch.output_file_id).text
    out_path: Path | None = None
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
    return {
        "batch_id":       batch.id,
        "status":         batch.status,
        "output_file_id": batch.output_file_id,
        "output_path":    str(out_path) if out_path else None,
        "bytes":          len(content.encode("utf-8")),
    }
