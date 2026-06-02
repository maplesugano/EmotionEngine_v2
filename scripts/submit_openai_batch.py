from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv


BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_COMPLETION_WINDOW = "24h"
TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})


def _resolve_api_key(api_key: str | None = None) -> str:
    load_dotenv()
    resolved = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    return resolved


def make_client(api_key: str | None = None) -> openai.OpenAI:
    return openai.OpenAI(api_key=_resolve_api_key(api_key))


def summarize_jsonl(jsonl_path: str | Path) -> dict[str, Any]:
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
        "path": str(path),
        "requests": line_count,
        "size_bytes": path.stat().st_size,
        "first_custom_id": first_custom_id,
    }


def submit_batch_file(
    jsonl_path: str | Path,
    *,
    description: str,
    metadata: dict[str, str] | None = None,
    api_key: str | None = None,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    endpoint: str = BATCH_ENDPOINT,
) -> dict[str, Any]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(path)

    client = make_client(api_key)
    with path.open("rb") as fh:
        file_obj = client.files.create(file=fh, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata={"description": description, **(metadata or {})},
    )
    return {
        "input_path": str(path),
        "input_file_id": file_obj.id,
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": endpoint,
        "completion_window": completion_window,
        "metadata": batch.metadata,
    }


def retrieve_batch(batch_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    client = make_client(api_key)
    batch = client.batches.retrieve(batch_id)
    return batch.model_dump(mode="json")


def download_batch_output(
    batch_id: str,
    output_path: str | Path | None = None,
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    client = make_client(api_key)
    batch = client.batches.retrieve(batch_id)
    if batch.status not in TERMINAL_STATUSES:
        raise RuntimeError(f"Batch {batch_id} is not finished yet (status={batch.status}).")
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch_id} has no output_file_id (status={batch.status}).")

    content = client.files.content(batch.output_file_id).text
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
    else:
        out_path = None

    return {
        "batch_id": batch.id,
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "output_path": str(out_path) if out_path else None,
        "bytes": len(content.encode("utf-8")),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit or inspect an OpenAI Batch API JSONL input file."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Upload a JSONL file and create a batch.")
    submit_parser.add_argument("jsonl", type=Path, help="Input JSONL file for the Batch API.")
    submit_parser.add_argument(
        "--description",
        required=True,
        help="Human-readable description stored in batch metadata.",
    )
    submit_parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra metadata pairs to attach to the batch.",
    )
    submit_parser.add_argument(
        "--completion-window",
        default=DEFAULT_COMPLETION_WINDOW,
        help=f"Batch completion window (default: {DEFAULT_COMPLETION_WINDOW}).",
    )
    submit_parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write the submission response JSON.",
    )

    status_parser = subparsers.add_parser("status", help="Fetch the current batch status.")
    status_parser.add_argument("batch_id", help="OpenAI batch ID.")

    download_parser = subparsers.add_parser(
        "download", help="Download the output JSONL for a completed batch."
    )
    download_parser.add_argument("batch_id", help="OpenAI batch ID.")
    download_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the downloaded output JSONL.",
    )

    return parser


def _parse_metadata(items: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid metadata item: {item!r}. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        metadata[key] = value
    return metadata


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "submit":
        summary = summarize_jsonl(args.jsonl)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        response = submit_batch_file(
            args.jsonl,
            description=args.description,
            metadata=_parse_metadata(args.metadata),
            completion_window=args.completion_window,
        )
        print(json.dumps(response, indent=2, ensure_ascii=False))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(response, indent=2, ensure_ascii=False) + "\n")
    elif args.command == "status":
        print(json.dumps(retrieve_batch(args.batch_id), indent=2, ensure_ascii=False))
    elif args.command == "download":
        print(json.dumps(download_batch_output(args.batch_id, args.out), indent=2, ensure_ascii=False))
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()