from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AffectiveAxisLiteral = Literal[
    "warmth",
    "emotional_intensity",
    "vulnerability",
    "restraint",
    "interpersonal_distance",
    "addressivity",
    "detachment",
]

AFFECTIVE_AXIS_VALUES: list[str] = [
    "warmth",
    "emotional_intensity",
    "vulnerability",
    "restraint",
    "interpersonal_distance",
    "addressivity",
    "detachment",
]

# JSON schema submitted to OpenAI for structured output.
REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "base_text": {"type": "string"},
        "positive_rewrite": {"type": "string"},
        "negative_rewrite": {"type": "string"},
        "preserved_meaning_score": {"type": "number"},
        "affective_axis": {
            "type": "string",
            "enum": AFFECTIVE_AXIS_VALUES,
        },
        "positive_description": {"type": "string"},
        "negative_description": {"type": "string"},
        "rejection_reason": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
        "base_valence": {"type": "number"},
        "base_arousal": {"type": "number"},
        "base_dominance": {"type": "number"},
        "positive_valence": {"type": "number"},
        "positive_arousal": {"type": "number"},
        "positive_dominance": {"type": "number"},
        "negative_valence": {"type": "number"},
        "negative_arousal": {"type": "number"},
        "negative_dominance": {"type": "number"},
    },
    "required": [
        "base_text",
        "positive_rewrite",
        "negative_rewrite",
        "preserved_meaning_score",
        "affective_axis",
        "positive_description",
        "negative_description",
        "rejection_reason",
        "base_valence",
        "base_arousal",
        "base_dominance",
        "positive_valence",
        "positive_arousal",
        "positive_dominance",
        "negative_valence",
        "negative_arousal",
        "negative_dominance",
    ],
    "additionalProperties": False,
}

CSV_FIELDNAMES: list[str] = [
    "source_dataset",
    "source_split",
    "source_id",
    "base_text",
    "positive_rewrite",
    "negative_rewrite",
    "affective_axis",
    "preserved_meaning_score",
    "positive_description",
    "negative_description",
    "base_valence",
    "base_arousal",
    "base_dominance",
    "positive_valence",
    "positive_arousal",
    "positive_dominance",
    "negative_valence",
    "negative_arousal",
    "negative_dominance",
]

SYSTEM_PROMPT: str = """\
You are an expert in affective linguistics and conversational pragmatics.

Your task: given a conversational utterance, produce a contrastive rewrite triple.

CORE CONSTRAINT
Preserve the proposition (what is asserted), the described event, and the speaker's
social intention. Change ONLY the emotional temperature along exactly one affective
axis.
This is NOT sentiment flipping. Joy stays joy. Anger stays anger.
The difference is how explicitly, warmly, vulnerably, or intensely the emotion
is expressed.

POSITIVE REWRITE
Same proposition, same event, same speaker intention.
Increase the emotional intensity to be more dramatic, explicit, warm, vulnerable, or intense.
Do NOT change the emotion category (sadness→joy, fear→anger, etc. are forbidden).

NEGATIVE REWRITE
Same proposition, same event, same speaker intention.
Reduce emotional intensity to be more restrained, detached, or formal.
Do NOT change the emotion category (joy→sadness, anger→fear, etc. are forbidden).

EXAMPLES
Base      : "I'm glad you came."
Positive  : "I'm really, genuinely happy you're here."
Negative  : "Thank you for coming."

Base      : "I was worried when you didn't reply."
Positive  : "I was honestly really worried when you didn't reply."
Negative  : "I noticed you hadn't replied and felt concerned."

VAD LABELS (Valence / Arousal / Dominance)
For each of the three texts (base_text, positive_rewrite, negative_rewrite),
provide continuous ratings on a 0.0–1.0 scale:
  Valence   : 0.0 = maximally negative affect, 1.0 = maximally positive affect
  Arousal   : 0.0 = fully calm / deactivated, 1.0 = highly excited / activated
  Dominance : 0.0 = fully powerless / submissive, 1.0 = fully dominant / in control
Use precise decimals (e.g. 0.72, not just 0, 0.5, or 1).
Expect the positive_rewrite to score higher on the modulated axis's arousal or
valence than the base, and the negative_rewrite to score lower.

OUTPUT FORMAT
Return the JSON object with all required fields.
Set rejection_reason to a non-null string ONLY if the input is:
  - offensive or inappropriate
  - grammatically incoherent / not a natural utterance
  - impossible to rewrite while preserving propositional content
Otherwise set rejection_reason to null.
"""

# Offensive-content heuristic: extended pattern covering common slurs/profanity
_OFFENSIVE_RE = re.compile(
    r"\b(fuck(?:ing|er|s)?|shit(?:ty)?|bitch(?:es)?|asshole|cunt|nigger|faggot|retard)\b",
    re.IGNORECASE,
)

MIN_WORDS: int = 8
MAX_WORDS: int = 80

_DATASET_DIR = Path(__file__).parent.parent / "dataset"


class AffectiveRewriteRecord(BaseModel):
    """Structured output from the language model for one affective rewrite triple."""

    base_text: str = Field(description="Original utterance, reproduced verbatim.")
    positive_rewrite: str = Field(
        description=(
            "Same proposition and social intent; affective dimension increased "
            "(warmer, more explicit, more vulnerable, or more intense)."
        )
    )
    negative_rewrite: str = Field(
        description=(
            "Same proposition and social intent; affective dimension reduced "
            "(more restrained, detached, or formal). Emotion category unchanged."
        )
    )
    preserved_meaning_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Estimated proportion of propositional and social content preserved "
            "across all three variants (0–1)."
        ),
    )
    affective_axis: AffectiveAxisLiteral = Field(
        description="The single affective dimension modulated across the triple."
    )
    positive_description: str = Field(
        description="One-sentence explanation of what changed in the positive rewrite."
    )
    negative_description: str = Field(
        description="One-sentence explanation of what changed in the negative rewrite."
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description=(
            "Non-null string if the input cannot be cleanly rewritten. Otherwise null."
        ),
    )
    # VAD labels (0.0–1.0) for each of the three texts
    base_valence: float = Field(
        ge=0.0, le=1.0,
        description="Valence of base_text: 0=very negative, 1=very positive.",
    )
    base_arousal: float = Field(
        ge=0.0, le=1.0,
        description="Arousal of base_text: 0=calm/deactivated, 1=excited/activated.",
    )
    base_dominance: float = Field(
        ge=0.0, le=1.0,
        description="Dominance of base_text: 0=submissive/powerless, 1=dominant/in control.",
    )
    positive_valence: float = Field(
        ge=0.0, le=1.0,
        description="Valence of positive_rewrite.",
    )
    positive_arousal: float = Field(
        ge=0.0, le=1.0,
        description="Arousal of positive_rewrite.",
    )
    positive_dominance: float = Field(
        ge=0.0, le=1.0,
        description="Dominance of positive_rewrite.",
    )
    negative_valence: float = Field(
        ge=0.0, le=1.0,
        description="Valence of negative_rewrite.",
    )
    negative_arousal: float = Field(
        ge=0.0, le=1.0,
        description="Arousal of negative_rewrite.",
    )
    negative_dominance: float = Field(
        ge=0.0, le=1.0,
        description="Dominance of negative_rewrite.",
    )


# Text-quality filters

def word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens in *text*."""
    return len(text.split())


def is_english(text: str) -> bool:
    """Heuristic: classify *text* as English when ≥90% of characters are ASCII."""
    if not isinstance(text, str) or not text:
        return False
    return sum(c < "\x80" for c in text) / len(text) >= 0.9


def is_offensive(text: str) -> bool:
    """Return True if *text* contains obviously offensive language."""
    return bool(_OFFENSIVE_RE.search(text))


def passes_filters(text: str) -> bool:
    """Return True if *text* is a suitable candidate for affective rewriting."""
    if not isinstance(text, str) or not text.strip():
        return False
    if not is_english(text):
        return False
    wc = word_count(text)
    if not (MIN_WORDS <= wc <= MAX_WORDS):
        return False
    if is_offensive(text):
        return False
    return True


# Dataset loading

def _iter_dailydialog() -> Iterator[tuple[str, str, str]]:
    """Yield ``(source_id, utterance, split)`` tuples from local DailyDialog CSVs.

    Each CSV row contains a ``dialog`` field that is a Python list literal of
    utterance strings (one conversation per row).
    """
    for split in ("train", "validation", "test"):
        csv_path = _DATASET_DIR / "dailydialog" / f"{split}.csv"
        if not csv_path.exists():
            logger.warning("DailyDialog %s split not found: %s", split, csv_path)
            continue
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for conv_idx, row in enumerate(reader):
                try:
                    utterances: list[str] = ast.literal_eval(row["dialog"])
                except Exception:
                    continue
                for turn_idx, utterance in enumerate(utterances):
                    source_id = f"dd_{split}_{conv_idx}_{turn_idx}"
                    yield source_id, utterance.strip(), split


def _iter_empatheticdialogues() -> Iterator[tuple[str, str, str]]:
    """Yield ``(source_id, utterance, split)`` tuples from the local
    EmpatheticDialogues CSV (``emotion-emotion_69k.csv``).

    The ``Situation`` column contains the first-person emotional narrative
    written by each crowd worker — the primary utterance for rewriting.
    """
    csv_path = _DATASET_DIR / "empatheticdialog" / "emotion-emotion_69k.csv"
    if not csv_path.exists():
        logger.warning("EmpatheticDialogues file not found: %s", csv_path)
        return
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            utterance = row.get("Situation", "").strip()
            if utterance:
                source_id = f"ed_all_{idx}"
                yield source_id, utterance, "all"


def load_all_candidates(
    seed: int,
    n_samples: Optional[int] = None,
) -> list[tuple[str, str, str, str]]:
    """Load, filter, deduplicate, and sample utterances from both datasets.

    Parameters
    ----------
    seed:
        Random seed for reproducible shuffling.
    n_samples:
        Maximum number of candidates to return.  ``None`` means return all.

    Returns
    -------
    list of (source_id, utterance, source_dataset, source_split)
    """
    logger.info("Loading DailyDialog …")
    dd_candidates: list[tuple[str, str, str, str]] = [
        (sid, utt, "dailydialog", split)
        for sid, utt, split in _iter_dailydialog()
        if passes_filters(utt)
    ]
    logger.info("DailyDialog candidates after filtering: %d", len(dd_candidates))

    logger.info("Loading EmpatheticDialogues …")
    ed_candidates: list[tuple[str, str, str, str]] = [
        (sid, utt, "empatheticdialogues", split)
        for sid, utt, split in _iter_empatheticdialogues()
        if passes_filters(utt)
    ]
    logger.info(
        "EmpatheticDialogues candidates after filtering: %d", len(ed_candidates)
    )

    # Deduplicate by normalised text (case-folded, stripped)
    seen_texts: set[str] = set()
    unique: list[tuple[str, str, str, str]] = []
    for candidate in dd_candidates + ed_candidates:
        key = candidate[1].strip().lower()
        if key not in seen_texts:
            seen_texts.add(key)
            unique.append(candidate)

    logger.info(
        "Total unique candidates after dedup: %d (DD=%d, ED=%d)",
        len(unique),
        len(dd_candidates),
        len(ed_candidates),
    )

    rng = random.Random(seed)
    rng.shuffle(unique)
    return unique if n_samples is None else unique[:n_samples]


# OpenAI API helpers

class _TokenBucket:
    """Async token bucket for enforcing a global requests-per-minute cap.

    Each call to ``acquire()`` waits until the inter-request interval has
    elapsed, serialising admission via an asyncio lock.
    """

    def __init__(self, rpm: float) -> None:
        self._interval = 60.0 / rpm  # minimum seconds between requests
        self._next_allowed: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_allowed = (
                max(self._next_allowed, asyncio.get_event_loop().time())
                + self._interval
            )


def _build_messages(base_text: str) -> list[dict[str, str]]:
    """Construct the system + user message list for the API call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Produce an affective rewrite triple for the utterance below.\n\n"
                f'Utterance: "{base_text}"'
            ),
        },
    ]


def _has_responses_api(client: openai.OpenAI) -> bool:
    """Return True if the installed openai SDK exposes the Responses API."""
    return hasattr(client, "responses")


async def call_openai_async(
    client: openai.AsyncOpenAI,
    model: str,
    base_text: str,
    *,
    use_responses_api: bool,
) -> str:
    """Async version of ``call_openai`` using ``openai.AsyncOpenAI``."""
    messages = _build_messages(base_text)
    structured_format: dict[str, Any] = {
        "type": "json_schema",
        "name": "AffectiveRewriteOutput",
        "schema": REWRITE_SCHEMA,
        "strict": True,
    }
    if use_responses_api:
        response = await client.responses.create(  # type: ignore[attr-defined]
            model=model,
            input=messages,
            text={"format": structured_format},
        )
        return response.output_text  # type: ignore[attr-defined]
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            response_format={  # type: ignore[arg-type]
                "type": "json_schema",
                "json_schema": {
                    "name": "AffectiveRewriteOutput",
                    "schema": REWRITE_SCHEMA,
                    "strict": True,
                },
            },
        )
        return response.choices[0].message.content or ""


def _parse_retry_after(exc: openai.RateLimitError) -> float:
    """Extract the server-recommended wait time (seconds) from a RateLimitError.

    Reads ``retry-after`` and ``x-ratelimit-reset-tokens`` response headers.
    Falls back to 60 s when neither is present.
    """
    _DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")

    def _parse_duration(s: str) -> float:
        total = 0.0
        for m in _DURATION_RE.finditer(s):
            val, unit = float(m.group(1)), m.group(2)
            if unit == "ms":
                total += val / 1000.0
            elif unit == "s":
                total += val
            elif unit == "m":
                total += val * 60.0
            elif unit == "h":
                total += val * 3600.0
        return total if total > 0 else 0.0

    try:
        headers = exc.response.headers  # type: ignore[union-attr]
        ra = headers.get("retry-after")
        if ra:
            return float(ra)
        reset = headers.get("x-ratelimit-reset-tokens") or headers.get(
            "x-ratelimit-reset-requests"
        )
        if reset:
            parsed = _parse_duration(str(reset))
            if parsed > 0:
                return parsed
    except Exception:
        pass
    return 60.0  # conservative fallback


async def _call_with_backoff_async(
    client: openai.AsyncOpenAI,
    model: str,
    base_text: str,
    *,
    use_responses_api: bool,
    max_retries: int = 8,
) -> Optional[str]:
    """Async call with retry-after-aware exponential backoff on ``RateLimitError``."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return await call_openai_async(
                client, model, base_text, use_responses_api=use_responses_api
            )
        except openai.RateLimitError as exc:
            if attempt == max_retries - 1:
                logger.error("Rate limit: giving up after %d retries. %s", max_retries, exc)
                return None
            server_wait = _parse_retry_after(exc)
            wait = max(server_wait, delay) + random.uniform(0.0, 0.5 * max(server_wait, delay))
            logger.warning(
                "Rate limit (attempt %d/%d); backing off %.2fs (server asked %.2fs).",
                attempt + 1, max_retries, wait, server_wait,
            )
            await asyncio.sleep(wait)
            delay = min(delay * 2.0, 60.0)
        except openai.OpenAIError as exc:
            logger.error("OpenAI error (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            return None
    return None


# Output validation

def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def validate_record(record: AffectiveRewriteRecord) -> tuple[bool, str]:
    """Validate a completed rewrite record.

    Returns ``(is_valid, rejection_reason)``.  When *is_valid* is ``False``,
    *rejection_reason* explains why the record was discarded.
    """
    if record.rejection_reason is not None:
        return False, f"model_rejection: {record.rejection_reason}"

    if record.preserved_meaning_score < 0.85:
        return False, (
            f"preserved_meaning_score={record.preserved_meaning_score:.3f} < 0.85"
        )

    base = record.base_text.strip()
    pos = record.positive_rewrite.strip()
    neg = record.negative_rewrite.strip()

    if pos.lower() == base.lower():
        return False, "positive_rewrite is identical to base"
    if neg.lower() == base.lower():
        return False, "negative_rewrite is identical to base"

    sim = _jaccard(pos, neg)
    if sim > 0.85:
        return False, f"positive/negative rewrites too similar (Jaccard={sim:.3f})"

    base_wc = word_count(base)
    if base_wc > 0:
        for rewrite, label in ((pos, "positive"), (neg, "negative")):
            ratio = word_count(rewrite) / base_wc
            if not (0.5 <= ratio <= 2.0):
                return False, (
                    f"{label}_rewrite length ratio {ratio:.2f} outside [0.5, 2.0]"
                )

    return True, ""


# I/O helpers


def load_processed_ids(jsonl_path: Path) -> set[str]:
    """Return the set of ``source_id`` values already present in *jsonl_path*."""
    processed: set[str] = set()
    if not jsonl_path.exists():
        return processed
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("source_id")
                if sid:
                    processed.add(sid)
            except json.JSONDecodeError:
                continue
    return processed


def _to_output_dict(
    rewrite: AffectiveRewriteRecord,
    source_dataset: str,
    source_split: str,
    source_id: str,
) -> dict[str, Any]:
    """Merge model output with dataset metadata into a flat dict for storage."""
    return {
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_id": source_id,
        "base_text": rewrite.base_text,
        "positive_rewrite": rewrite.positive_rewrite,
        "negative_rewrite": rewrite.negative_rewrite,
        "affective_axis": rewrite.affective_axis,
        "preserved_meaning_score": rewrite.preserved_meaning_score,
        "positive_description": rewrite.positive_description,
        "negative_description": rewrite.negative_description,
        "base_valence": rewrite.base_valence,
        "base_arousal": rewrite.base_arousal,
        "base_dominance": rewrite.base_dominance,
        "positive_valence": rewrite.positive_valence,
        "positive_arousal": rewrite.positive_arousal,
        "positive_dominance": rewrite.positive_dominance,
        "negative_valence": rewrite.negative_valence,
        "negative_arousal": rewrite.negative_arousal,
        "negative_dominance": rewrite.negative_dominance,
    }


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    """Append *record* as a JSON line to *path*."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv(record: dict[str, Any], path: Path) -> None:
    """Append *record* to *path* as a CSV row, writing a header if needed."""
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)


# CLI


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build affective rewrite triples (base, positive, negative) from "
            "DailyDialog and EmpatheticDialogues via OpenAI structured outputs."
        )
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of candidate utterances to process. Omit to process all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/rewrites"),
        help="Directory for output files (default: dataset/rewrites).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model identifier (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate sampling (default: 42).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help=(
            "Maximum simultaneous in-flight API requests (async mode, default: 10). "
            "Tune with --rpm to stay within TPM limits. "
            "Optimal ≈ rpm x api_latency_s / 60."
        ),
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=None,
        help=(
            "Global requests-per-minute cap enforced by a token bucket "
            "(async mode only). Omit to rely solely on --max-concurrency."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip source_ids already present in the output JSONL (default: enabled).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode; reprocess all sampled candidates.",
    )
    return parser.parse_args()


# Per-utterance worker

_SCORE_FIELDS: tuple[str, ...] = (
    "preserved_meaning_score",
    "base_valence", "base_arousal", "base_dominance",
    "positive_valence", "positive_arousal", "positive_dominance",
    "negative_valence", "negative_arousal", "negative_dominance",
)


async def _process_one_async(
    source_id: str,
    base_text: str,
    source_dataset: str,
    source_split: str,
    *,
    client: openai.AsyncOpenAI,
    model: str,
    use_responses_api: bool,
    sem: asyncio.Semaphore,
    bucket: Optional[_TokenBucket],
) -> tuple[str, Optional[dict[str, Any]]]:
    """Call the API and validate one utterance (async).

    Acquires *sem* to cap concurrency and *bucket* to enforce RPM.
    Returns ``(source_id, out_dict)`` where *out_dict* is ``None`` on failure.
    """
    async with sem:
        if bucket is not None:
            await bucket.acquire()
        raw_json = await _call_with_backoff_async(
            client, model, base_text, use_responses_api=use_responses_api
        )

    if raw_json is None:
        return source_id, None

    try:
        data = json.loads(raw_json)
        for field in _SCORE_FIELDS:
            val = data.get(field)
            if isinstance(val, (int, float)) and val > 1.0:
                data[field] = round(val / 10.0, 4)
        record = AffectiveRewriteRecord.model_validate(data)
    except Exception as exc:
        logger.warning(
            "Parse/validation error for %s: %s | raw=%r",
            source_id, exc, raw_json[:300],
        )
        return source_id, None

    is_valid, reason = validate_record(record)
    if not is_valid:
        logger.debug("Rejected %s: %s", source_id, reason)
        return source_id, None

    return source_id, _to_output_dict(record, source_dataset, source_split, source_id)


# Mode orchestrators


async def _run_async_mode(
    todo: list[tuple[str, str, str, str]],
    *,
    api_key: str,
    model: str,
    max_concurrency: int,
    rpm: Optional[float],
    jsonl_path: Path,
    csv_path: Path,
) -> tuple[int, int]:
    """Process all candidates concurrently using ``asyncio`` + ``AsyncOpenAI``.

    Parameters
    ----------
    max_concurrency:
        Maximum number of in-flight API requests at any one time.
    rpm:
        Optional global requests-per-minute cap enforced by a token bucket.
        ``None`` means no additional rate limiting beyond *max_concurrency*.
    """
    async with openai.AsyncOpenAI(api_key=api_key) as async_client:
        use_responses_api = _has_responses_api(async_client)
        logger.info(
            "Async mode: max_concurrency=%d%s",
            max_concurrency,
            f"  rpm_cap={rpm:.0f}" if rpm is not None else "",
        )
        logger.info(
            "Tip: if you hit TPM limits, use --rpm to cap throughput "
            "(e.g., --rpm 200 for 200k TPM / ~1000 tok/req)."
        )
        sem = asyncio.Semaphore(max_concurrency)
        bucket = _TokenBucket(rpm) if rpm is not None else None
        write_lock = asyncio.Lock()
        n_written = n_rejected = 0

        tasks = [
            asyncio.create_task(
                _process_one_async(
                    sid, utt, ds, sp,
                    client=async_client,
                    model=model,
                    use_responses_api=use_responses_api,
                    sem=sem,
                    bucket=bucket,
                )
            )
            for sid, utt, ds, sp in todo
        ]

        with logging_redirect_tqdm():
            pbar = tqdm(total=len(tasks), desc="Generating rewrites", unit="utt")
            for coro in asyncio.as_completed(tasks):
                _, out_dict = await coro
                if out_dict is None:
                    n_rejected += 1
                else:
                    async with write_lock:
                        append_jsonl(out_dict, jsonl_path)
                        append_csv(out_dict, csv_path)
                    n_written += 1
                pbar.update(1)
            pbar.close()

    return n_written, n_rejected


# Main


def main() -> None:
    """Entry point: build the affective rewrite dataset."""
    load_dotenv()
    args = _parse_args()
    max_concurrency: int = args.max_concurrency

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "affective_rewrites.jsonl"
    csv_path = output_dir / "affective_rewrites.csv"

    # Validate API key early
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it before running this script."
        )

    logger.info(
        "model: %s  max_concurrency: %d%s",
        args.model,
        max_concurrency,
        f"  rpm_cap={args.rpm:.0f}" if args.rpm is not None else "",
    )

    # Resume: collect already-processed source_ids
    processed_ids: set[str] = set()
    if args.resume:
        processed_ids = load_processed_ids(jsonl_path)
        if processed_ids:
            logger.info("Resume: %d records already in %s.", len(processed_ids), jsonl_path)

    # Load candidates (all, or capped by --n-samples)
    candidates = load_all_candidates(seed=args.seed, n_samples=args.n_samples)
    logger.info(
        "Candidates to process: %d%s",
        len(candidates),
        " (all)" if args.n_samples is None else f" (capped at {args.n_samples})",
    )

    # Filter out already-processed candidates up front
    todo = [
        (sid, utt, ds, sp)
        for sid, utt, ds, sp in candidates
        if not (args.resume and sid in processed_ids)
    ]
    n_skipped = len(candidates) - len(todo)
    if n_skipped:
        logger.info("Skipping %d already-processed candidates.", n_skipped)

    # Suppress noisy httpx transport logs (HTTP Request: POST ....)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    n_written, n_rejected = asyncio.run(
        _run_async_mode(
            todo,
            api_key=api_key,
            model=args.model,
            max_concurrency=max_concurrency,
            rpm=args.rpm,
            jsonl_path=jsonl_path,
            csv_path=csv_path,
        )
    )

    logger.info(
        "Done.  written=%d  skipped(resume)=%d  rejected=%d",
        n_written,
        n_skipped,
        n_rejected,
    )
    logger.info("JSONL → %s", jsonl_path.resolve())
    logger.info("CSV  → %s", csv_path.resolve())


if __name__ == "__main__":
    main()
