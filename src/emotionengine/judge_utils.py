"""OpenAI-based affective judge utilities shared across analysis notebooks."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class JudgeConfig:
    """Configuration for the affective judge.

    Parameters
    ----------
    judge_attrs:
        Ordered list of attribute names to score.
    attr_ranges:
        Dict mapping each attribute name to a (lo, hi) float tuple.
    judge_system:
        System prompt passed to the model.
    judge_user_tmpl:
        User prompt template; must contain a ``{text}`` placeholder.
    judge_schema:
        OpenAI ``response_format`` JSON-schema dict.
    cache_dir:
        Directory used to persist per-text JSON result files.
    judge_model:
        OpenAI model name (default: ``"gpt-4o-mini"``).
    prompt_version:
        Short string included in the cache key so schema changes invalidate the
        cache (default: ``"v1"``).
    dry_run:
        When ``True``, skip API calls and return zeros (useful for testing).
    """

    judge_attrs: list[str]
    attr_ranges: dict[str, tuple[float, float]]
    judge_system: str
    judge_user_tmpl: str
    judge_schema: dict[str, Any]
    cache_dir: Path
    judge_model: str = "gpt-4o-mini"
    prompt_version: str = "v1"
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_clients: dict[str, Any] = {}


def _get_client(api_key: str | None = None):
    """Return a cached ``openai.OpenAI`` client."""
    import os
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if key not in _clients:
        from openai import OpenAI
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or use dry_run=True."
            )
        _clients[key] = OpenAI(api_key=key) if api_key else OpenAI()
    return _clients[key]


def _cache_key(text: str, cfg: JudgeConfig) -> str:
    h = hashlib.sha1()
    h.update(cfg.judge_model.encode())
    h.update(b"|")
    h.update(cfg.prompt_version.encode())
    h.update(b"|")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:24]


def _cache_path(text: str, cfg: JudgeConfig) -> Path:
    return cfg.cache_dir / f"{_cache_key(text, cfg)}.json"


def _clip(d: dict, cfg: JudgeConfig) -> dict:
    out: dict[str, float] = {}
    for k, (lo, hi) in cfg.attr_ranges.items():
        v = float(d.get(k, 0.0))
        out[k] = float(np.clip(v, lo, hi))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def judge_one(
    text: str,
    cfg: JudgeConfig,
    *,
    api_key: str | None = None,
) -> dict[str, float]:
    """Score a single text using the OpenAI judge.

    Results are cached on disk inside ``cfg.cache_dir``. Subsequent calls with
    the same text and config (model + prompt version) return the cached result
    without making an API call.

    Parameters
    ----------
    text:
        The text to score.
    cfg:
        A :class:`JudgeConfig` instance.
    api_key:
        Optional explicit OpenAI API key. Falls back to ``OPENAI_API_KEY``.

    Returns
    -------
    dict mapping each attribute in ``cfg.judge_attrs`` to its clipped score.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(text, cfg)

    if p.exists():
        with open(p) as f:
            return json.load(f)

    if cfg.dry_run:
        scores: dict[str, float] = {k: 0.0 for k in cfg.judge_attrs}
        with open(p, "w") as f:
            json.dump(scores, f)
        return scores

    client = _get_client(api_key)
    resp = client.chat.completions.create(
        model=cfg.judge_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": cfg.judge_system},
            {"role": "user", "content": cfg.judge_user_tmpl.format(text=text)},
        ],
        response_format={"type": "json_schema", "json_schema": cfg.judge_schema},
    )
    raw = resp.choices[0].message.content
    scores = _clip(json.loads(raw), cfg)
    with open(p, "w") as f:
        json.dump(scores, f)
    return scores


def judge_many(
    texts: list[str],
    cfg: JudgeConfig,
    *,
    progress_every: int = 25,
    api_key: str | None = None,
) -> list[dict[str, float]]:
    """Score a list of texts, printing progress every *progress_every* items."""
    out: list[dict[str, float]] = []
    n = len(texts)
    t0 = time.time()
    for i, t in enumerate(texts):
        out.append(judge_one(t, cfg, api_key=api_key))
        if (i + 1) % progress_every == 0 or i == n - 1:
            dt = time.time() - t0
            print(
                f"  judged {i + 1}/{n}"
                f"  ({dt:.1f}s, {(i + 1) / max(dt, 1e-9):.2f}/s)"
            )
    return out
