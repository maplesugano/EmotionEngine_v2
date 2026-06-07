"""Shared constants for the EmotionEngine pipeline."""
from __future__ import annotations

# Plutchik's eight basic emotions — defines the emotion dimension throughout the pipeline.
PLUTCHIK_EMOTIONS: list[str] = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
]

INTENSITY_LEVELS: list[str] = ["low", "medium", "high"]

# OpenAI Batch API
BATCH_ENDPOINT: str = "/v1/chat/completions"
COMPLETION_WINDOW: str = "24h"
MAX_REQUESTS_PER_BATCH: int = 50_000
DEFAULT_MAX_TOKENS_PER_BATCH: int = 20_000_000  # Tier-2 queue limit
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "expired", "cancelled"})
POLL_INTERVAL_SEC: int = 60
