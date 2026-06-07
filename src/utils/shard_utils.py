"""Checkpoint helpers for memory-bounded shard extraction."""
from __future__ import annotations

import json
from pathlib import Path


def load_shard_checkpoint(ckpt_path: Path) -> set[int]:
    """Return the set of already-completed shard indices."""
    if not ckpt_path.exists():
        return set()
    return set(json.loads(ckpt_path.read_text()).get("completed", []))


def save_shard_checkpoint(ckpt_path: Path, completed: set[int]) -> None:
    ckpt_path.write_text(json.dumps({"completed": sorted(completed)}))
