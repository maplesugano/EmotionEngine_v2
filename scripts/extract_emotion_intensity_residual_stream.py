from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import gc

import numpy as np
import psutil
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotionengine.model_utils import extract_batch, load_config, load_model_and_tokenizer
from emotionengine.text_utils import make_instruction_prefix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Plutchik's 8 basic emotions — defines dimension-1 of the output tensor.
EMOTIONS: list[str] = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
]

# Three intensity levels — defines dimension-2 of the output tensor.
INTENSITIES: list[str] = ["low", "medium", "high"]

# Checkpoint filename for incremental resume (JSON; tracks completed shards).
SHARD_CHECKPOINT_FILENAME = "shard_checkpoint.json"


# Data loading

def _parse_batch_response(line: str) -> tuple[str, str, dict[str, Any]] | None:
    """Parse one line of a returned OpenAI batch output file.

    Returns ``(source_id, intensity, content_dict)`` or ``None`` on error.
    """
    try:
        record = json.loads(line.strip())
    except json.JSONDecodeError:
        return None
    if record.get("error") is not None:
        return None

    custom_id: str = record.get("custom_id", "")
    # custom_id format: "{source_id}__intensity_{level}"
    if "__intensity_" not in custom_id:
        return None
    source_id, intensity = custom_id.rsplit("__intensity_", 1)

    try:
        raw_content: str = (
            record["response"]["body"]["choices"][0]["message"]["content"]
        )
        content_dict: dict[str, Any] = json.loads(raw_content)
    except (KeyError, IndexError, json.JSONDecodeError):
        return None

    return source_id, intensity, content_dict


def load_dataset(data: str | Path) -> list[dict[str, Any]]:
    """Load emotion-rewrite records from a flat ``.jsonl`` file or directory.

    Each row must have fields:
      source_id, base_text, intensity_level,
      {emotion}_rewrite × 8, meaning_preserved_score
    (produced by ``build_emotion_rewrite_dataset.py``).

    Returns a flat list of records, one per (source_id, intensity_level).
    """
    data = Path(data)
    if data.is_dir():
        files = sorted(data.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in {data}")
    else:
        files = [data]

    # source_id → intensity → row dict
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source_id = raw.get("source_id", "")
                intensity = raw.get("intensity_level", "")
                if not source_id or not intensity:
                    continue
                grouped[source_id][intensity] = raw

    # Keep only source_ids that have all three intensity levels.
    complete: list[dict[str, Any]] = []
    n_incomplete = 0
    for source_id, intensity_map in sorted(grouped.items()):
        missing = [iv for iv in INTENSITIES if iv not in intensity_map]
        if missing:
            n_incomplete += 1
            continue
        for intensity in INTENSITIES:
            c = intensity_map[intensity]
            rec: dict[str, Any] = {
                "source_id":               source_id,
                "base_text":               c.get("base_text", ""),
                "intensity_level":         intensity,
                "meaning_preserved_score": c.get("meaning_preserved_score"),
            }
            for emotion in EMOTIONS:
                rewrite = c.get(f"{emotion}_rewrite", "")
                if not rewrite.strip():
                    logger.warning(
                        "Empty %s_rewrite for source_id=%r intensity=%s — skipping source.",
                        emotion, source_id, intensity,
                    )
                    n_incomplete += 1
                    complete = [r for r in complete if r["source_id"] != source_id]
                    break
                rec[f"{emotion}_rewrite"] = rewrite
            else:
                complete.append(rec)
                continue
            break  # skip remaining intensities for this source

    n_complete = len(complete) // len(INTENSITIES)
    logger.info(
        "Loaded %d complete source texts (%d records).  "
        "Skipped %d incomplete.",
        n_complete, len(complete), n_incomplete,
    )
    return complete


# Text formatting

def _make_texts(
    rec: dict[str, Any],
    tokenizer: AutoTokenizer,
) -> list[str]:
    """Return 8 forward-pass strings (one per emotion) for one intensity record.

    The instruction prefix is shared across all emotions so that
    ``h(emotion_A) − h(emotion_B)`` captures purely affective contrast,
    not base-text variation.  Order matches EMOTIONS.
    """
    prefix: str = make_instruction_prefix(rec["base_text"], tokenizer)
    return [prefix + rec[f"{emotion}_rewrite"] for emotion in EMOTIONS]


# Shard checkpoint helpers

def _load_shard_checkpoint(ckpt_path: Path) -> set[int]:
    """Return the set of already-completed shard indices."""
    if not ckpt_path.exists():
        return set()
    return set(json.loads(ckpt_path.read_text()).get("completed", []))


def _save_shard_checkpoint(ckpt_path: Path, completed: set[int]) -> None:
    ckpt_path.write_text(json.dumps({"completed": sorted(completed)}))


# Config / model loading

# load_config, load_model_and_tokenizer, and extract_batch are imported from
# emotionengine.model_utils above.


# Full dataset extraction

def extract_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict[str, Any]],
    hook_layers: list[int],
    device: str,
    batch_size: int,
    out_dir: Path,
    max_length: int = 512,
    save_every: int = 50,  # kept for API compatibility; no longer used
    shard_size: int = 500,
) -> list[Path]:
    """Extract activations in memory-bounded shards.

    Processes ``shard_size`` source texts at a time, saving each completed
    shard to ``shard_NNNN.pt`` (shape ``(shard_size, 8, 3, L, D)``).
    Progress is recorded in ``shard_checkpoint.json`` so interrupted runs
    can resume from the last completed shard.

    Unlike the previous implementation, ``all_vecs`` never grows beyond one
    shard, bounding peak CPU-RAM to roughly
    ``shard_size × 3 × 8 × L × D × 4 bytes``  (≈ 1 GB at shard_size=500).

    Returns an ordered list of shard file paths.
    """
    n_sources = len(records) // len(INTENSITIES)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path   = out_dir / SHARD_CHECKPOINT_FILENAME
    completed   = _load_shard_checkpoint(ckpt_path)
    n_shards    = (n_sources + shard_size - 1) // shard_size
    shard_paths = [out_dir / f"shard_{i:04d}.pt" for i in range(n_shards)]

    logger.info(
        "Total: %d sources → %d shards of ≤%d sources each.",
        n_sources, n_shards, shard_size,
    )

    for shard_idx in range(n_shards):
        if shard_idx in completed:
            logger.info("Shard %d/%d already done, skipping.", shard_idx + 1, n_shards)
            continue

        src_start = shard_idx * shard_size
        src_end   = min(src_start + shard_size, n_sources)
        n_shard   = src_end - src_start

        # Each source occupies len(INTENSITIES) consecutive records.
        shard_records = records[src_start * len(INTENSITIES) : src_end * len(INTENSITIES)]
        shard_texts: list[str] = [
            t for rec in shard_records for t in _make_texts(rec, tokenizer)
        ]

        logger.info(
            "Shard %d/%d: sources %d–%d  (%d forward passes)",
            shard_idx + 1, n_shards, src_start, src_end - 1, len(shard_texts),
        )

        shard_vecs: list[torch.Tensor] = []
        for start in tqdm(
            range(0, len(shard_texts), batch_size),
            desc=f"Shard {shard_idx + 1}/{n_shards}",
            unit="batch",
        ):
            shard_vecs.append(extract_batch(
                model, tokenizer,
                shard_texts[start : start + batch_size],
                hook_layers, device, max_length,
            ))

        flat      = torch.cat(shard_vecs, dim=0)   # (n_shard*3*8, L, D)
        del shard_vecs
        L, D      = len(hook_layers), flat.shape[-1]
        shard_out = flat.view(n_shard, len(INTENSITIES), len(EMOTIONS), L, D)
        shard_out = shard_out.permute(0, 2, 1, 3, 4).contiguous()  # (n_shard, 8, 3, L, D)
        del flat

        torch.save(shard_out, shard_paths[shard_idx])
        logger.info(
            "Shard %d saved: shape=%s  dtype=%s",
            shard_idx, list(shard_out.shape), shard_out.dtype,
        )
        del shard_out
        gc.collect()
        if "cuda" in device:
            torch.cuda.empty_cache()

        completed.add(shard_idx)
        _save_shard_checkpoint(ckpt_path, completed)

    return shard_paths


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default="model.yaml",
        help="Path to model.yaml",
    )
    p.add_argument(
        "--data", default="dataset/emotion_rewrites/emotion_rewrites.jsonl",
        help="Flat .jsonl file or directory of .jsonl files",
    )
    p.add_argument(
        "--out-dir", default="activation/emotion_rewrites",
        help="Output directory (default: activation/emotion_rewrites)",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string",
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of texts per forward pass (default: 8)",
    )
    p.add_argument(
        "--save-every", type=int, default=100,
        help="[deprecated, no longer used with shard-based extraction]",
    )
    p.add_argument(
        "--shard-size", type=int, default=500,
        help="Number of source texts per memory shard (default: 500). "
             "Lower values use less CPU RAM at the cost of more shard files.",
    )
    return p.parse_args()


# Entry point

def main() -> None:
    args = parse_args()

    cfg     = load_config(args.config)
    records = load_dataset(args.data)

    hook_layers: list[int] = cfg["hook_layers"]
    logger.info("Hook layers: %s", hook_layers)

    model, tokenizer = load_model_and_tokenizer(cfg, args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Extract (shard-based, bounded CPU-RAM) ----------------------------
    shard_paths = extract_dataset(
        model, tokenizer, records, hook_layers, args.device, args.batch_size,
        out_dir,
        max_length=cfg.get("max_length", 512),
        shard_size=args.shard_size,
    )

    # Unload model before the merge step to free GPU + CPU memory.
    logger.info("Unloading model to free memory for merge step …")
    del model
    gc.collect()
    if "cuda" in args.device:
        torch.cuda.empty_cache()

    # ---- Collect ordered source IDs ----------------------------------------
    seen: set[str] = set()
    source_ids: list[str] = []
    duplicated: list[str] = []
    for rec in records:
        sid = rec["source_id"]
        if sid not in seen:
            seen.add(sid)
            source_ids.append(sid)
        else:
            duplicated.append(sid)
    if duplicated:
        raise ValueError(
            f"Duplicated source_ids detected in dataset: {sorted(set(duplicated))[:10]} …\n"
            "Each source_id must appear exactly once per intensity level."
        )
    # Sort source_ids alphabetically so order matches neutral-paraphrase extraction.
    source_ids = sorted(source_ids)
    # Reorder records to match sorted source_ids
    sid_to_int_recs: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        sid_to_int_recs.setdefault(rec["source_id"], []).append(rec)
    records = [rec for sid in source_ids for rec in sid_to_int_recs[sid]]

    n_sources   = len(source_ids)
    first_shard = torch.load(shard_paths[0], weights_only=True)
    _L, _D      = first_shard.shape[-2], first_shard.shape[-1]
    del first_shard

    shared_meta: dict[str, Any] = {
        "layer_indices":   hook_layers,
        "emotion_order":   EMOTIONS,
        "intensity_order": INTENSITIES,
        "source_ids":      source_ids,
    }

    npy_path  = out_dir / "emotion_intensity_residual_stream.npy"
    info_path = out_dir / "emotion_intensity_residual_stream_info.json"
    meta_path = out_dir / "emotion_intensity_residual_stream_meta.jsonl"

    # ---- Merge shards → final .npy artifact --------------------------------
    # Always output .npy (never .pt) so downstream code has a single path.
    # Stream via memmap when data exceeds 75% of available RAM.
    shape        = (n_sources, len(EMOTIONS), len(INTENSITIES), _L, _D)
    needed_bytes = n_sources * len(EMOTIONS) * len(INTENSITIES) * _L * _D * 4
    avail_bytes  = psutil.virtual_memory().available

    if needed_bytes < avail_bytes * 0.75:
        logger.info(
            "Merging %d shards into RAM then saving .npy (%.1f GB) …",
            len(shard_paths), needed_bytes / 1e9,
        )
        shards      = [torch.load(p, weights_only=True) for p in shard_paths]
        activations = torch.cat(shards, dim=0)      # (N, 8, 3, L, D)
        del shards
        np.save(str(npy_path), activations.numpy())
        logger.info("Saved .npy  → %s  shape=%s", npy_path, list(activations.shape))
        del activations
        gc.collect()
    else:
        logger.info(
            "Tensor (%.1f GB) exceeds 75 %% of available RAM (%.1f GB). "
            "Streaming shards to .npy via memmap …",
            needed_bytes / 1e9, avail_bytes / 1e9,
        )
        mmap = np.memmap(str(npy_path), dtype=np.float32, mode="w+", shape=shape)
        idx  = 0
        for p in tqdm(shard_paths, desc="Merging shards", unit="shard"):
            shard = torch.load(p, weights_only=True)
            n     = shard.shape[0]
            mmap[idx : idx + n] = shard.numpy()
            del shard
            idx += n
        mmap.flush()
        del mmap
        gc.collect()
        logger.info("Saved .npy  → %s", npy_path)

    with open(info_path, "w") as f:
        json.dump(
            {
                **shared_meta,
                "shape": list(shape),
                "dtype": "float32",
                "description": (
                    "Last-token residual stream of emotion-intensity rewrites. "
                    "Shape: (N, E, I, L, D) = (n_sources, 8 emotions, 3 intensities, "
                    "n_hook_layers, hidden_dim). "
                    "Subtract neutral_paraphrase_residual_stream[n, l] to obtain CAA deltas."
                ),
            },
            f, indent=2, ensure_ascii=False,
        )
    logger.info("Saved info  → %s", info_path)
    logger.info(
        "Load in downstream code with:\n"
        "  import json, numpy as np\n"
        "  info = json.load(open('%s'))\n"
        "  act  = np.memmap('%s', dtype=info['dtype'], mode='r', shape=tuple(info['shape']))",
        info_path, npy_path,
    )

    # ---- Clean up shards ---------------------------------------------------
    ckpt_path = out_dir / SHARD_CHECKPOINT_FILENAME
    for p in shard_paths:
        if p.exists():
            p.unlink()
    if ckpt_path.exists():
        ckpt_path.unlink()
    logger.info("Shard files removed.")

    # ---- Metadata sidecar — one record per (source_id, intensity) ---------
    with open(meta_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved metadata to %s", meta_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
