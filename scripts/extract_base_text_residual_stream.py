from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
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

SHARD_CHECKPOINT_FILENAME = "base_text_shard_checkpoint.json"


# Data loading

def load_unique_base_texts(jsonl_path: str | Path) -> list[dict[str, str]]:
    """Return ordered list of {source_id, base_text} dicts — one per unique source.

    Ordering matches the source_id order written by
    ``extract_emotion_intensity_residual_stream.py`` (insertion order of the
    sorted source_id set).
    """
    jsonl_path = Path(jsonl_path)
    seen: dict[str, str] = {}  # source_id → base_text (preserves insertion order)
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["source_id"]
            if sid not in seen:
                seen[sid] = rec["base_text"]

    records = [{"source_id": sid, "base_text": bt} for sid, bt in seen.items()]
    logger.info("Loaded %d unique source texts from %s", len(records), jsonl_path)
    return records


# Shard checkpoint helpers

def _load_shard_checkpoint(ckpt_path: Path) -> set[int]:
    if not ckpt_path.exists():
        return set()
    return set(json.loads(ckpt_path.read_text()).get("completed", []))


def _save_shard_checkpoint(ckpt_path: Path, completed: set[int]) -> None:
    ckpt_path.write_text(json.dumps({"completed": sorted(completed)}))


# Extraction

def extract_base_texts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict[str, str]],
    hook_layers: list[int],
    device: str,
    batch_size: int,
    out_dir: Path,
    max_length: int = 512,
    shard_size: int = 500,
) -> list[Path]:
    """Extract base-text activations in memory-bounded shards.

    Each shard is saved as ``base_text_shard_NNNN.pt`` with shape
    ``(shard_size, L, D)``.  A JSON checkpoint tracks completed shards so
    that interrupted runs can resume.

    Returns an ordered list of shard file paths.
    """
    n_sources = len(records)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path   = out_dir / SHARD_CHECKPOINT_FILENAME
    completed   = _load_shard_checkpoint(ckpt_path)
    n_shards    = (n_sources + shard_size - 1) // shard_size
    shard_paths = [out_dir / f"base_text_shard_{i:04d}.pt" for i in range(n_shards)]

    logger.info(
        "Total: %d sources → %d shards of ≤%d each.",
        n_sources, n_shards, shard_size,
    )

    for shard_idx in range(n_shards):
        if shard_idx in completed:
            logger.info("Shard %d/%d already done, skipping.", shard_idx + 1, n_shards)
            continue

        src_start = shard_idx * shard_size
        src_end   = min(src_start + shard_size, n_sources)
        shard_recs = records[src_start:src_end]

        # Each base text is formatted as the instruction prefix only —
        # the same prefix that precedes each emotion rewrite.
        shard_texts = [
            make_instruction_prefix(rec["base_text"], tokenizer)
            for rec in shard_recs
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

        shard_out = torch.cat(shard_vecs, dim=0)   # (n_shard, L, D)
        del shard_vecs

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


# Merge shards → final artifact

def merge_shards(
    shard_paths: list[Path],
    source_ids: list[str],
    hook_layers: list[int],
    out_dir: Path,
) -> None:
    """Concatenate shard .pt files into a single .npy + info JSON."""
    npy_path  = out_dir / "base_text_residual_stream.npy"
    info_path = out_dir / "base_text_residual_stream_info.json"

    first = torch.load(shard_paths[0], weights_only=True)
    _L, _D = first.shape[-2], first.shape[-1]
    del first

    n_sources    = len(source_ids)
    shape        = (n_sources, _L, _D)
    needed_bytes = n_sources * _L * _D * 4
    avail_bytes  = psutil.virtual_memory().available

    if needed_bytes < avail_bytes * 0.75:
        logger.info(
            "Merging %d shards into single .pt (%.2f GB) …",
            len(shard_paths), needed_bytes / 1e9,
        )
        shards      = [torch.load(p, weights_only=True) for p in shard_paths]
        activations = torch.cat(shards, dim=0)   # (N, L, D)
        del shards
        np.save(str(npy_path), activations.numpy())
        logger.info("Saved .npy  → %s  shape=%s", npy_path, list(activations.shape))
        del activations
    else:
        logger.info(
            "Tensor (%.2f GB) exceeds 75%% of available RAM. Streaming via memmap …",
            needed_bytes / 1e9,
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

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "layer_indices": hook_layers,
                "source_ids":    source_ids,
                "shape":         list(shape),
                "dtype":         "float32",
                "description":   (
                    "Last-token residual stream of the instruction prefix "
                    "(base text only, no emotion rewrite appended). "
                    "Shape: (N, L, D)."
                ),
            },
            f, indent=2, ensure_ascii=False,
        )
    logger.info("Saved info  → %s", info_path)
    logger.info(
        "Load with:\n"
        "  import json, numpy as np\n"
        "  info = json.load(open('%s'))\n"
        "  act  = np.load('%s')  # (N, L, D) float32",
        info_path, npy_path,
    )


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default="model.yaml",
        help="Path to model.yaml",
    )
    p.add_argument(
        "--data", default="dataset/emotion_rewrites/emotion_rewrites.jsonl",
        help="Path to emotion_rewrites.jsonl",
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
        "--shard-size", type=int, default=500,
        help="Number of sources per memory shard (default: 500)",
    )
    return p.parse_args()


# Entry point

def main() -> None:
    args = parse_args()

    cfg     = load_config(args.config)
    records = load_unique_base_texts(args.data)

    hook_layers: list[int] = cfg["hook_layers"]
    logger.info("Hook layers: %s", hook_layers)

    model, tokenizer = load_model_and_tokenizer(cfg, args.device)

    out_dir = Path(args.out_dir)

    shard_paths = extract_base_texts(
        model, tokenizer, records, hook_layers, args.device, args.batch_size,
        out_dir,
        max_length=cfg.get("max_length", 512),
        shard_size=args.shard_size,
    )

    logger.info("Unloading model …")
    del model
    gc.collect()
    if "cuda" in args.device:
        torch.cuda.empty_cache()

    source_ids = [rec["source_id"] for rec in records]
    merge_shards(shard_paths, source_ids, hook_layers, out_dir)

    # Clean up shard files
    ckpt_path = out_dir / SHARD_CHECKPOINT_FILENAME
    for p in shard_paths:
        if p.exists():
            p.unlink()
    if ckpt_path.exists():
        ckpt_path.unlink()
    logger.info("Shard files cleaned up.")


if __name__ == "__main__":
    main()
