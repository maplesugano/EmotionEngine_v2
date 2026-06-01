from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

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

# Text keys extracted per triplet (order defines dimension-1 of the tensor).
TEXT_KEYS: list[str] = ["base_text", "positive_rewrite", "negative_rewrite"]

# Checkpoint filename for incremental resume
CHECKPOINT_FILENAME = "residual_stream_checkpoint.pt"


def _make_texts(rec: dict[str, Any], tokenizer: AutoTokenizer) -> list[str]:
    """Return the three forward-pass strings for one triplet.

    All three variants share the same instruction prefix so that
    ``d_pos_neg = h_pos − h_neg`` captures only the emotional contrast and
    not base-text variation.  Order matches TEXT_KEYS: [base, pos, neg].
    """
    prefix: str = make_instruction_prefix(rec["base_text"], tokenizer)
    return [
        prefix,                            # h_base  (generation prompt only)
        prefix + rec["positive_rewrite"],  # h_pos
        prefix + rec["negative_rewrite"],  # h_neg
    ]


def _save_checkpoint(
    ckpt_path: Path, completed_texts: int, vecs: list[torch.Tensor]
) -> None:
    """Atomically write a resume checkpoint (tmp-file + rename)."""
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save({"completed_texts": completed_texts, "vecs": vecs}, tmp)
    tmp.replace(ckpt_path)


def _load_checkpoint(
    ckpt_path: Path,
) -> tuple[int, list[torch.Tensor]] | None:
    """Return ``(completed_texts, accumulated_vecs)`` if a checkpoint exists."""
    if not ckpt_path.exists():
        return None
    data = torch.load(ckpt_path, weights_only=True)
    return int(data["completed_texts"]), list(data["vecs"])


# Data helpers

def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict[str, Any]],
    hook_layers: list[int],
    device: str,
    batch_size: int,
    out_dir: Path,
    max_length: int = 512,
    save_every: int = 50,
) -> torch.Tensor:
    """Extract activations for all triplets, returning shape ``(N, 3, L, D)``.

    All 3N texts are flattened into a single sequence and processed in
    mini-batches of ``batch_size``.  A checkpoint is saved every ``save_every``
    batches so that an interrupted run can be resumed by re-running the script.
    The flat result is reshaped to ``(N, 3, L, D)`` after all batches complete.
    """
    # Flatten: [base_0, pos_0, neg_0, base_1, pos_1, neg_1, ...]
    # apply_chat_template formats each triplet with the model's native special
    # tokens; positive/negative rewrites are appended as raw continuations.
    all_texts: list[str] = [
        text for rec in records for text in _make_texts(rec, tokenizer)
    ]
    total = len(all_texts)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / CHECKPOINT_FILENAME

    # Resume from a previous checkpoint if one exists
    resume = _load_checkpoint(ckpt_path)
    if resume is not None:
        start_idx, all_vecs = resume
        logger.info(
            "Checkpoint found — resuming from text %d / %d  (%d batches done)",
            start_idx, total, len(all_vecs),
        )
    else:
        start_idx, all_vecs = 0, []

    n_total_batches = (total + batch_size - 1) // batch_size
    n_done_batches  = len(all_vecs)
    new_batches     = 0
    completed_texts = start_idx

    for start in tqdm(
        range(start_idx, total, batch_size),
        desc="Extracting activations",
        unit="batch",
        initial=n_done_batches,
        total=n_total_batches,
    ):
        batch_texts = all_texts[start : start + batch_size]
        vecs = extract_batch(model, tokenizer, batch_texts, hook_layers, device, max_length)
        all_vecs.append(vecs)
        new_batches    += 1
        completed_texts = start + len(batch_texts)
        if new_batches % save_every == 0:
            _save_checkpoint(ckpt_path, completed_texts, all_vecs)

    # Always persist the final state
    if new_batches > 0 and new_batches % save_every != 0:
        _save_checkpoint(ckpt_path, completed_texts, all_vecs)

    flat = torch.cat(all_vecs, dim=0)              # (N*3, L, D)
    return flat.view(len(records), 3, len(hook_layers), -1)  # (N, 3, L, D)


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",      default="model.yaml",
                   help="Path to model.yaml")
    p.add_argument("--data",        default="dataset/rewrites/affective_rewrites.jsonl",
                   help="Path to affective_rewrites.jsonl")
    p.add_argument("--out-dir",     default="dataset/activations",
                   help="Output directory")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device string")
    p.add_argument("--batch-size",  type=int, default=8,
                   help="Number of texts per forward pass (default: 8)")
    p.add_argument("--save-every",  type=int, default=100,
                   help="Save resume checkpoint every N batches (default: 100)")
    return p.parse_args()


# Entry point

def main() -> None:
    args = parse_args()

    cfg     = load_config(args.config)
    records = load_dataset(args.data)
    logger.info("Loaded %d rewrite triplets from %s", len(records), args.data)

    hook_layers: list[int] = cfg["hook_layers"]
    logger.info("Hook layers: %s", hook_layers)

    model, tokenizer = load_model_and_tokenizer(cfg, args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #extraction
    activations = extract_dataset(
        model, tokenizer, records, hook_layers, args.device, args.batch_size,
        out_dir,
        max_length=cfg.get("max_length", 512),
        save_every=args.save_every,
    )
    # activations : (N, 3, L, D)
    logger.info("Activations tensor shape: %s", list(activations.shape))

    # compute contrast vectors
    # activations[:, 0] = h_base, [:, 1] = h_pos, [:, 2] = h_neg  → (N, L, D)
    h_base = activations[:, 0]  # (N, L, D)
    h_pos  = activations[:, 1]  # (N, L, D)
    h_neg  = activations[:, 2]  # (N, L, D)

    # Primary contrast (used for steering vector estimation)
    d_pos_neg  = h_pos - h_neg   # (N, L, D)
    # Auxiliary contrasts
    d_pos_base = h_pos - h_base  # (N, L, D)
    d_base_neg = h_base - h_neg  # (N, L, D)

    logger.info(
        "Contrast tensors computed.  d_pos_neg %s  d_pos_base %s  d_base_neg %s",
        list(d_pos_neg.shape), list(d_pos_base.shape), list(d_base_neg.shape),
    )

    pt_path   = out_dir / "residual_stream.pt"
    meta_path = out_dir / "residual_stream_meta.jsonl"

    torch.save(
        {
            "activations":    activations,    # (N, 3, L, D) float32
            "d_pos_neg":      d_pos_neg,      # h_pos − h_neg   (primary)
            "d_pos_base":     d_pos_base,     # h_pos − h_base  (auxiliary)
            "d_base_neg":     d_base_neg,     # h_base − h_neg  (auxiliary)
            "layer_indices":  hook_layers,
            "source_ids":     [r["source_id"]     for r in records],
            "text_order":     TEXT_KEYS,
            # Per-sample metadata
            "preserved_meaning_scores": [r.get("preserved_meaning_score") for r in records],
            "base_valence":             [r.get("base_valence")             for r in records],
            "base_arousal":             [r.get("base_arousal")             for r in records],
            "base_dominance":           [r.get("base_dominance")           for r in records],
            "positive_valence":         [r.get("positive_valence")         for r in records],
            "positive_arousal":         [r.get("positive_arousal")         for r in records],
            "positive_dominance":       [r.get("positive_dominance")       for r in records],
            "negative_valence":         [r.get("negative_valence")         for r in records],
            "negative_arousal":         [r.get("negative_arousal")         for r in records],
            "negative_dominance":       [r.get("negative_dominance")       for r in records],
            "split":                    [r.get("split")                    for r in records],
        },
        pt_path,
    )
    logger.info("Saved tensor to %s", pt_path)

    # Remove checkpoint now that the final artifact is safely written
    ckpt_path = out_dir / CHECKPOINT_FILENAME
    if ckpt_path.exists():
        ckpt_path.unlink()
        logger.info("Checkpoint removed.")

    # metadata sidecar (original records, for reference / joining)
    with open(meta_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved metadata to %s", meta_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
