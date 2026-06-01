"""Merge two emotion-intensity activation datasets.

Concatenates the (N, 8, 3, L, D) activation tensors from two directories
along the N (source) dimension and appends the corresponding meta.jsonl
records, writing the combined result back to --base-dir.

When the combined tensor fits in available RAM (75 % threshold) it is saved
as the standard ``emotion_intensity_residual_stream.pt``.  For larger
datasets the merge streams shard-by-shard via numpy memmap, producing
``emotion_intensity_residual_stream.npy`` + ``…_info.json`` instead.

Load the .npy output in downstream code with::

    import json, numpy as np, torch
    info = json.load(open("emotion_intensity_residual_stream_info.json"))
    act  = np.memmap("emotion_intensity_residual_stream.npy",
                     dtype=info["dtype"], mode="r", shape=tuple(info["shape"]))

Typical usage
-------------
# 1. Extract new activations to a staging directory:
python extract_emotion_intensity_residual_stream.py \\
    --data   dataset/emotion_rewrite_batch_results \\
    --out-dir activation_new

# 2. Merge staging into the existing activation directory:
python merge_emotion_intensity_activations.py \\
    --base-dir  activation \\
    --new-dir   activation_new
"""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PT_FILENAME   = "emotion_intensity_residual_stream.pt"
NPY_FILENAME  = "emotion_intensity_residual_stream.npy"
INFO_FILENAME = "emotion_intensity_residual_stream_info.json"
META_FILENAME = "emotion_intensity_residual_stream_meta.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_new_data(new_dir: Path) -> tuple[np.ndarray | torch.Tensor, list[str], dict[str, Any]]:
    """Load activation tensor and metadata from new_dir.

    Supports both .pt (standard) and .npy + _info.json (large-dataset) formats.
    Returns (activations, source_ids, shared_meta).
    ``activations`` may be a numpy memmap (zero-copy) or a torch Tensor.
    """
    new_pt  = new_dir / PT_FILENAME
    new_npy = new_dir / NPY_FILENAME

    if new_pt.exists():
        logger.info("Loading new data from %s …", new_pt)
        data = torch.load(new_pt, weights_only=True)
        act  = data["activations"]
        meta = {k: data[k] for k in ("layer_indices", "emotion_order", "intensity_order")}
        return act, list(data["source_ids"]), meta

    if new_npy.exists():
        info_path = new_dir / INFO_FILENAME
        if not info_path.exists():
            raise FileNotFoundError(
                f"{new_npy} exists but {info_path} is missing. "
                "Re-run extraction to regenerate both files."
            )
        logger.info("Loading new data from %s (memmap) …", new_npy)
        with open(info_path) as f:
            info = json.load(f)
        # The file is a raw binary written by np.memmap (no .npy header).
        # Must load with np.memmap (not np.load) to avoid pickle error.
        shape = tuple(info["shape"])
        dtype = info.get("dtype", "float32")
        act = np.memmap(str(new_npy), dtype=dtype, mode="r", shape=shape)
        meta = {k: info[k] for k in ("layer_indices", "emotion_order", "intensity_order")}
        return act, list(info["source_ids"]), meta

    raise FileNotFoundError(
        f"No activation file found in {new_dir}. "
        f"Expected {new_pt} or {new_npy}."
    )


# Rows written per chunk when streaming new data to memmap.
# 200 rows × (8×3×6×4096×4 bytes) ≈ 450 MB per chunk.
_COPY_CHUNK_ROWS = 200


def _streaming_merge(
    base_pt:      Path,
    new_act:      np.ndarray | torch.Tensor,
    combined_ids: list[str],
    shared_meta:  dict[str, Any],
    shape_suffix: tuple[int, ...],   # (8, 3, L, D)
) -> None:
    """Write the merged tensor, choosing .pt or .npy based on available RAM.

    Accepts ``base_pt`` (path) rather than a pre-loaded tensor so the base
    data can be loaded, written, and freed *before* the large new-data copy
    begins.  This keeps peak RAM to roughly max(base_size, chunk_size) rather
    than base_size + new_size.
    """
    n_combined   = len(combined_ids)
    full_shape   = (n_combined,) + tuple(shape_suffix)
    needed_bytes = int(np.prod(full_shape)) * 4   # float32
    avail_bytes  = psutil.virtual_memory().available

    out_meta = {
        **shared_meta,
        "source_ids": combined_ids,
    }

    if needed_bytes < avail_bytes * 0.75:
        # ---- In-RAM merge → .pt ----
        logger.info(
            "Combined tensor (%.1f GB) fits in RAM. Merging to .pt …",
            needed_bytes / 1e9,
        )
        _base = torch.load(base_pt, weights_only=True)
        base_t = _base["activations"]
        del _base
        if isinstance(new_act, np.ndarray):
            new_t = torch.from_numpy(np.array(new_act))
        else:
            new_t = new_act
        combined = torch.cat([base_t, new_t], dim=0)
        del base_t, new_t
        tmp = base_pt.with_suffix(".tmp")
        torch.save({"activations": combined, **out_meta}, tmp)
        tmp.replace(base_pt)
        logger.info("Saved .pt → %s  shape=%s", base_pt, list(combined.shape))

    else:
        # ---- Streaming merge via numpy memmap → .npy ----
        npy_path  = base_pt.with_name(NPY_FILENAME)
        info_path = base_pt.with_name(INFO_FILENAME)
        logger.info(
            "Combined tensor (%.1f GB) > 75 %% of available RAM (%.1f GB). "
            "Streaming to .npy via memmap …",
            needed_bytes / 1e9, avail_bytes / 1e9,
        )
        mmap = np.memmap(str(npy_path), dtype=np.float32, mode="w+", shape=full_shape)

        # --- Write base: load tensor, copy, free immediately ---
        # base_pt tensor is ~8 GB; freeing it before the 45 GB new-data copy
        # ensures peak RAM stays bounded.
        logger.info("Writing base activations …")
        _base = torch.load(base_pt, weights_only=True)
        base_np = _base["activations"].numpy()   # zero-copy view
        n_base  = base_np.shape[0]
        for s in range(0, n_base, _COPY_CHUNK_ROWS):
            e = min(s + _COPY_CHUNK_ROWS, n_base)
            mmap[s:e] = base_np[s:e]
        mmap.flush()
        del _base, base_np
        gc.collect()
        logger.info("Base written (%d rows). RAM freed.", n_base)

        # --- Write new data in chunks so OS page cache stays bounded ---
        # Even between two memmaps, a single slice assignment can pull the
        # entire source into RAM.  Chunked writes cap it to ~450 MB per step.
        if isinstance(new_act, torch.Tensor):
            new_act = new_act.numpy()
        n_new = new_act.shape[0]
        for s in tqdm(range(0, n_new, _COPY_CHUNK_ROWS), desc="Writing new data", unit="chunk"):
            e = min(s + _COPY_CHUNK_ROWS, n_new)
            mmap[n_base + s : n_base + e] = new_act[s:e]
            mmap.flush()
        del new_act

        del mmap
        gc.collect()

        with open(info_path, "w") as f:
            json.dump(
                {**out_meta, "shape": list(full_shape), "dtype": "float32"},
                f, indent=2, ensure_ascii=False,
            )
        logger.info("Saved .npy  → %s", npy_path)
        logger.info("Saved info  → %s", info_path)
        logger.info(
            "Load in downstream code with:\n"
            "  import json, numpy as np\n"
            "  info = json.load(open('%s'))\n"
            "  act  = np.memmap('%s', dtype=info['dtype'], mode='r', shape=tuple(info['shape']))",
            info_path, npy_path,
        )
        if base_pt.exists():
            logger.warning(
                "The original %s is superseded by the new .npy. "
                "You may delete it to reclaim disk space.",
                base_pt,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-dir", default="activation",
        help="Directory containing the existing .pt and .jsonl to extend.",
    )
    p.add_argument(
        "--new-dir", default="activation_new",
        help="Directory containing the newly extracted .pt/.npy and .jsonl.",
    )
    p.add_argument(
        "--allow-overlap", action="store_true",
        help="If set, source_ids that already exist in --base-dir are skipped "
             "rather than raising an error.",
    )
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    base_dir = Path(args.base_dir)
    new_dir  = Path(args.new_dir)

    base_pt   = base_dir / PT_FILENAME
    base_meta = base_dir / META_FILENAME
    new_meta  = new_dir  / META_FILENAME

    for p in (base_pt, base_meta, new_meta):
        if not p.exists():
            raise FileNotFoundError(p)

    # ------------------------------------------------------------------
    # Load base metadata only — extract as plain Python objects then
    # immediately free the tensor so it is NOT held in RAM during the
    # large new-data copy.  _streaming_merge reloads base_pt internally.
    # ------------------------------------------------------------------
    logger.info("Loading base metadata: %s", base_pt)
    _base = torch.load(base_pt, weights_only=True)
    base_layer_indices   = list(_base["layer_indices"])
    base_emotion_order   = list(_base["emotion_order"])
    base_intensity_order = list(_base["intensity_order"])
    base_source_ids      = list(_base["source_ids"])
    base_shape_suffix    = tuple(_base["activations"].shape[1:])   # (8, 3, L, D)
    del _base
    gc.collect()
    logger.info("Base: N=%d  shape[1:]=%s", len(base_source_ids), list(base_shape_suffix))

    # ------------------------------------------------------------------
    # Load new (supports .pt or .npy memmap — zero RAM for .npy)
    # ------------------------------------------------------------------
    new_act, new_ids, new_meta_info = _load_new_data(new_dir)

    # ------------------------------------------------------------------
    # Compatibility checks
    # ------------------------------------------------------------------
    for key, base_val in [
        ("layer_indices",   base_layer_indices),
        ("emotion_order",   base_emotion_order),
        ("intensity_order", base_intensity_order),
    ]:
        new_val = new_meta_info[key]
        if base_val != new_val:
            raise ValueError(
                f"Mismatch in '{key}':\n  base={base_val}\n  new={new_val}"
            )

    new_shape_suffix = tuple(new_act.shape[1:])
    if base_shape_suffix != new_shape_suffix:
        raise ValueError(
            f"Activation shapes incompatible beyond N dimension: "
            f"base=(N, {list(base_shape_suffix)}), new=(N, {list(new_shape_suffix)})"
        )

    # ------------------------------------------------------------------
    # Overlap check
    # ------------------------------------------------------------------
    existing_ids: set[str] = set(base_source_ids)
    overlap = [sid for sid in new_ids if sid in existing_ids]
    if overlap:
        if not args.allow_overlap:
            raise ValueError(
                f"{len(overlap)} source_ids already exist in base. "
                "Re-run with --allow-overlap to skip them."
            )
        logger.warning(
            "Skipping %d overlapping source_ids (--allow-overlap set).",
            len(overlap),
        )
        overlap_set  = set(overlap)
        keep_indices = [i for i, sid in enumerate(new_ids) if sid not in overlap_set]
        new_act = new_act[keep_indices]
        new_ids = [new_ids[i] for i in keep_indices]

    logger.info(
        "Merging: base N=%d  +  new N=%d  →  combined N=%d",
        len(base_source_ids), len(new_ids),
        len(base_source_ids) + len(new_ids),
    )

    combined_ids = base_source_ids + new_ids
    shared_meta  = {
        "layer_indices":   base_layer_indices,
        "emotion_order":   base_emotion_order,
        "intensity_order": base_intensity_order,
    }

    # ------------------------------------------------------------------
    # Merge activations (RAM-aware: .pt or .npy)
    # ------------------------------------------------------------------
    _streaming_merge(base_pt, new_act, combined_ids, shared_meta, base_shape_suffix)

    # ------------------------------------------------------------------
    # Merge meta.jsonl
    # ------------------------------------------------------------------
    skip_ids: set[str] = set(overlap) if overlap else set()
    logger.info("Reading new meta: %s", new_meta)
    new_meta_lines: list[str] = []
    with open(new_meta) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skip_ids:
                rec = json.loads(line)
                if rec.get("source_id") in skip_ids:
                    continue
            new_meta_lines.append(line)

    logger.info("Appending %d meta lines to %s …", len(new_meta_lines), base_meta)
    with open(base_meta, "a") as f:
        for line in new_meta_lines:
            f.write(line + "\n")
    logger.info("Meta updated.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
