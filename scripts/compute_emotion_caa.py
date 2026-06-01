from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Helpers

def unit_norm(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Return v normalised to unit length along *axis*."""
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, eps)


# Core computation

def compute_caa(
    emo_act:  np.ndarray,   # (N, E, I, L, D)
    base_act: np.ndarray,   # (N, L, D)
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean delta vectors and unit-normalised CAA directions.

    Processes sources in chunks to bound peak RAM usage.

    Returns
    -------
    delta_mean : (E, I, L, D) float64 — raw mean of (h_emo − h_base)
    caa        : (E, I, L, D) float32 — unit-normalised CAA per (emotion, intensity, layer)
    caa_pooled : (E, L, D)    float32 — unit-normalised CAA pooled across intensities
    """
    N, E, I, L, D = emo_act.shape
    delta_sum = np.zeros((E, I, L, D), dtype=np.float64)

    logger.info("Computing deltas in chunks of %d …", chunk)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        # h_base broadcast: (chunk, 1, 1, L, D) → subtracts from (chunk, E, I, L, D)
        h_emo  = emo_act[start:end].astype(np.float64)   # (chunk, E, I, L, D)
        h_base = base_act[start:end, np.newaxis, np.newaxis, :, :].astype(np.float64)
        delta_sum += (h_emo - h_base).sum(axis=0)
        if (start // chunk) % 10 == 0:
            logger.info("  processed %d / %d sources", end, N)

    delta_mean  = delta_sum / N                              # (E, I, L, D)
    caa         = unit_norm(delta_mean.astype(np.float32))  # (E, I, L, D)
    caa_pooled  = unit_norm(
        delta_mean.mean(axis=1).astype(np.float32)          # (E, L, D)
    )
    return delta_mean.astype(np.float32), caa, caa_pooled


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--emotion-act",
        default="activation/emotion_rewrites/emotion_intensity_residual_stream.npy",
        help="Path to emotion×intensity .npy  (N, E, I, L, D)",
    )
    p.add_argument(
        "--emotion-info",
        default="activation/emotion_rewrites/emotion_intensity_residual_stream_info.json",
        help="Path to the corresponding _info.json",
    )
    p.add_argument(
        "--base-act",
        default="activation/emotion_rewrites/base_text_residual_stream.npy",
        help="Path to base-text .npy  (N, L, D)",
    )
    p.add_argument(
        "--base-info",
        default="activation/emotion_rewrites/base_text_residual_stream_info.json",
        help="Path to the base-text _info.json",
    )
    p.add_argument(
        "--out-dir",
        default="activation/emotion_rewrites",
        help="Output directory (default: activation/emotion_rewrites)",
    )
    p.add_argument(
        "--chunk", type=int, default=256,
        help="Number of sources per computation chunk (default: 256)",
    )
    return p.parse_args()


# Entry point

def main() -> None:
    args = parse_args()

    # Load metadata
    with open(args.emotion_info, encoding="utf-8") as f:
        emo_info = json.load(f)
    with open(args.base_info, encoding="utf-8") as f:
        base_info = json.load(f)

    emo_shape  = tuple(emo_info["shape"])    # (N, E, I, L, D)
    base_shape = tuple(base_info["shape"])   # (N, L, D)

    # Sanity-check source alignment
    if emo_info["source_ids"] != base_info["source_ids"]:
        raise ValueError(
            "source_ids in emotion and base-text info files do not match. "
            "Re-run extraction scripts to ensure identical ordering."
        )
    if emo_info["layer_indices"] != base_info["layer_indices"]:
        raise ValueError(
            "layer_indices differ between emotion and base-text info files."
        )

    logger.info("Emotion act shape : %s", emo_shape)
    logger.info("Base-text shape   : %s", base_shape)
    logger.info("Emotions          : %s", emo_info["emotion_order"])
    logger.info("Intensities       : %s", emo_info["intensity_order"])
    logger.info("Layers            : %s", emo_info["layer_indices"])

    # Memory-map tensors (avoid loading into RAM twice)
    emo_act  = np.memmap(args.emotion_act,  dtype="float32", mode="r", shape=emo_shape)
    base_act = np.memmap(args.base_act,     dtype="float32", mode="r", shape=base_shape)

    # Compute CAA directions
    delta_mean, caa, caa_pooled = compute_caa(emo_act, base_act, chunk=args.chunk)

    # Save
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path  = out_dir / "caa_emotion_directions.npz"
    info_path = out_dir / "caa_emotion_directions_info.json"

    np.savez(
        str(npz_path),
        caa=caa,               # (E, I, L, D)  unit-normalised
        caa_pooled=caa_pooled, # (E, L, D)      unit-normalised, pooled across intensities
        delta_mean=delta_mean, # (E, I, L, D)   raw mean deltas (un-normalised)
    )
    logger.info("Saved CAA directions → %s", npz_path)

    E, I, L, D = caa.shape
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "emotion_order":   emo_info["emotion_order"],
                "intensity_order": emo_info["intensity_order"],
                "layer_indices":   emo_info["layer_indices"],
                "n_sources":       int(emo_shape[0]),
                "shape_caa":          [E, I, L, D],
                "shape_caa_pooled":   [E, L, D],
                "shape_delta_mean":   [E, I, L, D],
                "dtype":           "float32",
                "description": (
                    "caa[e, i, l, :] = unit_norm( mean_n( h_emotion[n,e,i,l] - h_base[n,l] ) )  "
                    "caa_pooled[e, l, :] = unit_norm( mean_{n,i}( h_emotion[n,e,i,l] - h_base[n,l] ) )"
                ),
            },
            f, indent=2, ensure_ascii=False,
        )
    logger.info("Saved info          → %s", info_path)

    # Quick sanity check: print cosine similarity between pooled CAA directions
    logger.info("Pooled CAA inter-emotion cosine (layer index 2 = layer 13):")
    L_IDX = 2  # index of layer 13 in hook_layers [8,10,13,16,19,22]
    emotions = emo_info["emotion_order"]
    vecs = caa_pooled[:, L_IDX, :]   # (E, D)
    cos_mat = vecs @ vecs.T
    header = f"{'':12s}" + "".join(f"{e[:6]:>8s}" for e in emotions)
    logger.info(header)
    for i, e in enumerate(emotions):
        row = f"{e:<12s}" + "".join(f"{cos_mat[i,j]:>8.3f}" for j in range(len(emotions)))
        logger.info(row)


if __name__ == "__main__":
    main()
