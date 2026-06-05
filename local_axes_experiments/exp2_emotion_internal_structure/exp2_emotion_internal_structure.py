"""Experiment 2: Do individual emotions have internal sub-axes?

For each emotion e, collect per-source residual deltas after removing the
common affect direction g:

    delta_resid[n, e] = delta[n, e] - (delta[n, e] · g) * g

Then remove the source-level random effect by subtracting the per-source
mean across all emotions:

    delta_demeaned[n, e] = delta_resid[n, e] - mean_e( delta_resid[n, e] )

This eliminates source-specific variance (topic, style) that is shared
across all emotions and would otherwise inflate the PR.

Then run local PCA on X_e = {delta_demeaned[n, e]}_n and compute the
Participation Ratio (PR):

    PR(e) = (sum_k lambda_k)^2 / sum_k lambda_k^2

where lambda_k are the explained-variance eigenvalues of X_e.

PR ≈ 1   → emotion is essentially one-dimensional
PR ≈ K   → emotion spans K equally-weighted axes

Outputs are written to local_axes_experiments/exp2_emotion_internal_structure/results/.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = REPO_ROOT / "activation" / "emotion_rewrites"
OUT_DIR = REPO_ROOT / "local_axes_experiments" / "exp2_emotion_internal_structure" / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def project_out(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Remove component along unit vector g from each row of X.

    Parameters
    ----------
    X : (M, D)
    g : (D,)  — must be unit norm
    """
    coeff = X @ g          # (M,)
    return X - np.outer(coeff, g)


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Compute PR = (sum lambda)^2 / sum(lambda^2)."""
    lam = eigenvalues[eigenvalues > 0]
    if lam.size == 0:
        return float("nan")
    return float(lam.sum() ** 2 / (lam ** 2).sum())


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_emotion_subspace(
    X_e: np.ndarray,            # (N, D) float32 — already residualised
    emotion: str,
    layer: int,
    n_components: int,
) -> dict:
    """Run local PCA on X_e and return summary metrics."""
    N, D = X_e.shape
    logger.info(
        "  %s: N=%d, running PCA with %d components …",
        emotion, N, min(n_components, N, D),
    )

    X = X_e.astype(np.float64)

    # Centre
    mu = X.mean(axis=0)
    X_c = X - mu

    # PCA
    n_comp = min(n_components, N - 1, D)
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X_c)

    evr = pca.explained_variance_ratio_      # (n_comp,)
    ev  = pca.explained_variance_            # (n_comp,)

    # Full-variance PR using all components fitted
    pr = participation_ratio(ev)

    # EVR cumulative
    evr_cumsum = float(evr[:5].sum()) if len(evr) >= 5 else float(evr.sum())

    # Top-K EVR
    evr_top1 = float(evr[0]) if len(evr) >= 1 else float("nan")
    evr_top3 = float(evr[:3].sum()) if len(evr) >= 3 else float(evr.sum())

    return {
        "layer": layer,
        "emotion": emotion,
        "n_sources": N,
        "participation_ratio": pr,
        "evr_pc1": evr_top1,
        "evr_top3": evr_top3,
        "evr_top5": evr_cumsum,
        "n_components_fitted": n_comp,
    }


def build_per_emotion_residuals(
    emo_act: np.ndarray,       # (N, E, I, L, D)  mmap
    neu_act: np.ndarray,       # (N, L, D)         mmap
    caa_pooled: np.ndarray,    # (E, L, D)         unit-normed
    caa: np.ndarray,           # (E, I, L, D)      per-emotion per-intensity CAA
    layer_idx: int,            # index into the L dimension
    intensity_low_idx: int = 0,
    intensity_high_idx: int = 2,
    CHUNK: int = 512,
) -> np.ndarray:
    """Return delta_resid of shape (N, E, D) for a single layer.

    Removes two directions per emotion:
      1. Common affect direction g (mean of all emotion CAA directions)
      2. Per-emotion intensity direction u_e (high - low CAA direction)
    """
    N, E, I, L, D = emo_act.shape

    # Common direction g
    g_raw = caa_pooled[:, layer_idx, :].mean(axis=0)   # (D,)
    g = unit_norm(g_raw)                               # (D,)

    # Per-emotion intensity directions u_e: (E, D)
    intensity_dirs = unit_norm(
        caa[:, intensity_high_idx, layer_idx, :].astype(np.float64)
        - caa[:, intensity_low_idx, layer_idx, :].astype(np.float64)
    ).astype(np.float32)   # (E, D)

    delta_resid = np.empty((N, E, D), dtype=np.float32)

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        h_emo = emo_act[start:end, :, :, layer_idx, :].mean(axis=2).astype(np.float32)  # (c, E, D)
        h_neu = neu_act[start:end, layer_idx, :].astype(np.float32)                     # (c, D)
        delta = h_emo - h_neu[:, np.newaxis, :]                                          # (c, E, D)
        for e in range(E):
            d = project_out(delta[:, e, :], g)           # remove common direction
            d = project_out(d, intensity_dirs[e])        # remove per-emotion intensity axis
            delta_resid[start:end, e, :] = d

    return delta_resid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--emo-act",
        default=str(ACT_DIR / "emotion_intensity_residual_stream.npy"),
        help="Emotion×intensity residual stream (N, E, I, L, D)",
    )
    p.add_argument(
        "--emo-info",
        default=str(ACT_DIR / "emotion_intensity_residual_stream_info.json"),
    )
    p.add_argument(
        "--neu-act",
        default=str(ACT_DIR / "neutral_paraphrase_residual_stream.npy"),
        help="Neutral paraphrase residual stream (N, L, D)",
    )
    p.add_argument(
        "--caa-directions",
        default=str(ACT_DIR / "caa_emotion_directions.npz"),
    )
    p.add_argument(
        "--layer", type=int, default=13,
        help="Layer index to analyse (default: 13)",
    )
    p.add_argument(
        "--n-components", type=int, default=50,
        help="Number of PCA components to fit per emotion",
    )
    p.add_argument(
        "--out-dir", default=str(OUT_DIR),
    )
    p.add_argument(
        "--n-shuffle", type=int, default=10,
        help="Number of shuffle repetitions for noise baseline (0 to skip)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load metadata ---
    logger.info("Loading metadata …")
    with open(args.emo_info) as f:
        info = json.load(f)
    emotions: list[str] = info["emotion_order"]
    layer_indices: list[int] = info["layer_indices"]

    if args.layer not in layer_indices:
        raise ValueError(
            f"Layer {args.layer} not in available layers: {layer_indices}"
        )
    layer_idx = layer_indices.index(args.layer)

    # --- load activations (memory-mapped) ---
    logger.info("Loading emotion activations (mmap) …")
    emo_shape = tuple(info["shape"])
    emo_raw = np.memmap(args.emo_act, dtype=info["dtype"], mode="r", shape=emo_shape)

    logger.info("Loading neutral activations (mmap) …")
    neu_info_path = args.neu_act.replace(".npy", "_info.json")
    with open(neu_info_path) as f:
        neu_info = json.load(f)
    neu_shape = tuple(neu_info["shape"])
    neu_raw = np.memmap(args.neu_act, dtype=neu_info["dtype"], mode="r", shape=neu_shape)

    # --- load CAA directions ---
    logger.info("Loading CAA directions …")
    caa_npz = np.load(args.caa_directions)
    caa_pooled: np.ndarray = caa_npz["caa_pooled"]   # (E, L, D) unit-normed
    caa: np.ndarray = caa_npz["caa"]                 # (E, I, L, D)

    intensity_order: list[str] = info["intensity_order"]
    intensity_low_idx  = intensity_order.index("low")
    intensity_high_idx = intensity_order.index("high")

    # --- build per-emotion residuals for the target layer ---
    logger.info("Building residual deltas for layer %d (index %d) …", args.layer, layer_idx)
    delta_resid = build_per_emotion_residuals(
        emo_raw, neu_raw, caa_pooled, caa, layer_idx=layer_idx,
        intensity_low_idx=intensity_low_idx,
        intensity_high_idx=intensity_high_idx,
    )
    # delta_resid: (N, E, D)

    # --- remove source-level random effect ---
    # Subtract each source's mean across emotions to eliminate topic/style variance
    # that is shared across all emotions and would otherwise inflate the PR.
    logger.info("Removing per-source mean across emotions …")
    source_mean = delta_resid.mean(axis=1, keepdims=True)  # (N, 1, D)
    delta_resid = delta_resid - source_mean                # (N, E, D)

    # --- per-emotion local PCA ---
    logger.info("Running per-emotion local PCA …")
    rows = []
    for e_idx, emotion in enumerate(emotions):
        X_e = delta_resid[:, e_idx, :]   # (N, D)
        result = analyse_emotion_subspace(
            X_e, emotion, layer=args.layer, n_components=args.n_components,
        )
        rows.append(result)
        logger.info(
            "  %-14s  PR=%.2f  EVR-PC1=%.3f  EVR-top3=%.3f  EVR-top5=%.3f",
            emotion,
            result["participation_ratio"],
            result["evr_pc1"],
            result["evr_top3"],
            result["evr_top5"],
        )

    # --- save results ---
    df = pd.DataFrame(rows)

    # --- shuffle baseline ---
    if args.n_shuffle > 0:
        logger.info("Computing shuffle baseline (n_shuffle=%d) …", args.n_shuffle)
        rng = np.random.default_rng(42)
        shuffle_prs: dict[str, list[float]] = {e: [] for e in emotions}
        N_s, E_s, D_s = delta_resid.shape
        # Pool all emotion vectors together: (N*E, D)
        all_pool = delta_resid.reshape(-1, D_s)
        for _ in range(args.n_shuffle):
            for emotion in emotions:
                # Draw N random rows from the full pool — breaks emotion-specific structure
                idx = rng.integers(0, N_s * E_s, size=N_s)
                X_shuf = all_pool[idx].astype(np.float64)
                X_c = X_shuf - X_shuf.mean(axis=0)
                n_comp = min(args.n_components, N_s - 1, D_s)
                pca = PCA(n_components=n_comp, random_state=0)
                pca.fit(X_c)
                shuffle_prs[emotion].append(participation_ratio(pca.explained_variance_))
        df["pr_shuffle_mean"] = [float(np.mean(shuffle_prs[e])) for e in emotions]
        df["pr_shuffle_std"]  = [float(np.std(shuffle_prs[e]))  for e in emotions]
        df["pr_excess"]       = df["participation_ratio"] - df["pr_shuffle_mean"]
    else:
        df["pr_shuffle_mean"] = float("nan")
        df["pr_shuffle_std"]  = float("nan")
        df["pr_excess"]       = float("nan")

    out_csv = out_dir / f"layer{args.layer}_participation_ratio.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Saved summary → %s", out_csv)

    # --- also save per-emotion EVR curves ---
    for e_idx, emotion in enumerate(emotions):
        X_e = delta_resid[:, e_idx, :].astype(np.float64)
        mu = X_e.mean(axis=0)
        X_c = X_e - mu
        n_comp = min(args.n_components, X_e.shape[0] - 1, X_e.shape[1])
        pca = PCA(n_components=n_comp, random_state=0)
        pca.fit(X_c)

        evr_df = pd.DataFrame({
            "pc": np.arange(1, n_comp + 1),
            "evr": pca.explained_variance_ratio_,
            "evr_cumsum": np.cumsum(pca.explained_variance_ratio_),
            "eigenvalue": pca.explained_variance_,
        })
        evr_df["layer"] = args.layer
        evr_df["emotion"] = emotion
        evr_out = out_dir / f"layer{args.layer}_{emotion}_evr.csv"
        evr_df.to_csv(evr_out, index=False)

    logger.info("Done.")

    # --- print final summary ---
    print("\n=== Emotion Internal Structure - Layer %d ===" % args.layer)
    cols = [
        "emotion", "n_sources", "participation_ratio", "pr_shuffle_mean",
        "pr_excess", "evr_pc1", "evr_top3", "evr_top5",
    ]
    print(df[cols].to_string(index=False))
    print()
    print("Interpretation (PR_excess = PR_real - PR_shuffle):")
    for _, row in df.iterrows():
        pr = row["participation_ratio"]
        excess = row["pr_excess"]
        em = row["emotion"]
        if np.isnan(excess):
            interp = f"PR={pr:.2f}"
        else:
            interp = f"PR={pr:.2f}  PR_shuffle={row['pr_shuffle_mean']:.2f}  excess={excess:.2f}"
        print(f"  {em:<14}  {interp}")


if __name__ == "__main__":
    main()
