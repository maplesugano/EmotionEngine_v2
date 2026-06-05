"""Experiment 1: Is r_{e} truly an emotion-specific direction?

For each source n and emotion e, compute the per-source delta:

    delta[n, e] = h_emotion[n, e] - h_neutral[n]

where h_emotion[n, e] is averaged over intensity levels.

Then remove the common direction g (mean of all emotion-wise CAA directions,
pooled over intensity):

    delta_resid[n, e] = delta[n, e] - (delta[n, e] · g) * g

Finally, train a linear emotion classifier on delta_resid using
source-held-out GroupKFold cross-validation and report balanced accuracy.

Outputs are written to analysis/caa/residualized_per_source/.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, make_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = REPO_ROOT / "activation" / "emotion_rewrites"
OUT_DIR = REPO_ROOT / "local_axes_experiments" / "exp1_emotion_specificity" / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalise *v* along its last axis."""
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


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def build_dataset(
    emo_act: np.ndarray,   # (N, E, I, L, D)
    neu_act: np.ndarray,   # (N, L, D)
    caa_pooled: np.ndarray,  # (E, L, D)
    source_ids: list[str],
    emotions: list[str],
    layer_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_resid, y, groups) for a single layer.

    X_resid : (N*E, D) float32 — residualised deltas
    y       : (N*E,)  int      — emotion label index
    groups  : (N*E,)  int      — source group index for GroupKFold
    """
    N, E, I, L, D = emo_act.shape

    # Common direction g: mean of unit-normed pooled CAA directions over emotions
    # caa_pooled[e, layer_idx, :] is already unit-normed per emotion
    g_raw = caa_pooled[:, layer_idx, :].mean(axis=0)   # (D,)
    g = unit_norm(g_raw)                               # (D,)

    # Build delta_resid in chunks to avoid materialising (N, E, I, L, D) at once.
    # Output: (N*E, D) float32
    CHUNK = 512
    delta_resid = np.empty((N * E, D), dtype=np.float32)
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        # Load only this chunk's layer slice from mmap
        h_emo_c = emo_act[start:end, :, :, layer_idx, :].mean(axis=2).astype(np.float32)  # (c, E, D)
        h_neu_c = neu_act[start:end, layer_idx, :].astype(np.float32)                     # (c, D)
        delta_c = (h_emo_c - h_neu_c[:, np.newaxis, :]).reshape(-1, D)                   # (c*E, D)
        delta_resid[start * E : end * E] = project_out(delta_c, g)

    # Labels and groups
    labels = np.tile(np.arange(E), N)                  # (N*E,)  emotion index
    groups_idx = np.repeat(np.arange(N), E)            # (N*E,)  source index

    return delta_resid, labels, groups_idx


def run_classification(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    n_components: int | None,
    C: float,
) -> tuple[float, float]:
    """GroupKFold logistic regression. Returns (mean_bal_acc, std_bal_acc)."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    # PCA requires float64 to avoid overflow on large float32 activations
    X = X.astype(np.float64)

    steps = []

    if n_components is not None:
        max_comp = min(n_components, X.shape[1], X.shape[0] - 1)
        steps.append(("pca", PCA(n_components=max_comp, random_state=0)))

    # Scale after PCA so lbfgs converges reliably
    steps.append(("scaler", StandardScaler()))

    steps.append(
        ("clf", LogisticRegression(
            max_iter=2000,
            C=C,
            solver="lbfgs",
            random_state=0,
        ))
    )

    pipe = Pipeline(steps)
    cv = GroupKFold(n_splits=n_splits)
    scorer = make_scorer(balanced_accuracy_score)

    results = cross_validate(
        pipe, X, y,
        groups=groups,
        cv=cv,
        scoring=scorer,
        n_jobs=1,
    )
    scores = results["test_score"]
    return float(scores.mean()), float(scores.std())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--emo-act",
        default=str(ACT_DIR / "emotion_intensity_residual_stream.npy"),
        help="Path to emotion×intensity residual stream (N, E, I, L, D)",
    )
    p.add_argument(
        "--emo-info",
        default=str(ACT_DIR / "emotion_intensity_residual_stream_info.json"),
    )
    p.add_argument(
        "--neu-act",
        default=str(ACT_DIR / "neutral_paraphrase_residual_stream.npy"),
        help="Path to neutral paraphrase residual stream (N, L, D)",
    )
    p.add_argument(
        "--caa-directions",
        default=str(ACT_DIR / "caa_emotion_directions.npz"),
        help="CAA directions npz (keys: caa, caa_pooled, delta_mean)",
    )
    p.add_argument(
        "--n-splits", type=int, default=5,
        help="Number of GroupKFold splits",
    )
    p.add_argument(
        "--n-components", type=int, default=64,
        help="PCA components before classifier (None to disable)",
    )
    p.add_argument(
        "--C", type=float, default=1.0,
        help="Logistic regression regularisation strength",
    )
    p.add_argument(
        "--run-full-dim", action="store_true",
        help="Also run full-dimensional (4096-D) classification (slow, ~hours)",
    )
    p.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
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
    source_ids: list[str] = info["source_ids"]

    # --- load activations (memory-mapped) ---
    logger.info("Loading emotion activations (mmap) …")
    # Shape described in info: (N, E, I, L, D)
    emo_shape = tuple(info["shape"])   # e.g. (22782, 8, 3, 6, 4096)
    emo_raw = np.memmap(args.emo_act, dtype=info["dtype"], mode="r", shape=emo_shape)

    logger.info("Loading neutral activations (mmap) …")
    with open(str(args.neu_act).replace(".npy", "_info.json")) as f:
        neu_info = json.load(f)
    neu_shape = tuple(neu_info["shape"])   # (N, L, D)
    neu_raw = np.memmap(args.neu_act, dtype=neu_info["dtype"], mode="r", shape=neu_shape)

    # --- load CAA directions ---
    logger.info("Loading CAA directions …")
    caa_npz = np.load(args.caa_directions)
    caa_pooled: np.ndarray = caa_npz["caa_pooled"]  # (E, L, D) unit-normed

    # --- run per-layer ---
    L = len(layer_indices)
    rows = []

    for li, layer in enumerate(layer_indices):
        if layer != 13:
            continue
        logger.info("Layer %d (%d/%d) …", layer, li + 1, L)

        X_resid, y, groups = build_dataset(
            emo_raw, neu_raw, caa_pooled,
            source_ids, emotions, layer_idx=li,
        )

        # --- full-dimensional baseline (optional, slow) ---
        if args.run_full_dim:
            logger.info("  Full-dim classification …")
            bal_full, std_full = run_classification(
                X_resid, y, groups,
                n_splits=args.n_splits,
                n_components=None,
                C=args.C,
            )
            logger.info("  Full-dim  bal_acc = %.4f ± %.4f", bal_full, std_full)
        else:
            bal_full, std_full = float("nan"), float("nan")

        # --- reduced-dim (PCA) ---
        if args.n_components is not None:
            logger.info("  PCA-%d classification …", args.n_components)
            bal_pca, std_pca = run_classification(
                X_resid, y, groups,
                n_splits=args.n_splits,
                n_components=args.n_components,
                C=args.C,
            )
            logger.info("  PCA-%d    bal_acc = %.4f ± %.4f", args.n_components, bal_pca, std_pca)
        else:
            bal_pca, std_pca = float("nan"), float("nan")

        rows.append({
            "layer": layer,
            "bal_acc_full": bal_full,
            "std_full": std_full,
            "bal_acc_pca": bal_pca,
            "std_pca": std_pca,
            "n_components": args.n_components,
        })

    df = pd.DataFrame(rows)
    out_csv = out_dir / "exp1_emotion_specificity.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Results saved to %s", out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
