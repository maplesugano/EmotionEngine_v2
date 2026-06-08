"""Experiment 6.1: Does a single per-emotion-centered pooled PCA recover the
exp6 meta-axes?

Hypothesis (from the exp6 analysis):

    pooled PC1 = top eigvec of  Σ_{e,k} λ_{e,k} · u_{e,k} u_{e,k}^T   (variance-weighted)
    meta-PC1   = top eigvec of  Σ_{e,k}        1 · u_{e,k} u_{e,k}^T   (equal-weighted)

Both are top eigenvectors of the SAME sum of rank-1 outer products u u^T,
differing only in the weight (variance λ vs. 1). When a sub-axis is shared it
has large λ in every emotion AND appears many times, so it dominates the top
eigenvector of BOTH matrices. Therefore the top pooled components should align
with the exp6 meta-PCs.

This script:
  1. Rebuilds delta_resid via the exact exp6 pipeline (same residuals).
  2. Centers each emotion's cloud separately, pools all (N*E, D), runs one PCA.
  3. Loads the saved exp6 meta-PCA components and compares them to the pooled
     components via |cosine| (sign-neutral, since PC sign is arbitrary).
  4. Writes a comparison CSV + an |cos| heatmap and prints a summary table.

Run:
    python exp6_1_pooled_pca.py --layer 13 --top-k 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the exact exp6 residual pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp6_shared_subaxes import build_per_emotion_residuals  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR   = REPO_ROOT / "activation" / "emotion_rewrites"
EXP6_DIR  = REPO_ROOT / "local_axes_experiments" / "exp6_shared_subaxes"
OUT_DIR   = EXP6_DIR / "results_pooled"


# ---------------------------------------------------------------------------
# Pooled PCA (per-emotion centered)
# ---------------------------------------------------------------------------

def pooled_pca(
    delta_resid: np.ndarray,   # (N, E, D)
    n_comp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-emotion center, pool to (N*E, D), run a single PCA.

    Returns
    -------
    components : (n_comp, D)  unit-norm pooled PC directions
    evr        : (n_comp,)    explained variance ratio
    """
    N, E, D = delta_resid.shape
    blocks = []
    for e in range(E):
        X_e = delta_resid[:, e, :].astype(np.float64)
        X_c = X_e - X_e.mean(axis=0)          # center THIS emotion only
        blocks.append(X_c)
    X_pool = np.vstack(blocks)                 # (N*E, D), already ~zero mean

    n_comp = min(n_comp, X_pool.shape[0] - 1, D)
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X_pool)                            # sklearn re-centers (mean ~0 here)
    return pca.components_.astype(np.float64), pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def abs_cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """|cos| between every row of A (m,D) and every row of B (n,D) -> (m,n)."""
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return np.abs(An @ Bn.T)


def subspace_alignment(pooled: np.ndarray, meta: np.ndarray, r: int) -> dict:
    """Principal angles + mutual energy capture between the top-r subspaces.

    Axis-by-axis |cos| is misleading when meta-PCA components are near-degenerate
    (EVR_PC1 ~= EVR_PC2), since the axes are then free to rotate within the plane.
    The span overlap is the rotation-invariant comparison.
    """
    P = pooled[:r].T                       # (D, r)
    M = meta[:r].T                         # (D, r)
    cos_angles = np.cos(subspace_angles(P, M))   # r values, descending
    Qp, _ = np.linalg.qr(P)
    Qm, _ = np.linalg.qr(M)
    cap_meta_in_pooled = float(np.mean([np.linalg.norm(Qp.T @ meta[i]) ** 2 for i in range(r)]))
    cap_pooled_in_meta = float(np.mean([np.linalg.norm(Qm.T @ pooled[i]) ** 2 for i in range(r)]))
    return {
        "r": r,
        "principal_cos": cos_angles,
        "mean_principal_cos": float(cos_angles.mean()),
        "meta_captured_by_pooled": cap_meta_in_pooled,
        "pooled_captured_by_meta": cap_pooled_in_meta,
    }


def plot_abs_cos_heatmap(
    M: np.ndarray, row_labels: list[str], col_labels: list[str],
    layer: int, out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, M.shape[1] * 0.6), max(5, M.shape[0] * 0.6)))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=6)
    plt.colorbar(im, ax=ax, label="|cosine|")
    ax.set_title(f"Layer {layer}: pooled PCA vs exp6 meta-PCA  (|cos|)")
    fig.tight_layout()
    fig.savefig(out_dir / f"layer{layer}_pooled_vs_meta_cos.png", dpi=150)
    plt.close(fig)
    logger.info("Saved comparison heatmap.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emo-act",
                   default=str(ACT_DIR / "emotion_intensity_residual_stream.npy"))
    p.add_argument("--emo-info",
                   default=str(ACT_DIR / "emotion_intensity_residual_stream_info.json"))
    p.add_argument("--neu-act",
                   default=str(ACT_DIR / "neutral_paraphrase_residual_stream.npy"))
    p.add_argument("--caa-directions",
                   default=str(ACT_DIR / "caa_emotion_directions.npz"))
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--top-k", type=int, default=5,
                   help="How many pooled PCs to compute (default: 5)")
    p.add_argument("--n-meta-compare", type=int, default=3,
                   help="How many meta-PCs to compare against (default: 3)")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- metadata ---
    logger.info("Loading metadata …")
    with open(args.emo_info) as f:
        info = json.load(f)
    emotions: list[str] = info["emotion_order"]
    layer_indices: list[int] = info["layer_indices"]
    if args.layer not in layer_indices:
        raise ValueError(f"Layer {args.layer} not in {layer_indices}")
    layer_idx = layer_indices.index(args.layer)

    intensity_order: list[str] = info["intensity_order"]
    intensity_low_idx  = intensity_order.index("low")
    intensity_high_idx = intensity_order.index("high")

    # --- activations (mmap) ---
    logger.info("Memory-mapping activations …")
    emo_raw = np.memmap(args.emo_act, dtype=info["dtype"], mode="r",
                        shape=tuple(info["shape"]))
    with open(args.neu_act.replace(".npy", "_info.json")) as f:
        neu_info = json.load(f)
    neu_raw = np.memmap(args.neu_act, dtype=neu_info["dtype"], mode="r",
                        shape=tuple(neu_info["shape"]))

    caa_npz = np.load(args.caa_directions)
    caa_pooled = caa_npz["caa_pooled"]
    caa = caa_npz["caa"]

    # --- residuals (identical pipeline to exp6) ---
    logger.info("Building residual deltas for layer %d …", args.layer)
    delta_resid = build_per_emotion_residuals(
        emo_raw, neu_raw, caa_pooled, caa,
        layer_idx=layer_idx,
        intensity_low_idx=intensity_low_idx,
        intensity_high_idx=intensity_high_idx,
    )  # (N, E, D)
    logger.info("delta_resid shape = %s", delta_resid.shape)

    # --- pooled PCA ---
    logger.info("Running per-emotion-centered POOLED PCA …")
    pooled_comp, pooled_evr = pooled_pca(delta_resid, n_comp=args.top_k)
    logger.info("Pooled EVR: %s",
                "  ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(pooled_evr)))

    np.save(out_dir / f"layer{args.layer}_pooled_pcs.npy", pooled_comp.astype(np.float32))
    pd.DataFrame({
        "pooled_pc": np.arange(1, len(pooled_evr) + 1),
        "evr": pooled_evr,
        "evr_cumsum": np.cumsum(pooled_evr),
    }).to_csv(out_dir / f"layer{args.layer}_pooled_evr.csv", index=False)

    # --- load exp6 meta-PCA components ---
    meta_path = EXP6_DIR / "results" / f"layer{args.layer}_meta_pca_components.csv"
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_comp = meta_df.values.astype(np.float64)          # (n_meta, D)
    meta_labels = list(meta_df.index)
    n_cmp = min(args.n_meta_compare, meta_comp.shape[0])

    # --- |cos| comparison ---
    pooled_labels = [f"pooled_PC{i+1}" for i in range(pooled_comp.shape[0])]
    M = abs_cosine_matrix(pooled_comp, meta_comp[:n_cmp])   # (top_k, n_cmp)

    cmp_df = pd.DataFrame(M, index=pooled_labels, columns=meta_labels[:n_cmp])
    cmp_df.to_csv(out_dir / f"layer{args.layer}_pooled_vs_meta_cos.csv")

    # best meta match per pooled PC
    print(f"\n=== Exp 6.1 – Layer {args.layer}: pooled PCA vs exp6 meta-PCA ===\n")
    print("Pooled PCA explained-variance ratio:")
    for i, v in enumerate(pooled_evr):
        print(f"  pooled_PC{i+1}: EVR={v:.3f}  cum={np.cumsum(pooled_evr)[i]:.3f}")

    print(f"\n|cosine| pooled PC  ×  meta PC (top {n_cmp}):")
    print(cmp_df.round(3).to_string())

    print("\nBest meta-PC match for each pooled PC:")
    for i, plabel in enumerate(pooled_labels):
        j = int(np.argmax(M[i]))
        print(f"  {plabel:12s} ↔ {meta_labels[j]:10s}  |cos|={M[i, j]:.3f}")

    # diagonal check: pooled_PCi vs meta_PCi
    print("\nDiagonal alignment (pooled_PCi vs meta_PCi):")
    for i in range(min(n_cmp, pooled_comp.shape[0])):
        print(f"  pooled_PC{i+1} · meta_PC{i+1}  |cos|={M[i, i]:.3f}")

    plot_abs_cos_heatmap(M, pooled_labels, meta_labels[:n_cmp], args.layer, out_dir)

    # --- subspace alignment (rotation-invariant) ---
    print("\nSubspace alignment (principal angles + mutual energy capture):")
    sub_rows = []
    for r in range(1, n_cmp + 1):
        res = subspace_alignment(pooled_comp, meta_comp, r)
        print(f"  top-{r}: principal |cos| = {np.round(res['principal_cos'], 3)}  "
              f"mean={res['mean_principal_cos']:.3f}  "
              f"meta⊂pooled={res['meta_captured_by_pooled']:.3f}  "
              f"pooled⊂meta={res['pooled_captured_by_meta']:.3f}")
        sub_rows.append({
            "top_r": r,
            "mean_principal_cos": res["mean_principal_cos"],
            "meta_captured_by_pooled": res["meta_captured_by_pooled"],
            "pooled_captured_by_meta": res["pooled_captured_by_meta"],
            "principal_cos": ";".join(f"{c:.4f}" for c in res["principal_cos"]),
        })
    pd.DataFrame(sub_rows).to_csv(
        out_dir / f"layer{args.layer}_subspace_alignment.csv", index=False)

    # --- universal-PC1 check: does pooled_PC1 equal the mean of per-emotion PC1? ---
    U = np.load(EXP6_DIR / "results" / f"layer{args.layer}_local_pcs.npy").astype(np.float64)
    with open(EXP6_DIR / "results" / f"layer{args.layer}_local_pcs_labels.json") as f:
        u_labels = json.load(f)["labels"]
    pc1 = U[[i for i, l in enumerate(u_labels) if l.endswith("_PC1")]]
    pc1 = pc1 / np.linalg.norm(pc1, axis=1, keepdims=True)
    mean_pc1 = pc1.mean(0); mean_pc1 /= np.linalg.norm(mean_pc1)
    print("\nUniversal-PC1 check (mean of 8 per-emotion PC1 directions):")
    print(f"  pooled_PC1 |cos| universal-PC1 = {abs(pooled_comp[0] @ mean_pc1):.3f}")
    print(f"  meta_PC1   |cos| universal-PC1 = {abs(meta_comp[0] @ mean_pc1):.3f}")
    print(f"  meta_PC2   |cos| universal-PC1 = {abs(meta_comp[1] @ mean_pc1):.3f}")
    print(f"  pooled_PC1 |cos| each emotion PC1 = {np.round(np.abs(pc1 @ pooled_comp[0]), 2)}")

    logger.info("All outputs written to %s", out_dir)
    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
