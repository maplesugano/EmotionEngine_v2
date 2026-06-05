"""Experiment 7: Extract shared meta-axes m_k from local PC vectors.

Building on exp6's finding that local PCs cluster by PC number rather than
emotion label, this experiment formally extracts the shared affective axes m_k:

    m_k = sign_aligned_mean({ u_{e,k} | e in emotions })

Two extraction methods are compared:
  1. Sign-aligned mean: align signs within each PC-number group, then average
     and unit-normalise.  Simple and interpretable.
  2. Meta-PCA first component: take the first PC of each PC-number group.
     More robust to noisy outlier vectors.

Both produce one vector per PC rank (m_1 … m_K).

Post-extraction analysis:
  - Per-axis cosine alignment to each u_{e,k}  (how "universal" is m_k?)
  - Mutual orthogonality of m_1 … m_K
  - Overlap with g (common emotionalization) and each r_e (emotion residuals)
  - Reconstruction quality: |r_e - proj(r_e onto m_1…m_K)| / ||r_e||

Outputs written to results/:
  layer{L}_meta_axes.npy              — (K, D) float32 meta-axis vectors
  layer{L}_meta_axes_info.json        — metadata + per-axis stats
  layer{L}_meta_axes_orthogonality.csv— K×K cosine matrix among m_k
  layer{L}_meta_axes_emotion_overlap.csv — cos(m_k, r_e) for all e, k
  layer{L}_reconstruction_quality.csv— r_e reconstruction error per emotion

Run:
    python exp7_meta_axis_extraction.py --layer 13 --top-k 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR   = REPO_ROOT / "activation" / "emotion_rewrites"
EXP6_DIR  = REPO_ROOT / "local_axes_experiments" / "exp6_shared_subaxes" / "results"
OUT_DIR   = Path(__file__).resolve().parent / "results"

EMOTION_COLOURS = {
    "anger":        "#d62728",
    "anticipation": "#ff7f0e",
    "disgust":      "#8c564b",
    "fear":         "#9467bd",
    "joy":          "#2ca02c",
    "sadness":      "#1f77b4",
    "surprise":     "#bcbd22",
    "trust":        "#17becf",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


# ---------------------------------------------------------------------------
# Method 1: Sign-aligned mean
# ---------------------------------------------------------------------------

def extract_by_sign_aligned_mean(
    U: np.ndarray,       # (E*K, D)
    labels: list[str],
    emotions: list[str],
    top_k: int,
) -> tuple[np.ndarray, list[dict]]:
    """Return m_k vectors by sign-aligning each PC-number group and averaging.

    For each k in 1..top_k:
      - collect all u_{e,k}
      - reference = u_{emotions[0],k}
      - flip sign of each u_{e,k} if cos(u_{e,k}, ref) < 0
      - average and unit-normalise

    Returns
    -------
    M : (K, D) float64  meta-axis vectors (unit-normed)
    stats : list of K dicts with per-axis statistics
    """
    E = len(emotions)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    M_list: list[np.ndarray] = []
    stats_list: list[dict] = []

    for k in range(1, top_k + 1):
        group_labels = [f"{e}_PC{k}" for e in emotions]
        vecs = np.array([U[label_to_idx[lbl]] for lbl in group_labels], dtype=np.float64)
        # unit-normalise each vector first
        vecs = unit_norm(vecs)

        ref = vecs[0]
        cos_to_ref = vecs @ ref
        signs = np.where(cos_to_ref >= 0, 1.0, -1.0)
        aligned = vecs * signs[:, np.newaxis]

        m_k = unit_norm(aligned.mean(axis=0))

        # per-axis statistics
        cos_after_align = aligned @ m_k          # (E,) should all be positive
        cos_before_flip = np.abs(vecs @ m_k)     # absolute agreement ignoring sign

        stats_list.append({
            "axis": f"m{k}",
            "pc_rank": k,
            "mean_cos_aligned": float(cos_after_align.mean()),
            "min_cos_aligned":  float(cos_after_align.min()),
            "std_cos_aligned":  float(cos_after_align.std()),
            "mean_abs_cos":     float(cos_before_flip.mean()),
            "n_sign_flips":     int((signs < 0).sum()),
            "per_emotion_cos":  {e: float(cos_after_align[i]) for i, e in enumerate(emotions)},
        })
        M_list.append(m_k)

        logger.info(
            "  m%d  mean_cos=%.3f  min_cos=%.3f  n_flips=%d",
            k,
            stats_list[-1]["mean_cos_aligned"],
            stats_list[-1]["min_cos_aligned"],
            stats_list[-1]["n_sign_flips"],
        )

    M = np.array(M_list, dtype=np.float64)
    return M, stats_list


# ---------------------------------------------------------------------------
# Method 2: Meta-PCA first component per group
# ---------------------------------------------------------------------------

def extract_by_group_pca(
    U: np.ndarray,
    labels: list[str],
    emotions: list[str],
    top_k: int,
) -> np.ndarray:
    """Return m_k vectors as the first PCA component of each PC-number group."""
    from sklearn.decomposition import PCA

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    M_list: list[np.ndarray] = []

    for k in range(1, top_k + 1):
        group_labels = [f"{e}_PC{k}" for e in emotions]
        vecs = np.array([U[label_to_idx[lbl]] for lbl in group_labels], dtype=np.float64)
        vecs = unit_norm(vecs)
        pca = PCA(n_components=1, random_state=0)
        pca.fit(vecs)
        m_k = unit_norm(pca.components_[0])
        M_list.append(m_k)

    return np.array(M_list, dtype=np.float64)


# ---------------------------------------------------------------------------
# Post-extraction analysis
# ---------------------------------------------------------------------------

def compute_orthogonality(M: np.ndarray, axis_ids: list[str]) -> pd.DataFrame:
    """Return K×K DataFrame of cosine similarities among m_k."""
    cos = M @ M.T
    np.fill_diagonal(cos, 1.0)
    return pd.DataFrame(cos, index=axis_ids, columns=axis_ids)


def compute_emotion_overlap(
    M: np.ndarray,              # (K, D)
    caa_pooled: np.ndarray,     # (E, L, D)
    layer_idx: int,
    emotions: list[str],
    axis_ids: list[str],
) -> pd.DataFrame:
    """Return DataFrame of cos(m_k, r_e) for all k, e.

    r_e = unit_norm(caa_pooled[e, layer_idx, :])
    """
    rows = []
    for e_idx, emo in enumerate(emotions):
        r_e = unit_norm(caa_pooled[e_idx, layer_idx, :].astype(np.float64))
        cos_vals = M @ r_e
        row = {"emotion": emo}
        for k_idx, aid in enumerate(axis_ids):
            row[aid] = float(cos_vals[k_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def compute_g_overlap(
    M: np.ndarray,
    caa_pooled: np.ndarray,
    layer_idx: int,
    axis_ids: list[str],
) -> pd.DataFrame:
    """Return cos(m_k, g) for each k."""
    g_raw = caa_pooled[:, layer_idx, :].mean(axis=0)
    g = unit_norm(g_raw.astype(np.float64))
    cos_vals = M @ g
    return pd.DataFrame(
        [{"axis": aid, "cos_to_g": float(cos_vals[i])} for i, aid in enumerate(axis_ids)]
    )


def compute_reconstruction_quality(
    M: np.ndarray,
    caa_pooled: np.ndarray,
    caa: np.ndarray,
    layer_idx: int,
    emotions: list[str],
    intensity_low_idx: int,
    intensity_high_idx: int,
) -> pd.DataFrame:
    """How much of r_e is explained by the m_k subspace?

    Metric:
      r_e_proj = Σ_k (r_e · m_k) * m_k
      reconstruction_error = ||r_e - r_e_proj|| / ||r_e||

    Also reports:
      explained_var_ratio = 1 - reconstruction_error^2 (cosine-based)
    """
    rows = []
    g_raw = caa_pooled[:, layer_idx, :].mean(axis=0)
    g = unit_norm(g_raw.astype(np.float64))

    for e_idx, emo in enumerate(emotions):
        r_e_full = caa_pooled[e_idx, layer_idx, :].astype(np.float64)
        # Remove g component to get the residual direction
        r_e = r_e_full - (r_e_full @ g) * g
        r_e = unit_norm(r_e)

        # Project onto m_k subspace
        coeffs = M @ r_e       # (K,)
        r_e_proj = (coeffs[:, np.newaxis] * M).sum(axis=0)
        proj_norm = float(np.linalg.norm(r_e_proj))
        residual_norm = float(np.linalg.norm(r_e - r_e_proj))

        rows.append({
            "emotion": emo,
            "proj_norm": proj_norm,
            "residual_norm": residual_norm,
            "explained_var_ratio": float(1.0 - residual_norm**2),  # = Σ (r_e·m_k)^2
            **{f"coeff_m{k+1}": float(coeffs[k]) for k in range(len(coeffs))},
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_method_comparison(
    M_mean: np.ndarray,
    M_pca: np.ndarray,
    axis_ids: list[str],
    layer: int,
    out_dir: Path,
) -> None:
    """Bar chart of |cos(m_k_mean, m_k_pca)| for each k."""
    cos_vals = np.abs(np.einsum("kd,kd->k", M_mean, M_pca))
    fig, ax = plt.subplots(figsize=(max(5, len(axis_ids) * 0.8), 4))
    ax.bar(axis_ids, cos_vals, color="#4c72b0")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Meta-axis")
    ax.set_ylabel("|cos(m_k_mean, m_k_pca)|")
    ax.set_title(f"Layer {layer}: Agreement between sign-aligned-mean and PCA extraction")
    ax.axhline(0.9, linestyle="--", color="#c44e52", linewidth=1, label="0.9 threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"layer{layer}_method_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved method comparison plot.")


def plot_emotion_overlap_heatmap(
    df: pd.DataFrame,
    axis_ids: list[str],
    layer: int,
    out_dir: Path,
) -> None:
    """Heatmap of cos(m_k, r_e) for all e, k."""
    mat = df.set_index("emotion")[axis_ids].values.astype(float)
    emotions = list(df["emotion"])

    fig, ax = plt.subplots(figsize=(max(5, len(axis_ids) * 0.9), max(4, len(emotions) * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(axis_ids)))
    ax.set_xticklabels(axis_ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions, fontsize=8)
    plt.colorbar(im, ax=ax, label="cos(m_k, r_e)")
    ax.set_title(f"Layer {layer}: Emotion–meta-axis overlap  cos(m_k, r_e)")
    fig.tight_layout()
    fig.savefig(out_dir / f"layer{layer}_emotion_overlap_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved emotion overlap heatmap.")


def plot_reconstruction_quality(
    df: pd.DataFrame,
    layer: int,
    out_dir: Path,
) -> None:
    """Bar chart of explained_var_ratio per emotion."""
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.9), 4))
    colours = [EMOTION_COLOURS.get(e, "#555555") for e in df["emotion"]]
    ax.bar(df["emotion"], df["explained_var_ratio"], color=colours)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Explained variance ratio (r_e via m_k subspace)")
    ax.set_xlabel("Emotion")
    ax.set_title(f"Layer {layer}: How much of r_e is captured by shared m_k axes?")
    ax.axhline(0.8, linestyle="--", color="black", linewidth=1, alpha=0.5, label="0.8")
    ax.legend(fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / f"layer{layer}_reconstruction_quality.png", dpi=150)
    plt.close(fig)
    logger.info("Saved reconstruction quality plot.")


def plot_alignment_per_emotion(
    stats_list: list[dict],
    emotions: list[str],
    layer: int,
    out_dir: Path,
) -> None:
    """Grid of bar charts: for each m_k, show cos to each u_{e,k}."""
    K = len(stats_list)
    fig, axes = plt.subplots(1, K, figsize=(K * 2.5, 3.5), sharey=True)
    if K == 1:
        axes = [axes]
    for k_idx, (ax, stat) in enumerate(zip(axes, stats_list)):
        per_emo = stat["per_emotion_cos"]
        vals = [per_emo[e] for e in emotions]
        cols = [EMOTION_COLOURS.get(e, "#555555") for e in emotions]
        ax.bar(range(len(emotions)), vals, color=cols)
        ax.set_xticks(range(len(emotions)))
        ax.set_xticklabels([e[:3] for e in emotions], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"m{k_idx+1}\nμ={stat['mean_cos_aligned']:.2f}", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(0.5, color="#999999", linewidth=0.5, linestyle="--")
        if k_idx == 0:
            ax.set_ylabel("cos(m_k, u_{e,k})")
        ax.set_ylim(-0.3, 1.05)

    fig.suptitle(f"Layer {layer}: Alignment of shared m_k with each emotion's local PC", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"layer{layer}_alignment_per_emotion.png", dpi=150)
    plt.close(fig)
    logger.info("Saved per-emotion alignment plot.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--exp6-dir",
        default=str(EXP6_DIR),
        help="Directory containing exp6 results (layer{L}_local_pcs.npy etc.)",
    )
    p.add_argument(
        "--caa-directions",
        default=str(ACT_DIR / "caa_emotion_directions.npz"),
    )
    p.add_argument(
        "--emo-info",
        default=str(ACT_DIR / "emotion_intensity_residual_stream_info.json"),
    )
    p.add_argument("--layer", type=int, default=13)
    p.add_argument(
        "--top-k", type=int, default=5,
        help="Number of meta-axes to extract (must match exp6 --top-k)",
    )
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp6_dir = Path(args.exp6_dir)

    # --- load exp6 outputs ---
    logger.info("Loading exp6 local PC vectors …")
    U_path = exp6_dir / f"layer{args.layer}_local_pcs.npy"
    meta_path = exp6_dir / f"layer{args.layer}_local_pcs_labels.json"

    if not U_path.exists():
        raise FileNotFoundError(
            f"exp6 local PCs not found: {U_path}\n"
            "Run exp6_shared_subaxes.py first."
        )

    U = np.load(U_path).astype(np.float64)
    with open(meta_path) as f:
        meta = json.load(f)
    labels: list[str] = meta["labels"]
    emotions: list[str] = meta["emotions"]
    top_k_used: int = meta["top_k"]

    if args.top_k > top_k_used:
        raise ValueError(
            f"Requested --top-k {args.top_k} > exp6 top_k {top_k_used}. "
            "Re-run exp6 with a higher --top-k."
        )

    top_k = args.top_k
    axis_ids = [f"m{k}" for k in range(1, top_k + 1)]

    logger.info("  Loaded U: %s  labels: %d", U.shape, len(labels))
    logger.info("  Emotions: %s", emotions)

    # --- load CAA directions ---
    logger.info("Loading CAA directions …")
    caa_npz = np.load(args.caa_directions)
    caa_pooled: np.ndarray = caa_npz["caa_pooled"]   # (E, L, D)
    caa: np.ndarray = caa_npz["caa"]                 # (E, I, L, D)

    with open(args.emo_info) as f:
        info = json.load(f)
    layer_indices: list[int] = info["layer_indices"]
    if args.layer not in layer_indices:
        raise ValueError(f"Layer {args.layer} not in {layer_indices}")
    layer_idx = layer_indices.index(args.layer)

    intensity_order: list[str] = info["intensity_order"]
    intensity_low_idx  = intensity_order.index("low")
    intensity_high_idx = intensity_order.index("high")

    # --- Method 1: sign-aligned mean ---
    logger.info("Extracting meta-axes by sign-aligned mean …")
    M_mean, stats_list = extract_by_sign_aligned_mean(U, labels, emotions, top_k)

    # --- Method 2: group PCA ---
    logger.info("Extracting meta-axes by group PCA …")
    M_pca = extract_by_group_pca(U, labels, emotions, top_k)

    # Agreement between methods
    method_cos = np.abs(np.einsum("kd,kd->k", M_mean, M_pca))
    for k_idx, c in enumerate(method_cos):
        logger.info("  m%d: |cos(mean, pca)| = %.4f", k_idx + 1, c)

    # --- post-extraction analysis ---
    logger.info("Computing orthogonality matrix …")
    ortho_df = compute_orthogonality(M_mean, axis_ids)

    logger.info("Computing emotion overlap …")
    emo_overlap_df = compute_emotion_overlap(
        M_mean, caa_pooled, layer_idx, emotions, axis_ids
    )

    logger.info("Computing overlap with g …")
    g_overlap_df = compute_g_overlap(M_mean, caa_pooled, layer_idx, axis_ids)

    logger.info("Computing reconstruction quality of r_e …")
    recon_df = compute_reconstruction_quality(
        M_mean, caa_pooled, caa, layer_idx, emotions,
        intensity_low_idx, intensity_high_idx,
    )

    # --- save vectors ---
    np.save(out_dir / f"layer{args.layer}_meta_axes.npy", M_mean.astype(np.float32))
    np.save(out_dir / f"layer{args.layer}_meta_axes_pca.npy", M_pca.astype(np.float32))
    logger.info("Saved meta-axis vectors.")

    # --- save CSV tables ---
    ortho_df.to_csv(out_dir / f"layer{args.layer}_meta_axes_orthogonality.csv")
    emo_overlap_df.to_csv(out_dir / f"layer{args.layer}_meta_axes_emotion_overlap.csv", index=False)
    g_overlap_df.to_csv(out_dir / f"layer{args.layer}_g_overlap.csv", index=False)
    recon_df.to_csv(out_dir / f"layer{args.layer}_reconstruction_quality.csv", index=False)

    # --- save info JSON ---
    info_out = {
        "layer": args.layer,
        "n_axes": top_k,
        "axis_ids": axis_ids,
        "method": "sign_aligned_mean",
        "emotions": emotions,
        "method_agreement": {f"m{k+1}": float(c) for k, c in enumerate(method_cos)},
        "per_axis_stats": stats_list,
        "g_overlap": {row["axis"]: row["cos_to_g"] for _, row in g_overlap_df.iterrows()},
        "mean_reconstruction_evr": float(recon_df["explained_var_ratio"].mean()),
    }
    with open(out_dir / f"layer{args.layer}_meta_axes_info.json", "w") as f:
        json.dump(info_out, f, indent=2)

    # --- print summary ---
    print(f"\n=== Exp 7 — Layer {args.layer}: Meta-axis extraction summary ===\n")
    print(f"{'Axis':<6}  {'mean_cos':>8}  {'min_cos':>8}  {'n_flips':>7}  "
          f"{'|cos(mean,pca)|':>15}  {'cos_to_g':>8}")
    print("-" * 60)
    for k_idx, stat in enumerate(stats_list):
        gcos = info_out["g_overlap"].get(f"m{k_idx+1}", float("nan"))
        print(f"  m{k_idx+1}    {stat['mean_cos_aligned']:8.3f}  {stat['min_cos_aligned']:8.3f}  "
              f"{stat['n_sign_flips']:7d}  {method_cos[k_idx]:15.3f}  {gcos:8.3f}")

    print("\n=== Reconstruction quality (how much of r_e lives in m_k subspace) ===\n")
    print(recon_df[["emotion", "explained_var_ratio"] +
                    [f"coeff_m{k+1}" for k in range(top_k)]].to_string(index=False))

    print("\n=== Mutual orthogonality among m_k ===\n")
    print(ortho_df.round(3).to_string())

    # --- plots ---
    logger.info("Generating plots …")
    plot_method_comparison(M_mean, M_pca, axis_ids, args.layer, out_dir)
    plot_emotion_overlap_heatmap(emo_overlap_df, axis_ids, args.layer, out_dir)
    plot_reconstruction_quality(recon_df, args.layer, out_dir)
    plot_alignment_per_emotion(stats_list, emotions, args.layer, out_dir)

    logger.info("All outputs written to %s", out_dir)
    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
