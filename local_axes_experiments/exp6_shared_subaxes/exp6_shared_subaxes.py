"""Experiment 6: Are local sub-axes emotion-specific or shared across emotions?

For each emotion e, run local PCA on the residual delta cloud (same pipeline
as exp2) and collect the top-K PC vectors:

    U = { u_{e,k} }   for e in emotions, k in 1..K

Then:

  1. Cosine-similarity matrix  cos(u_{e,k}, u_{e',k'}) for all pairs.
  2. Meta-PCA: PCA on the stacked PC vectors to find label-free affective axes.
  3. Hierarchical clustering (Ward linkage on angular distance) to detect
     which local axes cluster together across emotion labels.
  4. Summary CSV and plots.

Run:
    python exp6_shared_subaxes.py --layer 13 --top-k 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
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
OUT_DIR   = REPO_ROOT / "local_axes_experiments" / "exp6_shared_subaxes" / "results"

# Colour palette per emotion for plots
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
# Shared helpers (identical to exp2)
# ---------------------------------------------------------------------------

def unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def project_out(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Remove component along unit vector g from each row of X (M, D)."""
    coeff = X @ g
    return X - np.outer(coeff, g)


def build_per_emotion_residuals(
    emo_act: np.ndarray,       # (N, E, I, L, D)
    neu_act: np.ndarray,       # (N, L, D)
    caa_pooled: np.ndarray,    # (E, L, D)  unit-normed
    caa: np.ndarray,           # (E, I, L, D)
    layer_idx: int,
    intensity_low_idx: int = 0,
    intensity_high_idx: int = 2,
    CHUNK: int = 512,
) -> np.ndarray:
    """Return delta_resid of shape (N, E, D) for a single layer.

    Same pipeline as exp2: removes common affect direction g and per-emotion
    intensity direction u_e, then subtracts per-source mean across emotions.
    """
    N, E, I, L, D = emo_act.shape

    g_raw = caa_pooled[:, layer_idx, :].mean(axis=0)
    g = unit_norm(g_raw)

    intensity_dirs = unit_norm(
        caa[:, intensity_high_idx, layer_idx, :].astype(np.float64)
        - caa[:, intensity_low_idx, layer_idx, :].astype(np.float64)
    ).astype(np.float32)   # (E, D)

    delta_resid = np.empty((N, E, D), dtype=np.float32)

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        h_emo = emo_act[start:end, :, :, layer_idx, :].mean(axis=2).astype(np.float32)
        h_neu = neu_act[start:end, layer_idx, :].astype(np.float32)
        delta = h_emo - h_neu[:, np.newaxis, :]
        for e in range(E):
            d = project_out(delta[:, e, :].astype(np.float64), g)
            d = project_out(d, intensity_dirs[e].astype(np.float64))
            delta_resid[start:end, e, :] = d.astype(np.float32)

    # Remove per-source mean across emotions (topic/style)
    source_mean = delta_resid.mean(axis=1, keepdims=True)
    delta_resid -= source_mean

    return delta_resid


# ---------------------------------------------------------------------------
# Local PCA: extract top-K PC vectors per emotion
# ---------------------------------------------------------------------------

def extract_local_pcs(
    delta_resid: np.ndarray,   # (N, E, D)
    emotions: list[str],
    top_k: int,
    layer: int,
) -> tuple[np.ndarray, list[str]]:
    """Return unit-normed PC matrix and labels.

    Returns
    -------
    U : (E * top_k, D)  – each row is one local PC vector (unit norm)
    labels : list of str, length E * top_k, e.g. "joy_PC1"
    """
    N, E, D = delta_resid.shape
    U_list: list[np.ndarray] = []
    labels: list[str] = []

    for e_idx, emotion in enumerate(emotions):
        X_e = delta_resid[:, e_idx, :].astype(np.float64)
        X_c = X_e - X_e.mean(axis=0)
        n_comp = min(top_k, N - 1, D)
        pca = PCA(n_components=n_comp, random_state=0)
        pca.fit(X_c)
        vecs = pca.components_[:n_comp]   # (n_comp, D)  already unit norm

        # Pad with zeros if fewer components available
        if n_comp < top_k:
            pad = np.zeros((top_k - n_comp, D))
            vecs = np.vstack([vecs, pad])

        for k in range(top_k):
            label = f"{emotion}_PC{k + 1}"
            U_list.append(vecs[k])
            labels.append(label)
        logger.info(
            "  %-14s  fitted %d PCs  EVR-PC1=%.3f",
            emotion,
            n_comp,
            float(pca.explained_variance_ratio_[0]) if n_comp > 0 else float("nan"),
        )

    U = np.array(U_list, dtype=np.float64)   # (E*K, D)
    return U, labels


# ---------------------------------------------------------------------------
# Cosine similarity matrix
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(U: np.ndarray) -> np.ndarray:
    """Return (M, M) cosine similarity matrix for row-vectors U (M, D)."""
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U_hat = U / np.maximum(norms, 1e-12)
    return U_hat @ U_hat.T


# ---------------------------------------------------------------------------
# Meta-PCA: PCA on the PC vectors themselves
# ---------------------------------------------------------------------------

def meta_pca(
    U: np.ndarray,              # (M, D)
    labels: list[str],
    emotions: list[str],
    n_meta: int,
    layer: int,
    out_dir: Path,
) -> pd.DataFrame:
    """Run PCA on U and return DataFrame of projections."""
    M, D = U.shape
    n_comp = min(n_meta, M - 1, D)
    pca = PCA(n_components=n_comp, random_state=0)
    coords = pca.fit_transform(U)   # (M, n_comp)

    evr = pca.explained_variance_ratio_
    logger.info("Meta-PCA EVR: %s", "  ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(evr)))

    # Save component vectors
    comp_df = pd.DataFrame(
        pca.components_,
        index=[f"meta_PC{i+1}" for i in range(n_comp)],
    )
    comp_df.to_csv(out_dir / f"layer{layer}_meta_pca_components.csv")

    # Save EVR
    evr_df = pd.DataFrame({
        "meta_pc": np.arange(1, n_comp + 1),
        "evr": evr,
        "evr_cumsum": np.cumsum(evr),
    })
    evr_df.to_csv(out_dir / f"layer{layer}_meta_pca_evr.csv", index=False)

    # Save projections
    proj_rows = []
    for i, label in enumerate(labels):
        parts = label.rsplit("_PC", 1)
        emotion = parts[0]
        pc_num = int(parts[1])
        row = {"label": label, "emotion": emotion, "pc_num": pc_num}
        for j in range(n_comp):
            row[f"meta_PC{j+1}"] = float(coords[i, j])
        proj_rows.append(row)
    proj_df = pd.DataFrame(proj_rows)
    proj_df.to_csv(out_dir / f"layer{layer}_meta_pca_projections.csv", index=False)

    return proj_df


# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------

def hierarchical_clustering(
    cos_sim: np.ndarray,      # (M, M)
    labels: list[str],
    layer: int,
    n_clusters: int,
    out_dir: Path,
) -> pd.DataFrame:
    """Cluster local PC axes by angular distance (Ward linkage)."""
    # Angular distance in [0, 1]: uses |cosine| so sign-flip of a PC is neutral
    angular_dist = 1.0 - np.abs(cos_sim)
    np.fill_diagonal(angular_dist, 0.0)
    # Clip numerical noise
    angular_dist = np.clip(angular_dist, 0.0, 1.0)

    condensed = squareform(angular_dist, checks=False)
    Z = linkage(condensed, method="ward")

    cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")

    rows = []
    for i, label in enumerate(labels):
        parts = label.rsplit("_PC", 1)
        rows.append({
            "label": label,
            "emotion": parts[0],
            "pc_num": int(parts[1]),
            "cluster": int(cluster_ids[i]),
        })
    clust_df = pd.DataFrame(rows)
    clust_df.to_csv(out_dir / f"layer{layer}_clusters_k{n_clusters}.csv", index=False)

    return clust_df, Z, angular_dist


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cosine_heatmap(
    cos_sim: np.ndarray,
    labels: list[str],
    emotions: list[str],
    layer: int,
    out_dir: Path,
) -> None:
    M = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, M * 0.45), max(6, M * 0.4)))
    im = ax.imshow(np.abs(cos_sim), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(M))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(M))
    ax.set_yticklabels(labels, fontsize=6)

    # Draw emotion block boundaries
    top_k = M // len(emotions)
    for i in range(1, len(emotions)):
        pos = i * top_k - 0.5
        ax.axvline(pos, color="white", linewidth=0.8, alpha=0.7)
        ax.axhline(pos, color="white", linewidth=0.8, alpha=0.7)

    plt.colorbar(im, ax=ax, label="|cosine similarity|")
    ax.set_title(f"Layer {layer}: |cosine similarity| between local PC axes")
    fig.tight_layout()
    fig.savefig(out_dir / f"layer{layer}_cosine_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved cosine heatmap.")


def plot_dendrogram(
    Z: np.ndarray,
    labels: list[str],
    emotions: list[str],
    layer: int,
    n_clusters: int,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 6))
    colour_map = {e: EMOTION_COLOURS.get(e, "#333333") for e in emotions}

    def leaf_colour(leaf_label: str) -> str:
        emotion = leaf_label.rsplit("_PC", 1)[0]
        return colour_map.get(emotion, "#333333")

    ddata = dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
        color_threshold=0,
        above_threshold_color="#aaaaaa",
    )
    # Colour leaf labels by emotion
    for lbl_text in ax.get_xticklabels():
        txt = lbl_text.get_text()
        lbl_text.set_color(leaf_colour(txt))

    ax.set_title(f"Layer {layer}: Hierarchical clustering of local PC axes (Ward, |cos| distance)")
    ax.set_ylabel("Angular distance (Ward)")

    # Legend
    handles = [
        plt.Line2D([0], [0], color=EMOTION_COLOURS.get(e, "#333333"), linewidth=4, label=e)
        for e in emotions
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_dir / f"layer{layer}_dendrogram.png", dpi=150)
    plt.close(fig)
    logger.info("Saved dendrogram.")


def plot_meta_pca_scatter(
    proj_df: pd.DataFrame,
    emotions: list[str],
    layer: int,
    out_dir: Path,
) -> None:
    """2-D scatter of local PCs projected onto meta-PC1/2.

    Colour encodes PC rank (PC1–PC5); marker shape encodes emotion.
    """
    if "meta_PC1" not in proj_df.columns or "meta_PC2" not in proj_df.columns:
        return

    top_k = int(proj_df["pc_num"].max())

    # Colour palette for PC rank
    pc_cmap = plt.get_cmap("tab10")
    PC_COLOURS = {k: pc_cmap(k - 1) for k in range(1, top_k + 1)}

    # Marker cycle for emotions
    MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
    emotion_marker = {e: MARKERS[i % len(MARKERS)] for i, e in enumerate(emotions)}

    fig, ax = plt.subplots(figsize=(8, 7))

    for emotion in emotions:
        sub = proj_df[proj_df["emotion"] == emotion]
        marker = emotion_marker[emotion]
        for _, row in sub.iterrows():
            pc = int(row["pc_num"])
            ax.scatter(
                row["meta_PC1"], row["meta_PC2"],
                color=PC_COLOURS[pc], marker=marker,
                s=80, alpha=0.88, zorder=3,
                edgecolors="white", linewidths=0.4,
            )
            ax.annotate(
                emotion[:3],
                (row["meta_PC1"], row["meta_PC2"]),
                fontsize=5, alpha=0.65,
                xytext=(3, 3), textcoords="offset points",
            )

    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)
    ax.set_xlabel("Meta-PC 1")
    ax.set_ylabel("Meta-PC 2")
    ax.set_title(f"Layer {layer}: Local PCs projected onto meta-PCA space\n"
                 "(colour = PC rank, shape = emotion)")

    # Legend: PC colours
    pc_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PC_COLOURS[k], markersize=8, label=f"PC{k}")
        for k in range(1, top_k + 1)
    ]
    # Legend: emotion shapes
    emo_handles = [
        plt.Line2D([0], [0], marker=emotion_marker[e], color="w",
                   markerfacecolor="#555555", markersize=8, label=e)
        for e in emotions
    ]
    # Place both legends side-by-side in the upper-left using a combined handler
    all_handles = (
        [plt.Line2D([0], [0], linestyle="none", label="— PC rank —")]
        + pc_handles
        + [plt.Line2D([0], [0], linestyle="none", label="— Emotion —")]
        + emo_handles
    )
    ax.legend(handles=all_handles, fontsize=7, loc="upper left",
              framealpha=0.85, ncol=2, columnspacing=1.0,
              handletextpad=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / f"layer{layer}_meta_pca_scatter.png", dpi=150)
    plt.close(fig)
    logger.info("Saved meta-PCA scatter.")

def plot_cluster_composition(
    clust_df: pd.DataFrame,
    emotions: list[str],
    layer: int,
    n_clusters: int,
    out_dir: Path,
) -> None:
    """Stacked bar chart: how many axes from each emotion land in each cluster."""
    pivot = (
        clust_df.groupby(["cluster", "emotion"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=emotions, fill_value=0)
    )
    n_clust = pivot.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n_clust * 0.9), 5))
    bottom = np.zeros(n_clust)
    for emo in emotions:
        vals = pivot[emo].values.astype(float)
        ax.bar(
            pivot.index, vals, bottom=bottom,
            label=emo, color=EMOTION_COLOURS.get(emo, "#333333"),
        )
        bottom += vals
    ax.set_xticks(pivot.index)
    ax.set_xticklabels([f"Cluster {c}" for c in pivot.index])
    ax.set_ylabel("Number of local PC axes")
    ax.set_title(f"Layer {layer}: Emotion composition per cluster (K={n_clusters})")
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_dir / f"layer{layer}_cluster_composition_k{n_clusters}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved cluster composition bar chart.")


# ---------------------------------------------------------------------------
# Cross-emotion sharing summary
# ---------------------------------------------------------------------------

def summarise_cross_emotion_sharing(
    cos_sim: np.ndarray,
    labels: list[str],
    emotions: list[str],
    top_k: int,
    layer: int,
    out_dir: Path,
) -> None:
    """For each local PC, find its nearest neighbour from a *different* emotion."""
    M = len(labels)
    rows = []
    abs_sim = np.abs(cos_sim)
    np.fill_diagonal(abs_sim, -1.0)   # exclude self

    label_emotions = [lbl.rsplit("_PC", 1)[0] for lbl in labels]

    for i, label in enumerate(labels):
        em_i = label_emotions[i]
        sims = abs_sim[i].copy()
        # Mask same-emotion axes
        for j in range(M):
            if label_emotions[j] == em_i:
                sims[j] = -1.0
        best_cross_j = int(np.argmax(sims))
        best_cross_sim = float(sims[best_cross_j])

        # Also best same-emotion (excluding self)
        sims_same = abs_sim[i].copy()
        for j in range(M):
            if label_emotions[j] != em_i:
                sims_same[j] = -1.0
        best_same_j   = int(np.argmax(sims_same))
        best_same_sim = float(sims_same[best_same_j]) if np.any(sims_same >= 0) else float("nan")

        rows.append({
            "label": label,
            "emotion": em_i,
            "pc_num": int(label.rsplit("_PC", 1)[1]),
            "best_cross_label": labels[best_cross_j],
            "best_cross_emotion": label_emotions[best_cross_j],
            "best_cross_abs_cos": best_cross_sim,
            "best_same_label": labels[best_same_j] if not np.isnan(best_same_sim) else "",
            "best_same_abs_cos": best_same_sim,
            "sharing_score": (
                best_cross_sim / best_same_sim
                if (not np.isnan(best_same_sim) and best_same_sim > 0)
                else float("nan")
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"layer{layer}_cross_emotion_sharing.csv", index=False)

    # Print summary
    print(f"\n=== Exp 6 – Layer {layer}: Cross-emotion sharing ===")
    print(df[[
        "label", "best_cross_label", "best_cross_abs_cos",
        "best_same_abs_cos", "sharing_score",
    ]].to_string(index=False))

    # Identify axes with sharing_score > 0.5 as strong candidates for shared meta-axes
    strong = df[df["sharing_score"] > 0.5].sort_values("best_cross_abs_cos", ascending=False)
    if len(strong):
        print(f"\n  → {len(strong)} axes with sharing_score > 0.5 (strong cross-emotion alignment):")
        for _, r in strong.iterrows():
            print(f"     {r['label']:20s}  ↔  {r['best_cross_label']:20s}  |cos|={r['best_cross_abs_cos']:.3f}")
    else:
        print("\n  → No axes with sharing_score > 0.5  (emotion-specific structure dominates)")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--emo-act",
        default=str(ACT_DIR / "emotion_intensity_residual_stream.npy"),
    )
    p.add_argument(
        "--emo-info",
        default=str(ACT_DIR / "emotion_intensity_residual_stream_info.json"),
    )
    p.add_argument(
        "--neu-act",
        default=str(ACT_DIR / "neutral_paraphrase_residual_stream.npy"),
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
        "--top-k", type=int, default=5,
        help="Number of top PCs to extract per emotion (default: 5)",
    )
    p.add_argument(
        "--n-meta", type=int, default=10,
        help="Number of meta-PCA components (default: 10)",
    )
    p.add_argument(
        "--n-clusters", type=int, default=8,
        help="Number of hierarchical clusters (default: 8)",
    )
    p.add_argument(
        "--out-dir", default=str(OUT_DIR),
    )
    p.add_argument(
        "--replot-only", action="store_true",
        help=(
            "Skip all heavy computation and regenerate plots only. "
            "Requires existing CSV/NPY outputs in --out-dir."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fast path: regenerate plots from existing saved data
    # ------------------------------------------------------------------
    if args.replot_only:
        logger.info("--replot-only: loading saved data from %s …", out_dir)
        with open(out_dir / f"layer{args.layer}_local_pcs_labels.json") as f:
            meta = json.load(f)
        emotions_rp: list[str] = meta["emotions"]
        labels_rp: list[str] = meta["labels"]

        cos_sim_rp = np.load(out_dir / f"layer{args.layer}_cosine_sim.npy").astype(np.float64)
        proj_df_rp = pd.read_csv(out_dir / f"layer{args.layer}_meta_pca_projections.csv")
        clust_df_rp = pd.read_csv(out_dir / f"layer{args.layer}_clusters_k{args.n_clusters}.csv")

        angular_dist_rp = np.clip(1.0 - np.abs(cos_sim_rp), 0.0, 1.0)
        np.fill_diagonal(angular_dist_rp, 0.0)
        Z_rp = linkage(squareform(angular_dist_rp, checks=False), method="ward")

        logger.info("Regenerating plots …")
        plot_cosine_heatmap(cos_sim_rp, labels_rp, emotions_rp, args.layer, out_dir)
        plot_dendrogram(Z_rp, labels_rp, emotions_rp, args.layer, args.n_clusters, out_dir)
        plot_meta_pca_scatter(proj_df_rp, emotions_rp, args.layer, out_dir)
        plot_cluster_composition(clust_df_rp, emotions_rp, args.layer, args.n_clusters, out_dir)
        logger.info("Done (replot only).")
        return

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
    emo_shape = tuple(info["shape"])
    emo_raw = np.memmap(args.emo_act, dtype=info["dtype"], mode="r", shape=emo_shape)

    neu_info_path = args.neu_act.replace(".npy", "_info.json")
    with open(neu_info_path) as f:
        neu_info = json.load(f)
    neu_shape = tuple(neu_info["shape"])
    neu_raw = np.memmap(args.neu_act, dtype=neu_info["dtype"], mode="r", shape=neu_shape)

    # --- CAA directions ---
    caa_npz = np.load(args.caa_directions)
    caa_pooled: np.ndarray = caa_npz["caa_pooled"]
    caa: np.ndarray = caa_npz["caa"]

    # --- residuals ---
    logger.info("Building residual deltas for layer %d …", args.layer)
    delta_resid = build_per_emotion_residuals(
        emo_raw, neu_raw, caa_pooled, caa,
        layer_idx=layer_idx,
        intensity_low_idx=intensity_low_idx,
        intensity_high_idx=intensity_high_idx,
    )
    # delta_resid: (N, E, D)

    # --- extract local PC vectors ---
    logger.info("Extracting top-%d local PCs per emotion …", args.top_k)
    U, labels = extract_local_pcs(delta_resid, emotions, top_k=args.top_k, layer=args.layer)
    # U: (E*K, D)

    # Save PC vectors for downstream use
    np.save(out_dir / f"layer{args.layer}_local_pcs.npy", U.astype(np.float32))
    with open(out_dir / f"layer{args.layer}_local_pcs_labels.json", "w") as f:
        json.dump({"labels": labels, "emotions": emotions,
                   "top_k": args.top_k, "layer": args.layer}, f, indent=2)
    logger.info("Saved %d PC vectors.", len(labels))

    # --- cosine similarity matrix ---
    logger.info("Computing cosine similarity matrix …")
    cos_sim = cosine_similarity_matrix(U)
    np.save(out_dir / f"layer{args.layer}_cosine_sim.npy", cos_sim.astype(np.float32))

    cos_df = pd.DataFrame(cos_sim, index=labels, columns=labels)
    cos_df.to_csv(out_dir / f"layer{args.layer}_cosine_sim.csv")

    # --- cross-emotion sharing summary ---
    summarise_cross_emotion_sharing(
        cos_sim, labels, emotions, args.top_k, args.layer, out_dir
    )

    # --- meta-PCA ---
    logger.info("Running meta-PCA …")
    proj_df = meta_pca(U, labels, emotions, n_meta=args.n_meta, layer=args.layer, out_dir=out_dir)

    # --- hierarchical clustering ---
    logger.info("Running hierarchical clustering (K=%d) …", args.n_clusters)
    clust_df, Z, angular_dist = hierarchical_clustering(
        cos_sim, labels, layer=args.layer,
        n_clusters=args.n_clusters, out_dir=out_dir,
    )

    # Print cluster membership
    print(f"\n=== Cluster assignments (K={args.n_clusters}) ===")
    for c in sorted(clust_df["cluster"].unique()):
        members = clust_df[clust_df["cluster"] == c]["label"].tolist()
        # Check if cross-emotion
        member_emotions = set(m.rsplit("_PC", 1)[0] for m in members)
        tag = "SHARED" if len(member_emotions) > 1 else "single-emotion"
        print(f"  Cluster {c:2d} [{tag:13s}]: {', '.join(members)}")

    # --- plots ---
    logger.info("Generating plots …")
    plot_cosine_heatmap(cos_sim, labels, emotions, args.layer, out_dir)
    plot_dendrogram(Z, labels, emotions, args.layer, args.n_clusters, out_dir)
    plot_meta_pca_scatter(proj_df, emotions, args.layer, out_dir)
    plot_cluster_composition(clust_df, emotions, args.layer, args.n_clusters, out_dir)

    logger.info("All outputs written to %s", out_dir)
    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
