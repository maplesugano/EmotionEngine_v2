"""
Experiment 13: Decomposition validity — source-paired cosine geometry and linear probe.

Validates the g / r̂_e decomposition via three complementary tests:

1. Source-paired cosine similarity BEFORE (c_e) vs AFTER (r̂_e) residualization.
   Shows that r̂_e has lower inter-emotion similarity → more discriminative directions.

2. Linear probe: logistic regression classifies emotion from r̂_e more accurately
   than from c_e, confirming r̂_e captures emotion-specific structure.

3. α_G sweep from Exp 12: α_G > 0 does not improve target-emotion match,
   confirming that g carries no emotion-specific information.

Outputs
-------
results/cosine_matrix_before_L13.csv
results/cosine_matrix_after_L13.csv
results/probe_accuracy.csv
thesis/figures/caa_decomposition_cosine_before_after_L13.pdf  — 2-panel heatmap
thesis/figures/caa_decomposition_validity_L13.pdf             — 3-panel combined figure

RAM requirements: ~2.5 GB (activation slice + feature matrices).

Usage
-----
    cd /home/maplesugano/proj/EmotionEngine_v2
    python local_axes_experiments/exp13_decomposition_validity/exp13_decomposition_validity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
ACT_DIR   = ROOT / "activation" / "emotion_rewrites"
EMO_NPY   = ACT_DIR / "emotion_intensity_residual_stream.npy"
EMO_INFO  = ACT_DIR / "emotion_intensity_residual_stream_info.json"
NEU_NPY   = ACT_DIR / "neutral_paraphrase_residual_stream.npy"
NEU_INFO  = ACT_DIR / "neutral_paraphrase_residual_stream_info.json"
CAA_PATH  = ACT_DIR / "caa_emotion_directions.npz"
CAA_INFO  = ACT_DIR / "caa_emotion_directions_info.json"
EXP12_CSV = (ROOT / "local_axes_experiments" / "exp12_alpha_g_sweep"
             / "results" / "alpha_g_sweep_summary.csv")
OUT_DIR   = Path(__file__).resolve().parent / "results"
FIG_DIR   = ROOT / "thesis" / "figures"
ANA_DIR   = ROOT / "analysis" / "caa" / "geometry"

PRIMARY_LAYER    = 13
MAX_SOURCES      = 2500
SEED             = 0
INTENSITY_PAIRS  = [(0, 1), (0, 2), (1, 2)]
EMOTIONS_ORDERED = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
]

plt.rcParams.update({"font.family": "serif", "font.size": 10})


# ── helpers ────────────────────────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    """Unit-normalize along the last axis."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-12)


# ── data loading ───────────────────────────────────────────────────────────────

def load_shared_direction() -> np.ndarray:
    """
    Compute g = unit( mean_e unit(c_e) ) from pooled CAA at PRIMARY_LAYER.
    Matches the decomposition used in the main CAA experiment and Exp 12.
    """
    with open(CAA_INFO) as f:
        meta = json.load(f)
    li   = meta["layer_indices"].index(PRIMARY_LAYER)
    data = np.load(CAA_PATH)
    C    = unit(np.asarray(data["caa_pooled"][:, li, :], dtype=np.float32))  # [8, D]
    return unit(C.mean(axis=0))  # [D]


def load_delta_slice() -> tuple[np.ndarray, list[str]]:
    """
    Load delta[n, e, i, D] = h_emo[n,e,i,layer] - h_neu[n,layer]
    for a random subset of MAX_SOURCES sources (memory-mapped, then copied).

    Returns
    -------
    delta : (N_SUB, 8, 3, D) float32
    emotion_order : list of 8 emotion names
    """
    with open(EMO_INFO) as f: em = json.load(f)
    with open(NEU_INFO) as f: nu = json.load(f)

    EMO_SHAPE     = tuple(em["shape"])        # (N, 8, 3, 6, D)
    NEU_SHAPE     = tuple(nu["shape"])        # (N, 6, D)
    emotion_order = em["emotion_order"]
    li            = em["layer_indices"].index(PRIMARY_LAYER)
    N_TOTAL       = EMO_SHAPE[0]

    rng = np.random.default_rng(SEED)
    idx = (np.sort(rng.choice(N_TOTAL, size=MAX_SOURCES, replace=False))
           if N_TOTAL > MAX_SOURCES else np.arange(N_TOTAL))
    print(f"Sources: {len(idx)}/{N_TOTAL}  |  layer: {PRIMARY_LAYER} (index {li})")

    H_emo = np.memmap(str(EMO_NPY), dtype="float32", mode="r", shape=EMO_SHAPE)
    H_neu = np.memmap(str(NEU_NPY), dtype="float32", mode="r", shape=NEU_SHAPE)
    delta = np.asarray(H_emo[idx, :, :, li, :], dtype=np.float32)   # (N_SUB, 8, 3, D)
    neu   = np.asarray(H_neu[idx, li, :],        dtype=np.float32)   # (N_SUB, D)
    delta -= neu[:, np.newaxis, np.newaxis, :]
    del H_emo, H_neu, neu
    return delta, emotion_order


# ── cosine matrix ──────────────────────────────────────────────────────────────

def cosine_matrix(delta: np.ndarray, g: np.ndarray | None = None) -> np.ndarray:
    """
    Source-paired (8×8) cosine similarity matrix.

    Off-diagonal (e1 ≠ e2):
        mean_n  cos( unit(δ̄_{n,e1}), unit(δ̄_{n,e2}) )
        δ̄_{n,e} = mean over intensities of δ_{n,e,i}

    Diagonal [e, e]:
        mean_n  mean_{i1<i2}  cos( unit(δ_{n,e,i1}), unit(δ_{n,e,i2}) )
        (within-emotion intensity consistency)

    If g is given, each δ is residualized before unit-norming:
        δ → δ − (δ·g) g
    """
    N_SUB, N_EMO, N_INT, D = delta.shape

    # ── off-diagonal: pool intensities ───────────────────────────────────────
    d_pool = delta.mean(axis=2)                          # (N_SUB, 8, D)
    if g is not None:
        proj   = (d_pool @ g)[:, :, np.newaxis]          # (N_SUB, 8, 1)
        d_pool = d_pool - proj * g                        # residualize (new array)
    d_pool_u = unit(d_pool)                              # (N_SUB, 8, D)
    mat = np.einsum("ned,nfd->ef", d_pool_u, d_pool_u) / N_SUB   # (8, 8)
    del d_pool, d_pool_u

    # ── diagonal: per-intensity, per-emotion to save memory ──────────────────
    for e in range(N_EMO):
        d_e = delta[:, e, :, :].copy()                  # (N_SUB, 3, D)
        if g is not None:
            proj_e = (d_e @ g)[:, :, np.newaxis]        # (N_SUB, 3, 1)
            d_e    = d_e - proj_e * g
        d_e_u = unit(d_e)                               # (N_SUB, 3, D)
        mat[e, e] = float(np.mean([
            np.einsum("nd,nd->n", d_e_u[:, i1], d_e_u[:, i2]).mean()
            for i1, i2 in INTENSITY_PAIRS
        ]))
        del d_e, d_e_u

    return mat


def reorder_matrix(mat: np.ndarray, src_order: list[str]) -> np.ndarray:
    idx = [src_order.index(e) for e in EMOTIONS_ORDERED]
    return mat[np.ix_(idx, idx)]


# ── Plutchik organization ──────────────────────────────────────────────────────

def wheel_distance(e1: str, e2: str) -> int:
    """Circular distance on Plutchik wheel (1=adjacent, 4=opposite)."""
    i = EMOTIONS_ORDERED.index(e1)
    j = EMOTIONS_ORDERED.index(e2)
    d = abs(i - j)
    return min(d, 8 - d)


def plutchik_by_distance(mat: np.ndarray) -> dict:
    """Return {distance: [cosine values]} for all off-diagonal pairs."""
    result: dict = {d: [] for d in range(1, 5)}
    n = mat.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = wheel_distance(EMOTIONS_ORDERED[i], EMOTIONS_ORDERED[j])
            result[d].append(float(mat[i, j]))
    return result


def save_plutchik_figure(mat_b: np.ndarray, mat_a: np.ndarray, path: Path) -> None:
    dist_b = plutchik_by_distance(mat_b)
    dist_a = plutchik_by_distance(mat_a)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    rng = np.random.default_rng(1)
    x_ticks = [1, 2, 3, 4]
    x_labels = ["Adjacent\n(dist 1)", "Dist 2", "Dist 3", "Opposite\n(dist 4)"]
    specs = [
        (r"Before  ($\mathbf{c}_e$)",     dist_b, "#2166ac", -0.07),
        (r"After  ($\hat{\mathbf{r}}_e$)", dist_a, "#d6604d",  0.07),
    ]

    for label, dist_dict, color, x_off in specs:
        means = [np.mean(dist_dict[d]) for d in x_ticks]
        ax.plot(x_ticks, means, "o-", color=color, linewidth=2.0,
                markersize=7, label=label, zorder=5)
        for d in x_ticks:
            vals = dist_dict[d]
            jitter = rng.uniform(-0.05, 0.05, size=len(vals))
            ax.scatter(
                [d + x_off + jitter[k] for k in range(len(vals))],
                vals, color=color, alpha=0.25, s=18, zorder=3,
            )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8.5)
    ax.set_xlabel("Plutchik wheel distance", fontsize=9)
    ax.set_ylabel("Source-paired cosine similarity", fontsize=9)
    ax.set_title(
        "Plutchik organization: cosine similarity by wheel distance (Layer 13)\n"
        r"Before vs after $\mathbf{g}$ residualization",
        fontsize=9,
    )
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _mds_circle_order(mat: np.ndarray) -> list:
    """
    Fit 1D MDS on cosine distance matrix, return list of emotion indices
    sorted by the embedding (defines circular placement order).
    """
    from sklearn.manifold import MDS
    dist = np.clip(1.0 - mat, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    mds = MDS(n_components=1, dissimilarity="precomputed",
              random_state=0, n_init=20, max_iter=500)
    embedding = mds.fit_transform(dist).ravel()
    return list(np.argsort(embedding))


def _best_circular_alignment(order: list, reference: list) -> list:
    """
    Rotate `order` (a circular permutation of 0..n-1) to minimise the
    sum of circular distances to `reference` (Plutchik order = 0..n-1).
    Also tries the reversed order (reflection).
    Returns the best-aligned version of `order`.
    """
    n = len(order)
    best, best_cost = order, float("inf")
    for candidate in [order, list(reversed(order))]:
        for rot in range(n):
            rotated = candidate[rot:] + candidate[:rot]
            cost = sum(min(abs(rotated[k] - k), n - abs(rotated[k] - k))
                       for k in range(n))
            if cost < best_cost:
                best_cost = cost
                best = rotated
    return best


def save_mds_circle_figure(mat_b: np.ndarray, mat_a: np.ndarray, path: Path) -> None:
    """
    Two-panel circular chart where emotion nodes are placed by 1D MDS order
    (data-driven) rather than the fixed Plutchik order.
    Edge style reflects Plutchik adjacency; edge color encodes cosine value.
    """
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors

    PLUTCHIK_ADJACENT = {frozenset([i, (i + 1) % 8]) for i in range(8)}
    PLUTCHIK_OPPOSITE = {frozenset([i, (i + 4) % 8]) for i in range(4)}
    PLUTCHIK_REF = list(range(8))   # EMOTIONS_ORDERED is already Plutchik order

    cmap = plt.cm.RdYlBu_r
    vmin, vmax = 0.70, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2))
    fig.subplots_adjust(left=0.02, right=0.87, top=0.93, bottom=0.04, wspace=0.08)

    for ax, mat, title in [
        (axes[0], mat_b, r"Before  ($\mathbf{c}_e$)"),
        (axes[1], mat_a, r"After  ($\hat{\mathbf{r}}_e$)"),
    ]:
        raw_order = _mds_circle_order(mat)
        order = _best_circular_alignment(raw_order, PLUTCHIK_REF)

        n = 8
        # position k on circle → emotion index order[k]
        angles = [np.pi / 2 - 2 * np.pi * k / n for k in range(n)]
        angle_of = {order[k]: angles[k] for k in range(n)}   # emotion → angle
        pos = np.array([(np.cos(angle_of[i]), np.sin(angle_of[i])) for i in range(n)])

        ax.set_aspect("equal")
        ax.set_xlim(-1.65, 1.65)
        ax.set_ylim(-1.80, 1.65)
        ax.axis("off")

        # Draw all 28 edges as bezier arcs
        for i in range(n):
            for j in range(i + 1, n):
                v = float(mat[i, j])
                color = cmap(norm(v))
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                mx = (x1 + x2) / 2 * 0.35
                my = (y1 + y2) / 2 * 0.35
                verts = [(x1, y1), (mx, my), (x2, y2)]
                codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
                pair = frozenset([i, j])
                is_adj = pair in PLUTCHIK_ADJACENT
                is_opp = pair in PLUTCHIK_OPPOSITE
                lw    = 2.8 if is_adj else (1.8 if is_opp else 0.8)
                alpha = 0.90 if is_adj else (0.70 if is_opp else 0.22)
                ax.add_patch(PathPatch(MPath(verts, codes), facecolor="none",
                                       edgecolor=color, linewidth=lw, alpha=alpha))

        # Nodes and labels
        for i in range(n):
            x, y = pos[i]
            mds_rank = order.index(i)
            plutchik_rank = i
            # Check if emotion is in the expected Plutchik slot (±1 tolerance)
            circ_dist = min(abs(mds_rank - plutchik_rank),
                            n - abs(mds_rank - plutchik_rank))
            node_color = "#d6604d" if circ_dist > 1 else "white"

            ax.scatter(x, y, s=260, zorder=6, color=node_color,
                       edgecolors="#333333", linewidth=1.5)
            a = angle_of[i]
            lx, ly = 1.28 * np.cos(a), 1.28 * np.sin(a)
            ha = "left" if lx > 0.15 else ("right" if lx < -0.15 else "center")
            ax.text(lx, ly, EMOTIONS_ORDERED[i].capitalize(),
                    ha=ha, va="center", fontsize=8.5)

        # MDS order label at bottom
        order_str = " → ".join(EMOTIONS_ORDERED[k].capitalize()[:3] for k in order)
        ax.text(0, -1.68, f"MDS order: {order_str}", ha="center", va="top",
                fontsize=6.2, style="italic", color="#444444")

        ax.set_title(title, fontsize=10.5, pad=14)

    # Shared colorbar — placed in a reserved strip outside the two panels
    cbar_ax = fig.add_axes([0.89, 0.22, 0.016, 0.55])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Source-paired cosine similarity", fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=8)

    # Legend
    legend_elems = [
        Line2D([0], [0], color="gray", lw=2.8, alpha=0.9,  label="Plutchik adjacent (dist 1)"),
        Line2D([0], [0], color="gray", lw=1.8, alpha=0.70, label="Plutchik opposite (dist 4)"),
        Line2D([0], [0], color="gray", lw=0.8, alpha=0.30, label="Other pairs"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d6604d",
               markeredgecolor="#333", markersize=9, label="Out of Plutchik order (>1 slot)"),
    ]
    axes[1].legend(handles=legend_elems, loc="lower right", fontsize=7.2, framealpha=0.85)

    fig.suptitle(
        "Emotion directions placed by 1D MDS on cosine distance (Layer 13)\n"
        r"Node position = data-driven order  |"
        r"  Red node = deviates from Plutchik  |"
        r"  Edge style = Plutchik adjacency",
        fontsize=9.2,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_plutchik_circle_figure(mat_b: np.ndarray, mat_a: np.ndarray, path: Path) -> None:
    """2-panel circular chord diagram: emotions on Plutchik wheel, edges colored by cosine."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors

    n = 8
    # Start at top (joy), go clockwise
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    pos = np.array([(np.cos(a), np.sin(a)) for a in angles])

    cmap = plt.cm.RdYlBu_r      # warm = high cosine, cool = low
    vmin, vmax = 0.70, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5))
    fig.subplots_adjust(left=0.02, right=0.87, top=0.92, bottom=0.02, wspace=0.08)

    for ax, mat, title in [
        (axes[0], mat_b, r"Before  ($\mathbf{c}_e$)"),
        (axes[1], mat_a, r"After  ($\hat{\mathbf{r}}_e$)"),
    ]:
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.axis("off")

        # Draw all 28 pairwise edges as inward-curving bezier arcs
        for i in range(n):
            for j in range(i + 1, n):
                v = float(mat[i, j])
                color = cmap(norm(v))
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                # Control point: midpoint pulled toward center
                mx, my = (x1 + x2) / 2 * 0.35, (y1 + y2) / 2 * 0.35
                verts = [(x1, y1), (mx, my), (x2, y2)]
                codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
                d = wheel_distance(EMOTIONS_ORDERED[i], EMOTIONS_ORDERED[j])
                lw    = 2.8 if d == 1 else (2.0 if d == 4 else 0.9)
                alpha = 0.90 if d == 1 else (0.75 if d == 4 else 0.30)
                patch = PathPatch(MPath(verts, codes), facecolor="none",
                                  edgecolor=color, linewidth=lw, alpha=alpha)
                ax.add_patch(patch)

        # Nodes on top of edges
        for i, (x, y) in enumerate(pos):
            ax.scatter(x, y, s=230, zorder=6, color="white",
                       edgecolors="#333333", linewidth=1.5)
            # Labels just outside the circle
            lx = 1.25 * np.cos(angles[i])
            ly = 1.25 * np.sin(angles[i])
            ha = "left" if lx > 0.15 else ("right" if lx < -0.15 else "center")
            ax.text(lx, ly, EMOTIONS_ORDERED[i].capitalize(),
                    ha=ha, va="center", fontsize=8.5)

        ax.set_title(title, fontsize=10.5, pad=14)

    # Shared colorbar — placed in a reserved strip outside the two panels
    cbar_ax = fig.add_axes([0.89, 0.22, 0.016, 0.55])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Source-paired cosine similarity", fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=8)

    # Edge-style legend
    legend_elems = [
        Line2D([0], [0], color="gray", lw=2.8, alpha=0.9,  label="Adjacent (dist 1)"),
        Line2D([0], [0], color="gray", lw=2.0, alpha=0.75, label="Opposite (dist 4)"),
        Line2D([0], [0], color="gray", lw=0.9, alpha=0.35, label="Other pairs"),
    ]
    axes[1].legend(handles=legend_elems, loc="lower right",
                   fontsize=7.5, framealpha=0.85)

    fig.suptitle(
        "Plutchik emotion wheel — source-paired cosine similarity (Layer 13)\n"
        r"Edge color = cosine  |  Thick = adjacent (dist 1)  |  Medium = opposite (dist 4)",
        fontsize=9.5,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── linear probe ──────────────────────────────────────────────────────────────

def run_linear_probe(delta: np.ndarray, g: np.ndarray) -> dict:
    """
    8-class logistic-regression linear probe.

    For each source n and emotion e, the feature is:
        before: unit( δ̄_{n,e} )
        after:  unit( δ̄_{n,e} − (δ̄_{n,e}·g)g )

    Train/test split by source (80/20) to prevent leakage across a source's
    8 emotion variants.

    Returns dict with 'before', 'after', 'chance' accuracy values.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    N_SUB, N_EMO, N_INT, D = delta.shape
    n_train = int(0.8 * N_SUB)

    d_pool = delta.mean(axis=2)                           # (N_SUB, 8, D)
    proj   = (d_pool @ g)[:, :, np.newaxis]              # (N_SUB, 8, 1)
    d_resid = d_pool - proj * g                          # (N_SUB, 8, D)

    feat_b = unit(d_pool)                                # before: (N_SUB, 8, D)
    feat_a = unit(d_resid)                               # after:  (N_SUB, 8, D)
    del d_pool, d_resid

    # Labels: emotion index for each (source, emotion) row
    y = np.tile(np.arange(N_EMO), N_SUB)                 # (N_SUB*8,)

    # Row index arrays grouped by source to avoid cross-source leakage
    train_rows = np.concatenate(
        [np.arange(s * N_EMO, (s + 1) * N_EMO) for s in range(n_train)]
    )
    test_rows = np.concatenate(
        [np.arange(s * N_EMO, (s + 1) * N_EMO) for s in range(n_train, N_SUB)]
    )

    results: dict = {"chance": 1.0 / N_EMO}
    for tag, feat in [("before", feat_b), ("after", feat_a)]:
        X_flat = feat.reshape(N_SUB * N_EMO, D)
        X_tr, X_te = X_flat[train_rows], X_flat[test_rows]
        y_tr, y_te = y[train_rows],      y[test_rows]

        clf = LogisticRegression(
            C=1.0, max_iter=300, solver="lbfgs",
            n_jobs=-1, random_state=0,
        )
        clf.fit(X_tr, y_tr)
        acc = float(accuracy_score(y_te, clf.predict(X_te)))
        results[tag] = acc
        print(f"  {tag:8s}: acc = {acc:.4f}  ({100*acc:.1f}%)")

    return results


# ── plotting ───────────────────────────────────────────────────────────────────

def _draw_heatmap(ax, mat: np.ndarray, cmap: str, vmin: float, vmax: float,
                  title: str) -> object:
    """Render a cosine heatmap with annotated values; return the AxesImage."""
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    n = mat.shape[0]
    labels = [e.capitalize() for e in EMOTIONS_ORDERED]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7.5)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            txt = f"[{v:.3f}]" if i == j else f"{v:.3f}"
            norm = (v - vmin) / max(vmax - vmin, 1e-9)
            color = "white" if norm > 0.65 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.5, color=color)
    ax.set_title(title, fontsize=9, pad=4)
    return im


def save_heatmap_pair(mat_b: np.ndarray, mat_a: np.ndarray, path: Path) -> None:
    """2-panel figure: before (Blues) | after (RdBu_r), exp11 format."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.8))

    im1 = _draw_heatmap(
        ax1, mat_b, "Blues", vmin=0.85, vmax=1.0,
        title=r"Before residualization  ($\mathbf{c}_e$, pooled CAA)",
    )
    im2 = _draw_heatmap(
        ax2, mat_a, "RdBu_r", vmin=-0.3, vmax=1.0,
        title=r"After residualization  ($\hat{\mathbf{r}}_e$)",
    )

    for ax, im in [(ax1, im1), (ax2, im2)]:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Cosine similarity", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Source-paired cosine similarity of CAA emotion directions (Layer 13)\n"
        r"Off-diagonal: inter-emotion  |  Diagonal $[\cdot]$: intra-emotion intensity consistency",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_combined_figure(
    mat_b: np.ndarray, mat_a: np.ndarray,
    df_ag: pd.DataFrame, path: Path,
) -> None:
    """3-panel combined decomposition-validity figure for the thesis."""
    fig = plt.figure(figsize=(18.2, 5.8))
    gs  = fig.add_gridspec(1, 3, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    im1 = _draw_heatmap(
        ax1, mat_b, "Blues", vmin=0.85, vmax=1.0,
        title=r"(a) Before: $\mathbf{c}_e$",
    )
    im2 = _draw_heatmap(
        ax2, mat_a, "RdBu_r", vmin=-0.3, vmax=1.0,
        title=r"(b) After: $\hat{\mathbf{r}}_e$",
    )

    for ax, im in [(ax1, im1), (ax2, im2)]:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Cosine sim.", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)

    # ── Panel (c): α_G sweep target-emotion match ─────────────────────────────
    ag = df_ag["alpha_g"].values
    tm = df_ag["tm_mean"].values
    ci = df_ag["tm_ci"].values   # already ±95% CI in the CSV

    ax3.fill_between(ag, tm - ci, tm + ci, alpha=0.20, color="#2166ac")
    ax3.plot(ag, tm, "o-", color="#2166ac", linewidth=1.8, markersize=5,
             label="Target-emotion match ± 95% CI")
    ax3.axvline(0, color="grey", linestyle="--", linewidth=0.9, alpha=0.7,
                label=r"Residual-only ($\alpha_G = 0$)")
    ax3.set_xlabel(r"$\alpha_G$  (shared direction coefficient)", fontsize=9)
    ax3.set_ylabel("Target-emotion match", fontsize=9)
    ax3.set_title(
        r"(c) $\mathbf{g}$ does not improve emotion match" + "\n"
        r"($\alpha_R = 5.0$ fixed,  $n = 100$ sources)",
        fontsize=9,
    )
    ax3.xaxis.set_major_locator(mticker.FixedLocator(list(ag)))
    ax3.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax3.tick_params(axis="x", labelsize=8)
    ax3.grid(axis="y", linestyle=":", alpha=0.5)
    ax3.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        r"Decomposition validity: $\mathbf{g}$ carries no emotion-specific information (Layer 13)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    for d in [OUT_DIR, FIG_DIR, ANA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── load ─────────────────────────────────────────────────────────────────
    print("Loading shared direction g ...")
    g = load_shared_direction()
    print(f"  ||g|| = {np.linalg.norm(g):.6f}  (should be ≈1.0)")

    print("\nLoading activation slice ...")
    delta, emotion_order = load_delta_slice()
    print(f"  delta shape: {delta.shape}  dtype: {delta.dtype}")

    # ── cosine matrices ───────────────────────────────────────────────────────
    print("\nComputing cosine matrix (before) ...")
    mat_b_raw = cosine_matrix(delta, g=None)

    print("Computing cosine matrix (after) ...")
    mat_a_raw = cosine_matrix(delta, g=g)

    mat_b = reorder_matrix(mat_b_raw, emotion_order)
    mat_a = reorder_matrix(mat_a_raw, emotion_order)

    off_diag_mask = ~np.eye(8, dtype=bool)
    print(f"\n  Before off-diagonal mean: {mat_b[off_diag_mask].mean():.4f}")
    print(f"  After  off-diagonal mean: {mat_a[off_diag_mask].mean():.4f}")
    print(f"\n  Before diagonal mean (intensity consistency): {np.diag(mat_b).mean():.4f}")
    print(f"  After  diagonal mean (intensity consistency): {np.diag(mat_a).mean():.4f}")

    df_b = pd.DataFrame(mat_b, index=EMOTIONS_ORDERED, columns=EMOTIONS_ORDERED)
    df_a = pd.DataFrame(mat_a, index=EMOTIONS_ORDERED, columns=EMOTIONS_ORDERED)

    df_b.to_csv(OUT_DIR / "cosine_matrix_before_L13.csv")
    df_a.to_csv(OUT_DIR / "cosine_matrix_after_L13.csv")
    df_a.to_csv(ANA_DIR / "cosine_matrix_after_L13.csv")
    print(f"  Saved CSVs → {OUT_DIR}")

    # ── linear probe ──────────────────────────────────────────────────────────
    print("\nRunning linear probe ...")
    print(f"  (8-class LR, {int(0.8 * delta.shape[0])} train / {delta.shape[0] - int(0.8 * delta.shape[0])} test sources)")
    probe = run_linear_probe(delta, g)
    pd.DataFrame([probe]).to_csv(OUT_DIR / "probe_accuracy.csv", index=False)
    print(f"  chance level: {probe['chance']:.4f} ({100*probe['chance']:.1f}%)")
    print(f"  Saved → {OUT_DIR / 'probe_accuracy.csv'}")

    del delta   # free ~1 GB before plotting

    # ── figures ───────────────────────────────────────────────────────────────
    df_ag = pd.read_csv(EXP12_CSV)

    print("\nSaving 2-panel heatmap ...")
    save_heatmap_pair(mat_b, mat_a,
                      FIG_DIR / "caa_decomposition_cosine_before_after_L13.pdf")

    print("Saving 3-panel combined figure ...")
    save_combined_figure(mat_b, mat_a, df_ag,
                         FIG_DIR / "caa_decomposition_validity_L13.pdf")

    print("Saving Plutchik organization figure ...")
    save_plutchik_figure(mat_b, mat_a,
                         FIG_DIR / "caa_decomposition_plutchik_L13.pdf")

    print("Saving Plutchik circle figure ...")
    save_plutchik_circle_figure(mat_b, mat_a,
                                FIG_DIR / "caa_decomposition_plutchik_circle_L13.pdf")

    print("Saving MDS circle figure ...")
    save_mds_circle_figure(mat_b, mat_a,
                           FIG_DIR / "caa_decomposition_mds_circle_L13.pdf")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — Decomposition Validity (Layer 13)")
    print("=" * 60)
    print(f"  Source-paired inter-emotion cosine (off-diagonal)")
    print(f"    before: {mat_b[off_diag_mask].mean():.4f}  "
          f"(range {mat_b[off_diag_mask].min():.3f}–{mat_b[off_diag_mask].max():.3f})")
    print(f"    after:  {mat_a[off_diag_mask].mean():.4f}  "
          f"(range {mat_a[off_diag_mask].min():.3f}–{mat_a[off_diag_mask].max():.3f})")
    print(f"  Intensity consistency (diagonal)")
    print(f"    before: {np.diag(mat_b).mean():.4f}")
    print(f"    after:  {np.diag(mat_a).mean():.4f}")
    print(f"  Linear probe accuracy")
    print(f"    chance: {probe['chance']:.4f}")
    print(f"    before: {probe['before']:.4f}")
    print(f"    after:  {probe['after']:.4f}")
    print(f"  Plutchik organization (cosine by wheel distance)")
    dist_labels = {1: "Adjacent (dist 1)", 2: "Dist 2         ", 3: "Dist 3         ", 4: "Opposite (dist 4)"}
    dist_b = plutchik_by_distance(mat_b)
    dist_a = plutchik_by_distance(mat_a)
    for d in [1, 2, 3, 4]:
        mb = np.mean(dist_b[d])
        ma = np.mean(dist_a[d])
        print(f"    {dist_labels[d]}: before={mb:.4f}  after={ma:.4f}")
    gap_b = np.mean(dist_b[1]) - np.mean(dist_b[4])
    gap_a = np.mean(dist_a[1]) - np.mean(dist_a[4])
    print(f"    Gap (adj − opp)        : before={gap_b:.4f}  after={gap_a:.4f}  ({gap_a/gap_b:.1f}× larger)")
    print("=" * 60)


if __name__ == "__main__":
    main()
