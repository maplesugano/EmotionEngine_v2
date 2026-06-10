"""Publication-quality figures for Experiment 6.1 (pooled PCA validation).

Figures saved to results_pooled/figures/:
  fig1_scree_subspace.{pdf,png}   – pooled EVR scree + subspace alignment
  fig2_universal_pc1.{pdf,png}    – universal PC1 alignment (method + per-emotion)
  fig3_pooled_mk_heatmap.{pdf,png} – |cos| heatmap: pooled PCs vs thesis meta-axes m_k

Run:
    python plot_thesis_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
POOLED_DIR = HERE / "results_pooled"
META_DIR   = HERE / "results"
EXP7_DIR   = HERE.parent / "exp7_meta_axis_extraction" / "results"
OUT_DIR    = POOLED_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
LAYER = 13

# ── Style ──────────────────────────────────────────────────────────────────────
C_BLUE       = "#2171b5"
C_ORANGE     = "#d94801"
C_LIGHT_BLUE = "#9ecae1"
C_LIGHT_ORG  = "#fdae6b"
C_GREEN      = "#238b45"
C_GRAY       = "#737373"

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "savefig.dpi":        300,
    "text.usetex":        False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "lines.linewidth":    1.2,
    "patch.linewidth":    0.8,
})


# ── Load data ──────────────────────────────────────────────────────────────────
evr_df    = pd.read_csv(POOLED_DIR / f"layer{LAYER}_pooled_evr.csv")
pooled_evr = evr_df["evr"].values * 100  # → percent

pooled_comp = np.load(POOLED_DIR / f"layer{LAYER}_pooled_pcs.npy").astype(np.float64)

meta_df   = pd.read_csv(META_DIR / f"layer{LAYER}_meta_pca_components.csv", index_col=0)
meta_comp = meta_df.values.astype(np.float64)          # (10, D) – exp6 meta-PCA

local_pcs = np.load(META_DIR / f"layer{LAYER}_local_pcs.npy").astype(np.float64)
with open(META_DIR / f"layer{LAYER}_local_pcs_labels.json") as fh:
    local_labels = json.load(fh)["labels"]

mk = np.load(EXP7_DIR / f"layer{LAYER}_meta_axes.npy").astype(np.float64)  # (5, D)


# ── Helpers ────────────────────────────────────────────────────────────────────
def abs_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return np.abs(An @ Bn.T)


def energy_capture(pooled: np.ndarray, meta: np.ndarray, r: int) -> float:
    """Fraction of meta top-r energy captured by pooled top-r span."""
    Qp, _ = np.linalg.qr(pooled[:r].T)
    return float(np.mean([np.linalg.norm(Qp.T @ meta[i]) ** 2 for i in range(r)]))


def panel_label(ax, label: str, x=0.97, y=0.97) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold")


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight")
    print(f"  Saved  {name}.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1  –  EVR scree (a)  +  subspace energy capture (b)
# ══════════════════════════════════════════════════════════════════════════════
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(6.4, 2.75))
fig1.subplots_adjust(wspace=0.40)

# (a) Pooled PCA scree
n_show = 5
xs = np.arange(1, n_show + 1)
bar_colors = [C_BLUE if i == 0 else C_LIGHT_BLUE for i in range(n_show)]
bars = ax1a.bar(xs, pooled_evr[:n_show], color=bar_colors, width=0.55, zorder=3)
ax1a.set_xticks(xs)
ax1a.set_xticklabels([f"PC{i}" for i in xs])
ax1a.set_xlabel("Pooled PC")
ax1a.set_ylabel("Explained variance (%)")
ax1a.set_ylim(0, pooled_evr[0] * 1.30)
ax1a.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
ax1a.set_axisbelow(True)
for bar, val in zip(bars, pooled_evr[:n_show]):
    ax1a.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.12,
        f"{val:.1f}%",
        ha="center", va="bottom", fontsize=7,
    )
# panel_label(ax1a, "(a)")

# (b) Subspace energy capture for top-r (pooled vs exp6 meta-PCA)
r_vals = [1, 2, 3]
energy = [energy_capture(pooled_comp, meta_comp, r) for r in r_vals]
shade  = [C_LIGHT_BLUE, C_BLUE, "#084594"]
ax1b.bar(r_vals, energy, color=shade, width=0.5, zorder=3)
ax1b.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=0.8, alpha=0.5)
ax1b.set_xticks(r_vals)
ax1b.set_xticklabels([f"$r={r}$" for r in r_vals])
ax1b.set_xlabel("Subspace dimension $r$")
ax1b.set_ylabel("Energy capture")
ax1b.set_ylim(0, 1.15)
ax1b.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
ax1b.set_axisbelow(True)
for r, e in zip(r_vals, energy):
    ax1b.text(r, e + 0.015, f"{e:.3f}", ha="center", va="bottom", fontsize=8)
# panel_label(ax1b, "(b)")

save(fig1, "fig1_scree_subspace")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2  –  Universal PC1 comparison (a)  +  per-emotion breakdown (b)
# ══════════════════════════════════════════════════════════════════════════════

# Universal PC1 = sign-normalised mean of per-emotion PC1 vectors
pc1_idx   = [i for i, lbl in enumerate(local_labels) if lbl.endswith("_PC1")]
pc1_vecs  = local_pcs[pc1_idx].copy()
pc1_vecs /= np.linalg.norm(pc1_vecs, axis=1, keepdims=True)
universal = pc1_vecs.mean(0)
universal /= np.linalg.norm(universal)

emotion_names = [local_labels[i].split("_PC1")[0].capitalize() for i in pc1_idx]

cos_methods = [
    ("Pooled PC1",  float(abs(pooled_comp[0] @ universal)), C_BLUE),
    ("Meta-PC1",    float(abs(meta_comp[0]   @ universal)), C_ORANGE),
    ("Meta-PC2",    float(abs(meta_comp[1]   @ universal)), C_LIGHT_ORG),
]
per_em_cos = np.abs(pc1_vecs @ pooled_comp[0])   # (E,)
sort_idx   = np.argsort(per_em_cos)               # ascending for barh

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(6.4, 2.75))
fig2.subplots_adjust(wspace=0.50)

# (a) Method comparison  (horizontal bars)
m_names  = [m[0] for m in cos_methods[::-1]]
m_vals   = [m[1] for m in cos_methods[::-1]]
m_colors = [m[2] for m in cos_methods[::-1]]
hbars = ax2a.barh(m_names, m_vals, color=m_colors, height=0.45, zorder=3)
ax2a.set_xlim(0, 1.12)
ax2a.axvline(1.0, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.5)
ax2a.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
ax2a.set_axisbelow(True)
ax2a.set_xlabel("|cos| with universal PC1")
for bar, val in zip(hbars, m_vals):
    ax2a.text(
        val + 0.015, bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}", va="center", fontsize=8,
    )
# panel_label(ax2a, "(a)", x=0.97, y=0.08)

# (b) Per-emotion breakdown of pooled PC1
em_sorted  = [emotion_names[i] for i in sort_idx]
cos_sorted = per_em_cos[sort_idx]
ec = [C_BLUE if v >= 0.85 else C_LIGHT_BLUE for v in cos_sorted]
ax2b.barh(em_sorted, cos_sorted, color=ec, height=0.6, zorder=3)
ax2b.set_xlim(0, 1.10)
ax2b.axvline(0.85, color=C_GREEN, linestyle="--", linewidth=0.8, alpha=0.6,
             label="0.85 threshold")
ax2b.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
ax2b.set_axisbelow(True)
ax2b.set_xlabel("|cos|  pooled PC1 ↔ emotion PC1")
for i, val in enumerate(cos_sorted):
    ax2b.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=7.5)
ax2b.legend(loc="lower right", frameon=False, fontsize=7)
# panel_label(ax2b, "(b)", x=0.97, y=0.08)

save(fig2, "fig2_universal_pc1")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3  –  |cos| heatmap: pooled PCs vs thesis meta-axes m_k
# ══════════════════════════════════════════════════════════════════════════════
K = min(5, mk.shape[0])
P = min(5, pooled_comp.shape[0])
C_mat = abs_cos(pooled_comp[:P], mk[:K])   # (P, K)

fig3, ax3 = plt.subplots(figsize=(4.2, 3.3))
im = ax3.imshow(C_mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")

ax3.set_xticks(range(K))
ax3.set_xticklabels([f"$\\mathbf{{m}}_{{{k+1}}}$" for k in range(K)], fontsize=9)
ax3.set_yticks(range(P))
ax3.set_yticklabels([f"pooled PC{i+1}" for i in range(P)], fontsize=8.5)
ax3.set_xlabel("Thesis meta-axis", fontsize=9)

for i in range(P):
    for j in range(K):
        val = C_mat[i, j]
        txt_col = "white" if val > 0.55 else "black"
        weight  = "bold" if i == j else "normal"
        ax3.text(j, i, f"{val:.2f}",
                 ha="center", va="center",
                 fontsize=8.5, color=txt_col, fontweight=weight)

cbar = plt.colorbar(im, ax=ax3, fraction=0.034, pad=0.04)
cbar.set_label("|cosine|", fontsize=8)
cbar.ax.tick_params(labelsize=7.5)
fig3.tight_layout()

save(fig3, "fig3_pooled_mk_heatmap")

plt.close("all")
print(f"\nDone. All figures in: {OUT_DIR}")
