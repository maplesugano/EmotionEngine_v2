"""Exp 6.1 robustness figures.

Generates:
  1. A two-panel figure (EVR bar chart + |cos| diagonal comparison).
  2. A LaTeX table snippet.

Run:
    python exp6_1_robustness_figures.py --layer 13
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR  = Path(__file__).resolve().parent
OUT_DIR   = BASE_DIR / "results_pooled"


# ---------------------------------------------------------------------------
# Sign-aligned mean meta-axes (m1, m2, m3)
# ---------------------------------------------------------------------------

def sign_aligned_mean(vecs: np.ndarray) -> np.ndarray:
    """Iterative sign alignment → unit mean vector."""
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    ref = vecs[0].copy()
    for _ in range(20):
        signs = np.sign(vecs @ ref)
        signs[signs == 0] = 1.0
        aligned = vecs * signs[:, None]
        new_ref = aligned.mean(0)
        new_ref /= np.linalg.norm(new_ref)
        if np.allclose(new_ref, ref, atol=1e-9):
            break
        ref = new_ref
    return ref


def compute_meta_axes(U: np.ndarray, labels: list[str], n_ranks: int) -> np.ndarray:
    """Return (n_ranks, D) sign-aligned mean meta-axes."""
    axes = []
    for k in range(1, n_ranks + 1):
        idx = [i for i, l in enumerate(labels) if l.endswith(f"_PC{k}")]
        axes.append(sign_aligned_mean(U[idx]))
    return np.stack(axes)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(layer: int, out_dir: Path) -> None:
    # Load data
    evr_df    = pd.read_csv(out_dir / f"layer{layer}_pooled_evr.csv")
    pooled    = np.load(out_dir / f"layer{layer}_pooled_pcs.npy").astype(np.float64)
    U         = np.load(BASE_DIR / "results" / f"layer{layer}_local_pcs.npy").astype(np.float64)
    with open(BASE_DIR / "results" / f"layer{layer}_local_pcs_labels.json") as f:
        u_labels = json.load(f)["labels"]

    meta = compute_meta_axes(U, u_labels, n_ranks=3)

    n_show  = min(5, len(evr_df))
    evr     = evr_df["evr"].values[:n_show] * 100          # percent
    pc_idx  = np.arange(1, n_show + 1)

    cos_diag = np.array([abs(float(pooled[k] @ meta[k])) for k in range(3)])

    # ---- layout ----
    fig = plt.figure(figsize=(10, 4.2))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.38)

    ax_evr = fig.add_subplot(gs[0])
    ax_cos = fig.add_subplot(gs[1])

    BLUE    = "#3A7DBF"
    GREY    = "#AABCCE"
    GREEN   = "#2BAD63"

    # ── Panel A: Explained Variance Ratio ──────────────────────────────────
    bar_colors = [BLUE if i == 0 else GREY for i in range(n_show)]
    bars = ax_evr.bar(pc_idx, evr, color=bar_colors, edgecolor="white",
                      linewidth=0.8, width=0.55, zorder=3)
    ax_evr.set_xticks(pc_idx)
    ax_evr.set_xticklabels([f"PC{i}" for i in pc_idx], fontsize=10)
    ax_evr.set_ylabel("Explained variance (%)", fontsize=10)
    ax_evr.set_title("(a)  Pooled PCA — explained variance", fontsize=11, pad=8)
    ax_evr.set_ylim(0, evr[0] * 1.45)
    ax_evr.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax_evr.set_axisbelow(True)
    ax_evr.spines["top"].set_visible(False)
    ax_evr.spines["right"].set_visible(False)

    for bar, v in zip(bars, evr):
        ax_evr.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9,
                    color="#222222")

    ratio = evr[0] / evr[1]
    ax_evr.annotate(
        f"PC1 = {ratio:.1f}× PC2",
        xy=(1, evr[0]), xytext=(2.3, evr[0] * 1.25),
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.2),
        fontsize=9, color="#444",
    )

    # ── Panel B: |cos| diagonal ────────────────────────────────────────────
    x     = np.arange(1, 4)
    xlbls = ["PC₁ vs $\\mathbf{m}_1$",
             "PC₂ vs $\\mathbf{m}_2$",
             "PC₃ vs $\\mathbf{m}_3$"]

    ax_cos.bar(x, cos_diag, color=GREEN, edgecolor="white",
               linewidth=0.8, width=0.45, zorder=3)
    ax_cos.axhline(1.0, color="#888", linewidth=0.8, linestyle="--", zorder=2)
    ax_cos.set_xticks(x)
    ax_cos.set_xticklabels(xlbls, fontsize=10)
    ax_cos.set_ylabel("|cosine similarity|", fontsize=10)
    ax_cos.set_ylim(0.0, 1.12)
    ax_cos.set_title("(b)  Pooled PCA vs sign-aligned mean\n"
                     "         (diagonal cosine)", fontsize=11, pad=8)
    ax_cos.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax_cos.set_axisbelow(True)
    ax_cos.spines["top"].set_visible(False)
    ax_cos.spines["right"].set_visible(False)

    for xi, v in zip(x, cos_diag):
        ax_cos.text(xi, v + 0.014, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#222222")

    fig.suptitle(
        f"Layer {layer}: robustness of shared sub-axes — "
        "two independent methods converge on the same 3 directions",
        fontsize=11, y=1.03,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])

    out_path = out_dir / f"layer{layer}_robustness_figure.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

LATEX_TEMPLATE = r"""\begin{{table}}[ht]
\centering
\caption{{Robustness of shared sub-axes at layer {layer}.
  Left: explained variance of pooled PCA (the first pooled principal component
  accounts for more than twice the variance of the second).
  Right: diagonal cosine similarity between pooled principal components and the
  sign-aligned mean meta-axes $\mathbf{{m}}_k$; both methods are run
  independently on the same residuals.}}
\label{{tab:pooled_pca_robustness}}
\begin{{tabular}}{{lcc|lcc}}
\toprule
& \multicolumn{{2}}{{c|}}{{\textbf{{Explained variance}}}}
& & \multicolumn{{2}}{{c}}{{\textbf{{Convergence: $|\cos|$}}}} \\
Comp. & EVR & Cumul. & Pair & Pooled PCA & Sign-aligned mean \\
\midrule
{evr_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

def make_table(layer: int, out_dir: Path) -> None:
    evr_df = pd.read_csv(out_dir / f"layer{layer}_pooled_evr.csv")
    pooled = np.load(out_dir / f"layer{layer}_pooled_pcs.npy").astype(np.float64)
    U      = np.load(BASE_DIR / "results" / f"layer{layer}_local_pcs.npy").astype(np.float64)
    with open(BASE_DIR / "results" / f"layer{layer}_local_pcs_labels.json") as f:
        u_labels = json.load(f)["labels"]

    meta     = compute_meta_axes(U, u_labels, n_ranks=3)
    cos_diag = [abs(float(pooled[k] @ meta[k])) for k in range(3)]

    n_evr = min(5, len(evr_df))
    rows  = []
    for i in range(n_evr):
        evr_v = evr_df["evr"].values[i] * 100
        cum_v = evr_df["evr_cumsum"].values[i] * 100
        evr_str = f"PC$_{i+1}$ & {evr_v:.1f}\\% & {cum_v:.1f}\\%"
        if i < 3:
            cos_str = (f"PC$_{i+1}$ vs $\\mathbf{{m}}_{i+1}$"
                       f" & \\multicolumn{{2}}{{c}}{{{cos_diag[i]:.3f}}}")
            rows.append(f"  {evr_str} & {cos_str} \\\\")
        else:
            rows.append(f"  {evr_str} & & & \\\\")

    tex = LATEX_TEMPLATE.format(layer=layer, evr_rows="\n".join(rows))
    out_path = out_dir / f"layer{layer}_robustness_table.tex"
    out_path.write_text(tex)
    print(f"Saved LaTeX table: {out_path}")
    print("\n--- preview ---")
    print(tex)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer",   type=int, default=13)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_figure(args.layer, out_dir)
    make_table(args.layer, out_dir)


if __name__ == "__main__":
    main()
