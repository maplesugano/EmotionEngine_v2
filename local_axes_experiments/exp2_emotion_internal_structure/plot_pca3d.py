"""3-D PCA scatter of per-source residual shifts (layer 13).

Projects all emotion residuals onto the first three principal components of
the globally-pooled (N*E, D) matrix so every emotion lives in the same
linear coordinate system.  The tight-vs-diffuse gradient predicted by the
participation ratios should be visible directly as cloud compactness.

Usage
-----
python plot_pca3d.py                   # defaults: layer 13, 800 pts/emotion
python plot_pca3d.py --n-per-emotion 400 --out custom_path.png
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR   = REPO_ROOT / "activation" / "emotion_rewrites"
FIG_DIR   = REPO_ROOT / "thesis" / "figures"

EMOTION_COLORS = {
    "joy":           "#F5C518",
    "trust":         "#2E8B57",
    "fear":          "#9B59B6",
    "surprise":      "#E67E22",
    "sadness":       "#2980B9",
    "disgust":       "#922B21",
    "anger":         "#E74C3C",
    "anticipation":  "#17A589",
}


# ---------------------------------------------------------------------------
# Helpers (mirror of exp2 — kept local to avoid import-path gymnastics)
# ---------------------------------------------------------------------------

def _unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _project_out(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    return X - np.outer(X @ g, g)


def build_residuals(
    emo_act: np.ndarray,
    neu_act: np.ndarray,
    caa_pooled: np.ndarray,
    caa: np.ndarray,
    layer_idx: int,
    low_idx: int,
    high_idx: int,
    chunk: int = 512,
) -> np.ndarray:
    """Return (N, E, D) residuals with g and u_e projected out."""
    N, E, I, L, D = emo_act.shape
    g = _unit_norm(caa_pooled[:, layer_idx, :].mean(axis=0).astype(np.float64)).astype(np.float32)
    u_e = _unit_norm(
        caa[:, high_idx, layer_idx, :].astype(np.float64)
        - caa[:, low_idx,  layer_idx, :].astype(np.float64)
    ).astype(np.float32)  # (E, D)

    out = np.empty((N, E, D), dtype=np.float32)
    for s in range(0, N, chunk):
        e_ = min(s + chunk, N)
        h_emo = emo_act[s:e_, :, :, layer_idx, :].mean(axis=2).astype(np.float32)
        h_neu = neu_act[s:e_, layer_idx, :].astype(np.float32)
        delta = h_emo - h_neu[:, np.newaxis, :]
        for ei in range(E):
            d = _project_out(delta[:, ei, :], g)
            d = _project_out(d, u_e[ei])
            out[s:e_, ei, :] = d
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer",         type=int,   default=13)
    p.add_argument("--n-per-emotion", type=int,   default=800,
                   help="Samples per emotion to plot (default 800)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--elev",          type=float, default=20,
                   help="Elevation angle of 3-D view")
    p.add_argument("--azim",          type=float, default=45,
                   help="Azimuth angle of 3-D view")
    p.add_argument("--out",           default=str(FIG_DIR / "pca3d_residuals.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # --- load metadata -------------------------------------------------------
    log.info("Loading metadata …")
    with open(ACT_DIR / "emotion_intensity_residual_stream_info.json") as f:
        info = json.load(f)
    emotions: list[str] = info["emotion_order"]
    layer_idx = info["layer_indices"].index(args.layer)

    with open(ACT_DIR / "neutral_paraphrase_residual_stream_info.json") as f:
        neu_info = json.load(f)

    # --- memory-map activations ----------------------------------------------
    log.info("Mapping activations …")
    emo_raw = np.memmap(ACT_DIR / "emotion_intensity_residual_stream.npy",
                        dtype=info["dtype"], mode="r", shape=tuple(info["shape"]))
    neu_raw = np.memmap(ACT_DIR / "neutral_paraphrase_residual_stream.npy",
                        dtype=neu_info["dtype"], mode="r", shape=tuple(neu_info["shape"]))

    caa_npz   = np.load(ACT_DIR / "caa_emotion_directions.npz")
    low_idx   = info["intensity_order"].index("low")
    high_idx  = info["intensity_order"].index("high")

    # --- build residuals -----------------------------------------------------
    log.info("Building residuals for layer %d …", args.layer)
    delta = build_residuals(emo_raw, neu_raw,
                            caa_npz["caa_pooled"], caa_npz["caa"],
                            layer_idx, low_idx, high_idx)
    delta -= delta.mean(axis=1, keepdims=True)   # remove per-source mean
    N, E, D = delta.shape

    # --- subsample and stack -------------------------------------------------
    n = min(args.n_per_emotion, N)
    idx = rng.choice(N, size=n, replace=False)

    X_list, lab_list = [], []
    for ei, emo in enumerate(emotions):
        X_list.append(delta[idx, ei, :].astype(np.float64))
        lab_list.extend([emo] * n)
    X = np.vstack(X_list)           # (n*E, D)
    labels = np.array(lab_list)

    # --- global PCA ----------------------------------------------------------
    log.info("Fitting 3-D PCA on %d points …", len(X))
    X_c = X - X.mean(axis=0)
    pca = PCA(n_components=3, random_state=0)
    P   = pca.fit_transform(X_c)    # (n*E, 3)
    evr = pca.explained_variance_ratio_
    log.info("EVR: PC1=%.2f%%  PC2=%.2f%%  PC3=%.2f%%",
             evr[0]*100, evr[1]*100, evr[2]*100)

    # --- plot ----------------------------------------------------------------
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)

    for emo in emotions:
        mask = labels == emo
        ax.scatter(P[mask, 0], P[mask, 1], P[mask, 2],
                   c=EMOTION_COLORS.get(emo, "#888888"),
                   label=emo, alpha=0.30, s=5, linewidths=0)

    ax.set_xlabel(f"PC1 ({evr[0]:.1%})", fontsize=9, labelpad=6)
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})", fontsize=9, labelpad=6)
    ax.set_zlabel(f"PC3 ({evr[2]:.1%})", fontsize=9, labelpad=6)
    ax.tick_params(labelsize=7)

    ax.set_title(
        f"3-D PCA of per-source residual shifts — layer {args.layer}\n"
        f"({n} samples per emotion, global projection)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8, markerscale=4, framealpha=0.7)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    log.info("Saved → %s", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
