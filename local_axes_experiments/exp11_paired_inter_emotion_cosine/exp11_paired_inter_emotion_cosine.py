"""
Compute source-paired inter-emotion cosine similarity matrix at layer 13.

Off-diagonal (e1, e2):
    mean_n  cos( unit(delta_n_e1), unit(delta_n_e2) )
    where delta_n_e = mean over intensities of (h_emo[n,e,i] - h_neu[n])

Diagonal (e, e):
    mean_n  mean_{intensity pairs}  cos( unit(delta_n_e_i1), unit(delta_n_e_i2) )
    within-emotion intensity consistency, preserving source pairing.

Outputs
-------
local_axes_experiments/exp11_paired_inter_emotion_cosine/results/caa_paired_cosine_matrix_L13.csv
analysis/caa/geometry/caa_paired_cosine_matrix_L13.csv   — also copied here for thesis pipeline
thesis/figures/caa_paired_inter_emotion_cosine_heatmap_L13.pdf

Usage
-----
    cd /home/maplesugano/proj/EmotionEngine_v2
    python local_axes_experiments/exp11_paired_inter_emotion_cosine/exp11_paired_inter_emotion_cosine.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
ACT_DIR     = ROOT / "activation" / "emotion_rewrites"
EMO_NPY     = ACT_DIR / "emotion_intensity_residual_stream.npy"
EMO_INFO    = ACT_DIR / "emotion_intensity_residual_stream_info.json"
NEU_NPY     = ACT_DIR / "neutral_paraphrase_residual_stream.npy"
NEU_INFO    = ACT_DIR / "neutral_paraphrase_residual_stream_info.json"
OUT_RESULTS = Path(__file__).resolve().parent / "results"
OUT_CSV     = ROOT / "analysis" / "caa" / "geometry" / "caa_paired_cosine_matrix_L13.csv"
OUT_FIG     = ROOT / "thesis" / "figures" / "caa_paired_inter_emotion_cosine_heatmap_L13.pdf"

PRIMARY_LAYER = 13
MAX_SOURCES   = 2500   # RAM cap (same as caa_per_source_structure.ipynb)
SEED          = 0

plt.rcParams.update({"font.family": "serif", "font.size": 10})

# ── load metadata ──────────────────────────────────────────────────────────────
with open(EMO_INFO)  as f: emo_meta  = json.load(f)
with open(NEU_INFO)  as f: neu_meta  = json.load(f)

EMOTION_ORDER   = emo_meta["emotion_order"]
INTENSITY_ORDER = emo_meta["intensity_order"]
LAYER_INDICES   = emo_meta["layer_indices"]
LI_MAP          = {l: i for i, l in enumerate(LAYER_INDICES)}
li              = LI_MAP[PRIMARY_LAYER]

EMO_SHAPE = tuple(emo_meta["shape"])   # (N, 8, 3, 6, D)
NEU_SHAPE = tuple(neu_meta["shape"])   # (N, 6, D)
N_TOTAL, N_EMO, N_INT, N_LAY, D = EMO_SHAPE

rng     = np.random.default_rng(SEED)
src_idx = (np.sort(rng.choice(N_TOTAL, size=MAX_SOURCES, replace=False))
           if N_TOTAL > MAX_SOURCES else np.arange(N_TOTAL))
N_SUB   = len(src_idx)
print(f"Using {N_SUB}/{N_TOTAL} sources, layer {PRIMARY_LAYER} (index {li}), D={D}")

# ── load activations (memory-mapped) ──────────────────────────────────────────
print("Memory-mapping activations ...")
H_emo = np.memmap(str(EMO_NPY), dtype="float32", mode="r", shape=EMO_SHAPE)
H_neu = np.memmap(str(NEU_NPY), dtype="float32", mode="r", shape=NEU_SHAPE)

# ── load slice into RAM ────────────────────────────────────────────────────────
print("Loading slice into RAM ...")
# delta[n, e, i] = h_emo[n, e, i, layer] - h_neu[n, layer]
delta = np.asarray(H_emo[src_idx, :, :, li, :], dtype=np.float32)   # (N_SUB, 8, 3, D)
neu   = np.asarray(H_neu[src_idx, li, :],        dtype=np.float32)   # (N_SUB, D)
delta -= neu[:, np.newaxis, np.newaxis, :]   # in-place, no copy
del H_emo, H_neu, neu

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-12)

# ── off-diagonal: paired inter-emotion cosine ──────────────────────────────────
# pool intensities first, then unit-normalise per source
delta_pooled = delta.mean(axis=2)           # (N_SUB, 8, D)
delta_u      = unit(delta_pooled)           # (N_SUB, 8, D)

# mat[e1, e2] = mean_n  dot(delta_u[n,e1], delta_u[n,e2])
# einsum: sum over n and d, leave e1 and e2
mat = np.einsum("ned,nfd->ef", delta_u, delta_u) / N_SUB   # (8, 8)

# ── diagonal: within-emotion intensity consistency (source-paired) ─────────────
delta_int_u = unit(delta)   # (N_SUB, 8, 3, D)
intensity_pairs = [(0, 1), (0, 2), (1, 2)]

for e in range(N_EMO):
    d_e = delta_int_u[:, e, :, :]   # (N_SUB, 3, D)
    pair_cos = [
        float(np.einsum("nd,nd->n", d_e[:, i1, :], d_e[:, i2, :]).mean())
        for i1, i2 in intensity_pairs
    ]
    mat[e, e] = np.mean(pair_cos)

del delta, delta_pooled, delta_u, delta_int_u

print("Paired cosine matrix:")
df_mat = pd.DataFrame(mat, index=EMOTION_ORDER, columns=EMOTION_ORDER)
print(df_mat.round(4).to_string())

# ── save CSV ───────────────────────────────────────────────────────────────────
OUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_mat.to_csv(OUT_RESULTS / "caa_paired_cosine_matrix_L13.csv")
df_mat.to_csv(OUT_CSV)
print(f"\nSaved: {OUT_RESULTS / 'caa_paired_cosine_matrix_L13.csv'}")
print(f"Saved: {OUT_CSV}")

# ── plot heatmap ───────────────────────────────────────────────────────────────
emotions_order = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
mat_ordered = df_mat.reindex(index=emotions_order, columns=emotions_order).values

fig, ax = plt.subplots(figsize=(6.2, 5.4))
cmap = plt.get_cmap("Blues").copy()
im = ax.imshow(mat_ordered, aspect="equal", cmap=cmap, vmin=0.0, vmax=1.0)

n = len(emotions_order)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
labels = [e.capitalize() for e in emotions_order]
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
ax.set_yticklabels(labels, fontsize=8.5)

for i in range(n):
    for j in range(n):
        val = mat_ordered[i, j]
        # diagonal gets a different marker to distinguish it
        fmt = f"{val:.3f}" if i != j else f"[{val:.3f}]"
        ax.text(j, i, fmt, ha="center", va="center",
                fontsize=6.0, color="white" if val > 0.6 else "black")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Cosine similarity", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title(
    "Source-paired cosine similarity of CAA directions (Layer 13)\n"
    "Off-diagonal: inter-emotion  |  Diagonal []: intra-emotion intensity consistency",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(OUT_FIG, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved: {OUT_FIG}")
