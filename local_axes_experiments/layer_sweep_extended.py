"""Compute headline layer-diagnostic metrics for layers 8, 19, 22.

Appends new rows to analysis/affective_subspace_coverage/layer_comparison_headline.csv
using the same procedure and source subsample as the affective_subspace_coverage notebook.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.extmath import randomized_svd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.steering_utils import unit

ACT_DIR  = REPO_ROOT / "activation" / "emotion_rewrites"
NPY_PATH = ACT_DIR / "emotion_intensity_residual_stream.npy"
INFO_PATH = ACT_DIR / "emotion_intensity_residual_stream_info.json"
CSV_PATH = REPO_ROOT / "analysis" / "affective_subspace_coverage" / "layer_comparison_headline.csv"

NEW_LAYERS   = [8, 19, 22]
K_PCS        = 4
MAX_SOURCES  = 2500
CHUNK        = 256
SEED         = 0

# ---------------------------------------------------------------------------

def build_global_basis(R: np.ndarray, int_pos: dict, k: int = K_PCS) -> dict:
    C    = R.mean(axis=(0, 2))                   # (N_EMO, D)
    mu_c = C.mean(axis=0)
    Cc   = (C - mu_c).astype(np.float64)
    _, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    evr  = (S ** 2) / (S ** 2).sum()
    PCs  = Vt[:k].astype(np.float32)
    hi   = R[:, :, int_pos["high"], :].mean(axis=(0, 1))
    lo   = R[:, :, int_pos["low"],  :].mean(axis=(0, 1))
    v_int = unit((hi - lo).astype(np.float32))
    v_int_orth = unit(v_int - PCs.T @ (PCs @ v_int))
    return dict(C=C, evr=evr, PCs=PCs, v_int=v_int, v_int_orth=v_int_orth)


def cv_bacc(X: np.ndarray, y: np.ndarray, grp: np.ndarray, C: float = 0.1) -> float:
    gkf   = GroupKFold(n_splits=min(3, len(np.unique(grp))))
    clf   = LogisticRegression(C=C, max_iter=1000, solver="saga", n_jobs=-1)
    preds = np.zeros_like(y)
    for tr, te in gkf.split(X, y, grp):
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    return float(balanced_accuracy_score(y, preds))


def run_layer(layer: int, ACT, SRC_IDX, LI_MAP, N_SUB, N_EMO, N_INT, D,
              int_pos, y_emo, y_int, groups) -> dict:
    t0 = time.time()
    li = LI_MAP[layer]
    X  = np.asarray(ACT[SRC_IDX, :, :, li, :], dtype=np.float32)

    # v_int before in-place centering
    hi_raw = X[:, :, int_pos["high"], :].mean(axis=(0, 1))
    lo_raw = X[:, :, int_pos["low"],  :].mean(axis=(0, 1))
    v_int_raw = unit((hi_raw - lo_raw).astype(np.float32))
    del hi_raw, lo_raw

    mu = X.mean(axis=1, keepdims=True)
    X -= mu
    del mu
    gc.collect()
    R = X; X = None

    basis  = build_global_basis(R, int_pos, k=K_PCS)
    basis["v_int"]      = v_int_raw
    basis["v_int_orth"] = unit(v_int_raw - basis["PCs"].T @ (basis["PCs"] @ v_int_raw))
    del v_int_raw
    gc.collect()

    PCs = basis["PCs"]
    vo  = basis["v_int_orth"]
    gm  = R.reshape(-1, D).mean(axis=0)

    # reconstruction EVR
    tv = vp4 = voi = 0.0
    for s0 in range(0, N_SUB, CHUNK):
        s1  = min(s0 + CHUNK, N_SUB)
        ch  = (R[s0:s1].reshape(-1, D) - gm).astype(np.float32)
        tv  += float((ch ** 2).sum())
        vp4 += float(((ch @ PCs.T) ** 2).sum())
        voi += float(((ch @ vo) ** 2).sum())
        del ch
    evr4 = vp4 / tv
    evr5 = (vp4 + voi) / tv
    cent = float(basis["evr"][:K_PCS].sum())

    # emotion probe
    X_pc4 = np.empty((N_SUB * N_EMO * N_INT, K_PCS), dtype=np.float32)
    for s0 in range(0, N_SUB, CHUNK):
        s1   = min(s0 + CHUNK, N_SUB)
        ch   = (R[s0:s1].reshape(-1, D) - gm).astype(np.float32)
        rows = slice(s0 * N_EMO * N_INT, s1 * N_EMO * N_INT)
        X_pc4[rows] = ch @ PCs.T
        del ch
    bacc_emo = cv_bacc(X_pc4, y_emo, groups)
    bacc_int = cv_bacc(X_pc4, y_int, groups)

    # local PC1 × intensity Spearman ρ
    rho_list = []
    int_tile = np.tile(np.arange(N_INT), N_SUB)
    for e_idx in range(N_EMO):
        C_e  = basis["C"][e_idx]
        R_e  = R[:, e_idx, :, :].reshape(-1, D) - C_e
        _, _, Vt_e = randomized_svd(R_e.astype(np.float64), n_components=2,
                                    random_state=SEED, n_iter=4)
        pc1  = Vt_e[0].astype(np.float32)
        rho  = float(stats.spearmanr(R_e @ pc1, int_tile).statistic)
        rho_list.append(abs(rho))
        del R_e
    mean_rho = float(np.mean(rho_list))

    elapsed = time.time() - t0
    print(f"  Layer {layer}: centroid_evr4={cent:.4f}  emo_bacc={bacc_emo:.4f}"
          f"  mean_rho={mean_rho:.4f}  ({elapsed:.0f}s)")

    del R, X_pc4, basis
    gc.collect()

    return {
        "layer":                   layer,
        "centroid_evr4":           round(cent,     4),
        "indiv_evr_pc4":           round(evr4,     4),
        "indiv_evr_pc4_vint":      round(evr5,     4),
        "emo_probe_bacc_pc4":      round(bacc_emo, 4),
        "int_probe_bacc_pc4":      round(bacc_int, 4),
        "mean_local_pc1_int_rho":  round(mean_rho, 4),
    }


def main() -> None:
    with open(INFO_PATH) as f:
        info = json.load(f)

    layer_indices  = info["layer_indices"]
    emotion_order  = info["emotion_order"]
    intensity_order = info["intensity_order"]
    shape = tuple(info["shape"])
    dtype = info.get("dtype", "float32")
    N, N_EMO, N_INT, N_LAY, D = shape

    LI_MAP  = {l: i for i, l in enumerate(layer_indices)}
    int_pos = {lvl: i for i, lvl in enumerate(intensity_order)}

    rng     = np.random.default_rng(SEED)
    SRC_IDX = np.sort(rng.choice(N, size=MAX_SOURCES, replace=False)) if N > MAX_SOURCES else np.arange(N)
    N_SUB   = len(SRC_IDX)

    ACT = np.memmap(str(NPY_PATH), dtype=dtype, mode="r", shape=shape)

    y_emo  = np.tile(np.repeat(np.arange(N_EMO), N_INT), N_SUB)
    y_int  = np.tile(np.arange(N_INT), N_SUB * N_EMO)
    groups = np.repeat(np.arange(N_SUB), N_EMO * N_INT)

    existing = pd.read_csv(CSV_PATH, index_col="layer")
    already_done = set(existing.index.tolist())
    to_run = [l for l in NEW_LAYERS if l not in already_done]

    if not to_run:
        print("All layers already present in CSV — nothing to do.")
        return

    print(f"Running layers: {to_run}")
    new_rows = []
    for layer in to_run:
        if layer not in LI_MAP:
            print(f"  Layer {layer} not in activation file — skipping.")
            continue
        row = run_layer(layer, ACT, SRC_IDX, LI_MAP, N_SUB, N_EMO, N_INT, D,
                        int_pos, y_emo, y_int, groups)
        new_rows.append(row)

    if not new_rows:
        print("Nothing new to append.")
        return

    new_df  = pd.DataFrame(new_rows).set_index("layer")
    updated = pd.concat([existing, new_df]).sort_index()
    updated.to_csv(CSV_PATH)
    print(f"\nUpdated {CSV_PATH}")
    print(updated.to_string())


if __name__ == "__main__":
    main()
