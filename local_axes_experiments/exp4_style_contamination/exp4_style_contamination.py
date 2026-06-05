"""Experiment 4: Is the local sub-axis pre-verbal affect or style contamination?

Three complementary tests on layer-level local PCA axes:

Method A — Semantic / style probe
    Regress each per-source PC score  z_{n,e,k}  against a battery of surface
    features extracted from the emotion rewrite text.  Low R² → the axis is NOT
    explained by surface style.

    Features
    --------
    output_length       : word count of the emotion rewrite
    type_token_ratio    : unique words / total words
    mean_word_length    : average character count per word
    first_person_count  : occurrences of first-person tokens (I, I'm, I've, …)
    exclamation_count   : '!' count
    question_count      : '?' count
    sentence_count      : approximate sentence count (period / ! / ?)
    meaning_preserved   : LLM judge score from meta JSONL
    neutral_topic_pc{1..5}: projection onto top-5 PCs of neutral activations
                            (captures topic / source-level variation)

Method B — Neutral-control PCA
    δ_neutral[n] = h_neutral_paraphrase[n, L] − h_base_text[n, L]
    Run PCA on {δ_neutral[n]}_n → obtain neutral-style axes v_j.
    For each emotion PC u_{e,k} report max |cos(u_{e,k}, v_j)| over j=1..K.
    High alignment → the axis tracks paraphrase style, not emotion.

Method C — Meaning-preservation filter
    Keep only sources whose meaning_preserved_score ≥ threshold (default 0.7).
    Re-run local PCA on the filtered subset; compute |cos(u_{e,k}^full,
    u_{e,k}^filtered)| for each PC.  High alignment → the axis is robust to
    meaning drift, supporting an affective rather than semantic interpretation.

Outputs → analysis/style_contamination/layer{L}/
    method_a_style_r2.csv         R² per (emotion, pc, feature) and overall
    method_b_neutral_cos.csv      max neutral-PCA cos per (emotion, pc)
    method_c_filter_cos.csv       filtered vs full PC cos per (emotion, pc)
    summary.json                  high-level verdict per (emotion, pc)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = REPO_ROOT / "activation" / "emotion_rewrites"
BASE_ACT_DIR = REPO_ROOT / "activation" / "base_text"
OUT_BASE_DIR = REPO_ROOT / "analysis" / "style_contamination"

FIRST_PERSON_TOKENS = re.compile(
    r"\b(i|i'm|i've|i'd|i'll|i'm|i've|i'd|i'll|me|my|myself|mine)\b",
    flags=re.IGNORECASE,
)
SOCIAL_TOKENS = re.compile(
    r"\b(we|us|our|ours|ourselves|you|your|yours|yourself|yourselves|"
    r"he|she|they|him|her|them|his|hers|their|theirs)\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers shared with exp2 / exp3
# ---------------------------------------------------------------------------

def unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def project_out(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Remove component along unit vector g from each row of X."""
    coeff = X @ g
    return X - np.outer(coeff, g)


def participation_ratio(eigenvalues: np.ndarray) -> float:
    lam = eigenvalues[eigenvalues > 0]
    if lam.size == 0:
        return float("nan")
    return float(lam.sum() ** 2 / (lam ** 2).sum())


# ---------------------------------------------------------------------------
# Text feature extraction
# ---------------------------------------------------------------------------

def text_features(text: str) -> dict[str, float]:
    """Extract surface / style features from a single text string."""
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return {
            "output_length": 0.0,
            "type_token_ratio": 0.0,
            "mean_word_length": 0.0,
            "first_person_count": 0.0,
            "social_word_count": 0.0,
            "exclamation_count": 0.0,
            "question_count": 0.0,
            "sentence_count": 0.0,
        }
    ttr = len(set(w.lower() for w in words)) / n_words
    mean_wl = float(np.mean([len(w) for w in words]))
    fp = len(FIRST_PERSON_TOKENS.findall(text))
    sw = len(SOCIAL_TOKENS.findall(text))
    exc = text.count("!")
    que = text.count("?")
    sent = len(re.split(r"[.!?]+", text.strip()))
    return {
        "output_length": float(n_words),
        "type_token_ratio": float(ttr),
        "mean_word_length": mean_wl,
        "first_person_count": float(fp),
        "social_word_count": float(sw),
        "exclamation_count": float(exc),
        "question_count": float(que),
        "sentence_count": float(sent),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_meta(meta_path: Path) -> dict[str, dict]:
    """Load emotion_intensity_residual_stream_meta.jsonl → {source_id: record}."""
    records: dict[str, dict] = {}
    with open(meta_path) as f:
        for line in f:
            row = json.loads(line)
            sid = str(row["source_id"])
            intensity = str(row.get("intensity_level", ""))
            if sid not in records:
                records[sid] = {
                    "source_id": sid,
                    "base_text": row.get("base_text", ""),
                    "neutral_paraphrase": row.get("neutral_paraphrase", ""),
                    "emotion_rewrites": {},
                    "intensities": [],
                    "meaning_preserved_scores": [],
                }
            rec = records[sid]
            for emo in ["joy", "trust", "fear", "surprise",
                        "sadness", "disgust", "anger", "anticipation"]:
                key = f"{emo}_rewrite"
                rec["emotion_rewrites"].setdefault(emo, [])
                rec["emotion_rewrites"][emo].append(row.get(key, ""))
            rec["intensities"].append(intensity)
            score = row.get("meaning_preserved_score", None)
            if score is not None:
                try:
                    rec["meaning_preserved_scores"].append(float(score))
                except (ValueError, TypeError):
                    rec["meaning_preserved_scores"].append(float("nan"))
    return records


def build_per_emotion_residuals(
    emo_act: np.ndarray,
    neu_act: np.ndarray,
    caa_pooled: np.ndarray,
    caa: np.ndarray,
    layer_idx: int,
    intensity_low_idx: int = 0,
    intensity_high_idx: int = 2,
    CHUNK: int = 512,
) -> np.ndarray:
    """Return delta_resid (N, E, D) for one layer, common + intensity-projected out."""
    N, E, I, L, D = emo_act.shape

    g = unit_norm(caa_pooled[:, layer_idx, :].mean(axis=0))

    intensity_dirs = unit_norm(
        caa[:, intensity_high_idx, layer_idx, :].astype(np.float64)
        - caa[:, intensity_low_idx, layer_idx, :].astype(np.float64)
    ).astype(np.float32)

    delta_resid = np.empty((N, E, D), dtype=np.float32)

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        h_emo = emo_act[start:end, :, :, layer_idx, :].mean(axis=2).astype(np.float32)
        h_neu = neu_act[start:end, layer_idx, :].astype(np.float32)
        delta = h_emo - h_neu[:, np.newaxis, :]
        for e in range(E):
            d = project_out(delta[:, e, :], g)
            d = project_out(d, intensity_dirs[e])
            delta_resid[start:end, e, :] = d

    return delta_resid


def fit_local_pca(X_e: np.ndarray, n_components: int) -> tuple[PCA, np.ndarray]:
    """Centre and fit PCA; return (pca, scores) where scores is (N, n_comp)."""
    # Keep float32; randomized SVD avoids allocating a full N×D float64 copy
    X = X_e.astype(np.float32)
    mu = X.mean(axis=0)
    X -= mu  # in-place to avoid an extra copy
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=0, svd_solver="randomized")
    pca.fit(X)
    scores = pca.transform(X)   # (N, n_comp)
    return pca, scores


# ---------------------------------------------------------------------------
# Method A — style probe regression
# ---------------------------------------------------------------------------

def _build_style_matrix(
    source_ids: list[str],
    meta: dict[str, dict],
    emotion: str,
    neutral_topic_pcs: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build (N, F) style feature matrix for one emotion using vectorised pandas ops."""
    rewrites: list[str] = []
    mps_vals: list[float] = []
    for sid in source_ids:
        rec = meta.get(sid, {})
        emo_rewrites = rec.get("emotion_rewrites", {}).get(emotion, [])
        rewrites.append(emo_rewrites[1] if len(emo_rewrites) > 1 else (emo_rewrites[0] if emo_rewrites else ""))
        mps_list = [s for s in rec.get("meaning_preserved_scores", []) if not np.isnan(s)]
        mps_vals.append(float(np.mean(mps_list)) if mps_list else 0.0)

    texts = pd.Series(rewrites)
    words_ser = texts.str.split()

    output_length = words_ser.str.len().fillna(0).astype(float).values
    safe_len      = np.maximum(output_length, 1)
    unique_words  = words_ser.apply(lambda ws: len(set(w.lower() for w in ws)) if ws else 0).values.astype(float)
    ttr           = unique_words / safe_len
    mean_wl       = words_ser.apply(lambda ws: float(np.mean([len(w) for w in ws])) if ws else 0.0).values
    fp_count      = texts.str.count(FIRST_PERSON_TOKENS.pattern).fillna(0).values.astype(float)
    sw_count      = texts.str.count(SOCIAL_TOKENS.pattern).fillna(0).values.astype(float)
    excl          = texts.str.count(r"!").fillna(0).values.astype(float)
    ques          = texts.str.count(r"\?").fillna(0).values.astype(float)
    sent          = texts.str.count(r"[.!?]+").fillna(0).values.astype(float) + 1
    mps_arr       = np.array(mps_vals, dtype=float)

    style_names = [
        "output_length", "type_token_ratio", "mean_word_length",
        "first_person_count", "social_word_count",
        "exclamation_count", "question_count", "sentence_count",
        "meaning_preserved",
    ]
    style_mat = np.column_stack([
        output_length, ttr, mean_wl,
        fp_count, sw_count,
        excl, ques, sent,
        mps_arr,
    ])

    topic_names = [f"neutral_topic_pc{j+1}" for j in range(neutral_topic_pcs.shape[1])]
    feat_mat   = np.hstack([style_mat, neutral_topic_pcs]).astype(np.float64)
    feat_names = style_names + topic_names
    return feat_mat, feat_names


def method_a_style_probe(
    delta_resid: np.ndarray,          # (N, E, D)  demeaned
    source_ids: list[str],
    meta: dict[str, dict],
    emotions: list[str],
    n_components: int,
    neutral_topic_pcs: np.ndarray,    # (N, K_topic) pre-computed topic features
) -> pd.DataFrame:
    """Regress local PC scores against style features; return R² DataFrame."""
    rows = []

    for e_idx, emotion in enumerate(emotions):
        logger.info("  Method A — %s …", emotion)
        X_e = delta_resid[:, e_idx, :]

        feat_mat, feat_names = _build_style_matrix(source_ids, meta, emotion, neutral_topic_pcs)

        scaler = StandardScaler()
        feat_mat_sc = scaler.fit_transform(feat_mat)  # (N, F)

        pca, scores = fit_local_pca(X_e, n_components)
        n_comp_fitted = scores.shape[1]

        # Pre-centre all feature columns once for Pearson r² computation
        F_c = feat_mat_sc - feat_mat_sc.mean(axis=0)  # (N, F)
        F_norms = np.linalg.norm(F_c, axis=0) + 1e-12  # (F,)

        for k in range(n_comp_fitted):
            z = scores[:, k].astype(np.float64)

            # Overall R² via OLS (single fit, all features)
            reg = LinearRegression().fit(feat_mat_sc, z)
            r2_total = float(reg.score(feat_mat_sc, z))

            # Per-feature R² = Pearson r² (univariate OLS equivalence)
            z_c = z - z.mean()
            z_norm = float(np.linalg.norm(z_c)) + 1e-12
            r_vec = (F_c.T @ z_c) / (F_norms * z_norm)  # (F,)
            per_feat_r2 = {fname: float(r * r) for fname, r in zip(feat_names, r_vec)}

            row = {
                "emotion": emotion,
                "pc": k + 1,
                "evr": float(pca.explained_variance_ratio_[k]),
                "r2_all_features": r2_total,
            }
            row.update({f"r2_{fn}": v for fn, v in per_feat_r2.items()})
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Method B — neutral-control PCA
# ---------------------------------------------------------------------------

def method_b_neutral_control(
    delta_resid: np.ndarray,          # (N, E, D)  demeaned
    neutral_delta: np.ndarray,        # (N, D)  h_neutral - h_base
    emotions: list[str],
    n_components: int,
    n_neutral_pcs: int,
) -> pd.DataFrame:
    """Compare emotion PCs to neutral-paraphrase style PCs via cosine similarity."""
    logger.info("  Method B — fitting neutral PCA (n=%d) …", neutral_delta.shape[0])

    # PCA on neutral paraphrase deltas (captures style / topic variation)
    neu_c = (neutral_delta - neutral_delta.mean(axis=0)).astype(np.float32)
    n_neu = min(n_neutral_pcs, neu_c.shape[0] - 1, neu_c.shape[1])
    pca_neu = PCA(n_components=n_neu, random_state=0, svd_solver="randomized")
    pca_neu.fit(neu_c)
    del neu_c  # free (N, D) float32 copy
    V = pca_neu.components_.astype(np.float32)   # (n_neu, D)  neutral style axes

    rows = []
    for e_idx, emotion in enumerate(emotions):
        logger.info("  Method B — %s …", emotion)
        X_e = delta_resid[:, e_idx, :]
        pca_e, _ = fit_local_pca(X_e, n_components)
        U = pca_e.components_.astype(np.float32)   # (n_comp, D)

        for k in range(U.shape[0]):
            u = U[k]
            # Cosine similarities to all neutral PCs
            cos_vals = np.abs(V @ u)   # (n_neu,)
            rows.append({
                "emotion": emotion,
                "pc": k + 1,
                "evr": float(pca_e.explained_variance_ratio_[k]),
                "max_cos_neutral": float(cos_vals.max()),
                "mean_cos_neutral_top5": float(np.sort(cos_vals)[-5:].mean()),
                "argmax_neutral_pc": int(np.argmax(cos_vals)) + 1,
                "neutral_pca_evr_at_argmax": float(pca_neu.explained_variance_ratio_[int(np.argmax(cos_vals))]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Method C — meaning-preservation filter
# ---------------------------------------------------------------------------

def method_c_filter(
    delta_resid: np.ndarray,          # (N, E, D)  demeaned
    source_ids: list[str],
    meta: dict[str, dict],
    emotions: list[str],
    n_components: int,
    meaning_threshold: float = 0.7,
) -> pd.DataFrame:
    """Re-run PCA after dropping low meaning-preservation sources; compare axes."""
    # Compute per-source mean meaning_preserved_score
    mps_arr = np.array([
        float(np.nanmean(meta.get(sid, {}).get("meaning_preserved_scores", [float("nan")])))
        for sid in source_ids
    ])
    # Scores can be 0–10 or 0–1 depending on LLM judge format; normalise to 0–1
    if np.nanmax(mps_arr) > 2.0:
        mps_arr = mps_arr / 10.0
    keep_mask = mps_arr >= meaning_threshold
    n_keep = int(keep_mask.sum())
    n_total = len(source_ids)
    logger.info(
        "  Method C — threshold=%.2f  keeping %d / %d sources (%.1f%%)",
        meaning_threshold, n_keep, n_total, 100 * n_keep / max(n_total, 1),
    )

    rows = []
    for e_idx, emotion in enumerate(emotions):
        logger.info("  Method C — %s …", emotion)
        X_full = delta_resid[:, e_idx, :]   # view, no copy

        pca_full, _ = fit_local_pca(X_full, n_components)
        U_full = pca_full.components_.astype(np.float32)   # (n_comp, D)

        if n_keep < 10:
            logger.warning("  Too few samples after filter for %s, skipping.", emotion)
            for k in range(U_full.shape[0]):
                rows.append({
                    "emotion": emotion, "pc": k + 1,
                    "evr_full": float(pca_full.explained_variance_ratio_[k]),
                    "cos_full_filtered": float("nan"),
                    "n_sources_full": n_total,
                    "n_sources_filtered": n_keep,
                })
            continue

        X_filt = X_full[keep_mask]   # copy only the filtered slice
        pca_filt, _ = fit_local_pca(X_filt, n_components)
        U_filt = pca_filt.components_.astype(np.float32)   # (n_comp_filt, D)

        # Compute all pairwise cosines at once: (n_comp_filt, n_comp_full)
        cos_mat = np.abs(U_filt @ U_full.T)

        for k in range(min(U_full.shape[0], U_filt.shape[0])):
            best_k_filt = int(np.argmax(cos_mat[:, k]))
            rows.append({
                "emotion": emotion,
                "pc": k + 1,
                "evr_full": float(pca_full.explained_variance_ratio_[k]),
                "evr_filtered": float(pca_filt.explained_variance_ratio_[best_k_filt]),
                "cos_full_filtered": float(cos_mat[best_k_filt, k]),
                "matched_filtered_pc": best_k_filt + 1,
                "n_sources_full": n_total,
                "n_sources_filtered": n_keep,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary verdict
# ---------------------------------------------------------------------------

def build_summary(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    df_c: pd.DataFrame,
    r2_threshold: float = 0.15,
    cos_neutral_threshold: float = 0.5,
    cos_filter_threshold: float = 0.7,
) -> list[dict]:
    """Combine results into a per (emotion, pc) verdict."""
    summary = []
    for _, row_b in df_b.iterrows():
        emotion = row_b["emotion"]
        pc = row_b["pc"]

        row_a_match = df_a[(df_a["emotion"] == emotion) & (df_a["pc"] == pc)]
        row_c_match = df_c[(df_c["emotion"] == emotion) & (df_c["pc"] == pc)]

        r2 = float(row_a_match["r2_all_features"].iloc[0]) if len(row_a_match) else float("nan")
        max_cos_neu = float(row_b["max_cos_neutral"])
        cos_filt = float(row_c_match["cos_full_filtered"].iloc[0]) if len(row_c_match) else float("nan")

        # Verdicts
        style_contaminated_a = r2 > r2_threshold
        style_contaminated_b = max_cos_neu > cos_neutral_threshold
        robust_to_filter_c = cos_filt >= cos_filter_threshold if not np.isnan(cos_filt) else None

        # Aggregate verdict
        style_flags = sum([style_contaminated_a, style_contaminated_b])
        if robust_to_filter_c is False:
            style_flags += 1

        if style_flags == 0:
            verdict = "likely_affective"
        elif style_flags == 1:
            verdict = "ambiguous"
        else:
            verdict = "likely_style"

        summary.append({
            "emotion": emotion,
            "pc": int(pc),
            "evr": float(row_b["evr"]),
            "method_a_r2_all": r2,
            "method_a_style_contaminated": bool(style_contaminated_a),
            "method_b_max_cos_neutral": max_cos_neu,
            "method_b_style_contaminated": bool(style_contaminated_b),
            "method_c_cos_filtered": cos_filt,
            "method_c_robust": robust_to_filter_c,
            "verdict": verdict,
        })

    return summary


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
        "--base-act",
        default=str(BASE_ACT_DIR / "base_text_residual_stream.npy"),
        help="Base text residual stream (N, L, D) for Method B",
    )
    p.add_argument(
        "--base-info",
        default=str(BASE_ACT_DIR / "base_text_residual_stream_info.json"),
    )
    p.add_argument(
        "--caa-directions",
        default=str(ACT_DIR / "caa_emotion_directions.npz"),
    )
    p.add_argument(
        "--meta",
        default=str(ACT_DIR / "emotion_intensity_residual_stream_meta.jsonl"),
    )
    p.add_argument(
        "--layer", type=int, default=13,
        help="Layer to analyse (default: 13)",
    )
    p.add_argument(
        "--n-components", type=int, default=5,
        help="Number of local PCA components per emotion",
    )
    p.add_argument(
        "--n-neutral-pcs", type=int, default=20,
        help="Number of neutral-control PCA components for Method B",
    )
    p.add_argument(
        "--n-topic-pcs", type=int, default=5,
        help="Number of neutral-activation topic PCs to use as style features in Method A",
    )
    p.add_argument(
        "--meaning-threshold", type=float, default=0.7,
        help="Meaning-preservation score threshold for Method C (0–1 scale)",
    )
    p.add_argument(
        "--r2-threshold", type=float, default=0.15,
        help="R² threshold for declaring style contamination in Method A",
    )
    p.add_argument(
        "--cos-neutral-threshold", type=float, default=0.50,
        help="Max-cosine threshold for neutral alignment in Method B",
    )
    p.add_argument(
        "--cos-filter-threshold", type=float, default=0.70,
        help="Cosine threshold for meaning-filter robustness in Method C",
    )
    p.add_argument(
        "--methods", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"],
        metavar="{A,B,C}",
        help="Which methods to run (default: A B C).  E.g. --methods C",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: analysis/style_contamination/layer{L}/)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_BASE_DIR / f"layer{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    logger.info("Loading metadata …")
    meta = load_meta(Path(args.meta))

    logger.info("Loading activation info …")
    with open(args.emo_info) as f:
        emo_info = json.load(f)
    emotions: list[str] = emo_info["emotion_order"]
    layer_indices: list[int] = emo_info["layer_indices"]

    if args.layer not in layer_indices:
        raise ValueError(f"Layer {args.layer} not in available layers: {layer_indices}")
    layer_idx = layer_indices.index(args.layer)

    source_ids: list[str] = [str(s) for s in emo_info["source_ids"]]

    logger.info("Loading emotion activations (mmap) …")
    emo_shape = tuple(emo_info["shape"])
    emo_raw = np.memmap(args.emo_act, dtype=emo_info["dtype"], mode="r", shape=emo_shape)

    logger.info("Loading neutral activations (mmap) …")
    neu_info_path = args.neu_act.replace(".npy", "_info.json")
    with open(neu_info_path) as f:
        neu_info = json.load(f)
    neu_shape = tuple(neu_info["shape"])
    neu_raw = np.memmap(args.neu_act, dtype=neu_info["dtype"], mode="r", shape=neu_shape)

    logger.info("Loading base-text activations (mmap) …")
    with open(args.base_info) as f:
        base_info = json.load(f)
    base_shape = tuple(base_info["shape"])
    base_raw = np.memmap(args.base_act, dtype=base_info["dtype"], mode="r", shape=base_shape)

    logger.info("Loading CAA directions …")
    caa_npz = np.load(args.caa_directions)
    caa_pooled: np.ndarray = caa_npz["caa_pooled"]
    caa: np.ndarray = caa_npz["caa"]

    intensity_order: list[str] = emo_info["intensity_order"]
    intensity_low_idx  = intensity_order.index("low")
    intensity_high_idx = intensity_order.index("high")

    # ----------------------------------------- build per-emotion residuals
    logger.info("Building per-emotion residuals for layer %d …", args.layer)
    delta_resid = build_per_emotion_residuals(
        emo_raw, neu_raw, caa_pooled, caa, layer_idx=layer_idx,
        intensity_low_idx=intensity_low_idx,
        intensity_high_idx=intensity_high_idx,
    )
    # Remove per-source mean across emotions (shared topic / style) — in-place
    delta_resid -= delta_resid.mean(axis=1, keepdims=True)  # (N, E, D)

    run_a = "A" in args.methods
    run_b = "B" in args.methods
    run_c = "C" in args.methods

    neutral_topic_pcs: np.ndarray | None = None
    neutral_delta: np.ndarray | None = None
    neu_layer: np.ndarray | None = None

    if run_a or run_b:
        # ----------------------------------------------- neutral topic features
        logger.info("Computing neutral-activation topic PCs for style features …")
        # float32 to halve memory (~375 MB instead of ~750 MB)
        neu_layer = neu_raw[:, layer_idx, :].astype(np.float32)    # (N, D)
        if run_a:
            k_topic = min(args.n_topic_pcs, neu_layer.shape[0] - 1, neu_layer.shape[1])
            neu_c = neu_layer - neu_layer.mean(axis=0)
            pca_topic = PCA(n_components=k_topic, random_state=0, svd_solver="randomized")
            neutral_topic_pcs = pca_topic.fit_transform(neu_c).astype(np.float64)  # (N, k_topic)
            del neu_c

        if run_b:
            # ----------------------------------- Method B neutral delta (h_neu - h_base)
            logger.info("Computing neutral delta (h_neutral − h_base) for Method B …")
            base_layer = base_raw[:, layer_idx, :].astype(np.float32)  # (N, D)
            neutral_delta = neu_layer.astype(np.float64) - base_layer.astype(np.float64)  # (N, D)
            del base_layer  # free ~375 MB

    # ===================================================================
    # Method A
    # ===================================================================
    out_a = out_dir / "method_a_style_r2.csv"
    if run_a:
        logger.info("=== Method A: Style probe regression ===")
        df_a = method_a_style_probe(
            delta_resid, source_ids, meta, emotions,
            n_components=args.n_components,
            neutral_topic_pcs=neutral_topic_pcs,
        )
        df_a.to_csv(out_a, index=False)
        logger.info("Saved → %s", out_a)
        del df_a  # free before Method B (will reload from CSV for summary)
    else:
        logger.info("Skipping Method A.")

    # ===================================================================
    # Method B
    # ===================================================================
    out_b = out_dir / "method_b_neutral_cos.csv"
    if run_b:
        logger.info("=== Method B: Neutral-control PCA ===")
        df_b = method_b_neutral_control(
            delta_resid, neutral_delta, emotions,
            n_components=args.n_components,
            n_neutral_pcs=args.n_neutral_pcs,
        )
        del neutral_delta  # free ~750 MB
        df_b.to_csv(out_b, index=False)
        logger.info("Saved → %s", out_b)
    else:
        logger.info("Skipping Method B.")
        df_b = pd.read_csv(out_b) if out_b.exists() else None

    # ===================================================================
    # Method C
    # ===================================================================
    out_c = out_dir / "method_c_filter_cos.csv"
    if run_c:
        logger.info("=== Method C: Meaning-preservation filter ===")
        df_c = method_c_filter(
            delta_resid, source_ids, meta, emotions,
            n_components=args.n_components,
            meaning_threshold=args.meaning_threshold,
        )
        df_c.to_csv(out_c, index=False)
        logger.info("Saved → %s", out_c)
    else:
        logger.info("Skipping Method C.")
        df_c = pd.read_csv(out_c) if out_c.exists() else None

    # ===================================================================
    # Summary (only when all three outputs are available)
    # ===================================================================
    out_summary = out_dir / "summary.json"
    df_a_summary = pd.read_csv(out_a) if out_a.exists() else None
    if df_a_summary is not None and df_b is not None and df_c is not None:
        logger.info("Building summary …")
        summary = build_summary(
            df_a_summary, df_b, df_c,
            r2_threshold=args.r2_threshold,
            cos_neutral_threshold=args.cos_neutral_threshold,
            cos_filter_threshold=args.cos_filter_threshold,
        )
        with open(out_summary, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved → %s", out_summary)

        # ============================= pretty-print summary table
        print(f"\n=== Style Contamination Analysis — Layer {args.layer} ===")
        print(f"{'emotion':<14} {'PC':>3}  {'EVR':>6}  {'R²(A)':>7}  "
              f"{'maxCos(B)':>10}  {'cos_filt(C)':>12}  verdict")
        print("-" * 75)
        for s in summary:
            r2_str = f"{s['method_a_r2_all']:.3f}" if not np.isnan(s["method_a_r2_all"]) else "  nan"
            cos_b_str = f"{s['method_b_max_cos_neutral']:.3f}"
            cos_c_str = f"{s['method_c_cos_filtered']:.3f}" if s["method_c_cos_filtered"] is not None \
                        and not np.isnan(s["method_c_cos_filtered"]) else "    nan"
            print(f"{s['emotion']:<14} {s['pc']:>3}  {s['evr']:>6.3f}  "
                  f"{r2_str:>7}  {cos_b_str:>10}  {cos_c_str:>12}  {s['verdict']}")

        verdicts = [s["verdict"] for s in summary]
        for v in ["likely_affective", "ambiguous", "likely_style"]:
            print(f"  {v}: {verdicts.count(v)}")
    else:
        logger.info(
            "Summary skipped — not all method outputs available. "
            "Run with --methods A B C (or omit --methods) to generate summary."
        )

    logger.info("Done. All outputs in %s", out_dir)


if __name__ == "__main__":
    main()
