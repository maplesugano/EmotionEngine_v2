"""Experiment 3: Interpret local sub-axes from extreme examples.

For each emotion e and local PC u_{e,k}, compute the projection score:

    z_{n,e,k} = delta_resid[n,e] · u_{e,k}

Find the top-K (high pole) and bottom-K (low pole) source examples by score.
For each extreme example, record:
  - base_text
  - neutral_paraphrase
  - emotion_rewrite
  - source metadata
  - intensity
  - PC score

Then generate a per-axis LLM judge prompt file asking the model to name the
axis and describe both poles without imposing a hypothesis.

Outputs are written to analysis/local_axis_interpretation/.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = REPO_ROOT / "activation" / "emotion_rewrites"
OUT_DIR = Path(__file__).resolve().parent / "results"

LLM_PROMPT_TEMPLATE = """\
You are an expert in affective science and computational linguistics.

Below are {n_pcs} latent axes extracted from the internal activations of a language
model for the emotion "{emotion}". Each axis is a Principal Component of the
residual activations after the shared emotion direction AND the intensity direction
have been projected out. The axes are ordered by explained variance (EVR).

Your tasks:
1. NAME each axis based solely on the examples — do NOT assume meaning in advance.
2. Describe what distinguishes the high and low poles.
3. Assess whether the axis reflects a genuine pre-verbal affective dimension or is
   contaminated by text style / topic / semantic content.
4. DIFFERENTIATE the axes from each other — if two axes seem similar, explain how
   they differ.

--- EVALUATION CRITERIA ---

style_or_semantics_contamination:
  low    — the contrast is clearly affective/emotional (e.g. arousal level, social
            vs private, acute vs chronic). The topic or linguistic register of the
            examples does NOT explain the split.
  medium — the affective contrast is real but entangled with a topic shift or
            register difference (e.g. high pole = work situations, low = personal).
  high   — the axis is primarily explained by topic, vocabulary, or writing style
            rather than an emotional dimension.

confidence:
  5 — the axis name is clear and both poles are distinctly characterised.
  4 — the axis is interpretable but one pole is somewhat mixed.
  3 — the axis is plausible but the examples are noisy or the poles overlap.
  2 — the examples do not strongly suggest a coherent affective dimension.
  1 — no interpretable pattern found.

--- AXES FOR "{emotion}" ---
{all_pc_blocks}
--- OUTPUT FORMAT ---

Respond ONLY with a valid JSON array — one object per axis, in PC order:
[
  {{
    "emotion": "{emotion}",
    "pc": <int>,
    "evr": <float>,
    "axis_name": "<high-pole label> vs <low-pole label>",
    "high_pole_description": "<what characterises the high-pole examples>",
    "low_pole_description": "<what characterises the low-pole examples>",
    "differentiation_from_other_pcs": "<how this axis differs from the other PCs above>",
    "preverbal_affective_interpretation": "<what pre-verbal affective dimension this axis may capture>",
    "style_or_semantics_contamination": "low|medium|high",
    "confidence": <int 1-5>
  }},
  ...
]
"""

PC_BLOCK_TEMPLATE = """\
=== PC{pc}  (EVR = {evr_pct:.1f}% of within-emotion variance) ===

-- HIGH-POLE EXAMPLES (high score on PC{pc}) --
{high_pole_block}
-- LOW-POLE EXAMPLES (low score on PC{pc}) --
{low_pole_block}
"""

EXAMPLE_BLOCK_TEMPLATE = """\
[Example {i}]
Source: {source_id}
Intensity: {intensity}
PC score: {score:.4f}
Base text:
  {base_text}
Neutral paraphrase:
  {neutral_paraphrase}
Emotion rewrite ({emotion}):
  {emotion_rewrite}
"""


# ---------------------------------------------------------------------------
# Helpers (shared with exp1/exp2)
# ---------------------------------------------------------------------------

def unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def project_out(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Remove component along unit vector g from each row of X."""
    coeff = X @ g
    return X - np.outer(coeff, g)


def build_per_emotion_residuals(
    emo_act: np.ndarray,       # (N, E, I, L, D)  mmap
    neu_act: np.ndarray,       # (N, L, D)         mmap
    caa_pooled: np.ndarray,    # (E, L, D)         unit-normed
    caa: np.ndarray,           # (E, I, L, D)      per-emotion per-intensity CAA
    layer_idx: int,            # index into the L dimension
    intensity_low_idx: int = 0,
    intensity_high_idx: int = 2,
    CHUNK: int = 512,
) -> np.ndarray:
    """Return delta_resid of shape (N, E, D) for a single layer.

    Removes two directions per emotion:
      1. Common affect direction g (mean of all emotion CAA directions)
      2. Per-emotion intensity direction u_e (high - low CAA direction)
    """
    N, E, I, L, D = emo_act.shape

    # Common direction g
    g_raw = caa_pooled[:, layer_idx, :].mean(axis=0)   # (D,)
    g = unit_norm(g_raw)                               # (D,)

    # Per-emotion intensity directions u_e: (E, D)
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
            d = project_out(delta[:, e, :], g)           # remove common direction
            d = project_out(d, intensity_dirs[e])        # remove per-emotion intensity axis
            delta_resid[start:end, e, :] = d

    return delta_resid


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def load_text_records(text_data_path: str) -> dict[str, dict]:
    """Load text records keyed by source_id.

    Supported formats
    -----------------
    1. **JSONL** (native meta format from this project):
       One line per (source_id, intensity_level) triple, 3 lines per source.
       Fields: source_id, base_text, intensity_level, {emotion}_rewrite, ...
       Example path: activation/emotion_rewrites/emotion_intensity_residual_stream_meta.jsonl

    2. **JSON** list or dict:
       List of records with nested ``emotion_rewrites`` dict, or a dict keyed
       by source_id in the same structure.

    3. **CSV / TSV**:
       Columns: source_id, base_text, neutral_paraphrase, emotion, intensity,
       emotion_rewrite  (one row per source × emotion × intensity).
    """
    path = Path(text_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Text data file not found: {path}")

    suffix = path.suffix.lower()

    # --- JSONL (native project meta format) ---
    if suffix == ".jsonl":
        records: dict[str, dict] = {}
        emotion_keys = [
            "joy", "trust", "fear", "surprise",
            "sadness", "disgust", "anger", "anticipation",
        ]
        # Intensity order as used in info["intensity_order"]
        intensity_order = ["low", "medium", "high"]

        with open(path) as f:
            for line in f:
                row = json.loads(line)
                sid = str(row["source_id"])
                intensity = str(row.get("intensity_level", ""))

                if sid not in records:
                    records[sid] = {
                        "source_id": sid,
                        "base_text": row.get("base_text", ""),
                        "neutral_paraphrase": row.get("neutral_paraphrase", ""),
                        "emotion_rewrites": {e: [] for e in emotion_keys},
                        "intensities": [],
                    }

                # Append rewrites in intensity-sorted order
                rec = records[sid]
                for emo in emotion_keys:
                    key = f"{emo}_rewrite"
                    rec["emotion_rewrites"][emo].append(row.get(key, ""))
                if intensity not in rec["intensities"]:
                    rec["intensities"].append(intensity)

        # Sort each source's lists by intensity_order so index is consistent
        for rec in records.values():
            order = {lvl: i for i, lvl in enumerate(intensity_order)}
            idx = sorted(range(len(rec["intensities"])),
                         key=lambda i: order.get(rec["intensities"][i], 99))
            rec["intensities"] = [rec["intensities"][i] for i in idx]
            for emo in rec["emotion_rewrites"]:
                rec["emotion_rewrites"][emo] = [
                    rec["emotion_rewrites"][emo][i] for i in idx
                ]
        return records

    # --- JSON list or dict ---
    elif suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return {r["source_id"]: r for r in raw}
        elif isinstance(raw, dict):
            return raw
        else:
            raise ValueError("Unexpected JSON structure in text data file.")

    # --- CSV / TSV ---
    elif suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        records = {}
        for _, row in df.iterrows():
            sid = str(row["source_id"])
            if sid not in records:
                records[sid] = {
                    "source_id": sid,
                    "base_text": row.get("base_text", ""),
                    "neutral_paraphrase": row.get("neutral_paraphrase", ""),
                    "emotion_rewrites": {},
                    "intensities": [],
                }
            emo = str(row.get("emotion", ""))
            intensity = row.get("intensity", "")
            rewrite = row.get("emotion_rewrite", "")
            if emo:
                records[sid].setdefault("emotion_rewrites", {}).setdefault(emo, [])
                records[sid]["emotion_rewrites"][emo].append(rewrite)
                if intensity not in records[sid]["intensities"]:
                    records[sid]["intensities"].append(intensity)
        return records

    else:
        raise ValueError(f"Unsupported text data format: {suffix}")


# ---------------------------------------------------------------------------
# Core: extract extremes and generate prompts
# ---------------------------------------------------------------------------

def get_extreme_examples(
    scores: np.ndarray,       # (N,)
    source_ids: list[str],
    text_records: dict[str, dict],
    emotion: str,
    pc: int,
    n_extreme: int,
    intensity_index: int,
) -> tuple[list[dict], list[dict]]:
    """Return (high_pole_examples, low_pole_examples) as lists of dicts."""
    sorted_asc = np.argsort(scores)
    low_indices = sorted_asc[:n_extreme]
    high_indices = sorted_asc[-n_extreme:][::-1]

    def make_example(idx: int) -> dict:
        sid = source_ids[idx]
        rec = text_records.get(sid, {})
        # Get the emotion rewrite at the given intensity index, or first available
        rewrites = rec.get("emotion_rewrites", {}).get(emotion, [])
        if rewrites:
            rewrite = rewrites[min(intensity_index, len(rewrites) - 1)]
        else:
            rewrite = ""
        intensities = rec.get("intensities", [])
        intensity = intensities[min(intensity_index, len(intensities) - 1)] if intensities else ""
        return {
            "source_id": sid,
            "base_text": rec.get("base_text", ""),
            "neutral_paraphrase": rec.get("neutral_paraphrase", ""),
            "emotion_rewrite": rewrite,
            "intensity": intensity,
            "pc_score": float(scores[idx]),
            "emotion": emotion,
            "pc": pc,
        }

    high_examples = [make_example(i) for i in high_indices]
    low_examples = [make_example(i) for i in low_indices]
    return high_examples, low_examples


def format_example_block(examples: list[dict]) -> str:
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(EXAMPLE_BLOCK_TEMPLATE.format(
            i=i,
            source_id=ex["source_id"],
            intensity=ex["intensity"],
            score=ex["pc_score"],
            base_text=ex["base_text"] or "(not available)",
            neutral_paraphrase=ex["neutral_paraphrase"] or "(not available)",
            emotion=ex["emotion"],
            emotion_rewrite=ex["emotion_rewrite"] or "(not available)",
        ))
    return "\n".join(blocks)


def generate_emotion_prompt(
    emotion: str,
    pc_data: list[dict],  # list of {pc, evr, high_examples, low_examples}
) -> str:
    """Generate a single prompt covering all PCs for one emotion."""
    pc_blocks = []
    for entry in pc_data:
        high_block = format_example_block(entry["high_examples"])
        low_block = format_example_block(entry["low_examples"])
        pc_blocks.append(PC_BLOCK_TEMPLATE.format(
            pc=entry["pc"],
            evr_pct=entry["evr"] * 100,
            high_pole_block=high_block,
            low_pole_block=low_block,
        ))
    return LLM_PROMPT_TEMPLATE.format(
        emotion=emotion,
        n_pcs=len(pc_data),
        all_pc_blocks="\n".join(pc_blocks),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--emo-act",
        default=str(ACT_DIR / "emotion_intensity_residual_stream.npy"),
        help="Emotion×intensity residual stream (N, E, I, L, D)",
    )
    p.add_argument(
        "--emo-info",
        default=str(ACT_DIR / "emotion_intensity_residual_stream_info.json"),
    )
    p.add_argument(
        "--neu-act",
        default=str(ACT_DIR / "neutral_paraphrase_residual_stream.npy"),
        help="Neutral paraphrase residual stream (N, L, D)",
    )
    p.add_argument(
        "--caa-directions",
        default=str(ACT_DIR / "caa_emotion_directions.npz"),
    )
    p.add_argument(
        "--text-data",
        default=str(ACT_DIR / "emotion_intensity_residual_stream_meta.jsonl"),
        help=(
            "Path to text records file (.jsonl / .json / .csv). "
            "Default: emotion_intensity_residual_stream_meta.jsonl in ACT_DIR. "
            "If not found, text fields will be empty in the output."
        ),
    )
    p.add_argument(
        "--neutral-data",
        default=str(REPO_ROOT / "dataset" / "neutral_paraphrases" / "neutral_paraphrases.jsonl"),
        help=(
            "Path to neutral_paraphrases.jsonl produced by build_neutral_paraphrase_dataset.py. "
            "Each line: {source_id, base_text, neutral_paraphrase, meaning_preserved_score}. "
            "If found, neutral_paraphrase fields in the prompts will be filled in. "
            "Default: dataset/neutral_paraphrases/neutral_paraphrases.jsonl"
        ),
    )
    p.add_argument(
        "--layer", type=int, default=13,
        help="Layer to analyse (default: 13)",
    )
    p.add_argument(
        "--n-components", type=int, default=5,
        help="Number of local PCs to interpret per emotion",
    )
    p.add_argument(
        "--n-extreme", type=int, default=5,
        help="Number of extreme examples per pole",
    )
    p.add_argument(
        "--intensity-index", type=int, default=1,
        help="Which intensity level to display in examples (0-indexed)",
    )
    p.add_argument(
        "--out-dir", default=str(OUT_DIR),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "llm_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    extremes_dir = out_dir / "extremes"
    extremes_dir.mkdir(parents=True, exist_ok=True)

    # --- load metadata ---
    logger.info("Loading metadata …")
    with open(args.emo_info) as f:
        info = json.load(f)
    emotions: list[str] = info["emotion_order"]
    layer_indices: list[int] = info["layer_indices"]
    source_ids: list[str] = info["source_ids"]

    if args.layer not in layer_indices:
        raise ValueError(f"Layer {args.layer} not in available layers: {layer_indices}")
    layer_idx = layer_indices.index(args.layer)

    # --- load text records (optional; degrade gracefully) ---
    text_records: dict[str, dict] = {}
    if Path(args.text_data).exists():
        logger.info("Loading text records from %s …", args.text_data)
        text_records = load_text_records(args.text_data)
        logger.info("  Loaded %d text records.", len(text_records))
    else:
        logger.warning(
            "Text data file not found: %s — text fields will be empty. "
            "Pass --text-data to supply a JSON or CSV with base_text / neutral_paraphrase / emotion_rewrites.",
            args.text_data,
        )

    # --- merge neutral paraphrases (optional) ---
    neutral_data_path = Path(args.neutral_data)
    if neutral_data_path.exists():
        logger.info("Loading neutral paraphrases from %s …", neutral_data_path)
        n_merged = 0
        with open(neutral_data_path) as f:
            for line in f:
                row = json.loads(line)
                sid = str(row["source_id"])
                np_text = row.get("neutral_paraphrase", "")
                if not np_text:
                    continue
                if sid in text_records:
                    text_records[sid]["neutral_paraphrase"] = np_text
                else:
                    # source exists in neutral data but not in text_records (e.g. text_data missing)
                    text_records[sid] = {
                        "source_id": sid,
                        "base_text": row.get("base_text", ""),
                        "neutral_paraphrase": np_text,
                        "emotion_rewrites": {},
                        "intensities": [],
                    }
                n_merged += 1
        logger.info("  Merged neutral_paraphrase for %d sources.", n_merged)
    else:
        logger.warning(
            "Neutral paraphrase file not found: %s — neutral_paraphrase fields will be empty. "
            "Pass --neutral-data to supply dataset/neutral_paraphrases/neutral_paraphrases.jsonl.",
            neutral_data_path,
        )

    # --- load activations (memory-mapped) ---
    logger.info("Loading emotion activations (mmap) …")
    emo_shape = tuple(info["shape"])
    emo_raw = np.memmap(args.emo_act, dtype=info["dtype"], mode="r", shape=emo_shape)

    logger.info("Loading neutral activations (mmap) …")
    neu_info_path = args.neu_act.replace(".npy", "_info.json")
    with open(neu_info_path) as f:
        neu_info = json.load(f)
    neu_shape = tuple(neu_info["shape"])
    neu_raw = np.memmap(args.neu_act, dtype=neu_info["dtype"], mode="r", shape=neu_shape)

    # --- load CAA directions ---
    logger.info("Loading CAA directions …")
    caa_npz = np.load(args.caa_directions)
    caa_pooled: np.ndarray = caa_npz["caa_pooled"]   # (E, L, D) unit-normed
    caa: np.ndarray = caa_npz["caa"]                 # (E, I, L, D)

    intensity_order: list[str] = info["intensity_order"]
    intensity_low_idx  = intensity_order.index("low")
    intensity_high_idx = intensity_order.index("high")

    # --- build residual deltas for the target layer ---
    logger.info("Building residual deltas for layer %d (index %d) …", args.layer, layer_idx)
    delta_resid = build_per_emotion_residuals(
        emo_raw, neu_raw, caa_pooled, caa, layer_idx=layer_idx,
        intensity_low_idx=intensity_low_idx,
        intensity_high_idx=intensity_high_idx,
    )
    # delta_resid: (N, E, D)

    # --- remove source-level random effect ---
    # Subtract each source's mean across emotions to eliminate topic/style variance
    # that is shared across all emotions and would otherwise inflate the PR.
    logger.info("Removing per-source mean across emotions …")
    source_mean = delta_resid.mean(axis=1, keepdims=True)  # (N, 1, D)
    delta_resid = delta_resid - source_mean                # (N, E, D)

    # --- per-emotion local PCA → projection scores → extreme examples ---
    all_extreme_rows: list[dict] = []
    summary_rows: list[dict] = []

    for e_idx, emotion in enumerate(emotions):
        logger.info("Emotion: %s", emotion)
        X_e = delta_resid[:, e_idx, :].astype(np.float64)   # (N, D)
        N, D = X_e.shape

        # Centre and fit PCA
        mu = X_e.mean(axis=0)
        X_c = X_e - mu
        n_comp = min(args.n_components, N - 1, D)
        pca = PCA(n_components=n_comp, random_state=0)
        pca.fit(X_c)

        # Collect all PC data for this emotion before generating the prompt
        emotion_pc_data: list[dict] = []

        for k in range(n_comp):
            u_k = pca.components_[k]          # (D,)
            scores = X_c @ u_k                # (N,)  projection scores

            pc_number = k + 1
            evr = float(pca.explained_variance_ratio_[k])
            logger.info("  PC%d  EVR=%.4f", pc_number, evr)

            high_examples, low_examples = get_extreme_examples(
                scores,
                source_ids,
                text_records,
                emotion=emotion,
                pc=pc_number,
                n_extreme=args.n_extreme,
                intensity_index=args.intensity_index,
            )

            # Save per-PC extremes as JSON
            axis_id = f"layer{args.layer}_{emotion}_pc{pc_number}"
            extremes_payload = {
                "layer": args.layer,
                "emotion": emotion,
                "pc": pc_number,
                "evr": evr,
                "n_extreme": args.n_extreme,
                "high_pole": high_examples,
                "low_pole": low_examples,
            }
            with open(extremes_dir / f"{axis_id}.json", "w") as f:
                json.dump(extremes_payload, f, ensure_ascii=False, indent=2)

            emotion_pc_data.append({
                "pc": pc_number,
                "evr": evr,
                "high_examples": high_examples,
                "low_examples": low_examples,
            })

            # Collect flat rows for summary CSV
            for pole, examples in [("high", high_examples), ("low", low_examples)]:
                for rank, ex in enumerate(examples, 1):
                    row = dict(ex)
                    row.update({
                        "layer": args.layer,
                        "pc": pc_number,
                        "evr": evr,
                        "pole": pole,
                        "rank": rank,
                    })
                    all_extreme_rows.append(row)

            summary_rows.append({
                "layer": args.layer,
                "emotion": emotion,
                "pc": pc_number,
                "evr": evr,
                "high_pole_score_max": float(scores.max()),
                "low_pole_score_min": float(scores.min()),
                "score_std": float(scores.std()),
            })

        # Generate one combined prompt per emotion (all PCs together)
        emotion_prompt_id = f"layer{args.layer}_{emotion}_all_pcs"
        prompt_text = generate_emotion_prompt(emotion, emotion_pc_data)
        with open(prompts_dir / f"{emotion_prompt_id}_prompt.txt", "w") as f:
            f.write(prompt_text)
        logger.info("  Written combined prompt → %s_prompt.txt", emotion_prompt_id)

    # --- save summary CSV ---
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"layer{args.layer}_axis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved axis summary → %s", summary_path)

    # --- save flat extremes CSV ---
    if all_extreme_rows:
        extremes_df = pd.DataFrame(all_extreme_rows)
        extremes_path = out_dir / f"layer{args.layer}_extremes.csv"
        extremes_df.to_csv(extremes_path, index=False)
        logger.info("Saved extremes table → %s", extremes_path)

    logger.info(
        "Done. %d combined emotion-prompt files written to %s",
        len(emotions), prompts_dir,
    )
    print("\n=== Axis Summary — Layer %d ===" % args.layer)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
