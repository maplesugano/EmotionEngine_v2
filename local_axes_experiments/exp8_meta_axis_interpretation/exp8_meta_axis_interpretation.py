"""Experiment 8: Interpret shared meta-axes m_k via LLM judge.

Analogous to exp3 (which interpreted emotion-specific local PCs), but working
on the cross-emotion shared axes m_k extracted in exp7.

For each meta-axis m_k:
  1. Project ALL residual delta vectors (N × E sources) onto m_k.
  2. Find the top-K (high-pole) and bottom-K (low-pole) examples globally.
  3. Ensure pole examples span multiple emotions to demonstrate cross-emotion nature.
  4. Build an LLM prompt showing the extreme examples with their emotion labels.
  5. Ask the judge to name the axis and describe both poles without assuming meaning.

The cross-emotion design is the key difference from exp3:
  - exp3: "what distinguishes high vs low within one emotion?"
  - exp8: "what distinguishes high vs low ACROSS ALL emotions?"

This reveals what affective dimension m_k captures that transcends emotion labels.

Outputs written to results/:
  layer{L}_extremes.csv         — source_id, emotion, score, pole for each m_k
  layer{L}_llm_prompts/         — one .txt file per m_k
  layer{L}_llm_judge_results.jsonl — LLM responses

Run:
    python local_axes_experiments/exp8_meta_axis_interpretation/exp8_meta_axis_interpretation.py --layer 13 --n-extremes 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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
ACT_DIR   = REPO_ROOT / "activation" / "emotion_rewrites"
EXP7_DIR  = REPO_ROOT / "local_axes_experiments" / "exp7_meta_axis_extraction" / "results"
OUT_DIR   = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

LLM_SYSTEM = """\
You are an expert in affective science and computational linguistics.
Respond ONLY with valid JSON matching the requested schema.
"""

LLM_PROMPT_TEMPLATE = """\
You are an expert in affective science and computational linguistics.

Below is a latent axis extracted from the internal activations of a language model.
This axis is a shared Principal Component computed ACROSS ALL EIGHT EMOTIONS
(anger, anticipation, disgust, fear, joy, sadness, surprise, trust).

The axis was extracted from the residual activations after removing:
  1. The common emotionalization direction g (shared across all emotions)
  2. The per-emotion intensity direction u_e (emotion-specific CAA direction)

Because this axis is shared across emotions, it is expected to capture a
PRE-VERBAL AFFECTIVE DIMENSION — a quality that exists within emotional experience
independently of which named emotion is being expressed.

Your tasks:
1. NAME the axis based solely on the examples — do NOT assume meaning in advance.
2. Describe what distinguishes the high-pole from the low-pole examples.
3. Identify the AFFECTIVE quality that cuts across emotions — what is shared among
   high-pole anger, high-pole joy, high-pole sadness, etc.?
4. Assess whether the axis captures a genuine pre-verbal affective dimension or is
   contaminated by text style / topic / semantic content.
5. Suggest an interpretable label for use in a UI knob (e.g. "resolved ↔ conflicted").

--- META-AXIS {axis_id} ---
(EVR within each emotion's residual space: see below)

-- HIGH-POLE EXAMPLES (high score on {axis_id}: emotionally "more X") --
{high_pole_block}

-- LOW-POLE EXAMPLES (low score on {axis_id}: emotionally "more Y") --
{low_pole_block}

--- OUTPUT FORMAT ---

Respond ONLY with a valid JSON object:
{{
  "axis_id": "{axis_id}",
  "axis_name": "<high-pole label> ↔ <low-pole label>",
  "high_pole_description": "<what characterises the HIGH-pole examples across all emotions>",
  "low_pole_description": "<what characterises the LOW-pole examples across all emotions>",
  "cross_emotion_pattern": "<the pre-verbal affective quality that transcends emotion labels>",
  "ui_knob_label": "<short label for the high pole (2-4 words)> ↔ <label for low pole>",
  "preverbal_affective_interpretation": "<which pre-verbal affective dimension this may capture>",
  "style_or_semantics_contamination": "low|medium|high",
  "confidence": <int 1-5>,
  "reasoning": "<2-3 sentences explaining the axis interpretation>"
}}
"""

EXAMPLE_BLOCK_TEMPLATE = """\
[Example {i}]
Emotion label: {emotion}  |  Score: {score:.4f}
Base text:
  {base_text}
Neutral paraphrase:
  {neutral_paraphrase}
Emotion rewrite ({emotion}):
  {emotion_rewrite}

"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def project_out(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    coeff = X @ g
    return X - np.outer(coeff, g)


def build_per_emotion_residuals(
    emo_act: np.ndarray,       # (N, E, I, L, D)
    neu_act: np.ndarray,       # (N, L, D)
    caa_pooled: np.ndarray,    # (E, L, D)
    caa: np.ndarray,           # (E, I, L, D)
    layer_idx: int,
    intensity_low_idx: int = 0,
    intensity_high_idx: int = 2,
    CHUNK: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (delta_resid, g, intensity_dirs) for a single layer.

    Same pipeline as exp3/5/6 — removes common direction g and per-emotion
    intensity direction, then subtracts per-source mean across emotions.
    """
    N, E, I, L, D = emo_act.shape

    g_raw = caa_pooled[:, layer_idx, :].mean(axis=0)
    g = unit_norm(g_raw.astype(np.float64)).astype(np.float32)

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
            d = project_out(delta[:, e, :].astype(np.float64), g.astype(np.float64))
            d = project_out(d, intensity_dirs[e].astype(np.float64))
            delta_resid[start:end, e, :] = d.astype(np.float32)

    # Remove per-source mean across emotions (matches exp3/5/6)
    source_mean = delta_resid.mean(axis=1, keepdims=True)
    delta_resid -= source_mean

    return delta_resid, g, intensity_dirs


def load_text_records(path: Path) -> dict[str, dict]:
    """Load source text records keyed by source_id (same as exp3)."""
    emotion_keys = [
        "joy", "trust", "fear", "surprise",
        "sadness", "disgust", "anger", "anticipation",
    ]
    intensity_order = ["low", "medium", "high"]
    records: dict[str, dict] = {}

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
            rec = records[sid]
            for emo in emotion_keys:
                rec["emotion_rewrites"][emo].append(row.get(f"{emo}_rewrite", ""))
            if intensity not in rec["intensities"]:
                rec["intensities"].append(intensity)

    for rec in records.values():
        order_map = {lvl: i for i, lvl in enumerate(intensity_order)}
        idx = sorted(range(len(rec["intensities"])),
                     key=lambda i: order_map.get(rec["intensities"][i], 99))
        rec["intensities"] = [rec["intensities"][i] for i in idx]
        for emo in rec["emotion_rewrites"]:
            rec["emotion_rewrites"][emo] = [rec["emotion_rewrites"][emo][i] for i in idx]

    return records


def find_extreme_examples(
    scores: np.ndarray,         # (N,) projected scores
    source_ids: list[str],
    emotion: str,
    n: int,
) -> list[dict]:
    """Return top-n and bottom-n entries for one emotion."""
    order = np.argsort(scores)
    top_idx = order[-n:][::-1]
    bot_idx = order[:n]
    entries = []
    for pole, idxs in [("high", top_idx), ("low", bot_idx)]:
        for rank, i in enumerate(idxs):
            entries.append({
                "source_id": source_ids[i],
                "emotion": emotion,
                "score": float(scores[i]),
                "pole": pole,
                "rank": rank + 1,
            })
    return entries


def select_diverse_extremes(
    all_entries: list[dict],
    n_total: int,
    emotions: list[str],
) -> list[dict]:
    """From all (source, emotion) entries for one pole, pick n_total examples
    that span as many different emotions as possible.

    Strategy:
      1. Sort by |score| descending.
      2. Round-robin over emotions to ensure diversity.
      3. Fall back to top-scoring if some emotions run out.
    """
    high = sorted([e for e in all_entries if e["pole"] == "high"],
                  key=lambda x: -abs(x["score"]))
    low  = sorted([e for e in all_entries if e["pole"] == "low"],
                  key=lambda x: -abs(x["score"]))

    result = []
    for pool in [high, low]:
        by_emo: dict[str, list[dict]] = {e: [] for e in emotions}
        for entry in pool:
            by_emo[entry["emotion"]].append(entry)

        chosen: list[dict] = []
        round_idx = 0
        while len(chosen) < n_total:
            added = False
            for emo in emotions:
                if len(chosen) >= n_total:
                    break
                if round_idx < len(by_emo[emo]):
                    chosen.append(by_emo[emo][round_idx])
                    added = True
            if not added:
                break
            round_idx += 1
        result.extend(chosen[:n_total])

    return result


def build_example_block(
    entries: list[dict],
    pole: str,
    text_records: dict[str, dict],
    intensity_idx: int = 2,  # "high" intensity (index 2 in low/medium/high)
) -> str:
    pole_entries = [e for e in entries if e["pole"] == pole]
    blocks = []
    for i, entry in enumerate(pole_entries, 1):
        sid = entry["source_id"]
        emo = entry["emotion"]
        rec = text_records.get(sid, {})

        rewrites = rec.get("emotion_rewrites", {}).get(emo, [])
        rewrite = rewrites[min(intensity_idx, len(rewrites) - 1)] if rewrites else "(not found)"

        blocks.append(EXAMPLE_BLOCK_TEMPLATE.format(
            i=i,
            emotion=emo,
            score=entry["score"],
            base_text=rec.get("base_text", "(not found)"),
            neutral_paraphrase=rec.get("neutral_paraphrase", "(not found)"),
            emotion_rewrite=rewrite,
        ))
    return "".join(blocks) if blocks else "(no examples)"


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def call_judge(
    prompt: str,
    model: str,
    cache_dir: Path,
) -> dict:
    import hashlib
    h = hashlib.sha1()
    h.update(model.encode())
    h.update(b"|")
    h.update(prompt.encode("utf-8"))
    cache_path = cache_dir / f"{h.hexdigest()[:24]}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    raw = json.loads(resp.choices[0].message.content)
    with open(cache_path, "w") as f:
        json.dump(raw, f)
    return raw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=13)
    p.add_argument(
        "--exp7-dir", default=str(EXP7_DIR),
        help="Directory containing exp7 results (layer{L}_meta_axes.npy etc.)",
    )
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
        "--text-meta",
        default=str(ACT_DIR / "emotion_intensity_residual_stream_meta.jsonl"),
        help="JSONL file with base_text, neutral_paraphrase, emotion rewrites.",
    )
    p.add_argument("--n-extremes", type=int, default=8,
                   help="Number of extreme examples per pole, per axis (total, across emotions).")
    p.add_argument("--judge-model", default="gpt-4.1",
                   help="OpenAI model for interpretation.")
    p.add_argument("--prompts-only", action="store_true",
                   help="Write prompt files but do not call the LLM.")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / f"layer{args.layer}_llm_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    judge_cache_dir = out_dir / "judge_cache"
    judge_cache_dir.mkdir(parents=True, exist_ok=True)

    # --- load exp7 meta-axes ---
    exp7_dir = Path(args.exp7_dir)
    axes_path = exp7_dir / f"layer{args.layer}_meta_axes.npy"
    info_path = exp7_dir / f"layer{args.layer}_meta_axes_info.json"

    if not axes_path.exists():
        raise FileNotFoundError(
            f"exp7 meta-axes not found: {axes_path}\nRun exp7_meta_axis_extraction.py first."
        )

    M = np.load(axes_path).astype(np.float64)    # (K, D)
    with open(info_path) as f:
        axes_info = json.load(f)
    axis_ids: list[str] = axes_info["axis_ids"]
    emotions: list[str] = axes_info["emotions"]
    K = len(axis_ids)
    logger.info("Loaded %d meta-axes.  Emotions: %s", K, emotions)

    # --- load activations ---
    logger.info("Loading activation metadata …")
    with open(args.emo_info) as f:
        info = json.load(f)
    layer_indices: list[int] = info["layer_indices"]
    source_ids: list[str] = info["source_ids"]

    if args.layer not in layer_indices:
        raise ValueError(f"Layer {args.layer} not in {layer_indices}")
    layer_idx = layer_indices.index(args.layer)

    intensity_order: list[str] = info["intensity_order"]
    intensity_low_idx  = intensity_order.index("low")
    intensity_high_idx = intensity_order.index("high")

    logger.info("Memory-mapping activations …")
    emo_shape = tuple(info["shape"])
    emo_raw = np.memmap(args.emo_act, dtype=info["dtype"], mode="r", shape=emo_shape)

    neu_info_path = args.neu_act.replace(".npy", "_info.json")
    with open(neu_info_path) as f:
        neu_info = json.load(f)
    neu_raw = np.memmap(args.neu_act, dtype=neu_info["dtype"], mode="r", shape=tuple(neu_info["shape"]))

    caa_npz = np.load(args.caa_directions)
    caa_pooled: np.ndarray = caa_npz["caa_pooled"]
    caa: np.ndarray = caa_npz["caa"]

    # --- build residuals ---
    logger.info("Building residual deltas for layer %d …", args.layer)
    delta_resid, g, intensity_dirs = build_per_emotion_residuals(
        emo_raw, neu_raw, caa_pooled, caa,
        layer_idx=layer_idx,
        intensity_low_idx=intensity_low_idx,
        intensity_high_idx=intensity_high_idx,
    )
    # delta_resid: (N, E, D)
    N, E, D = delta_resid.shape
    logger.info("  delta_resid: %s", delta_resid.shape)

    # --- load text records ---
    logger.info("Loading text records …")
    text_records = load_text_records(Path(args.text_meta))
    logger.info("  %d text records loaded.", len(text_records))

    # --- compute projections and find extremes for each m_k ---
    all_extreme_rows: list[dict] = []
    results: list[dict] = []

    for k_idx, axis_id in enumerate(axis_ids):
        m_k = M[k_idx].astype(np.float32)  # (D,)
        logger.info("Processing %s …", axis_id)

        # Collect all (source, emotion) entries with their projection scores
        all_entries: list[dict] = []
        for e_idx, emo in enumerate(emotions):
            scores = delta_resid[:, e_idx, :].astype(np.float32) @ m_k  # (N,)
            entries = find_extreme_examples(scores, source_ids, emo, n=args.n_extremes)
            for entry in entries:
                entry["axis_id"] = axis_id
            all_entries.extend(entries)

        # Select diverse extremes spanning multiple emotions
        n_per_pole = args.n_extremes
        chosen = select_diverse_extremes(all_entries, n_total=n_per_pole, emotions=emotions)

        # Save to master extremes table
        for entry in chosen:
            all_extreme_rows.append(entry)

        # Build prompt
        high_block = build_example_block(chosen, "high", text_records)
        low_block  = build_example_block(chosen, "low",  text_records)

        prompt = LLM_PROMPT_TEMPLATE.format(
            axis_id=axis_id,
            high_pole_block=high_block,
            low_pole_block=low_block,
        )

        # Save prompt file
        prompt_path = prompts_dir / f"{axis_id}_prompt.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt)
        logger.info("  Saved prompt: %s", prompt_path)

        if args.prompts_only:
            logger.info("  --prompts-only: skipping LLM call for %s", axis_id)
            continue

        # Call LLM judge
        logger.info("  Calling judge for %s …", axis_id)
        try:
            raw = call_judge(prompt, args.judge_model, judge_cache_dir)
            logger.info("  %s → axis_name: %s", axis_id, raw.get("axis_name", "(none)"))
            results.append(raw)
        except Exception as e:
            logger.error("  LLM call failed for %s: %s", axis_id, e)
            results.append({"axis_id": axis_id, "error": str(e)})

    # --- save extremes CSV ---
    extremes_df = pd.DataFrame(all_extreme_rows)
    extremes_df.to_csv(out_dir / f"layer{args.layer}_extremes.csv", index=False)
    logger.info("Saved extremes CSV.")

    # --- save LLM results ---
    if not args.prompts_only and results:
        results_path = out_dir / f"layer{args.layer}_llm_judge_results.jsonl"
        with open(results_path, "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        logger.info("Saved LLM judge results: %s", results_path)

        # Print summary
        print(f"\n=== Exp 8 — Layer {args.layer}: Meta-axis interpretations ===\n")
        for row in results:
            if "error" in row:
                print(f"  {row.get('axis_id', '?')}  ERROR: {row['error']}")
            else:
                print(f"  {row.get('axis_id', '?'):4s}  {row.get('axis_name', '?')}")
                print(f"        high: {row.get('high_pole_description', '')[:70]}")
                print(f"        low:  {row.get('low_pole_description', '')[:70]}")
                print(f"        knob: {row.get('ui_knob_label', '')}")
                print(f"        contamination: {row.get('style_or_semantics_contamination', '?')}  "
                      f"confidence: {row.get('confidence', '?')}")
                print()

    logger.info("All outputs written to %s", out_dir)
    print(f"\nDone. Outputs in: {out_dir}")
    if args.prompts_only:
        print("(Prompts written; LLM not called.  Re-run without --prompts-only.)")


if __name__ == "__main__":
    main()
