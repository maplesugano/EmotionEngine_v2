"""Experiment 8 v2: Contrastive meta-axis labeling — all axes shown simultaneously.

Motivation: exp8 v1 labeled each axis in isolation, and all 5 converged on
"engaged vs detached". Hypothesis: the LLM can only distinguish axes when it
sees them side-by-side and is forced to assign *distinct* names.

Approach: present all K meta-axes in a single LLM prompt, showing high/low
examples for each, and explicitly require that every axis_name be different.

Run:
    python local_axes_experiments/exp8_meta_axis_interpretation/exp8v2_contrastive_labeling.py --layer 13 --n-extremes 6
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

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
# Prompt templates
# ---------------------------------------------------------------------------

LLM_SYSTEM = """\
You are an expert in affective science and computational linguistics.
Respond ONLY with valid JSON matching the requested schema.
"""

# One block per axis inside the joint prompt
AXIS_BLOCK_TEMPLATE = """\
=== META-AXIS {axis_id} ===

-- HIGH-POLE examples (texts that score HIGH on {axis_id}) --
{high_pole_block}

-- LOW-POLE examples (texts that score LOW on {axis_id}) --
{low_pole_block}
"""

EXAMPLE_BLOCK_TEMPLATE = """\
[{i}] emotion={emotion}  score={score:.4f}
  Emotion rewrite: {emotion_rewrite}

"""

LLM_PROMPT_TEMPLATE = """\
You are an expert in affective science and computational linguistics.

Below are {K} latent axes extracted from the internal activations of a language model
(Llama-3.1-8B-Instruct, layer {layer}).

All axes were extracted from the SAME residual activation space, after removing:
  1. The shared emotionalization direction g
  2. The per-emotion intensity direction u_e

Because the axes are mathematically orthogonal to each other and to g/u_e, they
are expected to capture DISTINCT pre-verbal affective dimensions.

## CRITICAL CONSTRAINT
You MUST assign a DIFFERENT axis_name to each axis.
If two axes seem superficially similar, look harder for what makes them different.
Compare the axes against each other, not just high vs low within each axis.

## Your tasks for EACH axis
1. Compare its high/low examples to ALL OTHER AXES — what does THIS axis capture
   that the others do NOT?
2. Name the unique affective dimension it captures.
3. Suggest a short UI knob label (e.g. "resolved ↔ conflicted").
4. Rate contamination (low/medium/high) and confidence (1–5).

---

{all_axis_blocks}

---

## OUTPUT FORMAT

Respond ONLY with a valid JSON object of this exact shape:

{{
  "axes": [
    {{
      "axis_id": "<e.g. m1>",
      "axis_name": "<high-pole> ↔ <low-pole>  [MUST be unique across all axes]",
      "what_distinguishes_from_others": "<1-2 sentences: how is this axis different from the other {Km1} axes?>",
      "high_pole_description": "<what characterises the HIGH-pole examples>",
      "low_pole_description": "<what characterises the LOW-pole examples>",
      "ui_knob_label": "<2-4 word high label> ↔ <2-4 word low label>",
      "preverbal_affective_interpretation": "<which pre-verbal affective dimension this captures>",
      "style_or_semantics_contamination": "low|medium|high",
      "confidence": <int 1-5>
    }}
  ],
  "overall_summary": "<2-3 sentences: do the axes capture genuinely distinct dimensions or are they variations of one theme?>"
}}
"""


# ---------------------------------------------------------------------------
# Helpers (copied from exp8 v1)
# ---------------------------------------------------------------------------

def unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def project_out(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    coeff = X @ g
    return X - np.outer(coeff, g)


def build_per_emotion_residuals(
    emo_act: np.ndarray,
    neu_act: np.ndarray,
    caa_pooled: np.ndarray,
    caa: np.ndarray,
    layer_idx: int,
    intensity_low_idx: int = 0,
    intensity_high_idx: int = 2,
    CHUNK: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    source_mean = delta_resid.mean(axis=1, keepdims=True)
    delta_resid -= source_mean

    return delta_resid, g, intensity_dirs


def load_text_records(path: Path) -> dict[str, dict]:
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
    scores: np.ndarray,
    source_ids: list[str],
    emotion: str,
    n: int,
) -> list[dict]:
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
    intensity_idx: int = 2,
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
            emotion_rewrite=rewrite,
        ))
    return "".join(blocks) if blocks else "(no examples)"


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def call_judge(prompt: str, model: str, cache_dir: Path) -> dict:
    import hashlib
    h = hashlib.sha1()
    h.update(model.encode())
    h.update(b"|v2|")
    h.update(prompt.encode("utf-8"))
    cache_path = cache_dir / f"v2_{h.hexdigest()[:24]}.json"

    if cache_path.exists():
        logger.info("  Cache hit: %s", cache_path.name)
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
        json.dump(raw, f, ensure_ascii=False, indent=2)
    return raw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--exp7-dir", default=str(EXP7_DIR))
    p.add_argument("--emo-act",
                   default=str(ACT_DIR / "emotion_intensity_residual_stream.npy"))
    p.add_argument("--emo-info",
                   default=str(ACT_DIR / "emotion_intensity_residual_stream_info.json"))
    p.add_argument("--neu-act",
                   default=str(ACT_DIR / "neutral_paraphrase_residual_stream.npy"))
    p.add_argument("--caa-directions",
                   default=str(ACT_DIR / "caa_emotion_directions.npz"))
    p.add_argument("--text-meta",
                   default=str(ACT_DIR / "emotion_intensity_residual_stream_meta.jsonl"))
    p.add_argument("--n-extremes", type=int, default=6,
                   help="Number of extreme examples per pole per axis.")
    p.add_argument("--judge-model", default="gpt-4.1")
    p.add_argument("--prompts-only", action="store_true",
                   help="Write prompt file but do not call the LLM.")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_cache_dir = out_dir / "judge_cache"
    judge_cache_dir.mkdir(parents=True, exist_ok=True)

    # --- load exp7 meta-axes ---
    exp7_dir = Path(args.exp7_dir)
    axes_path = exp7_dir / f"layer{args.layer}_meta_axes.npy"
    info_path = exp7_dir / f"layer{args.layer}_meta_axes_info.json"

    if not axes_path.exists():
        raise FileNotFoundError(f"exp7 meta-axes not found: {axes_path}")

    M = np.load(axes_path).astype(np.float64)
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
    neu_raw = np.memmap(args.neu_act, dtype=neu_info["dtype"], mode="r",
                        shape=tuple(neu_info["shape"]))

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
    N, E, D = delta_resid.shape
    logger.info("  delta_resid: %s", delta_resid.shape)

    logger.info("Loading text records …")
    text_records = load_text_records(Path(args.text_meta))
    logger.info("  %d text records loaded.", len(text_records))

    # --- build all axis blocks for the joint prompt ---
    all_axis_blocks_str = ""
    for k_idx, axis_id in enumerate(axis_ids):
        m_k = M[k_idx].astype(np.float32)
        logger.info("Computing extremes for %s …", axis_id)

        all_entries: list[dict] = []
        for e_idx, emo in enumerate(emotions):
            scores = delta_resid[:, e_idx, :].astype(np.float32) @ m_k
            entries = find_extreme_examples(scores, source_ids, emo, n=args.n_extremes)
            for entry in entries:
                entry["axis_id"] = axis_id
            all_entries.extend(entries)

        chosen = select_diverse_extremes(all_entries, n_total=args.n_extremes,
                                         emotions=emotions)

        high_block = build_example_block(chosen, "high", text_records)
        low_block  = build_example_block(chosen, "low",  text_records)

        all_axis_blocks_str += AXIS_BLOCK_TEMPLATE.format(
            axis_id=axis_id,
            high_pole_block=high_block,
            low_pole_block=low_block,
        )

    # --- build and save joint prompt ---
    prompt = LLM_PROMPT_TEMPLATE.format(
        K=K,
        layer=args.layer,
        all_axis_blocks=all_axis_blocks_str,
        Km1=K - 1,
    )

    prompt_path = out_dir / f"layer{args.layer}_v2_joint_prompt.txt"
    with open(prompt_path, "w") as f:
        f.write(prompt)
    logger.info("Joint prompt saved: %s  (%d chars)", prompt_path, len(prompt))

    if args.prompts_only:
        print(f"Prompt written to: {prompt_path}")
        print("(--prompts-only: LLM not called)")
        return

    # --- single joint LLM call ---
    logger.info("Calling judge (joint prompt) …")
    try:
        raw = call_judge(prompt, args.judge_model, judge_cache_dir)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise

    # --- save results ---
    results_path = out_dir / f"layer{args.layer}_v2_llm_judge_results.json"
    with open(results_path, "w") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    logger.info("Saved results: %s", results_path)

    # --- print summary ---
    print(f"\n=== Exp 8 v2 — Layer {args.layer}: Contrastive meta-axis labeling ===\n")
    axes_out = raw.get("axes", [])
    for row in axes_out:
        print(f"  {row.get('axis_id', '?'):4s}  {row.get('axis_name', '?')}")
        print(f"        knob:         {row.get('ui_knob_label', '')}")
        print(f"        distinguishes: {row.get('what_distinguishes_from_others', '')[:90]}")
        print(f"        preverbal:    {row.get('preverbal_affective_interpretation', '')[:80]}")
        print(f"        contamination: {row.get('style_or_semantics_contamination', '?')}  "
              f"confidence: {row.get('confidence', '?')}")
        print()

    overall = raw.get("overall_summary", "")
    if overall:
        print(f"Overall summary:\n  {overall}\n")

    print(f"Full results: {results_path}")


if __name__ == "__main__":
    main()
