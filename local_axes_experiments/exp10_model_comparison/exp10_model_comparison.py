"""Experiment 10: Compare steering models A, B, C.

The central test of the Preverbal Affective Vector Projection framework.
For each of the 24 seed texts (8 emotions × 3 texts), three conditions are run:

  Condition A (named-emotion baseline):
      Δ = α_G * g + α_R * r_e

  Condition B (label-free meta-axes):
      Δ = α_G * g + Σ_k β_k * m_k
      where β_k = α_R * (r_e · m_k)
      — same intended affective target as A, but decomposed through shared axes.

  Condition C (hybrid full model):
      Δ = α_G * g + α_R * r_e + Σ_k β_k * m_k
      where β_k = α_R * (r_e · m_k)
      — adds the m_k projection ON TOP of r_e to see if meta-axes complement it.

Note on Condition B design:
  β_k = α_R * (r_e · m_k) ensures B targets the same region of affective space
  as A but through the shared m_k basis.  Any difference in output quality is
  therefore due to the decomposition, not a different intended target.
  The ε_e component (r_e perpendicular to the m_k subspace) is absent in B.

Judge criteria (scored 0–1 each, same as exp5 plus one new criterion):
  1. target_emotion_match
  2. meaning_preservation
  3. fluency
  4. style_contamination
  5. subtle_affective_state   (exp5 criterion: any preverbal nuance?)
  6. label_free_nuance [NEW]: does the rewrite express qualities BEYOND what
     the bare emotion label alone would convey?  Mixed, transitional, or
     nuanced states score high.  Plain "I felt sad" type outputs score low.

Key questions answered by this experiment:
  Q1: Can B match A on target_emotion_match?
      (If yes: m_k projection captures the named emotion well.)
  Q2: Does B > A on subtle_affective_state or label_free_nuance?
      (If yes: m_k decomposition generates richer affective texture.)
  Q3: Does C > B?
      (If yes: r_e adds information not captured by m_k alone — i.e., ε_e matters.)
  Q4: Does C > A?
      (If yes: augmenting r_e with m_k improves the steering quality.)

Outputs written to results/layer{L}/:
  results_A.jsonl / results_A.csv
  results_B.jsonl / results_B.csv
  results_C.jsonl / results_C.csv
  comparison_summary.csv    — mean scores per condition per criterion
  comparison_summary_by_emotion.csv

Run:
    python local_axes_experiments/exp10_model_comparison/exp10_model_comparison.py --conditions A B C
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.model_utils import load_config, load_model_and_tokenizer
from utils.steering_utils import steering_hook, flush_device_cache, append_jsonl_csv_row
from utils.text_utils import INSTRUCTION_TEMPLATE, make_instruction_prefix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR   = REPO_ROOT / "activation" / "emotion_rewrites"
EXP7_DIR  = REPO_ROOT / "local_axes_experiments" / "exp7_meta_axis_extraction" / "results"
EXP8_DIR  = REPO_ROOT / "local_axes_experiments" / "exp8_meta_axis_interpretation" / "results"
OUT_BASE  = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an expert in affective science and computational linguistics.
You evaluate emotional text rewrites for their affective quality.
Respond ONLY with valid JSON matching the requested schema.
"""

JUDGE_USER_TMPL = """\
## Task
Evaluate the following rewritten text.

## Original text (base)
{base_text}

## Rewritten text
{rewrite_text}

## Target emotion
{target_emotion}

## Note on evaluation
The output does NOT need to become a pure instance of the target emotion.
Context-dependent shifts (e.g. sadness steered toward joy yielding "bittersweet
relief") are valid and desirable.  Subtle, nameless, or transitional affective
states are especially valuable.

## Criteria (score each 0.0–1.0)

1. target_emotion_match: Does the rewrite express or move toward the target emotion
   (including blended/mixed states that lean toward it)?
2. meaning_preservation: Is the core propositional meaning of the original preserved?
3. fluency: Is the rewrite fluent and natural-sounding?
4. style_contamination: Does the rewrite avoid introducing obvious style artifacts
   (repetition, melodrama, boilerplate)?
   (1.0 = no artifact; 0.0 = unusable)
5. subtle_affective_state: Does the rewrite express a nameless or subtle pre-verbal
   affective state BEYOND what the emotion label alone conveys?
   Mixed, bittersweet, or transitional states score high.
   A plain "I felt [emotion]" style output scores low.
6. label_free_nuance [KEY CRITERION]: Does the rewrite express affective qualities
   that go beyond what you would predict from the emotion label alone?
   Examples of high-scoring qualities:
     - "resolved sadness" vs plain "sadness"
     - "anxious anticipation vs hopeful anticipation" vs plain "anticipation"
     - "raw vs integrated anger"
   Score 1.0 if the rewrite expresses clear affective nuance beyond the label.
   Score 0.5 if there is mild nuance.
   Score 0.0 if the output is a flat expression of the named emotion.

Respond ONLY with valid JSON:
{{
  "target_emotion_match": <float 0-1>,
  "meaning_preservation": <float 0-1>,
  "fluency": <float 0-1>,
  "style_contamination": <float 0-1>,
  "subtle_affective_state": <float 0-1>,
  "label_free_nuance": <float 0-1>,
  "reasoning": "<1-2 sentences on what affective quality (if any) goes beyond the label>"
}}
"""

JUDGE_SCHEMA: dict[str, Any] = {
    "name": "exp10_model_judge",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "target_emotion_match":   {"type": "number"},
            "meaning_preservation":   {"type": "number"},
            "fluency":                {"type": "number"},
            "style_contamination":    {"type": "number"},
            "subtle_affective_state": {"type": "number"},
            "label_free_nuance":      {"type": "number"},
            "reasoning":              {"type": "string"},
        },
        "required": [
            "target_emotion_match", "meaning_preservation", "fluency",
            "style_contamination", "subtle_affective_state", "label_free_nuance",
            "reasoning",
        ],
        "additionalProperties": False,
    },
}

JUDGE_ATTRS = [
    "target_emotion_match", "meaning_preservation", "fluency",
    "style_contamination", "subtle_affective_state", "label_free_nuance",
]

OUTPUT_FIELDNAMES = [
    "layer", "condition", "emotion", "source_id",
    "base_text", "rewrite_text",
    "alpha_g", "alpha_r",
    "delta_description",
    "target_emotion_match", "meaning_preservation", "fluency",
    "style_contamination", "subtle_affective_state", "label_free_nuance",
    "judged", "reasoning",
]

# ---------------------------------------------------------------------------
# Steering parameters (same as exp5)
# ---------------------------------------------------------------------------

DEFAULT_EMOTION_PARAMS: dict[str, dict[str, float]] = {
    "joy":          {"alpha_g": 2.0, "alpha_r": 1.5},
    "trust":        {"alpha_g": 2.0, "alpha_r": 1.5},
    "fear":         {"alpha_g": 2.0, "alpha_r": 1.5},
    "anticipation": {"alpha_g": 2.0, "alpha_r": 1.5},
    "surprise":     {"alpha_g": 2.0, "alpha_r": 1.5},
    "sadness":      {"alpha_g": 1.5, "alpha_r": 1.0},
    "disgust":      {"alpha_g": 1.5, "alpha_r": 1.0},
    "anger":        {"alpha_g": 1.5, "alpha_r": 1.0},
}

# Seed texts (identical to exp5/exp9)
SEED_TEXTS: list[dict[str, str]] = [
    {"seed_id": "seed_joy_1", "seed_emotion": "joy",
     "seed_text": "Today was a wonderful day. Everything went smoothly and I felt genuinely happy."},
    {"seed_id": "seed_joy_2", "seed_emotion": "joy",
     "seed_text": "I just got the news that I passed the exam. I could not stop smiling all afternoon."},
    {"seed_id": "seed_joy_3", "seed_emotion": "joy",
     "seed_text": "We spent the evening laughing and dancing and it felt like nothing else mattered."},
    {"seed_id": "seed_trust_1", "seed_emotion": "trust",
     "seed_text": "She has always kept her promises and I know she will come through for me again."},
    {"seed_id": "seed_trust_2", "seed_emotion": "trust",
     "seed_text": "He has never let me down in ten years and I have no reason to doubt him now."},
    {"seed_id": "seed_trust_3", "seed_emotion": "trust",
     "seed_text": "I handed the whole project over to her without hesitation because I know her work."},
    {"seed_id": "seed_fear_1", "seed_emotion": "fear",
     "seed_text": "Walking alone through the dark alley I felt a cold shiver run down my spine."},
    {"seed_id": "seed_fear_2", "seed_emotion": "fear",
     "seed_text": "Every creak of the floorboard made my heart race and I could not calm down."},
    {"seed_id": "seed_fear_3", "seed_emotion": "fear",
     "seed_text": "The test results were taking too long and all I could think about was the worst case."},
    {"seed_id": "seed_surprise_1", "seed_emotion": "surprise",
     "seed_text": "I opened the door and could not believe what was waiting for me on the other side."},
    {"seed_id": "seed_surprise_2", "seed_emotion": "surprise",
     "seed_text": "Out of nowhere my old friend called and said she was already standing outside my building."},
    {"seed_id": "seed_surprise_3", "seed_emotion": "surprise",
     "seed_text": "The envelope contained a cheque for an amount I never expected and I read it three times."},
    {"seed_id": "seed_sadness_1", "seed_emotion": "sadness",
     "seed_text": "The old photographs reminded me of everything I had lost and would never get back."},
    {"seed_id": "seed_sadness_2", "seed_emotion": "sadness",
     "seed_text": "I sat alone in the empty house after everyone had left and the silence was unbearable."},
    {"seed_id": "seed_sadness_3", "seed_emotion": "sadness",
     "seed_text": "No matter how hard I tried I could not hold back the tears when I finally said goodbye."},
    {"seed_id": "seed_disgust_1", "seed_emotion": "disgust",
     "seed_text": "The smell from the bin was overwhelming and the sight of it made my stomach turn."},
    {"seed_id": "seed_disgust_2", "seed_emotion": "disgust",
     "seed_text": "I found mold covering the entire back of the fridge and felt a wave of revulsion."},
    {"seed_id": "seed_disgust_3", "seed_emotion": "disgust",
     "seed_text": "Watching the video I felt sick. Some things simply should not exist in the world."},
    {"seed_id": "seed_anger_1", "seed_emotion": "anger",
     "seed_text": "He broke his word again and I am absolutely fed up with being taken for granted."},
    {"seed_id": "seed_anger_2", "seed_emotion": "anger",
     "seed_text": "They ignored every complaint I filed and now I am furious and done being patient."},
    {"seed_id": "seed_anger_3", "seed_emotion": "anger",
     "seed_text": "She talked over me again in the meeting and I could feel the rage building inside me."},
    {"seed_id": "seed_anticipation_1", "seed_emotion": "anticipation",
     "seed_text": "The results are due tomorrow and I keep checking my email every few minutes."},
    {"seed_id": "seed_anticipation_2", "seed_emotion": "anticipation",
     "seed_text": "Only three days left before the trip and I have been counting down since last week."},
    {"seed_id": "seed_anticipation_3", "seed_emotion": "anticipation",
     "seed_text": "The offer is still pending and every time my phone buzzes I hope it is the good news."},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def project_out(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    coeff = X @ g
    return X - np.outer(coeff, g)


def build_common_vectors(caa_pooled, caa, layer_idx, intensity_low_idx, intensity_high_idx, emotions_all):
    """Return g (D,) and intensity_dirs (E, D)."""
    g_raw = caa_pooled[:, layer_idx, :].mean(axis=0)
    g = unit_norm(g_raw.astype(np.float64)).astype(np.float32)
    intensity_dirs = unit_norm(
        caa[:, intensity_high_idx, layer_idx, :].astype(np.float64)
        - caa[:, intensity_low_idx, layer_idx, :].astype(np.float64)
    ).astype(np.float32)
    return g, intensity_dirs


def make_r_e(caa_pooled, layer_idx, e_idx, g):
    """Return unit-normed r_e (g-projected-out CAA direction for emotion e)."""
    r_raw = caa_pooled[e_idx, layer_idx, :].astype(np.float64)
    g64 = g.astype(np.float64)
    r_resid = r_raw - (r_raw @ g64) * g64
    return unit_norm(r_resid[np.newaxis, :])[0].astype(np.float32)


def build_delta(
    condition: str,
    g: np.ndarray,
    r_e: np.ndarray,
    M: np.ndarray,          # (K, D) meta-axes
    alpha_g: float,
    alpha_r: float,
) -> tuple[np.ndarray, str]:
    """Return (delta_vec, description_string) for condition A, B, or C.

    Condition B: β_k = α_R * (r_e · m_k)   → same affective target as A via m_k
    Condition C: both r_e and m_k projection active
    """
    if condition == "A":
        delta = alpha_g * g + alpha_r * r_e
        desc = f"alpha_g={alpha_g}*g + alpha_r={alpha_r}*r_e"
        return delta.astype(np.float32), desc

    # Compute beta_k for all axes
    beta_k = alpha_r * (M.astype(np.float64) @ r_e.astype(np.float64))  # (K,)
    mk_contribution = (beta_k[:, np.newaxis] * M.astype(np.float64)).sum(axis=0).astype(np.float32)

    if condition == "B":
        delta = alpha_g * g + mk_contribution
        terms = " + ".join(f"{b:.3f}*m{k+1}" for k, b in enumerate(beta_k))
        desc = f"alpha_g={alpha_g}*g + [{terms}]"
        return delta.astype(np.float32), desc

    if condition == "C":
        delta = alpha_g * g + alpha_r * r_e + mk_contribution
        terms = " + ".join(f"{b:.3f}*m{k+1}" for k, b in enumerate(beta_k))
        desc = f"alpha_g={alpha_g}*g + alpha_r={alpha_r}*r_e + [{terms}]"
        return delta.astype(np.float32), desc

    raise ValueError(f"Unknown condition: {condition}")


def generate_with_delta(model, tokenizer, device, base_text, delta_vec, layer_idx,
                        max_new_tokens, do_sample, temperature, top_p):
    prompt = make_instruction_prefix(base_text, tokenizer, INSTRUCTION_TEMPLATE)
    inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    hook_stats: dict = {"hook_calls": 0, "hook_prefill_calls": 0,
                        "hook_decode_calls": 0, "hook_applied_calls": 0}
    gen_kwargs: dict = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                        "pad_token_id": tokenizer.pad_token_id}
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    try:
        with torch.inference_mode():
            with steering_hook(model, layer_idx, delta_vec, "completion_tokens", hook_stats):
                out_ids = model.generate(**inputs, **gen_kwargs)
        prompt_width = inputs["input_ids"].shape[1]
        return tokenizer.decode(out_ids[0, prompt_width:], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False).strip()
    except torch.cuda.OutOfMemoryError:
        flush_device_cache()
        return "[OOM]"
    finally:
        del inputs
        if "out_ids" in locals():
            del out_ids


def call_judge(prompt: str, model: str, cache_dir: Path) -> dict:
    import hashlib
    h = hashlib.sha1()
    h.update(model.encode())
    h.update(b"|v1|")
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
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": JUDGE_SCHEMA},
    )
    raw = json.loads(resp.choices[0].message.content)
    with open(cache_path, "w") as f:
        json.dump(raw, f)
    return raw


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(summary_df: pd.DataFrame, layer: int, out_dir: Path) -> None:
    """Grouped bar chart comparing A vs B vs C across all judge criteria."""
    criteria = JUDGE_ATTRS
    conditions = summary_df["condition"].tolist()
    n_crit = len(criteria)
    x = np.arange(n_crit)
    width = 0.25
    offsets = np.linspace(-width, width, len(conditions))

    colours = {"A": "#4c72b0", "B": "#dd8452", "C": "#55a868"}

    fig, ax = plt.subplots(figsize=(max(8, n_crit * 1.2), 5))
    for i, cond in enumerate(conditions):
        row = summary_df[summary_df["condition"] == cond].iloc[0]
        vals = [float(row.get(c, 0.0)) for c in criteria]
        ax.bar(x + offsets[i], vals, width, label=f"Condition {cond}",
               color=colours.get(cond, "#999999"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(criteria, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score")
    ax.set_title(f"Layer {layer}: Model A vs B vs C — judge scores")
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="#cccccc", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / f"layer{layer}_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved comparison plot.")


def plot_delta_per_emotion(summary_by_emo: pd.DataFrame, layer: int, out_dir: Path) -> None:
    """Scatter: per-emotion B-minus-A delta on label_free_nuance and subtle_affective_state."""
    emotions_in = summary_by_emo["emotion"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, criterion in zip(axes, ["label_free_nuance", "subtle_affective_state"]):
        A = summary_by_emo[summary_by_emo["condition"] == "A"].set_index("emotion")[criterion]
        B = summary_by_emo[summary_by_emo["condition"] == "B"].set_index("emotion")[criterion]
        delta = (B - A).dropna()

        colours = [
            "#2ca02c" if v > 0 else "#d62728" for v in delta.values
        ]
        ax.barh(delta.index, delta.values, color=colours, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"B – A  ({criterion})")
        ax.set_title(f"B minus A: {criterion}")
        ax.set_xlim(-0.5, 0.5)

    fig.suptitle(f"Layer {layer}: Per-emotion improvement of B over A", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"layer{layer}_delta_B_minus_A.png", dpi=150)
    plt.close(fig)
    logger.info("Saved delta B-A plot.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--conditions", nargs="+", choices=["A", "B", "C"],
                   default=["A", "B", "C"], help="Which conditions to run.")
    p.add_argument("--emo-act",  default=str(ACT_DIR / "emotion_intensity_residual_stream.npy"))
    p.add_argument("--emo-info", default=str(ACT_DIR / "emotion_intensity_residual_stream_info.json"))
    p.add_argument("--neu-act",  default=str(ACT_DIR / "neutral_paraphrase_residual_stream.npy"))
    p.add_argument("--caa-directions", default=str(ACT_DIR / "caa_emotion_directions.npz"))
    p.add_argument("--exp7-dir", default=str(EXP7_DIR))
    p.add_argument("--model-config", default=str(REPO_ROOT / "model.yaml"))
    p.add_argument("--emotions", nargs="+", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--do-sample", action="store_true", default=False)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default="cuda")
    p.add_argument("--judge-model", default="gpt-5.4")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--out-dir", default=str(OUT_BASE))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir) / f"layer{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_cache_dir = out_dir / "judge_cache"
    judge_cache_dir.mkdir(parents=True, exist_ok=True)

    # --- metadata ---
    with open(args.emo_info) as f:
        info = json.load(f)
    emotions_all: list[str] = info["emotion_order"]
    layer_indices: list[int] = info["layer_indices"]
    if args.layer not in layer_indices:
        raise ValueError(f"Layer {args.layer} not in {layer_indices}")
    layer_idx = layer_indices.index(args.layer)

    intensity_order: list[str] = info["intensity_order"]
    intensity_low_idx  = intensity_order.index("low")
    intensity_high_idx = intensity_order.index("high")

    emotions = [e for e in emotions_all if args.emotions is None or e in args.emotions]
    logger.info("Emotions: %s", emotions)
    logger.info("Conditions: %s", args.conditions)

    # --- CAA directions ---
    caa_npz = np.load(args.caa_directions)
    caa_pooled: np.ndarray = caa_npz["caa_pooled"]
    caa: np.ndarray = caa_npz["caa"]

    g, intensity_dirs = build_common_vectors(
        caa_pooled, caa, layer_idx, intensity_low_idx, intensity_high_idx, emotions_all
    )

    # --- meta-axes (exp7) ---
    exp7_dir = Path(args.exp7_dir)
    M_path = exp7_dir / f"layer{args.layer}_meta_axes.npy"
    if not M_path.exists():
        raise FileNotFoundError(f"exp7 meta-axes not found: {M_path}\nRun exp7 first.")
    M = np.load(M_path).astype(np.float64)   # (K, D)
    with open(exp7_dir / f"layer{args.layer}_meta_axes_info.json") as f:
        axes_info = json.load(f)
    axis_ids: list[str] = axes_info["axis_ids"]
    logger.info("Loaded %d meta-axes: %s", len(axis_ids), axis_ids)

    # --- load model ---
    if not args.dry_run:
        model_cfg = load_config(args.model_config)
        model, tokenizer = load_model_and_tokenizer(model_cfg, args.device)
        model_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        model = tokenizer = model_device = None
        logger.info("[dry-run] Skipping model load.")

    emotion_to_idx = {e: i for i, e in enumerate(emotions_all)}

    # --- result files and resume state ---
    result_files: dict[str, tuple[Path, Path]] = {}
    done_keys: dict[str, set] = {}
    for cond in args.conditions:
        rj = out_dir / f"results_{cond}.jsonl"
        rc = out_dir / f"results_{cond}.csv"
        result_files[cond] = (rj, rc)
        done: set[tuple] = set()
        if rj.exists():
            with open(rj) as f:
                for line in f:
                    row = json.loads(line)
                    done.add((row["emotion"], row["source_id"]))
            logger.info("Condition %s: %d rows already done.", cond, len(done))
        done_keys[cond] = done

    # --- main loop ---
    for seed in SEED_TEXTS:
        emotion = seed["seed_emotion"]
        if emotion not in emotions:
            continue

        source_id = seed["seed_id"]
        base_text  = seed["seed_text"]
        e_idx = emotion_to_idx[emotion]

        ep = DEFAULT_EMOTION_PARAMS.get(emotion, {"alpha_g": 2.0, "alpha_r": 1.5})
        alpha_g = ep["alpha_g"]
        alpha_r = ep["alpha_r"]

        r_e = make_r_e(caa_pooled, layer_idx, e_idx, g)

        for cond in args.conditions:
            cache_key = (emotion, source_id)
            if cache_key in done_keys[cond]:
                logger.debug("Skip [%s/%s/%s] (done)", cond, emotion, source_id)
                continue

            delta_vec, delta_desc = build_delta(cond, g, r_e, M, alpha_g, alpha_r)

            # --- generate ---
            if args.dry_run:
                rewrite_text = f"[dry-run cond={cond}] {base_text[:60]}"
            else:
                rewrite_text = generate_with_delta(
                    model, tokenizer, model_device,
                    base_text, delta_vec, layer_idx,
                    args.max_new_tokens, args.do_sample,
                    args.temperature, args.top_p,
                )

            logger.info(
                "  [%s/cond=%s] %s → %s",
                emotion, cond, base_text[:40], rewrite_text[:50],
            )

            # --- judge ---
            if args.skip_judge or args.dry_run:
                judge_scores: dict[str, float] = {a: 0.0 for a in JUDGE_ATTRS}
                reasoning = ""
                judged = False
            else:
                judge_prompt = JUDGE_USER_TMPL.format(
                    base_text=base_text,
                    rewrite_text=rewrite_text,
                    target_emotion=emotion,
                )
                try:
                    raw = call_judge(judge_prompt, args.judge_model, judge_cache_dir)
                    judge_scores = {k: float(np.clip(raw.get(k, 0.0), 0.0, 1.0))
                                    for k in JUDGE_ATTRS}
                    reasoning = str(raw.get("reasoning", ""))
                    judged = True
                except Exception as e:
                    logger.error("Judge failed for [%s/%s/%s]: %s", cond, emotion, source_id, e)
                    judge_scores = {a: float("nan") for a in JUDGE_ATTRS}
                    reasoning = f"ERROR: {e}"
                    judged = False

            row = {
                "layer": args.layer,
                "condition": cond,
                "emotion": emotion,
                "source_id": source_id,
                "base_text": base_text,
                "rewrite_text": rewrite_text,
                "alpha_g": alpha_g,
                "alpha_r": alpha_r,
                "delta_description": delta_desc,
                **judge_scores,
                "judged": judged,
                "reasoning": reasoning,
            }
            rj_path, rc_path = result_files[cond]
            append_jsonl_csv_row(rj_path, rc_path, OUTPUT_FIELDNAMES, row)
            done_keys[cond].add(cache_key)

    # --- summary statistics ---
    all_dfs: list[pd.DataFrame] = []
    for cond in args.conditions:
        rj_path, _ = result_files[cond]
        if not rj_path.exists():
            continue
        df = pd.read_json(rj_path, lines=True)
        if df.empty:
            continue
        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        judged = combined[combined["judged"] == True]  # noqa: E712

        # Overall summary
        summary_rows = []
        for cond in args.conditions:
            sub = judged[judged["condition"] == cond]
            if sub.empty:
                continue
            means = sub[JUDGE_ATTRS].mean()
            summary_rows.append({"condition": cond,
                                  **{k: round(float(v), 4) for k, v in means.items()}})
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "comparison_summary.csv", index=False)

        # Per-emotion summary
        emo_rows = []
        for cond in args.conditions:
            for emo in emotions:
                sub = judged[(judged["condition"] == cond) & (judged["emotion"] == emo)]
                if sub.empty:
                    continue
                means = sub[JUDGE_ATTRS].mean()
                emo_rows.append({"condition": cond, "emotion": emo,
                                 **{k: round(float(v), 4) for k, v in means.items()}})
        emo_df = pd.DataFrame(emo_rows)
        emo_df.to_csv(out_dir / "comparison_summary_by_emotion.csv", index=False)

        # Print results
        print(f"\n=== Exp 10 — Layer {args.layer}: Model A vs B vs C ===\n")
        print(summary_df.to_string(index=False))

        # Answer the four key questions
        if len(summary_df) >= 2:
            print("\n=== Key Questions ===\n")
            cond_map = {row["condition"]: row for _, row in summary_df.iterrows()}

            def diff(c1: str, c2: str, criterion: str) -> str:
                a = cond_map.get(c1, {}).get(criterion, float("nan"))
                b = cond_map.get(c2, {}).get(criterion, float("nan"))
                if a != a or b != b:
                    return "N/A"
                d = b - a
                sign = "+" if d >= 0 else ""
                return f"{sign}{d:.3f}"

            if "A" in cond_map and "B" in cond_map:
                print(f"  Q1 (B vs A on target_emotion_match): {diff('A','B','target_emotion_match')}")
                print(f"  Q2a (B vs A on subtle_affective_state): {diff('A','B','subtle_affective_state')}")
                print(f"  Q2b (B vs A on label_free_nuance): {diff('A','B','label_free_nuance')}")
            if "B" in cond_map and "C" in cond_map:
                print(f"  Q3 (C vs B on label_free_nuance): {diff('B','C','label_free_nuance')}")
            if "A" in cond_map and "C" in cond_map:
                print(f"  Q4 (C vs A on label_free_nuance): {diff('A','C','label_free_nuance')}")

        # Plots
        if not summary_df.empty:
            plot_comparison(summary_df, args.layer, out_dir)
        if not emo_df.empty and "A" in args.conditions and "B" in args.conditions:
            plot_delta_per_emotion(emo_df, args.layer, out_dir)

    logger.info("Done. Results written to %s", out_dir)


if __name__ == "__main__":
    main()
