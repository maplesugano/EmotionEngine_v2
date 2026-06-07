"""Experiment 9: Steer along shared meta-axes m_k vs emotion-specific u_{e,k}.

Extends exp5 with a second steering condition (condition M) to test whether
shared meta-axes produce steering quality equal to or better than
emotion-specific local PCs.

Two conditions for each (emotion, seed_text, pole):

  Condition S (specific):  Δ = α_G * g + α_R * r_e + β * u_{e,1}
    — emotion-specific local PC (same as exp5, run with --condition s)

  Condition M (meta/shared):  Δ = α_G * g + α_R * r_e + β * m_1
    — shared meta-axis m_1 from exp7 (run with --condition m)

Both conditions use the same α_G, α_R, β values and the same seed texts
as exp5 (SEED_TEXTS: 8 emotions × 3 texts = 24 base texts).

The judge uses the same 6 criteria as exp5 (target_emotion_match,
local_axis_match, meaning_preservation, fluency, style_contamination,
subtle_affective_state), but the axis_name / pole descriptions come from:
  - Condition S: exp3 emotion-specific interpretation
  - Condition M: exp8 cross-emotion meta-axis interpretation

After scoring, a comparison_summary.csv is produced with per-condition means.

Key question:
  Does condition M (shared m_k) produce steering that is as good as or better
  than condition S (emotion-specific u_{e,k}) on subtle_affective_state?

If yes:  shared meta-axes suffice — the DecomposedSteering framework can use
         m_k instead of u_{e,k}, simplifying the system.
If no:   emotion-specific axes carry information not captured by m_k alone.

Run:
    # Condition M (new):
    python local_axes_experiments/exp9_meta_axis_steering/exp9_meta_axis_steering.py --condition m --use-seeds

    # Condition S (replication of exp5, optional baseline):
    python local_axes_experiments/exp9_meta_axis_steering/exp9_meta_axis_steering.py --condition s --use-seeds
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
from sklearn.decomposition import PCA

from utils.judge_utils import JudgeConfig, judge_one
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
EXP3_DIR  = REPO_ROOT / "local_axes_experiments" / "exp3_local_axis_interpretation" / "results"
OUT_BASE  = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Judge prompt (same as exp5)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an expert in affective science and computational linguistics.
You evaluate texts for their emotional quality along a specific affective axis.
Respond ONLY with valid JSON matching the requested schema.
"""

JUDGE_USER_TMPL = """\
## Task
Evaluate the following rewritten text according to the criteria below.

## Original text (base)
{base_text}

## Rewritten text
{rewrite_text}

## Target emotion
{target_emotion}

## Affective axis steered
Name: {axis_name}
Type: {axis_type}
High-pole description: {high_pole_description}
Low-pole description: {low_pole_description}

## Expected pole
{expected_pole}

## Important note on evaluation
The output does NOT need to become a pure instance of the target emotion.
It may express a context-dependent shift from the seed emotion toward the
target/local pole — for example, a sadness seed steered toward joy may yield
"bittersweet relief" or "peaceful acceptance" rather than outright joy.
Reward such subtle, mixed affective states if they are consistent with the
expected local pole.  The most important criteria are (2) local_axis_match
and (6) subtle_affective_state.

## Criteria (score each 0.0–1.0)
1. target_emotion_match: Does the rewrite express or move toward the target
   emotion (including blended/mixed states that lean toward it)?
2. local_axis_match [MOST IMPORTANT]: Does the rewrite move toward the
   EXPECTED POLE of the affective axis?
   (1.0 = clearly moves toward expected pole; 0.0 = moves away or no movement)
3. meaning_preservation: Is the core propositional meaning of the original preserved?
4. fluency: Is the rewrite fluent and natural-sounding?
5. style_contamination: Does the rewrite introduce an obvious style artifact
    unrelated to the affective axis?
    (1.0 = no artifact; 0.0 = unusable due to contamination)
6. subtle_affective_state [VERY IMPORTANT]: Does the rewrite express a
   nameless or subtle pre-verbal affective state beyond what the emotion label
   alone conveys?  Mixed, bittersweet, or transitional states score high.

Respond ONLY with valid JSON:
{{
  "target_emotion_match": <float 0-1>,
  "local_axis_match": <float 0-1>,
  "meaning_preservation": <float 0-1>,
  "fluency": <float 0-1>,
  "style_contamination": <float 0-1>,
  "subtle_affective_state": <float 0-1>,
  "reasoning": "<one sentence explaining local_axis_match score>"
}}
"""

JUDGE_SCHEMA: dict[str, Any] = {
    "name": "exp9_axis_judge",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "target_emotion_match":  {"type": "number"},
            "local_axis_match":      {"type": "number"},
            "meaning_preservation":  {"type": "number"},
            "fluency":               {"type": "number"},
            "style_contamination":   {"type": "number"},
            "subtle_affective_state":{"type": "number"},
            "reasoning":             {"type": "string"},
        },
        "required": [
            "target_emotion_match", "local_axis_match", "meaning_preservation",
            "fluency", "style_contamination", "subtle_affective_state", "reasoning",
        ],
        "additionalProperties": False,
    },
}

JUDGE_ATTRS = [
    "target_emotion_match", "local_axis_match", "meaning_preservation",
    "fluency", "style_contamination", "subtle_affective_state",
]

ATTR_RANGES = {attr: (0.0, 1.0) for attr in JUDGE_ATTRS}

# ---------------------------------------------------------------------------
# Per-emotion steering parameters (same as exp5)
# ---------------------------------------------------------------------------

DEFAULT_EMOTION_PARAMS: dict[str, dict[str, float]] = {
    "joy":          {"alpha_g": 2.0, "alpha_r": 1.5, "beta": 1.0},
    "trust":        {"alpha_g": 2.0, "alpha_r": 1.5, "beta": 1.0},
    "fear":         {"alpha_g": 2.0, "alpha_r": 1.5, "beta": 1.0},
    "anticipation": {"alpha_g": 2.0, "alpha_r": 1.5, "beta": 1.0},
    "surprise":     {"alpha_g": 2.0, "alpha_r": 1.5, "beta": 0.75},
    "sadness":      {"alpha_g": 1.5, "alpha_r": 1.0, "beta": 0.5},
    "disgust":      {"alpha_g": 1.5, "alpha_r": 1.0, "beta": 0.3},
    "anger":        {"alpha_g": 1.5, "alpha_r": 1.0, "beta": 0.35},
}

# Seed texts: identical to exp5
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

OUTPUT_FIELDNAMES = [
    "layer", "condition", "emotion", "axis_type", "axis_id",
    "pole", "source_id", "base_text", "rewrite_text",
    "alpha_g", "alpha_r", "beta",
    "target_emotion_match", "local_axis_match", "meaning_preservation",
    "fluency", "style_contamination", "subtle_affective_state",
    "judged", "reasoning",
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


def build_per_emotion_residuals(
    emo_act, neu_act, caa_pooled, caa,
    layer_idx, intensity_low_idx, intensity_high_idx,
    CHUNK=512,
):
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


def load_exp3_axis_info(exp3_dir: Path, layer: int) -> dict[tuple[str, int], dict]:
    """Load emotion-specific axis interpretations from exp3 JSONL."""
    axis_info: dict[tuple[str, int], dict] = {}
    jsonl_path = None
    for candidate in sorted(exp3_dir.glob("llm_judge_results*.jsonl"), reverse=True):
        jsonl_path = candidate
        break
    if jsonl_path is None or not jsonl_path.exists():
        logger.warning("exp3 results not found in %s — axis names will be empty.", exp3_dir)
        return axis_info
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            for entry in row.get("response_data", []):
                key = (entry["emotion"], int(entry["pc"]))
                axis_info[key] = {
                    "axis_name": entry.get("axis_name", ""),
                    "high_pole_description": entry.get("high_pole_description", ""),
                    "low_pole_description": entry.get("low_pole_description", ""),
                }
    return axis_info


def load_exp8_axis_info(exp8_dir: Path, layer: int) -> dict[str, dict]:
    """Load shared meta-axis interpretations from exp8 JSONL."""
    axis_info: dict[str, dict] = {}
    jsonl_path = exp8_dir / f"layer{layer}_llm_judge_results.jsonl"
    if not jsonl_path.exists():
        logger.warning("exp8 results not found: %s — axis names will be empty.", jsonl_path)
        return axis_info
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            axis_id = row.get("axis_id", "")
            if axis_id:
                axis_info[axis_id] = {
                    "axis_name": row.get("axis_name", axis_id),
                    "high_pole_description": row.get("high_pole_description", ""),
                    "low_pole_description": row.get("low_pole_description", ""),
                }
    return axis_info


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=13)
    p.add_argument(
        "--condition", choices=["s", "m", "both"], default="m",
        help=(
            "s = emotion-specific u_{e,1} (replicates exp5); "
            "m = shared meta-axis m_1 (new); "
            "both = run both conditions."
        ),
    )
    p.add_argument(
        "--meta-axis-idx", type=int, default=1,
        help="Which meta-axis to steer along (1-based index into m_k). Default: 1 (m1).",
    )
    p.add_argument("--emo-act",  default=str(ACT_DIR / "emotion_intensity_residual_stream.npy"))
    p.add_argument("--emo-info", default=str(ACT_DIR / "emotion_intensity_residual_stream_info.json"))
    p.add_argument("--neu-act",  default=str(ACT_DIR / "neutral_paraphrase_residual_stream.npy"))
    p.add_argument("--caa-directions", default=str(ACT_DIR / "caa_emotion_directions.npz"))
    p.add_argument("--exp7-dir", default=str(EXP7_DIR))
    p.add_argument("--exp8-dir", default=str(EXP8_DIR))
    p.add_argument("--exp3-dir", default=str(EXP3_DIR))
    p.add_argument("--model-config", default=str(REPO_ROOT / "model.yaml"))
    p.add_argument("--use-seeds", action="store_true", default=True)
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

    # --- load metadata ---
    with open(args.emo_info) as f:
        info = json.load(f)
    emotions_all: list[str] = info["emotion_order"]
    layer_indices: list[int] = info["layer_indices"]
    source_ids: list[str] = info["source_ids"]

    if args.layer not in layer_indices:
        raise ValueError(f"Layer {args.layer} not in {layer_indices}")
    layer_idx = layer_indices.index(args.layer)

    intensity_order: list[str] = info["intensity_order"]
    intensity_low_idx  = intensity_order.index("low")
    intensity_high_idx = intensity_order.index("high")

    emotions = [e for e in emotions_all if args.emotions is None or e in args.emotions]
    logger.info("Processing emotions: %s", emotions)

    # --- load activations (mmap) ---
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
    logger.info("Building residuals for layer %d …", args.layer)
    delta_resid, g, intensity_dirs = build_per_emotion_residuals(
        emo_raw, neu_raw, caa_pooled, caa,
        layer_idx, intensity_low_idx, intensity_high_idx,
    )

    # --- load meta-axes (exp7) ---
    exp7_dir = Path(args.exp7_dir)
    M_path = exp7_dir / f"layer{args.layer}_meta_axes.npy"
    if not M_path.exists():
        raise FileNotFoundError(f"exp7 meta-axes not found: {M_path}\nRun exp7 first.")
    M = np.load(M_path).astype(np.float32)    # (K, D)
    with open(exp7_dir / f"layer{args.layer}_meta_axes_info.json") as f:
        axes_info = json.load(f)
    meta_axis_ids: list[str] = axes_info["axis_ids"]

    if args.meta_axis_idx < 1 or args.meta_axis_idx > len(M):
        raise ValueError(f"--meta-axis-idx {args.meta_axis_idx} out of range [1..{len(M)}]")
    m_k = M[args.meta_axis_idx - 1]   # (D,)
    meta_axis_id = meta_axis_ids[args.meta_axis_idx - 1]
    logger.info("Using shared meta-axis: %s", meta_axis_id)

    # --- load axis interpretations ---
    exp3_axis_info = load_exp3_axis_info(Path(args.exp3_dir), args.layer)
    exp8_axis_info = load_exp8_axis_info(Path(args.exp8_dir), args.layer)
    logger.info("Loaded exp3 axes: %d  exp8 axes: %d",
                len(exp3_axis_info), len(exp8_axis_info))

    # Fallback axis descriptions if exp8 not available
    meta_axis_meta = exp8_axis_info.get(meta_axis_id, {
        "axis_name": meta_axis_id,
        "high_pole_description": f"{meta_axis_id} high pole",
        "low_pole_description":  f"{meta_axis_id} low pole",
    })

    # --- load model ---
    if not args.dry_run:
        model_cfg = load_config(args.model_config)
        model, tokenizer = load_model_and_tokenizer(model_cfg, args.device)
        model_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        model = tokenizer = model_device = None
        logger.info("[dry-run] Skipping model load.")

    # --- judge config ---
    judge_cfg = JudgeConfig(
        judge_attrs=JUDGE_ATTRS,
        attr_ranges=ATTR_RANGES,
        judge_system=JUDGE_SYSTEM,
        judge_user_tmpl="{text}",
        judge_schema=JUDGE_SCHEMA,
        cache_dir=judge_cache_dir,
        judge_model=args.judge_model,
        prompt_version="v1",
        dry_run=args.dry_run or args.skip_judge,
    )

    emotion_to_idx = {e: i for i, e in enumerate(emotions_all)}

    # Decide which conditions to run
    conditions_to_run: list[str] = (
        ["s", "m"] if args.condition == "both" else [args.condition]
    )

    # --- per-condition output files ---
    results_files: dict[str, tuple[Path, Path]] = {}
    done_keys: dict[str, set] = {}

    for cond in conditions_to_run:
        rj = out_dir / f"results_{cond}.jsonl"
        rc = out_dir / f"results_{cond}.csv"
        results_files[cond] = (rj, rc)
        done: set[tuple] = set()
        if rj.exists():
            with open(rj) as f:
                for line in f:
                    row = json.loads(line)
                    done.add((row["emotion"], row["axis_id"], row["pole"], row["source_id"]))
            logger.info("Condition %s: %d rows already done.", cond, len(done))
        done_keys[cond] = done

    # --- main loop ---
    for emotion in emotions:
        e_idx = emotion_to_idx[emotion]
        ep = DEFAULT_EMOTION_PARAMS.get(emotion, {"alpha_g": 2.0, "alpha_r": 1.5, "beta": 1.0})
        alpha_g_e = ep["alpha_g"]
        alpha_r_e = ep["alpha_r"]
        beta_e    = ep["beta"]

        # r_e: CAA direction with g projected out
        r_e_raw = caa_pooled[e_idx, layer_idx, :].astype(np.float64)
        g64 = g.astype(np.float64)
        r_e_resid = r_e_raw - (r_e_raw @ g64) * g64
        r_e = (r_e_resid / max(np.linalg.norm(r_e_resid), 1e-12)).astype(np.float32)

        # Fit local PCA for condition S
        if "s" in conditions_to_run:
            X_e = delta_resid[:, e_idx, :].astype(np.float64)
            X_c = X_e - X_e.mean(axis=0)
            n_comp = min(5, X_e.shape[0] - 1, X_e.shape[1])
            local_pca = PCA(n_components=n_comp, random_state=0)
            local_pca.fit(X_c)
            u_e1 = local_pca.components_[0].astype(np.float32)

            s_axis_meta = exp3_axis_info.get((emotion, 1), {
                "axis_name": f"{emotion}_pc1",
                "high_pole_description": "(not available)",
                "low_pole_description":  "(not available)",
            })

        for pole, sign in [("high", +1.0), ("low", -1.0)]:
            beta_signed = sign * beta_e

            for seed in SEED_TEXTS:
                if emotion not in (seed["seed_emotion"], ):
                    # Only use seeds matching the target emotion
                    if seed["seed_emotion"] != emotion:
                        continue
                source_id = seed["seed_id"]
                base_text  = seed["seed_text"]

                for cond in conditions_to_run:
                    done_set = done_keys[cond]
                    axis_id_cond = f"{emotion}_pc1" if cond == "s" else meta_axis_id
                    cache_key = (emotion, axis_id_cond, pole, source_id)
                    if cache_key in done_set:
                        continue

                    # Build steering vector
                    if cond == "s":
                        axis_vec = u_e1 if "s" in conditions_to_run else None
                        axis_type = "emotion-specific u_{e,1}"
                        axis_name = s_axis_meta.get("axis_name", f"{emotion}_pc1")
                        high_desc = s_axis_meta.get("high_pole_description", "")
                        low_desc  = s_axis_meta.get("low_pole_description",  "")
                    else:
                        axis_vec = m_k
                        axis_type = f"shared meta-axis {meta_axis_id}"
                        axis_name = meta_axis_meta.get("axis_name", meta_axis_id)
                        high_desc = meta_axis_meta.get("high_pole_description", "")
                        low_desc  = meta_axis_meta.get("low_pole_description", "")

                    delta_vec = (
                        alpha_g_e * g.astype(np.float32)
                        + alpha_r_e * r_e
                        + beta_signed * axis_vec
                    )

                    expected_pole_str = (
                        f"HIGH pole: {high_desc}" if pole == "high"
                        else f"LOW pole: {low_desc}"
                    )

                    # --- generate ---
                    if args.dry_run:
                        rewrite_text = f"[dry-run] {base_text[:60]}"
                    else:
                        rewrite_text = generate_with_delta(
                            model, tokenizer, model_device,
                            base_text, delta_vec, layer_idx,
                            args.max_new_tokens, args.do_sample,
                            args.temperature, args.top_p,
                        )

                    logger.info(
                        "  [%s/cond=%s/PC1/%s] %s → %s",
                        emotion, cond, pole,
                        base_text[:40], rewrite_text[:50],
                    )

                    # --- judge ---
                    if args.skip_judge or args.dry_run:
                        judge_scores = {a: 0.0 for a in JUDGE_ATTRS}
                        reasoning = ""
                    else:
                        judge_prompt = JUDGE_USER_TMPL.format(
                            base_text=base_text,
                            rewrite_text=rewrite_text,
                            target_emotion=emotion,
                            axis_name=axis_name,
                            axis_type=axis_type,
                            high_pole_description=high_desc,
                            low_pole_description=low_desc,
                            expected_pole=expected_pole_str,
                        )
                        import hashlib as _hashlib
                        _h = _hashlib.sha1()
                        _h.update(args.judge_model.encode())
                        _h.update(b"|v1|")
                        _h.update(judge_prompt.encode("utf-8"))
                        _cache_p = judge_cache_dir / f"{_h.hexdigest()[:24]}.json"
                        if _cache_p.exists():
                            with open(_cache_p) as _cf:
                                _raw = json.load(_cf)
                        else:
                            from openai import OpenAI
                            _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                            _resp = _client.chat.completions.create(
                                model=args.judge_model,
                                temperature=0.0,
                                messages=[
                                    {"role": "system", "content": JUDGE_SYSTEM},
                                    {"role": "user",   "content": judge_prompt},
                                ],
                                response_format={"type": "json_schema", "json_schema": JUDGE_SCHEMA},
                            )
                            _raw = json.loads(_resp.choices[0].message.content)
                            with open(_cache_p, "w") as _cf:
                                json.dump(_raw, _cf)
                        judge_scores = {k: float(np.clip(_raw.get(k, 0.0), 0.0, 1.0))
                                        for k in JUDGE_ATTRS}
                        reasoning = str(_raw.get("reasoning", ""))

                    row = {
                        "layer": args.layer,
                        "condition": cond,
                        "emotion": emotion,
                        "axis_type": axis_type,
                        "axis_id": axis_id_cond,
                        "pole": pole,
                        "source_id": source_id,
                        "base_text": base_text,
                        "rewrite_text": rewrite_text,
                        "alpha_g": alpha_g_e,
                        "alpha_r": alpha_r_e,
                        "beta": beta_signed,
                        **judge_scores,
                        "judged": not (args.skip_judge or args.dry_run),
                        "reasoning": reasoning,
                    }
                    rj_path, rc_path = results_files[cond]
                    append_jsonl_csv_row(rj_path, rc_path, OUTPUT_FIELDNAMES, row)
                    done_keys[cond].add(cache_key)

    # --- comparison summary ---
    summary_rows = []
    for cond in conditions_to_run:
        rj_path, _ = results_files[cond]
        if not rj_path.exists():
            continue
        df = pd.read_json(rj_path, lines=True)
        if df.empty:
            continue
        judged = df[df["judged"] == True]  # noqa: E712
        if judged.empty:
            continue
        means = judged[JUDGE_ATTRS].mean()
        row = {"condition": cond, **{k: round(float(v), 4) for k, v in means.items()}}
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "comparison_summary.csv", index=False)
        print(f"\n=== Exp 9 — Layer {args.layer}: Condition S vs M comparison ===\n")
        print(summary_df.to_string(index=False))

    logger.info("Done. Results written to %s", out_dir)


if __name__ == "__main__":
    main()
