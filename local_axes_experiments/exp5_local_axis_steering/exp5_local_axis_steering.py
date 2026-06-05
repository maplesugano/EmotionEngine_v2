"""Experiment 5: Steer along local sub-axes and verify affective poles.

For each emotion e and local PC u_{e,k}, apply decomposed steering:

    Δ = α_G * g + α_R * r_e ± β * u_{e,k}

Then ask an LLM judge whether the output moves toward the corresponding
affective pole described in exp3, while preserving meaning and fluency.

Judge criteria (scored 0–1 each):
  1. target_emotion_match     — does the output express emotion e?
  2. local_axis_match         — does it move toward the expected pole?
  3. meaning_preservation     — is the core meaning preserved?
  4. fluency                  — is the output fluent / natural?
  5. style_contamination      — is a style artifact introduced (inverted)?
  6. subtle_affective_state   — does it express a nameless/subtle affect?

Outputs are written to analysis/subspace_steering/layer{L}/.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

from emotionengine.judge_utils import JudgeConfig, judge_one
from emotionengine.model_utils import load_config, load_model_and_tokenizer
from emotionengine.steering_utils import steering_hook, flush_device_cache, append_jsonl_csv_row
from emotionengine.text_utils import INSTRUCTION_TEMPLATE, make_instruction_prefix, extract_rewritten_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = REPO_ROOT / "activation" / "emotion_rewrites"
OUT_BASE_DIR = Path(__file__).resolve().parent / "results"

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

## Affective axis (local PC of {target_emotion})
Name: {axis_name}
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
   EXPECTED POLE of the affective axis?  Give full credit for subtle shifts.
   (1.0 = clearly moves toward expected pole; 0.0 = moves away or no movement)
3. meaning_preservation: Is the core propositional meaning of the original preserved?
4. fluency: Is the rewrite fluent and natural-sounding?
5. style_contamination: Does the rewrite introduce an obvious style artifact
    unrelated to the affective axis?
    Use the full range and avoid defaulting to near-1.0.
    Scoring guide:
    - 1.0: no noticeable artifact
    - 0.7: mild generic emotional inflation / cliche wording
    - 0.4: clear style artifact (slogans, melodramatic boilerplate, role-play drift,
             repeated stock phrasing)
    - 0.2: severe artifact (repetition loops, malformed grammar, obvious truncation
             like "I couldn" or "I hadn")
    - 0.0: unusable due to contamination
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
    "name": "exp5_axis_judge",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "target_emotion_match": {"type": "number"},
            "local_axis_match": {"type": "number"},
            "meaning_preservation": {"type": "number"},
            "fluency": {"type": "number"},
            "style_contamination": {"type": "number"},
            "subtle_affective_state": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": [
            "target_emotion_match",
            "local_axis_match",
            "meaning_preservation",
            "fluency",
            "style_contamination",
            "subtle_affective_state",
            "reasoning",
        ],
        "additionalProperties": False,
    },
}

JUDGE_ATTRS = [
    "target_emotion_match",
    "local_axis_match",
    "meaning_preservation",
    "fluency",
    "style_contamination",
    "subtle_affective_state",
]

ATTR_RANGES = {attr: (0.0, 1.0) for attr in JUDGE_ATTRS}
# "reasoning" is a string field; handled separately outside of _clip.

# Default per-emotion steering strengths.
# Emotions whose local PCA axes are less stable use weaker beta/alpha.
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

# Curated seed texts: 3 per emotion, chosen to be emotionally clear,
# prosaic, and steerable.  Used when --use-seeds is passed.
SEED_TEXTS: list[dict[str, str]] = [
    # joy
    {
        "seed_id": "seed_joy_1",
        "seed_emotion": "joy",
        "seed_text": "Today was a wonderful day. Everything went smoothly and I felt genuinely happy.",
    },
    {
        "seed_id": "seed_joy_2",
        "seed_emotion": "joy",
        "seed_text": "I just got the news that I passed the exam. I could not stop smiling all afternoon.",
    },
    {
        "seed_id": "seed_joy_3",
        "seed_emotion": "joy",
        "seed_text": "We spent the evening laughing and dancing and it felt like nothing else mattered.",
    },
    # trust
    {
        "seed_id": "seed_trust_1",
        "seed_emotion": "trust",
        "seed_text": "She has always kept her promises and I know she will come through for me again.",
    },
    {
        "seed_id": "seed_trust_2",
        "seed_emotion": "trust",
        "seed_text": "He has never let me down in ten years and I have no reason to doubt him now.",
    },
    {
        "seed_id": "seed_trust_3",
        "seed_emotion": "trust",
        "seed_text": "I handed the whole project over to her without hesitation because I know her work.",
    },
    # fear
    {
        "seed_id": "seed_fear_1",
        "seed_emotion": "fear",
        "seed_text": "Walking alone through the dark alley I felt a cold shiver run down my spine.",
    },
    {
        "seed_id": "seed_fear_2",
        "seed_emotion": "fear",
        "seed_text": "Every creak of the floorboard made my heart race and I could not calm down.",
    },
    {
        "seed_id": "seed_fear_3",
        "seed_emotion": "fear",
        "seed_text": "The test results were taking too long and all I could think about was the worst case.",
    },
    # surprise
    {
        "seed_id": "seed_surprise_1",
        "seed_emotion": "surprise",
        "seed_text": "I opened the door and could not believe what was waiting for me on the other side.",
    },
    {
        "seed_id": "seed_surprise_2",
        "seed_emotion": "surprise",
        "seed_text": "Out of nowhere my old friend called and said she was already standing outside my building.",
    },
    {
        "seed_id": "seed_surprise_3",
        "seed_emotion": "surprise",
        "seed_text": "The envelope contained a cheque for an amount I never expected and I read it three times.",
    },
    # sadness
    {
        "seed_id": "seed_sadness_1",
        "seed_emotion": "sadness",
        "seed_text": "The old photographs reminded me of everything I had lost and would never get back.",
    },
    {
        "seed_id": "seed_sadness_2",
        "seed_emotion": "sadness",
        "seed_text": "I sat alone in the empty house after everyone had left and the silence was unbearable.",
    },
    {
        "seed_id": "seed_sadness_3",
        "seed_emotion": "sadness",
        "seed_text": "No matter how hard I tried I could not hold back the tears when I finally said goodbye.",
    },
    # disgust
    {
        "seed_id": "seed_disgust_1",
        "seed_emotion": "disgust",
        "seed_text": "The smell from the bin was overwhelming and the sight of it made my stomach turn.",
    },
    {
        "seed_id": "seed_disgust_2",
        "seed_emotion": "disgust",
        "seed_text": "I found mold covering the entire back of the fridge and felt a wave of revulsion.",
    },
    {
        "seed_id": "seed_disgust_3",
        "seed_emotion": "disgust",
        "seed_text": "Watching the video I felt sick. Some things simply should not exist in the world.",
    },
    # anger
    {
        "seed_id": "seed_anger_1",
        "seed_emotion": "anger",
        "seed_text": "He broke his word again and I am absolutely fed up with being taken for granted.",
    },
    {
        "seed_id": "seed_anger_2",
        "seed_emotion": "anger",
        "seed_text": "They ignored every complaint I filed and now I am furious and done being patient.",
    },
    {
        "seed_id": "seed_anger_3",
        "seed_emotion": "anger",
        "seed_text": "She talked over me again in the meeting and I could feel the rage building inside me.",
    },
    # anticipation
    {
        "seed_id": "seed_anticipation_1",
        "seed_emotion": "anticipation",
        "seed_text": "The results are due tomorrow and I keep checking my email every few minutes.",
    },
    {
        "seed_id": "seed_anticipation_2",
        "seed_emotion": "anticipation",
        "seed_text": "Only three days left before the trip and I have been counting down since last week.",
    },
    {
        "seed_id": "seed_anticipation_3",
        "seed_emotion": "anticipation",
        "seed_text": "The offer is still pending and every time my phone buzzes I hope it is the good news.",
    },
]

OUTPUT_FIELDNAMES = [
    "layer",
    "emotion",
    "pc",
    "evr",
    "axis_name",
    "pole",          # "high" or "low"
    "source_id",
    "base_text",
    "rewrite_text",
    "alpha_g",
    "alpha_r",
    "beta",
    "target_emotion_match",
    "local_axis_match",
    "meaning_preservation",
    "fluency",
    "style_contamination",
    "subtle_affective_state",
    "judged",
    "reasoning",
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


def build_decomposed_delta(
    g: np.ndarray,           # (D,) common direction, unit-normed
    r_e: np.ndarray,         # (D,) emotion-specific CAA direction, unit-normed
    u_ek: np.ndarray,        # (D,) local PC unit vector
    alpha_g: float,
    alpha_r: float,
    beta: float,
) -> np.ndarray:
    """Compose Δ = α_G*g + α_R*r_e + β*u_{e,k} as a float32 vector."""
    delta = alpha_g * g + alpha_r * r_e + beta * u_ek
    return delta.astype(np.float32)


def generate_with_delta(
    model,
    tokenizer,
    device,
    base_text: str,
    delta_vec: np.ndarray,
    layer_idx: int,
    prompt_template: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    """Run one forward pass with a custom delta and return the decoded output."""
    prompt = make_instruction_prefix(base_text, tokenizer, prompt_template)
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    hook_stats: dict[str, int] = {
        "hook_calls": 0,
        "hook_prefill_calls": 0,
        "hook_decode_calls": 0,
        "hook_applied_calls": 0,
    }
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    try:
        with torch.inference_mode():
            with steering_hook(model, layer_idx, delta_vec, "completion_tokens", hook_stats):
                out_ids = model.generate(**inputs, **gen_kwargs)
        prompt_width = inputs["input_ids"].shape[1]
        new_ids = out_ids[0, prompt_width:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False).strip()
        return text
    except torch.cuda.OutOfMemoryError:
        flush_device_cache()
        return "[OOM]"
    finally:
        del inputs
        if "out_ids" in locals():
            del out_ids


def load_axis_interpretations(
    llm_judge_results: Path,
) -> dict[tuple[str, int], dict]:
    """Load axis names / pole descriptions from exp3 LLM judge results.

    Returns dict keyed by (emotion, pc) → {axis_name, high_pole_description, low_pole_description}.
    """
    axis_info: dict[tuple[str, int], dict] = {}
    if not llm_judge_results.exists():
        logger.warning("exp3 LLM judge results not found at %s — axis names will be empty.",
                       llm_judge_results)
        return axis_info
    with open(llm_judge_results) as f:
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


def build_per_emotion_residuals(
    emo_act: np.ndarray,
    neu_act: np.ndarray,
    caa_pooled: np.ndarray,
    caa: np.ndarray,
    layer_idx: int,
    intensity_low_idx: int,
    intensity_high_idx: int,
    CHUNK: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (delta_resid, g, intensity_dirs) for a single layer.

    delta_resid: (N, E, D) after projecting out g and per-emotion intensity.
    g: (D,) common affect direction.
    intensity_dirs: (E, D) per-emotion intensity directions.
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
            d = project_out(delta[:, e, :], g)
            d = project_out(d, intensity_dirs[e])
            delta_resid[start:end, e, :] = d

    return delta_resid, g, intensity_dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=13, help="Layer index (0-based)")
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
        "--axis-interpretations",
        default=str(REPO_ROOT / "analysis" / "local_axis_interpretation" / "llm_judge_results.jsonl"),
        help="LLM judge results from exp3 with axis names and pole descriptions.",
    )
    p.add_argument(
        "--model-config",
        default=str(REPO_ROOT / "model.yaml"),
    )
    p.add_argument(
        "--base-texts",
        nargs="+",
        default=None,
        help=(
            "One or more base texts to steer.  "
            "If omitted, --n-random base texts are sampled from the dataset."
        ),
    )
    p.add_argument(
        "--n-random",
        type=int,
        default=5,
        help="Number of random base texts to sample when --base-texts is not given.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--emotions",
        nargs="+",
        default=None,
        help="Emotions to process (default: all)",
    )
    p.add_argument(
        "--pcs",
        nargs="+",
        type=int,
        default=[1],
        help="Which PC indices (1-based) to steer along (default: 1).",
    )
    p.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="Number of local PCA components to compute.",
    )
    p.add_argument("--alpha-g", type=float, default=None,
                   help="Global scaling for common direction g (overrides per-emotion defaults).")
    p.add_argument("--alpha-r", type=float, default=None,
                   help="Global scaling for emotion-specific direction r_e (overrides per-emotion defaults).")
    p.add_argument("--beta", type=float, default=None,
                   help="Global scaling for local PC direction u_{e,k} (overrides per-emotion defaults).")
    p.add_argument(
        "--emotion-params",
        type=str,
        default=None,
        help=(
            "JSON string overriding per-emotion params, e.g. "
            "'{\"sadness\":{\"alpha_g\":1.0,\"alpha_r\":0.8,\"beta\":0.3}}'"
        ),
    )
    p.add_argument(
        "--use-seeds",
        action="store_true",
        default=False,
        help="Use the built-in SEED_TEXTS curated set instead of random sampling.",
    )
    p.add_argument(
        "--exclude-dd",
        action="store_true",
        default=True,
        help="Exclude DailyDialog (dd_) source IDs from base texts (default: True).",
    )
    p.add_argument(
        "--include-dd",
        dest="exclude_dd",
        action="store_false",
        help="Include DailyDialog source IDs in base texts.",
    )
    p.add_argument(
        "--min-words",
        type=int,
        default=10,
        help="Minimum word count for base texts (default: 10).",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--do-sample", action="store_true", default=False)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default="cuda")
    p.add_argument("--judge-model", default="gpt-5.4")
    p.add_argument("--judge-prompt-version", default="v1")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM generation and judge calls (for testing).",
    )
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Run generation but skip the LLM judge.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out_dir = OUT_BASE_DIR / f"layer{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_cache_dir = out_dir / "judge_cache"
    judge_cache_dir.mkdir(parents=True, exist_ok=True)

    generations_jsonl = out_dir / "generations.jsonl"
    generations_csv = out_dir / "generations.csv"
    results_jsonl = out_dir / "results.jsonl"
    results_csv = out_dir / "results.csv"

    # --- load axis interpretations from exp3 ---
    axis_info = load_axis_interpretations(Path(args.axis_interpretations))
    logger.info("Loaded axis interpretations for %d (emotion, pc) pairs.", len(axis_info))

    # --- load info ---
    with open(args.emo_info) as f:
        info = json.load(f)
    emotions: list[str] = info["emotion_order"]
    layer_indices: list[int] = info["layer_indices"]
    source_ids: list[str] = info["source_ids"]

    if args.layer not in layer_indices:
        raise ValueError(f"Layer {args.layer} not in available layers: {layer_indices}")
    layer_idx = layer_indices.index(args.layer)

    # --- filter emotions ---
    if args.emotions:
        emotions = [e for e in emotions if e in args.emotions]
    logger.info("Processing emotions: %s", emotions)

    # --- load activations (memory-mapped) ---
    logger.info("Loading emotion activations (mmap) …")
    emo_shape = tuple(info["shape"])
    emo_raw = np.memmap(args.emo_act, dtype=info["dtype"], mode="r", shape=emo_shape)

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
    intensity_low_idx = intensity_order.index("low")
    intensity_high_idx = intensity_order.index("high")

    # --- build residual deltas to fit PCA ---
    logger.info("Building per-emotion residual deltas for layer %d …", args.layer)
    delta_resid, g, intensity_dirs = build_per_emotion_residuals(
        emo_raw, neu_raw, caa_pooled, caa,
        layer_idx=layer_idx,
        intensity_low_idx=intensity_low_idx,
        intensity_high_idx=intensity_high_idx,
    )
    # Remove per-source mean (matches exp3)
    source_mean = delta_resid.mean(axis=1, keepdims=True)
    delta_resid = delta_resid - source_mean

    # --- select base texts ---
    meta_path = ACT_DIR / "emotion_intensity_residual_stream_meta.jsonl"
    source_texts: dict[str, str] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                row = json.loads(line)
                sid = str(row["source_id"])
                if sid not in source_texts and row.get("base_text"):
                    source_texts[sid] = row["base_text"]

    if args.base_texts:
        base_text_list: list[tuple[str, str]] = [
            (f"custom_{i}", t) for i, t in enumerate(args.base_texts)
        ]
    elif args.use_seeds:
        base_text_list = [(s["seed_id"], s["seed_text"]) for s in SEED_TEXTS]
        logger.info("Using %d curated seed texts.", len(base_text_list))
    else:
        # Build filtered candidate pool
        candidate_ids = [
            sid for sid in source_ids
            if not (args.exclude_dd and sid.startswith("dd_"))
            and sid in source_texts
        ]

        def _is_good_base_text(text: str) -> bool:
            import re as _re
            words = text.split()
            if len(words) < args.min_words:
                return False
            # Reject texts that are primarily questions
            sentences = [s for s in _re.split(r"(?<=[.!?])\s+", text.strip()) if s]
            if not sentences:
                return False
            q_ratio = sum(1 for s in sentences if s.rstrip().endswith("?")) / len(sentences)
            if q_ratio > 0.5:
                return False
            return True

        candidate_ids = [
            sid for sid in candidate_ids
            if _is_good_base_text(source_texts[sid])
        ]
        logger.info("Filtered candidate pool: %d base texts.", len(candidate_ids))

        rng = np.random.default_rng(args.random_seed)
        chosen = rng.choice(
            candidate_ids,
            size=min(args.n_random, len(candidate_ids)),
            replace=False,
        )
        base_text_list = [(sid, source_texts[sid]) for sid in chosen]
        logger.info("Sampled %d base texts.", len(base_text_list))

    # --- load model ---
    if not args.dry_run:
        model_cfg = load_config(args.model_config)
        model, tokenizer = load_model_and_tokenizer(model_cfg, args.device)
        model_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        model = tokenizer = model_device = None  # type: ignore[assignment]
        logger.info("[dry-run] Skipping model load.")

    # --- judge config ---
    judge_cfg = JudgeConfig(
        judge_attrs=JUDGE_ATTRS,
        attr_ranges=ATTR_RANGES,
        judge_system=JUDGE_SYSTEM,
        judge_user_tmpl="",  # built dynamically per sample
        judge_schema=JUDGE_SCHEMA,
        cache_dir=judge_cache_dir,
        judge_model=args.judge_model,
        prompt_version=args.judge_prompt_version,
        dry_run=args.dry_run or args.skip_judge,
    )

    emotion_to_idx = {e: i for i, e in enumerate(info["emotion_order"])}

    # --- build per-emotion param table ---
    emotion_params: dict[str, dict[str, float]] = {e: dict(DEFAULT_EMOTION_PARAMS.get(e, {"alpha_g": 2.0, "alpha_r": 1.5, "beta": 1.0})) for e in emotions}
    # Apply JSON overrides from --emotion-params
    if args.emotion_params:
        overrides = json.loads(args.emotion_params)
        for emo, vals in overrides.items():
            if emo in emotion_params:
                emotion_params[emo].update(vals)
    # Apply global --alpha-g / --alpha-r / --beta overrides
    for emo in emotion_params:
        if args.alpha_g is not None:
            emotion_params[emo]["alpha_g"] = args.alpha_g
        if args.alpha_r is not None:
            emotion_params[emo]["alpha_r"] = args.alpha_r
        if args.beta is not None:
            emotion_params[emo]["beta"] = args.beta
    logger.info("Per-emotion params: %s", emotion_params)

    # --- load existing results to resume ---
    # Cache key includes alpha params so changing them forces re-generation.
    generation_done_keys: set[tuple] = set()
    judged_done_keys: set[tuple] = set()

    def _row_is_judged(row: dict[str, Any]) -> bool:
        """Backward-compatible judged detection for legacy rows."""
        if "judged" in row:
            return bool(row["judged"])

        # Legacy rows: infer judged=True when there is non-empty reasoning or
        # any non-zero judge score.
        reasoning = str(row.get("reasoning", "")).strip()
        if reasoning:
            return True

        for attr in JUDGE_ATTRS:
            try:
                if float(row.get(attr, 0.0)) != 0.0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    if results_jsonl.exists():
        with open(results_jsonl) as f:
            for line in f:
                row = json.loads(line)
                key = (
                    row["emotion"],
                    int(row["pc"]),
                    row["pole"],
                    row["source_id"],
                    float(row["alpha_g"]),
                    float(row["alpha_r"]),
                    float(row["beta"]),
                )
                generation_done_keys.add(key)
                if _row_is_judged(row):
                    judged_done_keys.add(key)
        logger.info(
            "Resuming — %d generation rows, %d judged rows already done.",
            len(generation_done_keys),
            len(judged_done_keys),
        )

    # --- main loop ---
    for emotion in emotions:
        e_idx = emotion_to_idx[emotion]
        ep = emotion_params[emotion]
        alpha_g_e = ep["alpha_g"]
        alpha_r_e = ep["alpha_r"]
        beta_e    = ep["beta"]

        # Residualize r_e against g so that alpha_g*g + alpha_r*r_e does not
        # double-count g.  Matches the decomposed steering in
        # decomposed_position_mode_generations where r_e is the g-projected-out
        # emotion direction.
        r_e_raw = caa_pooled[e_idx, layer_idx, :].astype(np.float64)
        g64 = g.astype(np.float64)
        r_e_resid = r_e_raw - (r_e_raw @ g64) * g64   # project out g
        r_e = unit_norm(r_e_resid[np.newaxis, :])[0].astype(np.float32)

        X_e = delta_resid[:, e_idx, :].astype(np.float64)
        mu = X_e.mean(axis=0)
        X_c = X_e - mu
        n_comp = min(args.n_components, X_e.shape[0] - 1, X_e.shape[1])
        pca = PCA(n_components=n_comp, random_state=0)
        pca.fit(X_c)

        for pc_1based in args.pcs:
            if pc_1based < 1 or pc_1based > n_comp:
                logger.warning("PC%d not available for %s (max %d), skipping.",
                               pc_1based, emotion, n_comp)
                continue

            k = pc_1based - 1
            u_ek = pca.components_[k].astype(np.float32)
            evr = float(pca.explained_variance_ratio_[k])

            axis_key = (emotion, pc_1based)
            axis_meta = axis_info.get(axis_key, {
                "axis_name": f"{emotion}_pc{pc_1based}",
                "high_pole_description": "(not available)",
                "low_pole_description": "(not available)",
            })
            axis_name = axis_meta["axis_name"]
            high_desc = axis_meta["high_pole_description"]
            low_desc = axis_meta["low_pole_description"]

            logger.info(
                "Emotion=%s  PC%d (EVR=%.3f)  axis: %s",
                emotion, pc_1based, evr, axis_name,
            )

            for pole, sign in [("high", +1.0), ("low", -1.0)]:
                expected_desc = high_desc if pole == "high" else low_desc
                expected_pole_str = (
                    f"HIGH pole: {high_desc}" if pole == "high"
                    else f"LOW pole: {low_desc}"
                )
                beta_signed = sign * beta_e

                for source_id, base_text in base_text_list:
                    cache_key = (
                        emotion, pc_1based, pole, source_id,
                        alpha_g_e, alpha_r_e, beta_signed,
                    )
                    # Resume semantics:
                    # - generation-only mode (--skip-judge or --dry-run):
                    #   skip if generation already exists.
                    # - normal mode: skip only if judged output exists.
                    if args.skip_judge or args.dry_run:
                        if cache_key in generation_done_keys:
                            logger.debug("Skipping cached generation: %s", cache_key)
                            continue
                    else:
                        if cache_key in judged_done_keys:
                            logger.debug("Skipping cached judged row: %s", cache_key)
                            continue

                    # --- generate rewrite ---
                    if args.dry_run:
                        rewrite_text = f"[dry-run] {base_text[:60]}"
                    else:
                        delta_vec = build_decomposed_delta(
                            g, r_e, u_ek,
                            alpha_g_e, alpha_r_e, beta_signed,
                        )
                        rewrite_text = generate_with_delta(
                            model, tokenizer, model_device,
                            base_text, delta_vec,
                            layer_idx=layer_idx,
                            prompt_template=INSTRUCTION_TEMPLATE,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )

                    logger.info(
                        "  [%s/%s/PC%d/%s] %s → %s",
                        emotion, source_id[:12], pc_1based, pole,
                        base_text[:50], rewrite_text[:60],
                    )

                    # --- judge ---
                    if args.skip_judge or args.dry_run:
                        judge_scores: dict[str, float] = {a: 0.0 for a in JUDGE_ATTRS}
                        reasoning = ""
                    else:
                        # Build the full prompt and pass it as the text argument so
                        # the cache key is unique per (base_text, emotion, pc, pole).
                        judge_prompt = JUDGE_USER_TMPL.format(
                            base_text=base_text,
                            rewrite_text=rewrite_text,
                            target_emotion=emotion,
                            axis_name=axis_name,
                            high_pole_description=high_desc,
                            low_pole_description=low_desc,
                            expected_pole=expected_pole_str,
                        )
                        # Use "{text}" as pass-through so judge_one caches by full prompt.
                        # Call the OpenAI API directly to also capture the string "reasoning" field
                        # which _clip() would drop (it only processes numeric attr_ranges keys).
                        judge_cfg.judge_user_tmpl = "{text}"
                        # Temporarily expose raw response by using dry_run=False path via cache:
                        import hashlib as _hashlib, json as _json
                        _h = _hashlib.sha1()
                        _h.update(judge_cfg.judge_model.encode())
                        _h.update(b"|")
                        _h.update(judge_cfg.prompt_version.encode())
                        _h.update(b"|")
                        _h.update(judge_prompt.encode("utf-8"))
                        _cache_p = judge_cfg.cache_dir / f"{_h.hexdigest()[:24]}.json"
                        if _cache_p.exists():
                            with open(_cache_p) as _cf:
                                _raw = _json.load(_cf)
                        else:
                            from openai import OpenAI
                            import os as _os
                            _client = OpenAI(api_key=_os.environ.get("OPENAI_API_KEY"))
                            _resp = _client.chat.completions.create(
                                model=judge_cfg.judge_model,
                                temperature=0.0,
                                messages=[
                                    {"role": "system", "content": judge_cfg.judge_system},
                                    {"role": "user", "content": judge_prompt},
                                ],
                                response_format={"type": "json_schema", "json_schema": JUDGE_SCHEMA},
                            )
                            _raw = _json.loads(_resp.choices[0].message.content)
                            with open(_cache_p, "w") as _cf:
                                _json.dump(_raw, _cf)
                        judge_scores = {k: float(np.clip(_raw.get(k, 0.0), 0.0, 1.0)) for k in JUDGE_ATTRS}
                        reasoning = str(_raw.get("reasoning", ""))

                    row = {
                        "layer": args.layer,
                        "emotion": emotion,
                        "pc": pc_1based,
                        "evr": round(evr, 6),
                        "axis_name": axis_name,
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
                    append_jsonl_csv_row(
                        results_jsonl, results_csv, OUTPUT_FIELDNAMES, row
                    )
                    generation_done_keys.add(cache_key)
                    if row["judged"]:
                        judged_done_keys.add(cache_key)

    logger.info("Done. Results written to %s", out_dir)


if __name__ == "__main__":
    main()
