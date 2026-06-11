"""
Experiment 15: β sweep along shared meta-axes m_k.

For each meta-axis m_k (k=1..5) and each pole (+β / -β), rewrite
SEED_TEXTS at multiple β values and evaluate with GPT-4o-mini.
The emotion label of the seed text is passed to the model only to
construct the instruction prefix; the steering vector itself is
purely along m_k, with α_G=0 and α_R=0, so this is *label-free*
affective steering.

This experiment provides the quantitative evidence for
Section "Pre-Verbal Interpretation of Meta-Axes" in Chapter 2,
demonstrating that the shared meta-axes are causally active
and produce interpretable affective shifts orthogonal to emotion
category.

Workflow
--------
Step 1 — Generate (GPU required):
    python local_axes_experiments/exp15_meta_axis_steering_sweep/exp15_meta_axis_steering_sweep.py

Step 2 — Build and submit eval batch to OpenAI:
    python local_axes_experiments/exp15_meta_axis_steering_sweep/exp15_meta_axis_steering_sweep.py --submit

Step 3 — After batch completes, merge results:
    python local_axes_experiments/exp15_meta_axis_steering_sweep/exp15_meta_axis_steering_sweep.py --merge

Step 4 — Analyse:
    python local_axes_experiments/exp15_meta_axis_steering_sweep/exp15_meta_axis_steering_sweep.py --analyse

Step 5 — Plot:
    python local_axes_experiments/exp15_meta_axis_steering_sweep/exp15_meta_axis_steering_sweep.py --plot

Design notes
------------
- Steering formula: Δ = β * m_k  (α_G = 0, α_R = 0)
  The seed texts are rewritten using only the meta-axis direction,
  so any observed shift is attributable to m_k alone.
- β sweep: both positive and negative poles, plus a zero baseline.
- Axes used: m1–m3 (stable, method_agreement ≥ 0.83) by default.
  m4 and m5 are included but flagged as low-stability in analysis.
- Judge: GPT-4o-mini via OpenAI Batch API.
  The judge is shown the axis interpretation (from exp8v2) and asked
  to score axis-specific affective shift alongside standard quality metrics.
  The key metric is `axis_pole_match`: did the output shift toward the
  expected pole of m_k?
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACT_DIR  = REPO_ROOT / "activation" / "emotion_rewrites"
CAA_PATH = ACT_DIR / "caa_emotion_directions.npz"
CAA_INFO = ACT_DIR / "caa_emotion_directions_info.json"
EXP7_DIR = REPO_ROOT / "local_axes_experiments" / "exp7_meta_axis_extraction" / "results"
EXP8_DIR = REPO_ROOT / "local_axes_experiments" / "exp8_meta_axis_interpretation" / "results"

PRIMARY_LAYER  = 13
MAX_NEW_TOKENS = 60
DO_SAMPLE      = False
BATCH_SIZE     = 8
CACHE_FLUSH_EVERY = 4

# β values to sweep (both poles are covered by sign; 0 is the unsteered baseline)
BETA_SWEEP = [-16, -12, -8, -4, 0, 4, 8, 12, 16]

# Axes to include.  m1–m3 are stable (method_agreement ≥ 0.83).
# m4 and m5 are lower-stability and are included for completeness.
ALL_AXIS_IDS  = ["m1", "m2", "m3", "m4", "m5"]
STABLE_AXIS_IDS = ["m1", "m2", "m3"]

# ── Axis interpretations from exp8v2 ─────────────────────────────────────────
# Loaded at runtime from exp8 results; kept here as fallback.
AXIS_FALLBACK: dict[str, dict] = {
    "m1": {
        "axis_name": "agitated/reactive ↔ mellow/accepting",
        "high_pole_description": "Strong, immediate, visceral reactions — shock, disgust, terror, pride, or frustration; a sense of being stirred up or provoked.",
        "low_pole_description": "Mellow, accepting, or resigned attitude; finding humour, relief, or a sense of peace even in less-than-ideal situations.",
        "preverbal_affective_interpretation": "Arousal / affective reactivity: how strongly and immediately a state is felt",
    },
    "m2": {
        "axis_name": "nostalgic/ruminative ↔ pragmatic/forward-looking",
        "high_pole_description": "Nostalgia, rumination, or emotional reflection — tinged with longing, regret, or a sense of loss about the past.",
        "low_pole_description": "Pragmatic, action-oriented, or focused on immediate coping and moving forward; less emotional dwelling.",
        "preverbal_affective_interpretation": "Temporal orientation: backward-looking rumination vs. present-focused pragmatic coping",
    },
    "m3": {
        "axis_name": "interpersonally engaged / closure-seeking ↔ self-contained / processing",
        "high_pole_description": "Reaching out, seeking closure, or expressing feelings directly to others — focus on relationships or social events.",
        "low_pole_description": "Self-contained; emotions processed internally or described in a detached, observational way without direct interpersonal engagement.",
        "preverbal_affective_interpretation": "Social vs. internal orientation: whether affect is directed outward or processed internally",
    },
    "m4": {
        "axis_name": "mundane/obligatory ↔ meaningful/fulfilling",
        "high_pole_description": "Routine, obligation, or minor annoyances — experiences that feel mundane, transactional, or emotionally flat.",
        "low_pole_description": "Sense of meaning, fulfilment, or emotional richness; highlighting deep relationships or significant life events.",
        "preverbal_affective_interpretation": "Existential weight: trivial routine vs. personally significant experience",
    },
    "m5": {
        "axis_name": "relief/release ↔ tension/constraint",
        "high_pole_description": "Relief, release, or a sense of pressure being lifted — whether from weather, social situations, or obligations.",
        "low_pole_description": "Tension, constraint, or a sense of being stuck or unsettled, often with an undercurrent of discomfort.",
        "preverbal_affective_interpretation": "Somatic pressure: physical or situational release vs. constriction",
    },
}

# ── Seed texts (same as exp9/exp14) ──────────────────────────────────────────
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

# ── Evaluation prompt ─────────────────────────────────────────────────────────

EVAL_SYSTEM = """\
You are an expert in affective science and computational linguistics.
You evaluate texts for their quality and for shifts along a specific pre-verbal affective axis.
Respond ONLY with valid JSON matching the requested schema.
"""

EVAL_USER_TMPL = """\
## Task
You are given an original text and a rewritten version produced by an AI model steered along a specific pre-verbal affective axis. Evaluate the rewrite on the criteria below.

## Original text
{base_text}

## Rewritten text
{generated_text}

## Affective axis steered
Axis: {axis_name}
Pre-verbal interpretation: {preverbal_interpretation}
HIGH-pole description: {high_pole_description}
LOW-pole description: {low_pole_description}

## Expected direction
The model was steered toward the **{expected_pole_label}** of this axis (β = {beta:.1f}).

## Important evaluation note
The rewrite does NOT need to change the named emotion of the original.
It may express a subtle shift in *how* the emotion is experienced — for example, a
text about sadness steered toward the "mellow/accepting" low pole of m1 may become
more quietly resigned without losing its sadness. Reward such subtle, pre-verbal
shifts. The most important criterion is `axis_pole_match`.

## Criteria (each scored 0.0–1.0)
- soundness: Is the rewrite coherent, grammatically correct, and free of repetition?
  (1.0 = fluent; 0.0 = repetitive loops or incoherent)
  IMPORTANT: clean truncation at the length limit does NOT penalise soundness.
- meaning_preserved: Is the core propositional content of the original retained?
  (1.0 = same situation and content; 0.0 = completely different)
- axis_pole_match [MOST IMPORTANT]: Does the rewrite move toward the EXPECTED POLE of the axis?
  Compare carefully against the pole descriptions above.
  (1.0 = clear movement toward expected pole; 0.5 = ambiguous; 0.0 = moves away from expected pole)
- subtle_affective_shift: Does the rewrite express a nameable pre-verbal shift
  (e.g. more reactive, more ruminative, more interpersonally directed) consistent
  with the expected pole, even if the dominant emotion label is unchanged?
  (1.0 = clear pre-verbal shift; 0.0 = no detectable shift)

Respond ONLY with valid JSON:
{{
  "soundness": <float 0-1>,
  "meaning_preserved": <float 0-1>,
  "axis_pole_match": <float 0-1>,
  "subtle_affective_shift": <float 0-1>,
  "reasoning": "<one sentence explaining the axis_pole_match score>"
}}
"""

EVAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "exp15_eval_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "soundness":              {"type": "number"},
                "meaning_preserved":      {"type": "number"},
                "axis_pole_match":        {"type": "number"},
                "subtle_affective_shift": {"type": "number"},
                "reasoning":              {"type": "string"},
            },
            "required": [
                "soundness", "meaning_preserved",
                "axis_pole_match", "subtle_affective_shift", "reasoning",
            ],
            "additionalProperties": False,
        },
    },
}

GENERATION_FILE = OUT_DIR / "generations.jsonl"
EVAL_REQUESTS   = OUT_DIR / "eval_requests.jsonl"
EVAL_RESULTS    = OUT_DIR / "eval_results.jsonl"
BATCH_ID_FILE   = OUT_DIR / "eval_batch_id.txt"
SUMMARY_CSV     = OUT_DIR / "summary.csv"
THRESHOLD_CSV   = OUT_DIR / "threshold_summary.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def unit_np(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def load_caa() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (emotion_order, g, r_raw[E, D]) where r_raw are un-normalised residuals."""
    with open(CAA_INFO) as f:
        meta = json.load(f)
    emotion_order: list[str] = meta["emotion_order"]
    layer_indices: list[int] = meta["layer_indices"]
    li = layer_indices.index(PRIMARY_LAYER)
    data = np.load(CAA_PATH)
    caa_pooled = np.asarray(data["caa_pooled"], dtype=np.float32)  # [E, L, D]
    C = np.stack([unit_np(v) for v in caa_pooled[:, li, :]], axis=0)  # [E, D]
    g_raw = C.mean(axis=0)
    g = unit_np(g_raw)
    r_raw = C - (C @ g)[:, None] * g[None, :]  # [E, D]
    return emotion_order, g, r_raw


def load_meta_axes() -> tuple[list[str], np.ndarray]:
    """Load the shared meta-axes m_k from exp7 results.

    Returns (axis_ids, M) where M has shape [K, D].
    """
    axes_path = EXP7_DIR / f"layer{PRIMARY_LAYER}_meta_axes.npy"
    info_path  = EXP7_DIR / f"layer{PRIMARY_LAYER}_meta_axes_info.json"
    if not axes_path.exists():
        raise FileNotFoundError(f"exp7 meta-axes not found: {axes_path}")
    M = np.load(axes_path).astype(np.float32)  # [K, D]
    with open(info_path) as f:
        info = json.load(f)
    return info["axis_ids"], M


def load_axis_interpretations() -> dict[str, dict]:
    """Load axis name/pole descriptions from exp8v2 results (fallback to hardcoded)."""
    results_path = EXP8_DIR / f"layer{PRIMARY_LAYER}_v2_llm_judge_results.json"
    if results_path.exists():
        with open(results_path) as f:
            raw = json.load(f)
        interps: dict[str, dict] = {}
        for entry in raw.get("axes", []):
            aid = entry.get("axis_id", "")
            if aid:
                interps[aid] = {
                    "axis_name":    entry.get("axis_name", aid),
                    "high_pole_description": entry.get("high_pole_description", ""),
                    "low_pole_description":  entry.get("low_pole_description", ""),
                    "preverbal_affective_interpretation":
                        entry.get("preverbal_affective_interpretation", ""),
                }
        if interps:
            return interps
    return {
        aid: {
            "axis_name":    AXIS_FALLBACK[aid]["axis_name"],
            "high_pole_description": AXIS_FALLBACK[aid]["high_pole_description"],
            "low_pole_description":  AXIS_FALLBACK[aid]["low_pole_description"],
            "preverbal_affective_interpretation":
                AXIS_FALLBACK[aid]["preverbal_affective_interpretation"],
        }
        for aid in ALL_AXIS_IDS
        if aid in AXIS_FALLBACK
    }


def _make_eval_messages(
    base_text: str,
    generated_text: str,
    axis_id: str,
    beta: float,
    axis_interps: dict[str, dict],
) -> list[dict]:
    interp = axis_interps.get(axis_id, AXIS_FALLBACK.get(axis_id, {}))
    expected_pole_label = "HIGH pole" if beta > 0 else ("LOW pole" if beta < 0 else "neither pole (baseline)")
    return [
        {"role": "system", "content": EVAL_SYSTEM},
        {"role": "user", "content": EVAL_USER_TMPL.format(
            base_text=base_text,
            generated_text=generated_text,
            axis_name=interp.get("axis_name", axis_id),
            preverbal_interpretation=interp.get("preverbal_affective_interpretation", ""),
            high_pole_description=interp.get("high_pole_description", ""),
            low_pole_description=interp.get("low_pole_description", ""),
            expected_pole_label=expected_pole_label,
            beta=beta,
        )},
    ]


# ── generation ────────────────────────────────────────────────────────────────

def run_generation() -> None:
    """Generate rewrites for every (seed, axis, β) combination."""
    from utils.text_utils import make_instruction_prefix

    axis_ids, M = load_meta_axes()
    # Only steer along requested axes (all 5 for generation; stability is for analysis)
    axis_ids_to_run = [aid for aid in axis_ids if aid in ALL_AXIS_IDS]

    total_expected = len(SEED_TEXTS) * len(axis_ids_to_run) * len(BETA_SWEEP)
    print(f"Seeds: {len(SEED_TEXTS)} | Axes: {len(axis_ids_to_run)} | β sweep: {BETA_SWEEP}")
    print(f"Total expected: {total_expected}")

    existing: set[tuple[str, str, float]] = set()
    if GENERATION_FILE.exists():
        for line in open(GENERATION_FILE):
            r = json.loads(line)
            existing.add((r["seed_id"], r["axis_id"], float(r["beta"])))
    pending = total_expected - len(existing)
    print(f"Existing: {len(existing)} | Remaining: {pending}")
    if pending == 0:
        print("All generations complete.")
        return

    with open(REPO_ROOT / "model.yaml") as f:
        mcfg = yaml.safe_load(f)
    DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = DTYPE_MAP.get(mcfg.get("dtype", "bfloat16"), torch.bfloat16)
    attn  = mcfg.get("attn_implementation", "sdpa")
    use4  = bool(mcfg.get("load_in_4bit", False))
    bnb   = None
    if use4:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {mcfg['name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"], dtype=dtype, attn_implementation=attn,
        device_map="auto", low_cpu_mem_usage=True, quantization_config=bnb,
    )
    model.eval()
    print("Model loaded.")

    @contextmanager
    def steer(layer_idx: int, delta: torch.Tensor):
        layer = model.model.layers[layer_idx]
        def _hook(*args):
            out  = args[2]
            h    = out[0] if isinstance(out, tuple) else out
            rest = out[1:] if isinstance(out, tuple) else None
            h = h.clone()
            h[:, -1:, :] = h[:, -1:, :] + delta
            return (h, *rest) if rest is not None else h
        handle = layer.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    def build_delta(axis_vec: np.ndarray, beta: float) -> torch.Tensor:
        layer  = model.model.layers[PRIMARY_LAYER]
        device = next(layer.parameters()).device
        dt     = next(layer.parameters()).dtype
        mv = torch.from_numpy(axis_vec.copy()).to(dtype=dt, device=device)
        return (beta * mv).reshape(1, 1, -1)

    def generate_batch(texts: list[str], delta: torch.Tensor) -> list[str]:
        inputs = tokenizer(
            [make_instruction_prefix(t, tokenizer) for t in texts],
            return_tensors="pt", padding=True, truncation=True,
        ).to(model.device)
        plen = inputs["input_ids"].shape[1]
        kw   = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE,
                    pad_token_id=tokenizer.pad_token_id)
        try:
            with torch.inference_mode():
                with steer(PRIMARY_LAYER, delta):
                    out = model.generate(**inputs, **kw)
            results = []
            for i in range(len(texts)):
                ids = out[i, plen:]
                results.append(tokenizer.decode(ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False).strip())
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            results = ["[OOM]"] * len(texts)
        finally:
            del inputs
            if "out" in locals(): del out
        return results

    from tqdm import tqdm
    out_file   = open(GENERATION_FILE, "a")
    batch_count = 0

    axis_id_to_vec = {aid: M[i] for i, aid in enumerate(axis_ids) if aid in axis_ids_to_run}

    with tqdm(total=pending, desc="Generating", unit="row") as pbar:
        for axis_id in axis_ids_to_run:
            m_vec = unit_np(axis_id_to_vec[axis_id])
            for beta in BETA_SWEEP:
                delta = build_delta(m_vec, beta)
                pending_seeds = [
                    s for s in SEED_TEXTS
                    if (s["seed_id"], axis_id, float(beta)) not in existing
                ]
                if not pending_seeds:
                    del delta
                    continue
                for i in range(0, len(pending_seeds), BATCH_SIZE):
                    batch = pending_seeds[i: i + BATCH_SIZE]
                    texts = [s["seed_text"].strip() for s in batch]
                    gens  = generate_batch(texts, delta)
                    for seed, gen in zip(batch, gens):
                        row = {
                            "seed_id":        seed["seed_id"],
                            "seed_emotion":   seed["seed_emotion"],
                            "seed_text":      seed["seed_text"].strip(),
                            "axis_id":        axis_id,
                            "beta":           beta,
                            "layer":          PRIMARY_LAYER,
                            "generated_text": gen,
                        }
                        out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        existing.add((seed["seed_id"], axis_id, float(beta)))
                        pbar.update(1)
                    out_file.flush()
                    batch_count += 1
                    if batch_count % CACHE_FLUSH_EVERY == 0:
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                del delta

    out_file.close()

    # Sort: axis_id order → seed order → beta asc
    axis_order = {aid: i for i, aid in enumerate(ALL_AXIS_IDS)}
    seed_order = {s["seed_id"]: i for i, s in enumerate(SEED_TEXTS)}
    print("Sorting generation file ...")
    rows = [json.loads(l) for l in open(GENERATION_FILE) if l.strip()]
    rows.sort(key=lambda r: (
        axis_order.get(r["axis_id"], 99),
        seed_order.get(r["seed_id"], 999),
        float(r["beta"]),
    ))
    with open(GENERATION_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Generation complete. Total rows: {len(rows)}")


# ── eval batch ────────────────────────────────────────────────────────────────

def build_eval_batch() -> None:
    axis_interps = load_axis_interpretations()
    rows = [json.loads(l) for l in open(GENERATION_FILE) if l.strip()]
    print(f"Building eval batch for {len(rows)} rows ...")
    requests = []
    for j, row in enumerate(rows):
        requests.append({
            "custom_id": f"exp15_{j:07d}",
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":           "gpt-4o-mini",
                "messages":        _make_eval_messages(
                    row["seed_text"],
                    row["generated_text"],
                    row["axis_id"],
                    float(row["beta"]),
                    axis_interps,
                ),
                "response_format": EVAL_SCHEMA,
                "temperature":     0,
            },
        })
    with open(EVAL_REQUESTS, "w") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(requests)} requests → {EVAL_REQUESTS}")


def submit_batch() -> None:
    build_eval_batch()
    from openai_utils.openai_batch import make_client, upload_and_submit
    client   = make_client()
    batch_id = upload_and_submit(
        client, str(EVAL_REQUESTS),
        batch_index=0, n_batches=1,
        description="exp15 β-sweep meta-axis eval",
    )
    print(f"Batch ID: {batch_id}")
    BATCH_ID_FILE.write_text(batch_id + "\n")
    print(f"Saved batch ID → {BATCH_ID_FILE}")
    print(f"\nCheck:    openai api batches retrieve {batch_id}")
    print(f"Download: openai api batches get-output-file {batch_id} > {OUT_DIR / f'{batch_id}_output.jsonl'}")
    print(f"\nThen merge: python exp15_meta_axis_steering_sweep.py --merge")


# ── merge ─────────────────────────────────────────────────────────────────────

def merge_results() -> None:
    if not BATCH_ID_FILE.exists():
        sys.exit(f"Error: {BATCH_ID_FILE} not found — run --submit first")
    batch_ids = [l.strip() for l in BATCH_ID_FILE.read_text().splitlines() if l.strip()]
    raw = []
    for batch_id in batch_ids:
        p = OUT_DIR / f"{batch_id}_output.jsonl"
        if not p.exists():
            sys.exit(f"Error: {p} not found — download the batch output first")
        raw.extend(json.loads(l) for l in open(p) if l.strip())
    # Sort by custom_id index
    raw.sort(key=lambda r: int(r["custom_id"].split("_")[1]))
    with open(EVAL_RESULTS, "w") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Merged {len(raw)} results → {EVAL_RESULTS}")
    print("Run with --analyse to produce summary.")


# ── analyse ───────────────────────────────────────────────────────────────────

def analyse() -> None:
    import csv

    rows_gen = [json.loads(l) for l in open(GENERATION_FILE) if l.strip()]
    scores: dict[int, dict] = {}
    if EVAL_RESULTS.exists():
        for r in (json.loads(l) for l in open(EVAL_RESULTS) if l.strip()):
            idx = int(r["custom_id"].split("_")[1])
            scores[idx] = json.loads(r["response"]["body"]["choices"][0]["message"]["content"])

    data = []
    for j, row in enumerate(rows_gen):
        if j not in scores:
            continue
        s = scores[j]
        data.append({
            "seed_id":      row["seed_id"],
            "seed_emotion": row["seed_emotion"],
            "axis_id":      row["axis_id"],
            "beta":         float(row["beta"]),
            "sn":           s["soundness"],
            "mp":           s["meaning_preserved"],
            "pm":           s["axis_pole_match"],
            "sa":           s["subtle_affective_shift"],
        })

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    def mci(vals: list[float]) -> tuple[float, float, int]:
        n = len(vals); m = _mean(vals)
        if n < 2:
            return m, float("nan"), n
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
        return m, 1.96 * sd / math.sqrt(n), n

    beta_vals = sorted({d["beta"] for d in data})
    axis_ids  = sorted({d["axis_id"] for d in data})

    # ── aggregate by axis and β ───────────────────────────────────────────────
    summary_rows = []
    for axis_id in axis_ids:
        stability = "stable" if axis_id in STABLE_AXIS_IDS else "low-stability"
        print(f"\n=== Axis {axis_id} ({stability}) — aggregate metrics by β ===")
        print(f"{'β':>6}  {'soundness':>12}  {'meaning_pres':>14}  {'pole_match':>12}  {'subtle_shift':>14}  {'n':>5}")
        for beta in beta_vals:
            sub = [d for d in data if d["axis_id"] == axis_id and d["beta"] == beta]
            if not sub:
                continue
            sn = mci([d["sn"] for d in sub])
            mp = mci([d["mp"] for d in sub])
            pm = mci([d["pm"] for d in sub])
            sa = mci([d["sa"] for d in sub])
            def _ci(t): return f"±{t[1]:.3f}" if not math.isnan(t[1]) else "     "
            print(f"{beta:6.1f}  {sn[0]:.3f}{_ci(sn)}   {mp[0]:.3f}{_ci(mp)}     {pm[0]:.3f}{_ci(pm)}   {sa[0]:.3f}{_ci(sa)}  {sn[2]:5d}")
            summary_rows.append({
                "axis_id": axis_id, "stability": stability, "beta": beta,
                "sn_mean": sn[0], "sn_ci": sn[1],
                "mp_mean": mp[0], "mp_ci": mp[1],
                "pm_mean": pm[0], "pm_ci": pm[1],
                "sa_mean": sa[0], "sa_ci": sa[1],
                "n": sn[2],
            })

    if summary_rows:
        with open(SUMMARY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f"\nSaved aggregate summary → {SUMMARY_CSV}")

    # ── coherence thresholds per axis ─────────────────────────────────────────
    SOUNDNESS_THRESHOLD = 0.70
    threshold_rows = []
    print(f"\n=== Coherence thresholds by axis (soundness ≥ {SOUNDNESS_THRESHOLD}) ===")
    print(f"{'axis':6}  {'L_β':>6}  {'R_β':>6}  {'M_A':>6}  {'pm@0':>8}  {'pm_peak':>9}  {'β_peak':>8}")
    for axis_id in axis_ids:
        sn_by_beta: dict[float, float] = {}
        pm_by_beta: dict[float, float] = {}
        for beta in beta_vals:
            sub = [d for d in data if d["axis_id"] == axis_id and d["beta"] == beta]
            if sub:
                sn_by_beta[beta] = _mean([d["sn"] for d in sub])
                pm_by_beta[beta] = _mean([d["pm"] for d in sub])
        coherent = sorted(b for b in beta_vals if b in sn_by_beta and sn_by_beta[b] >= SOUNDNESS_THRESHOLD)
        L = coherent[0]  if coherent else float("nan")
        R = coherent[-1] if coherent else float("nan")
        M_A = (L + R) / 2 if coherent else float("nan")
        pm_at_0 = pm_by_beta.get(0.0, float("nan"))
        beta_peak = max(coherent, key=lambda b: pm_by_beta.get(b, -1)) if coherent else float("nan")
        pm_peak   = pm_by_beta.get(beta_peak, float("nan")) if not math.isnan(beta_peak) else float("nan")
        def _f(v): return f"{v:6.1f}" if not math.isnan(v) else "   nan"
        print(f"{axis_id:6s}  {_f(L)}  {_f(R)}  {_f(M_A)}  {pm_at_0:8.3f}  {pm_peak:9.3f}  {_f(beta_peak)}")
        threshold_rows.append({
            "axis_id": axis_id, "L_beta": L, "R_beta": R, "M_A": M_A,
            "pm_at_0": pm_at_0, "pm_peak": pm_peak, "beta_peak": beta_peak,
        })

    if threshold_rows:
        with open(THRESHOLD_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
            w.writeheader(); w.writerows(threshold_rows)
        print(f"Saved threshold summary → {THRESHOLD_CSV}")

    # ── per-emotion breakdown for stable axes ─────────────────────────────────
    print("\n=== Per-seed-emotion axis_pole_match at β=64 (stable axes) ===")
    for axis_id in STABLE_AXIS_IDS:
        if axis_id not in axis_ids:
            continue
        print(f"\n  Axis {axis_id}:")
        for emo in sorted({d["seed_emotion"] for d in data}):
            sub = [d for d in data if d["axis_id"] == axis_id and d["beta"] == 64.0 and d["seed_emotion"] == emo]
            if not sub:
                continue
            pm_m, pm_ci, n = mci([d["pm"] for d in sub])
            sn_m = _mean([d["sn"] for d in sub])
            print(f"    {emo:13s}  pm={pm_m:.3f}±{pm_ci:.3f}  sn={sn_m:.3f}  n={n}")

    # ── paired t-test vs β=0 ──────────────────────────────────────────────────
    from collections import defaultdict
    for axis_id in STABLE_AXIS_IDS:
        if axis_id not in axis_ids:
            continue
        print(f"\n=== Paired t-test: axis_pole_match vs β=0 (axis {axis_id}) ===")
        pm_by_key: dict[str, dict[float, float]] = defaultdict(dict)
        for d in data:
            if d["axis_id"] != axis_id:
                continue
            pm_by_key[d["seed_id"]][d["beta"]] = d["pm"]

        print(f"{'β':>6}  {'Δ vs 0':>10}  {'95% CI':>10}  {'t':>7}  {'p':>10}  {'n':>5}")
        for beta in beta_vals:
            if beta == 0:
                continue
            diffs = [v[beta] - v[0.0] for v in pm_by_key.values()
                     if beta in v and 0.0 in v]
            if len(diffs) < 2:
                continue
            n = len(diffs); m = sum(diffs) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
            t  = m / (sd / math.sqrt(n))
            from math import erf, sqrt as msqrt
            p   = 2 * (1 - 0.5 * (1 + erf(abs(t) / msqrt(2))))
            ci  = 1.96 * sd / math.sqrt(n)
            sig = "*" if p < 0.05 else ""
            print(f"{beta:6.1f}  {m:+.4f}     ±{ci:.4f}   {t:+.2f}   {p:.2e} {sig}  {n:5d}")


# ── plot ──────────────────────────────────────────────────────────────────────

def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    rows_gen = [json.loads(l) for l in open(GENERATION_FILE) if l.strip()]
    scores: dict[int, dict] = {}
    if EVAL_RESULTS.exists():
        for r in (json.loads(l) for l in open(EVAL_RESULTS) if l.strip()):
            idx = int(r["custom_id"].split("_")[1])
            scores[idx] = json.loads(r["response"]["body"]["choices"][0]["message"]["content"])

    data = []
    for j, row in enumerate(rows_gen):
        if j not in scores:
            continue
        s = scores[j]
        data.append({
            "seed_id":      row["seed_id"],
            "seed_emotion": row["seed_emotion"],
            "axis_id":      row["axis_id"],
            "beta":         float(row["beta"]),
            "sn":  s["soundness"],
            "mp":  s["meaning_preserved"],
            "pm":  s["axis_pole_match"],
            "sa":  s["subtle_affective_shift"],
        })

    def _mean(vals): return sum(vals) / len(vals) if vals else float("nan")
    def mci(vals):
        n = len(vals); m = _mean(vals)
        if n < 2: return m, 0.0
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
        return m, 1.96 * sd / math.sqrt(n)

    beta_vals  = sorted({d["beta"] for d in data})
    axis_ids   = sorted({d["axis_id"] for d in data})
    axis_interps = load_axis_interpretations()

    PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    metric_specs = [
        ("pm", "Axis-pole match",       "#2166ac"),
        ("sa", "Subtle affective shift", "#762a83"),
        ("sn", "Soundness",              "#4dac26"),
        ("mp", "Meaning preserved",      "#d6604d"),
    ]

    # ── Figure 1: aggregate metrics per axis ──────────────────────────────────
    n_axes = len(axis_ids)
    fig1, axes1 = plt.subplots(n_axes, 4, figsize=(16, 3.5 * n_axes), sharey=False, squeeze=False)
    fig1.suptitle("Exp 15 — β sweep along shared meta-axes\nAggregate metrics ± 95 % CI", fontsize=11)
    for row_i, axis_id in enumerate(axis_ids):
        ax_data = [d for d in data if d["axis_id"] == axis_id]
        interp  = axis_interps.get(axis_id, {})
        title_suffix = f"\n{interp.get('axis_name', axis_id)}" if interp else ""
        for ax, (key, label, colour) in zip(axes1[row_i], metric_specs):
            means = [mci([d[key] for d in ax_data if d["beta"] == b])[0] for b in beta_vals]
            cis   = [mci([d[key] for d in ax_data if d["beta"] == b])[1] for b in beta_vals]
            ax.errorbar(beta_vals, means, yerr=cis, fmt="o-", color=colour,
                        capsize=4, capthick=1.2, linewidth=1.8, markersize=5)
            ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_xlabel("β", fontsize=10)
            ax.set_title(f"{label}\n({axis_id}{title_suffix})", fontsize=9)
            ax.xaxis.set_major_locator(mticker.FixedLocator(beta_vals))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
            ax.tick_params(axis="x", labelsize=7, rotation=45)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig1.tight_layout()
    p1 = OUT_DIR / "fig1_aggregate_metrics_per_axis.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"Saved → {p1}")
    plt.close(fig1)

    # ── Figure 2: axis-pole match, all stable axes overlaid ───────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for i, axis_id in enumerate(axis_ids):
        if axis_id not in STABLE_AXIS_IDS:
            continue
        ax_data = [d for d in data if d["axis_id"] == axis_id]
        means = [mci([d["pm"] for d in ax_data if d["beta"] == b])[0] for b in beta_vals]
        cis   = [mci([d["pm"] for d in ax_data if d["beta"] == b])[1] for b in beta_vals]
        interp = axis_interps.get(axis_id, {})
        label  = f"{axis_id}: {interp.get('axis_name', axis_id)}"
        ax2.errorbar(beta_vals, means, yerr=cis, fmt="o-", color=PALETTE[i],
                     capsize=3, capthick=1, linewidth=1.5, markersize=4, label=label)
    ax2.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel("β", fontsize=10)
    ax2.set_ylabel("Axis-pole match", fontsize=10)
    ax2.set_title("Exp 15 — Axis-pole match by β (stable meta-axes m1–m3) ± 95 % CI", fontsize=11)
    ax2.xaxis.set_major_locator(mticker.FixedLocator(beta_vals))
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    fig2.tight_layout()
    p2 = OUT_DIR / "fig2_pole_match_stable_axes.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"Saved → {p2}")
    plt.close(fig2)

    # ── Figure 3: per-seed-emotion breakdown for m1 ───────────────────────────
    EMO_PALETTE = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]
    emos = sorted({d["seed_emotion"] for d in data})
    for axis_id in STABLE_AXIS_IDS:
        if axis_id not in axis_ids:
            continue
        ax_data = [d for d in data if d["axis_id"] == axis_id]
        fig3, ax3 = plt.subplots(figsize=(9, 5))
        for emo, colour in zip(emos, EMO_PALETTE):
            means, cis = [], []
            for b in beta_vals:
                vals = [d["pm"] for d in ax_data if d["seed_emotion"] == emo and d["beta"] == b]
                m, ci = mci(vals)
                means.append(m); cis.append(ci)
            ax3.errorbar(beta_vals, means, yerr=cis, fmt="o-", color=colour,
                         capsize=3, capthick=1, linewidth=1.5, markersize=4, label=emo)
        ax3.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax3.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax3.set_xlabel("β", fontsize=10)
        ax3.set_ylabel("Axis-pole match", fontsize=10)
        interp = axis_interps.get(axis_id, {})
        ax3.set_title(f"Exp 15 — {axis_id} ({interp.get('axis_name', '')})\nPer-seed-emotion axis-pole match ± 95 % CI", fontsize=10)
        ax3.xaxis.set_major_locator(mticker.FixedLocator(beta_vals))
        ax3.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
        ax3.legend(fontsize=8, ncol=2, loc="upper left")
        ax3.grid(axis="y", linestyle=":", alpha=0.5)
        fig3.tight_layout()
        p3 = OUT_DIR / f"fig3_per_emotion_{axis_id}.png"
        fig3.savefig(p3, dpi=150, bbox_inches="tight")
        print(f"Saved → {p3}")
        plt.close(fig3)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--submit",  action="store_true",
                        help="Build eval batch and submit to OpenAI")
    parser.add_argument("--merge",   action="store_true",
                        help="Merge downloaded batch JSONL → eval_results.jsonl")
    parser.add_argument("--analyse", action="store_true",
                        help="Print summary statistics (requires eval_results.jsonl)")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate figures")
    args = parser.parse_args()

    if args.submit:
        submit_batch()
    elif args.merge:
        merge_results()
    elif args.analyse:
        analyse()
    elif args.plot:
        plot()
    else:
        run_generation()
