"""
Experiment 12: α_G sweep — does a small shared-direction coefficient help?

Background
----------
The main CAA decomposition experiment (n=376) showed that at α_G=3 the shared
direction **g** is counterproductive: every condition containing g at that scale
lowers emotionality and either matches or falls below the no-steering baseline
for target-emotion match.  However, only two values of α_G were tested:
α_G=0 (residual-only) and α_G=3 (decomposed/shared-only/original-caa).

α_G=3 may simply be too large.  A small α_G (e.g. 0.25, 0.5, 1.0) might provide
a complementary emotionalisation nudge without overwhelming r̂_e.  This experiment
sweeps α_G across a fine grid while holding α_R fixed at the value that worked best.

Research question
-----------------
Is there a value of α_G in (0, 3) at which the combined vector
    Δ = α_G·g + α_R·r̂_e
outperforms residual-only (α_G=0) on target-emotion match, without sacrificing
semantic preservation or emotionality?

Design
------
α_R fixed at 5.0 (best performer from main experiment).
α_G swept over: 0, 0.25, 0.5, 1.0, 2.0, 3.0
  → 0 replicates residual-only as an anchor; 3.0 replicates the main experiment.

Source texts: first N_SOURCES test-split items (default 50).
  50 sources × 8 emotions × 6 α_G values = 2400 generations.
  (Adjust N_SOURCES via --n_sources; use --n_sources 376 for the full set.)

Evaluation: GPT-4o-mini judge (same rubric as main experiment).

Workflow
--------
Step 1 — Generate (GPU required, ~15 min at N_SOURCES=50, batch_size=8):
    cd /path/to/EmotionEngine_v2
    python local_axes_experiments/exp12_alpha_g_sweep/exp12_alpha_g_sweep.py

Step 2 — Submit eval batch to OpenAI:
    python local_axes_experiments/exp12_alpha_g_sweep/exp12_alpha_g_sweep.py --submit

Step 3 — After batch completes (~10 min), merge results:
    python local_axes_experiments/exp12_alpha_g_sweep/exp12_alpha_g_sweep.py \\
        --merge <path/to/batch_results.jsonl>

Step 4 — Analyse:
    python local_axes_experiments/exp12_alpha_g_sweep/exp12_alpha_g_sweep.py --analyse
    (writes results/alpha_g_sweep_summary.csv and prints a table)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT_DIR       = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENERATION_FILE = OUT_DIR / "generations.jsonl"
EVAL_REQUESTS   = OUT_DIR / "eval_requests.jsonl"
EVAL_RESULTS    = OUT_DIR / "eval_results.jsonl"
BATCH_ID_FILE   = OUT_DIR / "eval_batch_id.txt"
SUMMARY_CSV     = OUT_DIR / "alpha_g_sweep_summary.csv"

ACT_DIR   = REPO_ROOT / "activation" / "emotion_rewrites"
CAA_PATH  = ACT_DIR / "caa_emotion_directions.npz"
CAA_INFO  = ACT_DIR / "caa_emotion_directions_info.json"
REWRITE   = REPO_ROOT / "dataset" / "emotion_rewrites" / "emotion_rewrites.jsonl"

PRIMARY_LAYER  = 13
MAX_NEW_TOKENS = 60
DO_SAMPLE      = False
ALPHA_R        = 64
ALPHA_G_SWEEP  = [0.0, 2, 4, 8]
FILTER_EMOTIONS = None # set to None to run all emotions
N_SOURCES      = 100      # override with --n_sources
BATCH_SIZE     = 8
CACHE_FLUSH_EVERY = 4

EVAL_SYSTEM = """\
You are an expert evaluator of emotional text rewrites. Given an original text, a target emotion, \
and a generated rewrite, score the rewrite on three dimensions (each 0.0–1.0):

- meaning_preserved: Is the core propositional content of the original retained?
  1.0 = same events, entities, and relationships; 0.0 = completely different content.
- emotionality: Is the rewrite emotionally expressive at all?
  1.0 = vivid emotional language throughout; 0.0 = flat, neutral, affectively empty.
- target_emotion_match: Does the rewrite's dominant tone match the target emotion?
  1.0 = unmistakably expresses that emotion; 0.0 = expresses a different or no emotion.

Also identify the single dominant_emotion of the rewrite \
(joy | trust | fear | surprise | sadness | disgust | anger | anticipation | neutral | other) \
and write a one-sentence notes explanation."""

# ── few-shot examples ─────────────────────────────────────────────────────────
# Each entry is (user_message, assistant_json_string).
# Three examples calibrate the scoring space: high-quality match, wrong emotion,
# and meaning drift.

FEW_SHOT_EXAMPLES = [
    (
        # Ex 1 — good steering: meaning preserved, target emotion clearly expressed
        """\
Original text:
I have a job interview tomorrow.

Target emotion: fear

Generated rewrite:
I have a job interview tomorrow, and I can't stop dreading it. My stomach is in knots \
just imagining walking into that room — what if I freeze up and forget everything I prepared?

Rate the generated rewrite.""",
        '{"meaning_preserved": 0.95, "emotionality": 0.92, "target_emotion_match": 0.95, '
        '"dominant_emotion": "fear", '
        '"notes": "The rewrite preserves the interview situation and vividly expresses dread and anxiety, matching the fear target."}'
    ),
    (
        # Ex 2 — wrong emotion: target is anger but output expresses joy
        """\
Original text:
My package arrived today.

Target emotion: anger

Generated rewrite:
My package arrived today! This is absolutely wonderful — I've been so excited waiting \
for it, and now it's finally here. What a great day!

Rate the generated rewrite.""",
        '{"meaning_preserved": 0.9, "emotionality": 0.85, "target_emotion_match": 0.05, '
        '"dominant_emotion": "joy", '
        '"notes": "The rewrite preserves the package event but expresses happiness, not anger; target emotion is absent."}'
    ),
    (
        # Ex 3 — meaning drift: original specific event replaced by generic reflection
        """\
Original text:
I ran into my old teacher at the supermarket.

Target emotion: surprise

Generated rewrite:
Life is full of unexpected twists. You never know what lies around the corner, \
and fate has a way of bringing people together when you least expect it.

Rate the generated rewrite.""",
        '{"meaning_preserved": 0.1, "emotionality": 0.55, "target_emotion_match": 0.45, '
        '"dominant_emotion": "anticipation", '
        '"notes": "The specific encounter is replaced by generic reflection; meaning is mostly lost despite some surprise-adjacent language."}'
    ),
]

EVAL_USER = """\
Original text:
{base_text}

Target emotion: {target_emotion}

Generated rewrite:
{generated_text}

Rate the generated rewrite."""

# ── structured output schema ──────────────────────────────────────────────────
EVAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "eval_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "meaning_preserved":    {"type": "number"},
                "emotionality":         {"type": "number"},
                "target_emotion_match": {"type": "number"},
                "dominant_emotion": {
                    "type": "string",
                    "enum": ["joy", "trust", "fear", "surprise", "sadness",
                             "disgust", "anger", "anticipation", "neutral", "other"],
                },
                "notes": {"type": "string"},
            },
            "required": ["meaning_preserved", "emotionality", "target_emotion_match",
                         "dominant_emotion", "notes"],
            "additionalProperties": False,
        },
    },
}


# ── helpers ──────────────────────────────────────────────────────────────────

def unit_np(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def load_caa():
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
    r_raw = C - (C @ g)[:, None] * g[None, :]                         # [E, D]
    return emotion_order, g, r_raw, C


def load_test_sources(n: int) -> list[dict]:
    seen: set[str] = set()
    sources: list[dict] = []
    for line in open(REWRITE):
        r = json.loads(line)
        sid = r.get("source_id", "")
        if sid and sid not in seen and r.get("source_split") == "test":
            seen.add(sid)
            sources.append(r)
            if len(sources) >= n:
                break
    return sources


# ── generation ───────────────────────────────────────────────────────────────

def run_generation(n_sources: int) -> None:
    from utils.text_utils import make_instruction_prefix

    emotion_order, g, r_raw, C = load_caa()
    if FILTER_EMOTIONS:
        emotion_order = [e for e in emotion_order if e in FILTER_EMOTIONS]
    E = len(emotion_order)
    D = g.shape[0]

    sources = load_test_sources(n_sources)
    print(f"Sources: {len(sources)}  |  Emotions: {E}  |  α_G sweep: {ALPHA_G_SWEEP}")
    total_expected = len(sources) * E * len(ALPHA_G_SWEEP)
    print(f"Total expected: {total_expected}")

    # resumability
    existing: set[tuple[str, str, float]] = set()
    if GENERATION_FILE.exists():
        for line in open(GENERATION_FILE):
            r = json.loads(line)
            existing.add((r["source_id"], r["target_emotion"], float(r["alpha_g"])))
    pending = total_expected - len(existing)
    print(f"Existing: {len(existing)}  |  Remaining: {pending}")
    if pending == 0:
        print("All generations complete.")
        return

    # load model
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
            out = args[2]
            h = out[0] if isinstance(out, tuple) else out
            rest = out[1:] if isinstance(out, tuple) else None
            h = h.clone(); h[:, -1:, :] = h[:, -1:, :] + delta
            return (h, *rest) if rest is not None else h
        handle = layer.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    def build_delta(ag: float, ei: int) -> torch.Tensor | None:
        layer  = model.model.layers[PRIMARY_LAYER]
        device = next(layer.parameters()).device
        dt     = next(layer.parameters()).dtype
        delta  = torch.zeros(D, dtype=dt, device=device)
        # residual component (always applied, at ALPHA_R)
        rv = torch.from_numpy(r_raw[ei].copy()).to(dtype=dt, device=device)
        delta = delta + ALPHA_R * rv          # natural-norm scaling (same as main exp)
        # shared component (scale by ag, unit-normalised)
        if ag > 0:
            gv = torch.from_numpy(g.copy()).to(dtype=dt, device=device)
            delta = delta + ag * gv
        return delta.reshape(1, 1, -1)

    def generate_batch(texts: list[str], delta: torch.Tensor | None) -> list[str]:
        inputs = tokenizer(
            [make_instruction_prefix(t, tokenizer) for t in texts],
            return_tensors="pt", padding=True, truncation=True,
        ).to(model.device)
        plen = inputs["input_ids"].shape[1]
        kw   = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE,
                    pad_token_id=tokenizer.pad_token_id)
        try:
            with torch.inference_mode():
                if delta is not None:
                    with steer(PRIMARY_LAYER, delta):
                        out = model.generate(**inputs, **kw)
                else:
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
    out_f = open(GENERATION_FILE, "a")
    batch_count = 0

    with tqdm(total=pending, desc="Generating", unit="row") as pbar:
        for emo in emotion_order:
            ei = emotion_order.index(emo)
            for ag in ALPHA_G_SWEEP:
                delta = build_delta(ag, ei)
                pending_recs = [
                    rec for rec in sources
                    if (rec["source_id"], emo, ag) not in existing
                ]
                if not pending_recs:
                    continue
                for i in range(0, len(pending_recs), BATCH_SIZE):
                    batch = pending_recs[i: i + BATCH_SIZE]
                    texts = [r["base_text"].strip() for r in batch]
                    gens  = generate_batch(texts, delta)
                    for rec, gen in zip(batch, gens):
                        row = {
                            "source_id":      rec["source_id"],
                            "base_text":      rec["base_text"].strip(),
                            "target_emotion": emo,
                            "alpha_g":        ag,
                            "alpha_r":        ALPHA_R,
                            "layer":          PRIMARY_LAYER,
                            "generated_text": gen,
                        }
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        existing.add((rec["source_id"], emo, ag))
                        pbar.update(1)
                    out_f.flush()
                    batch_count += 1
                    if batch_count % CACHE_FLUSH_EVERY == 0:
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                if delta is not None:
                    del delta
    out_f.close()
    print(f"\nGeneration complete. Total rows: {len(existing)}")


# ── build eval batch ──────────────────────────────────────────────────────────

def _make_messages(base_text: str, target_emotion: str, generated_text: str) -> list[dict]:
    """Build the few-shot message list for a single eval request."""
    messages: list[dict] = [{"role": "system", "content": EVAL_SYSTEM}]
    for user_msg, assistant_json in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_json})
    messages.append({
        "role": "user",
        "content": EVAL_USER.format(
            base_text=base_text,
            target_emotion=target_emotion,
            generated_text=generated_text,
        ),
    })
    return messages


def build_eval_batch() -> None:
    rows = [json.loads(l) for l in open(GENERATION_FILE) if l.strip()]
    print(f"Building eval batch for {len(rows)} rows ...")
    requests = []
    for i, row in enumerate(rows):
        requests.append({
            "custom_id": f"exp12_{i:06d}",
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":           "gpt-4o-mini",
                "messages":        _make_messages(
                    row["base_text"],
                    row["target_emotion"],
                    row["generated_text"],
                ),
                "response_format": EVAL_SCHEMA,
                "temperature":     0,
            },
        })
    with open(EVAL_REQUESTS, "w") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(requests)} requests → {EVAL_REQUESTS}")


# ── submit to OpenAI ──────────────────────────────────────────────────────────

def submit_batch() -> None:
    build_eval_batch()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from openai_utils.openai_batch import make_client, upload_and_submit
    client   = make_client()
    batch_id = upload_and_submit(
        client, str(EVAL_REQUESTS),
        batch_index=0, n_batches=1,
        description="exp12 α_G sweep eval",
    )
    print(f"Batch ID: {batch_id}")
    BATCH_ID_FILE.write_text(batch_id)
    print(f"Saved → {BATCH_ID_FILE}")
    print(f"\nCheck status:  openai api batches retrieve {batch_id}")
    print(f"Download:      openai api batches get-output-file {batch_id} > results/batch_raw.jsonl")
    print(f"Then merge:    python exp12_alpha_g_sweep.py --merge results/batch_raw.jsonl")


# ── merge results ─────────────────────────────────────────────────────────────

def merge_results(raw_path: str) -> None:
    p = Path(raw_path)
    if not p.exists():
        sys.exit(f"Error: {p} not found")
    raw = [json.loads(l) for l in open(p) if l.strip()]
    raw.sort(key=lambda r: int(r["custom_id"].split("_")[1]))
    with open(EVAL_RESULTS, "w") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Merged {len(raw)} results → {EVAL_RESULTS}")
    print("Run with --analyse to produce summary.")


# ── analyse ───────────────────────────────────────────────────────────────────

def analyse() -> None:
    import math, csv

    rows     = [json.loads(l) for l in open(GENERATION_FILE) if l.strip()]
    res_raw  = [json.loads(l) for l in open(EVAL_RESULTS)    if l.strip()]
    scores   = {int(r["custom_id"].split("_")[1]):
                    json.loads(r["response"]["body"]["choices"][0]["message"]["content"])
                for r in res_raw}

    data = []
    for i, row in enumerate(rows):
        if i not in scores: continue
        s = scores[i]
        data.append({
            "source_id":   row["source_id"],
            "emotion":     row["target_emotion"],
            "alpha_g":     float(row["alpha_g"]),
            "tm":          s["target_emotion_match"],
            "mp":          s["meaning_preserved"],
            "em":          s["emotionality"],
        })

    def mci(vals):
        n = len(vals); m = sum(vals) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1)) if n > 1 else 0
        ci = 1.96 * sd / math.sqrt(n)
        return m, ci, n

    alpha_g_vals = sorted({d["alpha_g"] for d in data})
    emos         = sorted({d["emotion"] for d in data})
    metrics      = [("tm", "target_match"), ("mp", "meaning_preserved"), ("em", "emotionality")]

    print("\n=== Aggregate by α_G (all emotions pooled) ===")
    print(f"{'α_G':>6}  {'target_match':>14}  {'meaning_pres':>14}  {'emotionality':>14}  {'n':>5}")
    summary_rows = []
    for ag in alpha_g_vals:
        sub = [d for d in data if d["alpha_g"] == ag]
        tm  = mci([d["tm"] for d in sub])
        mp  = mci([d["mp"] for d in sub])
        em  = mci([d["em"] for d in sub])
        print(f"{ag:6.2f}  {tm[0]:.3f}±{tm[1]:.3f}     {mp[0]:.3f}±{mp[1]:.3f}     {em[0]:.3f}±{em[1]:.3f}  {tm[2]:5d}")
        summary_rows.append({"alpha_g": ag, "tm_mean": tm[0], "tm_ci": tm[1],
                              "mp_mean": mp[0], "mp_ci": mp[1],
                              "em_mean": em[0], "em_ci": em[1], "n": tm[2]})

    print("\n=== Per-emotion target_match by α_G ===")
    header = f"{'emotion':13s}" + "".join(f"  αG={ag:.2f}" for ag in alpha_g_vals)
    print(header)
    for e in emos:
        line = f"{e:13s}"
        for ag in alpha_g_vals:
            sub = [d["tm"] for d in data if d["emotion"] == e and d["alpha_g"] == ag]
            line += f"  {sum(sub)/len(sub):.3f}  " if sub else "   —    "
        print(line)

    # write CSV
    fieldnames = list(summary_rows[0].keys())
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(summary_rows)
    print(f"\nSaved summary → {SUMMARY_CSV}")

    # paired test: each α_G vs α_G=0 (residual-only anchor)
    print("\n=== Paired t-test vs α_G=0.0 (residual-only) ===")
    from collections import defaultdict
    tm_by_key = defaultdict(dict)
    for d in data:
        tm_by_key[(d["source_id"], d["emotion"])][d["alpha_g"]] = d["tm"]

    def paired_t(a, b):
        diffs = [v[a] - v[b] for v in tm_by_key.values() if a in v and b in v]
        n = len(diffs); m = sum(diffs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
        t = m / (sd / math.sqrt(n))
        from math import erf, sqrt as msqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / msqrt(2))))
        return m, 1.96 * sd / math.sqrt(n), t, p, n

    anchor = 0.0
    print(f"{'α_G':>6}  {'Δ vs 0':>10}  {'95% CI':>10}  {'t':>7}  {'p':>10}")
    for ag in alpha_g_vals:
        if ag == anchor: continue
        m, ci, t, p, n = paired_t(ag, anchor)
        sig = "*" if p < 0.05 else ""
        print(f"{ag:6.2f}  {m:+.4f}     ±{ci:.4f}   {t:+.2f}   {p:.2e} {sig}")


# ── plot ─────────────────────────────────────────────────────────────────────

def plot() -> None:
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    rows    = [json.loads(l) for l in open(GENERATION_FILE) if l.strip()]
    res_raw = [json.loads(l) for l in open(EVAL_RESULTS)    if l.strip()]
    scores  = {int(r["custom_id"].split("_")[1]):
                   json.loads(r["response"]["body"]["choices"][0]["message"]["content"])
               for r in res_raw}

    data = []
    for i, row in enumerate(rows):
        if i not in scores: continue
        s = scores[i]
        data.append({
            "source_id": row["source_id"],
            "emotion":   row["target_emotion"],
            "alpha_g":   float(row["alpha_g"]),
            "tm":        s["target_emotion_match"],
            "mp":        s["meaning_preserved"],
            "em":        s["emotionality"],
        })

    def mci(vals):
        n = len(vals); m = sum(vals) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1)) if n > 1 else 0
        return m, 1.96 * sd / math.sqrt(n)

    alpha_g_vals = sorted({d["alpha_g"] for d in data})
    emos         = sorted({d["emotion"] for d in data})

    # ── Figure 1: aggregate metrics ───────────────────────────────────────────
    metric_specs = [
        ("tm", "Target-emotion match",  "#2166ac"),
        ("em", "Emotionality",          "#4dac26"),
        ("mp", "Meaning preserved",     "#d6604d"),
    ]
    fig1, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig1.suptitle(
        f"Exp 12 — α_G sweep (α_R = {ALPHA_R} fixed)\nAggregate metrics ± 95 % CI",
        fontsize=11, y=1.02,
    )
    for ax, (key, label, colour) in zip(axes, metric_specs):
        means = [mci([d[key] for d in data if d["alpha_g"] == ag])[0] for ag in alpha_g_vals]
        cis   = [mci([d[key] for d in data if d["alpha_g"] == ag])[1] for ag in alpha_g_vals]
        ax.errorbar(alpha_g_vals, means, yerr=cis, fmt="o-", color=colour,
                    capsize=4, capthick=1.2, linewidth=1.8, markersize=5)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("α_G", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.xaxis.set_major_locator(mticker.FixedLocator(alpha_g_vals))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig1.tight_layout()
    p1 = OUT_DIR / "fig1_aggregate_metrics.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"Saved → {p1}")
    plt.close(fig1)

    # ── Figure 2: per-emotion target_match ────────────────────────────────────
    PALETTE = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for emo, colour in zip(emos, PALETTE):
        means = []
        cis   = []
        for ag in alpha_g_vals:
            vals = [d["tm"] for d in data if d["emotion"] == emo and d["alpha_g"] == ag]
            m, ci = mci(vals)
            means.append(m); cis.append(ci)
        ax2.errorbar(alpha_g_vals, means, yerr=cis, fmt="o-", color=colour,
                     capsize=3, capthick=1, linewidth=1.5, markersize=4, label=emo)
    ax2.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel("α_G", fontsize=10)
    ax2.set_ylabel("Target-emotion match", fontsize=10)
    ax2.set_title(
        f"Exp 12 — Per-emotion target_match ± 95 % CI\n(α_R = {ALPHA_R} fixed)",
        fontsize=11,
    )
    ax2.xaxis.set_major_locator(mticker.FixedLocator(alpha_g_vals))
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax2.legend(fontsize=8, ncol=2, loc="upper right")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    fig2.tight_layout()
    p2 = OUT_DIR / "fig2_per_emotion_target_match.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"Saved → {p2}")
    plt.close(fig2)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_sources", type=int, default=N_SOURCES,
                        help=f"Number of test-split sources to use (default {N_SOURCES})")
    parser.add_argument("--submit",  action="store_true",
                        help="Build eval batch and submit to OpenAI")
    parser.add_argument("--merge",   metavar="RAW_JSONL",
                        help="Sort raw batch results and write to eval_results.jsonl")
    parser.add_argument("--analyse", action="store_true",
                        help="Print summary statistics (requires eval_results.jsonl)")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate figures (requires generations + eval_results.jsonl)")
    args = parser.parse_args()

    if args.merge:
        merge_results(args.merge)
    elif args.submit:
        submit_batch()
    elif args.analyse:
        analyse()
    elif args.plot:
        plot()
    else:
        run_generation(args.n_sources)
