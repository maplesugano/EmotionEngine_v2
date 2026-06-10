"""
Experiment 14: α_R sweep — what is the useful range of the residual coefficient?
Test-split sources pipeline.

Background
----------
The main CAA decomposition experiment fixed α_R=5.0 (residual direction) and varied
α_G (shared direction).  Exp 12 confirmed that α_G>0 is largely counterproductive, and
that α_R=5.0 with α_G=0 (residual-only) is the best performer.  However, only one
value of α_R was ever tested in combination with the decomposition.  It is unknown:
  - Whether α_R=5.0 is near-optimal or just a convenient round number.
  - Whether negative α_R (steering *away* from the target emotion) produces coherent
    text, and if so, what effect it has.
  - Where the "text-breaking" thresholds lie on each side.

Research question
-----------------
Across what range of α_R does the residual steering vector Δ = α_R·r̂_e produce
intelligible rewrites, and where is target-emotion match maximised?

Design
------
α_G fixed at 0.0 (residual-only, best condition from Exp 12).
α_R swept over: -8, -5, -3, -1, 0, 1, 3, 5, 8
  → 0   replicates unsteered baseline.
  → 5   replicates the best condition from the main experiment.
  → ±8  are expected to produce degenerate or incoherent text.

Source texts: first N_SOURCES test-split items (default 100).
  100 sources x 8 emotions x 9 α_R values = 7200 generations.
  (Adjust N_SOURCES via --n_sources; use --n_sources 376 for the full set.)

Evaluation: GPT-4o-mini judge (same rubric as main experiment / Exp 12).

Workflow
--------
Step 1 — Generate (GPU required, ~25 min at N_SOURCES=100, batch_size=8):
    cd /path/to/EmotionEngine_v2
    python local_axes_experiments/exp14_alpha_r_sweep/exp14_alpha_r_sweep.py

Step 2 — Submit eval batch to OpenAI:
    python local_axes_experiments/exp14_alpha_r_sweep/exp14_alpha_r_sweep.py --submit

Step 3 — After batch completes (~10 min), merge results:
    python local_axes_experiments/exp14_alpha_r_sweep/exp14_alpha_r_sweep.py \\
        --merge <path/to/batch_results.jsonl>

Step 4 — Analyse:
    python local_axes_experiments/exp14_alpha_r_sweep/exp14_alpha_r_sweep.py --analyse
    (writes results/alpha_r_sweep_summary.csv and prints a table)
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
SUMMARY_CSV     = OUT_DIR / "alpha_r_sweep_summary.csv"

ACT_DIR   = REPO_ROOT / "activation" / "emotion_rewrites"
CAA_PATH  = ACT_DIR / "caa_emotion_directions.npz"
CAA_INFO  = ACT_DIR / "caa_emotion_directions_info.json"
REWRITE   = REPO_ROOT / "dataset" / "emotion_rewrites" / "emotion_rewrites.jsonl"

PRIMARY_LAYER  = 13
MAX_NEW_TOKENS = 60
DO_SAMPLE      = False
ALPHA_G        = 0.0
ALPHA_R_SWEEP  = [-128, -112, -96, -80, -64, -48, -32, -16, 0.0, 16, 32, 48, 64, 80, 96, 112, 128]
N_SOURCES      = 100      # override with --n_sources
BATCH_SIZE     = 8
CACHE_FLUSH_EVERY = 4

EVAL_SYSTEM = """\
You are an expert evaluator of emotional text rewrites. \
Given an original text, a target emotion label, and a generated rewrite, \
score the rewrite on three dimensions (each 0.0-1.0):

- soundness: Is the rewrite coherent, grammatically correct, and free of repetition?
  1.0 = fluent and natural; 0.0 = repetitive loops, incoherent, or grammatically broken.
  IMPORTANT: if the text ends abruptly mid-sentence (truncated by a length limit), \
evaluate only the text that is present — a clean cut-off does NOT penalise soundness. \
Only penalise for repetition, looping phrases, or genuine incoherence.
- meaning_preserved: Is the core propositional content of the original retained?
  1.0 = same situation and content; 0.0 = completely different.
- target_emotion_intensity: How strongly does the rewrite express the target emotion?
  1.0 = unmistakably and strongly expresses that emotion; \
0.5 = mild or ambiguous presence; 0.0 = not present at all.

Also identify the single dominant_emotion that best characterises the rewrite's overall tone \
(joy | trust | fear | surprise | sadness | disgust | anger | anticipation | neutral | other) \
and write a one-sentence notes explanation."""

FEW_SHOT_EXAMPLES = [
]

EVAL_USER = """\
Original text:
{base_text}

Target emotion: {target_emotion}

Generated rewrite:
{generated_text}

Rate the generated rewrite."""

EVAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "eval_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "soundness":                {"type": "number"},
                "meaning_preserved":        {"type": "number"},
                "target_emotion_intensity": {"type": "number"},
                "dominant_emotion": {
                    "type": "string",
                    "enum": ["joy", "trust", "fear", "surprise", "sadness",
                             "disgust", "anger", "anticipation", "neutral", "other"],
                },
                "notes": {"type": "string"},
            },
            "required": ["soundness", "meaning_preserved", "target_emotion_intensity",
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


def _make_messages(base_text: str, target_emotion: str, generated_text: str) -> list[dict]:
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


# ── generation ───────────────────────────────────────────────────────────────

def run_generation(n_sources: int) -> None:
    from utils.text_utils import make_instruction_prefix

    emotion_order, g, r_raw, _ = load_caa()
    E = len(emotion_order)
    D = g.shape[0]

    sources = load_test_sources(n_sources)
    print(f"Sources: {len(sources)}  |  Emotions: {E}  |  α_R sweep: {ALPHA_R_SWEEP}")
    total_expected = len(sources) * E * len(ALPHA_R_SWEEP)
    print(f"Total expected: {total_expected}")

    # resumability
    existing: set[tuple[str, str, float]] = set()
    if GENERATION_FILE.exists():
        _dec = json.JSONDecoder()
        for line in open(GENERATION_FILE):
            line = line.strip()
            if not line:
                continue
            pos = 0
            while pos < len(line):
                chunk = line[pos:].lstrip()
                if not chunk:
                    break
                r, end = _dec.raw_decode(chunk)
                existing.add((r["source_id"], r["target_emotion"], float(r["alpha_r"])))
                pos += (len(line[pos:]) - len(chunk)) + end
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

    def build_delta(ar: float, ei: int) -> torch.Tensor | None:
        layer  = model.model.layers[PRIMARY_LAYER]
        device = next(layer.parameters()).device
        dt     = next(layer.parameters()).dtype
        delta  = torch.zeros(D, dtype=dt, device=device)
        rv = torch.from_numpy(r_raw[ei].copy()).to(dtype=dt, device=device)
        delta = delta + ar * rv
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
            for ar in ALPHA_R_SWEEP:
                delta = build_delta(ar, ei)
                pending_recs = [
                    rec for rec in sources
                    if (rec["source_id"], emo, ar) not in existing
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
                            "alpha_r":        ar,
                            "alpha_g":        ALPHA_G,
                            "layer":          PRIMARY_LAYER,
                            "generated_text": gen,
                        }
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        existing.add((rec["source_id"], emo, ar))
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

def build_eval_batch() -> None:
    rows = [json.loads(l) for l in open(GENERATION_FILE) if l.strip()]
    print(f"Building eval batch for {len(rows)} rows ...")
    requests = []
    for i, row in enumerate(rows):
        requests.append({
            "custom_id": f"exp14_{i:06d}",
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
        description="exp14 α_R sweep eval",
    )
    print(f"Batch ID: {batch_id}")
    BATCH_ID_FILE.write_text(batch_id)
    print(f"Saved → {BATCH_ID_FILE}")
    print(f"\nCheck status:  openai api batches retrieve {batch_id}")
    print(f"Download:      openai api batches get-output-file {batch_id} > results/batch_raw.jsonl")
    print(f"Then merge:    python exp14_alpha_r_sweep.py --merge results/batch_raw.jsonl")


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
    from collections import defaultdict, Counter

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
            "alpha_r":   float(row["alpha_r"]),
            "sn":        s["soundness"],
            "mp":        s["meaning_preserved"],
            "ti":        s["target_emotion_intensity"],
            "de":        s["dominant_emotion"],
        })

    def _mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def mci(vals):
        n = len(vals); m = _mean(vals)
        if n < 2: return m, float("nan"), n
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
        return m, 1.96 * sd / math.sqrt(n), n

    alpha_r_vals    = sorted({d["alpha_r"] for d in data})
    emos            = sorted({d["emotion"]  for d in data})
    SOUNDNESS_THRESHOLD = 0.70   # coherence cutoff

    # ── Aggregate table ───────────────────────────────────────────────────────
    print("\n=== Aggregate by α_R (all emotions pooled) ===")
    print(f"{'α_R':>6}  {'soundness':>12}  {'meaning_pres':>14}  {'tgt_intensity':>15}  {'n':>5}")
    summary_rows = []
    for ar in alpha_r_vals:
        sub = [d for d in data if d["alpha_r"] == ar]
        if not sub: continue
        sn = mci([d["sn"] for d in sub])
        mp = mci([d["mp"] for d in sub])
        ti = mci([d["ti"] for d in sub])
        ci_sn = f"±{sn[1]:.3f}" if not math.isnan(sn[1]) else "     "
        ci_mp = f"±{mp[1]:.3f}" if not math.isnan(mp[1]) else "     "
        ci_ti = f"±{ti[1]:.3f}" if not math.isnan(ti[1]) else "     "
        print(f"{ar:6.2f}  {sn[0]:.3f}{ci_sn}   {mp[0]:.3f}{ci_mp}     {ti[0]:.3f}{ci_ti}  {sn[2]:5d}")
        summary_rows.append({
            "alpha_r": ar,
            "sn_mean": sn[0], "sn_ci": sn[1],
            "mp_mean": mp[0], "mp_ci": mp[1],
            "ti_mean": ti[0], "ti_ci": ti[1],
            "n": sn[2],
        })

    # ── Per-emotion thresholds and midpoints ──────────────────────────────────
    #
    # M_A (structural) = (L_e + R_e) / 2
    #   The arithmetic midpoint of the coherent α_R range.
    #   Interpretation: the geometric centre of where text stays intact.
    #
    # M_B (empirical)  = argmax_{α in coherent} mean(meaning_preserved)
    #   The α_R at which the rewrite best preserves the original base text.
    #   Interpretation: "closest to base" — minimises the steering effect on content.
    #
    # Comparing M_A vs M_B reveals whether the coherence window is symmetric
    # around the functionally neutral point.
    #
    print(f"\n=== Per-emotion coherence thresholds and midpoints "
          f"(soundness ≥ {SOUNDNESS_THRESHOLD}) ===")
    print(
        f"{'emotion':13s}  {'L_e':>6}  {'R_e':>6}  "
        f"{'M_A(arith)':>11}  {'M_B(mp_max)':>12}  "
        f"{'ti@M_A':>7}  {'ti@M_B':>7}  "
        f"{'opt_pos':>8}  {'opt_neg':>8}"
    )
    threshold_rows = []
    for emo in emos:
        sn_by_ar = {}
        mp_by_ar = {}
        ti_by_ar = {}
        for ar in alpha_r_vals:
            sub = [d for d in data if d["emotion"] == emo and d["alpha_r"] == ar]
            if sub:
                sn_by_ar[ar] = _mean([d["sn"] for d in sub])
                mp_by_ar[ar] = _mean([d["mp"] for d in sub])
                ti_by_ar[ar] = _mean([d["ti"] for d in sub])

        coherent = sorted(
            ar for ar in alpha_r_vals
            if ar in sn_by_ar and sn_by_ar[ar] >= SOUNDNESS_THRESHOLD
        )
        L_e = coherent[0]  if coherent else float("nan")
        R_e = coherent[-1] if coherent else float("nan")
        M_A = (L_e + R_e) / 2 if coherent else float("nan")

        # M_B: coherent α_R with highest mean meaning_preserved
        M_B = (max(coherent, key=lambda ar: mp_by_ar.get(ar, -1))
               if coherent else float("nan"))

        # ti at M_A via linear interpolation (M_A may fall between sweep points)
        def _ti_interp(alpha):
            if math.isnan(alpha): return float("nan")
            ars = sorted(ti_by_ar)
            if not ars: return float("nan")
            for k in range(len(ars) - 1):
                if ars[k] <= alpha <= ars[k + 1]:
                    span = ars[k + 1] - ars[k]
                    t = (alpha - ars[k]) / span if span else 0.0
                    return ti_by_ar[ars[k]] * (1 - t) + ti_by_ar[ars[k + 1]] * t
            return ti_by_ar[ars[0]] if alpha <= ars[0] else ti_by_ar[ars[-1]]

        ti_at_M_A = _ti_interp(M_A)
        ti_at_M_B = ti_by_ar.get(M_B, float("nan")) if not math.isnan(M_B) else float("nan")

        opt_pos = max(coherent, key=lambda ar: ti_by_ar.get(ar, -1)) if coherent else float("nan")
        opt_neg = min(coherent, key=lambda ar: ti_by_ar.get(ar,  2)) if coherent else float("nan")

        def _f(v): return f"{v:6.1f}" if not math.isnan(v) else "   nan"

        print(
            f"{emo:13s}  {_f(L_e)}  {_f(R_e)}  "
            f"{_f(M_A):>11}  {_f(M_B):>12}  "
            f"{ti_at_M_A:7.3f}  {ti_at_M_B:7.3f}  "
            f"{_f(opt_pos):>8}  {_f(opt_neg):>8}"
        )
        threshold_rows.append({
            "emotion": emo,
            "L_e": L_e, "R_e": R_e, "M_A": M_A, "M_B": M_B,
            "ti_at_M_A": ti_at_M_A, "ti_at_M_B": ti_at_M_B,
            "opt_pos_r": opt_pos, "opt_neg_r": opt_neg,
        })

    # ── Dominant-emotion spot-check ───────────────────────────────────────────
    print("\n=== Dominant-emotion distribution by α_R (all target emotions pooled) ===")
    for ar in alpha_r_vals:
        sub = [d["de"] for d in data if d["alpha_r"] == ar]
        if not sub: continue
        top3 = ", ".join(f"{k}:{v}" for k, v in Counter(sub).most_common(3))
        print(f"  α_R={ar:6.1f}: {top3}")

    # ── Paired t-test: intensity vs α_R=0 ────────────────────────────────────
    print("\n=== Paired t-test: target_emotion_intensity vs α_R=0.0 ===")
    ti_by_key = defaultdict(dict)
    for d in data:
        ti_by_key[(d["source_id"], d["emotion"])][d["alpha_r"]] = d["ti"]

    def paired_t(a, b):
        diffs = [v[a] - v[b] for v in ti_by_key.values() if a in v and b in v]
        if len(diffs) < 2:
            return float("nan"), float("nan"), float("nan"), float("nan"), len(diffs)
        n = len(diffs); m = sum(diffs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
        t  = m / (sd / math.sqrt(n))
        from math import erf, sqrt as msqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / msqrt(2))))
        return m, 1.96 * sd / math.sqrt(n), t, p, n

    anchor = 0.0
    print(f"{'α_R':>6}  {'Δ vs 0':>10}  {'95% CI':>10}  {'t':>7}  {'p':>10}  {'n':>5}")
    for ar in alpha_r_vals:
        if ar == anchor: continue
        m, ci, t, p, n = paired_t(ar, anchor)
        if math.isnan(m): continue
        sig = "*" if p < 0.05 else ""
        print(f"{ar:6.2f}  {m:+.4f}     ±{ci:.4f}   {t:+.2f}   {p:.2e} {sig}  {n:5d}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    if summary_rows:
        with open(SUMMARY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f"\nSaved aggregate summary → {SUMMARY_CSV}")

    threshold_csv = OUT_DIR / "alpha_r_thresholds.csv"
    if threshold_rows:
        with open(threshold_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
            w.writeheader(); w.writerows(threshold_rows)
        print(f"Saved threshold summary → {threshold_csv}")


# ── plot ─────────────────────────────────────────────────────────────────────

def plot() -> None:
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # ── publication-quality global style ─────────────────────────────────────
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          9,
        "axes.labelsize":     10,
        "axes.titlesize":     10,
        "axes.titleweight":   "bold",
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.fontsize":    8,
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "0.8",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "grid.alpha":         0.35,
        "grid.linestyle":     ":",
        "grid.linewidth":     0.6,
        "lines.linewidth":    1.6,
        "lines.markersize":   5,
        "errorbar.capsize":   3,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })

    DPI           = 300
    # Okabe-Ito 8-colour palette (colourblind-friendly)
    PALETTE = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999",
    ]

    def _save(fig, stem):
        for ext in ("pdf", "png"):
            p = OUT_DIR / f"{stem}.{ext}"
            fig.savefig(p, dpi=DPI)
            print(f"Saved → {p}")

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
            "alpha_r":   float(row["alpha_r"]),
            "sn":        s["soundness"],
            "mp":        s["meaning_preserved"],
            "ti":        s["target_emotion_intensity"],
        })

    def _mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def _mci_series(key, emo=None):
        """Return (means, lo, hi) arrays over alpha_r_vals for CI bands."""
        means, lo, hi = [], [], []
        for ar in alpha_r_vals:
            sub = [d[key] for d in data
                   if d["alpha_r"] == ar and (emo is None or d["emotion"] == emo)]
            if not sub:
                means.append(float("nan"))
                lo.append(float("nan")); hi.append(float("nan"))
                continue
            m = _mean(sub); n = len(sub)
            ci = 0.0
            if n > 1:
                sd = math.sqrt(sum((x - m) ** 2 for x in sub) / (n - 1))
                ci = 1.96 * sd / math.sqrt(n)
            means.append(m); lo.append(m - ci); hi.append(m + ci)
        return means, lo, hi

    alpha_r_vals     = sorted({d["alpha_r"] for d in data})
    emos             = sorted({d["emotion"]  for d in data})
    SOUNDNESS_THRESHOLD = 0.70
    X = alpha_r_vals  # shorthand

    # ── Figure 1: aggregate metrics (publication figure) ─────────────────────
    # 3-panel: intensity · soundness · meaning_preserved  vs α_R
    # Lines + 95 % CI fill_between bands
    metric_specs = [
        ("ti", "Target-emotion intensity", "#0072B2"),
        ("sn", "Soundness",                "#009E73"),
        ("mp", "Meaning preserved",        "#D55E00"),
    ]
    fig1, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=False)
    for ax, (key, label, col) in zip(axes, metric_specs):
        means, lo, hi = _mci_series(key)
        ax.plot(X, means, "o-", color=col, linewidth=1.6, markersize=4, zorder=3)
        ax.fill_between(X, lo, hi, color=col, alpha=0.18, zorder=2)
        ax.axvline(0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel(r"$\alpha_R$")
        ax.set_title(label)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_major_locator(mticker.FixedLocator(X))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.tick_params(axis="x", labelrotation=45)
    fig1.tight_layout(pad=0.6, w_pad=1.2)
    _save(fig1, "fig1_aggregate_metrics")
    plt.close(fig1)

    # ── Figure 2: per-emotion intensity (publication figure) ──────────────────
    # 8 emotion lines + CI bands.  Two-column layout, clean legend.
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.2))
    for emo, col in zip(emos, PALETTE):
        means, lo, hi = _mci_series("ti", emo=emo)
        ax2.plot(X, means, "o-", color=col, linewidth=1.4, markersize=3.5,
                 label=emo, zorder=3)
        ax2.fill_between(X, lo, hi, color=col, alpha=0.12, zorder=2)
    ax2.axvline(0,   color="#666666", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(0.5, color="#888888", linestyle=":",  linewidth=0.7, alpha=0.6)
    ax2.set_xlabel(r"$\alpha_R$")
    ax2.set_ylabel("Target-emotion intensity")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.xaxis.set_major_locator(mticker.FixedLocator(X))
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax2.tick_params(axis="x", labelrotation=45)
    leg = ax2.legend(ncol=2, loc="upper left", handlelength=1.4,
                     columnspacing=0.8, handletextpad=0.4)
    fig2.tight_layout(pad=0.6)
    _save(fig2, "fig2_per_emotion_intensity")
    plt.close(fig2)

    # ── Figure 3: per-emotion threshold diagnostic ────────────────────────────
    # One panel per emotion; left axis = soundness (green) + CI band,
    # right axis = intensity (blue) + CI band.
    # Vertical markers: α_R=0, L_e, R_e, M_A (--·), M_B (:).
    ncols = 4
    nrows = math.ceil(len(emos) / ncols)
    fig3, axes3 = plt.subplots(nrows, ncols,
                               figsize=(ncols * 3.0, nrows * 2.6))
    axes3_flat = list(axes3.flat) if hasattr(axes3, "flat") else [axes3]

    SN_COL = "#009E73"
    TI_COL = "#0072B2"

    for ax_sn, emo in zip(axes3_flat, emos):
        sn_m, sn_lo, sn_hi = _mci_series("sn", emo=emo)
        ti_m, ti_lo, ti_hi = _mci_series("ti", emo=emo)
        mp_m, _, _          = _mci_series("mp", emo=emo)

        sn_by_ar = dict(zip(X, sn_m))
        mp_by_ar = dict(zip(X, mp_m))
        ti_by_ar = dict(zip(X, ti_m))

        coherent = sorted(ar for ar in X
                          if not math.isnan(sn_by_ar.get(ar, float("nan")))
                          and sn_by_ar[ar] >= SOUNDNESS_THRESHOLD)
        L_e = coherent[0]  if coherent else None
        R_e = coherent[-1] if coherent else None
        M_A = (L_e + R_e) / 2 if coherent else None
        M_B = (max(coherent, key=lambda ar: mp_by_ar.get(ar, -1))
               if coherent else None)

        ax_ti = ax_sn.twinx()

        # CI bands first (behind lines)
        ax_sn.fill_between(X, sn_lo, sn_hi, color=SN_COL, alpha=0.15, zorder=1)
        ax_ti.fill_between(X, ti_lo, ti_hi, color=TI_COL, alpha=0.15, zorder=1)

        ax_sn.plot(X, sn_m, "o-", color=SN_COL, linewidth=1.3,
                   markersize=3, zorder=3, label="soundness")
        ax_ti.plot(X, ti_m, "s--", color=TI_COL, linewidth=1.3,
                   markersize=3, zorder=3, label="intensity")

        # Threshold reference line
        ax_sn.axhline(SOUNDNESS_THRESHOLD, color=SN_COL,
                      linestyle=":", linewidth=0.8, alpha=0.8)

        # Vertical markers
        for x_val, ls, lw, lbl in [
            (0,   "--", 0.7, None),
            (L_e, "-",  1.1, f"L={L_e}"),
            (R_e, "-",  1.1, f"R={R_e}"),
            (M_A, "-.", 1.3, f"A={M_A:.0f}" if M_A is not None else None),
            (M_B, ":",  1.3, f"B={M_B}"     if M_B is not None else None),
        ]:
            if x_val is None: continue
            ax_sn.axvline(x_val, color="grey", linestyle=ls,
                          linewidth=lw, alpha=0.65, zorder=0)
            if lbl:
                ax_sn.text(x_val, 0.01, lbl, fontsize=5.5, color="dimgrey",
                           rotation=90, va="bottom", ha="right",
                           transform=ax_sn.get_xaxis_transform())

        ax_sn.set_title(emo, fontsize=9, fontweight="bold", pad=3)
        ax_sn.set_xlabel(r"$\alpha_R$", fontsize=7)
        ax_sn.set_ylabel("soundness", fontsize=7, color=SN_COL, labelpad=2)
        ax_ti.set_ylabel("intensity", fontsize=7, color=TI_COL, labelpad=2)
        ax_sn.set_ylim(-0.05, 1.05)
        ax_ti.set_ylim(-0.05, 1.05)
        ax_sn.set_yticks([0, 0.5, 1.0])
        ax_ti.set_yticks([0, 0.5, 1.0])
        ax_sn.xaxis.set_major_locator(mticker.FixedLocator(X))
        ax_sn.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax_sn.tick_params(axis="x", labelsize=5.5, labelrotation=45)
        ax_sn.tick_params(axis="y", labelsize=6, colors=SN_COL)
        ax_ti.tick_params(axis="y", labelsize=6, colors=TI_COL)
        # Hide top/right spines for primary axis; right spine is twinx
        ax_sn.spines["top"].set_visible(False)
        ax_ti.spines["top"].set_visible(False)

    for ax in axes3_flat[len(emos):]:
        ax.set_visible(False)

    fig3.suptitle(
        r"Coherence thresholds and midpoints per emotion  ($\alpha_G = 0$)",
        fontsize=9, y=1.01,
    )
    fig3.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.8)
    _save(fig3, "fig3_per_emotion_thresholds")
    plt.close(fig3)

    # Reset rcParams so other code is unaffected
    plt.rcdefaults()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_sources", type=int, default=N_SOURCES,
                        help=f"Number of test-split sources to use (default {N_SOURCES})")
    parser.add_argument("--submit",  action="store_true",
                        help="Build eval batch and submit to OpenAI")
    parser.add_argument("--merge",   metavar="RAW_JSONL",
                        help="Sort raw batch results → eval_results.jsonl")
    parser.add_argument("--analyse", action="store_true",
                        help="Print summary statistics (requires eval_results.jsonl)")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate figures")
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
