"""
Experiment 14: α_R sweep — seed texts pipeline.

Rewrites hand-crafted SEED_TEXTS to every target emotion at every α_R value,
then evaluates with GPT-4o-mini.  Outputs are written to results/seeds/.

Workflow
--------
Step 1 — Generate (GPU required):
    python local_axes_experiments/exp14_alpha_r_sweep/exp14_seed.py

Step 2 — Submit eval batch to OpenAI:
    python local_axes_experiments/exp14_alpha_r_sweep/exp14_seed.py --submit

Step 3 — After batch completes, merge results:
    python local_axes_experiments/exp14_alpha_r_sweep/exp14_seed.py --merge

Step 4 — Analyse:
    python local_axes_experiments/exp14_alpha_r_sweep/exp14_seed.py --analyse

Step 5 — Plot:
    python local_axes_experiments/exp14_alpha_r_sweep/exp14_seed.py --plot
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

OUT_DIR  = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_DIR = OUT_DIR / "seeds"
SEED_DIR.mkdir(parents=True, exist_ok=True)

SEED_GENERATION_FILES = [
    SEED_DIR / "seed_generations_anger.jsonl",
    SEED_DIR / "seed_generations_anticipation.jsonl",
    SEED_DIR / "seed_generations_disgust.jsonl",
    SEED_DIR / "seed_generations_fear.jsonl",
    SEED_DIR / "seed_generations_joy.jsonl",
    SEED_DIR / "seed_generations_sadness.jsonl",
    SEED_DIR / "seed_generations_surprise.jsonl",
    SEED_DIR / "seed_generations_trust.jsonl",
]
SEED_EVAL_REQUESTS = [
    SEED_DIR / "seed_generations_anger_requests.jsonl",
    SEED_DIR / "seed_generations_anticipation_requests.jsonl",
    SEED_DIR / "seed_generations_disgust_requests.jsonl",
    SEED_DIR / "seed_generations_fear_requests.jsonl",
    SEED_DIR / "seed_generations_joy_requests.jsonl",
    SEED_DIR / "seed_generations_sadness_requests.jsonl",
    SEED_DIR / "seed_generations_surprise_requests.jsonl",
    SEED_DIR / "seed_generations_trust_requests.jsonl",
]
SEED_EVAL_RESULTS = [
    SEED_DIR / "seed_generations_anger_results.jsonl",
    SEED_DIR / "seed_generations_anticipation_results.jsonl",
    SEED_DIR / "seed_generations_disgust_results.jsonl",
    SEED_DIR / "seed_generations_fear_results.jsonl",
    SEED_DIR / "seed_generations_joy_results.jsonl",
    SEED_DIR / "seed_generations_sadness_results.jsonl",
    SEED_DIR / "seed_generations_surprise_results.jsonl",
    SEED_DIR / "seed_generations_trust_results.jsonl",
]
SEED_BATCH_ID_FILE = SEED_DIR / "seed_eval_batch_id.txt"
SEED_SUMMARY_CSV   = SEED_DIR / "seed_alpha_r_sweep_summary.csv"

ACT_DIR  = REPO_ROOT / "activation" / "emotion_rewrites"
CAA_PATH = ACT_DIR / "caa_emotion_directions.npz"
CAA_INFO = ACT_DIR / "caa_emotion_directions_info.json"

PRIMARY_LAYER  = 13
MAX_NEW_TOKENS = 60
DO_SAMPLE      = False
ALPHA_G_SWEEP  = [0.0, 4.0, 8.0]
ALPHA_R_SWEEP  = [-128, -112, -96, -80, -64, -48, -32, -16, 0.0, 16, 32, 48, 64, 80, 96, 112, 128]
BATCH_SIZE     = 8
CACHE_FLUSH_EVERY = 4

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

EVAL_SYSTEM = """\
You are an expert evaluator of emotional text rewrites. \
Given an original text, a target emotion label, and a generated rewrite, \
score the rewrite on three dimensions (each 0.0–1.0):

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


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── seed generation ───────────────────────────────────────────────────────────

def run_seed_generation() -> None:
    """Rewrite each SEED_TEXT to every target emotion at every α_R value."""
    from utils.text_utils import make_instruction_prefix

    emotion_order, g, r_raw, _ = load_caa()
    E = len(emotion_order)

    print(f"Seeds: {len(SEED_TEXTS)}  |  Emotions: {E}  |  α_R sweep: {ALPHA_R_SWEEP}  |  α_G sweep: {ALPHA_G_SWEEP}")
    total_expected = len(SEED_TEXTS) * E * len(ALPHA_R_SWEEP) * len(ALPHA_G_SWEEP)
    print(f"Total expected: {total_expected}")

    existing: set[tuple[str, str, float, float]] = set()
    for gen_file in SEED_GENERATION_FILES:
        if gen_file.exists():
            for line in open(gen_file):
                r = json.loads(line)
                existing.add((r["seed_id"], r["target_emotion"], float(r["alpha_r"]), float(r["alpha_g"])))
    pending = total_expected - len(existing)
    print(f"Existing: {len(existing)}  |  Remaining: {pending}")
    if pending == 0:
        print("All seed generations complete.")
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

    def build_delta(ar: float, ag: float, ei: int) -> torch.Tensor:
        layer  = model.model.layers[PRIMARY_LAYER]
        device = next(layer.parameters()).device
        dt     = next(layer.parameters()).dtype
        rv = torch.from_numpy(r_raw[ei].copy()).to(dtype=dt, device=device)
        gv = torch.from_numpy(g.copy()).to(dtype=dt, device=device)
        return (ar * rv + ag * gv).reshape(1, 1, -1)

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
    emo_to_file = {p.stem.replace("seed_generations_", ""): p for p in SEED_GENERATION_FILES}
    out_files   = {emo: open(path, "a") for emo, path in emo_to_file.items()}
    batch_count = 0

    with tqdm(total=pending, desc="Generating seeds", unit="row") as pbar:
        for emo in emotion_order:
            ei = emotion_order.index(emo)
            for ag in ALPHA_G_SWEEP:
                for ar in ALPHA_R_SWEEP:
                    delta = build_delta(ar, ag, ei)
                    pending_seeds = [
                        s for s in SEED_TEXTS
                        if (s["seed_id"], emo, ar, ag) not in existing
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
                                "target_emotion": emo,
                                "alpha_r":        ar,
                                "alpha_g":        ag,
                                "layer":          PRIMARY_LAYER,
                                "generated_text": gen,
                            }
                            out_files[emo].write(json.dumps(row, ensure_ascii=False) + "\n")
                            existing.add((seed["seed_id"], emo, ar, ag))
                            pbar.update(1)
                        out_files[emo].flush()
                        batch_count += 1
                        if batch_count % CACHE_FLUSH_EVERY == 0:
                            gc.collect()
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                    del delta
    for f in out_files.values():
        f.close()
    print(f"\nSeed generation complete. Total rows: {len(existing)}")

    # Re-sort each file: seed_id order (as in SEED_TEXTS) → alpha_r asc → alpha_g asc
    seed_order = {s["seed_id"]: i for i, s in enumerate(SEED_TEXTS)}
    print("Sorting generation files ...")
    for gen_file in SEED_GENERATION_FILES:
        if not gen_file.exists():
            continue
        rows = [json.loads(l) for l in open(gen_file) if l.strip()]
        rows.sort(key=lambda r: (seed_order.get(r["seed_id"], 9999),
                                  float(r["alpha_r"]), float(r["alpha_g"])))
        with open(gen_file, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Sort complete.")


# ── seed eval batch ───────────────────────────────────────────────────────────

def build_seed_eval_batch() -> None:
    for i, SEED_GENERATION_FILE in enumerate(SEED_GENERATION_FILES):
        rows = [json.loads(l) for l in open(SEED_GENERATION_FILE) if l.strip()]
        print(f"Building seed eval batch for {len(rows)} rows ...")
        requests = []
        for j, row in enumerate(rows):
            requests.append({
                "custom_id": f"exp14s_{i:06d}_{j:06d}",
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":           "gpt-4o-mini",
                    "messages":        _make_messages(
                        row["seed_text"],
                        row["target_emotion"],
                        row["generated_text"],
                    ),
                    "response_format": EVAL_SCHEMA,
                    "temperature":     0,
                },
            })
        with open(SEED_EVAL_REQUESTS[i], "w") as f:
            for r in requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(requests)} requests → {SEED_EVAL_REQUESTS[i]}")


def submit_seed_batch() -> None:
    build_seed_eval_batch()
    from openai_utils.openai_batch import make_client, upload_and_submit
    client    = make_client()
    batch_ids = []
    for SEED_EVAL_REQUEST in SEED_EVAL_REQUESTS:
        batch_id = upload_and_submit(
            client, str(SEED_EVAL_REQUEST),
            batch_index=0, n_batches=1,
            description="exp14 α_R sweep seed eval",
        )
        print(f"Batch ID: {batch_id}")
        batch_ids.append(batch_id)
    SEED_BATCH_ID_FILE.write_text("\n".join(batch_ids) + "\n")
    print(f"Saved {len(batch_ids)} batch IDs → {SEED_BATCH_ID_FILE}")
    for batch_id in batch_ids:
        print(f"\nCheck:    openai api batches retrieve {batch_id}")
        print(f"Download: openai api batches get-output-file {batch_id} > {SEED_DIR / f'{batch_id}_output.jsonl'}")
    print(f"\nThen merge: python exp14_seed.py --merge")


# ── merge results ─────────────────────────────────────────────────────────────

def merge_seed_results() -> None:
    from collections import defaultdict
    if not SEED_BATCH_ID_FILE.exists():
        sys.exit(f"Error: {SEED_BATCH_ID_FILE} not found — run --submit first")
    batch_ids = [l.strip() for l in SEED_BATCH_ID_FILE.read_text().splitlines() if l.strip()]
    raw = []
    for batch_id in batch_ids:
        p = SEED_DIR / f"{batch_id}_output.jsonl"
        if not p.exists():
            sys.exit(f"Error: {p} not found — download the batch output first")
        raw.extend(json.loads(l) for l in open(p) if l.strip())
    # custom_id format: exp14s_{i:06d}_{j:06d}
    buckets = defaultdict(list)
    for r in raw:
        parts = r["custom_id"].split("_")
        i, j = int(parts[1]), int(parts[2])
        buckets[i].append((j, r))
    total = 0
    for i, results_file in enumerate(SEED_EVAL_RESULTS):
        group = sorted(buckets[i], key=lambda x: x[0])
        with open(results_file, "w") as f:
            for _, r in group:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Merged {len(group)} results → {results_file}")
        total += len(group)
    print(f"Total: {total} results merged.")
    print("Run with --analyse to produce summary.")


# ── analyse ───────────────────────────────────────────────────────────────────

def analyse_seeds() -> None:
    import math, csv
    from collections import defaultdict

    all_rows = []
    for i, gen_file in enumerate(SEED_GENERATION_FILES):
        for j, row in enumerate(json.loads(l) for l in open(gen_file) if l.strip()):
            all_rows.append((i, j, row))

    scores = {}
    for res_file in SEED_EVAL_RESULTS:
        if not res_file.exists():
            continue
        for r in (json.loads(l) for l in open(res_file) if l.strip()):
            parts = r["custom_id"].split("_")
            i, j = int(parts[1]), int(parts[2])
            scores[(i, j)] = json.loads(r["response"]["body"]["choices"][0]["message"]["content"])

    data = []
    for i, j, row in all_rows:
        if (i, j) not in scores: continue
        s = scores[(i, j)]
        data.append({
            "seed_id":      row["seed_id"],
            "seed_emotion": row["seed_emotion"],
            "emotion":      row["target_emotion"],
            "alpha_r":      float(row["alpha_r"]),
            "sn":           s["soundness"],
            "mp":           s["meaning_preserved"],
            "ti":           s["target_emotion_intensity"],
            "de":           s["dominant_emotion"],
        })

    def _mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def mci(vals):
        n = len(vals); m = _mean(vals)
        if n < 2: return m, float("nan"), n
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
        return m, 1.96 * sd / math.sqrt(n), n

    alpha_r_vals     = sorted({d["alpha_r"] for d in data})
    alpha_g_vals     = sorted({d["alpha_g"] for d in data})
    emos             = sorted({d["emotion"]  for d in data})
    SOUNDNESS_THRESHOLD = 0.70

    summary_rows = []
    for ag in alpha_g_vals:
        ag_data = [d for d in data if d["alpha_g"] == ag]
        print(f"\n=== Seed aggregate by α_R (α_G={ag}, all emotions pooled) ===")
        print(f"{'α_R':>6}  {'soundness':>12}  {'meaning_pres':>14}  {'tgt_intensity':>15}  {'n':>5}")
        for ar in alpha_r_vals:
            sub = [d for d in ag_data if d["alpha_r"] == ar]
            if not sub: continue
            sn = mci([d["sn"] for d in sub])
            mp = mci([d["mp"] for d in sub])
            ti = mci([d["ti"] for d in sub])
            ci_sn = f"±{sn[1]:.3f}" if not math.isnan(sn[1]) else "     "
            ci_mp = f"±{mp[1]:.3f}" if not math.isnan(mp[1]) else "     "
            ci_ti = f"±{ti[1]:.3f}" if not math.isnan(ti[1]) else "     "
            print(f"{ar:6.2f}  {sn[0]:.3f}{ci_sn}   {mp[0]:.3f}{ci_mp}     {ti[0]:.3f}{ci_ti}  {sn[2]:5d}")
            summary_rows.append({
                "alpha_g": ag,
                "alpha_r": ar,
                "sn_mean": sn[0], "sn_ci": sn[1],
                "mp_mean": mp[0], "mp_ci": mp[1],
                "ti_mean": ti[0], "ti_ci": ti[1],
                "n": sn[2],
            })

    # ── Per-emotion thresholds and midpoints (mirrors exp14_alpha_r_sweep) ────
    threshold_rows = []
    for ag in alpha_g_vals:
        ag_data = [d for d in data if d["alpha_g"] == ag]
        print(f"\n=== Seed per-emotion coherence thresholds and midpoints "
              f"(α_G={ag}, soundness ≥ {SOUNDNESS_THRESHOLD}) ===")
        print(
            f"{'emotion':13s}  {'L_e':>6}  {'R_e':>6}  "
            f"{'M_A(arith)':>11}  {'M_B(mp_max)':>12}  "
            f"{'ti@M_A':>7}  {'ti@M_B':>7}  "
            f"{'opt_pos':>8}  {'opt_neg':>8}"
        )
        for emo in emos:
            sn_by_ar = {}; mp_by_ar = {}; ti_by_ar = {}
            for ar in alpha_r_vals:
                sub = [d for d in ag_data if d["emotion"] == emo and d["alpha_r"] == ar]
                if sub:
                    sn_by_ar[ar] = _mean([d["sn"] for d in sub])
                    mp_by_ar[ar] = _mean([d["mp"] for d in sub])
                    ti_by_ar[ar] = _mean([d["ti"] for d in sub])

            coherent = sorted(ar for ar in alpha_r_vals
                              if ar in sn_by_ar and sn_by_ar[ar] >= SOUNDNESS_THRESHOLD)
            L_e = coherent[0]  if coherent else float("nan")
            R_e = coherent[-1] if coherent else float("nan")
            M_A = (L_e + R_e) / 2 if coherent else float("nan")
            M_B = (max(coherent, key=lambda ar: mp_by_ar.get(ar, -1))
                   if coherent else float("nan"))

            def _ti_interp(alpha, _ti_by_ar=ti_by_ar):
                if math.isnan(alpha): return float("nan")
                ars = sorted(_ti_by_ar)
                if not ars: return float("nan")
                for k in range(len(ars) - 1):
                    if ars[k] <= alpha <= ars[k + 1]:
                        span = ars[k + 1] - ars[k]
                        t = (alpha - ars[k]) / span if span else 0.0
                        return _ti_by_ar[ars[k]] * (1 - t) + _ti_by_ar[ars[k + 1]] * t
                return _ti_by_ar[ars[0]] if alpha <= ars[0] else _ti_by_ar[ars[-1]]

            ti_at_M_A = _ti_interp(M_A)
            ti_at_M_B = (ti_by_ar.get(M_B, float("nan"))
                         if not math.isnan(M_B) else float("nan"))
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
                "alpha_g": ag,
                "emotion": emo, "L_e": L_e, "R_e": R_e, "M_A": M_A, "M_B": M_B,
                "ti_at_M_A": ti_at_M_A, "ti_at_M_B": ti_at_M_B,
                "opt_pos_r": opt_pos, "opt_neg_r": opt_neg,
            })

    # same-emotion rows: seed_emotion == target_emotion (identity rewrite)
    same = [d for d in data if d["seed_emotion"] == d["emotion"]]
    if same:
        for ag in alpha_g_vals:
            ag_same = [d for d in same if d["alpha_g"] == ag]
            if not ag_same: continue
            print(f"\n=== Identity rewrites (seed_emotion == target_emotion, α_G={ag}) by α_R ===")
            print(f"{'α_R':>6}  {'soundness':>12}  {'meaning_pres':>14}  {'tgt_intensity':>15}  {'n':>5}")
            for ar in alpha_r_vals:
                sub = [d for d in ag_same if d["alpha_r"] == ar]
                if not sub: continue
                sn = mci([d["sn"] for d in sub])
                mp = mci([d["mp"] for d in sub])
                ti = mci([d["ti"] for d in sub])
                ci_sn = f"±{sn[1]:.3f}" if not math.isnan(sn[1]) else "     "
                ci_mp = f"±{mp[1]:.3f}" if not math.isnan(mp[1]) else "     "
                ci_ti = f"±{ti[1]:.3f}" if not math.isnan(ti[1]) else "     "
                print(f"{ar:6.2f}  {sn[0]:.3f}{ci_sn}   {mp[0]:.3f}{ci_mp}     {ti[0]:.3f}{ci_ti}  {sn[2]:5d}")

    if summary_rows:
        with open(SEED_SUMMARY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f"\nSaved aggregate summary → {SEED_SUMMARY_CSV}")

    seed_threshold_csv = SEED_DIR / "seed_alpha_r_thresholds.csv"
    if threshold_rows:
        with open(seed_threshold_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
            w.writeheader(); w.writerows(threshold_rows)
        print(f"Saved threshold summary → {seed_threshold_csv}")

    # paired t-test vs α_R=0 anchor (per α_G)
    anchor = 0.0
    for ag in alpha_g_vals:
        ag_data = [d for d in data if d["alpha_g"] == ag]
        print(f"\n=== Paired t-test: target_emotion_intensity vs α_R=0.0 (α_G={ag}) ===")
        ti_by_key = defaultdict(dict)
        for d in ag_data:
            ti_by_key[(d["seed_id"], d["emotion"])][d["alpha_r"]] = d["ti"]

        def paired_t(a, b, _ti_by_key=ti_by_key):
            diffs = [v[a] - v[b] for v in _ti_by_key.values() if a in v and b in v]
            if len(diffs) < 2:
                return float("nan"), float("nan"), float("nan"), float("nan"), len(diffs)
            n = len(diffs); m = sum(diffs) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
            t  = m / (sd / math.sqrt(n))
            from math import erf, sqrt as msqrt
            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / msqrt(2))))
            return m, 1.96 * sd / math.sqrt(n), t, p, n

        print(f"{'α_R':>6}  {'Δ vs 0':>10}  {'95% CI':>10}  {'t':>7}  {'p':>10}  {'n':>5}")
        for ar in alpha_r_vals:
            if ar == anchor: continue
            m, ci, t, p, n = paired_t(ar, anchor)
            if math.isnan(m): continue
            sig = "*" if p < 0.05 else ""
            print(f"{ar:6.2f}  {m:+.4f}     ±{ci:.4f}   {t:+.2f}   {p:.2e} {sig}  {n:5d}")


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_seeds() -> None:
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    all_rows = []
    for i, gen_file in enumerate(SEED_GENERATION_FILES):
        for j, row in enumerate(json.loads(l) for l in open(gen_file) if l.strip()):
            all_rows.append((i, j, row))

    scores = {}
    for res_file in SEED_EVAL_RESULTS:
        if not res_file.exists():
            continue
        for r in (json.loads(l) for l in open(res_file) if l.strip()):
            parts = r["custom_id"].split("_")
            i, j = int(parts[1]), int(parts[2])
            scores[(i, j)] = json.loads(r["response"]["body"]["choices"][0]["message"]["content"])

    data = []
    for i, j, row in all_rows:
        if (i, j) not in scores: continue
        s = scores[(i, j)]
        data.append({
            "seed_id":      row["seed_id"],
            "seed_emotion": row["seed_emotion"],
            "emotion":      row["target_emotion"],
            "alpha_r":      float(row["alpha_r"]),
            "alpha_g":      float(row["alpha_g"]),
            "sn":           s["soundness"],
            "mp":           s["meaning_preserved"],
            "ti":           s["target_emotion_intensity"],
        })

    def _mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def mci(vals):
        n = len(vals); m = _mean(vals)
        if n < 2: return m, 0.0
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
        return m, 1.96 * sd / math.sqrt(n)

    alpha_r_vals = sorted({d["alpha_r"] for d in data})
    alpha_g_vals = sorted({d["alpha_g"] for d in data})
    emos         = sorted({d["emotion"]  for d in data})
    PALETTE = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]
    AG_STYLES = ["-", "--", ":"]

    metric_specs = [
        ("ti", "Target-emotion intensity", "#2166ac"),
        ("sn", "Soundness",                "#4dac26"),
        ("mp", "Meaning preserved",        "#d6604d"),
    ]

    # ── Figure 3: aggregate metrics — one row per α_G ────────────────────────
    n_ag = len(alpha_g_vals)
    fig3, axes3 = plt.subplots(n_ag, 3, figsize=(13, 4 * n_ag), sharey=False, squeeze=False)
    fig3.suptitle("Exp 14 — α_R sweep, seed texts\nAggregate metrics ± 95 % CI", fontsize=11)
    for row_i, ag in enumerate(alpha_g_vals):
        ag_data = [d for d in data if d["alpha_g"] == ag]
        for ax, (key, label, colour) in zip(axes3[row_i], metric_specs):
            means = [mci([d[key] for d in ag_data if d["alpha_r"] == ar])[0] for ar in alpha_r_vals]
            cis   = [mci([d[key] for d in ag_data if d["alpha_r"] == ar])[1] for ar in alpha_r_vals]
            ax.errorbar(alpha_r_vals, means, yerr=cis, fmt="o-", color=colour,
                        capsize=4, capthick=1.2, linewidth=1.8, markersize=5)
            ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_xlabel("α_R", fontsize=10)
            ax.set_title(f"{label}\n(α_G={ag})", fontsize=10)
            ax.xaxis.set_major_locator(mticker.FixedLocator(alpha_r_vals))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig3.tight_layout()
    p3 = OUT_DIR / "fig3_seed_aggregate_metrics.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"Saved → {p3}")
    plt.close(fig3)

    # ── Figure 4: per-target-emotion intensity — one figure per α_G ──────────
    for ag in alpha_g_vals:
        ag_data = [d for d in data if d["alpha_g"] == ag]
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        for emo, colour in zip(emos, PALETTE):
            means, cis = [], []
            for ar in alpha_r_vals:
                vals = [d["ti"] for d in ag_data if d["emotion"] == emo and d["alpha_r"] == ar]
                m, ci = mci(vals)
                means.append(m); cis.append(ci)
            ax4.errorbar(alpha_r_vals, means, yerr=cis, fmt="o-", color=colour,
                         capsize=3, capthick=1, linewidth=1.5, markersize=4, label=emo)
        ax4.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax4.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax4.set_xlabel("α_R", fontsize=10)
        ax4.set_ylabel("Target-emotion intensity", fontsize=10)
        ax4.set_title(
            f"Exp 14 — Seed texts: per-target-emotion intensity ± 95 % CI\n(α_G = {ag})",
            fontsize=11,
        )
        ax4.xaxis.set_major_locator(mticker.FixedLocator(alpha_r_vals))
        ax4.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
        ax4.legend(fontsize=8, ncol=2, loc="upper left")
        ax4.grid(axis="y", linestyle=":", alpha=0.5)
        fig4.tight_layout()
        ag_tag = f"{ag:g}".replace(".", "p")
        p4 = OUT_DIR / f"fig4_seed_per_emotion_intensity_ag{ag_tag}.png"
        fig4.savefig(p4, dpi=150, bbox_inches="tight")
        print(f"Saved → {p4}")
        plt.close(fig4)

    # ── Figure 4b: all α_G overlaid per emotion (linestyle per α_G) ──────────
    fig4b, ax4b = plt.subplots(figsize=(10, 5))
    for emo, colour in zip(emos, PALETTE):
        for ag, ls in zip(alpha_g_vals, AG_STYLES):
            ag_data = [d for d in data if d["alpha_g"] == ag]
            means = [mci([d["ti"] for d in ag_data if d["emotion"] == emo and d["alpha_r"] == ar])[0]
                     for ar in alpha_r_vals]
            lbl = f"{emo} (α_G={ag})" if ag == alpha_g_vals[0] else f"_ (α_G={ag})"
            ax4b.plot(alpha_r_vals, means, color=colour, linestyle=ls, linewidth=1.4,
                      marker="o", markersize=3, label=lbl)
    ax4b.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax4b.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax4b.set_xlabel("α_R", fontsize=10)
    ax4b.set_ylabel("Target-emotion intensity", fontsize=10)
    ax4b.set_title("Exp 14 — Seed: per-emotion intensity, all α_G overlaid\n"
                   "(solid=0, dashed=4, dotted=8)", fontsize=11)
    ax4b.xaxis.set_major_locator(mticker.FixedLocator(alpha_r_vals))
    ax4b.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
    ax4b.legend(fontsize=7, ncol=3, loc="upper left")
    ax4b.grid(axis="y", linestyle=":", alpha=0.5)
    fig4b.tight_layout()
    p4b = OUT_DIR / "fig4b_seed_per_emotion_intensity_all_ag.png"
    fig4b.savefig(p4b, dpi=150, bbox_inches="tight")
    print(f"Saved → {p4b}")
    plt.close(fig4b)

    # ── Figure 5: identity-rewrite quality (seed_emotion == target_emotion) ──
    same = [d for d in data if d["seed_emotion"] == d["emotion"]]
    if same:
        for ag in alpha_g_vals:
            ag_same = [d for d in same if d["alpha_g"] == ag]
            if not ag_same: continue
            fig5, ax5 = plt.subplots(figsize=(8, 4))
            for key, label, colour in metric_specs:
                means = [mci([d[key] for d in ag_same if d["alpha_r"] == ar])[0]
                         for ar in alpha_r_vals]
                cis   = [mci([d[key] for d in ag_same if d["alpha_r"] == ar])[1]
                         for ar in alpha_r_vals]
                ax5.errorbar(alpha_r_vals, means, yerr=cis, fmt="o-", color=colour,
                             capsize=4, capthick=1.2, linewidth=1.8, markersize=5, label=label)
            ax5.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
            ax5.set_xlabel("α_R", fontsize=10)
            ax5.set_title(
                f"Exp 14 — Identity rewrites (seed_emotion = target_emotion) ± 95 % CI\n(α_G = {ag})",
                fontsize=11,
            )
            ax5.xaxis.set_major_locator(mticker.FixedLocator(alpha_r_vals))
            ax5.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
            ax5.legend(fontsize=9)
            ax5.grid(axis="y", linestyle=":", alpha=0.5)
            fig5.tight_layout()
            ag_tag = f"{ag:g}".replace(".", "p")
            p5 = OUT_DIR / f"fig5_seed_identity_rewrite_ag{ag_tag}.png"
            fig5.savefig(p5, dpi=150, bbox_inches="tight")
            print(f"Saved → {p5}")
            plt.close(fig5)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submit",  action="store_true",
                        help="Build seed eval batch and submit to OpenAI")
    parser.add_argument("--merge",   action="store_true",
                        help="Merge downloaded batch JSONLs → per-emotion seed_eval_results.jsonl")
    parser.add_argument("--analyse", action="store_true",
                        help="Print seed summary statistics (requires seed_eval_results.jsonl)")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate seed figures")
    args = parser.parse_args()

    if args.submit:
        submit_seed_batch()
    elif args.merge:
        merge_seed_results()
    elif args.analyse:
        analyse_seeds()
    elif args.plot:
        plot_seeds()
    else:
        run_seed_generation()
