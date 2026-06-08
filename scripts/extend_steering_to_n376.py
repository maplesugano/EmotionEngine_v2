"""
Generates steering outputs for all 376 test-split source texts and writes
them to generation_outputs.jsonl (overwrites the original n=20 file).
Fully resumable: existing rows are skipped.  Batched generation for speed.

Workflow
--------
Step 1 — Generate (GPU required, ~1.5 hours with batch_size=8):
    cd /home/maplesugano/proj/EmotionEngine_v2
    python scripts/extend_steering_to_n376.py

Step 2 — Submit eval batch to OpenAI (run from repo root):
    python -c "
import sys; sys.path.insert(0, 'src')
from openai_utils.openai_batch import make_client, upload_and_submit
client = make_client()
batch_id = upload_and_submit(client,
    'analysis/caa/steering_decomposition/generation_eval_requests_n376.jsonl',
    batch_index=0, n_batches=1,
    description='CAA decomposition steering n=376 eval')
print('Batch ID:', batch_id)
open('analysis/caa/steering_decomposition/eval_batch_id_n376.txt', 'w').write(batch_id)
"

Step 3 — After batch completes, download results then merge
    (sorts by custom_id to match generation_outputs.jsonl row order):
    python scripts/extend_steering_to_n376.py --merge <path/to/results.jsonl>

Step 4 — Re-run thesis_artifacts_generation.ipynb to regenerate figures.
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parents[1]
ACT_DIR       = REPO_ROOT / "activation" / "emotion_rewrites"
CAA_PATH      = ACT_DIR / "caa_emotion_directions.npz"
CAA_INFO      = ACT_DIR / "caa_emotion_directions_info.json"
REWRITE_JSONL = REPO_ROOT / "dataset" / "emotion_rewrites" / "emotion_rewrites.jsonl"
OUT_DIR       = REPO_ROOT / "analysis" / "caa" / "steering_decomposition"

GENERATION_OUTFILE = OUT_DIR / "generation_outputs.jsonl"
EVAL_REQUESTS_FILE = OUT_DIR / "generation_eval_requests_n376.jsonl"
EVAL_RESULTS_FILE  = OUT_DIR / "generation_eval_results.jsonl"

sys.path.insert(0, str(REPO_ROOT / "src"))

# ── Parse args early so --merge skips GPU loading entirely ────────────────────
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--merge", metavar="RAW_RESULTS_JSONL",
    help="Sort raw batch results by custom_id and overwrite generation_eval_results.jsonl",
)
args = parser.parse_args()

# ── Merge mode (no GPU needed) ─────────────────────────────────────────────────
if args.merge:
    raw_path = Path(args.merge)
    if not raw_path.exists():
        sys.exit(f"Error: {raw_path} not found")
    raw_results = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
    raw_results.sort(key=lambda r: int(r.get("custom_id", "exp1_999999").split("_")[1]))
    with open(EVAL_RESULTS_FILE, "w") as f:
        for r in raw_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Merged {len(raw_results)} results → {EVAL_RESULTS_FILE}")
    print("Re-run thesis_artifacts_generation.ipynb to regenerate figures.")
    sys.exit(0)

from utils.text_utils import make_instruction_prefix

# ── Config ─────────────────────────────────────────────────────────────────────
PRIMARY_LAYER     = 13
MAX_NEW_TOKENS    = 60
DO_SAMPLE         = False
ALPHA_G_DEFAULT   = 3.0
ALPHA_R_DEFAULT   = 5.0
BATCH_SIZE        = 8
CACHE_FLUSH_EVERY = 4           # flush every N batches
EXP1_CONDITIONS   = ["none", "g", "resid", "g+resid", "original_caa"]

EVAL_SYSTEM_PROMPT = """\
You are an expert evaluator of emotional text rewrites.
Rate the output along three dimensions (each 0.0–1.0) and identify the dominant emotion.

Scoring rubric:
- meaning_preserved: 1.0 = core meaning fully retained; 0.0 = meaning changed completely
- emotionality: 1.0 = highly emotionally expressive; 0.0 = completely neutral/flat
- target_emotion_match: 1.0 = clearly expresses the target emotion; 0.0 = no match

Respond ONLY with valid JSON matching this schema:
{"meaning_preserved": float, "emotionality": float, "target_emotion_match": float,
 "dominant_emotion": "joy|trust|fear|surprise|sadness|disgust|anger|anticipation|neutral|other",
 "notes": "brief explanation"}"""

EVAL_USER_TEMPLATE = """\
Original text:
{base_text}

Target emotion: {target_emotion}

Generated rewrite:
{generated_text}

Rate the generated rewrite."""

# ── CAA vectors ────────────────────────────────────────────────────────────────
def unit_np(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

with open(CAA_INFO) as f:
    caa_meta = json.load(f)

EMOTION_ORDER: list[str] = caa_meta["emotion_order"]
LAYER_INDICES: list[int] = caa_meta["layer_indices"]
LI = LAYER_INDICES.index(PRIMARY_LAYER)

caa_data   = np.load(CAA_PATH)
caa_pooled = np.asarray(caa_data["caa_pooled"], dtype=np.float32)
C = np.stack([unit_np(v) for v in caa_pooled[:, LI, :]], axis=0)  # [E, D]
D = C.shape[-1]

g_raw      = C.mean(axis=0)
g          = unit_np(g_raw)
proj       = (C @ g)[:, None] * g[None, :]
resid_raw  = C - proj                  # natural scale, not unit-normalised
resid_unit = np.stack([unit_np(v) for v in resid_raw], axis=0)

print(f"CAA vectors: {len(EMOTION_ORDER)} emotions, D={D}")

# ── Source texts: test-split only ─────────────────────────────────────────────
records_raw: list[dict] = []
with open(REWRITE_JSONL) as f:
    for line in f:
        records_raw.append(json.loads(line))

seen_ids: set[str] = set()
test_sources: list[dict] = []
for rec in records_raw:
    sid = rec.get("source_id", "")
    if sid and sid not in seen_ids and rec.get("source_split") == "test":
        seen_ids.add(sid)
        test_sources.append(rec)

print(f"Test-split sources: {len(test_sources)}")   # should be 376
assert len(test_sources) >= 100, f"Too few test sources: {len(test_sources)}"

# ── Load existing rows (resumability) ─────────────────────────────────────────
existing_keys: set[tuple[str, str, str]] = set()
if GENERATION_OUTFILE.exists():
    with open(GENERATION_OUTFILE) as f:
        for line in f:
            row = json.loads(line)
            # Only count rows whose source_id is a test-split source
            if row["source_id"] in seen_ids:
                existing_keys.add((row["source_id"], row["target_emotion"],
                                    row["steering_condition"]))
    print(f"Existing test-source rows: {len(existing_keys)}")

total_expected = len(test_sources) * len(EMOTION_ORDER) * len(EXP1_CONDITIONS)
pending        = total_expected - len(existing_keys)
print(f"Total: {total_expected}  |  Done: {len(existing_keys)}  |  Remaining: {pending}")

if pending == 0:
    print("All rows already generated — jumping to eval batch generation.")

# ── Model ──────────────────────────────────────────────────────────────────────
if pending > 0:
    with open(REPO_ROOT / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    MODEL_NAME  = model_cfg["name"]
    DTYPE_MAP   = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    MODEL_DTYPE = DTYPE_MAP.get(model_cfg.get("dtype", "bfloat16"), torch.bfloat16)
    ATTN_IMPL   = model_cfg.get("attn_implementation", "sdpa")
    USE_4BIT    = bool(model_cfg.get("load_in_4bit", False))

    bnb_cfg = None
    if USE_4BIT:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPL,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_cfg,
    )
    model.eval()
    print("Model loaded.")

    # ── Steering hook ──────────────────────────────────────────────────────────
    @contextmanager
    def steer_at_layer(layer_idx: int, delta: torch.Tensor):
        """Apply a pre-built delta tensor to the last token at layer_idx."""
        layer = model.model.layers[layer_idx]

        def _hook(*args):
            out = args[2]
            h = out[0] if isinstance(out, tuple) else out
            rest = out[1:] if isinstance(out, tuple) else None
            h = h.clone()
            h[:, -1:, :] = h[:, -1:, :] + delta
            return (h, *rest) if rest is not None else h

        handle = layer.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    def build_delta(condition: str, emotion_idx: int) -> torch.Tensor | None:
        """Build the steering delta for a (condition, emotion) pair."""
        r_vec   = resid_raw[emotion_idx]
        r_alpha = ALPHA_R_DEFAULT * float(np.linalg.norm(r_vec))

        if condition == "none":
            return None
        layer     = model.model.layers[PRIMARY_LAYER]
        device    = next(layer.parameters()).device
        dtype     = next(layer.parameters()).dtype

        vecs: list[tuple[np.ndarray, float]] = []
        if condition == "g":
            vecs = [(g, ALPHA_G_DEFAULT)]
        elif condition == "resid":
            vecs = [(r_vec, r_alpha)]
        elif condition == "g+resid":
            vecs = [(g, ALPHA_G_DEFAULT), (r_vec, r_alpha)]
        elif condition == "original_caa":
            vecs = [(C[emotion_idx], ALPHA_G_DEFAULT)]
        else:
            raise ValueError(f"Unknown condition: {condition}")

        delta = torch.zeros(D, dtype=dtype, device=device)
        for vec, alpha in vecs:
            t = torch.from_numpy(vec.copy()).to(dtype=dtype, device=device)
            t = t / (t.norm() + 1e-10)
            delta = delta + alpha * t
        return delta.reshape(1, 1, -1)

    def generate_batch(base_texts: list[str], delta: torch.Tensor | None) -> list[str]:
        inputs = tokenizer(
            [make_instruction_prefix(t, tokenizer) for t in base_texts],
            return_tensors="pt", padding=True, truncation=True,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id,
        )
        try:
            with torch.inference_mode():
                if delta is not None:
                    with steer_at_layer(PRIMARY_LAYER, delta):
                        out = model.generate(**inputs, **gen_kwargs)
                else:
                    out = model.generate(**inputs, **gen_kwargs)
            results = []
            for i in range(len(base_texts)):
                ids = out[i, prompt_len:]
                results.append(tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ).strip())
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            results = ["[OOM]"] * len(base_texts)
        finally:
            del inputs
            if "out" in locals():
                del out
        return results

    # ── Generation loop: batched by (emotion, condition) ──────────────────────
    # Open file in append mode so resumability works
    out_f = open(GENERATION_OUTFILE, "a")
    batch_count = 0

    with tqdm(total=pending, desc="Generating", unit="row") as pbar:
        for emo in EMOTION_ORDER:
            ei = EMOTION_ORDER.index(emo)
            for cond in EXP1_CONDITIONS:
                delta = build_delta(cond, ei)

                # Collect all sources pending for this (emo, cond)
                pending_recs = [
                    rec for rec in test_sources
                    if (rec["source_id"], emo, cond) not in existing_keys
                ]
                if not pending_recs:
                    continue

                # Process in batches
                for i in range(0, len(pending_recs), BATCH_SIZE):
                    batch_recs = pending_recs[i : i + BATCH_SIZE]
                    texts      = [r["base_text"].strip() for r in batch_recs]
                    generated  = generate_batch(texts, delta)

                    for rec, gen in zip(batch_recs, generated):
                        row = {
                            "source_id":          rec["source_id"],
                            "base_text":          rec["base_text"].strip(),
                            "target_emotion":     emo,
                            "steering_condition": cond,
                            "layer":              PRIMARY_LAYER,
                            "alpha_g":            ALPHA_G_DEFAULT if "g" in cond else 0.0,
                            "alpha_resid":        ALPHA_R_DEFAULT if "resid" in cond else 0.0,
                            "generated_text":     gen,
                        }
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        existing_keys.add((rec["source_id"], emo, cond))
                        pbar.update(1)

                    out_f.flush()
                    batch_count += 1
                    if batch_count % CACHE_FLUSH_EVERY == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Free delta tensor after each (emo, cond) group
                if delta is not None:
                    del delta

    out_f.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nGeneration complete. Total rows in file: {len(existing_keys)}")

# ── Build eval batch (all test-source rows) ───────────────────────────────────
# Read only test-source rows (skip any legacy ed_all_* rows at top of file)
test_source_ids = {rec["source_id"] for rec in test_sources}
all_rows: list[dict] = []
with open(GENERATION_OUTFILE) as f:
    for line in f:
        row = json.loads(line)
        if row["source_id"] in test_source_ids:
            all_rows.append(row)

print(f"\nBuilding eval batch for {len(all_rows)} test-source rows …")
eval_requests: list[dict] = []
for i, row in enumerate(all_rows):
    eval_requests.append({
        "custom_id": f"exp1_{i:06d}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": EVAL_USER_TEMPLATE.format(
                    base_text      = row["base_text"],
                    target_emotion = row["target_emotion"],
                    generated_text = row["generated_text"],
                )},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
        "_meta": {
            "source_id":          row["source_id"],
            "target_emotion":     row["target_emotion"],
            "steering_condition": row["steering_condition"],
            "layer":              row["layer"],
            "alpha_g":            row["alpha_g"],
            "alpha_resid":        row["alpha_resid"],
        },
    })

with open(EVAL_REQUESTS_FILE, "w") as f:
    for req in eval_requests:
        f.write(json.dumps(req, ensure_ascii=False) + "\n")

print(f"Wrote {len(eval_requests)} eval requests → {EVAL_REQUESTS_FILE}")
print(f"""
Next steps:
  1. Submit batch to OpenAI (run from repo root):
       python -c "
import sys; sys.path.insert(0, 'src')
from openai_utils.openai_batch import make_client, upload_and_submit
client = make_client()
batch_id = upload_and_submit(client, '{EVAL_REQUESTS_FILE}', batch_index=0, n_batches=1,
    description='CAA steering n=376 eval')
print('Batch ID:', batch_id)
open('{OUT_DIR}/eval_batch_id_n376.txt', 'w').write(batch_id)
"

  2. After batch completes, download results then:
       python scripts/extend_steering_to_n376.py --merge <path/to/results.jsonl>

  3. Re-run thesis_artifacts_generation.ipynb.
""")
