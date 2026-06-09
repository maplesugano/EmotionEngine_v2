# Experiment 12: α_G Sweep

**Script:** `exp12_alpha_g_sweep.py`  
**Status:** Not yet run  
**Layer:** 13  
**α_R:** 5.0 (fixed — best performer from main experiment)  
**α_G sweep:** 0.0, 0.25, 0.5, 1.0, 2.0, 3.0  
**Default N_SOURCES:** 50 (use `--n_sources 376` for full replication)

---

## Motivation

The main CAA decomposition experiment tested only two values of α_G:
- α_G = 0 (residual-only): significantly beats baseline and original CAA
- α_G = 3 (decomposed/shared-only): at or below baseline, counterproductive

This leaves a gap: does any value of α_G in (0, 3) add value?
The shared direction **g** captures the mean emotionalisation axis across all eight
emotions. At small scales it might provide a mild "emotional activation" nudge that
complements r̂_e without overwhelming its emotion-specific signal.

**Research question:** Is there an α_G ∈ (0, 3) at which the combined vector
Δ = α_G·g + α_R·r̂_e outperforms residual-only (α_G=0) on target-emotion match?

---

## Design

| Variable | Value |
|---|---|
| Primary layer | 13 |
| α_R | 5.0 (fixed) |
| α_G values | 0.0, 0.25, 0.5, 1.0, 2.0, 3.0 |
| Source texts | first N_SOURCES test-split items |
| Target emotions | all 8 |
| Generation | greedy decode, max_new_tokens=60, last-token hook |
| Evaluator | GPT-4o-mini, same rubric as main experiment |
| Total (N=50) | 50 × 8 × 6 = 2,400 generations |
| Total (N=376) | 376 × 8 × 6 = 18,048 generations |

α_G = 0.0 serves as the **anchor** (= residual-only in the main experiment).  
α_G = 3.0 replicates the main experiment's decomposed condition.

**Note on delta construction:**  
Residual delta: `ALPHA_R * r_raw[e]` — natural norm scaling (not re-normalised),
matching the main experiment exactly.  
Shared delta: `alpha_g * unit(g)` — unit-normalised g scaled by α_G.

---

## How to Run

```bash
cd /path/to/EmotionEngine_v2

# Step 1 — Generate (GPU required, ~15 min for N=50)
python local_axes_experiments/exp12_alpha_g_sweep/exp12_alpha_g_sweep.py
# or full set:
python local_axes_experiments/exp12_alpha_g_sweep/exp12_alpha_g_sweep.py --n_sources 376

# Step 2 — Submit eval batch to OpenAI
python local_axes_experiments/exp12_alpha_g_sweep/exp12_alpha_g_sweep.py --submit

# Step 3 — After batch completes, download raw results then merge
python local_axes_experiments/exp12_alpha_g_sweep/exp12_alpha_g_sweep.py \
    --merge results/batch_raw.jsonl

# Step 4 — Analyse
python local_axes_experiments/exp12_alpha_g_sweep/exp12_alpha_g_sweep.py --analyse
```

---

## Expected Outputs

| File | Description |
|---|---|
| `results/generations.jsonl` | All generated texts with (source_id, emotion, alpha_g) |
| `results/eval_requests.jsonl` | OpenAI Batch API request file |
| `results/eval_batch_id.txt` | Batch ID after submit |
| `results/eval_results.jsonl` | Merged judge scores (after --merge) |
| `results/alpha_g_sweep_summary.csv` | Per-α_G aggregate stats (after --analyse) |

---

## Analysis Plan

1. **Aggregate by α_G** — plot target-match mean ± 95% CI vs α_G (line plot).
   Look for a U-shaped or monotonically decreasing curve.
   - If the curve is monotone (best at α_G=0) → g adds no value at any tested scale
   - If there is an interior maximum → there is an optimal α_G > 0

2. **Paired t-test vs anchor (α_G=0)** — for each α_G, test whether the paired
   difference from α_G=0 is significant.

3. **Per-emotion breakdown** — check whether any specific emotion benefits from
   a small α_G (g might help low-baseline emotions differently).

4. **Emotionality vs α_G** — check if small α_G recovers emotionality without
   hurting target-match (in the main exp, g at α_G=3 lowered emotionality from
   0.649 to 0.623).

---

## Results

*(To be filled after running)*

---

## Interpretation

*(To be filled after running)*
