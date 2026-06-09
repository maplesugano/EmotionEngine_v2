# CAA Decomposition Steering — Analysis

**Experiment:** Decomposed CAA steering evaluation on n=376 test-split source texts  
**Primary layer:** 13  
**Coefficients (default):** α_G = 3.0, α_R = 5.0  
**Evaluator:** GPT-4o-mini-2024-07-18 (LLM judge, 0–1 scale)  
**Date completed:** 2026-06-09

---

## 1. Research Question

Does decomposing a pooled CAA emotion vector into a **shared emotionalisation component g** and an **emotion-specific residual r̂_e** improve steered generation quality compared to using the monolithic pooled vector?

The tested parameterisation is:

$$\Delta_e(\alpha_G, \alpha_R) = \alpha_G \mathbf{g} + \alpha_R \hat{\mathbf{r}}_e$$

---

## 2. Method

### Decomposition

At layer 13, the pooled unit-normalised CAA direction for each emotion is `C[e]`.  
The shared direction is the mean: `g = unit(mean(C, axis=0))`.  
The emotion-specific residual is: `r_raw[e] = C[e] - (C[e]·g)·g` (projection removed).  
`r̂_e = unit(r_raw[e])`.

### Steering Hook

A forward hook adds a delta vector to the residual stream at the **last prompt token** of layer 13, before generation begins (greedy decode, `max_new_tokens=60`).

### Conditions (5)

| Condition | Delta applied |
|---|---|
| `none` | No intervention |
| `g` | α_G·unit(g), α_G = 3.0 |
| `resid` | α_R · unit(r_raw[e]) scaled by α_R · ‖r_raw[e]‖, α_R = 5.0 |
| `original_caa` | α_G · C[e], α_G = 3.0 (monolithic pooled unit vector) |
| `g+resid` | both g and resid combined |

**Note on resid scaling:** the residual delta is `α_R * r_raw[e]` (natural norm, not re-normalised), so its magnitude in activation space scales with the residual's natural size.

### Data

- Source texts: 376 test-split items from `dataset/emotion_rewrites/emotion_rewrites.jsonl`
- Target emotions: all 8 (joy, trust, fear, surprise, sadness, disgust, anger, anticipation)
- Total rows: 376 × 8 × 5 = **15,040 generations**
- Eval batch: 376 × 8 × 5 = **15,040 judge calls** (OpenAI Batch API)

### Evaluation Metrics (0–1 scale, per-sample)

- **target_emotion_match** — does the output express the target emotion?
- **meaning_preserved** — is the source propositional content retained?
- **emotionality** — is the output emotionally expressive at all?

### Key Files

| File | Description |
|---|---|
| `analysis/caa/steering_decomposition/extend_steering_to_n376.py` | Generation + eval-batch script |
| `analysis/caa/steering_decomposition/generation_outputs.jsonl` | All generations (incl. 800 legacy non-test rows at head — skip them) |
| `analysis/caa/steering_decomposition/generation_eval_results.jsonl` | Judge scores indexed by `custom_id` |
| `notebook/caa_analysis/thesis_artifacts_generation.ipynb` | Figure/table generation (uses custom_id alignment) |

**Alignment note:** `generation_outputs.jsonl` has 800 legacy (old n=20) non-test rows at the front. The eval batch was built from test-split rows only, indexed by `custom_id = "exp1_{i:06d}"`. Always join scores by `custom_id`, not by positional `zip`.

---

## 3. Results

### 3.1 Marginal Statistics (n = 3008 per condition)

Each condition is evaluated on 376 source texts × 8 target emotions = **3008 matched pairs**.  
CI = 95% from t-distribution with n−1 df.

| Condition | Target match | Semantic pres. | Emotionality |
|---|---|---|---|
| No steering | 0.339 ± 0.012 | 0.951 ± 0.003 | **0.649 ± 0.006** |
| Shared-only (α_G·g) | 0.328 ± 0.012 | 0.954 ± 0.003 | 0.623 ± 0.006 |
| **Residual-only (α_R·r̂_e)** | **0.352 ± 0.012** | 0.954 ± 0.003 | **0.650 ± 0.006** |
| Original CAA (α·c_e^pool) | 0.332 ± 0.012 | 0.955 ± 0.003 | 0.623 ± 0.006 |
| Decomposed (g + r̂_e) | 0.337 ± 0.012 | **0.957 ± 0.003** | 0.623 ± 0.006 |

The marginal CIs overlap because baseline alignment varies enormously across emotions (disgust=0.131, trust=0.648), which inflates between-item variance. The correct test is paired.

### 3.2 Paired Tests (target-emotion match)

All five conditions are run on the **same** (source_id, target_emotion) pairs, so a paired t-test differences out the large between-emotion baseline variance. This is the primary inferential analysis.

| Comparison | Δ (95% CI) | t(3007) | p | Emotions improved |
|---|---|---|---|---|
| Residual-only − No steering | +0.014 ± 0.004 | +6.42 | 1.4e-10 | **8/8** |
| Residual-only − Original CAA | +0.020 ± 0.005 | +7.26 | 3.7e-13 | **8/8** |
| Decomposed − Original CAA | +0.004 ± 0.004 | +2.31 | 0.021 | — |
| Decomposed − No steering | −0.002 ± 0.006 | −0.79 | 0.43 | — (n.s.) |
| Original CAA − No steering | −0.006 ± 0.005 | −2.32 | 0.020 | — (harmful) |
| Shared-only − No steering | −0.011 ± 0.005 | −3.98 | 7.0e-05 | — (harmful) |

### 3.3 Per-Emotion Breakdown (residual-only)

| Emotion | None | g | Resid | Orig. CAA | Decomposed | Δ vs none | Δ vs CAA |
|---|---|---|---|---|---|---|---|
| joy | 0.419 | 0.397 | **0.431** | 0.408 | 0.419 | +0.011 | +0.022 |
| trust | 0.648 | 0.648 | **0.664** | 0.651 | 0.647 | +0.016 | +0.013 |
| fear | 0.181 | 0.164 | **0.194** | 0.168 | 0.173 | +0.013 | +0.026 |
| surprise | 0.325 | 0.307 | **0.336** | 0.314 | 0.318 | +0.011 | +0.022 |
| sadness | 0.249 | 0.243 | **0.254** | 0.246 | 0.246 | +0.005 | +0.008 |
| disgust | 0.131 | 0.124 | **0.147** | 0.128 | 0.130 | +0.016 | +0.019 |
| anger | 0.157 | 0.140 | **0.186** | 0.147 | 0.163 | **+0.029** | **+0.039** |
| anticipation | 0.599 | 0.599 | **0.608** | 0.598 | 0.598 | +0.009 | +0.010 |

Residual-only is best for every emotion. Largest absolute gains for low-baseline emotions (anger, disgust, fear). Shared-only is below or at baseline for every emotion.

---

## 4. Interpretation

### What the decomposition achieves

The pooled CAA vector `c_e^pool` conflates two things:
1. A **shared emotionalisation direction g** — present in all eight emotion vectors, non-discriminative
2. An **emotion-specific residual r̂_e** — the part that distinguishes one emotion from another

The monolithic vector applies both at once and cannot separate their contributions. The decomposition makes them independently testable.

### What the data shows

- **g at α_G = 3 is counterproductive**: every condition containing g at this coefficient (shared-only, original CAA, decomposed) shows lower emotionality (0.623) than conditions without it (baseline/residual-only: 0.649–0.650), and both shared-only and original CAA are significantly *worse* than baseline for target-match. This does not imply g is useless at all scales — only that α_G = 3 is too large, and the effective region of the formula may be **α_G ≪ 3, α_R > 0**. Whether a small α_G (e.g., 0.25–1.0) recovers or adds value remains untested (→ exp12).

- **r̂_e is the signal**: residual-only is the only condition that reliably and significantly improves target-match — both over baseline (p=1.4e-10) and over the standard pooled CAA (p=3.7e-13), consistently across all 8/8 emotions.

- **The combined vector's mediocrity is explained by g**: when you add g back (decomposed = g+resid), g's negative contribution partially cancels the residual's gain, landing the combined vector back at the baseline level.

- **This was only discoverable through decomposition**: without separating g and r̂_e into independently-steerable components, you couldn't distinguish the residual's consistent positive effect from the shared direction's neutral/harmful contribution.

### Relation to the formula Δ_e(α_G, α_R) = α_G·g + α_R·r̂_e

The formula is supported as a **framework**: it defines two separable components and provides independent coefficients for each, which is what made residual-only steering testable. The empirical finding is that the effective setting is α_G ≈ 0, not α_G = 3 as originally guessed. This is a refinement of the framework, not a refutation of it.

### Pilot (n=20) vs full experiment (n=376)

At n=20 per cell, the combined decomposed vector appeared to give the largest gain, and confidence intervals were wide enough (~±0.05) that the ordering looked meaningful. At n=376 (~19× larger), the marginal CIs narrow to ±0.012 and the ranking reshuffles: residual-only is best, decomposed matches baseline. The paired test was the correct analysis throughout — it would have shown the same result even at n=20, but was not applied in the pilot.

---

## 5. Seed Robustness (α_G=3, α_R=3, last_token)

From `analysis/caa/seed_emotion_robustness/seed_robustness_generations.jsonl` (8 seeds per emotion):

- **No empty or [OOM] outputs** for any target emotion
- Output length standard deviation ranges from ~5 words (joy, anger) to ~10 words (fear, disgust, trust), indicating the seed context still substantially shapes generation even with fixed coefficients

---

## 6. Limitations

1. **Effect sizes are small** — paired Δ ≈ 0.014 above baseline, despite high statistical significance. Steering moves the distribution, not individual outputs reliably toward the target.
2. **Monotonicity is partial** — proxy sweep (output length/repetition vs α_R) shows broadly consistent directional trends but not strict monotonicity across all emotions.
3. **α_G not fully swept** — only α_G ∈ {0, 3} were compared. Whether a small α_G (0.25–1.0) adds a complementary signal or merely dilutes r̂_e is unknown (→ exp12).
4. **Baseline alignment is heterogeneous** — emotions like trust/anticipation are already well-matched without steering (0.6+), leaving little room for improvement; low-baseline emotions (disgust, anger) show the most benefit.

---

## 7. Figures Generated

| File | Description |
|---|---|
| `thesis/figures/steering_condition_comparison.pdf` | Bar chart: 3 metrics × 5 conditions (marginal, n=3008) |
| `thesis/figures/steering_per_emotion_target_match.pdf` | Per-emotion target-match, 5 conditions (n=376/cell) |
| `thesis/figures/steering_tradeoff_scatter.pdf` | Scatter: semantic preservation vs target-match |
| `thesis/figures/steering_paired_forest.pdf` | Forest plot: paired Δ with CI for 5 comparisons |
| `thesis/figures/steering_per_emotion_paired_delta.pdf` | Per-emotion paired Δ (resid vs none, resid vs CAA) |
| `thesis/figures/steering_coefficient_sweep_proxy.pdf` | Proxy monotonicity: output length & repetition vs α_R |
| `thesis/figures/steering_condition_table_ci.tex` | LaTeX table: marginal means ± CI |
