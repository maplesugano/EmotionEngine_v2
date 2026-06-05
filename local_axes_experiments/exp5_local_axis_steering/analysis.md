# Exp 5: Local Axis Steering — Analysis (Layer 13)

## Overview

**Goal**: For each emotion *e* and its first local PC $u_{e,1}$, apply decomposed steering

$$\Delta = \alpha_G \cdot g + \alpha_R \cdot r_e \pm \beta \cdot u_{e,k}$$

and verify whether the model output shifts toward the expected affective pole.  
An LLM judge (GPT) rates each rewrite on 6 axes (0–1 each).

**Dataset**: 8 emotions × 24 seed texts × 2 poles = 384 judged rows.  
**Layer**: 13 (Llama-3.1-8B-Instruct).  
**Seed texts**: 24 curated texts, 3 per emotion.

---

## Steering Parameters

| Emotion | α_G | α_R | β |
|---|---|---|---|
| joy | 2.0 | 1.5 | 1.0 |
| trust | 2.0 | 1.5 | 1.0 |
| fear | 2.0 | 1.5 | 1.0 |
| anticipation | 2.0 | 1.5 | 1.0 |
| surprise | 2.0 | 1.5 | 0.75 |
| sadness | 1.5 | 1.0 | 0.5 |
| disgust | 1.5 | 1.0 | 0.3 |
| anger | 1.5 | 1.0 | 0.35 |

`anger` and `disgust` use reduced β (≈20–40% lower than initial default of 0.5) to reduce style artifacts while still achieving affective shift.

---

## Local Axis Identities (PC1)

| Emotion | Axis Name |
|---|---|
| joy | Resolved Contentment vs Ambivalent Struggle |
| trust | Optimistic Trust / Benevolent Expectation vs Confident Self-Reliance and Satisfaction |
| fear | Immediate Situational Worry vs Reflective General Concern |
| surprise | Unexpected Negative/Inconvenient Events vs Pleasant or Neutral Unexpected Events |
| sadness | Experiencing loss and disappointment vs Managing everyday life and minor annoyances |
| disgust | Grossed-out by sensory repulsion vs Disgust mixed with negative judgment about norms or traditions |
| anger | anger_pc1 (axis name not yet interpreted) |
| anticipation | hopeful curiosity and forward-looking excitement vs anxious waiting and guarded pessimism |

---

## Per-Emotion Judge Score Summary (n=48 per emotion, both poles)

| Emotion | n | local_axis_match | target_match | fluency | style_contamination |
|---|---|---|---|---|---|
| anger | 48 | **0.445** | 0.242 | 0.906 | 0.667 |
| anticipation | 48 | 0.305 | 0.411 | 0.844 | 0.631 |
| disgust | 48 | 0.163 | 0.213 | 0.883 | 0.654 |
| fear | 48 | 0.307 | 0.388 | 0.842 | 0.626 |
| joy | 48 | 0.279 | 0.405 | 0.790 | 0.591 |
| sadness | 48 | 0.258 | 0.340 | 0.884 | 0.630 |
| surprise | 48 | 0.261 | 0.280 | 0.762 | 0.566 |
| trust | 48 | 0.184 | 0.259 | 0.814 | 0.610 |

### Key Observations

- **anger** achieves the highest `local_axis_match` (0.445) despite using a reduced β of 0.35, suggesting the anger PC1 direction is both stable and well-separated.
- **disgust** is the weakest (0.163), likely because its PC1 axis captures a subtle sensory-vs-normative distinction that the model does not reliably track at this β level.
- **fluency** is generally high (0.76–0.91), indicating the steering does not degrade grammaticality much.
- **style_contamination** averages 0.59–0.67 across emotions — no longer saturated at 1.0 after the judge prompt was tightened. Scores spread across the 0.4–0.99 range.

---

## Pole Breakdown: Anger and Disgust

| Emotion | Pole | n | local_axis_match | style_contamination |
|---|---|---|---|---|
| anger | high | 24 | 0.289 | 0.641 |
| anger | low | 24 | **0.602** | 0.693 |
| disgust | high | 24 | 0.120 | 0.638 |
| disgust | low | 24 | 0.207 | 0.671 |

- For anger, the **low pole** (suppressed anger / calm) steers much more reliably (0.602) than the high pole (agitated anger, 0.289). This may reflect an asymmetry in how the PC1 direction interacts with the base model's generation tendencies.
- Disgust is uniformly low on both poles. The sensory-repulsion vs. normative-disgust distinction is subtle and may require a larger β or additional PC components.

---

## Style Contamination Distribution (Anger / Disgust)

After tightening the judge rubric (explicit 1.0/0.7/0.4/0.2/0.0 scale with truncation penalties):

| Emotion | n | mean | 0.0–0.39 | 0.4–0.69 | 0.7–0.99 | 1.0 |
|---|---|---|---|---|---|---|
| anger | 48 | 0.667 | 0 | 28 (58%) | 20 (42%) | 0 |
| disgust | 48 | 0.654 | 2 (4%) | 24 (50%) | 22 (46%) | 0 |

- **Zero rows at 1.0**: saturation is fully resolved.
- The dominant band is 0.4–0.69 ("clear style artifact / melodramatic boilerplate"), which is expected given the steering strength needed to induce detectable affective shift.

---

## Truncation Stability Check

From `results/truncation_regen_samples.jsonl` (26 originally-truncated rows × 3 regenerations = 78 total):

| Metric | Value |
|---|---|
| Total regenerated rows | 78 |
| Truncation-like on regen | 9 (11.5%) |
| Clean on regen | 69 (88.5%) |
| Per-source: 0/3 truncated | 12 sources |
| Per-source: 1/3 truncated | 7 sources |
| Per-source: 2/3 truncated | 1 source |
| Per-source: 3/3 truncated | 0 sources |

**Interpretation**: Truncation in the original run was largely stochastic — most seeds that truncated once produce clean text in 2–3 out of 3 regenerations. No source consistently truncates (0 at 3/3), suggesting the issue is sampling variance rather than a systematic model failure for those inputs.

---

## Files

| File | Description |
|---|---|
| `results/results.jsonl` | 384 judged rows (all emotions, both poles, layer 13) |
| `results/results.csv` | Same as above in CSV format |
| `results/truncation_regen_samples.jsonl` | 78-row truncation stability check |
| `results/judge_cache/` | Cached LLM judge responses (SHA1-keyed JSON) |

---

## Next Steps / Open Issues

1. **Disgust PC1 weakness**: Interpret the `disgust_pc1` axis more carefully (exp3 output) and consider increasing β or using PC2 for the sensory-repulsion pole.
2. **Anger pole asymmetry**: Investigate why the low pole steers so much better. Check if `anger_pc1` has a directional bias toward the calm/suppressed end in the activation space.
3. **Multi-PC steering**: Experiment with `--pcs 1 2` to combine PC1 and PC2 for emotions with low `local_axis_match`.
4. **Trust / surprise**: Both show flat `local_axis_match` (0.18–0.26). The trust axis (Optimistic Trust vs. Self-Reliance) may be too abstract for the judge to score at this β level.
