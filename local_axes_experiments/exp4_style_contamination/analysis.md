# Exp4 Layer13 Analysis (Style Contamination Check)

## Scope
- Experiment: local sub-axis validation (Methods A/B/C)
- Layer: 13
- Inputs:
  - method_a_style_r2.csv
  - method_b_neutral_cos.csv
  - method_c_filter_cos.csv
  - summary.json

## File-level sanity checks
- Number of (emotion, PC) entries: 40 (8 emotions x 5 PCs)
- Method C filtered sample size: 2775 / 22782 (~12.2%)

## Main results
- Verdict counts from summary.json:
  - likely_affective: 34
  - ambiguous: 6
  - likely_style: 0

- Mean EVR by PC:
  - PC1: 0.1292
  - PC2: 0.0494
  - PC3: 0.0343
  - PC4: 0.0282
  - PC5: 0.0239

Interpretation:
- Most explainable variance is in PC1, then decays smoothly.
- This pattern is consistent with a dominant affective direction plus weaker sub-axes.

## Method-wise interpretation

### Method A (style/semantic probe)
- Mean R2 over all 40 axes: 0.0690
- Axes above threshold (R2 > 0.15): 5 / 40

Interpretation:
- Most local axes are not strongly explained by the style/topic proxy feature set.
- A small subset shows non-trivial explainability by style proxies.

### Method B (neutral-control PCA alignment)
- Mean max cosine to neutral PCs: 0.0340
- Maximum observed max cosine: 0.0614

Interpretation:
- Alignment with neutral paraphrase variation axes is uniformly low.
- This argues against a strong paraphrase-style contamination account.

### Method C (meaning-preservation filter robustness)
- Mean cosine between full vs filtered axes: 0.9444
- Minimum cosine: 0.4400
- Axes below robustness threshold (cos < 0.7): 1 / 40

Interpretation:
- Nearly all axes are stable after restricting to meaning-preserved rewrites.
- One notable unstable axis exists: surprise PC5.

## Ambiguous axes (from summary)
Top ambiguous rows by Method A R2:
1. anger PC4: R2=0.2431, cos_filtered=0.9736
2. surprise PC2: R2=0.2415, cos_filtered=0.9489
3. fear PC5: R2=0.2190, cos_filtered=0.9696
4. joy PC2: R2=0.1832, cos_filtered=0.9927
5. sadness PC4: R2=0.1574, cos_filtered=0.9679

Interpretation:
- Ambiguity is primarily driven by Method A (style feature explainability),
  while Methods B/C remain supportive of affective interpretation.

## Overall conclusion
- At layer 13, evidence overall favors the "pre-verbal affective sub-axis"
  interpretation for most local PCs.
- Residual risk remains for a small subset of axes where Method A indicates
  moderate style-topic explainability.
- No axis is classified as likely_style in the current output set.
