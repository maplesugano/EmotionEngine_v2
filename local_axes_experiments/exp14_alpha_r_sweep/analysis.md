# Exp 14 — α_R Sweep Analysis

**Date**: 2026-06-10  
**Data**: 100 test-split sources × 8 emotions × 17 α_R values (−128 to +128, step 16) = **13,600 generations**, scored by GPT-4o-mini Batch API.  
**α_G fixed at 0.0** (pure emotion-specific residual steering, shared "general emotionality" component removed).

---

## 1. Aggregate metrics

| α_R | Soundness | Meaning preserved | Target-emotion intensity |
|----:|----------:|------------------:|------------------------:|
| −128 | 0.359 | 0.145 | 0.010 |
| −96 | 0.712 | 0.341 | 0.047 |
| −80 | 0.865 | 0.512 | 0.114 |
| **0** | **0.985** | **0.912** | **0.339** |
| 64 | 0.949 | 0.818 | 0.549 |
| 80 | 0.859 | 0.677 | 0.581 |
| 96 | 0.686 | 0.483 | 0.526 |
| 128 | 0.347 | 0.146 | 0.320 |

- Soundness is nearly symmetric around α_R = 0, with rapid degradation outside ±96.
- Meaning preservation erodes faster on the negative side than the positive side (e.g. α_R = −64: 0.655 vs α_R = +64: 0.818), consistent with negative steering pushing text away from coherent content.
- Target-emotion intensity peaks around α_R = 64–80 then **decreases** at higher values — over-steering causes incoherent outputs that no longer strongly express any recognizable emotion.

---

## 2. Coherence thresholds and midpoints

Coherence threshold: **soundness ≥ 0.70**.  
M_A = arithmetic midpoint of coherent range; M_B = α_R within coherent range with highest mean meaning_preserved.

| Emotion | L_e | R_e | Width | M_A | M_B | \|M_A−M_B\| | ti @ M_A | ti @ M_B |
|---------|----:|----:|------:|----:|----:|-----------:|--------:|--------:|
| anger | −112 | 80 | 192 | −16.0 | 48 | 64.0 | 0.122 | 0.570 |
| anticipation | −80 | 128 | 208 | 24.0 | 0 | 24.0 | 0.591 | 0.567 |
| disgust | −128 | 80 | 208 | −24.0 | 48 | 72.0 | 0.090 | 0.351 |
| fear | −112 | 96 | 208 | −8.0 | 32 | 40.0 | 0.172 | 0.280 |
| joy | −80 | 96 | 176 | 8.0 | 0 | 8.0 | 0.503 | 0.472 |
| sadness | −128 | 128 | 256 | 0.0 | 0 | 0.0 | 0.249 | 0.249 |
| surprise | −96 | 64 | 160 | −16.0 | 0 | 16.0 | 0.350 | 0.413 |
| trust | −64 | 96 | 160 | 16.0 | 0 | 16.0 | 0.546 | 0.536 |

### Key observations

**M_A vs M_B discrepancy**  
Large gaps between the two midpoint definitions appear for anger (Δ = 64) and disgust (Δ = 72). For these emotions, the arithmetic midpoint (M_A) falls at slightly negative α_R (−16 and −24), but the point of highest meaning preservation (M_B) is at α_R = +48. This indicates an asymmetric steering manifold: the "neutral" content-preserving point is displaced toward the positive steering direction, and the coherent range is shifted left relative to the text's semantic neutral.

Emotions with M_B = 0 (joy, surprise, trust, sadness, anticipation) suggest that α_R = 0 is already the most meaning-preserving point within the coherent range, and the range is approximately symmetric in content terms even when L_e and R_e are not equidistant.

**Coherent range width**  
- Widest: sadness (256, covers the entire sweep), anticipation/disgust/fear (208)
- Narrowest: trust and surprise (160)
- The widest ranges correspond to emotions with a strong "general distress/activation" component that overlaps with the shared g direction already removed, giving the residual r_e a milder effect on fluency.

---

## 3. Per-emotion steering effectiveness

### Optimal positive steering (highest ti within coherent range)

| Emotion | opt_pos α_R | ti | Soundness |
|---------|-----------:|---:|----------:|
| anger | 80 | 0.763 | 0.832 |
| sadness | 112 | 0.738 | 0.902 |
| joy | 64 | 0.709 | 0.950 |
| fear | 80 | 0.659 | 0.941 |
| anticipation | 80 | 0.662 | 0.915 |
| disgust | 80 | 0.640 | 0.833 |
| trust | 16 | 0.546 | 0.986 |
| surprise | 32 | 0.485 | 0.972 |

**Anger** achieves the highest emotion intensity (ti = 0.763) at opt_pos while remaining mostly coherent (sn = 0.832).  
**Trust** peaks at α_R = 16 — unusually early, with only marginal improvement over α_R = 0 — and then *decreases* (see below).  
**Surprise** has low peak intensity (0.485) even at its optimal point, making it the weakest of the eight directions.

### Optimal negative steering (lowest ti within coherent range)

| Emotion | opt_neg α_R | ti | Soundness |
|---------|-----------:|---:|----------:|
| anger | −96 | 0.006 | 0.925 |
| disgust | −128 | 0.000 | 0.721 |
| fear | −96 | 0.003 | 0.902 |
| sadness | −112 | 0.038 | 0.906 |
| surprise | −96 | 0.001 | 0.850 |
| joy | −80 | 0.130 | 0.712 |
| anticipation | −80 | 0.323 | 0.708 |
| trust | +96 | 0.181 | 0.880 |

The top-4 discrete emotions (anger, fear, disgust, surprise) can be nearly completely suppressed (ti ≈ 0) while maintaining acceptable fluency. Positive/social emotions (joy, anticipation, trust) are harder to suppress: even at the most negative coherent α_R, trust retains ti = 0.181 and anticipation ti = 0.323.

---

## 4. Trust anomaly

Trust shows an anomalous response curve. Unlike every other emotion, its TI *decreases* monotonically above α_R = 16:

| α_R | sn | mp | ti |
|----:|---:|---:|---:|
| 0 | 0.982 | 0.949 | 0.536 |
| 16 | 0.986 | 0.938 | **0.546** ← peak |
| 32 | 0.987 | 0.900 | 0.521 |
| 64 | 0.973 | 0.619 | 0.413 |
| 80 | 0.963 | 0.414 | 0.302 |
| 96 | 0.880 | 0.213 | 0.181 |

Furthermore, opt_neg_r = +96 (positive!), meaning the point of lowest emotion intensity within the coherent range is at a *positive* α_R. This means steering "toward trust" counterintuitively *reduces* trust expression.

**Possible explanations**:
1. The trust residual direction captures a surface-level stylistic feature (e.g. formal register, hedging language) that the evaluator does not interpret as "trust emotion".
2. The source texts may already be high in trust-adjacent language (ti = 0.536 at α_R = 0 is the highest baseline of all 8 emotions). Adding more trust-direction signal overshoots and produces unnatural, sycophantic text.
3. Trust is the most semantically diffuse of Plutchik's basic emotions and may not have a stable residual direction in this model's activation space.

---

## 5. Dominant emotion recognition at α_R = 64

| Emotion | Correct (%) | Top confusion |
|---------|------------:|---------------|
| anger | 90% | neutral (5%), sadness (3%) |
| joy | 86% | neutral (11%), trust (2%) |
| fear | 78% | neutral (19%), anger (1%) |
| anticipation | 66% | neutral (17%), trust (7%) |
| trust | 59% | neutral (39%), anticipation (1%) |
| disgust | 45% | neutral (47%), anger (4%) |
| sadness | 46% | neutral (49%), trust (2%) |
| surprise | 16% | neutral (40%), joy (23%) |

Anger and joy have the strongest, most recognizable steering signals. Disgust, sadness, and especially surprise are frequently classified as neutral even at α_R = 64, indicating that these directions produce stylistic shifts that are not perceived as strongly emotional by the evaluator.

---

## 6. Baseline emotionality (α_R = 0)

| Emotion | ti @ 0 | mp @ 0 |
|---------|-------:|-------:|
| anticipation | 0.567 | 0.958 |
| trust | 0.536 | 0.949 |
| surprise | 0.413 | 0.928 |
| joy | 0.472 | 0.931 |
| sadness | 0.249 | 0.906 |
| fear | 0.177 | 0.886 |
| anger | 0.181 | 0.861 |
| disgust | 0.120 | 0.879 |

The test corpus has high baseline "positive/calm" emotionality: anticipation, trust, and joy show ti > 0.45 even at α_R = 0. Negative discrete emotions (anger, fear, disgust) start very low. This asymmetry partially explains why negative steering for positive emotions has less headroom and why their M_B tends to coincide with α_R = 0.

---

## 7. Recommended operating points

Based on the sweep, recommended α_R values per emotion (balancing ti and soundness):

| Emotion | Recommended α_R | Rationale |
|---------|----------------:|-----------|
| anger | 64–80 | ti > 0.70, sn ≥ 0.83 |
| sadness | 96–112 | strong ti, sn ≥ 0.90 |
| joy | 48–64 | ti ≈ 0.65–0.71, sn ≥ 0.95 |
| fear | 64–80 | ti ≈ 0.65, sn ≥ 0.94 |
| anticipation | 64–80 | ti ≈ 0.65, sn ≥ 0.91 |
| disgust | 64–80 | moderate ti (0.55–0.64), sn ≥ 0.83 |
| trust | 0–16 | anomalous curve; max effective at α_R = 16 |
| surprise | 32–48 | peak ti only ~0.49; direction is weak |

For the "neutral midpoint" (starting point for bidirectional interpolation): use **M_B** (argmax meaning_preserved within coherent range) rather than M_A for anger, disgust, and fear, where the discrepancy is large. For joy, sadness, surprise, trust, and anticipation, M_B = 0 ≈ M_A, so α_R = 0 is a reasonable neutral baseline.

---

## 9. Confidence level assessment

### CI methodology

95% CI は **1.96 × σ / √n** (正規近似) で計算。各(emotion, α_R) セルは n = 100。

代表的なCI幅:

| α_R | CI soundness | CI target-emotion intensity |
|----:|-------------:|----------------------------:|
| −128 | ±0.029 | ±0.004 |
| −96 | ±0.025 | ±0.009 |
| 0 | ±0.003 | ±0.020 |
| 64 | ±0.007 | ±0.018 |
| 96 | ±0.024 | ±0.022 |
| 128 | ±0.027 | ±0.022 |

中程度のα_R（±64付近）では intensityのCIが±0.02程度。極端な値（±128）では soundness のCIが±0.03まで広がる（破綻テキストのばらつきが大きいため）。

### 発見ごとの信頼度

**高信頼（CIが非重複）**

- α_R ≈ ±80 以上での coherence 急落 → 隣接グリッド点間の差がCIを大きく超える
- disgust/sadness/surprise の dominant emotion 認識率が低い（16〜46%）→ n=100 でこの水準なら誤差での説明は不可
- 全体のTIが α_R=0 から 64 にかけて単調増加し 80 でピーク → 方向性は明確

**中信頼（方向は確かだがpoint estimateは揺れる）**

- **L_e / R_e の正確な値**: グリッドは 16ステップ刻みなので、真の閾値は ±8 以内の誤差がある。例: "anger の L_e = −112" は厳密には "L_e ∈ (−128, −96]" と読むべき。
- **M_B (argmax meaning_preserved) の位置**: anger M_B=48 付近の実測値:

  | α_R | mp mean | 95% CI |
  |----:|--------:|-------:|
  | 32 | 0.936 | ±0.024 |
  | **48** | **0.951** | **±0.018** |
  | 64 | 0.935 | ±0.026 |

  48 がわずかに高いが、32/64 との差（Δ ≈ 0.015）はCI幅と同程度。M_B=48 という点推定より「M_B ∈ {32, 48}」と幅を持たせて解釈するのが適切。

**低信頼（CI内で説明可能）**

- **Trust anomaly のピーク位置**: α_R=0 と α_R=16 の ti の差は **0.010**（±0.030 と ±0.025 のCIが大きく重複）。「TI peak @ α_R=16」という主張は統計的に有意でない。言えるのは「trustは α_R ∈ [−16, +32] でほぼ平坦なプラトーを持つ」程度。
- **Surprise の opt_pos α_R=32**: α_R=32/48/64 の TI はそれぞれ 0.485/0.479/0.478 で、CIが ±0.035〜0.042 と広く、どの点が「最適」かは判別不能。

### Multiple comparison

8感情 × 17α_R値の組み合わせに対して個別の比較を行っており、Family-wise error rate (FWER) の補正をしていない。formal な統計検定を論文に載せる場合は Bonferroni または Holm 補正が必要。現時点の分析はすべて **exploratory (仮説生成的)** であり、confirmatory な検定とは区別すること。

### Evaluator reliability

GPT-4o-mini のスコア自体のノイズが未定量。同一テキストを複数回評価した場合の test-retest reliability（ICC: Intraclass Correlation Coefficient）を測定していないため、スコアの絶対値より **相対的なパターン（傾向・順序）** を信頼するべき。特に ti スコアは evaluator の主観的閾値に依存するため、0.5 という中点の意味が不明確である点に注意。

---

## 8. Caveats

- **α_G = 0 throughout**: the general emotionality component is not applied. Adding α_G > 0 may shift both coherence thresholds and effective emotion intensity.
- **Trust direction**: should be re-evaluated or re-estimated before using in downstream experiments.
- **Surprise direction**: low peak ti (0.49) suggests either a weak residual or a surface-level feature not well-captured by GPT-4o-mini's evaluator.
- **Soundness threshold = 0.70** is a hard cutoff; gradual degradation starts earlier (~α_R = ±64 for most emotions).
