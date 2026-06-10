# Exp 12 — α_G Sweep Analysis

**Date**: 2026-06-10  
**Data**: 100 test-split sources × 8 emotions × 4 α_G values (0, 2, 4, 8) = **3,200 generations**, scored by GPT-4o-mini Batch API.  
**α_R fixed at 64** (best performer from main experiment); steering vector: Δ = α_G·g + α_R·r̂_e.

---

## 1. Research question

The main decomposition experiment (n=376) found that the shared direction **g** at α_G=3 is counterproductive: all conditions containing **g** lower emotionality and target-emotion match relative to the residual-only baseline. However, only two values were tested (α_G=0 and α_G=3). This experiment asks: **Is there a value of α_G in (0, 3) that improves over residual-only without sacrificing semantic quality?**

**Answer: No.** Every non-zero α_G reduces target-emotion match, meaning preservation, and emotionality, in a monotonically degrading pattern. The shared direction **g** is unconditionally harmful at all tested scales.

---

## 2. Aggregate metrics

| α_G | Target-emotion match | Meaning preserved | Emotionality | n |
|----:|--------------------:|------------------:|-------------:|--:|
| **0.0** (residual-only) | **0.471 ± 0.023** | **0.794 ± 0.012** | **0.625 ± 0.016** | 800 |
| 2.0 | 0.396 ± 0.023 | 0.784 ± 0.013 | 0.575 ± 0.017 | 800 |
| 4.0 | 0.314 ± 0.023 | 0.670 ± 0.021 | 0.477 ± 0.019 | 800 |
| 8.0 | 0.001 ± 0.001 | 0.004 ± 0.002 | 0.004 ± 0.003 | 800 |

- All three metrics degrade monotonically as α_G increases.
- At α_G=8, all metrics collapse to near zero — the model produces unintelligible or empty output (see Section 5).
- The drop from α_G=0 to α_G=2 is already substantial: −0.075 in target-emotion match, smaller in meaning preservation (−0.010), suggesting that **g** specifically disrupts emotion specificity before damaging fluency.

---

## 3. Statistical tests: paired comparison vs α_G=0

Paired t-tests on per-(source, emotion) target-emotion match scores:

| α_G | Δ target-emotion match | 95% CI | t | p |
|----:|----------------------:|-------:|--:|--:|
| 2.0 | −0.0751 | ±0.0191 | −7.68 | 1.55e-14 * |
| 4.0 | −0.1573 | ±0.0218 | −14.16 | < 1e-40 * |
| 8.0 | −0.4708 | ±0.0234 | −39.48 | ≈ 0 * |

All comparisons are highly significant (p ≪ 0.05). The g component causes a statistically unambiguous decrease at every tested value.

---

## 4. Per-emotion breakdown

Target-emotion match by α_G for each emotion:

| Emotion | α_G=0 | α_G=2 | α_G=4 | α_G=8 |
|---------|------:|------:|------:|------:|
| anger | 0.750 | 0.592 | 0.516 | 0.000 |
| anticipation | 0.627 | 0.595 | 0.538 | 0.003 |
| disgust | 0.400 | 0.303 | 0.195 | 0.000 |
| fear | 0.523 | 0.324 | 0.245 | 0.000 |
| **joy** | **0.605** | **0.625** | **0.546** | **0.002** |
| sadness | 0.300 | 0.279 | 0.234 | 0.000 |
| surprise | 0.228 | 0.174 | 0.028 | 0.000 |
| trust | 0.338 | 0.281 | 0.210 | 0.000 |

**Joy anomaly**: joy is the only emotion where α_G=2 slightly exceeds α_G=0 (0.625 vs 0.605, Δ=+0.020). This may reflect that **g** has a positive-valence component that incidentally reinforces the joy direction. However, joy also degrades sharply at α_G=4 and collapses at α_G=8, so this is not a stable operating point.

**Worst affected at α_G=2**: fear (Δ=−0.199), anger (Δ=−0.158), surprise (Δ=−0.054). Fear and anger are high-specificity negative emotions whose residual directions **r̂_e** may be most orthogonal to **g**, so adding **g** dilutes the signal most.

**Most g-robust at low scale**: anticipation (Δ=−0.032) and sadness (Δ=−0.021) show smaller drops at α_G=2, possibly because their CAA directions have larger projections onto **g** (less residual signal to dilute).

---

## 5. α_G=8 collapse

At α_G=8, all three metrics drop to ≈0 across all emotions. This is a generation failure, not a gradual degradation: the residual direction **r̂_e** (unnormalized, natural norm) has its effect overwhelmed and the model outputs incoherent or repetitive text. The result confirms that **g** is not merely less effective than **r̂_e** — it actively disrupts generation when applied at sufficient scale. Even at α_G=4, meaning preservation falls to 0.670 (vs 0.794 at α_G=0), indicating early-stage generation degradation before full collapse.

---

## 6. Answer to research question

> Is there a value of α_G ∈ (0, 3) at which Δ = α_G·g + α_R·r̂_e outperforms residual-only?

**No.** Even the smallest tested value, α_G=2, produces a statistically significant reduction in target-emotion match (Δ=−0.075, p=1.55e-14). There is no "sweet spot" — the shared direction **g** degrades performance at all scales. The original finding from the main experiment holds: **g should be excluded entirely from the steering vector** (α_G=0 is optimal).

The lone exception — joy at α_G=2 — is a +2% effect that collapses at α_G=4 and is not robust enough to motivate including **g**.

---

## 7. Implications

1. **CAA decomposition**: the residual decomposition Δ = α_R·r̂_e (with g removed) is the correct operating mode. The shared direction captures cross-emotion general arousal but not emotion-specific content, and adding it at any scale degrades specificity.
2. **g as a diagnostic only**: **g** may be useful as a probe of "general emotionality" but should not appear in the generation hook.
3. **Scale sensitivity**: the transition from degraded (α_G=4) to collapsed (α_G=8) is abrupt. This mirrors the exp14 finding that coherence degrades faster than emotionality on one side of the sweep.
4. **Joy direction**: the weak positive effect of **g** on joy warrants a follow-up — possibly joy's residual direction has a small norm and **g** provides supplementary signal. Testing with a per-emotion-normalized δ could clarify.

---

## 9. Confidence level assessment

### CI methodology

95% CI は **1.96 × σ / √n** (正規近似) で計算。各 α_G レベルは n = 800（100 sources × 8 emotions）。Section 2 の表の ±値はこの値。

### 発見ごとの信頼度

**高信頼（p ≪ 0.05, paired t-test で確認済み）**

- 「α_G=2 でも target-emotion match が有意に低下する (Δ=−0.075, p=1.55e-14)」→ n=800 の paired t-test で大きなマージンがある。CI: ±0.019。
- 「α_G=8 で全メトリクスが崩壊する (≈0)」→ 議論の余地なし。
- 「α_G が増加するにつれ全メトリクスが単調減少する」→ fear/anger/surprise で特に明確。

**中信頼（方向は確かだが定量的な値は揺れる）**

- 各感情の α_G=2 における低下幅の大小関係（fear が最大、anticipation が最小）→ 感情間の比較は n=100 per cell に基づくため、順位は信頼できるが差の絶対値には ±0.02〜0.03 程度の誤差がある。

**低信頼（CI内で説明可能）**

- **Joy anomaly (α_G=2 で α_G=0 を上回る: Δ=+0.020)**: n=100 での単一観測であり、CI幅 ±0.023 の範囲内。統計的に有意でない。この効果が本物かどうかは、より多いサンプルか異なる α_G 値（例: 0.5, 1.0）での再現実験なしには確認できない。

### Multiple comparison

8感情 × 3 α_G 条件の組み合わせを比較しており、Section 3 の paired t-test は個別検定。FWER 補正（Bonferroni: α=0.05/24≈0.002）を適用しても p=1.55e-14 は有意水準を大きく下回るため、主要な結論（「g は有害」）は変わらない。ただし joy anomaly については補正後に有意でないことが明確になる。

### Evaluator reliability

exp12 の評価スキーマ（旧: `target_emotion_match`, `emotionality`）は exp14 で改訂された（`target_emotion_intensity`, `soundness`）。exp12 の数値は旧スキーマに基づくため、**exp14 との直接的な数値比較はできない**（exp12 の α_G=0, α_R=64 の target-emotion match = 0.471 と exp14 の α_R=64 の TI = 0.549 は異なる尺度）。同一実験内での相対比較のみが有効。

---

## 8. Caveats

- **α_G sweep is coarse**: only four values tested (0, 2, 4, 8). The interval [0, 2] was not sampled. A finer grid (e.g., 0.25, 0.5, 1.0) would confirm whether the degradation from α_G=0 to α_G=2 is linear or has a threshold.
- **α_R=64 fixed throughout**: results may differ at lower α_R where the residual signal is weaker and **g** might matter more relatively.
- **Joy anomaly**: based on a single data point (α_G=2); should be replicated on the full test set before any conclusion is drawn.
- **Generation collapse at α_G=8** may be partly due to unnormalized **r̂_e** combined with a large **g** exceeding the model's activation range — not purely a semantic effect.
