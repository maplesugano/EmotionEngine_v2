# Experiment 9 Analysis: Meta-Axis Steering (condition M) vs Emotion-Specific Steering (condition S)

## 概要

**スクリプト**: `exp9_meta_axis_steering.py`
**対象**: Layer 13 / Llama-3.1-8B-Instruct
**実行条件**: condition M のみ実行（condition S = exp5 の結果を再利用）
**シード文**: SEED_TEXTS（8感情 × 3テキスト = 24文）× 2 極 (high/low) = 48 行
**ステアリング式**:
- Condition S (exp5): `Δ = α_G g + α_R r_e + β u_{e,1}` （感情固有 local PC）
- Condition M (exp9): `Δ = α_G g + α_R r_e + β m_1` （共有 meta-axis）

---

## 方法上の注記：exp5 との比較について

Exp5 は全感情 × シード文 × 2 極 = 384 行を生成しており、その中に「感情が一致するシード文だけ」を含む 48 行のサブセットがある（例：seed_joy_1 → target=joy）。Exp9 はこの within-emotion 48 行と同一の条件で実施した。

以下の比較はすべて **within-emotion seed texts に絞った 48 行** で行う。

---

## 主要結果

### 全体スコア比較（48行, within-emotion seeds, PC1）

| 指標 | Condition S（u_{e,1}） | Condition M（m_1） | M − S |
|------|----------------------|-------------------|-------|
| target\_emotion\_match | **0.926** | 0.923 | -0.003 |
| local\_axis\_match | **0.525** | 0.519 | -0.006 |
| meaning\_preservation | 0.885 | **0.882** | -0.003 |
| fluency | **0.846** | 0.779 | **-0.067** |
| style\_contamination | **0.638** | 0.792 | **+0.154** |
| subtle\_affective\_state | **0.675** | 0.608 | **-0.067** |

*注: style_contamination は高いほど汚染が少ない（1.0 = clean）。*

---

### Pole 別比較（条件 M）

| 指標 | High pole | Low pole | 差 (high − low) |
|------|-----------|----------|-----------------|
| target\_emotion\_match | 0.924 | 0.922 | +0.002 |
| **local\_axis\_match** | **0.896** | **0.142** | **+0.754** |
| meaning\_preservation | 0.878 | 0.886 | -0.008 |
| fluency | 0.776 | 0.782 | -0.006 |
| style\_contamination | 0.776 | 0.808 | -0.032 |
| subtle\_affective\_state | **0.827** | **0.389** | **+0.438** |

**High pole と low pole の間に大きな非対称性がある**：
- `local_axis_match`: high = 0.896 vs low = 0.142（差 +0.754）
- `subtle_affective_state`: high = 0.827 vs low = 0.389

Condition S（exp5 within-emotion）でも：
- High: local_axis_match = 0.782, low: 0.269（差 +0.513）

m_1 の**high pole は非常に効果的（local_axis_match = 0.896）**だが、**low pole の steering はほとんど機能していない（0.142）**。

---

### Pole 非対称性の原因分析

m_1 の解釈（exp8）: **intense/immersed ↔ calm/detached**

High pole（immersed/intense 方向）はモデルが生成しやすい方向と一致していることが多い：
- "pure elation lifted my entire being" (joy high, local_axis_match=0.95)
- "heart into a frantic gallop" (fear high, local_axis_match=0.90)
- "immense satisfaction" (trust high, local_axis_match=0.87)

一方、Low pole（calm/detached 方向）は生成が難しい：
- 多くの low pole サンプルでもモデルが intensify した出力を生成している（truncated や反転した効果）
- fear low pole で3件中2件が local_axis_match = 0.25 以下（`+β m_1` でなく `-β m_1` で生成したが、モデルが high arousal direction に向かってしまった）

これは residual stream への直接介入が **arousal を上げる方向（positive beta）では機能するが、下げる方向（negative beta）では生成プロセスが抵抗する** ことを示唆する。

---

### 感情別スコア比較

| 感情 | S: local\_axis\_match | M: local\_axis\_match | S: subtle\_affect | M: subtle\_affect |
|------|----------------------|----------------------|-------------------|-------------------|
| anger | **0.767** | 0.522 | **0.835** | 0.680 |
| anticipation | **0.562** | 0.525 | **0.735** | 0.687 |
| disgust | 0.445 | **0.517** | 0.558 | **0.547** |
| fear | **0.527** | 0.503 | **0.690** | 0.510 |
| joy | 0.420 | **0.540** | 0.520 | **0.620** |
| sadness | 0.510 | **0.483** | **0.802** | 0.722 |
| surprise | **0.487** | 0.515 | 0.733 | **0.558** |
| trust | 0.485 | **0.550** | 0.528 | **0.540** |
| **平均** | **0.525** | 0.519 | **0.675** | 0.608 |

感情によって S > M または M > S のケースが混在しており、**どちらが一方的に優れているとは言えない**。

注目すべき点：
- **joy** では M が S を上回る（local_axis_match: +0.12、subtle_affect: +0.10）
- **anger** では S が M を大きく上回る（local_axis_match: -0.25）

---

### Fluency と Style Contamination の差異

最大の差異は **fluency**（S: 0.846, M: 0.779, 差 -0.067）と **style_contamination**（S: 0.638, M: 0.792, 差 +0.154）。

逆説的に：
- Condition M は fluency が **低い** が style_contamination が **高い**（より clean）
- Condition S は fluency が **高い** が style_contamination が **低い**（より汚染されている）

この理由は、fear の truncated 出力が多いこと（fluency < 0.3 が6件）。特に fear では fluency の平均が 0.310 まで低下しており、m_1 の high amplitude な fear steering が token generation を不安定にしている。

Truncated 出力のパターン：
```
"As I stepped into the old mansion, every faint creak of the floorboards ...
 sent my heart into a frantic gallop, and I couldn"  (fear high, fluency=0.28)
```

u_{e,1}（emotion-specific）では beta が既にチューニング済み（fear: 1.0→1.5 に調整なし）だが、m_1 は共有軸のためスケールが異なる可能性がある。

---

## 考察

### Condition M vs S: どちらが優れているか？

**数値だけで見ると S がわずかに優れている**（local_axis_match: +0.006, subtle_affective_state: +0.067）が、差は非常に小さく、感情ごとに逆転するケースが多い。

この結果は次の解釈を支持する：

> **共有 meta-axis m_1 は emotion-specific local PC u_{e,1} と同等の steering 効果を持つ。**

ただし、重要な注意点がある：

1. **Low pole が機能しない**: m_1 の low pole（calm/detached）は steering の主要な弱点。これは m_1 が arousal-up の方向に偏った one-sided axis である可能性を示唆する。

2. **感情依存性**: joy と disgust では M が S を上回り、anger と fear では S が M を上回る。M は joy/trust のような「高揚感情」に相性が良く、S は anger/fear のような「身体的緊張感情」に相性が良い傾向がある。

3. **Fluency の問題**: Condition M での高 beta 値が fear で truncation を引き起こしている。Beta スケールの再調整が必要かもしれない。

### EmotionEngine への示唆

Exp9 の結果から、u_{e,k}（emotion-specific）と m_k（shared）の間に質的な差異はほとんどない。これは：

> **EmotionEngine の制御軸として、emotion-specific な u_{e,k} を用いるか shared m_k を用いるかは、ほぼ同等の結果をもたらす。**

実装上は m_k を使う方が**単純**（感情ごとに PCA を計算する必要がない）という利点がある。ただし low pole の機能不全は解決が必要。

---

## 制限と留意点

1. **Condition S の比較がそのままではない**: exp5 と exp9 は生成パラメータが完全一致しているか確認が必要（judge model, temperature など）。

2. **Beta スケールの非整合**: m_1 は unit-normed だが exp5 の u_{e,1} も unit-normed なので、スケールは同じはず。ただし effective な activation への影響が異なる可能性がある。

3. **48 行という小規模サンプル**: 各感情 6 行（3テキスト × 2極）での評価は統計的ノイズが大きい。有意性の主張には慎重さが必要。
