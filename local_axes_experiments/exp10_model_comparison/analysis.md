# Experiment 10 Analysis: Model A vs B vs C Comparison

## 概要

**スクリプト**: `exp10_model_comparison.py`
**対象**: Layer 13 / Llama-3.1-8B-Instruct
**シード文**: SEED_TEXTS（8感情 × 3文 = 24文）× 3 条件 = 72 行
**Judge**: GPT-5.4（6軸評価）

### 3条件の定義

| 条件 | Δ の式 | 意図 |
|------|--------|------|
| **A** | `α_G g + α_R r_e` | Named emotion prototype のみ（旧来の decomposed steering）|
| **B** | `α_G g + Σ_k β_k m_k` where β_k = α_R (r_e · m_k) | r_e を m_k 基底で近似（label-free）|
| **C** | `α_G g + α_R r_e + Σ_k β_k m_k` | r_e と m_k 両方（hybrid full model） |

**条件 B の設計原理**: β_k = α_R × (r_e · m_k) により、B が targeting する方向は A と（m_k subspace 内で）同じ。条件 A と B の差異は **m_k では捉えられない r_e の成分（ε_e）を含むかどうか**のみ。

---

## 主要結果

### 全体スコア比較

| 指標 | A | B | C | B − A | C − A |
|------|---|---|---|-------|-------|
| target\_emotion\_match | 0.936 | **0.938** | 0.935 | +0.002 | -0.001 |
| meaning\_preservation | 0.881 | **0.902** | 0.879 | **+0.021** | -0.002 |
| fluency | 0.820 | **0.891** | 0.793 | **+0.071** | **-0.027** |
| style\_contamination | 0.699 | **0.735** | 0.690 | **+0.036** | -0.009 |
| subtle\_affective\_state | 0.712 | **0.723** | 0.686 | +0.011 | -0.026 |
| label\_free\_nuance | 0.678 | **0.697** | 0.650 | **+0.019** | -0.028 |

---

## 4つの核心的問いへの回答

### Q1: B は A と同等の target_emotion_match を達成できるか？

**→ Yes。B は A とほぼ同一（0.938 vs 0.936）。**

条件 B は名前付き感情の方向 r_e を使わずに、m_k への投影だけで感情カテゴリを再現できている。これは、感情の prototype 方向の約 22%（exp7 の再構築 EVR）が m_k subspace 内にあることで十分に感情の "flavor" を引き出せることを示す。

### Q2: B は A より subtle_affective_state または label_free_nuance が高いか？

**→ 僅かに Yes。B > A（+0.011 と +0.019）。**

差は小さいが、方向は一貫している。感情横断 arousal 軸の m_k 分解を通すことで、微妙な感情テクスチャが若干向上している。特に **joy (+0.08), fear (+0.09)** で改善が顕著。

### Q3: C は B より良いか？（r_e は m_k 分解に追加価値をもたらすか）

**→ No。C < B（すべての指標で B が上回る）。**

| 指標 | C − B |
|------|-------|
| fluency | **-0.098** |
| style_contamination | **-0.045** |
| label_free_nuance | -0.047 |
| subtle_affective_state | -0.037 |

C は r_e（m_k subspace 外の成分 ε_e）を加えることで、むしろ品質が下がる。これは **ε_e の追加がノイズとして働いている**可能性を示す。r_e の ε_e 成分（m_k と直交する 78% の部分）は感情の nuance を高めるどころか、出力の fluency と naturalness を下げる。

### Q4: C は A より良いか？（m_k は r_e の補完になるか）

**→ No。C ≤ A（全指標で A が C と同等か上回る）。**

特に fluency（A: 0.820, C: 0.793, -0.027）と label_free_nuance（A: 0.678, C: 0.650, -0.028）で A が C を上回る。

**まとめ**:
> 条件 B（label-free m_k 分解）が最良の結果をもたらす。r_e を加えてもm_kで得た品質は向上せず、むしろ低下する。

---

## 感情別分析

### B − A の差（感情別）

| 感情 | fluency | style\_contam | subtle\_affect | label\_free |
|------|---------|--------------|----------------|-------------|
| joy | **+0.257** | **+0.183** | +0.073 | +0.080 |
| fear | **+0.357** | **+0.147** | +0.080 | +0.087 |
| sadness | +0.030 | +0.067 | +0.007 | +0.020 |
| disgust | +0.050 | +0.053 | +0.043 | +0.053 |
| anger | +0.003 | +0.003 | -0.040 | -0.037 |
| trust | +0.053 | +0.067 | -0.087 | **-0.100** |
| anticipation | -0.003 | -0.020 | +0.027 | +0.040 |
| **surprise** | **-0.180** | **-0.213** | -0.020 | +0.007 |

**B が大きく改善する感情**: joy, fear
- joy: fluency +0.257（condition A の joy は fluency = 0.483 が低かった。B では 0.740 に改善）
- fear: fluency +0.357（A: 0.583 → B: 0.940）

**B が悪化する感情**: surprise, trust
- surprise: fluency -0.180, style_contamination -0.213（A: 0.920 → B: 0.740）
- trust: label_free_nuance -0.100

joy と fear で大きく改善した理由：これらの感情の m_k 分解 β_k 値が大きい（特に joy: m1=+0.24, m5=+0.40; fear: m1=-0.66）。r_e だけより m_k 分解の方が生成が smooth になっている可能性がある。

surprise で悪化した理由：surprise の r_e → m_k 分解は m1 や m2 への投影が大きい（m2=+0.55, exp7 で surprise は m2 への整合性が最高 0.083）。しかし surprise の内部構造が他の感情と異なるため（exp2 で PR_excess が最大だった）、m_k 分解が surprise 固有の成分を十分に捉えられていない。

---

## 質的分析：サンプル rewrite 比較

### sadness（seed_sadness_1）

> **ベース文**: "The old photographs reminded me of everything I had lost and would never get back."

| 条件 | Rewrite | label_free | subtle |
|------|---------|------------|--------|
| A | *"The old photographs stood as a haunting reminder of all that I'd left behind, a bittersweet nostalgia that lingered, a poignant echo of memories I could never reclaim."* | 0.90 | 0.92 |
| B | *"The old photographs stood as a haunting reminder of all that I'd left behind, a bittersweet nostalgia that only served to intensify the ache of what could never be regained."* | **0.92** | 0.90 |
| C | （A と同一の出力）| 0.90 | 0.92 |

sadness では A, B, C が非常に類似した出力を生成した。B の「intensify the ache」は A の「lingered, poignant echo」より若干より active な悲しみを表現している。条件 C が A と全く同じ出力になったことは、r_e + m_k 投影が A の r_e のみと実質的に同じ表現領域に収束したことを示す。

### joy（seed_joy_3）

> **ベース文**: "We spent the evening laughing and dancing and it felt like nothing else mattered."

| 条件 | Rewrite（抜粋）| label_free |
|------|--------------|------------|
| A | *"In that magical moment, time stood still, and the world melted away, leaving only the joy of being present."* | 0.74 |
| B | *"In that magical moment, time stood still as the world melted away, and all that was left was the warmth of our joy, the rhythm of our laughter, and the beat of our hearts, beating as one."* | 0.74 |
| C | *"...as we lost ourselves in the joy of the night, surrounded by the people and music that mad[e]..."*（truncated）| 0.68 |

B は A より社会的結合感と具体的感覚描写を加えた豊かな表現になっているが、judge の label_free_nuance スコアは同じ (0.74)。C はまた truncated で品質が低下。

---

## β_k 値の解釈

条件 B/C で使用される β_k = α_R × (r_e · m_k) の実際の値：

| 感情 | β_m1 | β_m2 | β_m3 | β_m4 | β_m5 |
|------|------|------|------|------|------|
| joy | +0.240 | -0.391 | -0.110 | +0.234 | +0.404 |
| sadness | +0.070 | +0.174 | -0.089 | **+0.429** | -0.227 |
| anger | -0.034 | -0.008 | +0.035 | -0.060 | **-0.242** |
| fear | **-0.662** | +0.090 | +0.243 | -0.109 | -0.153 |
| surprise | +0.336 | **+0.830** | -0.384 | -0.125 | -0.065 |

**注目点**：
- fear は m1 への投影が -0.662 と非常に大きく負（= 条件 B では m1 の low 方向に大きく steers）。Exp8 の解釈では m1 low = calm/detached。joy と反対方向。
- sadness は m4 への投影が最大（+0.429）。Sadness の特徴は m4 の高い係数で表現されている。
- surprise は m2 への投影が最大（+0.830）。Surprise の独自性は m2 方向に集中。

これらの β_k パターンは、**各感情が m_k 空間上の異なる点として分布している**ことを示す。条件 B はこのマッピングを通じて感情の特徴を再現している。

---

## 論文への含意

### 主要主張の支持

> **「named emotion label ではなく、label-free な affective dimension の組み合わせとして感情を操作する方が、品質が高い（少なくとも同等）」**

Q1–Q4 の結果は以下の主張を支持する：

1. m_k 分解だけで感情の prototype を再現できる（Q1）
2. m_k 分解は微妙な感情テクスチャを僅かに高める（Q2）
3. r_e は m_k 分解への追加価値をほとんどもたらさない（Q3, Q4）

特に **fluency（B: 0.891 vs A: 0.820）** の改善は実用的に重要：label-free 操作の方が自然な文を生成する。

### ε_e の役割について

条件 C（= A + B の m_k 投影）が A を下回ることは、r_e の ε_e 成分（m_k と直交する 78%）が affective nuance を高めず、むしろノイズとして働くことを示唆する。これは exp7 の「r_e の 78% が m_k で説明できない」という発見と整合的：

> ε_e は名前付き感情に固有の情報を含むが、そのほとんどは nuanced 出力に貢献しない冗長な成分かもしれない。

ただし、これは今回の小規模（24文）評価での暫定的な結論であり、特定の感情カテゴリや文脈では ε_e が有効な場合もある（anger/trust では C と A が近似）。

---

## 制限と留意点

1. **N=24 の小規模サンプル**：感情ごとに 3 文しかなく、統計的有意性が低い。特に感情別の結論は tentative。

2. **条件 B の β_k はすべての感情で小さい**：再構築 EVR が平均 22% であるため、B の実効的な steering は A よりも弱い。fluency が改善しているのは「弱い steering = 自然な文」という効果かもしれない。

3. **m4, m5 の信頼性**: exp7 で method agreement が低かった m4, m5 を含む β_k は信頼性が低い。m1–m3 だけに絞った条件 B' を比較するのが望ましい。

4. **同一シード文の繰り返し問題**：A, B, C が似た出力を生成する場合（sadness の例参照）、judge が3条件に同スコアをつける傾向がある。真の差異は標準的条件での大規模評価でのみ見えてくる可能性がある。
