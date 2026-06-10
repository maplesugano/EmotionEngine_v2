# Exp3: Local Axis Interpretation — Analysis

## 概要

Layer 13 の感情活性化から残差を構築し、感情ごとに PCA を適用して上位 5 軸を抽出。
各軸の極端例（top-5 / bottom-5）を LLM judge（GPT-4o, Batch API）に提示し、
軸の意味を解釈させた。

**残差化パイプライン:**
1. `delta = h_emo - h_neu`（感情方向の差分）
2. 共通感情方向 `g`（CAA pooled）を射影除去
3. 感情ごとの intensity 方向 `u_e`（CAA per-emotion）を射影除去
4. source ごとの感情間平均を減算

---

## EVR 上位軸（説明分散比）

| Emotion      | PC1 EVR |
|-------------|---------|
| joy         | 22.7%   |
| trust       | 15.8%   |
| sadness     | 13.6%   |
| disgust     | 13.0%   |
| anticipation| 10.8%   |
| anger       | 10.7%   |
| fear        |  9.1%   |
| surprise    |  7.6%   |

PC1 以降は急激に低下し（PC2: 4〜6%、PC3 以降: 2〜4%）、
感情ごとの内部構造は PC1 に強く集約されている。

---

## LLM Judge 結果（v3 — 全8感情）

### Quality 評価サマリー

| Emotion     | PC | EVR  | Axis Name                                                          | Contamination | Confidence |
|------------|-----|------|--------------------------------------------------------------------|--------------|------------|
| joy         |  1 | 22.7 | Reflective/Perspective-taking joy vs Immediate/External event joy  | medium       | 4          |
| joy         |  2 |  4.7 | Hopeful/Positive anticipation joy vs Anxious/Uncertain vigilance   | medium       | 4          |
| joy         |  3 |  3.0 | Social/Interpersonal joy vs Self-focused Contentment               | medium       | 4          |
| joy         |  4 |  2.4 | Resilient Reframing joy vs Risk/Concern-tinged joy                 | medium       | 4          |
| joy         |  5 |  1.9 | Achievement/Reward joy vs Disappointment-with-Silver-Lining        | medium       | 4          |
| trust       |  1 | 15.8 | Trust as hopeful acceptance vs Trust as self-assured confidence    | medium       | **5**      |
| trust       |  2 |  4.7 | Trust as pragmatic reassurance vs Trust as anticipatory concern    | medium       | 4          |
| trust       |  3 |  3.0 | Institutional/Normative Trust vs Distrust/Anxiety                  | medium       | 4          |
| trust       |  4 |  2.5 | Steady Reliance vs Mild Doubt and Disappointment                   | medium       | 4          |
| trust       |  5 |  2.2 | Forgiving Attribution of Misbehaviors vs Confident Expectation     | medium       | 4          |
| fear        |  1 |  9.1 | Immediate situational anxiety vs Chronic/existential worries       | **low**      | **5**      |
| fear        |  2 |  4.4 | Concern about personal faults vs General anticipatory uncertainty  | medium       | 4          |
| fear        |  3 |  3.8 | Uncertainty in Novel Situations vs Disappointment/Distrust         | medium       | 4          |
| fear        |  4 |  3.1 | Tangible Threats vs Mild Social/Anticipatory Unease                | medium       | 4          |
| fear        |  5 |  2.4 | Loss/Personal Vulnerability vs Procedural Uncertainty              | **high**     | 3          |
| surprise    |  1 |  7.6 | Unexpected Events vs Expected/Anticipated Events                   | **low**      | **5**      |
| surprise    |  2 |  5.8 | Interpersonal Dialogue & Social Interaction vs Personal Anticipation| medium       | 4          |
| surprise    |  3 |  4.4 | Retrospective Emotional Surprise vs Present/Future Surprise        | medium       | 4          |
| surprise    |  4 |  3.6 | Achievement/Expectation Exceeded vs Disappointment                 | medium       | 4          |
| surprise    |  5 |  2.8 | Personal Connection & Meaning vs Unexpected Life Changes           | medium       | 4          |
| sadness     |  1 | 13.6 | Loss & Disappointment vs Everyday Contentment                      | **low**      | **5**      |
| sadness     |  2 |  4.3 | Relief & Resolution vs Anxiety & Loneliness                        | medium       | 4          |
| sadness     |  3 |  3.2 | Personal Intimate Loss vs Reflective/Empathetic Melancholy         | medium       | 4          |
| sadness     |  4 |  2.6 | Unfulfilled Desires vs Social Exclusion/Relational Poignancy       | medium       | 3          |
| sadness     |  5 |  2.4 | Positive Events with Muted Feelings vs Genuine Warmth              | medium       | 4          |
| disgust     |  1 | 13.0 | Visceral/Gross Physical Repulsion vs Mild/Abstract Discomfort      | medium       | 4          |
| disgust     |  2 |  4.5 | Frustration/Annoyance with External Events vs Melancholic Discomfort| medium      | 4          |
| disgust     |  3 |  3.2 | Moral Wrongness/Harm vs Mild Annoyance                             | **low**      | **5**      |
| disgust     |  4 |  2.6 | Critical Social Judgment vs Personal Mild Annoyance                | **high**     | 3          |
| disgust     |  5 |  2.5 | Neglect/Carelessness Disgust vs Overindulgence Disgust             | medium       | 4          |
| anger       |  1 | 10.7 | Irritation/Frustration vs Disappointment/Resigned Upset            | **low**      | **5**      |
| anger       |  2 |  4.9 | Frustration with External Uncontrollable Events vs Internal Concern | medium      | 4          |
| anger       |  3 |  3.4 | Moral Indignation vs Pragmatic Frustration                         | medium       | 4          |
| anger       |  4 |  2.6 | Social Confrontation/Unfairness vs Personal Setbacks               | medium       | 4          |
| anger       |  5 |  2.3 | Resentment/Bitterness vs Mild Irritation                           | medium       | 4          |
| anticipation|  1 | 10.8 | Hopeful curiosity vs Uneasy waiting                                | **low**      | **5**      |
| anticipation|  2 |  6.1 | Excited forward-looking vs Reflective memory-driven                | medium       | 4          |
| anticipation|  3 |  3.4 | Hopeful Openness vs Anticipatory Coping with Regret                | medium       | 4          |
| anticipation|  4 |  3.2 | Social Connection vs Isolation/Vigilance                           | medium       | 4          |
| anticipation|  5 |  2.7 | Anticipation of Opportunities vs Anticipation of Conflict          | medium       | 4          |

---

## 主要な知見

### 1. 高信頼・低汚染の軸（contamination=low, confidence=5）

v3 で新たに **anger PC1** と **surprise PC1**、**sadness PC1** が追加された。

- **anger PC1** (EVR=10.7%): 「能動的苛立ち/フラストレーション vs 諦め/失望を伴う怒り」  
  外向きの即時的な怒りと内向きの落胆した怒りを明確に分離。初めて anger の高信頼軸が確認された。
- **surprise PC1** (EVR=7.6%): 「本当に予期しない出来事 vs 予測・期待された出来事」  
  驚きの中核次元である「期待違反の有無」を純粋に捉えている。
- **sadness PC1** (EVR=13.6%): 「喪失・失望 vs 日常の平穏」  
  v2 では medium だったが v3 で low に改善。急性の悲嘆と低強度の悲しみの対比。
- **fear PC1** (EVR=9.1%): 「即時・状況的不安 vs 慢性的・実存的不安」  
  恐怖の時間的特性（acute vs chronic）を捉えている（v2 と同一）。
- **disgust PC3** (EVR=3.2%): 「道徳的危害・社会的違反 vs 軽微な不快感」  
  EVR は低いが意味的に純粋な道徳的嫌悪の軸（v2 と同一）。
- **anticipation PC1** (EVR=10.8%): 「希望的好奇心 vs 不安的待機」  
  anticipation の valence 次元（positive vs negative expectancy）に対応（v2 と同一）。

### 2. EVR ≠ 解釈可能性

joy PC1 は全軸中最大の EVR（22.7%）だが contamination=medium, confidence=4 に留まる。
一方 disgust PC3 は EVR=3.2% でも contamination=low, confidence=5。
これは、EVR が大きい軸が必ずしも意味的に純粋な感情次元を捉えているわけでないことを示唆する。
joy は他の感情と比べて、その内部構造が意味的というよりスタイル的・状況的な変動を含む可能性がある。

### 3. スタイル汚染の高い軸（contamination=high）

- **disgust PC4** (EVR=2.6%): 社会的批判 vs 個人的軽微な不快感 → テキストの対話的・非対話的な違いが混入
- **fear PC5** (EVR=2.4%): 個人的喪失 vs 手続き的不確実性 → 抽象度・具体性のスタイル差

これらは低 EVR 軸に多く、残差の微細な変動がスタイルに由来する可能性。

### 4. 感情間の共通パターン

全8感情の PC1 が揃い、それぞれの主要次元が明確になった：

| 感情         | PC1 の主要次元                              | Contamination | Confidence |
|-------------|---------------------------------------------|--------------|------------|
| joy         | 反省的/意味付け的喜び vs 即時的/外的喜び（認知深度）| medium       | 4          |
| trust       | 他者への希望的信頼 vs 自己確信（locus）        | medium       | 5          |
| fear        | 即時 vs 慢性（時間的特性）                    | low          | 5          |
| surprise    | 期待違反あり vs 期待内（expectancy）           | low          | 5          |
| sadness     | 深刻な喪失 vs 日常の平穏（intensity）          | low          | 5          |
| disgust     | 感覚的嫌悪 vs 規範的/抽象的不快（source）      | medium       | 4          |
| anger       | 能動的苛立ち vs 諦め・失望（arousal/locus）    | low          | 5          |
| anticipation| 希望的好奇心 vs 不安的待機（valence）           | low          | 5          |

注目点: 5/8 感情で PC1 が contamination=low を達成。joy と disgust のみ medium に留まる。
joy は認知的深度（反省的 vs 即時的）という次元を捉えているが、これはスタイルと絡みやすい。

---

## 課題・次ステップ

1. **定量的スタイル汚染測定** (exp4): LLM judge による定性評価を超えた、text embeddings 等による汚染の定量化
2. **軸間の相関分析**: 感情をまたいだ共通の潜在次元（例：全感情の arousal 軸、fear/anticipation の acute vs chronic 類似性）の探索
3. **EVR と軸品質の関係**: EVR が低い軸の信頼性を体系的に評価する基準の確立
4. **joy・disgust PC1 の深掘り**: contamination=medium が残る2感情について、追加の極端例や異なる intensity での検証
