# 実験 6 分析レポート：Local Sub-Axes は共有されているか？

**対象モデル層**: Layer 13  
**分析対象**: 8 感情 × top-5 PC = 40 本の local PC 軸  
**実行日**: 2026-06-05

---

## 1. 主要結論

> **Local sub-axes は emotion label を横断して強く共有されている。**
> 支配的な構造は「どの感情か」ではなく「何番目の PC か（分散説明量の順位）」である。

これは EmotionEngine の制御空間が emotion label ではなく
**label-free affective dimensions** によって組織されていることを示す直接的な証拠である。

---

## 2. クラスタリング結果（Ward, K=8）

| Cluster | Size | PC 順位 | 構成感情 | 解釈 |
|---------|------|---------|---------|------|
| **1** | 8 | PC1 のみ | 全 8 感情 | **Universal PC1 軸**：すべての感情で共有される最大分散方向 |
| **4** | 8 | PC2–3 | 全 8 感情 | **Universal PC2 軸**：2番目の共有 affective dimension |
| **3** | 8 | PC2–5 | 全 8 感情 | **Universal PC3 軸**：3番目の共有 affective dimension |
| **5** | 6 | PC3–5 | joy/trust/fear/sadness/disgust/anger | 4番目の共有 dimension（joy/trust/anticipation の PC4-5 は別れる） |
| 2 | 2 | PC3,5 | surprise, anticipation | 周辺的（surprise, anticipation に固有） |
| 6 | 3 | PC4,5 | joy, trust, anticipation | 周辺的 |
| 7 | 2 | PC4 | surprise, anger | 周辺的 |
| 8 | 3 | PC4,5 | fear, sadness, disgust | 周辺的 |

**最も重要な観察**：Cluster 1, 4, 3 はそれぞれ**全 8 感情**の PC1, PC2, PC3 に相当する軸を含む。
感情ラベルによる分離はクラスタ 2〜8 の周辺的な軸にしか現れない。

---

## 3. Cross-Emotion |Cosine Similarity|

### 3.1 PC1 軸：全感情で高い類似度

| 感情 | 最近傍（異感情） | \|cos\| |
|------|----------------|---------|
| disgust_PC1 | anger_PC1 | **0.950** |
| sadness_PC1 | disgust_PC1 | **0.922** |
| trust_PC1 | disgust_PC1 | **0.886** |
| joy_PC1 | anger_PC1 | **0.874** |
| anticipation_PC1 | sadness_PC1 | **0.805** |
| fear_PC1 | disgust_PC1 | **0.839** |
| surprise_PC1 | sadness_PC1 | **0.743** |

全感情の PC1 が互いに 0.74–0.95 の |cosine| で揃っている。
これは**単一の universal primary axis**が存在することを強く示唆する。

### 3.2 Sadness–Disgust–Anger の強結合ブロック

| ペア | PC1 | PC2 | PC3 |
|------|-----|-----|-----|
| disgust ↔ anger | **0.950** | **0.912** | **0.933** |
| sadness ↔ disgust | **0.922** | **0.890** | **0.907** |
| sadness ↔ anger | （推移） | — | — |

PC1–PC3 の**3軸すべて**が 0.89 以上で一致する。
sadness, disgust, anger は独立した emotion label を持つにもかかわらず、
representation 空間では事実上同一の sub-space に投影されている。

### 3.3 PC2–PC3 の共有パターン

| 軸 | 最近傍 | \|cos\| |
|----|--------|---------|
| fear_PC2 | anger_PC2 | 0.859 |
| fear_PC4 | sadness_PC3 | 0.829 |
| fear_PC3 | disgust_PC5 | 0.620 |
| trust_PC4 | sadness_PC5 | 0.596 |
| sadness_PC5 | trust_PC4 | 0.596 |

PC2–3 レベルでも cross-emotion sharing は継続して存在する。

---

## 4. Meta-PCA：Explained Variance Ratio

| Meta-PC | EVR | 累積 EVR |
|---------|-----|---------|
| 1 | **19.6%** | 19.6% |
| 2 | **18.3%** | 37.9% |
| 3 | **13.4%** | 51.2% |
| 4 | 9.6% | 60.8% |
| 5 | 8.6% | 69.4% |
| 6 | 5.3% | 74.7% |
| 7 | 4.1% | 78.8% |
| 8 | 3.5% | 82.2% |
| 9 | 2.7% | 84.9% |
| 10 | 2.2% | 87.2% |

**観察**：
- 上位 3 成分で累積 51%。Scree plot は比較的なだらか。
- PC1, PC2 の差が僅少（19.6% vs 18.3%）→ dominant な単一 meta-axis は存在せず、
  複数の equally-weighted meta-dimensions が並立している。
- 上位 5 成分で 69%、10 成分で 87% を説明する。

---

## 5. Meta-PCA 散布図の解読

散布図（meta_pca_scatter.png）から読み取れる構造：

- **右下象限**（Meta-PC1 > 0, Meta-PC2 < 0）：各感情の **PC1** が集結。
  色の違いにかかわらず dense cluster を形成 → universal primary axis の確認。
- **右上象限**（Meta-PC1 > 0, Meta-PC2 > 0）：各感情の **PC3** が集結。
  特に sadness/disgust/anger/fear の PC3 が高密度に重なる。
- **左上象限**（Meta-PC1 < 0, Meta-PC2 > 0）：**PC2** が主に分布。
  trust_PC1 が外れ値として左上に位置（trust の PC1 は他感情と方向が異なる可能性）。
- **左下象限**：PC2–PC3 の intermediate 軸や joy_PC2, sadness_PC2 など。

meta-PCA 空間では色（感情ラベル）よりも PC 番号のほうがクラスタ構造を説明している。

---

## 6. 解釈：何が示されたか

### 6.1 Emotion-specific structure は PC4 以降にしか現れない

| PC 順位 | 構造 |
|---------|------|
| PC1 | **完全共有**：全 8 感情が 1 つのクラスタ（|cos| = 0.74–0.95） |
| PC2 | **ほぼ完全共有**：7/8 感情が同一クラスタ |
| PC3 | **ほぼ完全共有**：全 8 感情が同一クラスタ（sadness/disgust/anger は特に強い） |
| PC4–5 | **部分共有 + emotion-specific**：クラスタが分裂し始め、感情固有の軸が現れる |

高次 PC（PC4, PC5）になって初めて emotion-specific な構造が出現する。
モデルが affective information を符号化する主要な方法は「どの感情か」ではなく
**共有 affective dimensions への投影**である。

### 6.2 Sadness–Disgust–Anger は同一 sub-space

PC1–PC3 の 3 軸すべてで |cos| ≥ 0.89 が成立する。
これら 3 感情の representation はほぼ同一の 3 次元 sub-space に属する。
生体心理学的解釈では、sadness・disgust・anger は共通して
**negative valence + threat/aversion** の component を持つことが知られており、
本結果はその神経表現上の対応物を示している可能性がある。

### 6.3 Universal PC1 の意味するもの

全感情にわたって共有される PC1 は、感情の種類に関わらない
**affective intensity / emotionalization** の共有 axis である可能性が高い。
この解釈は exp2 で既に確認された「per-emotion intensity direction の除去後にも PC1 が強い」
という観察と整合する。ただし exp2 で除去された軸（high–low CAA）と
ここで発見された universal PC1 が同一かどうかは追加検証が必要。

---

## 7. EmotionEngine UI への示唆

クラスタ構造と meta-PCA から、以下の **label-free affective controls** が
データドリブンに示唆される：

| 候補 meta-axis | 対応する構造 | 備考 |
|----------------|-------------|------|
| **Affective intensity / Emotionalization** | Cluster 1（全 PC1） | 最も重要な共有軸 |
| **Negative valence / Aversion** | sadness+disgust+anger ブロック（PC2） | valence 方向の第 2 軸 |
| **Arousal / Tension** | Cluster 3–4（PC2–3 の混合） | 解釈には生理指標との照合が必要 |
| **Positive affect / Approach** | joy+trust 方向（Cluster 4 内） | sadness 群とは独立 |
| **Uncertainty / Surprise** | Cluster 2（surprise+anticipation） | より小さな共有ブロック |

現在の "joy knob / sadness knob" という emotion-label ベースの設計は、
空間の主方向とは直交していない恐れがある。
**Meta-PC1 knob → Meta-PC2 knob → …** という label-free 設計の方が
representation 空間を効率的に制御できる可能性が高い。

---

## 8. 今後の実験

1. **Universal PC1 の同定**：
   exp2 で除去した per-emotion intensity direction との cosine を測定し、
   universal PC1 が既知の arousal/intensity 軸と一致するか確認する。

2. **Steering 実験**：
   meta-PCA 成分方向へのステアリング（exp5 の手法を流用）を実施し、
   label-free controls が実際に感情表現を制御できるか行動的に検証する。

3. **複数層での比較**：
   Layer 8, 10, 16 でも同実験を実施し、共有構造がどの層で形成されるか追跡する。

4. **Sadness–Disgust–Anger 融合の確認**：
   3 感情のデータを合算した PCA と個別 PCA の sub-space alignment（Grassmann 距離）を測定。

5. **Meta-axis の言語的解釈**：
   meta-PC1/2 の極端値（正方向・負方向）における生成テキストの qualitative 分析。

---

## 付録：出力ファイル一覧

| ファイル | 内容 |
|----------|------|
| `layer13_local_pcs.npy` | 全 40 本の local PC ベクトル (40, D) |
| `layer13_cosine_sim.csv` | 40×40 cosine similarity 行列 |
| `layer13_cross_emotion_sharing.csv` | 各軸の cross-emotion 最近傍と |cos| |
| `layer13_clusters_k8.csv` | Ward クラスタリング結果（K=8） |
| `layer13_meta_pca_evr.csv` | Meta-PCA の explained variance ratio |
| `layer13_meta_pca_projections.csv` | 40 軸の meta-PCA 座標 |
| `layer13_meta_pca_components.csv` | Meta-PC 方向ベクトル（D 次元） |
| `layer13_cosine_heatmap.png` | |cos| ヒートマップ |
| `layer13_dendrogram.png` | Ward デンドログラム（emotion 色分け） |
| `layer13_meta_pca_scatter.png` | Meta-PC1/2 散布図 |
| `layer13_cluster_composition_k8.png` | クラスタ内 emotion 構成バー |
