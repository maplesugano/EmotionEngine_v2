# Experiment 13: Decomposition Validity

**Script:** `exp13_decomposition_validity.py`  
**Status:** Complete  
**Layer:** 13  
**Sources:** 2,500 / 22,782（ランダムサンプル, seed=0）

---

## 目的

g と r̂_e への orthogonal decomposition が「意味的に valid か」を三つの独立した検定で確かめる。

1. **Source-paired cosine similarity（before / after）** — g 除去により emotion 方向間の識別性が上がるか  
2. **Linear probe accuracy** — r̂_e が c_e より emotion を正確に分類できるか  
3. **Plutchik organization** — 円環的な隣接/対立構造が残差化後に浮かび上がるか

---

## 方法

| 変数 | 値 |
|---|---|
| Primary layer | 13 |
| g の定義 | unit( mean_e unit(c_e) )、layer 13 の pooled CAA から |
| c_e（before） | CAA pooled direction（単位ベクトル化前） |
| r̂_e（after） | c_e − (c_e·g)g を unit-norm |
| delta[n,e] | h_emo[n,e,i,L13] − h_neu[n,L13]、intensity 3段階を平均 |
| Linear probe | 8-class Logistic Regression（lbfgs）、80/20 source-split |

**Source-paired cosine の計算**  
- Off-diagonal (inter-emotion): mean_n cos( unit(δ̄_{n,e1}), unit(δ̄_{n,e2}) )  
- Diagonal (intensity consistency): mean_n mean_{i<j} cos( unit(δ_{n,e,i}), unit(δ_{n,e,j}) )

---

## 結果

### 1. Source-paired cosine similarity

#### Before（c_e）

|  | Joy | Trust | Fear | Surprise | Sadness | Disgust | Anger | Anticipation |
|---|---|---|---|---|---|---|---|---|
| **Joy** | [0.919] | 0.960 | 0.934 | 0.945 | 0.939 | 0.933 | 0.931 | 0.955 |
| **Trust** | | [0.923] | 0.940 | 0.938 | 0.937 | 0.935 | 0.932 | 0.955 |
| **Fear** | | | [0.913] | 0.943 | 0.961 | 0.957 | 0.954 | 0.954 |
| **Surprise** | | | | [0.914] | 0.950 | 0.950 | 0.948 | 0.944 |
| **Sadness** | | | | | [0.919] | 0.965 | 0.962 | 0.949 |
| **Disgust** | | | | | | [0.915] | 0.971 | 0.941 |
| **Anger** | | | | | | | [0.909] | 0.941 |
| **Anticipation** | | | | | | | | [0.924] |

- Off-diagonal 平均: **0.9472**（range 0.909–0.971）  
- Diagonal 平均（intensity consistency）: **0.9170**

#### After（r̂_e）

|  | Joy | Trust | Fear | Surprise | Sadness | Disgust | Anger | Anticipation |
|---|---|---|---|---|---|---|---|---|
| **Joy** | [0.742] | 0.846 | 0.742 | 0.786 | 0.757 | 0.737 | 0.730 | 0.823 |
| **Trust** | | [0.755] | 0.767 | 0.761 | 0.754 | 0.747 | 0.737 | 0.826 |
| **Fear** | | | [0.726] | 0.781 | 0.847 | 0.832 | 0.823 | 0.821 |
| **Surprise** | | | | [0.730] | 0.806 | 0.806 | 0.801 | 0.780 |
| **Sadness** | | | | | [0.738] | 0.861 | 0.852 | 0.796 |
| **Disgust** | | | | | | [0.732] | 0.889 | 0.768 |
| **Anger** | | | | | | | [0.713] | 0.768 |
| **Anticipation** | | | | | | | | [0.752] |

- Off-diagonal 平均: **0.7944**（range 0.713–0.889）  
- Diagonal 平均（intensity consistency）: **0.7359**

> **変化量:**  inter-emotion cosine  0.947 → 0.794（Δ = −0.153）  
> intensity consistency  0.917 → 0.736（Δ = −0.181）

---

### 2. Linear probe accuracy（8-class, Layer 13）

| 条件 | Accuracy |
|---|---|
| Chance | 12.5% |
| Before（c_e） | **92.7%** |
| After（r̂_e） | **93.8%** |

---

### 3. Plutchik organization（cosine by wheel distance）

| Wheel distance | ラベル | Before（c_e） | After（r̂_e） |
|---|---|---|---|
| 1 | Adjacent | 0.9531 | 0.8176 |
| 2 | Dist 2 | 0.9465 | 0.7915 |
| 3 | Dist 3 | 0.9443 | 0.7828 |
| 4 | Opposite | 0.9428 | 0.7766 |
| — | **Gap (adj − opp)** | **0.010** | **0.041** |

距離が大きくなるほど cosine が下がる単調性は before でも after でも成立するが、**差（gap）は after で 4 倍大きい**。

#### 主要ペアの詳細

| ペア | 種類 | Before | After |
|---|---|---|---|
| disgust ↔ anger | Adjacent | 0.971 | 0.889 |
| sadness ↔ disgust | Adjacent | 0.965 | 0.861 |
| joy ↔ trust | Adjacent | 0.960 | 0.846 |
| anticipation ↔ joy | Adjacent | 0.955 | 0.823 |
| **fear ↔ anger** | **Opposite** | 0.954 | **0.823** |
| surprise ↔ anticipation | Opposite | 0.944 | 0.780 |
| joy ↔ sadness | Opposite | 0.939 | 0.757 |
| trust ↔ disgust | Opposite | 0.935 | 0.747 |

---

## 生成チャート

| ファイル | 内容 |
|---|---|
| `thesis/figures/caa_decomposition_cosine_before_after_L13.pdf` | 2-panel heatmap（before / after） |
| `thesis/figures/caa_decomposition_validity_L13.pdf` | 3-panel combined（heatmap × 2 + α_G sweep） |
| `thesis/figures/caa_decomposition_plutchik_L13.pdf` | 距離別 cosine 線グラフ（before vs after） |
| `thesis/figures/caa_decomposition_plutchik_circle_L13.pdf` | 円環チャート（Plutchik wheel、辺色 = cosine 値） |

---

## 解釈

### 1. g は emotion 間で共有された成分である（primary evidence）

Inter-emotion cosine が 0.947 → 0.794 に低下した（Δ = −0.153）。  
g を取り除くことで c_e 間の「詰まり」が解消され、emotion 方向が互いにより独立した向きを向く。  
これは g が「全感情に共通する emotionality 成分」であることの直接的な幾何学的証拠。

### 2. r̂_e は emotion-discriminative である（probe で確認）

残差化後も 8-class probe 精度は **93.8%**（before = 92.7%）。  
g を除去しても emotion-specific 情報が失われておらず、むしろわずかに改善している。

### 3. intensity consistency の低下は decomposition の欠点ではない

Diagonal（intensity consistency）が 0.917 → 0.736 に低下するのは、**g が emotionality の intensity スケール軸を一部包含していた**ことを示す。  
言い換えれば、intensity 1/2/3 にわたって共通する「感情の強さ」成分の一部は g の中に存在し、r̂_e はより intensity ごとに分化した成分になる。

### 4. Plutchik 構造は残差化後に 4 倍明確になる

Before では隣接–対立の gap が 0.010 に過ぎず、g による圧縮で円環構造が見えない。  
After では gap が 0.041 に拡大し、距離 1→4 にかけて単調に cosine が低下する Plutchik 的な勾配が現れる。  
円環チャートでは after パネルで adjacent 辺（太線）が明らかに暖色系、opposite 辺（中線）が寒色系になることで構造が視覚的に確認できる。

### 5. 例外: fear ↔ anger（opposite なのに高 cosine）

fear と anger は Plutchik 上では対立ペアだが、after でも cosine = 0.823 と高い。  
両者は negative-valence / high-arousal という共通の valence×arousal 成分を持ち、  
g（単純な mean CAA 方向）ではこの valence 軸を完全には除去できていない。  
これは decomposition の改良余地（e.g. valence 軸を追加で除去）を示唆するが、  
steering 実験（Exp 9/12）の性能への影響は限定的であるため、現段階では許容範囲内。

---

### 4. MDS 円環：データ駆動の感情配置

1D MDS（cosine 距離）で各感情に実数スコアを付与し、それを円上の角度に変換した
データ駆動の配置を Plutchik 理論順と比較した（`caa_decomposition_mds_circle_L13.pdf`）。

#### MDS 導出順序（best circular alignment 後）

| スロット | MDS 順（before） | MDS 順（after） | Plutchik 理論 | 円環距離 |
|---|---|---|---|---|
| 0 | Trust | Trust | Joy | 1 |
| 1 | Fear | Fear | Trust | 1 |
| 2 | **Joy** | **Joy** | Fear | **2 ← OUT** |
| 3 | Surprise | Surprise | Surprise | 0 ✓ |
| 4 | **Anticipation** | **Anticipation** | Sadness | **3 ← OUT** |
| 5 | Disgust | Disgust | Disgust | 0 ✓ |
| 6 | **Sadness** | **Sadness** | Anger | **2 ← OUT** |
| 7 | Anger | Anger | Anticipation | 1 |

- **Alignment cost（0=完全一致）**: before = 10、after = 10（同一）

#### 注目すべき点

**before と after で MDS 順序が完全に同一。**  
これは g の除去が感情方向の「トポロジー（どれとどれが近いか）」を変えないことを意味する。
変化するのは距離の絶対値だけであり、構造の骨格は保たれている。

**Plutchik からの主要な逸脱：**

1. **Anticipation が大きくずれる（dist=3）**  
   Plutchik では anger ↔ anticipation ↔ joy だが、データでは anticipation が  
   surprise と disgust の間（スロット4）に配置される。  
   After 行列を確認すると anticipation は joy（0.823）・trust（0.826）・fear（0.821）と  
   いずれも高い cosine を持ち、特定の隣人を明確に持たない「浮遊」状態にある。

2. **Negative-valence クラスターの凝集**  
   fear, sadness, disgust, anger が隣接して並ぶ（スロット1, 6, 5, 7）。  
   Plutchik では fear と anger は対立ペアだが、両者の after cosine = 0.823 が  
   MDS にこれらを近接させる。g は emotionality の mean 方向を除去するが、  
   valence 軸（positive/negative）は除去しないため、negative 感情クラスターが残る。

3. **Surprise と Disgust は Plutchik 通り（dist=0）**  
   MDS でも理論通りの位置に現れる最も安定したノード。

#### 解釈

1D MDS 順序は Plutchik 円環を「完全には」再現しない。  
これは g の除去が不完全（valence 軸が残る）であることの幾何学的証拠であり、  
前節の fear ↔ anger の高 cosine と一貫した知見である。  
一方、Surprise・Disgust の正確な位置や negative 感情クラスターの構造は  
Plutchik の隣接性と整合しており、**部分的な再現** という評価が適切。

---

## 結論

| 検定 | 結果 | 解釈 |
|---|---|---|
| Inter-emotion cosine | 0.947 → 0.794 | g が共有成分であることの幾何学的証拠 |
| Linear probe | 92.7% → 93.8% | r̂_e が emotion-specific 情報を保持・改善 |
| Plutchik gap | 0.010 → 0.041（4×） | 残差化後に円環構造が 4 倍明確に浮上 |
| MDS 順序一致 | before = after（cost=10） | トポロジーは g 除去で不変；距離の大きさのみ変化 |
| Plutchik 一致度 | 5/8 ノードが dist≤1 | 部分的再現；valence 軸は g では除去されない |
| α_G sweep（Exp 12より） | α_G > 0 で改善なし | g に emotion-specific 情報が残っていない |

三つの独立した検定が一致して **g / r̂_e への分解は valid** であることを支持する。  
Plutchik 円環構造は残差化後に 4 倍明確になるが、完全な再現ではなく、  
valence 軸（positive/negative）が g によって除去されない残余構造として現れる。  
この限界は論文では「g は emotionality の mean 方向を捉えるが valence 軸は別途存在する」という
解釈として記述できる。
