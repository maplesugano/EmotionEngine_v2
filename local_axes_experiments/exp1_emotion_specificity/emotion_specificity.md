# Experiment 1: Is r_e Truly an Emotion-Specific Direction?

## 目的

共通方向 g（全感情に共通する "emotional activation" 成分）を取り除いたあとも、感情ラベルを識別できる signal が残るかを検証する。

## 方法

- **δ[n,e]** = h_emotion[n,e] − h_neutral[n]（各 source × emotion の per-source delta）
- **g** = unit_norm(mean over e of caa_pooled[e, layer])（感情共通方向）
- **δ_resid[n,e]** = δ[n,e] − (δ[n,e]·g)g（g 成分を射影除去）
- 分類器: PCA-64 → StandardScaler → LogisticRegression (lbfgs)
- 評価: GroupKFold (k=5, group=source_id)、balanced accuracy
- チャンス水準: **0.125**（8感情均等）

## 結果

| Layer | bal_acc_pca | std   |
|-------|-------------|-------|
| 8     | 0.269       | 0.289 |
| 10    | **0.853**   | 0.002 |
| 13    | **0.857**   | 0.001 |
| 16    | 0.755       | 0.002 |
| 19    | 0.652       | 0.002 |
| 22    | 0.563       | 0.002 |

## 解釈

### 1. r_e は emotion-specific な方向である

Layer 10–13 では balanced accuracy が **0.85 超** に達する。
これは共通方向 g を完全に除去した後でも感情ラベルを強く予測できることを意味し、次の主張を支持する：

> **r_e は単なる emotional intensity ではなく、名前付き感情間の質的差異（joy vs. sadness vs. anger 等）をエンコードする方向である。**

### 2. Layer 10–13 が emotion representation のピーク

Layer 8 では signal が弱く（0.27）、Layer 10 で急激に上昇し Layer 13 でピークを迎える。
LLM の中間層付近で感情の質的表現が形成されるという知見と整合する。

### 3. 後半層で signal が単調減衰

Layer 16 以降は 0.755 → 0.652 → 0.563 と低下する。
後半層では感情の質的情報がトークン予測に向けた汎用表現へ変換・圧縮されていると考えられる。

### 4. Layer 8 の異常な std（0.289）

Layer 8 では std が極端に大きい（他層は ~0.002）。
これは fold ごとに精度が大きく変動することを示しており、Layer 8 ではまだ感情方向が安定して分化していないことを示唆する。

## 結論

g を除去した残差方向 r_e は、Layer 10–13 においてチャンス水準（0.125）の **7倍近い** 感情識別精度を示す。
この実験は「r_e が emotion-specific な構造を持つ」という仮説を強く支持する。
