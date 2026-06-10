# 実験 6.1 分析レポート：pooled PCA は meta-axis を復元するか？

**対象モデル層**: Layer 13
**比較対象**: exp6 の meta-PCA components（2段階）vs. per-emotion centered の pooled PCA（1段階）
**実行日**: 2026-06-07
**スクリプト**: `exp6_1_pooled_pca.py` → 出力 `results_pooled/`

---

## 0. 動機

exp6 の解析で、pooled PCA と meta-PCA は次のように「重みだけが違う同じ固有値問題」に帰着すると予測した：

```
pooled PC1 = top eigvec of  Σ_{e,k} λ_{e,k} · u_{e,k} u_{e,k}^T   (分散 λ で重み付け)
meta-PC1   = top eigvec of  Σ_{e,k}      1   · u_{e,k} u_{e,k}^T   (等重み)
```

共有軸は λ も大きく登場回数も多いので、両者の上位成分は一致するはず、という仮説を検証する。

---

## 1. 主要結論

> **部分空間レベルでは仮説は成立する。軸レベルでは一致しない。**
> pooled PCA の top-3 と meta-PCA の top-3 は **96.8% のエネルギーが重なる同一 3 次元 affective subspace** を張る。
> ただし個々の軸は subspace 内で回転しており、対角 1:1 対応にはならない。
> 原因は予告どおり「分散重み vs 等重み」の違いと、**meta-PCA 上位2成分の縮退**（EVR 19.6% ≈ 18.3%）。

さらに副産物として：

> **pooled PCA のほうが universal PC1（emotionalization 軸）をクリーンに 1 本で抽出する。**
> meta-PCA はこの軸を meta_PC1 と meta_PC2 に分散させてしまう。UI の label-free knob 用途では pooled PCA のほうが望ましい可能性が高い。

---

## 2. Pooled PCA の Explained Variance Ratio

| Pooled PC | EVR | 累積 |
|-----------|-----|------|
| 1 | **11.4%** | 11.4% |
| 2 | 4.5% | 15.9% |
| 3 | 3.1% | 18.9% |
| 4 | 2.2% | 21.1% |
| 5 | 1.9% | 23.1% |

PC1 が突出（11.4%, 2位の 2.5 倍）。meta-PCA では PC1/PC2 が拮抗していた（19.6/18.3%）のと対照的で、**pooled PCA では単一の支配的軸が明確に出る**。これが下記 §4 の universal PC1 の clean 抽出に対応する。

---

## 3. 軸ごとの |cosine|（誤解を招く指標）

|            | meta_PC1 | meta_PC2 | meta_PC3 |
|------------|----------|----------|----------|
| pooled_PC1 | 0.557    | **0.691**| 0.448    |
| pooled_PC2 | **0.715**| 0.129    | 0.656    |
| pooled_PC3 | 0.400    | **0.700**| 0.555    |
| pooled_PC4 | 0.109    | 0.097    | 0.010    |
| pooled_PC5 | 0.037    | 0.003    | 0.159    |

- 対角（pooled_PCi ↔ meta_PCi）は **一致しない**：0.557, 0.129, 0.555。
- 順序が入れ替わっている：pooled_PC1↔meta_PC2、pooled_PC2↔meta_PC1。
- pooled_PC4/PC5 はどの meta-PC とも無相関（< 0.16）→ 下位成分は両手法で別物。

**軸ごとの比較は不適切**。meta-PCA の PC1/PC2 が縮退（EVR 拮抗）しているため、その平面内で軸は自由回転でき、ラベルの 1:1 対応に意味がない。正しくは部分空間で比べる（§4）。

---

## 4. 部分空間アライメント（回転不変・正しい指標）

主角度（principal angles）の余弦と、相互エネルギー捕捉率：

| top-r | 主角度 \|cos\|             | 平均   | meta ⊂ pooled span | pooled ⊂ meta span |
|-------|---------------------------|--------|--------------------|--------------------|
| 1     | [0.557]                   | 0.557  | 0.310              | 0.310              |
| 2     | [0.568, **0.997**]        | 0.782  | 0.658              | 0.658              |
| 3     | [**0.969, 0.985, 0.998**] | **0.984** | **0.968**       | **0.968**          |

**読み方**：
- **top-3 で完全一致に近い**：3 本すべての主角度 |cos| > 0.96、相互捕捉率 96.8%。
  → pooled PCA の張る 3 次元 affective subspace は meta-PCA のそれと事実上同一。**仮説は subspace レベルで成立**。
- top-2 の段階で 1 方向は既に |cos|=0.997 で完全一致、もう 1 方向（0.568）が縮退平面内の回転でズレている。
- top-1 だけだと 0.557 と低い：pooled の「最大分散方向」と meta の「最も登場頻度の高い方向」は**単独では別の軸**。これは weighting の違いそのもの。

→ 「pooled PCA は meta-axis を復元するか？」の答えは **「3次元の共有 affective subspace としては Yes。個別軸の向きとしては No（回転している）」**。

---

## 5. Universal PC1 の clean 抽出（副産物・重要）

8 感情それぞれの PC1 を単位化して平均した方向を **universal-PC1**（exp6 が「全感情で共有される最大分散軸＝emotionalization」と同定した軸）とし、各手法の第1成分との |cos| を測定：

| 比較 | \|cos\| |
|------|---------|
| **pooled_PC1** ↔ universal-PC1 | **0.902** |
| meta_PC1 ↔ universal-PC1 | 0.719 |
| meta_PC2 ↔ universal-PC1 | 0.588 |

**pooled_PC1 は universal-PC1 とほぼ同一（0.90）**。一方 meta-PCA はこの軸を meta_PC1（0.72）と meta_PC2（0.59）に**分散**させている（縮退＋等重みのため）。

pooled_PC1 と各感情 PC1 の |cos|：

```
[anger 0.94, anticipation 0.94, disgust 0.88, fear 0.71,
 joy 0.95, sadness 0.97, surprise 0.95, trust 0.71]
```

8 感情中 6 感情で 0.88–0.97。外れる 2 つ（fear 0.71, trust 0.71）は exp6 が既に「PC1 の向きが他と異なる候補」として指摘した surprise/trust 系と整合。

---

## 5.5 thesis の meta-axis $\mathbf{m}_k$ との直接比較（最重要・追記）

§3–4 は exp6 の **40 ベクトル meta-PCA**（rank 混在）との比較だった。thesis が実際に使う定義は
**$\mathbf{m}_k$ = rank ごとに符号整列した平均**（sign-aligned mean）であり、こちらと比べると
**部分空間どころか軸ごとにクリーンに一致する**：

| 比較 | \|cos\| |
|------|---------|
| pooled_PC1 ↔ $\mathbf{m}_1$ | **0.988** |
| pooled_PC2 ↔ $\mathbf{m}_2$ | **0.980** |
| pooled_PC3 ↔ $\mathbf{m}_3$ | **0.931** |

非対角はすべて小さい（$\mathbf{m}_1$: [0.99, 0.07, 0.02]、$\mathbf{m}_2$: [0.13, 0.98, 0.26]、$\mathbf{m}_3$: [0.00, 0.07, 0.93]）。
部分空間捕捉率も top-1/2/3 で 0.975 / 0.979 / 0.966。

**なぜ §3-4 では回転していたのか**：40 ベクトル meta-PCA は rank を混在させて分散順に並べ替えるため、
縮退平面内で軸が回る。一方 $\mathbf{m}_k$ は rank ごとに平均を取るので「平均共分散の固有ベクトル」に近く、
pooled PCA の固有ベクトル（=平均共分散の固有ベクトル）と**定義上ほぼ一致**する。
→ **比較対象は $\mathbf{m}_k$ にすべき。その場合 pooled PCA は m_1–m_3 を個別に 0.93–0.99 で復元する。**

これは「pooled PCA は emotion split も rank grouping も一切せずに、m_1–m_3 を再現する」という
**最も強い形の robustness**（meta-axis は2段階抽出の artifact ではない）を与える。

---

## 6. 解釈：仮説のどこが当たり、どこが外れたか

| 主張 | 結果 |
|------|------|
| 「両者は重みだけ違う同じ固有値問題」 | ✅ 正しい（数式どおり） |
| 「上位成分が**ほぼ一致**」 | △ **subspace としては一致（96.8%）／個別軸としては不一致** |
| 「λ重み vs 等重みの差は下位・不均一で効く」 | ✅ まさにそれが起きた（top-1 のズレ＝0.557、PC4/5 の無相関） |

最初の「ほぼ一致するはず」は **3次元部分空間の主張としては正しく、軸ごとの主張としては言い過ぎだった**。決定打は **meta-PCA 上位の縮退**（19.6% ≈ 18.3%）で、縮退平面内では軸が自由回転するため、等重み（meta）と分散重み（pooled）が**同じ平面を別々の向きで切る**。張る空間は同じでも、軸ラベルは揃わない。

---

## 7. thesis / EmotionEngine UI への示唆

1. **収束的証拠として強い**：独立した 2 経路（2段階 meta-PCA と 1段階 pooled PCA）が同一の 3 次元 affective subspace に 96.8% 収束した。「shared affective subspace は手法非依存に実在する」と主張できる。

2. **control knob の取り方は pooled PCA を推奨**：
   - pooled_PC1 = universal emotionalization 軸を**単一でクリーンに**与える（|cos|=0.90）。
   - meta-PCA はこれを 2 軸に分けてしまい、UI の第1ノブとして扱いにくい。
   - 「最も効く 1 本の knob」が欲しい実用上は pooled PCA が素直。

3. **"証明" と "実装" で手法を使い分ける**：
   - 共有の**証明**には依然 exp6 の 2段階が必要（per-emotion 独立抽出 → 揃うことを示す論理）。
   - 共有された軸の**抽出/実装**には pooled PCA が簡潔かつ安定。

---

## 8. 今後

1. pooled_PC1 方向への steering（exp5 流用）で、universal emotionalization 軸が行動的に効くか検証。
2. Layer 8/10/16 でも pooled vs meta の subspace 一致率を測り、層依存性を確認。
3. 縮退している meta_PC1/PC2 平面の言語的解釈（valence vs arousal の混合かどうか）。

---

## 付録：出力ファイル一覧（`results_pooled/`）

| ファイル | 内容 |
|----------|------|
| `layer13_pooled_pcs.npy` | pooled PCA 上位 5 成分 (5, D) |
| `layer13_pooled_evr.csv` | pooled PCA の EVR |
| `layer13_pooled_vs_meta_cos.csv` | pooled × meta の \|cos\| 行列 |
| `layer13_subspace_alignment.csv` | top-r 主角度・相互捕捉率 |
| `layer13_pooled_vs_meta_cos.png` | \|cos\| ヒートマップ |
