# Experiment 15: β Sweep Along Shared Meta-Axes

## 目的

共有メタ軸 $m_k$（exp7 で抽出）に沿った activation steering が、感情カテゴリとは直交した前言語的 (pre-verbal) な感情シフトを引き起こせるかを定量的に検証する。

steering ベクトルは $\Delta = \beta \cdot m_k$ のみ（$\alpha_G = \alpha_R = 0$）とし、感情ラベルに依存しない純粋な軸ベクトルによる操作が解釈可能な感情変化をもたらすことを示す。

---

## 実験設計

### モデルと層

- モデル: `model.yaml` で指定したモデル（Llama 系）
- 操作対象層: layer 13（primary layer）
- steering 方式: forward hook で最後のトークン位置の hidden state に $\beta \cdot m_k$ を加算

### メタ軸

exp7 から layer 13 の meta-axes を読み込む。軸は安定性で分類される：

| 軸 | 安定性 | 解釈（exp8v2 より） |
|---|---|---|
| m1 | stable (agreement ≥ 0.83) | agitated/reactive ↔ mellow/accepting（覚醒度・感情反応性） |
| m2 | stable | nostalgic/ruminative ↔ pragmatic/forward-looking（時間的指向性） |
| m3 | stable | interpersonally engaged ↔ self-contained/processing（社会的 vs. 内的指向） |
| m4 | low-stability | mundane/obligatory ↔ meaningful/fulfilling（実存的重み） |
| m5 | low-stability | relief/release ↔ tension/constraint（身体的圧力感） |

### シードテキスト

Plutchik の基本 8 感情（joy, trust, fear, surprise, sadness, disgust, anger, anticipation）それぞれ 3 テキスト、計 24 テキストを使用。テキストは exp9/exp14 と共通。

### β スイープ

$$\beta \in \{-16, -12, -8, -4, 0, 4, 8, 12, 16\}$$

$\beta = 0$ が非操作のベースライン。生成時 `max_new_tokens=60`、`do_sample=False`（greedy）。

### 評価

GPT-4o-mini（OpenAI Batch API）で各生成テキストをスコアリング。judge には軸の名前・両極の説明・期待されるポール方向を提示する。

評価指標（各 0–1）：

| 指標 | 説明 |
|---|---|
| `soundness` | 流暢性・文法的正確さ・繰り返しの無さ |
| `meaning_preserved` | 原文の命題内容の保持 |
| `axis_pole_match` | 期待ポールへのシフト（最重要指標） |
| `subtle_affective_shift` | 感情ラベルを変えない前言語的シフトの有無 |

コヒーレンス閾値: `soundness ≥ 0.70` を満たす $\beta$ 範囲を coherent window として集計。

---

## 結果

### サマリー統計（全軸、$n = 24$ seeds/条件）

#### m1 — agitated/reactive ↔ mellow/accepting

| β | soundness | meaning_pres | pole_match | subtle_shift |
|---|---|---|---|---|
| −16 | 0.008 | 0.354 | 0.000 | 0.000 |
| −12 | 0.379 | 0.875 | 0.117 | 0.146 |
| −8 | 0.913 | 1.000 | 0.462 | 0.529 |
| −4 | 0.992 | 1.000 | 0.550 | 0.596 |
| 0 | 0.996 | 1.000 | 0.188 | 0.271 |
| **4** | **1.000** | **1.000** | **0.917** | **0.938** |
| 8 | 1.000 | 1.000 | 0.833 | 0.917 |
| 12 | 0.983 | 1.000 | 0.638 | 0.696 |
| 16 | 0.792 | 0.388 | 0.092 | 0.088 |

coherent window: $\beta \in [-8, 16]$、ピーク pole match: **0.917** at $\beta = 4$

#### m2 — nostalgic/ruminative ↔ pragmatic/forward-looking

| β | soundness | meaning_pres | pole_match | subtle_shift |
|---|---|---|---|---|
| −16 | 0.663 | 0.083 | 0.104 | 0.063 |
| −12 | 0.954 | 0.729 | 0.646 | 0.479 |
| −8 | 1.000 | 0.958 | 0.504 | 0.583 |
| −4 | 1.000 | 1.000 | 0.421 | 0.538 |
| 0 | 0.996 | 1.000 | 0.208 | 0.229 |
| **4** | **1.000** | **1.000** | **0.971** | **0.971** |
| 8 | 0.958 | 1.000 | 0.917 | 0.913 |
| 12 | 0.792 | 1.000 | 0.692 | 0.742 |
| 16 | 0.354 | 0.696 | 0.296 | 0.317 |

coherent window: $\beta \in [-12, 12]$、ピーク pole match: **0.971** at $\beta = 4$

#### m3 — interpersonally engaged ↔ self-contained/processing

| β | soundness | meaning_pres | pole_match | subtle_shift |
|---|---|---|---|---|
| −16 | 0.479 | 0.767 | 0.392 | 0.379 |
| −12 | 0.875 | 1.000 | 0.496 | 0.558 |
| −8 | 0.963 | 1.000 | 0.296 | 0.450 |
| −4 | 0.992 | 1.000 | 0.317 | 0.433 |
| 0 | 0.996 | 1.000 | 0.271 | 0.396 |
| **4** | **1.000** | **0.979** | **0.850** | **0.879** |
| 8 | 1.000 | 0.754 | 0.242 | 0.258 |
| 12 | 0.950 | 0.188 | 0.000 | 0.000 |
| 16 | 0.958 | 0.958 | 0.438 | 0.438 |

coherent window: $\beta \in [-12, 16]$、ピーク pole match: **0.850** at $\beta = 4$

#### m4 — mundane/obligatory ↔ meaningful/fulfilling（低安定性）

coherent window: $\beta \in [-16, 8]$、ピーク pole match: **0.804** at $\beta = -4$

m4 は HIGH pole ($\beta > 0$) では pole match がほぼゼロ（0.04）であり、軸の方向性が exp8 解釈と逆転している可能性がある。

#### m5 — relief/release ↔ tension/constraint（低安定性）

coherent window: $\beta \in [-8, 12]$、ピーク pole match: **0.608** at $\beta = -4$

m5 の peak は m1–m3 より低く（0.608 vs. 0.85–0.971）、安定性の低さと一致する。

### コヒーレンス閾値まとめ

| 軸 | 安定性 | $L_\beta$ | $R_\beta$ | 中心 $M_A$ | pm@0 | pm_peak | $\beta_{\text{peak}}$ |
|---|---|---|---|---|---|---|---|
| m1 | stable | −8 | 16 | 4.0 | 0.188 | **0.917** | 4 |
| m2 | stable | −12 | 12 | 0.0 | 0.208 | **0.971** | 4 |
| m3 | stable | −12 | 16 | 2.0 | 0.271 | **0.850** | 4 |
| m4 | low | −16 | 8 | −4.0 | 0.000 | 0.804 | −4 |
| m5 | low | −8 | 12 | 2.0 | 0.104 | 0.608 | −4 |

---

## 主要な知見

1. **安定 3 軸（m1–m3）は因果的に活性**: $\beta = 4$ で pole match 0.85–0.97 に達し、感情カテゴリラベルを明示せずとも感情空間内での解釈可能なシフトが生じる。

2. **ベースライン（$\beta = 0$）の pole match は低い（0.19–0.27）**: steering なしでは judge がランダムに近い評価を行うことを確認。`axis_pole_match` 指標の感度を示す。

3. **$\beta = 4$ が安定軸で共通の最適点**: 3 軸すべてで $\beta = 4$ が最高 pole match を示し、coherent window の中心付近と一致する。過度な $|\beta|$ ではコヒーレンスが崩壊し意味保持が低下。

4. **前言語的シフト (`subtle_affective_shift`) は pole match とほぼ連動**: 感情ラベルを変えることなく、表現の「様式」が変化していることを示す。

5. **m4 の方向反転**: m4 は $\beta < 0$ でのみ有意な pole match を示す。PCA による軸方向が exp8 の解釈と逆向きである可能性があり、低安定性フラグの妥当性を裏付ける。

---

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `results/generations.jsonl` | 全 (seed, axis, β) 組み合わせの生成テキスト（1,080 行） |
| `results/eval_results.jsonl` | GPT-4o-mini バッチ評価結果 |
| `results/summary.csv` | 軸×β ごとの集計スコア（mean ± 95% CI） |
| `results/threshold_summary.csv` | コヒーレンス閾値と pm_peak の軸別まとめ |
| `results/fig1_aggregate_metrics_per_axis.png` | 全軸・全指標の β カーブ（5 軸 × 4 指標） |
| `results/fig2_pole_match_stable_axes.png` | 安定 3 軸の pole match オーバーレイ |
| `results/fig3_per_emotion_{m1,m2,m3}.png` | 安定軸ごとの感情種別 pole match |

---

## 論文内での位置づけ

本実験は第 2 章「Pre-Verbal Interpretation of Meta-Axes」節の定量的証拠を提供する。共有メタ軸が activation space において因果的に作用し、感情カテゴリとは独立した前言語的感情次元を符号化していることを示す。
