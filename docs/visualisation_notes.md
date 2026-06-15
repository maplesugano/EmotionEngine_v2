# EmotionEngine 可視化実装メモ

## 何を可視化しようとしているか

論文の中心的な主張：各命名感情は単一ベクトル $\hat{\mathbf{r}}_e$ ではなく、残差空間において**コーン状のプロトタイプ領域**を占めている。そのコーン内部の構造を記述するのが共有メタ軸 $\mathbf{m}_k$。

目標は「$s(x)$ がコーンのどこにいるか」「$\gamma_k$ を動かすとどう動くか」「コーンの境界はどこか」をインタラクティブに見せること。

---

## アーキテクチャ

### バックエンド (`src/`)

#### `emotion_state.py`
- `compute_emotion_profile(s_raw, mu, caa)` — $s(x) - \mu$ を各 $\hat{\mathbf{r}}_e$ に射影し、8感情スコアを返す（既存）
- `compute_meta_projections(s_raw, mu, meta_axes)` — 同じ $s(x) - \mu$ を各 $\mathbf{m}_k$ に射影し、`{"m1": ..., "m5": ...}` を返す（追加）

#### `api.py`
- `_meta_axes: np.ndarray` — `layer13_meta_axes_pca.npy` を lazy load（shape: `(5, 4096)`）
- `/analyze` レスポンスに `meta_projections: dict[str, float]` フィールドを追加

### フロントエンド (`frontend/src/App.tsx`)

#### 焼き込み定数

**`META_AXIS_EMO_PROJ`** — メタ軸→感情方向への射影行列（バックエンドで事前計算、フロントに定数として持つ）

```
(meta_axes_pca @ r_hat.T)  # shape: (5, 8)
```

各エントリ `META_AXIS_EMO_PROJ[mk][e]` = $\mathbf{m}_k \cdot \hat{\mathbf{r}}_e$。
$\mathbf{m}_k \perp \hat{\mathbf{r}}_e$ という論文の性質より、これらはすべてほぼゼロ（最大でも ~0.75）。

**`COHERENT_WINDOW`** — `exp15_n100_summary` の $\beta$-sweep から取った軸ごとの coherent window（soundness ≥ 0.70 が成立する $\gamma$ の範囲）

| 軸 | $L_\beta$ | $R_\beta$ |
|----|-----------|-----------|
| $\mathbf{m}_1$ | −8 | +16 |
| $\mathbf{m}_2$ | −16 | +12 |
| $\mathbf{m}_3$ | −12 | +16 |

---

## 3Dプロット (`EmotionPlot3D`)

### 座標系

プロット空間の軸は感情偏差スコア：ユーザーが選んだ3感情 $(e_x, e_y, e_z)$ の deviation（= raw score − neutral baseline）。

$s(x)$ の座標 = `(deviations[xEmo], deviations[yEmo], deviations[zEmo])`。

### 描画要素

| 要素 | 説明 |
|------|------|
| **灰色点 (neutral)** | 原点 = 中立状態 |
| **青い線 + 点 s(x)** | 原点から $s(x)$ へのベクトル |
| **3本の矢印** | $\mathbf{m}_1$（赤）、$\mathbf{m}_2$（緑）、$\mathbf{m}_3$（橙）。$s(x)$ を起点に感情空間へ射影した方向を示す。長さは `range * 0.4`（見やすさのためのスケール、$s(x)$ の値は反映しない） |
| **プロトタイプ領域（灰色点群）** | 下記参照 |
| **ダイアモンド** | $s(x) + \sum_k \gamma_k \mathbf{m}_k$ を感情空間に射影した位置。感情フリップ時は赤に変わる |

### プロトタイプ領域のサンプリング

`samplePrototypeRegion` で計算。アルゴリズム：

1. $(\gamma_1, \gamma_2, \gamma_3)$ を coherent window 内で $15^3 = 3375$ 点グリッドサンプリング
2. 各点で全8感情のジグルスコアを計算：
   $$\text{score}(e) = \text{deviation}(e) + \sum_k \gamma_k \cdot (\mathbf{m}_k \cdot \hat{\mathbf{r}}_e)$$
3. dominant 感情が $s(x)$ のものと異なれば除外（コーンの外）
4. 残った点を感情空間に射影して描画

この領域は**非対称**になる。competitor 感情に近い方向では早く切り取られ、遠い方向は coherent window 境界まで伸びる。

### 感情フリップ検出

現在のスライダー位置 $(\gamma_1, \gamma_2, \gamma_3)$ について同じスコア計算を実行し、dominant 感情が変わっていれば：
- ダイアモンドを赤にする
- レジェンドに `⚠ flipped→{emotion}` を表示
- プロット下に赤いバナーを表示：「Dominant emotion flipped: X → Y」

フリップの幾何学的な意味：8感情のうち competitor $e'$ が dominant $e^*$ を超える条件は

$$\sum_k \gamma_k \left( \mathbf{m}_k \cdot \hat{\mathbf{r}}_{e'} - \mathbf{m}_k \cdot \hat{\mathbf{r}}_{e^*} \right) = \text{deviation}(e^*) - \text{deviation}(e')$$

つまり $\gamma$-空間における**超平面**。各 competitor 感情が一枚の平面を与え、valid 領域はそれらの交差（box ∩ halfplanes）。

---

## 既知の制限

- **矢印の長さは $s(x)$ の $\mathbf{m}_k$ 射影値を反映していない**。方向のみ示す。実際の射影値は API が `meta_projections` で返しているが現在プロットには使っていない。
- **coherent window は単軸実験から取得**。複数の $\gamma_k$ を同時に動かした場合の相互作用は未検証。プロトタイプ領域の境界は近似。
- **$\mathbf{m}_4$, $\mathbf{m}_5$ はプロットに表示していない**（幾何学的安定性が低いため。スライダーにも出していない）。
- **プロトタイプ領域は感情空間への射影**。$\mathbf{m}_k \perp \hat{\mathbf{r}}_e$ なので射影はほぼゼロになり、点群は $s(x)$ 周辺の狭い帯に集まって見える可能性がある。

---

## ファイル

| ファイル | 変更内容 |
|----------|---------|
| `src/emotionengine/emotion_state.py` | `compute_meta_projections` 追加 |
| `src/api.py` | `_meta_axes` グローバル、`META_AXES_PATH`、`AnalyzeResponse.meta_projections` 追加 |
| `frontend/src/App.tsx` | `META_AXIS_EMO_PROJ`、`COHERENT_WINDOW`、`samplePrototypeRegion`、`EmotionPlot3D` 更新、gamma スライダー、フリップバナー追加 |
| `local_axes_experiments/exp7_meta_axis_extraction/results/layer13_meta_axes_pca.npy` | API がロードするメタ軸ファイル（既存） |
