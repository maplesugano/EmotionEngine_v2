# Experiment 2 Analysis (Layer 13)

## 対象
- スクリプト: `local_axes_experiments/exp2_emotion_internal_structure/exp2_emotion_internal_structure.py`
- 集計結果: `local_axes_experiments/exp2_emotion_internal_structure/results/layer13_participation_ratio.csv`
- EVR詳細: `local_axes_experiments/exp2_emotion_internal_structure/results/layer13_*_evr.csv`

## 主要結果
`pr_excess = participation_ratio - pr_shuffle_mean` を感情固有の多軸性の指標とみなす。

- `pr_excess < 0`: シャッフルより集中（少数支配軸）
- `pr_excess > 0`: シャッフルより拡散（多様な局所軸）

| emotion | PR | PR_shuffle | PR_excess | EVR_PC1 |
|---|---:|---:|---:|---:|
| joy | 6.72 | 13.62 | -6.90 | 0.227 |
| trust | 10.54 | 13.65 | -3.11 | 0.158 |
| sadness | 12.11 | 13.59 | -1.48 | 0.136 |
| disgust | 12.74 | 13.61 | -0.87 | 0.130 |
| anticipation | 15.30 | 13.65 | 1.65 | 0.108 |
| anger | 15.56 | 13.63 | 1.93 | 0.107 |
| fear | 18.16 | 13.60 | 4.57 | 0.091 |
| surprise | 18.94 | 13.63 | 5.31 | 0.076 |

## 解釈
- `joy`, `trust` は `pr_excess` が大きく負で、内部構造が比較的集中している。
- `sadness`, `disgust` はシャッフルに近く、局所軸はあるが強くはない。
- `anger`, `anticipation` は軽度に正で、やや多軸的。
- `fear`, `surprise` は大きく正で、最も多軸的・分散的な内部構造を示す。

## 補足
- 今回のシャッフル基準は「全感情プールから再サンプリング」なので、感情固有構造を壊したノイズ床として妥当。
- `pr_shuffle_std` は全感情で小さく、推定は安定している。

## 結論
Layer 13 では感情内部構造は一様ではなく、
- 集中型: `joy`, `trust`
- 中間: `sadness`, `disgust`, `anger`, `anticipation`
- 拡散型: `fear`, `surprise`
という分化が見られる。
