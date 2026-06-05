# Sources PDFごとの実験結果まとめ（EmotionEngine / CAA 系）

作成日: 2026-06-03  
対象: `/mnt/data` にある PDF ファイルのみ。CSV は対象外。  
対象PDF:

1. `2312.06681v4.pdf` — *Steering Llama 2 via Contrastive Activation Addition*
2. `caa_geometry.pdf` — CAA Emotion Direction Geometry
3. `caa_common_direction_residualization.pdf` — Experiment A: Common Emotionalization Direction Residualization
4. `caa_residualized_per_source.pdf` — Experiment B: Residualized Per-Source CAA Analysis
5. `caa_per_source_structure.pdf` — CAA Per-Source Emotion Direction Structure
6. `caa_manifold_linearity.pdf` — CAA Linear Manifold Continuity Test
7. `caa_local_pc_interpretation.pdf` — Experiment D: Interpret Local PCA Axes Within Each Emotion
8. `caa_seed_emotion_robustness.pdf` — Seed Emotion Robustness Test

---

## 0. 全体の読み方

この一連のPDFは、EmotionEngine の中心仮説、すなわち「LLM内部の感情表現を、離散ラベルではなく、低次元・連続・線形に操作可能な感情変化空間として扱えるか」を検証する実験群である。

基本となる操作は CAA（Contrastive Activation Addition）で、ここでは主に「neutral paraphrase から emotion rewrite への残差ストリーム差分」を感情方向として扱っている。

多くの実験で共通する記号は次の通り。

- `h_emotion[n, e, i, l]`: ソース `n`、感情 `e`、強度 `i`、レイヤー `l` の感情 rewrite 活性。
- `h_neutral[n, l]`: 同じソース `n` の neutral paraphrase 活性。
- 個別CAAベクトル: `δ[n,e] = h_emotion[n,e] - h_neutral[n]`。実験によって強度方向は平均される。
- グローバルCAAベクトル: `c_e = mean_n,i δ[n,e,i]` を単位正規化したもの。
- 共通 emotionalization 方向: `g = unit(mean_e c_e)`。
- 感情固有残差: `r_e = c_e - (c_e · g)g`、または個別ベクトルに対して `δ_resid[n,e] = δ[n,e] - (δ[n,e] · g)g`。

全体として得られている大きな結論は次のように整理できる。

1. 感情CAAの最大成分は、個別感情ではなく「neutral → emotional」という共通方向 `g` である。
2. ただし `g` を除去すると、Plutchik的な対立ペアや感情間の差がより見えやすくなる。
3. 個別ソースごとのCAAはグローバルCAAと非常によく揃うため、平均方向としてのCAAには意味がある。
4. 一方、個別CAA分布は大きく重なり、低次元では感情分類精度が限定的である。これは「8感情が完全に分離したクラスタ」ではなく、連続的で重なりのある空間であることを示す。
5. 各感情の内部には複数の局所PC軸があり、joy や sadness のような名前付き感情は単一方向ではなく、内部にサブタイプを持つ。
6. `α_g` と `α_r` に分解して steering すると、出力は target emotion だけで決まる絶対座標というより、seed emotion / seed text に強く依存する「文脈依存の感情変化ベクトル」として振る舞う。

---

# 1. `2312.06681v4.pdf` — Steering Llama 2 via Contrastive Activation Addition

## 1.1 位置づけ

このPDFは、EmotionEngine実験群の理論的・方法論的な基礎になっている CAA 論文である。EmotionEngine の各ノートブックは、この論文の CAA という考え方を「感情 rewrite」へ応用し、さらに共通方向・残差方向・局所PCA・seed依存性などを詳しく調べている。

論文の主張は、LLMの残差ストリームに対して、ある振る舞いを表す方向ベクトルを加算・減算することで、推論時にモデル出力を制御できるというもの。

## 1.2 CAA の基本方法

CAAは、ある振る舞いについて positive / negative の対照ペアを作り、それぞれの残差ストリーム活性差を平均して steering vector を作る。

形式的には、データセット `D` が `(prompt p, positive completion c_p, negative completion c_n)` の三つ組からなるとき、レイヤー `L` の Mean Difference ベクトルは次のように定義される。

```text
v_MD = (1/|D|) Σ_{p,c_p,c_n∈D} [a_L(p,c_p) - a_L(p,c_n)]
```

ここで `a_L()` はレイヤー `L` の残差ストリーム活性である。

重要なのは、positive / negative のプロンプトをほぼ同一にし、最後の answer letter だけを変えることで、トピック・文体・表層語彙などの交絡をキャンセルし、対象行動に対応する方向を抽出しようとしている点である。

## 1.3 CAA の適用方法

推論時には、ユーザープロンプトの後の token positions に対して、抽出した steering vector を倍率付きで加える。

```text
h_l ← h_l + α v
```

- `α > 0`: 対象行動を増やす方向。
- `α < 0`: 対象行動を減らす方向。
- 論文では、ユーザー prompt 後の全 token positions に加算する。

## 1.4 対象モデルと対象行動

主に Llama 2 Chat 系モデルで評価されている。

- Llama 2 7B Chat
- Llama 2 13B Chat
- 一部では base model と chat model の比較も行われる。

対象行動は alignment 関連の7カテゴリ。

- AI Coordination
- Corrigibility
- Hallucination
- Myopic Reward
- Survival Instinct
- Sycophancy
- Refusal

## 1.5 レイヤー選択に関する重要結果

複数レイヤーで steering vector を抽出し、`α=+1` と `α=-1` の効果を測定した結果、Llama 2 7B Chat では layer 13 付近、Llama 2 13B Chat では layer 14/15 付近が最も効果的だった。

EmotionEngine の多くの実験が `PRIMARY_LAYER = 13` を採用しているのは、この論文の結果と整合的である。layer 13 は、抽象的な振る舞い・感情方向が十分に形成され、かつ後段で出力へ影響できる中間〜後半層として自然な選択になっている。

## 1.6 Multiple-choice 評価の結果

論文では、held-out の multiple-choice behavioral questions に対して、CAAが7カテゴリすべてで行動の出現確率を上下させられることを示している。

- steering vector を加えると、対象行動の answer probability が増える。
- steering vector を引くと、対象行動の answer probability が減る。
- layer sweep の曲線は行動ごとに似たピークを持つ。

これは、行動に対応する表現が完全に個別タスク依存ではなく、似たレイヤー帯に抽象表現として現れている可能性を示す。

## 1.7 Open-ended generation 評価の結果

CAAは multiple-choice のような人工的設定だけでなく、open-ended generation にも転移する。

論文では GPT-4 を evaluator として使い、出力が対象行動をどの程度示すかを 1〜10 で採点している。

重要点:

- CAAは自由生成でも出力の傾向を変える。
- ただし倍率を大きくしすぎるとテキスト品質が落ちる。
- そのため、効果と品質のトレードオフが存在する。

この点は EmotionEngine の seed robustness 実験にも直結する。感情 steering でも `α_g` や `α_r` を大きくすると、感情方向は強まる一方で、文の自然さや意味保存が崩れる可能性がある。

## 1.8 System prompt との比較

論文は、CAA と system prompting を比較し、さらに両者を組み合わせた場合も評価している。

結果として、多くの行動で CAA は system prompt だけでは到達しない範囲まで行動を増減できる。これは、CAAが「プロンプトで明示的にお願いする」のではなく、内部表現を直接動かしているためだと解釈できる。

EmotionEngine にとっての示唆は大きい。ユーザーが言語化できない微妙な感情を、プロンプトで説明させるのではなく、UIノブで内部表現を操作するという構想は、この論文の「promptingとは別の制御経路がある」という結果と整合する。

## 1.9 Finetuning との比較

論文では supervised finetuning と CAA の比較も行っている。

- finetuning は held-out multiple-choice では高精度にできる。
- しかし open-ended generation への汎化では CAA の方が安定する場合がある。
- CAA は forward pass だけで作れるため、finetuning より計算コストが低い。
- finetuning 後に CAA を重ねると、さらに行動を調整できる場合がある。

EmotionEngine 的には、ユーザーごとに毎回 finetuning するより、抽出済みの感情方向を推論時に調整する方が実用的である。

## 1.10 一般能力への影響

MMLU と TruthfulQA で、CAAが一般能力に大きな悪影響を与えないことが確認されている。

- MMLU では、steering multiplier ±1 程度なら性能低下は大きくない。
- sycophancy vector を引くと TruthfulQA が少し改善する。

EmotionEngine では感情 rewrite が目的なので、能力保持というより意味保持・文法保持・自然さ保持が重要になるが、原論文の結果は「適切な倍率なら局所的な制御が可能」という基礎的な安心材料になる。

## 1.11 内部解釈の結果

論文では、steering vector と token activations の cosine similarity を見ることで、どのトークンに対象行動が現れているかを可視化している。

例:

- refusal vector は「I cannot help」「I strongly advise against」のような拒否表現で正の成分を持つ。
- myopic reward vector は「small one now」など即時報酬に関連する表現で正の成分を持つ。

これは EmotionEngine で local PC を「どんなニュアンス軸か」解釈する方針と近い。内部方向をただ steering に使うだけでなく、テキスト例・極端例・類似度可視化から意味を読むことができる。

## 1.12 レイヤー間・モデル間の転移

論文は、同じ行動の steering vector が近いレイヤー間で類似し、layer 13 で抽出した vector を他レイヤーで使ってもある程度効果が転移することを示す。

また、base model と chat model の steering vector 類似性も調べ、特に中間層で転移が見られる。

EmotionEngine では、layer 13 を固定して多数の解析を行っているが、これは合理的。ただし、感情方向がレイヤーごとにどう変化するかを調べる余地も残る。

## 1.13 限界

原論文が明示している限界:

- GPT-4 evaluator は便利だが、プロンプト依存やバイアスがある。
- finetuning baseline は完全に最適化されていない。
- prompting baseline も完全探索ではない。
- vector normalization の選択が layer optimality に影響しうる。
- steering は悪用可能である。

EmotionEngine 側でも、LLM-as-judge の評価、意味保存評価、感情判定評価、倍率スイープの安定性評価は注意深く設計する必要がある。

## 1.14 EmotionEngine への直接的な示唆

この論文から得られる設計原理:

1. 感情方向は、contrastive activation difference として抽出できる可能性がある。
2. layer 13 付近は有望な介入点。
3. steering multiplier は連続制御ノブとして使える。
4. prompt だけでは難しい微妙な制御を、内部方向で補助できる。
5. ただし大きすぎる介入は品質劣化を起こす。
6. CAA は絶対的な出力生成器ではなく、既存文脈の上に作用するベクトル操作として考えるべき。

---

# 2. `caa_geometry.pdf` — CAA Emotion Direction Geometry

## 2.1 実験の目的

このPDFは、8感情のグローバルCAAベクトルそのものの幾何を調べる実験である。

仮説は、LLMが感情を8つの離散カテゴリとして持っているのではなく、低次元の連続的・線形的な多様体上に表現しており、`joy` や `fear` はその多様体上の prototype region に人間が名前を付けたものにすぎない、というもの。

この実験では個別ソースではなく、まず `caa_pooled`、つまり感情ごとに平均された8本のグローバルCAA方向を分析している。

## 2.2 入力データ

入力は以下。

```text
activation/emotion_rewrites/caa_emotion_directions.npz
```

含まれる主要配列:

- `caa`: shape `(8, 3, 6, 4096)`
  - 8感情 × 3強度 × 6レイヤー × hidden dimension
  - 強度別・単位正規化済み CAA
- `caa_pooled`: shape `(8, 6, 4096)`
  - 強度をプールした感情ごとの CAA
- `delta_mean`: shape `(8, 3, 6, 4096)`
  - 単位正規化前の raw mean difference

感情順序:

```text
['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
```

レイヤー:

```text
[8, 10, 13, 16, 19, 22]
```

主分析レイヤーは `PRIMARY_LAYER = 13`。

## 2.3 なぜ残差ストリームではなく CAA か

残差ストリームの感情中心点を直接見ると、文体・語彙・トピックなどソーステキスト成分が混ざる。そのため以前の分析では source-centering が必要だった。

一方 CAA は定義上、

```text
mean_n(h_emotion[n] - h_neutral[n])
```

なので、各ソースの neutral baseline を引くことで、ソース依存成分がかなりキャンセルされている。そのため、より直接的に「感情変化方向」を見られる。

## 2.4 SVD / Participation Ratio の結果

8感情の `caa_pooled` ベクトルを行列として SVD し、有効次元を participation ratio で評価している。

結果:

| Layer | PR | EVR PC1 | EVR PC2 | EVR PC3 | EVR PC1-2 | EVR PC1-3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | NaN | NaN | NaN | NaN | NaN | NaN |
| 10 | 3.8789 | 0.4066 | 0.2309 | 0.1596 | 0.6374 | 0.7970 |
| 13 | 3.6149 | 0.4370 | 0.2321 | 0.1433 | 0.6691 | 0.8124 |
| 16 | 3.3299 | 0.4750 | 0.2194 | 0.1302 | 0.6944 | 0.8246 |
| 19 | 3.3285 | 0.4774 | 0.2092 | 0.1389 | 0.6866 | 0.8255 |
| 22 | 3.3675 | 0.4707 | 0.2098 | 0.1500 | 0.6805 | 0.8305 |

解釈:

- PR は 3.3〜3.9 程度で、8方向が完全独立しているわけではない。
- PC1だけで約40〜48%、PC1-2で約64〜69%、PC1-3で約80〜83% を説明する。
- これは、8感情CAAの幾何が比較的低次元に圧縮されていることを示す。
- ただし PR≈2 ではなく、3〜4程度なので、単純な2D circumplex だけでは足りない可能性が高い。

## 2.5 感情間 cosine similarity heatmap の結果

Layer 13 の感情間 cosine は、全ペアで非常に高い。

例:

- joy-trust: 0.992
- joy-sadness: 0.984
- fear-sadness: 0.993
- sadness-disgust: 0.995
- disgust-anger: 0.998

全体として、ほぼすべての感情ペアが `0.98〜0.998` 程度である。

これは非常に重要で、raw な CAA vector は「joy方向」「sadness方向」としてかなり違うものではなく、ほとんど同じ大方向を向いていることを意味する。

この結果だけを見ると、Plutchik のような対立構造は見えない。むしろ最大成分は「ニュートラル文を感情的にする」共通方向であり、個別感情の差はその上に乗る小さな残差として存在すると考えられる。

## 2.6 Plutchik opposite / adjacent pair の結果

Plutchik の対立ペア:

- joy ↔ sadness
- trust ↔ disgust
- fear ↔ anger
- anticipation ↔ surprise

Layer 13 での opposite cosine:

| Pair | Cosine |
|---|---:|
| joy↔sadness | 0.9843 |
| trust↔disgust | 0.9808 |
| fear↔anger | 0.9894 |
| anticipation↔surprise | 0.9801 |
| Mean | 0.9837 |

Adjacent pair の平均:

```text
Mean adjacent cosine = 0.9884
```

解釈:

- opposite は adjacent よりわずかに低い。
- しかし両方とも非常に高い正の値であり、raw CAA空間では「反対方向」にはなっていない。
- つまり、Plutchik的な双極構造は raw CAA では共通 emotionalization 方向に隠れている。

この結果が、後続の `caa_common_direction_residualization.pdf` の動機になっている。

## 2.7 2D/3D PCA 可視化

Layer 13 の PCA では:

```text
Top-3 EVR: 0.437, 0.232, 0.143 (sum = 0.812)
```

PC1-PC2, PC1-PC3 上で8感情をプロットすると、ある程度の配置差は見えるが、raw cosine がほぼ全て高いため、原点から見た方向差は小さい。

観察:

- joy/trust は近い。
- disgust/anger も近い。
- sadness/fear/disgust/anger は同じ側に寄りやすい。
- surprise は PC2/PC3 でやや独立した位置を持つ。
- anticipation は joy/trust 側に近いが、完全には同一ではない。

ただし、これはあくまで共通方向を含んだ空間なので、感情固有構造を評価するには残差化が必要。

## 2.8 強度別CAAベクトルの整合性

PDF後半では low/medium/high の各強度における同一感情CAAベクトルの方向整合性を評価している。

目的は、感情方向が強度に依存せず安定しているかを見ること。

解釈の軸:

- low/medium/high が同じ方向を向くなら、強度は方向ではなく長さ・倍率・共通方向成分として表現されている可能性がある。
- 強度ごとに方向が大きく変わるなら、感情の質そのものが強度で変質している可能性がある。

この実験の設計は、EmotionEngine の UI における「強度ノブ」と「感情タイプノブ」を分離できるかに関係する。

## 2.9 このPDFの結論

`caa_geometry.pdf` の中心結論は以下。

1. 8感情のCAAは完全な8独立方向ではなく、PR 3〜4程度の低次元構造を持つ。
2. しかし raw cosine はほぼ全ペアで0.98以上であり、最大成分は感情固有ではなく共通 emotionalization 方向である。
3. Plutchik の対立ペアは raw CAA では反対方向にならない。
4. したがって、感情固有構造を見るには、共通方向 `g` を除去する必要がある。
5. このPDFは、後続の「common direction residualization」実験の強い前提を提供している。

---

# 3. `caa_common_direction_residualization.pdf` — Experiment A

## 3.1 実験の目的

Experiment A は、グローバルCAAベクトルに共通して含まれる「neutral → emotional」方向 `g` を除去したときに、感情固有の幾何が現れるかを検証する実験である。

前の `caa_geometry.pdf` では、8感情の global CAA vectors が互いにほぼ同じ方向を向いていることが示された。つまり raw CAA では、joy-specific / sadness-specific な差よりも、文章を感情的にする共通成分が支配的である。

この実験は、その共通成分を明示的に取り除く。

## 3.2 方法

各感情の global pooled CAA vector を `c_e` とする。

1. 共通 emotionalization 方向を計算。

```text
g = c̄ / ||c̄||, where c̄ = (1/|E|) Σ_e c_e
```

2. 各感情方向から `g` への射影を除去。

```text
c_e^resid = c_e - (c_e · g) g
```

3. cosine geometry のために再度単位正規化。

```text
ĉ_e^resid = c_e^resid / ||c_e^resid||
```

## 3.3 何が証拠になるか

感情固有構造が存在するなら、残差化後に次が起きるはず。

- pairwise cosine の多様性が増える。
- PR が増える、または構造がより明瞭になる。
- Plutchik opposite pairs が adjacent pairs より低い cosine を持つ。
- PCA投影で感情がより広がる。

逆に、すべての残差がほぼゼロなら、感情差はほぼ存在せず、raw CAA はただの emotionalization 方向だったことになる。

## 3.4 入力と設定

入力:

```text
activation/emotion_rewrites/caa_emotion_directions.npz
```

使用データ:

- `CAA_POOLED`: shape `(8, 6, 4096)`
- `DELTA_MEAN`: shape `(8, 3, 6, 4096)`

主レイヤー:

```text
PRIMARY_LAYER = 13
```

## 3.5 Cosine heatmap: before / after

Layer 13 の平均 off-diagonal cosine:

```text
Original: 0.9856
Residualized: -0.1365
```

これは非常に大きな変化である。

元の空間では全感情方向がほぼ同じ方向を向いていた。残差化後は、平均的にはやや負の cosine になり、感情間の方向差が大きくなった。

## 3.6 残差化後の heatmap の特徴

Layer 13 の残差化後 cosine には、明確な正負の構造が現れる。

例:

- joy-trust: +0.505
- joy-fear: -0.599
- joy-sadness: -0.439
- joy-disgust: -0.582
- joy-anger: -0.570
- trust-disgust: -0.513
- disgust-anger: +0.812
- sadness-disgust: +0.403
- sadness-anger: +0.250
- anticipation-disgust: -0.689
- anticipation-anger: -0.574

解釈:

- joy/trust は近い。
- disgust/anger は非常に近い。
- joy/trust と disgust/anger はかなり反対側にある。
- fear は joy とは反対寄りだが、anger とは強い正になっていない。
- Plutchikに完全一致ではないが、raw CAAでは見えなかった構造がかなり見える。

## 3.7 SVD / PR before / after

PR と EVR1 の結果:

| Layer | PR original | PR residualized | EVR1 original | EVR1 residualized |
|---:|---:|---:|---:|---:|
| 8 | NaN | NaN | NaN | NaN |
| 10 | 3.8789 | 3.8751 | 0.4066 | 0.4195 |
| 13 | 3.6149 | 3.6301 | 0.4370 | 0.4483 |
| 16 | 3.3299 | 3.4441 | 0.4750 | 0.4705 |
| 19 | 3.3285 | 3.5542 | 0.4774 | 0.4570 |
| 22 | 3.3675 | 3.6612 | 0.4707 | 0.4429 |

解釈:

- PR は大きくは変わらない。
- ただし後半レイヤーでは residualized の PR がやや増える。
- これは、元々の低次元性がすべて `g` だけによるものではなく、感情固有残差にも3〜4次元程度の構造があることを示す。
- PR が劇的に増えないのは、感情固有構造もまた低次元であるためと考えられる。

## 3.8 Plutchik pair analysis

平均 cosine の結果:

### Original CAA

| Layer | Adjacent mean | Opposite mean |
|---:|---:|---:|
| 10 | 0.9874 | 0.9835 |
| 13 | 0.9884 | 0.9837 |
| 16 | 0.9895 | 0.9848 |
| 19 | 0.9916 | 0.9878 |
| 22 | 0.9936 | 0.9909 |

raw CAA では adjacent と opposite の差は非常に小さい。

### After g-residualization

| Layer | Adjacent mean | Opposite mean |
|---:|---:|---:|
| 10 | 0.0581 | -0.2571 |
| 13 | 0.0898 | -0.3060 |
| 16 | 0.1018 | -0.3107 |
| 19 | 0.1038 | -0.3172 |
| 22 | 0.0925 | -0.2986 |

残差化後は、opposite pairs が明確に負になり、adjacent pairs は弱い正になる。

これは非常に重要な結果で、Plutchik 的な対立構造は raw CAA には見えないが、共通 emotionalization 成分を引くことで出現する。

## 3.9 2D PCA before / after

2D PCA でも、残差化前は感情が共通方向に押しつぶされて見えにくい。一方、残差化後は感情間の配置が広がり、対立ペアを結ぶ線分もより意味を持つ。

ただし、残差化後も完全な円環・完全なPlutchik配置ではない。EmotionEngine では、Plutchikを厳密な地図として使うより、「初期ラベル・評価の参照枠」として使うのが妥当。

## 3.10 Layer-wise cosine diversity

この実験は、全レイヤーで cosine diversity を比較している。

大きな傾向:

- layer 8 は数値が異常または未形成で、構造が見えない。
- layer 10 以降で raw CAA は高い平均 cosine を持つ。
- layer 10 以降で residualized cosine は正負に広がる。
- opposite / adjacent の分離は layer 13〜19 あたりで安定している。

## 3.11 結論

Experiment A の結論:

1. 8感情CAAの最大成分は共通 emotionalization 方向 `g` である。
2. `g` を除去すると、感情固有の相対構造がかなり明瞭になる。
3. Plutchik opposite pairs は残差空間で負の cosine を持ち、adjacent pairs より明確に遠くなる。
4. ただし残差空間も完全な8独立方向ではなく、PR 3〜4程度の低次元構造を保つ。
5. EmotionEngine では、`g` を「感情化の強さ」、`r_e` を「感情タイプ・ニュアンス方向」として分ける設計が有望である。

---

# 4. `caa_residualized_per_source.pdf` — Experiment B

## 4.1 実験の目的

Experiment B は、Experiment A の residualization を個別ソースごとの CAA ベクトルにも適用し、感情ラベルの decodability が改善するかを検証する。

ベースラインの per-source analysis では、個別CAAから感情ラベルは chance よりは予測できるが弱かった。理由として、各個別 `δ[n,e]` に共通 emotionalization 方向 `g` が強く含まれ、感情固有差を隠している可能性がある。

この実験の中心質問:

```text
共通方向 g を各 δ[n,e] から取り除くと、emotion label decodability は改善するか？
```

## 4.2 方法

1. per-source deltas をロード。

```text
δ[n,e] = h_emotion[n,e] - h_base[n]
```

ここでは強度方向を平均する。

2. global pooled CAA から共通方向を計算。

```text
g = unit(c̄), c̄ = (1/|E|) Σ_e c_e
```

3. 個別CAAを residualize。

```text
δ_resid[n,e] = δ[n,e] - (δ[n,e] · g)g
```

4. original と residualized の両方で per-source analysis を再実行。

## 4.3 入力と設定

入力:

- `emotion_intensity_residual_stream.npy`: shape `(22782, 8, 3, 6, 4096)`
- `neutral_paraphrase_residual_stream.npy`: shape `(22782, 6, 4096)`
- `caa_emotion_directions.npz`

設定:

- `PRIMARY_LAYER = 13`
- `MAX_SOURCES = 1200`
- `DIMS_TO_TEST = [2, 4, 8, 16, 32, 64]`
- source-held-out `GroupKFold`
- classifier: logistic regression
- dimensionality reduction: memory-safe `TruncatedSVD`
- chance level: `1/8 = 0.125`

## 4.4 Per-source consistency before / after residualization

各 source/emotion の個別CAAが global CAA とどれだけ揃うかを cosine で測る。

### Original consistency mean

| Emotion | Mean cosine |
|---|---:|
| joy | 0.8526 |
| trust | 0.8512 |
| fear | 0.8510 |
| surprise | 0.8538 |
| sadness | 0.8541 |
| disgust | 0.8513 |
| anger | 0.8519 |
| anticipation | 0.8553 |

### Residualized consistency mean

| Emotion | Mean cosine |
|---|---:|
| joy | 0.2038 |
| trust | 0.2129 |
| fear | 0.1690 |
| surprise | 0.2250 |
| sadness | 0.1434 |
| disgust | 0.1585 |
| anger | 0.1801 |
| anticipation | 0.1842 |

解釈:

- original ではすべての感情が約0.85で global CAA と強く一致する。
- これは個別CAAがソースをまたいで安定している証拠であり、平均CAAに意味があることを示す。
- ただしこの強い一致の大部分は共通 `g` による可能性がある。
- residualized 後は cosine が0.14〜0.23に下がる。これは、感情固有残差は共通方向よりずっと小さく、個別ソースごとの揺らぎに埋もれやすいことを示す。

## 4.5 Emotion label classification: original vs residualized

source-held-out GroupKFold で、個別 `δ[n,e]` から emotion label を予測する。

結果:

| Dims | Original BAcc | Residualized BAcc |
|---:|---:|---:|
| 2 | 0.1331 ± 0.0025 | 0.1323 ± 0.0022 |
| 4 | 0.1370 ± 0.0022 | 0.1500 ± 0.0039 |
| 8 | 0.2414 ± 0.0061 | 0.3165 ± 0.0049 |
| 16 | 0.5194 ± 0.0150 | 0.5241 ± 0.0119 |
| 32 | 0.7370 ± 0.0084 | 0.7474 ± 0.0093 |
| 64 | 0.8215 ± 0.0094 | 0.8233 ± 0.0112 |

解釈:

- 2〜4次元では chance 付近。
- 8次元では residualized が明確に改善する。
- 16次元以上では original と residualized の差は小さい。
- 64次元では両方とも約0.82まで上がる。

つまり、共通方向を除去すると低〜中次元で感情固有信号が見えやすくなるが、高次元では元の特徴量にも十分な情報が含まれている。

## 4.6 2D visualization

個別 per-source delta を2Dに射影し、original と residualized を比較している。

目的:

- residualized space で emotion clouds がより分離するかを見る。

結果の読み方:

- original では共通方向が支配的で、感情雲の分離が見えにくい。
- residualized では感情別の重心差がやや見えやすくなる。
- ただし2Dだけでは完全分離はしない。

これは分類結果とも一致する。感情差は存在するが、2Dだけでは不足し、8〜64次元程度が必要。

## 4.7 Confusion matrix

residualized condition で最も良かった次元は:

```text
best dims = 64
balanced accuracy = 0.8233
```

64次元では、感情ラベルをかなり高精度に分類できる。ただし完全ではない。混同行列を見る目的は、どの感情が近いか・混同されやすいかを調べること。

想定される混同:

- sadness / disgust / anger など負感情側の近接。
- joy / trust / anticipation など正・期待側の近接。
- surprise は文脈によって positive/negative の両側に出る可能性。

## 4.8 Layer-wise classification sweep

固定次元 `16` で、各レイヤーに対して original / residualized の classification を比較している。

目的:

- layer 13 の選択が妥当か。
- residualization の効果がレイヤー依存か。

大きな読み方:

- layer 8 は構造が弱い/未形成。
- layer 10 以降で感情decodability が上がる。
- layer 13〜16あたりが良い候補。
- residualization は特に中間層で低次元decodabilityを改善しやすい。

## 4.9 結論

Experiment B の結論:

1. 個別CAAは original では global CAA と非常に強く揃う。
2. その一致の多くは共通 emotionalization 方向 `g` で説明できる。
3. `g` を取り除いても emotion-specific signal は残る。
4. residualization は特に8次元程度の低次元分類で有効。
5. 高次元では original でも residualized でも emotion label は高精度に分類可能。
6. EmotionEngine では、`g` と residual emotion directions を分ける設計は支持されるが、感情固有成分は小さく、UI操作では倍率・正規化・評価が重要になる。

---

# 5. `caa_per_source_structure.pdf` — CAA Per-Source Emotion Direction Structure

## 5.1 実験の目的

このPDFは、各ソーステキストごとに独立に計算した個別CAAベクトル `h_emotion[n] - h_neutral[n]` を分析し、感情方向の一貫性・感情内部の下位構造・感情間の重なりを調べる。

仮説:

- 感情は単なるラベル付きカテゴリではなく、連続的な変化方向として表現される。
- 各ソースごとのCAAが同じ感情で似た方向を向くなら、感情方向は安定している。
- しかし個別CAA分布に広がりがあるなら、各感情には内部構造・サブタイプがある。

## 5.2 入力と設定

入力:

```text
activation/emotion_rewrites/emotion_intensity_residual_stream.npy
activation/emotion_rewrites/neutral_paraphrase_residual_stream.npy
activation/emotion_rewrites/caa_emotion_directions.npz
```

テンソルサイズ:

- `EMO_SHAPE = (22782, 8, 3, 6, 4096)`
- `NEUTRAL_SHAPE = (22782, 6, 4096)`

設定:

- `PRIMARY_LAYER = 13`
- `MAX_SOURCES = 2500`
- `SEED = 0`

メモリ対策として memmap とソースサブサンプリングを使用。

## 5.3 個別CAAの定義

各ソース `n`、感情 `e`、強度 `i`、レイヤー `l` で:

```text
delta[n,e,i,l] = h_emotion[n,e,i,l] - h_neutral[n,l]
```

分析では多くの場合、強度方向を平均して:

```text
delta_pooled[n,e] = mean_i delta[n,e,i]
```

を使う。

## 5.4 感情方向の一貫性

個別CAA `delta[n,e]` を単位正規化し、global CAA `c_e` との cosine を計算。

Layer 13 の結果:

| Emotion | Mean | Std | Min | 25% | Median | 75% | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| joy | 0.8551 | 0.0550 | 0.5813 | 0.8292 | 0.8723 | 0.8937 | 0.9324 |
| trust | 0.8536 | 0.0542 | 0.5858 | 0.8292 | 0.8706 | 0.8915 | 0.9336 |
| fear | 0.8538 | 0.0539 | 0.5831 | 0.8301 | 0.8697 | 0.8914 | 0.9283 |
| surprise | 0.8555 | 0.0558 | 0.5575 | 0.8288 | 0.8740 | 0.8947 | 0.9320 |
| sadness | 0.8567 | 0.0545 | 0.5825 | 0.8309 | 0.8739 | 0.8945 | 0.9363 |
| disgust | 0.8540 | 0.0548 | 0.5699 | 0.8302 | 0.8708 | 0.8920 | 0.9320 |
| anger | 0.8548 | 0.0545 | 0.5721 | 0.8295 | 0.8717 | 0.8932 | 0.9351 |
| anticipation | 0.8570 | 0.0545 | 0.5838 | 0.8318 | 0.8739 | 0.8955 | 0.9330 |

解釈:

- 全感情で mean ≈ 0.85。
- 分布もかなり狭く、median は約0.87。
- これは、個別ソースのCAAがグローバル方向とかなり安定して揃うことを示す。
- つまり、CAA平均はノイズだらけの平均ではなく、ソースをまたいだ安定方向である。

## 5.5 感情ごとのローカルPCA

各感情 `e` について、個別CAA集合 `{delta[n,e]}` にPCAを適用し、感情内部の変動軸を調べる。

結果:

| Emotion | PR | EVR PC1 | EVR PC1-2 | EVR PC1-3 | cos(PC1, globalCAA) |
|---|---:|---:|---:|---:|---:|
| joy | 3.6117 | 0.1224 | 0.1804 | 0.2197 | 0.2520 |
| trust | 3.5609 | 0.1179 | 0.1747 | 0.2131 | 0.2539 |
| fear | 3.6037 | 0.1157 | 0.1717 | 0.2096 | 0.2564 |
| surprise | 3.5431 | 0.1217 | 0.1796 | 0.2180 | 0.2534 |
| sadness | 3.6908 | 0.1176 | 0.1750 | 0.2137 | 0.2583 |
| disgust | 3.7339 | 0.1149 | 0.1717 | 0.2108 | 0.2525 |
| anger | 3.6808 | 0.1159 | 0.1727 | 0.2117 | 0.2575 |
| anticipation | 3.5900 | 0.1187 | 0.1764 | 0.2150 | 0.2498 |

解釈:

- 各感情の内部PRは約3.5〜3.7。
- PC1だけで説明される分散は約11〜12%。
- PC1-3でも約21〜22%程度。
- つまり、各感情内部の変動は単一軸では説明できない。
- `cos(local PC1, global CAA)` は約0.25で、local PC1 は global CAA 方向そのものではない。

これは非常に重要で、名前付き感情は単一方向ではなく、内部に複数のニュアンス軸を持つことを示す。

## 5.6 感情「コーン」の2D可視化

全8感情の global CAA から作った2D PCA basis に、個別CAAを投影している。

観察:

- 各感情の個別CAAは、global centroid の周りに「コーン」状に広がる。
- ただしコーンはかなり重なる。
- joy/trust/anticipation 側、sadness/disgust/anger 側などの大まかな配置はありそう。
- しかし感情ごとに明確な独立クラスタがあるというより、連続的に重なった分布である。

解釈:

- 感情方向は安定しているが、ソース文脈によって大きく揺れる。
- この揺れこそが「名前にしにくい微妙な感情」の候補になる。
- EmotionEngine の UI では、global emotion direction だけでなく local PC 軸や residual 軸をノブにできる可能性がある。

## 5.7 感情間境界: k-NN / logistic classification

個別CAAベクトルから感情ラベルを分類する実験。

低次元 PCA に射影後の balanced accuracy:

| Dims | Balanced Accuracy |
|---:|---:|
| 2 | 0.2569 ± 0.0029 |
| 4 | 0.2691 ± 0.0038 |
| 8 | 0.2810 ± 0.0057 |
| 16 | 0.3044 ± 0.0029 |
| 32 | 0.3337 ± 0.0035 |
| 64 | 0.3629 ± 0.0019 |

解釈:

- chance は 0.125 なので、感情情報は確実にある。
- しかし 64次元でも 0.36 程度にとどまる。
- これは Experiment B の 0.82 と比べると低いが、原因として、このノートブックでは global basis / PCA設計 / label ordering / y構築の違いなどがありうる。
- 低次元可視化では感情分布の重なりがかなり大きいことは明確。

## 5.8 結論

このPDFの結論:

1. 個別CAAは global CAA と強く揃い、感情方向はソース非依存の安定軸を持つ。
2. ただし各感情内部にはPR約3.5〜3.7のサブ構造がある。
3. 各感情は細い1本の線ではなく、広がりを持つ「コーン」として表現される。
4. 感情間分布は重なりが大きく、8感情を完全に離散クラスタとして扱うのは不適切。
5. EmotionEngine の「DJノブ」構想には、global emotion axis だけでなく、感情内部の local PC 軸が重要になる。

---

# 6. `caa_manifold_linearity.pdf` — CAA Linear Manifold Continuity Test

## 6.1 実験の目的

このPDFは、CAA感情空間が線形・連続的な多様体として扱えるかを検証する。

中心仮説:

```text
LLM内部の感情表現は、8つの離散点ではなく、線形・連続的な多様体上に存在する。
```

ここでは、グローバルCAAベクトルを使って、感情間遷移・推移性・補間・円環モデル・レイヤー間安定性を調べる。

## 6.2 入力と設定

入力:

```text
activation/emotion_rewrites/caa_emotion_directions.npz
```

使用配列:

- `CAA_POOLED`: shape `(8, 6, 4096)`, unit-normed
- `DELTA_MEAN`: shape `(8, 3, 6, 4096)`, raw magnitude

主レイヤー:

```text
PRIMARY_LAYER = 13
```

## 6.3 感情間遷移ベクトルの定義

CAAベクトル `caa(A)` は「neutral から感情Aへ」の方向。したがって感情Aから感情Bへの遷移は:

```text
v_{A→B} = caa(B) - caa(A)
```

と定義する。

Layer 13 では遷移テンソル shape は:

```text
(8, 8, 4096)
```

例:

```text
||v(joy→sadness)|| = 0.1770
```

raw CAA の cosine が非常に高いため、感情間遷移ベクトルのノルムは比較的小さい。

## 6.4 対称性テスト

理論的には:

```text
v(A→B) = -v(B→A)
```

これは定義上必ず成り立つ。

実験では raw `delta_mean` を使い、方向とノルムの対称性を確認している。

Layer 13 の感情ごとの raw magnitude:

| Emotion | Norm |
|---|---:|
| sadness | 9.1179 |
| joy | 9.1129 |
| fear | 9.1070 |
| anticipation | 9.1070 |
| surprise | 9.1058 |
| trust | 9.0815 |
| anger | 9.0661 |
| disgust | 9.0466 |

対称性:

```text
Mean cos(v_AB, -v_BA) = 1.000000
```

これはコードと定義の整合性確認として機能する。

## 6.5 推移性テスト

線形空間なら:

```text
v(A→B) + v(B→C) = v(A→C)
```

ただしこれはユークリッド差分では自明なので、実験では unit-normalized transition vectors を使い、方向的推移性を評価している。

結果:

| Layer | Mean | Std | Min | Max |
|---:|---:|---:|---:|---:|
| 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10 | 0.9717 | 0.0376 | 0.7925 | 1.0000 |
| 13 | 0.9677 | 0.0391 | 0.7816 | 1.0000 |
| 16 | 0.9639 | 0.0403 | 0.7804 | 1.0000 |
| 19 | 0.9621 | 0.0431 | 0.7433 | 1.0000 |
| 22 | 0.9622 | 0.0429 | 0.7477 | 1.0000 |

解釈:

- layer 10以降では方向的推移性が非常に高い。
- layer 13 で mean ≈ 0.968。
- 感情間の相対配置はかなり線形的・加法的に振る舞う。
- ただし bottom triplets では 0.78 程度まで落ちるものもあり、完全な線形空間ではない。

Layer 13 の least transitive triplets には次が含まれる。

- trust→anger→disgust: 0.781592
- disgust→anger→trust: 0.781592
- joy→anger→disgust: 0.809277
- disgust→anger→joy: 0.809277

このあたりは、anger/disgust/trust/joy の関係が単純な直線的推移だけでは表しにくいことを示す。

## 6.6 線形補間テスト

感情AとBの中間点:

```text
mid = unit(α caa(A) + (1-α) caa(B))
```

を作り、最近傍感情がどう変わるかを見る。

### Plutchik dyad midpoint test

Plutchik の複合感情ペアについて、中点の最近傍を調べた。

例:

| Dyad | Pair | Nearest | Nearest excluding sources |
|---|---|---|---|
| love | joy+trust | trust | anticipation |
| submission | trust+fear | trust | anticipation |
| awe | fear+surprise | fear | sadness |
| disapproval | surprise+sadness | sadness | disgust |
| remorse | sadness+disgust | sadness | anger |
| contempt | disgust+anger | disgust | sadness |
| aggressiveness | anger+anticipation | anger | sadness |
| optimism | anticipation+joy | anticipation | trust |

解釈:

- 中点は多くの場合、元の2感情のどちらかに最も近い。
- これは raw CAA directions が互いに非常に近いため、中点が独立した第三の感情方向として明確に現れにくいから。
- source emotions を除外すると、それらに近い第三感情が出るが、Plutchik の複合感情と厳密対応するわけではない。

## 6.7 Interpolation switch points

例として、joy→sadness、fear→anger、joy→fear、trust→disgust の補間で最近傍がどこで切り替わるかを調べている。

ただし注意点として、実装上 `α=0` が `B`、`α=1` が `A` に対応しているため、表示文の「α=0 is joy」などとは逆に見える箇所がある。解釈時には式を優先する必要がある。

観察:

- 最近傍は多くの場合、片方の端点からもう片方へ急に切り替わる。
- joy→fear の中点付近では anticipation が最近傍になるなど、中間感情らしき挙動も一部ある。
- ただし、補間点をそのまま「新しい名前付き感情」とみなすには不十分。

## 6.8 Circumplex model fit

2D PCA 投影が円状に並ぶかを検証。

Layer 13 の 2D PCA:

```text
PC1 EVR = 0.437
PC2 EVR = 0.232
Total = 0.669
```

半径統計:

```text
mean radius = 0.0892
std = 0.0213
cv = 0.2390
```

角度順序:

```text
joy: -160.9°
surprise: -80.7°
anger: +2.9°
disgust: +4.8°
sadness: +20.4°
fear: +61.1°
anticipation: +147.4°
trust: +169.5°
```

角度ギャップ:

```text
[80.2, 83.6, 1.8, 15.7, 40.6, 86.3, 22.1, 29.6]
std = 31.50°
```

解釈:

- 2Dで約67%を説明するので、2D circumplex 的成分はある。
- しかし角度間隔は理想の45°からかなりズレる。
- anger/disgust/sadness が近くに固まり、joy/trust も近い。
- 完全なPlutchik円環ではない。

## 6.9 Circle fitting

2D点に円を最小二乗フィット。

結果:

```text
center = (-0.0100, -0.0278)
radius = 0.0963
mean residual = 0.0077
max residual = 0.0228
```

解釈:

- 点はある程度円に乗る。
- ただし等角間隔ではない。
- 「円環的配置」は部分的には支持されるが、EmotionEngine の内部感情空間は単純な2D円ではなく、3〜4D程度の歪んだ低次元多様体と見る方がよい。

## 6.10 全ペア遷移ベクトルのレイヤー間安定性

全28ペアの transition direction をレイヤー間で比較。

例:

```text
Mean transition direction consistency with layer 8: 0.0000
Mean transition direction consistency with layer 13: 0.3747
Mean transition direction consistency with layer 22: 0.3795
```

解釈:

- layer 8 は構造が形成されていない。
- layer 13 と後半レイヤーで、相対方向にある程度の安定性がある。
- ただし cosine 0.37 程度なので、遷移方向は完全にレイヤー不変ではない。
- 感情空間の幾何は層をまたいで変形する。

## 6.11 結論

このPDFの結論:

1. 感情間遷移は定義上対称性を持つ。
2. layer 10以降では方向的推移性が非常に高く、感情空間はかなり線形的に振る舞う。
3. ただし補間点はPlutchik複合感情と単純対応しない。
4. 2D circumplex は部分的に成り立つが、等角円環ではない。
5. 感情空間は「2D円」より「3〜4D程度の歪んだ低次元線形多様体」と見るのがよい。
6. EmotionEngine では、単純な valence-arousal 平面だけでなく、追加の局所軸・残差軸をUIに含める価値がある。

---

# 7. `caa_local_pc_interpretation.pdf` — Experiment D

## 7.1 実験の目的

Experiment D は、各感情内部の local PCA 軸を解釈可能にするための実験である。

これまでの分析で、各 named emotion は単一方向ではなく、participation ratio 3〜5程度の内部構造を持つことがわかった。そこでこの実験では、各 local PC の極端例を取り出し、人間またはLLMが読める形で保存する。

目的は、local PC を EmotionEngine の「DJノブ」として使えるかを判断すること。

例として期待される軸:

- calm joy vs excited joy
- private sadness vs social grief
- anger as irritation vs moral outrage
- fear as uncertainty vs immediate threat
- anticipation as hopeful planning vs anxious expectation

ただし、これらは仮説であり、実験では事前に決め打ちしない。

## 7.2 入力と設定

入力:

- `emotion_intensity_residual_stream.npy`
- `neutral_paraphrase_residual_stream.npy`
- `caa_emotion_directions.npz`
- `emotion_rewrites.jsonl`

設定:

```text
PRIMARY_LAYER = 13
K_LOCAL = 8
N_EX = 6
MAX_SOURCES = 3000
SEED = 42
RUN_RESIDUALIZED = True
```

感情順序:

```text
['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
```

使用ソース数:

```text
3000 / 22782 sources
D = 4096
```

## 7.3 テキストレジストリ

この実験では、数値ベクトルだけでなく、元の rewrite text を参照する。

構築されるレジストリ:

- `(source_id, emotion, intensity) → emotion rewrite`
- `source_id → base_text`

サイズ:

```text
Registry size: 546768
Base text registry size: 22782
```

これにより、PC score の高い/低い例について、base text と rewrite を並べて読める。

## 7.4 Local PCA の方法

各感情 `e` について:

```text
X_e = {δ[n,e]}_n
```

を作り、PCAを実行する。

さらに residualized run では:

```text
δ_resid[n,e] = δ[n,e] - (δ[n,e] · g)g
```

を使う。

メモリ対策として、1感情ずつ streaming load し、`randomized_svd` を使っている。

## 7.5 Local PCA の結果

Original:

| Emotion | PR | EVR PC1 | EVR PC2 | EVR PC3 |
|---|---:|---:|---:|---:|
| joy | 5.05 | 0.123 | 0.055 | 0.041 |
| trust | 5.09 | 0.118 | 0.054 | 0.041 |
| fear | 5.18 | 0.116 | 0.053 | 0.041 |
| surprise | 5.00 | 0.122 | 0.054 | 0.042 |
| sadness | 5.22 | 0.117 | 0.054 | 0.042 |
| disgust | 5.32 | 0.114 | 0.054 | 0.042 |
| anger | 5.25 | 0.116 | 0.054 | 0.042 |
| anticipation | 5.04 | 0.119 | 0.055 | 0.042 |

Residualized:

| Emotion | PR | EVR PC1 | EVR PC2 | EVR PC3 |
|---|---:|---:|---:|---:|
| joy | 5.13 | 0.119 | 0.055 | 0.041 |
| trust | 5.17 | 0.113 | 0.053 | 0.039 |
| fear | 5.27 | 0.112 | 0.053 | 0.040 |
| surprise | 5.10 | 0.117 | 0.054 | 0.040 |
| sadness | 5.31 | 0.113 | 0.054 | 0.041 |
| disgust | 5.41 | 0.110 | 0.054 | 0.042 |
| anger | 5.34 | 0.111 | 0.054 | 0.041 |
| anticipation | 5.12 | 0.115 | 0.054 | 0.040 |

解釈:

- `caa_per_source_structure.pdf` では K=5 のため PR ≈ 3.5〜3.7だったが、本実験では K=8 を抽出しているため PR ≈ 5.0〜5.4になっている。
- 各感情のPC1は約11〜12%しか説明しない。
- PC2, PC3 もそれぞれ5%, 4%程度。
- これは、感情内部の変動が単一の強い軸ではなく、複数の小さな軸に分散していることを示す。
- residualization 後もPRは少し上がるが、大きく構造が変わるわけではない。

## 7.6 Extreme examples per PC

各感情・各PCについて、score が最も高い例と低い例を取り出す。

設定:

- top PCs: 4
- 各 pole の例数: 6
- original と residualized の両方

保存された extreme example records:

```text
768 records
```

内訳:

```text
2 conditions × 8 emotions × 4 PCs × 2 poles × 6 examples = 768
```

保存先:

- `local_pc_extreme_examples.csv`
- `local_pc_extreme_examples.json`

## 7.7 Human-readable markdown report

人間が読める report も生成している。

保存先:

- `local_pc_report_original_L13.md`
- `local_pc_report_residualized_L13.md`

各セクションには以下が含まれる。

- emotion name
- participation ratio
- 各PCのEVR
- TOP POSITIVE examples
- TOP NEGATIVE examples
- base text
- emotion rewrite
- PC score

このファイルは、実際に local PC の意味を読むための主出力である。

## 7.8 2D PC scatter per emotion

各感情について local PC1 vs PC2 の散布図を描き、PC1の上位/下位例を star marker で表示している。

観察:

- どの感情も、PC1方向に明確な広がりを持つ。
- ただし分布は複雑で、単純な2クラスタではない。
- top/bottom extreme examples は分布の端にあり、解釈対象として妥当。
- residualized でも似たような形状が残る。

## 7.9 LLM interpretation prompt

local PC を意味づけるため、LLMに渡す構造化 prompt JSON を生成している。

保存先:

- `llm_interpretation_prompt_original_L13.json`
- `llm_interpretation_prompt_residualized_L13.json`

各 prompt は 24 axes を含む。

```text
8 emotions × 3 PCs = 24 axes
```

LLMへの指示内容:

1. high examples と low examples を読む。
2. emotional / semantic / stylistic dimension を特定する。
3. 軸に短い名前を付ける。
4. high と low の違いを2〜3文で説明する。
5. confidence 1〜5 を付ける。
6. JSON形式で返す。
7. Plutchik予測に無理に合わせない。
8. ランダムなら正直にそう言う。

これは EmotionEngine の UI ノブ命名のための重要な中間成果物である。

## 7.10 Local PCs と common direction g の alignment

original local PCs が共通方向 `g` にどれだけ揃うかを測定している。

PC1 の `|cos(PC1, g)|`:

| Emotion | EVR | cos_with_g |
|---|---:|---:|
| joy | 0.1232 | 0.2285 |
| trust | 0.1176 | 0.2294 |
| fear | 0.1161 | 0.2296 |
| surprise | 0.1216 | 0.2381 |
| sadness | 0.1174 | 0.2313 |
| disgust | 0.1142 | 0.2320 |
| anger | 0.1155 | 0.2340 |
| anticipation | 0.1188 | 0.2259 |

解釈:

- local PC1 は `g` と多少は関係するが、強くは揃っていない。
- つまり local PC1 は単なる emotionalization intensity ではない。
- residualized run では `g` 成分が構成上取り除かれるため、より感情固有/文脈固有の軸が見えるはず。

## 7.11 結論

Experiment D の結論:

1. 各感情内部にはPR約5程度の多軸構造がある。
2. PC1だけで感情内部変動を代表することはできない。
3. local PC の極端例を読むことで、各軸を semantic / stylistic / affective knob として命名できる可能性がある。
4. common direction `g` と local PC1 の alignment は約0.23で、local PCは単なる感情強度軸ではない。
5. EmotionEngine の「名前にできない感情」を作るには、global emotion directions だけでなく local PCs を使うのが有望。
6. 次に重要なのは、LLM interpretation prompt の結果を人間がレビューし、本当にUIノブとして意味がある軸だけを採用すること。

---

# 8. `caa_seed_emotion_robustness.pdf` — Seed Emotion Robustness Test

## 8.1 実験の目的

このPDFは、分解された CAA steering が「絶対的な感情座標」として働くのか、それとも「文脈依存の感情変化ベクトル」として働くのかを検証する。

重要な問い:

```text
同じ target emotion に steering したとき、seed text の元の感情に関係なく同じような出力になるか？
それとも seed emotion に強く依存して、出力が変わるか？
```

## 8.2 背景

これまでの実験では、感情CAAを次の2成分に分けられることが示唆された。

- 共通 emotionalization direction: `g`
- 感情ごとの residual direction: `r_e`

この実験では steering vector を次のように組み立てる。

```text
Δ_e(α_g, α_r) = α_g g + α_r r̂_e
```

ここで:

- `α_g`: 全体の感情化・文体的 emotionalization の強さ。
- `α_r`: target emotion の residual direction の強さ。
- `r̂_e`: 感情 `e` の residual を単位正規化したもの。

## 8.3 Seed texts

8つの seed text を用意し、それぞれが Plutchik の8感情のどれかに軽く寄っている。

| Seed emotion | Seed text の内容 |
|---|---|
| joy | 良い一日で、物事がうまくいき幸せだった。 |
| trust | 彼女はいつも約束を守り、今回も信頼できる。 |
| fear | 暗い路地を一人で歩き、背筋が寒くなる。 |
| surprise | ドアを開けると信じられないものが待っていた。 |
| sadness | 古い写真が失ったものを思い出させる。 |
| disgust | ゴミ箱の匂いと見た目に胃がむかつく。 |
| anger | また約束を破られ、軽んじられてうんざりしている。 |
| anticipation | 結果が明日出るので、何度もメールを確認している。 |

重要なのは、これらが最初から完全に neutral ではなく、各感情に軽く寄っている点である。これは、以前の「base text が最初から sad/apologetic に偏っているのでは？」という懸念に対応する設計になっている。

## 8.4 実験設定

モデル:

```text
meta-llama/Llama-3.1-8B-Instruct
```

設定:

```text
PRIMARY_LAYER = 13
MAX_NEW_TOKENS = 100
DO_SAMPLE = False
USE_CACHE = True
BATCH_SIZE = 4
STEERING_MODES = ['last_token', 'completion_tokens']
```

実験条件:

### Experiment 1

固定:

```text
α_g = 3.0
```

スイープ:

```text
α_r ∈ [-5.0, -3.0, -1.5, 1.5, 3.0, 5.0]
```

目的: target emotion identity が `α_r` で変わるか。

### Experiment 2

固定:

```text
α_r = 3.0
```

スイープ:

```text
α_g ∈ [-5.0, -3.0, 3.0, 5.0]
```

目的: `α_g` が emotionalization strength を変えるか。

### Experiment 3

小さな grid:

```text
α_g ∈ [-3.0, 3.0]
α_r ∈ [-3.0, 3.0]
```

目的: `α_g` と `α_r` の相互作用を見る。

## 8.5 CAA decomposition の数値

Layer 13 の `caa_pooled` を分解した結果:

```text
caa_pooled shape = (8, 6, 4096)
hidden size = 4096
||g|| = 1.0000
```

各感情の `g` への射影と residual norm:

| Emotion | dot_with_g | resid_raw_norm |
|---|---:|---:|
| joy | 0.992324 | 0.123664 |
| trust | 0.991750 | 0.128191 |
| fear | 0.994632 | 0.103481 |
| surprise | 0.990805 | 0.135296 |
| sadness | 0.996513 | 0.083442 |
| disgust | 0.995369 | 0.096128 |
| anger | 0.994058 | 0.108854 |
| anticipation | 0.994153 | 0.107979 |

解釈:

- すべての感情CAAは `g` と 0.99以上揃っている。
- residual norm は0.08〜0.14程度しかない。
- したがって、raw CAA の大半は共通 emotionalization 成分であり、感情固有成分はかなり小さい。
- `α_r` を大きくして residual を効かせる設計は妥当だが、過大にすると文が壊れやすい可能性がある。

## 8.6 Generation sweep

条件数:

```text
unique alpha pairs = 10
total conditions = 1280
```

内訳:

```text
8 seeds × 8 target emotions × 2 steering modes × 10 alpha pairs = 1280
```

全条件を生成し、JSONL/CSVに保存。

```text
Rows written: 1280
Total cached: 1280
```

## 8.7 Experiment 1: fixed αG, sweep αR の観察

代表設定:

```text
α_g = 3.0
steering_mode = last_token
```

### 8.7.1 target joy の例

seed joy では、`α_r` が負でも正でも joy/serenity/contentment 系に保たれやすい。

seed fear では、`α_r` が大きくなると暗い路地の文が少し calm / warm / soothing に変わるが、完全な joy 文にはならない。

seed disgust では、`α_r=3` や `5` で「ゴミ箱の匂いが welcome respite」「refreshing and uplifting scent」のような意味的に不自然な反転が起きる。

seed anger では、joy target でも anger/frustration の文脈がかなり残る。

解釈:

- target joy は、元の seed が joy/surprise/anticipation なら比較的効く。
- seed が disgust/anger/sadness のように意味的に強い負感情を持つ場合、target joy へ動かすと意味の歪みや不自然な再解釈が起きる。
- これは CAA が絶対座標ではなく、文脈上の既存感情を変形する vector として働くことを示す。

### 8.7.2 target fear の例

seed fear では fear が強まる。

seed joy/trust では、target fear にしても元の positive 文脈がかなり残り、完全な恐怖にはならない場合が多い。

seed anticipation では、fear target が anxiety / waiting / compulsive checking の方向へ自然に乗りやすい。

解釈:

- fear は anticipation seed と相性が良い。
- joy/trust から fear へ変えるには、内容そのものを大きく変える必要があり、steering だけでは難しい。

### 8.7.3 target sadness / anger / disgust の例

seed sadness では sadness target が自然に強まる。

seed anger では sadness target にしても、broken promise / disrespect という怒り文脈が残り、hurt / betrayal / exhaustion のような混合感情になる。

seed disgust では disgust target は自然に強まるが、joy target など正方向に振ると不自然になりやすい。

解釈:

- target emotion が seed emotion と近い、または同じ感情ファミリーにある場合は自然。
- 遠い target emotion に強く振ると、感情は変わるが意味保存が壊れやすい。

## 8.8 Experiment 2: fixed αR, sweep αG の観察

代表設定:

```text
α_r = 3.0
steering_mode = last_token
```

`α_g` を変えると、感情固有性というより、全体の文体・強調・情緒性・再叙述の強さが変わる傾向がある。

観察:

- `α_g` が正に大きいと、文がより lyrical / emotionally colored / elaborated になる場合がある。
- `α_g` が負でも、必ずしも単純に neutral になるわけではなく、モデルの生成癖や residual direction との相互作用が出る。
- `α_g=5` では一部で文法崩れや不自然表現が出る。

解釈:

- `g` は「感情強度」だけでなく、感情的な言い換え・文学的表現・主観化・強調を含む可能性がある。
- UIノブとしては、単純に “intensity” と呼ぶより、“emotionalization / expressive intensity” と呼ぶ方が正確。

## 8.9 Experiment 3: αG × αR grid

小さな grid により、`α_g` と `α_r` の符号・組み合わせによる相互作用を観察している。

考え方:

- `α_g > 0, α_r > 0`: emotionalization と target residual が同時に強まる。
- `α_g > 0, α_r < 0`: 感情的にはなるが、target residual の反対方向へ動く。
- `α_g < 0, α_r > 0`: 感情化を抑えながら target residual を入れる。
- `α_g < 0, α_r < 0`: emotionalization も target residual も抑える/反転させる。

結果の読み方:

- 組み合わせによって出力の感情・文体・意味保存が大きく変わる。
- `α_r` は target identity に関係するが、seed text の意味的制約に強く依存する。
- `α_g` は情緒的な表現量や文体に広く作用する。

## 8.10 Proxy analysis: output variation by seed emotion

代表点:

```text
α_g = 3
α_r = 3
steering_mode = last_token
```

で、target emotion ごとに seed をまたいだ output length / repetition rate を計算。

結果:

| Target emotion | Mean length | Std length | Mean repetition rate | n seeds | n empty |
|---|---:|---:|---:|---:|---:|
| anger | 26.375 | 5.3168 | 0.0700 | 8 | 0 |
| anticipation | 34.500 | 4.6904 | 0.1335 | 8 | 0 |
| disgust | 32.750 | 9.8380 | 0.1407 | 8 | 0 |
| fear | 35.000 | 9.9857 | 0.1138 | 8 | 0 |
| joy | 34.125 | 5.3302 | 0.1016 | 8 | 0 |
| sadness | 34.375 | 8.5513 | 0.1331 | 8 | 0 |
| surprise | 38.875 | 6.1281 | 0.1351 | 8 | 0 |
| trust | 30.625 | 9.6501 | 0.1287 | 8 | 0 |

解釈:

- target emotion ごとに出力長のばらつきがある。
- fear, disgust, trust, sadness は seed による長さのばらつきが大きめ。
- empty は0で、生成自体は安定している。
- ただし長さや反復率だけでは感情同一性を十分に測れないため、今後は LLM-as-judge や embedding-based similarity が必要。

## 8.11 HTML inspection の代表観察

代表点 `α_g=3, α_r=3, last_token` で、target emotionごとに8 seed出力をカード表示している。

target joy の例から見えること:

- seed joy: 本当に joy/contentment/gratitude になる。
- seed trust: trust/reliability の文脈が残る。
- seed fear: dark alley が calm/soothing に変わるが、場面は残る。
- seed sadness: bittersweet nostalgia になり、完全な joy ではない。
- seed disgust: garbage smell が welcome/uplifting として再解釈され、意味的にやや破綻。
- seed anger: anger context が残り、disappointment/being let down 系になる。
- seed anticipation: anxious waiting が残る。

target trust の例でも同様に、seedの意味が残り、target trust が絶対的に上書きするわけではない。

## 8.12 主要結論

このPDFの最重要結論は以下。

1. 分解CAA steering は、絶対的な emotion coordinate ではない。
2. 出力は seed emotion / seed text に強く依存する。
3. target emotion は、seed の文脈を完全に置き換えるのではなく、seed の意味構造の上に感情的な変形を加える。
4. `α_g` は emotionalization / expressiveness を広く制御する。
5. `α_r` は target emotion residual を制御するが、効果は seed との距離や相性に依存する。
6. seed と target が遠い場合、意味保存が壊れたり、不自然な再解釈が起きる。

## 8.13 EmotionEngine への示唆

この実験はUI設計に直結する。

### 8.13.1 絶対座標UIだけでは不十分

「joy knob を上げれば必ず joy になる」という単純なUIは危険。実際には、元文が fear/disgust/anger に強く寄っていると、joy steering は不自然な意味変換を起こすことがある。

### 8.13.2 変化ベクトルUIとして設計すべき

より正確には、UIは「今の文章からどちらへ感情を動かすか」を操作するものとして設計すべき。

例:

- current affect estimate をまず推定。
- target emotion との距離を測る。
- 遠すぎる場合は小刻みに移動する。
- 意味保存評価を入れて、安全な範囲で steering する。

### 8.13.3 Seedごとの初期感情を考慮する必要

EmotionEngine では、ユーザー入力文が最初からどの感情に寄っているかを推定し、その初期状態に応じて steering strength を調整する必要がある。

### 8.13.4 UIでは `α_g` と `α_r` を分ける価値がある

- `α_g`: 表現の情緒化、詩的さ、主観性、感情強度。
- `α_r`: 目標感情の方向性。

この2軸分解は、単一の emotion slider より柔軟。

### 8.13.5 次に必要な評価

このPDFでは主に生成例と proxy analysis を見ている。次に必要なのは:

1. LLM-as-judge による target emotion score。
2. seed emotion preservation / transformation score。
3. meaning preservation score。
4. unnatural reinterpretation の検出。
5. seed-target distance と steering success の相関分析。
6. last_token と completion_tokens の定量比較。

---

# 9. PDF間の関係まとめ

## 9.1 実験の流れ

```text
CAA原論文
  ↓
caa_geometry
  ↓
共通 emotionalization 方向 g が支配的だと判明
  ↓
Experiment A: global CAA から g を除去
  ↓
Experiment B: per-source CAA から g を除去して decodability 検証
  ↓
caa_per_source_structure: 個別CAAの一貫性・内部構造・重なりを見る
  ↓
caa_manifold_linearity: 感情空間の線形性・補間・円環性を検証
  ↓
Experiment D: local PC を解釈可能なUIノブ候補にする
  ↓
seed_emotion_robustness: 実際に decomposed steering して、絶対座標か変化ベクトルかを検証
```

## 9.2 強く支持されたこと

### 9.2.1 CAAは安定した感情変化方向を持つ

per-source CAA と global CAA の cosine が約0.85で安定しているため、平均方向としての感情CAAは有意味。

### 9.2.2 共通 emotionalization 方向が非常に強い

各感情CAAは `g` と 0.99以上揃う。raw CAA同士の cosine も約0.98〜0.99。したがって、感情CAAの最大成分は「何の感情か」ではなく「neutral から emotional になる」方向。

### 9.2.3 `g` を除くと感情固有構造が出る

Experiment A で、残差化後に opposite pairs が負、adjacent pairs が弱正になる。Plutchik的構造が raw CAA ではなく residual space に現れる。

### 9.2.4 感情空間は低次元だが2Dだけでは足りない

global CAA の PR は3〜4程度。local PCA の PR は5程度。2D circumplex は部分的にはあるが、完全ではない。

### 9.2.5 各感情には内部軸がある

local PC 分析で、各感情は単一方向ではなく複数のサブ軸を持つ。これは「名前にできない感情」を扱う EmotionEngine の根拠になる。

### 9.2.6 Steering は絶対座標ではなく文脈依存

seed robustness 実験で、同じ target emotion でも seed emotion によって出力が大きく変わる。CAAは「目標感情そのものを生成する座標」ではなく、「現在の文脈を感情的に変形するベクトル」として振る舞う。

## 9.3 まだ未解決のこと

1. Local PC 軸が本当に意味のあるUIノブとして安定するか。
2. LLM-as-judge で target emotion / meaning preservation / naturalness を定量化したとき、どの `α_g, α_r` 範囲が最適か。
3. seed-target distance をどう定義すればよいか。
4. 感情空間の最適次元は何次元か。現状は3〜5次元程度が有力だが、タスクによって異なる可能性がある。
5. 生成時に last_token と completion_tokens のどちらがより安定か。
6. layer 13 以外、特に layer 10/16/19 で同じ steering をした場合の品質差。
7. local PC を実際に steering vector として使ったとき、極端例で解釈されたニュアンスが出力に再現されるか。

## 9.4 論文・章に書くなら中心主張は何か

現時点で最も強く書ける主張は次の形。

> Emotion-related CAA directions are dominated by a common neutral-to-emotional direction. After removing this shared component, emotion-specific residual structure becomes visible and partially aligns with expected affective relations such as Plutchik oppositions. Per-source CAA vectors are highly consistent with global directions, but each named emotion also contains multi-dimensional internal variation. Therefore, named emotions are better understood as prototype regions or cones within a continuous affective change space, rather than as isolated discrete directions. Steering experiments further suggest that these directions act as context-dependent emotion-change vectors rather than absolute emotion coordinates.

日本語では:

> 感情CAA方向の最大成分は、個別感情ではなく neutral から emotional への共通変化方向である。この共通成分を取り除くと、Plutchik 的な対立関係を含む感情固有の残差構造が現れる。個別ソースごとのCAAはグローバル方向と高く一致する一方で、各名前付き感情の内部には複数の局所軸が存在する。したがって、LLM内部の感情表現は8つの孤立した離散方向というより、連続的な感情変化空間内の prototype region / cone として捉えるのが妥当である。さらに steering 実験は、これらの方向が絶対的な感情座標ではなく、文脈依存の感情変化ベクトルとして作用することを示している。

---

# 10. 次にやるべき実験

## 10.1 Local PC steering 実験

Experiment D で local PC の極端例を抽出しただけでは、PC軸が実際に操作可能なノブかはまだ分からない。

次にやるべきこと:

1. 各感情の local PC1/PC2/PC3 を steering vector として使う。
2. 対応する感情 rewrite に対して `+PC` / `-PC` steering を行う。
3. LLM-as-judge に、極端例から推定された軸名に沿って変化したかを評価させる。
4. 意味保存と自然さも同時に評価する。

## 10.2 Seed-target distance と成功率

seed robustness 実験を定量化する。

1. seed emotion と target emotion の距離を residual CAA 空間で計算。
2. 距離が近いほど steering success が高いかを見る。
3. 距離が遠い場合に意味破綻率が上がるかを見る。
4. 距離に応じた adaptive α scaling を設計する。

## 10.3 `α_g` / `α_r` の最適範囲探索

現在はかなり大きな値も試している。実用UIでは安全範囲が必要。

評価指標:

- target emotion score
- meaning preservation
- grammaticality
- naturalness
- unwanted semantic inversion
- repetition rate
- length inflation

## 10.4 last_token vs completion_tokens

seed robustness では2つの steering mode を生成しているが、詳細比較はまだ弱い。

必要な比較:

- target score
- meaning preservation
- stability
- output length
- repetition
- computation speed
- qualitative examples

## 10.5 Layer sweep for decomposed steering

`g` と `r_e` の steering を layer 10/13/16/19 で比較する。

目的:

- layer 13 が本当に最適か。
- `g` と `r_e` で最適レイヤーが違うか。
- local PC steering は global CAA steering と同じレイヤーで効くか。

---

# 11. 最終まとめ

この一連のPDFから見える EmotionEngine の現状はかなり明確である。

単に「joy vector」「sadness vector」を作って足すだけでは不十分である。raw CAA のほとんどは共通 emotionalization 方向であり、個別感情の違いは小さな residual 成分として現れる。その residual 成分を取り出すと、Plutchik 的な対立や近接が部分的に現れ、感情空間が連続的・低次元的に組織化されていることが見えてくる。

一方で、感情は完全な2D円環でも、8つの離散クラスタでもない。global には3〜4次元程度、local には5次元程度の内部構造があり、各名前付き感情は単一方向ではなく、文脈によって広がる cone / prototype region として振る舞う。

最も重要なのは、実際の steering が seed text に強く依存すること。これは欠点というより、EmotionEngine の設計思想に合っている。ユーザーの文章を別のテンプレートに置き換えるのではなく、元の意味や文脈を保ちながら、表現しきれない感情方向へ少しずつ動かすための技術として CAA を使うべきである。

したがって、EmotionEngine の次の段階では、絶対的な感情カテゴリを選ぶUIではなく、現在の文脈からの連続的な感情変化を操作するUIを設計するのが最も妥当である。
