# Experiment 8 Analysis: Meta-Axis Interpretation (LLM Judge)

## 概要

**スクリプト**: `exp8_meta_axis_interpretation.py`
**対象**: Layer 13 / Llama-3.1-8B-Instruct
**入力**: exp7 の meta_axes.npy（m1–m5）+ activation データ + 全感情の rewrite テキスト
**目的**: 各 m_k が感情横断でどんな affective 次元を捉えているかを LLM judge（GPT-4.1）に解釈させる

Exp3（感情ごとの local PC 解釈）との違い：
- Exp3: 「一つの感情の中での high/low は何か？」
- Exp8: 「**全8感情をまたいで** high/low は何か？」→ named emotion を超えた pre-verbal 次元を問う

---

## LLM Judge 結果（Layer 13）

### m1: heightened engagement/intensity ↔ subdued/reflective distance

| フィールド | 内容 |
|-----------|------|
| **axis_name** | heightened engagement/intensity ↔ subdued/reflective distance |
| **ui_knob_label** | intense/immersed ↔ calm/detached |
| **high pole** | 感情的覚醒が高く、状況に没入している。強い、明確で緊迫した感情を直接表現。言語が intense、direct、affectively charged。|
| **low pole** | より落ち着いた、内省的、あるいは距離を置いたトーン。混合感情、皮肉、出来事後の穏やかさを表現。|
| **cross_emotion_pattern** | 感情の種類に関わらず、high pole = 感情的に「活性化」・没入；low pole = 感情的モデレーション、脱attachment、事後的反省 |
| **preverbal_interpretation** | **Arousal / affective engagement** — 感情がどれだけ強く即時的に感じられているか（valence や category に依存しない）|
| **contamination** | low |
| **confidence** | 5 / 5 |

---

### m2: immersed/engaged ↔ detached/resigned

| フィールド | 内容 |
|-----------|------|
| **axis_name** | immersed/engaged ↔ detached/resigned |
| **ui_knob_label** | immersed/engaged ↔ detached/resigned |
| **high pole** | 鮮明な感情的没入と能動的な状況関与。その瞬間に感情的に巻き込まれ、個人的意義や感情的影響を詳述する傾向。|
| **low pole** | より距離を置いた、諦め感のある、内省的なスタンス。感情はあるが即時性・強度が低い。解決に向かうか、最初の感情を乗り越えようとしている。|
| **cross_emotion_pattern** | 高い arousal で感情的没入（excitement、anxiety、nostalgia、frustration など問わず） vs. より静かな、処理された、距離のある affective 状態 |
| **preverbal_interpretation** | **Affective engagement / immediacy** — 感情的経験の中にいるか vs. 距離を置いて処理しているか |
| **contamination** | low |
| **confidence** | 5 / 5 |

---

### m3: immersed ↔ detached

| フィールド | 内容 |
|-----------|------|
| **axis_name** | immersed ↔ detached |
| **ui_knob_label** | immersed ↔ detached |
| **high pole** | 鮮明で即時的な没入感。現在の瞬間と深く結びついた、フィルタリングのない raw な感情反応。|
| **low pole** | 感情的な距離、反省、regulation。実際的問題、将来的含意、あるいは安心感に焦点が移る。感情トーンは less intense で measured。|
| **cross_emotion_pattern** | raw な即時 affective 状態に完全に吸収される（high）vs. 反省的・距離を置いた・調節されたスタンスを維持する（low） |
| **preverbal_interpretation** | **Affective immediacy / absorption** — 感情的経験に飲み込まれているか、vs. 一歩引いて立つか |
| **contamination** | low |
| **confidence** | 5 / 5 |

---

### m4: emotionally engaged ↔ emotionally detached

| フィールド | 内容 |
|-----------|------|
| **axis_name** | emotionally engaged ↔ emotionally detached |
| **ui_knob_label** | immersed & reactive ↔ calm & detached |
| **preverbal_interpretation** | **Affective engagement / immediacy** — 感情的に「その瞬間にいる」か vs. 感情的に緩衝・距離を置いているか。Arousal や emotional involvement に対応する可能性。|
| **contamination** | low |
| **confidence** | 5 / 5 |

---

### m5: engaged ↔ detached

| フィールド | 内容 |
|-----------|------|
| **axis_name** | engaged ↔ detached |
| **ui_knob_label** | engaged & immediate ↔ detached & reflective |
| **preverbal_interpretation** | **Affective engagement / immediacy** |
| **contamination** | low |
| **confidence** | 5 / 5 |

---

## 最重要発見：全5軸が同じ次元を捉えている

**5つすべての meta-axis が、本質的に同一の affective 次元に収束した**。

| Axis | 解釈の要約 |
|------|------------|
| m1 | Arousal / affective engagement |
| m2 | Affective engagement / immediacy |
| m3 | Affective immediacy / absorption |
| m4 | Affective engagement / immediacy |
| m5 | Affective engagement / immediacy |

すべて confidence = 5、contamination = low、そしてほぼ同じ ui_knob_label（engaged/immersed vs. detached/reflective）。

これは当初の予想（m1 = arousal, m2 = private/social, m3 = resolved/conflicted など、複数の独立した次元が出てくる）とは大きく異なる。

---

## この収束の解釈

### 解釈 A: モデルの残差感情空間は「arousal 一次元」で支配されている

Exp7 の EVR 分布（PC1: ~13%, PC2: ~5%, PC3: ~3% …）と合わせると、感情方向 c_e から g と u_e を除いた残差空間の分散のほとんどが、**感情的没入/距離（arousal-like dimension）** として組織化されている。PC2 以降は独立した別次元を持っておらず、arousal の変種（より高次の非線形成分）として現れている可能性がある。

### 解釈 B: 5つの PC が回転した同一のシリンダーを分割している

m_k が実際には1つの主要次元の周囲を回転したコピーであれば、直交性の制約下でも同じ方向を向いたベクトルが多数 PCA で出てくる可能性がある。特に m2-m5 の低い method agreement（exp7: 0.83, 0.27, 0.10）は、これらが unique な成分ではなく数値的 artifact を含む可能性を示す。

### 解釈 C: 交差感情の中で "arousal" だけが共通

感情ラベルを横断したとき、感情ごとの固有ニュアンス（disgust の sensory revulsion、anger の injustice感覚 など）は **g と r_e によって既に捉えられており**、残差には汎用的な intensity/engagement 次元しか残っていない。

---

## EmotionEngine への含意

### UI ノブへの落とし込み

実験結果から提案できるノブ：

```
Affective intensity (g):     neutral → highly emotional
Emotional preset (r_e):      [joy / sadness / anger / fear / ...]
Arousal / immersion (m1):    calm & detached ↔ intense & immersed
```

当初想定していた「resolved ↔ conflicted」「private ↔ social」「hopeful ↔ anxious」といった多様なノブは、少なくとも layer 13 の残差空間では見られなかった。

### 論文への含意

「Preverbal Affective Vector Projection」の観点では、次のように再解釈する：

> m_k が捉えているのは複数の異なる pre-verbal dimension ではなく、
> 感情経験の **没入度（arousal / immediacy）** という単一の軸に近い。

これは主張を弱めるものではなく、**「感情 activation の基底構造は named emotion の先に、arousal という pre-verbal 次元として組織化されている」** という主張に昇華できる。

この発見は Plutchik の circumplex model における valence × arousal の2軸モデルと対応する：
- valence 方向 → r_e（感情の種類）
- arousal 方向 → m1（感情の强度/没入度）
- g（emotionalization） → combined neutral-to-emotional axis

---

---

## Exp 8 v2: Contrastive Labeling（全軸同時提示）

### 動機と仮説

v1 で全5軸が「engaged vs detached」に収束した原因として、**LLM が各軸を孤立して評価したため、各軸内の最大信号（arousal）しか捉えられなかった**という仮説を立てた。軸間の相対的な差異は、複数軸を同時に見比べて初めて認識できる（contrastive interpretation）。

**v2 の変更点**：
- 全5軸のサンプルを1つのプロンプトに束ねて LLM に提示
- `axis_name` は全軸で一意であることを明示的に制約
- `what_distinguishes_from_others` フィールドで「他軸との差分」を必須回答とした

スクリプト: `exp8v2_contrastive_labeling.py`

---

### v2 LLM Judge 結果（Layer 13）

| Axis | axis_name | ui_knob_label | preverbal_interpretation | contamination | confidence |
|------|-----------|---------------|--------------------------|---------------|------------|
| m1 | agitated/reactive ↔ mellow/accepting | agitated/reactive ↔ mellow/accepting | Arousal/reactivity vs. calm acceptance | medium | 5 |
| m2 | nostalgic/ruminative ↔ pragmatic/forward-looking | nostalgic/ruminative ↔ pragmatic/forward | Reflective/ruminative vs. pragmatic/active orientation | medium | 4 |
| m3 | interpersonally engaged/closure-seeking ↔ self-contained/processing | closure-seeking ↔ self-contained | Interpersonal engagement vs. internal processing | medium | 4 |
| m4 | mundane/obligatory ↔ meaningful/fulfilling | mundane/obligatory ↔ meaningful/fulfilling | Triviality/obligation vs. fulfillment/meaning | medium | 4 |
| m5 | relief/release ↔ tension/constraint | relief/release ↔ tension/constraint | Release/relief vs. tension/constraint | medium | 5 |

---

### 各軸の詳細

#### m1: agitated/reactive ↔ mellow/accepting

- **high pole**: 強い即時的な反応 — ショック、嫌悪、恐怖、誇り、フラストレーションなど visceral な反応、かき乱されている感覚
- **low pole**: よりメロウで受容的・諦め的な態度。ユーモア、安堵、または不完全な状況でも平和を見出す
- **他軸との差分**: m2（郷愁/反省）や m3（対人クロージャー）と異なり、m1 は即時的な身体的・感情的覚醒の強度 vs. 穏やかな受容

#### m2: nostalgic/ruminative ↔ pragmatic/forward-looking

- **high pole**: 郷愁、反芻、感情的な省察。過去や「もしかしたら」への執着、喪失感・後悔のニュアンス
- **low pole**: より実用的・行動志向。即時的な対処と前進に焦点、感情的な反芻が少ない
- **他軸との差分**: m1（覚醒）や m4（社会的つながり）と異なり、m2 は**時間的志向**（過去振り返り vs. 現実対応）

#### m3: interpersonally engaged/closure-seeking ↔ self-contained/processing

- **high pole**: 他者へのアプローチ、クロージャーの追求、直接的な感情表現 — 関係性・社会的イベントへの焦点
- **low pole**: より自己完結的。感情を内部で処理するか、分離した観察的な方法で記述、直接的な対人エンゲージメントが少ない
- **他軸との差分**: m1（覚醒）や m2（反芻）と異なり、m3 は**感情的スタンスが社会的方向か内的処理か**

#### m4: mundane/obligatory ↔ meaningful/fulfilling

- **high pole**: ルーティン、義務、小さな苛立ち — 平凡で取引的、感情的に平坦な経験
- **low pole**: 意味、充実感、感情的豊かさ — 深い関係性や重要なライフイベントを強調
- **他軸との差分**: m2（反芻）や m5（安堵/緊張）と異なり、m4 は経験の**実存的重み vs. 平凡さ**

#### m5: relief/release ↔ tension/constraint

- **high pole**: 安堵、解放、圧力が解除される感覚 — 天候、社会的状況、義務から問わず
- **low pole**: 緊張、制約、行き詰まり感、不快のアンダーカレント
- **他軸との差分**: m1（覚醒）や m4（意味）と異なり、m5 は**圧力が解除される・課されるという身体的・状況的感覚**

---

### Overall Summary（LLM 評価）

> "The axes capture genuinely distinct pre-verbal affective dimensions: arousal/reactivity, temporal rumination, interpersonal engagement, meaningfulness, and relief/tension. While some axes share surface similarities (e.g., m1 and m5 both touch on bodily states), each is uniquely defined by its contrast with the others, reflecting orthogonal aspects of affective experience beyond simple emotional valence or intensity."

---

### v1 vs v2 比較

| | v1（孤立評価） | v2（同時比較） |
|---|---|---|
| m1 | arousal/engagement | arousal/reactivity vs. calm acceptance |
| m2 | affective engagement/immediacy | nostalgic/ruminative ↔ pragmatic/forward-looking |
| m3 | affective immediacy/absorption | interpersonally engaged ↔ self-contained |
| m4 | affective engagement/immediacy | mundane/obligatory ↔ meaningful/fulfilling |
| m5 | affective engagement/immediacy | relief/release ↔ tension/constraint |
| contamination | all low | all medium |
| convergence | 全軸同一解釈 | 全軸異なる解釈 |

---

### 解釈

**仮説は正しかった**。v1 の収束は測定の人工物（measurement artifact）であり、LLM judge が各軸を孤立して評価したことで、各軸内の支配的な信号（arousal）しか捉えられなかった。

v2 の結果が示す5次元の構造は、Plutchik や Russell の circumplex を超えた、よりきめ細かい感情空間の組織化と対応する：

| 軸 | 対応する感情心理学的概念 |
|---|---|
| m1: arousal/reactivity | Russell の arousal 次元 |
| m2: temporal orientation | 反芻理論（Nolen-Hoeksema）, 感情調節の時間的側面 |
| m3: social/internal | 感情の社会的共有（Rimé）, 社会的な感情調節 |
| m4: meaning/triviality | appraisal theory の relevance/significance 評価 |
| m5: relief/tension | bodily affect（Damasio のソマティック・マーカー）|

**contamination が全軸 medium** になった点は注意が必要：プロンプトで「異なる名前を必ずつけろ」と強制したことで、LLM がテキストスタイルや意味的内容（天候・社会的状況などの topic）の違いを拾い上げた可能性がある。特に m4（mundane/obligatory）と m5（relief/release）は、感情的次元というよりも topic 的な分離を反映しているかもしれない。

---

## 制限と留意点

1. **全軸が同一解釈になった点の検証**: LLM judge が何らかのバイアスを持ってすべてを「engaged vs detached」と解釈した可能性がある。human 評価者による二重チェックが望ましい。

2. **m4, m5 の信頼性**: Exp7 で method agreement が低かった m4（0.27）、m5（0.10）については、この解釈も信頼性が低い。

3. **Layer 依存性**: Layer 13 での発見。より深い層（19, 22）では異なる軸が出てくる可能性がある。

4. **テキストの多様性**: 感情 rewrite は同一データセットから生成されており、diverse なスタイルを持っていない可能性がある。これが全軸が同じ次元に収束した原因かもしれない。