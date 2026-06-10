# Sentence-Level Writing Guide for ML/AI Papers
### ICML / NeurIPS / ICLR Style

Each entry follows this structure:
- **Function** — what the sentence does rhetorically
- **Abstract pattern** — the skeleton with `[SLOTS]`
- **Templates** — ready-to-adapt variants
- **Weak** — a common bad version
- **Improved** — the corrected version
- **Notes** — grammar, hedging, and tone guidance

---

## 1. Sentence Structure

---

### 1.1 Fronted purpose clause

**Function:** Opens a methods sentence with intent before action. Signals to the reader *why* the next step is taken before describing *what* is done.

**Abstract pattern:**
> To [GOAL], we [ACTION].

**Templates:**
- To isolate the effect of [COMPONENT], we ablate [COMPONENT] while holding [EVERYTHING ELSE] fixed.
- To validate this claim, we conduct [EXPERIMENT] on [BENCHMARK].
- To handle [EDGE CASE], we introduce [MECHANISM].

**Weak:**
> We did an ablation study where we removed the attention module to see if it was important.

**Improved:**
> To quantify the contribution of the attention module, we ablate it while keeping all other components fixed.

**Notes:** Front the infinitive; do not bury the goal in a relative clause. Avoid "we did X to see if" — that sounds exploratory. Prefer "to [measure/quantify/isolate/validate]" for specificity. The subject of the infinitive and the subject of the main clause must match (no dangling modifier).

---

### 1.2 Appositive contribution naming

**Function:** Names a new artifact and compresses its defining properties in one sentence. Prevents the need for a separate definitional sentence.

**Abstract pattern:**
> We introduce [NAME], a [DESCRIPTOR] of [SCALE] [TYPE].

**Templates:**
- We introduce [NAME], a [SCALE]-example benchmark for evaluating [PROPERTY] in [DOMAIN].
- We propose [NAME], a lightweight [ARCHITECTURE] that [KEY CAPABILITY] without [COST].
- We present [NAME], a [SCALE]-parameter model pretrained on [DATA] and finetuned for [TASK].

**Weak:**
> We made a new dataset. It is called [NAME]. The dataset has [SCALE] examples and covers [TOPIC].

**Improved:**
> We introduce [NAME], a [SCALE]-example dataset of [CONTENT] spanning [RANGE] [DOMAIN] conditions.

**Notes:** The appositive ("a [DESCRIPTOR]...") should pack in scale, type, and one defining property. Avoid stacking adjectives beyond three. "We introduce" is preferred over "we present" or "we propose" for datasets; "we propose" fits methods. Use "we release" if open-sourcing.

---

### 1.3 Relative clause embedding a property

**Function:** Defines a term and embeds its key property in the same sentence, avoiding a separate definitional sentence that breaks flow.

**Abstract pattern:**
> [TERM], which [PROPERTY], [MAIN CLAIM OR USE].

**Templates:**
- [REPRESENTATION], which encodes [PROPERTY] in the [LAYER] of the network, is extracted and used to [ACTION].
- [METRIC], which measures [WHAT IT CAPTURES], is reported across all conditions.
- [MECHANISM], which enables [CAPABILITY] without [COST], is described in Section [X].

**Weak:**
> We use [REPRESENTATION]. [REPRESENTATION] captures [PROPERTY]. We extract it from the network.

**Improved:**
> [REPRESENTATION], which captures [PROPERTY] at layer [L], is extracted and used as input to [DOWNSTREAM MODULE].

**Notes:** The relative clause must modify a specific noun — not a whole clause. Use "which" (non-restrictive) when the property is supplementary; use "that" (restrictive) when the property is definitionally necessary. Do not use this structure more than once per paragraph.

---

### 1.4 Fronted concessive clause

**Function:** Preemptively concedes a weakness before asserting a result. Shows the claim holds even under an acknowledged constraint.

**Abstract pattern:**
> Although [LIMITATION], [POSITIVE CLAIM].

**Templates:**
- Although [METHOD] does not model [ASPECT], it achieves competitive performance on [BENCHMARK].
- Although our evaluation is limited to [SCOPE], the results suggest [GENERALIZATION CLAIM].
- Although [RESOURCE] is not perfectly balanced, the distribution is comparable to [PRIOR WORK].

**Weak:**
> The method has some limitations but it still works well.

**Improved:**
> Although [METHOD] relies on [SIMPLIFYING ASSUMPTION], it achieves [METRIC] on [BENCHMARK], matching models that do not make this assumption.

**Notes:** The concessive must be specific — vague concessions ("has some limitations") read as defensive rather than honest. The main clause should follow immediately, without a hedge that weakens both halves. "Despite [NP]" is a compact alternative to "Although [clause]".

---

### 1.5 Contribution enumeration

**Function:** Enumerates discrete contributions at the end of the introduction. Gives reviewers and readers a navigational contract.

**Abstract pattern:**
> Our contributions are: (1) [CONTRIBUTION 1]; (2) [CONTRIBUTION 2]; (3) [CONTRIBUTION 3].

**Templates:**
- Our contributions are: (1) we propose [METHOD], which [KEY PROPERTY]; (2) we introduce [RESOURCE], a [DESCRIPTOR]; (3) we demonstrate [FINDING] on [BENCHMARK].
- In summary, this work contributes: a [ARTIFACT] for [TASK]; an analysis of [PHENOMENON]; and evidence that [CLAIM].

**Weak:**
> We do several things in this paper. First we propose a method. We also have a dataset. And we run experiments.

**Improved:**
> Our main contributions are: (1) [METHOD], a [DESCRIPTOR] that [KEY PROPERTY]; (2) [DATASET], a [SCALE] benchmark for [TASK]; and (3) an empirical demonstration that [CLAIM] across [CONDITIONS].

**Notes:** Each contribution should start with a noun phrase (artifact or claim), not a verb phrase starting with "we." Reviewers at ICML/NeurIPS scan this list first — make each point falsifiable and concrete. Three is the standard number; four is acceptable; five signals scope creep. Do not repeat the abstract here word for word.

---

### 1.6 Paired comparison in one sentence

**Function:** Contrasts two systems or approaches in a single compound sentence, making the functional difference immediately clear.

**Abstract pattern:**
> While [APPROACH A] [PROPERTY A], [APPROACH B] [PROPERTY B], enabling [BENEFIT].

**Templates:**
- While [PRIOR METHOD] requires [RESOURCE], our approach operates with [CHEAPER ALTERNATIVE], enabling deployment in [CONSTRAINED SETTING].
- While [BASELINE] models [ASPECT] globally, [OUR METHOD] models it at [FINER GRANULARITY], capturing [WHAT IS GAINED].

**Weak:**
> [PRIOR METHOD] uses a lot of memory. Our method does not use as much memory. So ours is better for low-resource settings.

**Improved:**
> While [PRIOR METHOD] requires [SCALE] GPU-hours at inference, our approach reduces this to [SCALE], enabling use in latency-sensitive applications.

**Notes:** Keep both halves syntactically parallel. Do not embed a separate result claim — the contrast itself should convey the advantage. Avoid "better" without specifying what dimension and by how much.

---

---

## 2. Hedging and Epistemic Modality

---

### 2.1 Conjecture marker

**Function:** Signals that an explanation is the authors' interpretation, not a proven causal account. Protects against overclaiming a mechanism.

**Abstract pattern:**
> We conjecture that [CLAIM] because [INDIRECT EVIDENCE OR REASONING].

**Templates:**
- We conjecture that [BEHAVIOR] arises because [MECHANISM], though confirming this would require [ADDITIONAL EXPERIMENT].
- We conjecture that [FINDING] is driven by [FACTOR], consistent with [PRIOR OBSERVATION].

**Weak:**
> The reason the model does this is because [MECHANISM].

**Improved:**
> We conjecture that this behavior is driven by [MECHANISM]; a controlled study varying [FACTOR] would be needed to confirm this.

**Notes:** "We conjecture" is stronger than "we speculate" (more reasoned) but weaker than "we show" (no proof). Use when you have indirect evidence for a mechanism but not a controlled experiment. Do not use for results you actually measured — that is "we find" or "we observe." Do not chain multiple conjectures in one paragraph without evidence.

---

### 2.2 "Appears to" hedge for observed trends

**Function:** Reports a pattern in data while acknowledging that the evidence is consistent with, but not conclusive for, the claim.

**Abstract pattern:**
> [SYSTEM] appears to [BEHAVIOR], suggesting [INTERPRETATION].

**Templates:**
- [MODEL] appears to [RELY ON / ENCODE / IGNORE] [FEATURE], as evidenced by [PROBE RESULT / ABLATION].
- The representations appear to form [STRUCTURE], suggesting that [INTERPRETATION].
- Performance appears to plateau beyond [SCALE], though longer training may shift this trend.

**Weak:**
> The model encodes [FEATURE] because the probe accuracy was 80%.

**Improved:**
> The model appears to encode [FEATURE] linearly: a linear probe achieves [ACCURACY] at layer [L], whereas a matched random baseline achieves [BASELINE].

**Notes:** "Appears to" is appropriate when the measurement is indirect (probing, attention visualization, behavioral testing). Do not use it for direct measurements — "we find" is correct then. Pair "appears to" with a concrete observable ("as evidenced by", "consistent with") to prevent the hedge from becoming vague.

---

### 2.3 Softened universal ("in general")

**Function:** States a trend without claiming it holds in every case. The exception clause shows the authors have checked the boundaries.

**Abstract pattern:**
> In general, [PATTERN], although [KNOWN EXCEPTION OR CONDITION] may apply.

**Templates:**
- In general, [FINDING] holds across [CONDITIONS], although performance degrades when [EDGE CONDITION].
- In general, larger [COMPONENT] improves [METRIC], although gains diminish beyond [THRESHOLD].

**Weak:**
> The method always works better when you add more layers.

**Improved:**
> In general, increasing [COMPONENT] depth improves [METRIC], although gains diminish beyond [THRESHOLD] layers, at which point [BEHAVIOR].

**Notes:** Use "in general" rather than "generally" when the hedging clause follows (stylistic preference at top venues). Do not use "always" for empirical claims. "Tends to" is a softer alternative to "in general, X" when no exception clause follows.

---

### 2.4 "May" for uncontrolled variables

**Function:** Flags that a result depends on a variable the current experiment did not control, preventing overconfident generalization.

**Abstract pattern:**
> [FACTOR] may affect [RESULT], though we do not vary it in this work.

**Templates:**
- Prompt wording may affect [METRIC]; all conditions use identical prompts to control for this.
- The choice of [HYPERPARAMETER] may influence [FINDING]; we fix it to [VALUE] following [PRIOR WORK].
- [DOMAIN SHIFT] may reduce [PERFORMANCE], which we leave for future evaluation.

**Weak:**
> We didn't test everything so there might be other factors.

**Improved:**
> [ARCHITECTURE CHOICE] may interact with [TRAINING PROCEDURE]; we leave a systematic study of this interaction to future work.

**Notes:** Use "may" (not "might") for scientific hedging in formal writing. Make the source of uncertainty specific — not "other factors" but "[named factor]." When you have controlled for it, say so explicitly rather than hedging.

---

### 2.5 Cautious result framing

**Function:** Presents a promising but preliminary result honestly, preventing reviewers from seeing you as overclaiming.

**Abstract pattern:**
> These results are [CHARACTERIZATION] but [QUALIFIER]; stronger conclusions would require [ADDITIONAL EVIDENCE].

**Templates:**
- These results are promising but preliminary; a larger-scale study with [N] conditions would be needed to confirm [SPECIFIC CLAIM].
- The improvement is statistically significant but modest ([VALUE]); we do not claim this constitutes a breakthrough.
- These findings are consistent with [HYPOTHESIS] but do not constitute a controlled test of it.

**Weak:**
> These results prove that our method is better.

**Improved:**
> These results suggest an advantage for [METHOD] on [TASK], though the gap ([VALUE]) is within the margin of [VARIANCE SOURCE]; we recommend replication on additional benchmarks.

**Notes:** Venue reviewers at ICML/NeurIPS respect epistemic honesty. Overclaiming ("proves", "demonstrates superiority") invites rebuttal. Pair the caveat with a specific requirement for stronger evidence, not a generic disclaimer.

---

### 2.6 Modal "could" for future directions

**Function:** Proposes an extension as possible without committing to it. Standard for future-work paragraphs.

**Abstract pattern:**
> [APPROACH] could be extended to [APPLICATION], which we leave as a direction for future work.

**Templates:**
- The framework could be applied to [RELATED TASK] with minimal modification.
- This approach could be combined with [COMPLEMENTARY METHOD] to address [LIMITATION].
- With [RESOURCE], this method could scale to [LARGER SETTING].

**Weak:**
> In the future we will do more experiments.

**Improved:**
> [METHOD] could be extended to [DOMAIN] by replacing [COMPONENT] with [ALTERNATIVE]; we leave this to future work given the scope of the present paper.

**Notes:** "Could" is hypothetical; do not use it for work you actually plan to publish soon (that reads as sandbagging). Do not list more than two or three future directions — it reads as though the current paper is incomplete.

---

---

## 3. Contrast Expressions

---

### 3.1 "However" as gap pivot

**Function:** Pivots from describing the status quo to identifying the gap your work addresses. The standard move at the end of the opening paragraph of an introduction.

**Abstract pattern:**
> However, [LIMITATION OF CURRENT STATE], which motivates [THIS WORK / OUR APPROACH].

**Templates:**
- However, existing methods require [RESOURCE] that is unavailable in [SETTING], limiting their applicability.
- However, [PRIOR WORK] assumes [CONDITION] that does not hold in [OUR TARGET DOMAIN].
- However, no prior work has [ACTION] at [SCALE / GRANULARITY], leaving [OPEN PROBLEM].

**Weak:**
> But there are some problems with existing work so we made something new.

**Improved:**
> However, existing approaches require [RESOURCE] at inference time, which precludes deployment in [CONSTRAINED SETTING] — the primary target of this work.

**Notes:** "However" should follow a positive description of prior work — it reads as contrast, not dismissal. Place it at the start of a sentence, not mid-sentence ("X is good; however, Y is missing" is correct; "X is, however, missing Y" is fine too but less emphatic). Do not use "but" as a sentence opener in formal writing.

---

### 3.2 "While X, Y" simultaneous contrast

**Function:** Places two simultaneous truths in parallel, allowing the reader to compare them without the author asserting which is better.

**Abstract pattern:**
> While [SYSTEM A] [PROPERTY A], [SYSTEM B] [PROPERTY B].

**Templates:**
- While [PRIOR METHOD] achieves high [METRIC A], it sacrifices [METRIC B]; our method retains both.
- While [ARCHITECTURE] excels on [TASK], it struggles with [RELATED TASK] due to [REASON].
- While [ASSUMPTION] simplifies training, it may not hold in [REAL SETTING].

**Weak:**
> [PRIOR METHOD] is good at [METRIC A] but bad at [METRIC B]. Our method is good at both.

**Improved:**
> While [PRIOR METHOD] achieves state-of-the-art [METRIC A], it requires [COST]; our approach reduces [COST] by [AMOUNT] with only [SMALL DROP] in [METRIC A].

**Notes:** Keep both halves syntactically parallel — same tense, same clause type. "While" is simultaneous contrast; use "whereas" for a stronger opposition or factual dichotomy. Do not use "whilst" in American-English venues.

---

### 3.3 "In contrast" inter-paragraph pivot

**Function:** Opens a sentence to distinguish the present work from the preceding description of a competing approach. The pivot that ends the related work and opens the contribution.

**Abstract pattern:**
> In contrast, [OUR APPROACH / PROPOSED METHOD] [DIFFERENTIATING PROPERTY].

**Templates:**
- In contrast, we do not assume [ASSUMPTION], allowing [BENEFIT] in [SETTING].
- In contrast to [PRIOR CLASS OF METHODS], our approach requires only [SIMPLER REQUIREMENT].
- In contrast, [METHOD] operates directly on [INPUT TYPE] without [PREPROCESSING STEP].

**Weak:**
> Our method is different because it doesn't need [ASSUMPTION].

**Improved:**
> In contrast, [METHOD] makes no assumption about [ASPECT], enabling [BEHAVIOR] without access to [RESOURCE].

**Notes:** "In contrast" requires that the preceding sentence has described the thing being contrasted. Do not open a paper or section with "In contrast" — the reader needs context first. Use "by contrast" as a stylistic alternative with the same meaning.

---

### 3.4 "Despite X, Y" robustness framing

**Function:** Turns an acknowledged weakness into evidence of robustness. Claims the result holds even under unfavorable conditions.

**Abstract pattern:**
> Despite [LIMITATION], [POSITIVE OUTCOME] is still observed.

**Templates:**
- Despite being trained on [SMALL / NOISY / BIASED] data, [METHOD] generalizes to [HELD-OUT CONDITION].
- Despite using no [RESOURCE], [METHOD] achieves [METRIC] on [BENCHMARK].
- Despite [CONSTRAINT], the representations retain [PROPERTY], as shown in [FIGURE/TABLE].

**Weak:**
> Even though we had limited data the model still worked.

**Improved:**
> Despite training on only [N] examples — roughly [FRACTION] of the data used by [PRIOR METHOD] — [MODEL] achieves [METRIC] on [BENCHMARK].

**Notes:** "Despite" takes a noun phrase, not a clause. Do not confuse with "although" (which takes a clause). The contrast is most persuasive when the limitation is quantified ("only [N] examples", "without any [RESOURCE]").

---

### 3.5 "Not X, but Y" corrective contrast

**Function:** Preempts a likely misreading of the results by explicitly naming and rejecting an alternative interpretation.

**Abstract pattern:**
> This is not [DISMISSED INTERPRETATION] but rather [CORRECT INTERPRETATION].

**Templates:**
- This improvement is not due to [CONFOUND] but rather to [PROPOSED MECHANISM], as evidenced by [ABLATION].
- The gap is not a result of [ARTIFACT] but reflects a genuine difference in [CAPABILITY].
- This is not a weakness of [DESIGN CHOICE] but an artifact of [EVALUATION PROTOCOL].

**Weak:**
> The results might be because of [CONFOUND] but we don't think so.

**Improved:**
> The performance gap is not attributable to [CONFOUND] — an ablation that removes [COMPONENT] while preserving [CONFOUND] shows only [SMALL EFFECT] — but rather to [PROPOSED MECHANISM].

**Notes:** Requires prior evidence (ablation, control) to be credible. Without evidence, this pattern reads as assertion. Place the evidence immediately after the rejection clause. Use "but rather" (not "but instead") in formal prose.

---

---

## 4. Motivation and Problem Introduction

---

### 4.1 Significance-first opener

**Function:** Opens the introduction by anchoring the task in dual relevance — scientific and practical — before narrowing to the specific problem.

**Abstract pattern:**
> [TASK / CAPABILITY] is fundamental to both [RESEARCH DOMAIN] and [PRACTICAL APPLICATION].

**Templates:**
- Understanding [PHENOMENON] is fundamental to both [SCIENTIFIC DOMAIN] and the development of [APPLICATION].
- [CAPABILITY] is central to [RESEARCH AREA] and has practical relevance for [DOWNSTREAM USE CASE].

**Weak:**
> [TASK] is an important and interesting problem in machine learning.

**Improved:**
> [CAPABILITY] is central to [RESEARCH AREA]: it underlies [SCIENTIFIC QUESTION] and is a prerequisite for [PRACTICAL SYSTEM] to operate reliably.

**Notes:** Avoid "important and interesting" — this is the most common opener and adds nothing. Instead, anchor in a specific scientific or engineering consequence. ICML/NeurIPS reviewers respond better to precision than to breadth of motivation.

---

### 4.2 Gap statement

**Function:** Concisely states the core deficiency of the current state after acknowledging real progress. The workhorse sentence of every introduction.

**Abstract pattern:**
> Despite [ACKNOWLEDGED PROGRESS], [RESOURCE / METHOD] for [SUB-TASK] remains [DEFICIENCY].

**Templates:**
- Despite significant progress in [AREA], methods for [SPECIFIC SETTING] remain limited by [BARRIER].
- Despite [PRIOR WORK], no existing approach addresses [ASPECT], leaving [OPEN PROBLEM].
- Despite strong performance on [BENCHMARK], current models fail to [CAPABILITY] when [CONDITION].

**Weak:**
> There are still some problems in this area that people haven't solved yet.

**Improved:**
> Despite strong benchmark performance, current [MODEL CLASS] fail to [BEHAVIOR] when [DISTRIBUTION SHIFT], limiting their use in [REAL-WORLD SETTING].

**Notes:** "Remains" and "still lacks" are the standard verbs. Quantify the gap when possible ("no dataset exceeds [N] examples for [TASK]"). Do not dismiss all prior work — acknowledge what has been achieved before stating what has not.

---

### 4.3 "Key challenge" framing

**Function:** Names the specific technical or empirical barrier that your paper attacks, focusing reader expectations.

**Abstract pattern:**
> A key challenge in [FIELD] is [PROBLEM], because [REASON THE PROBLEM IS HARD].

**Templates:**
- A key challenge in [TASK] is [PROBLEM]: [REASON], making [STANDARD APPROACH] insufficient.
- The central challenge is [TECHNICAL DIFFICULTY], which arises because [MECHANISM].
- [PROBLEM] poses a fundamental challenge: [WHY IT IS HARD], and existing approaches [HOW THEY FAIL].

**Weak:**
> It is hard to do [TASK] because there are many challenges.

**Improved:**
> A key challenge in [TASK] is [PROBLEM]: [REASON] means that [STANDARD APPROACH] degrades to [FAILURE MODE] under [CONDITION].

**Notes:** "Key challenge" is slightly overused but remains precise. Alternatives: "a central difficulty", "the primary obstacle". Always follow with the reason — the challenge alone is not informative. Lead reviewers to see the problem before they see your solution.

---

### 4.4 Raises-the-question framing

**Function:** Transitions from a known finding (often from prior work) to the specific question this paper addresses. Creates a logical bridge rather than an abrupt pivot.

**Abstract pattern:**
> This raises the question of whether [OPEN QUESTION], which we investigate here.

**Templates:**
- This raises the question of whether [CAPABILITY] can be [ACHIEVED / MAINTAINED] under [CONDITION].
- An open question is whether [FINDING] holds in [DIFFERENT SETTING]; we address this directly.
- Whether [CLAIM] generalizes beyond [ORIGINAL DOMAIN] is an open question that motivates this work.

**Weak:**
> So we decided to investigate [TOPIC].

**Improved:**
> It remains unclear whether [BEHAVIOR] observed in [ORIGINAL SETTING] persists under [DIFFERENT CONDITION]; we design [EXPERIMENT] to test this directly.

**Notes:** The question must be specific and answerable by the experiments in the paper. Do not raise questions the paper does not answer — this invites reviewers to ask why you didn't answer them. Use "we investigate" not "we study" when the outcome is not pre-known.

---

### 4.5 Paucity-of-data motivation

**Function:** Motivates a dataset contribution by specifying the exact resource gap, not just claiming that "more data is needed."

**Abstract pattern:**
> This task suffers from a lack of [RESOURCE TYPE] suitable for [TRAINING / EVALUATION].

**Templates:**
- No existing dataset captures [PROPERTY] at the [SCALE / GRANULARITY] needed for [GOAL].
- Existing [RESOURCE TYPE] for [TASK] are limited to [N] examples, insufficient for [APPROACH].
- The absence of [RESOURCE] with [PROPERTY] has prevented [APPROACH] from being tested on [TASK].

**Weak:**
> There is not enough data so we collected more.

**Improved:**
> No existing benchmark for [TASK] provides [PROPERTY]: the largest available resource contains only [N] instances, insufficient for training [MODEL TYPE].

**Notes:** Quantify the gap whenever possible. Cite the largest existing resource to show you have surveyed the space. Do not claim "no prior work" on a dataset unless you have done a thorough search — reviewers will find counterexamples.

---

---

## 5. Prior Work Description

---

### 5.1 Chronological field survey opener

**Function:** Opens a related work section with a historical anchor that situates the present work in the evolution of the field.

**Abstract pattern:**
> Early work on [TOPIC] primarily relied on [APPROACH], with later approaches moving toward [DIRECTION].

**Templates:**
- Early work on [TASK] relied on [METHOD TYPE]; subsequent approaches adopted [ARCHITECTURE], achieving [IMPROVEMENT].
- The field has progressed from [EARLY APPROACH] to [RECENT APPROACH], with [BENCHMARK] serving as the standard evaluation.
- Prior to [YEAR / MODEL], [TASK] was addressed primarily via [METHOD]; the advent of [PARADIGM] shifted the focus to [DIRECTION].

**Weak:**
> Many people have worked on [TASK] before.

**Improved:**
> Work on [TASK] has progressed from [EARLY APPROACH] to [RECENT APPROACH], with the introduction of [ARCHITECTURE] marking a clear inflection point in [METRIC].

**Notes:** Keep the historical sweep brief — one to two sentences. The goal is to orient the reader, not to write a survey. Cite one or two representative works per era rather than listing all of them. Proceed quickly to the specific prior work most related to your approach.

---

### 5.2 Representative-work summary

**Function:** Summarizes a single prior paper compactly — method, setting, and key finding — without requiring the reader to have read it.

**Abstract pattern:**
> [CITATION] propose [METHOD] for [TASK], achieving [KEY RESULT] on [DATASET].

**Templates:**
- [CITATION] propose [METHOD], which [KEY PROPERTY], and demonstrate [RESULT] on [BENCHMARK].
- [CITATION] show that [FINDING], using [METHOD] applied to [DATASET / SETTING].
- [CITATION] introduce [ARTIFACT], a [DESCRIPTOR] that is widely used as a baseline for [TASK].

**Weak:**
> [CITATION] did some interesting work on [TASK] and got good results.

**Improved:**
> [CITATION] propose [METHOD], which [KEY PROPERTY]; they report [METRIC VALUE] on [BENCHMARK], [IMPROVEMENT/COMPARISON TO PRIOR SOTA].

**Notes:** Use present tense for what the paper proposes/claims ("propose", "show", "argue"); use past tense for what they did experimentally if needed ("they trained on"). Do not copy result numbers verbatim from another paper without verifying them. One sentence per paper is almost always sufficient in related work.

---

### 5.3 Collective prior work limitation

**Function:** Identifies a limitation shared by a class of methods, motivating your approach without singling out any one paper unfairly.

**Abstract pattern:**
> Previous approaches have [COMMON PRACTICE], which [CONSEQUENCE OR LIMITATION].

**Templates:**
- Previous methods assume [CONDITION], which does not hold when [FAILURE SCENARIO].
- Prior work has focused on [NARROW SCOPE], leaving [BROADER SETTING] underexplored.
- Existing approaches require [RESOURCE] at [STAGE], limiting applicability in [CONSTRAINED SETTING].

**Weak:**
> People have tried to do [TASK] before but they didn't think about [ASPECT].

**Improved:**
> Prior work on [TASK] has largely assumed [CONDITION], a constraint that does not hold in [TARGET SETTING]; our approach relaxes this assumption by [METHOD].

**Notes:** "Previous approaches" is safer than "prior work" when you are attributing a specific failing, because it implies the limitation applies to a method class rather than all existing research. Always follow the limitation with why your approach avoids it.

---

### 5.4 "Shown to be" effectiveness justification

**Function:** Justifies a design choice by citing prior empirical evidence rather than making a theoretical argument alone.

**Abstract pattern:**
> [METHOD] has been shown to [PROPERTY] on [TASK / DOMAIN], which motivates its use here.

**Templates:**
- [ARCHITECTURE] has been shown to [CAPABILITY] in [PRIOR WORK], motivating its adoption as our backbone.
- [TECHNIQUE] has been shown to improve [METRIC] without [COST], making it a natural fit for [OUR SETTING].
- [TRAINING OBJECTIVE] has been shown to produce [REPRESENTATION PROPERTY] representations, which we exploit in [COMPONENT].

**Weak:**
> We used [METHOD] because it is good.

**Improved:**
> [METHOD] has been shown to [CAPABILITY] across [RANGE OF SETTINGS] in [CITATION]; we adopt it here for [SPECIFIC REASON].

**Notes:** "Has been shown to" is passive and backward-looking — appropriate for justifying a design choice. Do not use it for your own claims; use "we find" or "we show" for those. Cite the specific paper that showed it, not a general review.

---

### 5.5 Common-assumption attribution

**Function:** Identifies a structural assumption embedded in a family of methods that your work does not share. Sets up a principled distinction.

**Abstract pattern:**
> [APPROACH CLASS] typically assumes [ASSUMPTION], which may not hold when [FAILURE CONDITION].

**Templates:**
- [APPROACH CLASS] typically assumes access to [RESOURCE], a requirement we relax.
- Standard [APPROACH] assumes [PROPERTY] of the input distribution, which does not hold in [OUR DOMAIN].
- Most prior methods assume [CONDITION]; our approach operates without this assumption.

**Weak:**
> Existing methods have an assumption that is wrong in some cases.

**Improved:**
> Most [APPROACH CLASS] methods assume that [ASSUMPTION], which fails when [CONDITION] — a common scenario in [TARGET DOMAIN].

**Notes:** Attribute the assumption to a class, not a single paper, unless one paper introduced the assumption explicitly. Follow immediately with evidence that the assumption fails in your target setting. Do not claim an assumption is "wrong" — claim it "does not hold" in a specific regime.

---

---

## 6. Limitation Framing

---

### 6.1 Explicit limitation flag

**Function:** Names and bounds a limitation before discussing its scope. Shows scientific honesty and prevents reviewers from raising it as an unacknowledged flaw.

**Abstract pattern:**
> One limitation of [APPROACH / EVALUATION] is [LIMITATION], which [CONSEQUENCE].

**Templates:**
- One limitation of this evaluation is [SCOPE RESTRICTION], which may understate performance in [BROADER SETTING].
- A limitation of [METHOD] is its reliance on [RESOURCE], which may not be available in [CONSTRAINED SETTING].
- One limitation of our analysis is that [CONDITION], which prevents [STRONGER CLAIM].

**Weak:**
> There are some limitations but overall the method is good.

**Improved:**
> One limitation of [APPROACH] is [SPECIFIC LIMITATION]: [REASON IT MATTERS]. We mitigate this by [ACTION], though [REMAINING CONCERN].

**Notes:** Name the limitation precisely before explaining its impact. "Some limitations" without enumeration signals defensiveness. At top venues, one to three specific, named limitations are expected and respected. Do not bury limitations in footnotes.

---

### 6.2 Future work deferral

**Function:** Explicitly delineates what is not attempted, turning an omission into a deliberate scoping decision rather than an oversight.

**Abstract pattern:**
> [EXTENSION] is beyond the scope of this work; we leave [SPECIFIC ASPECT] for future investigation.

**Templates:**
- Exploring [DIRECTION] is beyond the scope of this work; we leave it for future investigation.
- A systematic study of [FACTOR] would require [RESOURCE]; we treat this as an avenue for future work.
- We do not [ACTION] in this paper, as it would require [CONSTRAINT]; this remains an open problem.

**Weak:**
> We didn't do [X] but maybe someone could do it later.

**Improved:**
> A principled study of [FACTOR] would require [CONTROLLED SETTING] we do not have access to; we leave this as an explicit open problem.

**Notes:** "We leave X to future work" has become formulaic — strengthen it by saying *why* this paper cannot address it and what would be needed. Do not use this to defer the core claim of the paper. Limit to two or three deferrals in a limitations section.

---

### 6.3 Comparison fairness caveat

**Function:** Prevents misrepresenting a baseline's ceiling by acknowledging that it was not exhaustively tuned.

**Abstract pattern:**
> We do not optimize [COMPONENT] for [COMPARISON SYSTEM]; better results may be achievable with additional tuning.

**Templates:**
- We use default hyperparameters for [BASELINE], and tuning may yield higher [METRIC].
- [BASELINE] is run without [ADDITIONAL TRAINING / RESOURCE] that its authors report as beneficial.
- We report [BASELINE] under the same training budget as [OUR METHOD] for a fair comparison.

**Weak:**
> The baseline probably could have done better if we had tuned it more.

**Improved:**
> We use publicly released checkpoints for [BASELINE] without further finetuning; results with task-specific tuning may differ.

**Notes:** Include this caveat whenever a comparison system is not run under its optimal conditions. Reviewers who know the compared system will check this. Frame it as a fairness acknowledgment, not an excuse for the baseline underperforming.

---

### 6.4 Metric adequacy caveat

**Function:** Acknowledges that the primary metric is an imperfect proxy and shows the limitation was anticipated.

**Abstract pattern:**
> [METRIC] is an imperfect measure of [QUALITY DIMENSION]; we supplement it with [ALTERNATIVE EVALUATION].

**Templates:**
- [METRIC] correlates imperfectly with human judgment on [TASK]; we additionally report [HUMAN EVAL] on a held-out subset.
- Automated [METRIC] may not capture [PROPERTY]; our human evaluation addresses this directly.
- [METRIC] favors [BIAS], which may inflate scores for [SYSTEM TYPE]; we interpret it alongside [ALTERNATIVE].

**Weak:**
> [METRIC] is not perfect but it's commonly used so we used it.

**Improved:**
> [METRIC] is widely used for comparability, but prior work has shown it correlates weakly with [HUMAN JUDGMENT DIMENSION] on [TASK] ([CITATION]); we therefore supplement it with [ALTERNATIVE EVALUATION].

**Notes:** Always cite the paper that showed the metric's limitation, if one exists. "Widely used for comparability" is the standard justification for keeping an imperfect metric — it is honest and accepted. Do not abandon the metric entirely unless you have a principled alternative.

---

### 6.5 Domain generalization caveat

**Function:** Restricts a generalization claim to the evaluated setting, preventing reviewers from asking about out-of-distribution performance.

**Abstract pattern:**
> Performance improvements observed on [DATASET A] may not transfer to [DATASET B] due to [DOMAIN DIFFERENCE].

**Templates:**
- These results are specific to [DOMAIN / LANGUAGE / MODALITY]; generalization to [ALTERNATIVE] is not guaranteed.
- Whether [FINDING] holds beyond [EVALUATED SETTING] remains to be established.
- Improvements on [TASK A] may not transfer to [TASK B], which requires [DIFFERENT CAPABILITY].

**Weak:**
> Our method works well but might not work everywhere.

**Improved:**
> Results on [BENCHMARK] may not transfer to [RELATED DOMAIN] because [SPECIFIC REASON]; we encourage evaluation on [DIVERSE BENCHMARK SET] in follow-up work.

**Notes:** Be specific about why transfer might fail — not "different domain" but "different [distribution property]." Suggesting what follow-up work should do is both honest and constructive.

---

---

## 7. Result Statements

---

### 7.1 "We find that" discovery framing

**Function:** Reports an empirical observation where the finding itself is the main point and interpretation is secondary.

**Abstract pattern:**
> We find that [FINDING], indicating [INTERPRETATION].

**Templates:**
- We find that [FINDING], suggesting that [INTERPRETATION].
- We find no significant effect of [FACTOR] on [METRIC], indicating that [NULL INTERPRETATION].
- We find that [METHOD] consistently outperforms [BASELINE] across all [CONDITIONS], with gains of [RANGE].

**Weak:**
> Our results show that the model is very good.

**Improved:**
> We find that [METHOD] outperforms [BASELINE] by [VALUE] on [METRIC] ([TEST]: p < [THRESHOLD]), with consistent gains across all [N] evaluation conditions.

**Notes:** "We find" is for observations; "we show" is for demonstrations of a claim; "we demonstrate" is for proofs of concept. Use "we observe" as a near-synonym for "we find." Always include the metric name and comparison point. Report variance or significance when available.

---

### 7.2 Metric delta statement

**Function:** States a quantitative improvement with all information needed for the reader to assess it: method, baseline, metric, magnitude.

**Abstract pattern:**
> [METHOD] achieves a [RELATIVE / ABSOLUTE] improvement of [VALUE] over [BASELINE] on [METRIC].

**Templates:**
- [METHOD] achieves [VALUE] on [METRIC], a [VALUE]-point improvement over the previous best ([BASELINE]).
- Relative to [BASELINE], [METHOD] reduces [ERROR METRIC] by [VALUE] while matching [BASELINE] on [OTHER METRIC].
- On [BENCHMARK], [METHOD] achieves [VALUE] [METRIC], compared to [BASELINE VALUE] for [BASELINE].

**Weak:**
> Our method is significantly better than the baseline.

**Improved:**
> [METHOD] achieves [VALUE] [METRIC] on [TEST SET], improving over [BASELINE] ([VALUE]) by [ABSOLUTE DELTA] points ([RELATIVE DELTA] relative improvement).

**Notes:** Report both absolute and relative deltas when the absolute is small. Always specify the test set, not just the dataset. "Significantly" requires a statistical test — use "substantially" or report the p-value. State the variance (±) or confidence interval when possible at NeurIPS/ICML.

---

### 7.3 Null / weak effect finding

**Function:** Reports a negative or non-significant result as an informative finding rather than a failure.

**Abstract pattern:**
> We observe [SMALL / NO SIGNIFICANT] effect of [INTERVENTION] on [METRIC], suggesting [INTERPRETATION].

**Templates:**
- We observe no significant effect of [FACTOR] on [METRIC] (Δ = [VALUE], p = [VALUE]), suggesting [INTERPRETATION].
- Removing [COMPONENT] yields only a [VALUE]-point drop on [METRIC], indicating [COMPONENT] contributes minimally.
- [METHOD] shows no improvement over [BASELINE] on [TASK], consistent with [PRIOR FINDING].

**Weak:**
> The ablation didn't really do much so [COMPONENT] doesn't matter.

**Improved:**
> Ablating [COMPONENT] reduces [METRIC] by [VALUE] points (from [VALUE] to [VALUE]), suggesting [COMPONENT] accounts for [FRACTION] of the total gain attributed to [APPROACH].

**Notes:** Null results are important at top venues. Frame them as informative, not as failures. Quantify "no effect" — say the delta was [VALUE] rather than leaving the reader to infer from a table. "Suggesting" is appropriate for interpreting null results since causal inference from ablations is limited.

---

### 7.4 Consistent trend across conditions

**Function:** Strengthens a result by showing it is not an artifact of a single setting, model, or dataset.

**Abstract pattern:**
> Across all [CONDITIONS], [FINDING] holds, demonstrating [ROBUSTNESS CLAIM].

**Templates:**
- Across [N] benchmarks, [METHOD] consistently outperforms [BASELINE], with [RANGE] improvement.
- The trend holds across all model sizes ([RANGE]), suggesting [INTERPRETATION] is scale-independent.
- Results are consistent across [LANGUAGES / DOMAINS / SEEDS], with variance below [THRESHOLD].

**Weak:**
> The results are generally good in different settings.

**Improved:**
> Across all [N] [CONDITION TYPE] (Table [X]), [METHOD] outperforms [BASELINE] by [RANGE] [METRIC] points, with no condition showing a reversal.

**Notes:** Saying "all conditions" is a strong claim — verify it before writing it. If one condition reverses, report it as an interesting exception rather than suppressing it. Reference the specific table or figure so the reader can verify the claim.

---

### 7.5 Aggregate vs. category breakdown

**Function:** Prevents a strong aggregate number from obscuring variance across subgroups. Shows the result is understood at fine granularity.

**Abstract pattern:**
> While [SYSTEM] performs well on average, performance varies across [CATEGORIES], with [CASE] being notably [STRONGER / WEAKER].

**Templates:**
- Aggregate [METRIC] masks variance: [CATEGORY A] reaches [VALUE] while [CATEGORY B] lags at [VALUE].
- Average performance is [VALUE], though [SUBCATEGORY] accounts for most of the gain ([VALUE] vs. [BASELINE VALUE]).
- Per-[CATEGORY] results (Table [X]) reveal that [SYSTEM] struggles on [CATEGORY], suggesting [INTERPRETATION].

**Weak:**
> The average result is good but some categories are harder.

**Improved:**
> Although average [METRIC] is [VALUE], performance on [DIFFICULT CATEGORY] is only [VALUE] — [GAP] points below the mean — suggesting [INTERPRETATION].

**Notes:** Category-level analysis is expected in NeurIPS/ICML evaluation sections. Aggregate-only reporting is a common reviewer complaint. Accompany this sentence with a per-category table.

---

---

## 8. Sentence and Paragraph Transitions

---

### 8.1 Section roadmap

**Function:** Closes the introduction by giving the reader a navigational map of the paper. Conventional and expected at all top venues.

**Abstract pattern:**
> Section [X] describes [COMPONENT]; Section [Y] presents [COMPONENT]; Section [Z] reports [COMPONENT].

**Templates:**
- We describe [COMPONENT] in Section [X], present [EXPERIMENT] in Section [Y], and discuss implications in Section [Z].
- The remainder of the paper is organized as follows: Section [X] reviews [TOPIC]; Section [Y] describes [METHOD]; Section [Z] reports results.

**Weak:**
> The rest of the paper is organized in the usual way.

**Improved:**
> Section [X] formalizes [PROBLEM] and introduces notation; Section [Y] describes [METHOD]; Sections [Z]–[W] report results on [BENCHMARK A] and [BENCHMARK B]; Section [V] discusses limitations and future directions.

**Notes:** Match the section descriptions precisely to what the sections actually contain. Reviewers use this roadmap to navigate — a mismatch erodes trust. Use past tense for the abstract ("we showed") but future tense here ("Section X describes").

---

### 8.2 "This suggests" implication pivot

**Function:** Lifts the discussion from data to meaning. The canonical move between a result sentence and an interpretation sentence.

**Abstract pattern:**
> This suggests that [INTERPRETATION], which [IMPLICATION].

**Templates:**
- This suggests that [BEHAVIOR] is driven by [MECHANISM], rather than [ALTERNATIVE].
- These results suggest that [APPROACH] is robust to [VARIATION], making it suitable for [PRACTICAL SETTING].
- Together, these findings suggest that [HIGHER-LEVEL CLAIM].

**Weak:**
> So the model works because of [MECHANISM].

**Improved:**
> These results suggest that [BEHAVIOR] is attributable to [MECHANISM], consistent with the hypothesis that [THEORETICAL CLAIM].

**Notes:** "Suggests" is weaker than "shows" or "demonstrates" — use it when the evidence is consistent with but not conclusive for the interpretation. Do not chain more than two "this suggests" pivots in succession or the logic becomes circular.

---

### 8.3 Elaboration pivot with "specifically"

**Function:** Drills from a general claim to a concrete detail without starting a new paragraph. Signals zoom-in, not topic shift.

**Abstract pattern:**
> Specifically, [DETAIL THAT REFINES THE PRECEDING CLAIM].

**Templates:**
- Specifically, [COMPONENT] is implemented as [IMPLEMENTATION DETAIL].
- Specifically, we find that [SPECIFIC VERSION OF THE PRIOR GENERAL CLAIM], with [QUANTIFICATION].
- Specifically, the gains are concentrated in [SUBCATEGORY], which accounts for [FRACTION] of the overall improvement.

**Weak:**
> To give more details, the model does X.

**Improved:**
> Specifically, [COMPONENT] applies [MECHANISM] at [GRANULARITY], operating over [SCOPE] rather than [ALTERNATIVE].

**Notes:** "Specifically" should always be followed by a sentence that is strictly more specific than the one before it. Do not use it to introduce a tangent. Alternatives: "Concretely", "In particular", "More precisely."

---

### 8.4 Additive accumulation pivot

**Function:** Adds a second piece of evidence to an already-established claim, signaling accumulation rather than contrast or elaboration.

**Abstract pattern:**
> [ADDITIVE CONNECTOR], [SUPPORTING FINDING] further confirms [MAIN CLAIM].

**Templates:**
- Additionally, [SUPPORTING EXPERIMENT] yields consistent results, confirming [CLAIM] is not an artifact of [SETTING].
- Furthermore, [OBSERVATION] aligns with [PRIOR FINDING], reinforcing [INTERPRETATION].
- Beyond [MAIN RESULT], [SECONDARY FINDING] suggests [EXTENSION OF CLAIM].

**Weak:**
> Also the results show the same thing in the other experiment too.

**Improved:**
> Furthermore, [SECONDARY EXPERIMENT] — which varies [FACTOR] not controlled in the primary setting — yields [CONSISTENT FINDING], suggesting the result is not an artifact of [SPECIFIC CONDITION].

**Notes:** "Additionally" is neutral accumulation; "furthermore" implies the added point is emphatic; "moreover" is slightly stronger still. Do not use "also" as a sentence opener in formal writing. Reserve "beyond" for scope expansion (section 8.6 pattern).

---

### 8.5 Temporal sequencing of methods

**Function:** Sequences procedural steps without overusing bullet points. Appropriate in methods sections.

**Abstract pattern:**
> First, [ACTION 1]. We then [ACTION 2]. Finally, [ACTION 3].

**Templates:**
- First, [STEP 1]. We then apply [STEP 2] to obtain [INTERMEDIATE]. Finally, [STEP 3] produces [OUTPUT].
- [STEP 1] is performed offline; at inference, we [STEP 2] and then [STEP 3].

**Weak:**
> We do step 1 and then step 2 and then step 3.

**Improved:**
> First, [STEP 1] is applied to [INPUT], yielding [INTERMEDIATE]. We then [STEP 2], which [TRANSFORMS INTERMEDIATE]. Finally, [STEP 3] produces [OUTPUT] used for [DOWNSTREAM].

**Notes:** "First… then… finally" works for three steps. For four or more, switch to a numbered list. "Subsequently" is a valid synonym for "then." Avoid "lastly" — use "finally." Do not use "firstly" in American-English venues.

---

---

## 9. Academic Verb Phrases (by function)

---

### Proposing and Introducing

| Phrase | Tone / Use |
|---|---|
| We propose [METHOD], which [PROPERTY]. | Standard for algorithmic contributions. |
| We introduce [ARTIFACT], a [DESCRIPTOR]. | Preferred for datasets and benchmarks. |
| We present [FRAMEWORK] for [TASK]. | Slightly broader than "propose"; fits systems papers. |
| We develop [RESOURCE] consisting of [SPEC]. | Appropriate for engineering-heavy contributions. |
| We formalize [CONCEPT] as [MATHEMATICAL OBJECT]. | Use when giving a formal treatment to an informal notion. |
| We define [TERM] as [DEFINITION]. | For notation-heavy papers; sets up later use. |

---

### Describing Methods

| Phrase | Use |
|---|---|
| We compute [QUANTITY] by [PROCEDURE]. | Precise; use in methods. |
| We extract [REPRESENTATION] from [SOURCE]. | For representation analysis papers. |
| We parameterize [COMPONENT] as [FORM]. | For model architecture descriptions. |
| We train [MODEL] on [DATA] using [OBJECTIVE]. | Standard training sentence. |
| We apply [TRANSFORMATION] to [INPUT] to obtain [OUTPUT]. | Pipeline steps. |
| We sample [N] [ITEMS] from [DISTRIBUTION]. | For stochastic procedures. |

---

### Making Comparisons

| Phrase | Use |
|---|---|
| We compare [METHOD] to [BASELINES] on [BENCHMARK]. | Standard evaluation sentence. |
| We evaluate [SYSTEM] against [SET OF BASELINES]. | Slight emphasis on the evaluation act. |
| We benchmark [METHOD] on [N] [TASK TYPE] tasks. | Use "benchmark" as a verb only when the task set is standard. |
| We replicate [PRIOR SYSTEM] as a comparison point. | Shows you ran the baseline yourself. |
| We adapt [PRIOR METHOD] to [OUR SETTING] by [MODIFICATION]. | Use when a baseline needed modification; prevents reviewer objections. |

---

### Reporting and Interpreting

| Phrase | Use |
|---|---|
| We find that [FINDING]. | Empirical observation. |
| We observe [BEHAVIOR] in [CONDITION]. | Slightly more descriptive than "find." |
| We demonstrate that [CLAIM]. | Stronger; implies controlled demonstration. |
| We show that [CLAIM], as evidenced by [SUPPORT]. | Use when evidence is immediate and cited. |
| We confirm that [REPLICATION]. | Specifically for replicating a prior result. |
| We verify [CLAIM] using [PROCEDURE]. | Validation-focused. |
| We report [METRIC] for all systems in Table [X]. | For directing readers to tabular results. |

---

### Acknowledging

| Phrase | Use |
|---|---|
| We note that [CAVEAT]. | Mild, parenthetical acknowledgment. |
| We acknowledge that [LIMITATION]. | Stronger; invites the reader to take the limitation seriously. |
| We leave [EXTENSION] to future work. | Scope delimitation. |
| We do not [ACTION]; this [CONSEQUENCE / REASON]. | Explicit scope exclusion. |

---

---

## 10. Noun Phrase Constructions

---

### Dataset and Benchmark NPs

- **a [SCALE]-[UNIT] [CONTENT TYPE] dataset** — *"a 50k-utterance dialogue dataset"*
- **a manually annotated [CORPUS / BENCHMARK] of [DOMAIN] [UNIT TYPE]**
- **a [ADJECTIVE] benchmark for evaluating [CAPABILITY] in [SETTING]**
- **[N] [LABEL TYPE]-annotated [UNIT TYPE] spanning [RANGE] [DIMENSION]**
- **a contrastive set of [POSITIVE] and [NEGATIVE] [EXAMPLE TYPE]**
- **a held-out test set of [N] [UNIT TYPE] not seen during training**

### Model and Architecture NPs

- **a [ADJECTIVE] [ARCHITECTURE] for [TASK]** — *"a retrieval-augmented Transformer for QA"*
- **a [SCALE]-parameter [MODEL FAMILY] model finetuned for [TASK]**
- **a [LIGHTWEIGHT / PARAMETER-EFFICIENT] variant of [BASE MODEL]**
- **a [DIMENSIONALITY]-dimensional [REPRESENTATION TYPE] of [SEMANTIC CONTENT]**
- **a [TRAINING OBJECTIVE]-trained encoder pretrained on [DATA SOURCE]**

### Evaluation NPs

- **[METRIC] on the [SPLIT] set of [BENCHMARK]**
- **human ratings on a [N]-point Likert scale for [DIMENSION A] and [DIMENSION B]**
- **an inter-annotator agreement of [VALUE] ([METRIC]) on [TASK]**
- **automatic [METRIC] alongside human evaluation on [N] sampled outputs**
- **per-[CATEGORY] [METRIC] breakdown across [N] conditions**

### Analytical and Conceptual NPs

- **a linear direction in the [LAYER / COMPONENT]'s [REPRESENTATION SPACE] corresponding to [CONCEPT]**
- **[PROPERTY]-relevant [SIGNAL / FEATURE] encoded in [COMPONENT]**
- **the [ADJECTIVE] distribution of [LABEL TYPE] across [CATEGORIES]**
- **[CONCEPT] representations at layer [L] of [MODEL]**
- **the [DIRECTION]-of-[CONCEPT] vector extracted via [METHOD]**

### Scope and Coverage NPs

- **a [ADJECTIVE] range of [DIMENSION], from [LOW END] to [HIGH END]**
- **[N] categories covering [RANGE DESCRIPTOR] [PHENOMENON TYPE]**
- **a broader coverage of [ASPECT] than [PRIOR RESOURCE]**
- **all [N] [CONDITION TYPE] examined in this paper**

---

*All patterns are abstracted from published NLP / ML papers and are intended for adaptation, not verbatim use.*
