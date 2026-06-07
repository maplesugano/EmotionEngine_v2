"""System prompts and JSON schemas shared across pipeline scripts.

Centralised here so that the same prompt is used during dataset construction
(batch API scripts) and at inference time (emotion_state.py), guaranteeing the
model always sees identical instruction framing.
"""
from __future__ import annotations

from typing import Any

from utils.constants import INTENSITY_LEVELS, PLUTCHIK_EMOTIONS

NEUTRAL_PARAPHRASE_SYSTEM_PROMPT: str = """\
You are an expert in affective linguistics and conversational pragmatics.

TASK
Rewrite the given conversational utterance into an affectively neutral control sentence.

The goal is not to summarise or shorten the utterance. The goal is to preserve the concrete situation, events, entities, and speaker's practical point, while removing emotional tone, sentiment, and evaluative wording.

CONSTRAINTS
- Preserve the same situation, entities, events, and practical intent.
- Do not summarise by deleting clauses unless the clause contains only emotional emphasis and no concrete information.
- Replace emotional evaluations with neutral factual or cognitive descriptions when possible.
- Keep approximately the same amount of information as the original.
- Use plain, matter-of-fact conversational language.
- Do not add new facts.
- Return only valid JSON.

MATCHING EXAMPLES
Original : "I finally got the results back and I'm so relieved that everything looks normal."
Neutral  : "I got the results back, and everything looks normal."

Original : "I can't believe they ignored my message again. That's so frustrating."
Neutral  : "They did not respond to my message again."

Original : "The trip was amazing, and I loved every minute of it."
Neutral  : "The trip went well, and I spent the full time there."
"""

NEUTRAL_PARAPHRASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "base_text":               {"type": "string"},
        "neutral_paraphrase":      {"type": "string"},
        "meaning_preserved_score": {"type": "number"},
    },
    "required": ["base_text", "neutral_paraphrase", "meaning_preserved_score"],
    "additionalProperties": False,
}

EMOTION_REWRITE_SYSTEM_PROMPT: str = """\
You are an expert in affective linguistics, appraisal theory, and conversational pragmatics.

TASK
Given a conversational utterance and a target INTENSITY LEVEL, produce eight
emotion-category rewrites — one for each of Plutchik's eight basic emotions:
joy, trust, fear, surprise, sadness, disgust, anger, and anticipation.

CORE CONSTRAINTS
- Preserve the same underlying situation and core facts.
- **Rewrite the speaker's reaction to the situation, not the situation itself.**
- Emotion should emerge from the speaker's interpretation, assumptions, expectations, and conversational framing.
- Avoid explicit emotion statements such as "I was happy", "I felt anxious", or "that made me angry".
- Prefer natural reactions, inferences, speculation, complaints, hopes, doubts, and observations.
- The utterance should sound like something a real person would actually say.
- Keep emotional intensity matched across all categories.

INTENSITY LEVELS
  low    : the emotion is present but subtle, muted, understated
  medium : the emotion is clear and naturally expressed (conversational baseline)
  high   : the emotion is strong, explicit, and fully expressed

EMOTION DEFINITIONS  (Plutchik's wheel)
  joy          : happiness, contentment, relief, gratitude, amusement
  trust        : acceptance, admiration, confidence, serenity toward others
  fear         : anxiety, dread, worry, apprehension, unease
  surprise     : astonishment, wonder, bewilderment, unexpectedness
  sadness      : sorrow, disappointment, loneliness, grief
  disgust      : revulsion, contempt, aversion, distaste
  anger        : frustration, irritation, fury, resentment
  anticipation : expectancy, eagerness, interest, looking forward (or dreading)

MATCHING EXAMPLE  (intensity: medium)
Base: "I waited for hours and nobody came."
  joy:          "Nobody showed up, but honestly, I ended up having a pretty nice afternoon anyway."
  trust:        "Nobody came. Something must have come up on their end."

  fear:         "Nobody came... I hope nothing happened."
  surprise:     "Wait, seriously? Nobody came at all?"
  sadness:      "For hours, nobody came. I guess that says enough."
  disgust:      "I was standing here for hours. Nobody came. That's a pretty lousy way to treat someone."
  anger:        "Nobody came. What a complete waste of my time."
  anticipation: "Nobody came, but I kept thinking the next minute would be the one."

MATCHING EXAMPLE (intensity: high)
Base: "My manager said they wanted to talk to me tomorrow."
  joy: "Damn!! My manager wants to talk tomorrow! Maybe I'm finally getting some good news."
  trust: "My manager wants to talk tomorrow. It's probably fine—they've always been straightforward with me."
  fear: "My manager wants to talk tomorrow? Oh no. What happened? Did I mess something up?"
  surprise: "Wait, what? Chat with my manager?? Tomorrow? That came completely out of nowhere."
  sadness: "My manager wants to talk tomorrow... Great. As if this week wasn't already bad enough."
  disgust: "My manager wants to talk tomorrow. Why do people always do this vague 'we need to talk' thing?"
  anger: "My manager wants to talk tomorrow? Then just tell me now instead of making me sit with it."
  anticipation: "My manager wants to talk tomorrow? Now I'm going to be thinking about that all night."

Notice that all eight rewrites are at equal expressiveness (none is understated while another is overwrought).
This matched-intensity constraint is essential.
"""

EMOTION_REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "base_text":               {"type": "string"},
        "intensity_level":         {"type": "string", "enum": INTENSITY_LEVELS},
        "meaning_preserved_score": {"type": "number"},
        **{f"{e}_rewrite": {"type": "string"} for e in PLUTCHIK_EMOTIONS},
    },
    "required": [
        "base_text",
        "intensity_level",
        "meaning_preserved_score",
        *[f"{e}_rewrite" for e in PLUTCHIK_EMOTIONS],
    ],
    "additionalProperties": False,
}

LLM_JUDGE_SYSTEM_PROMPT: str = (
    "You are an expert in affective science and computational linguistics. "
    "Respond ONLY with a valid JSON object as specified in the user message."
)
