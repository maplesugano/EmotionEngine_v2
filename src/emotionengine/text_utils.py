"""Chat-template text-formatting helpers shared across extraction scripts.

Both ``extract_residual_stream.py`` and
``extract_emotion_intensity_residual_stream.py`` use the same instruction
framing so that emotion/affective contrasts are not conflated with base-text
variation.
"""

from __future__ import annotations

import re

from transformers import AutoTokenizer

# Shared instruction prefix for all affective/emotion extraction passes.
# Only ``{base}`` is substituted; all rewrite variants share the same prefix
# so that h(A) − h(B) captures purely affective contrast.
INSTRUCTION_TEMPLATE = (
    "Please rewrite this so it aligns with my feelings more.\n"
    "Start straight from the rewritten text without any preamble, and only provide **one** rewritten version.\n"
    "Base: {base}\n\n"
)


def make_instruction_prefix(
    base_text: str,
    tokenizer: AutoTokenizer,
    template: str = INSTRUCTION_TEMPLATE,
) -> str:
    """Build the chat-template prefix for one base text.

    ``tokenizer.apply_chat_template`` inserts model-specific special tokens
    (e.g. ``<|im_start|>``, ``[INST]``, ``<|eot_id|>``).
    ``add_generation_prompt=True`` appends the assistant-turn header so the
    model is in generation mode when the continuation token is processed.
    """
    instruction = template.format(base=base_text, base_text=base_text)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=False,
    )


import re

def extract_rewritten_text(text: str) -> str:
    """
    Post-process model outputs from rewrite prompts.

    Goals:
    - Remove assistant framing: "Here's a revised version..."
    - If the model gives multiple candidates, keep only the first candidate.
    - Remove labels like "**Formal**:", "Option 1:", "Rewrite:", "Base:".
    - Prefer quoted candidate text when a clear quote is present.
    - Preserve evidence of severe degeneration instead of over-cleaning it.
    """
    if not text:
        return text

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return cleaned

    # Remove common special tokens / chat remnants.
    cleaned = re.sub(r"</?s>|<\|.*?\|>", "", cleaned).strip()

    # Normalize whitespace but keep line structure for candidate detection.
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    if not lines:
        return ""

    cleaned = "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # 1. Remove leading assistant framing paragraphs.
    # ------------------------------------------------------------------
    lead_in_patterns = [
        r"^here(?:'s| is)\s+(?:a|the|an)?\s*(?:revised|rewritten|rephrased)?\s*version(?:\s+of\s+(?:the\s+)?(?:text|statement|conversation))?.*?:\s*",
        r"^here(?:'s| are)\s+(?:a few|some|several)?\s*(?:rewritten|revised|rephrased)?\s*(?:options|versions|alternatives).*?:\s*",
        r"^i can (?:rewrite|rephrase) it.*?:\s*",
        r"^sure[:,]?\s*",
        r"^of course[:,]?\s*",
        r"^certainly[:,]?\s*",
        r"^rewritten text:?\s*",
        r"^rewrite:?\s*",
        r"^revised version:?\s*",
    ]

    changed = True
    while changed:
        changed = False
        before = cleaned
        for pat in lead_in_patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        changed = cleaned != before

    # ------------------------------------------------------------------
    # 2. If output has a numbered/bulleted list, keep first item only.
    #    Handles:
    #    1. text
    #    1) text
    #    - text
    #    * text
    #    Option 1: text
    # ------------------------------------------------------------------
    list_item_pattern = re.compile(
        r"^\s*(?:"
        r"(?P<num>\d+)[\.)]\s+|"
        r"[-*•]\s+|"
        r"(?:option|version|alternative)\s*(?P<opt_num>\d+)?\s*[:\-]\s*"
        r")(?P<body>.+)$",
        flags=re.IGNORECASE,
    )

    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    first_item_parts: list[str] = []
    in_first_item = False

    for line in lines:
        m = list_item_pattern.match(line)

        if m:
            num = m.group("num") or m.group("opt_num")
            body = m.group("body").strip()

            # Start first item.
            if not in_first_item:
                if num is None or num == "1":
                    first_item_parts.append(body)
                    in_first_item = True
                    continue

            # Stop at second item.
            if in_first_item:
                break

        elif in_first_item:
            # Continuation of first item, unless it looks like a new heading.
            if re.match(r"^(?:option|version|alternative)\s*\d*\s*[:\-]", line, re.I):
                break
            first_item_parts.append(line)

    if first_item_parts:
        cleaned = " ".join(first_item_parts).strip()

    # ------------------------------------------------------------------
    # 3. Remove inline labels/headings often produced by the model.
    # ------------------------------------------------------------------
    label_patterns = [
        r"^\*\*[^*\n]{1,80}\*\*\s*[:\-]\s*",      # **Formal**:
        r"^__[^_\n]{1,80}__\s*[:\-]\s*",          # __Formal__:
        r"^[A-Z][A-Za-z /&,\-]{1,80}\s*[:\-]\s*", # Formal and polite:
        r"^(?:option|version|alternative)\s*\d*\s*[:\-]\s*",
        r"^base\s*[:\-]\s*",
        r"^text\s*[:\-]\s*",
        r"^rewrite\s*[:\-]\s*",
        r"^revised\s*[:\-]\s*",
        r"^standard(?:\s+tone)?\s*[:\-]\s*",
    ]

    # Repeat because outputs can be like "**Formal**: Base: ..."
    for _ in range(4):
        before = cleaned
        for pat in label_patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned == before:
            break

    # Remove parenthetical tone labels at the start: "(Formal and polite) ..."
    cleaned = re.sub(
        r"^\((?:formal|polite|friendly|casual|sincere|apologetic|optimistic|sarcastic|standard|neutral|angry|frustrated)[^)]{0,80}\)\s*[:\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # ------------------------------------------------------------------
    # 4. Prefer quoted candidate if the remaining text contains one.
    #    Useful for: **Formal**: "actual rewrite"
    # ------------------------------------------------------------------
    quote_matches = re.findall(r'["“]([^"”]{10,500})["”]', cleaned)
    if quote_matches:
        # Prefer the first quoted string that does not look like an instruction.
        for q in quote_matches:
            q_clean = q.strip()
            low = q_clean.lower()
            if not (
                low.startswith("base:")
                or low.startswith("rewrite:")
                or "option" in low[:30]
            ):
                cleaned = q_clean
                break

    # ------------------------------------------------------------------
    # 5. Cut after obvious second candidate markers that survived inline.
    # ------------------------------------------------------------------
    split_patterns = [
        r"\s+\b2[\.)]\s+",
        r"\s+\bOption\s+2\s*[:\-]\s+",
        r"\s+\bVersion\s+2\s*[:\-]\s+",
        r"\s+\bAlternative\s+2\s*[:\-]\s+",
    ]
    for pat in split_patterns:
        cleaned = re.split(pat, cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    # ------------------------------------------------------------------
    # 6. Remove remaining markdown bullets / emphasis.
    # ------------------------------------------------------------------
    cleaned = re.sub(r"^\s*[-*•]\s*", "", cleaned).strip()
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned).strip()
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned).strip()

    # Collapse whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Strip wrapping quotes and dangling separators.
    cleaned = cleaned.strip(" \"'“”‘’`")
    cleaned = re.sub(r"^[\-\:\–\—\s]+", "", cleaned).strip()

    return cleaned