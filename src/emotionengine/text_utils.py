"""Chat-template text-formatting helpers shared across extraction scripts.

Both ``extract_residual_stream.py`` and
``extract_emotion_intensity_residual_stream.py`` use the same instruction
framing so that emotion/affective contrasts are not conflated with base-text
variation.
"""

from __future__ import annotations

from transformers import AutoTokenizer

# Shared instruction prefix for all affective/emotion extraction passes.
# Only ``{base}`` is substituted; all rewrite variants share the same prefix
# so that h(A) − h(B) captures purely affective contrast.
INSTRUCTION_TEMPLATE = (
    "Please rewrite this so it aligns with my feelings more: base: {base}"
)


def make_instruction_prefix(base_text: str, tokenizer: AutoTokenizer) -> str:
    """Build the chat-template prefix for one base text.

    ``tokenizer.apply_chat_template`` inserts model-specific special tokens
    (e.g. ``<|im_start|>``, ``[INST]``, ``<|eot_id|>``).
    ``add_generation_prompt=True`` appends the assistant-turn header so the
    model is in generation mode when the continuation token is processed.
    """
    instruction = INSTRUCTION_TEMPLATE.format(base=base_text)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=False,
    )
