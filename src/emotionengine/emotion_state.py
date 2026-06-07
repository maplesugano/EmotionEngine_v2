"""Utilities for computing s(x): the current emotional state of input text x.

The CAA training data is formatted as:
    make_instruction_prefix(base_text) + rewrite_text

where the last-token activation is taken at the final token of ``rewrite_text``.
To place s(x) in the same activation space, x is formatted as:
    make_instruction_prefix(x_neu) + x
where x_neu is the affectively neutral rewrite of x (obtained via OpenAI API).

Usage
-----
    # Precompute μ_neutral once:
    mu = compute_neutral_mean(
        "activation/emotion_rewrites/neutral_paraphrase_residual_stream.npy",
        "activation/emotion_rewrites/neutral_mean_layer13.npy",
    )

    # At inference time:
    # 1. Get x_neu from OpenAI (neutral rewrite of x)
    # 2. Extract s(x) in the same activation space as the CAA training data:
    #       instruction_prefix(x_neu) + x
    #    mirrors the CAA format: instruction_prefix(base_text) + emotion_rewrite
    s_raw = extract_current_state(model, tokenizer, x, x_neu, device)
    mu    = load_neutral_mean("activation/emotion_rewrites/neutral_mean_layer13.npy")
    caa   = np.load("activation/emotion_rewrites/caa_emotion_directions.npz")["caa_pooled"]
    profile = compute_emotion_profile(s_raw, mu, caa)
    # → {"joy": 0.42, "sadness": 1.83, ...}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.constants import PLUTCHIK_EMOTIONS
from utils.model_utils import extract_batch
from utils.prompts import NEUTRAL_PARAPHRASE_SCHEMA, NEUTRAL_PARAPHRASE_SYSTEM_PROMPT
from utils.text_utils import make_instruction_prefix

_NEUTRALISE_SYSTEM_PROMPT: str = NEUTRAL_PARAPHRASE_SYSTEM_PROMPT
_NEUTRALISE_SCHEMA: dict = NEUTRAL_PARAPHRASE_SCHEMA

_NEUTRALISE_MODEL = "gpt-4.1-mini"


def neutralise(
    text: str,
    client: openai.OpenAI | None = None,
    model: str = _NEUTRALISE_MODEL,
) -> str:
    """Return an affectively neutral rewrite of *text* via OpenAI API.

    Uses the identical system prompt and JSON schema as
    ``build_neutral_paraphrase_dataset.py`` so that x_neu lives in the same
    distribution as the CAA training baselines.

    Parameters
    ----------
    text : raw input sentence to neutralise
    client : openai.OpenAI instance; created from env OPENAI_API_KEY if None
    model : OpenAI model name (default: gpt-4.1-mini)

    Returns
    -------
    x_neu : affectively neutral rewrite of *text*
    """
    if client is None:
        client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _NEUTRALISE_SYSTEM_PROMPT},
            {"role": "user",   "content": f'Utterance: "{text}"'},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name":   "neutral_paraphrase",
                "strict": True,
                "schema": _NEUTRALISE_SCHEMA,
            },
        },
    )

    data = json.loads(response.choices[0].message.content)
    return data["neutral_paraphrase"]


# Layer 13 is at index 2 in hook_layers = [8, 10, 13, 16, 19, 22].
_LAYER = 13
_LAYER_POS = 2

EMOTION_ORDER: list[str] = PLUTCHIK_EMOTIONS


def compute_neutral_mean(
    neutral_act_path: str | Path,
    out_path: str | Path,
    layer_pos: int = _LAYER_POS,
) -> np.ndarray:
    """Compute μ_neutral from neutral-paraphrase activations and cache to disk.

    Parameters
    ----------
    neutral_act_path:
        Path to ``neutral_paraphrase_residual_stream.npy``, shape (N, L, D).
    out_path:
        Where to write the resulting (D,) float32 vector.
    layer_pos:
        Index into the L dimension corresponding to layer 13 (default: 2).

    Returns
    -------
    mu : np.ndarray, shape (D,), float32
    """
    acts = np.load(str(neutral_act_path), mmap_mode="r")  # (N, L, D)
    mu = acts[:, layer_pos, :].mean(axis=0).astype(np.float32)  # (D,)
    np.save(str(out_path), mu)
    return mu


def load_neutral_mean(path: str | Path) -> np.ndarray:
    """Load precomputed μ_neutral from disk.  Shape: (D,), float32."""
    return np.load(str(path))


def _format_for_extraction(
    text: str,
    text_neutral: str,
    tokenizer: AutoTokenizer,
) -> str:
    """Format x for activation extraction in the CAA training space.

    CAA training format:
        instruction_prefix(base_text) + emotion_rewrite_text

    s(x) mirrors this exactly:
        instruction_prefix(x_neu) + x

    where x_neu is the neutralised rewrite of x (base), and x is the
    emotionally coloured input (response).
    """
    return make_instruction_prefix(text_neutral, tokenizer) + text


def extract_current_state(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    text_neutral: str,
    device: str,
    max_length: int = 512,
) -> np.ndarray:
    """Extract s_raw(x): last-token residual at layer 13 for input text x.

    Parameters
    ----------
    text : raw input sentence (emotionally coloured; the response in the prompt)
    text_neutral : neutralised rewrite of text (the base in the prompt);
        obtain via OpenAI API before calling this function
    device : torch device string (e.g. ``"cuda"``)
    max_length : tokenizer truncation limit

    Returns
    -------
    s_raw : np.ndarray, shape (D,), float32
    """
    formatted = _format_for_extraction(text, text_neutral, tokenizer)
    acts = extract_batch(
        model, tokenizer,
        [formatted],
        hook_layers=[_LAYER],
        device=device,
        max_length=max_length,
    )
    # acts: (batch=1, L=1, D) → (D,)
    return acts[0, 0, :].numpy()


def compute_emotion_profile(
    s_raw: np.ndarray,
    mu_neutral: np.ndarray,
    caa_pooled: np.ndarray,
    layer_pos: int = _LAYER_POS,
) -> dict[str, float]:
    """Project s(x) = s_raw - μ_neutral onto each CAA emotion direction.

    Parameters
    ----------
    s_raw : (D,) float32 — output of :func:`extract_current_state`
    mu_neutral : (D,) float32 — output of :func:`load_neutral_mean`
    caa_pooled : (E, L, D) float32 — unit-normalised pooled CAA directions,
        loaded from ``caa_emotion_directions.npz["caa_pooled"]``
    layer_pos : index into L for layer 13 (default: 2)

    Returns
    -------
    dict mapping emotion name → scalar dot-product score.
    Positive values indicate alignment with that emotion direction.
    """
    s_centered = s_raw - mu_neutral        # (D,)
    vecs = caa_pooled[:, layer_pos, :]     # (E, D)
    scores = (vecs @ s_centered).tolist()  # (E,)
    return dict(zip(EMOTION_ORDER, scores))
