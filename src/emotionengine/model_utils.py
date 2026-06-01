"""Shared model-loading and batched activation-extraction utilities.

Used by both ``scripts/extract_residual_stream.py`` and
``scripts/extract_emotion_intensity_residual_stream.py`` to eliminate
duplicated code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(
    cfg: dict[str, Any],
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Instantiate a HuggingFace causal LM and its tokenizer from a model config.

    Parameters
    ----------
    cfg:
        Dict loaded from ``model.yaml``.  Must contain ``"name"``; optional keys
        ``"dtype"`` (default ``"bfloat16"``) and ``"attn_implementation"``
        (default ``"sdpa"``).
    device:
        Torch device string passed to ``device_map`` (e.g. ``"cuda"``).
    """
    model_name: str = cfg["name"]
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map.get(cfg.get("dtype", "bfloat16"), torch.bfloat16)
    attn_impl: str = cfg.get("attn_implementation", "sdpa")

    logger.info(
        "Loading %s  dtype=%s  attn=%s  device=%s …",
        model_name, dtype, attn_impl, device,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Left-padding: for a batch with variable-length sequences, the last
    # position (-1) always corresponds to the final real token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map=device,
    )
    model.eval()
    logger.info("Model loaded.  d_model=%d", model.config.hidden_size)
    return model, tokenizer


def extract_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    hook_layers: list[int],
    device: str,
    max_length: int = 512,
) -> torch.Tensor:
    """Forward a mini-batch of texts and return last-token residual vectors.

    ``hidden_states`` returned by HF is a tuple of length ``n_layers + 1``:
      - index 0  : embedding output (before any transformer block)
      - index i≥1: output of transformer block i-1  (0-indexed)

    So block ``k`` (0-indexed) → ``hidden_states[k + 1]``.

    Returns
    -------
    Tensor of shape ``(batch, L, D)`` on CPU in float32, where ``L`` is
    ``len(hook_layers)`` and ``D`` is the model hidden size.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    layer_vecs = [
        hidden_states[layer + 1][:, -1, :].cpu().to(torch.float32)
        for layer in hook_layers
    ]
    return torch.stack(layer_vecs, dim=1)  # (batch, L, D)
