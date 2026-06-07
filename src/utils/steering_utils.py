"""Shared helpers for activation steering notebooks and experiments."""

from __future__ import annotations

import csv
import gc
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.text_utils import make_instruction_prefix

T = TypeVar("T")


def unit_np(vec: np.ndarray, axis: int | None = None, eps: float = 1e-12) -> np.ndarray:
    """Normalise *vec* to unit L2 norm.

    Parameters
    ----------
    axis:
        Axis along which to normalise.  When ``None`` (default) the whole array
        is treated as a single vector (backward-compatible behaviour).
    eps:
        Small constant added to the norm to avoid division by zero.
    """
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec, axis=axis, keepdims=(axis is not None))
    return vec / np.maximum(norm, eps)


def unit(v: np.ndarray) -> np.ndarray:
    """Normalise the last axis of *v* to unit L2 norm.

    Convenience alias for ``unit_np(v, axis=-1)`` matching the calling
    convention used throughout the analysis notebooks.
    """
    return unit_np(v, axis=-1)


def cosine_sim_matrix(vecs: np.ndarray) -> np.ndarray:
    """Return a (K, K) cosine similarity matrix for a (K, D) array of vectors."""
    vecs = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = vecs / norms
    return normed @ normed.T


def build_delta_vector(
    caa_vectors: np.ndarray,
    emotion_to_index: dict[str, int],
    target_emotion: str,
    alpha: float,
) -> np.ndarray | None:
    """Scale one unit-normalised steering direction by alpha."""
    if abs(float(alpha)) < 1e-12:
        return None
    emotion_idx = emotion_to_index[target_emotion]
    return unit_np(caa_vectors[emotion_idx]) * float(alpha)


def chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    """Split a list into fixed-size chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def flush_device_cache() -> None:
    """Release Python and CUDA caches when available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def steering_hook(
    steer_model: AutoModelForCausalLM,
    layer_idx: int,
    delta_vec: np.ndarray | None,
    steering_mode: str,
    hook_stats: dict[str, int],
) -> Iterable[None]:
    """Temporarily add an activation-steering hook at one transformer layer."""
    if steering_mode == "none" or delta_vec is None:
        yield
        return

    if steering_mode not in {"last_token", "completion_tokens"}:
        raise ValueError(f"Unknown steering_mode: {steering_mode}")

    layer_module = steer_model.model.layers[layer_idx]
    layer_device = next(layer_module.parameters()).device
    layer_dtype = next(layer_module.parameters()).dtype
    delta = torch.as_tensor(delta_vec, dtype=layer_dtype, device=layer_device).view(1, 1, -1)

    def _hook(_module: Any, _args: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None

        seq_len = int(hidden.shape[1])
        is_decode_step = seq_len == 1
        hook_stats["hook_calls"] += 1
        hook_stats["hook_decode_calls"] += int(is_decode_step)
        hook_stats["hook_prefill_calls"] += int(not is_decode_step)

        should_apply = steering_mode == "last_token" or (
            steering_mode == "completion_tokens" and is_decode_step
        )
        if not should_apply:
            return output

        steered_hidden = hidden.clone()
        steered_hidden[:, -1:, :] = steered_hidden[:, -1:, :] + delta
        hook_stats["hook_applied_calls"] += 1
        return (steered_hidden, *rest) if rest is not None else steered_hidden

    handle = layer_module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()
        del delta


def generate_rewrite_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_input_device: torch.device,
    base_texts: list[str],
    target_emotion: str,
    alpha: float,
    steering_mode: str,
    *,
    caa_vectors: np.ndarray,
    emotion_to_index: dict[str, int],
    prompt_template: str,
    layer_idx: int,
    max_new_tokens: int,
    do_sample: bool,
    text_postprocessor: Callable[[str], str],
) -> list[dict[str, Any]]:
    """Generate steered rewrites for a batch of base texts."""
    if not base_texts:
        return []

    prompts = [make_instruction_prefix(base_text, tokenizer, prompt_template) for base_text in base_texts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model_input_device)
    hook_stats = {
        "hook_calls": 0,
        "hook_prefill_calls": 0,
        "hook_decode_calls": 0,
        "hook_applied_calls": 0,
    }

    delta_vec = None
    if steering_mode != "none":
        delta_vec = build_delta_vector(
            caa_vectors,
            emotion_to_index,
            target_emotion,
            alpha,
        )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }

    try:
        with torch.inference_mode():
            with steering_hook(model, layer_idx, delta_vec, steering_mode, hook_stats):
                out_ids = model.generate(**inputs, **gen_kwargs)

        prompt_width = inputs["input_ids"].shape[1]
        batch_results = []
        for row_idx in range(len(base_texts)):
            new_ids = out_ids[row_idx, prompt_width:]
            generated_text = tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            generated_text = text_postprocessor(generated_text)
            batch_results.append({"generated_text": generated_text, **hook_stats})
    except torch.cuda.OutOfMemoryError:
        flush_device_cache()
        batch_results = [
            {"generated_text": "[OOM]", **hook_stats}
            for _ in base_texts
        ]
    finally:
        del inputs
        if "out_ids" in locals():
            del out_ids

    return batch_results


def generate_rewrite(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_input_device: torch.device,
    base_text: str,
    target_emotion: str,
    alpha: float,
    steering_mode: str,
    *,
    caa_vectors: np.ndarray,
    emotion_to_index: dict[str, int],
    prompt_template: str,
    layer_idx: int,
    max_new_tokens: int,
    do_sample: bool,
    text_postprocessor: Callable[[str], str],
) -> dict[str, Any]:
    """Generate one steered rewrite."""
    return generate_rewrite_batch(
        model,
        tokenizer,
        model_input_device,
        [base_text],
        target_emotion,
        alpha,
        steering_mode,
        caa_vectors=caa_vectors,
        emotion_to_index=emotion_to_index,
        prompt_template=prompt_template,
        layer_idx=layer_idx,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        text_postprocessor=text_postprocessor,
    )[0]


def generation_key(
    source_id: str,
    target_emotion: str,
    alpha: float,
    steering_mode: str,
    layer: int,
) -> tuple[str, str, float, str, int]:
    """Build the composite cache key for one generation row."""
    return (source_id, target_emotion, float(alpha), steering_mode, int(layer))


def append_jsonl_csv_row(
    jsonl_path: str | Path,
    csv_path: str | Path,
    fieldnames: list[str],
    row: dict[str, Any],
) -> None:
    """Append one row to paired JSONL and CSV outputs."""
    csv_exists = Path(csv_path).exists()
    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})


def load_generation_cache(
    generations_jsonl: str | Path,
) -> tuple[
    dict[tuple[str, str, float, str, int], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Load generation rows and cached `none` baselines from JSONL."""
    rows_by_key: dict[tuple[str, str, float, str, int], dict[str, Any]] = {}
    none_cache: dict[str, dict[str, Any]] = {}
    if not Path(generations_jsonl).exists():
        return rows_by_key, none_cache

    with open(generations_jsonl, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = generation_key(
                row["source_id"],
                row["target_emotion"],
                row["alpha"],
                row["steering_mode"],
                row["layer"],
            )
            rows_by_key[key] = row
            if row["steering_mode"] == "none" and row["source_id"] not in none_cache:
                none_cache[row["source_id"]] = row
    return rows_by_key, none_cache


def load_generation_df(generations_jsonl: str | Path) -> pd.DataFrame | None:
    """Load generation JSONL as a DataFrame when present."""
    if not Path(generations_jsonl).exists():
        return None
    return pd.read_json(generations_jsonl, lines=True)


def extract_message_content(result_obj: dict[str, Any]) -> str | None:
    """Extract assistant message text from a chat-completions batch result."""
    try:
        content = result_obj["response"]["body"]["choices"][0]["message"]["content"]
    except Exception:
        return None

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts) if parts else None
    return None


def format_alpha_list(values: list[float], steering_mode: str) -> str:
    """Format a sorted alpha list for CSV/report display."""
    if steering_mode == "none":
        return "0"
    if not values:
        return ""
    values = sorted(float(v) for v in values)
    parts = []
    for value in values:
        parts.append(str(int(value)) if float(value).is_integer() else f"{value:g}")
    return ", ".join(parts)


def interpret_negative_alpha_row(row: pd.Series) -> str:
    """Summarize whether negative alpha suppresses target or boosts opposite emotion."""
    if pd.isna(row["delta_opposite_vs_zero"]):
        return "no baseline available"
    if row["delta_opposite_vs_zero"] >= 1.0:
        return "negative alpha increases opposite emotion"
    if row["delta_target_vs_zero"] <= -1.0:
        return "negative alpha mainly suppresses target emotion"
    return "negative alpha is weak or mixed"