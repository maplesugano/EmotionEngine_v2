"""Re-generate only truncation-like rows from exp5 results and compare stability.

This script:
1) loads analysis/subspace_steering/layer{L}/results.jsonl
2) detects truncation-like rewrites (e.g. trailing "I couldn", unfinished outputs)
3) re-generates multiple samples with the SAME steering condition
4) writes per-sample outputs for stability inspection

Usage:
  uv run python scripts/exp5_truncation_regen_check.py --layer 13 --n-samples 3
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

from emotionengine.model_utils import load_config, load_model_and_tokenizer
from emotionengine.steering_utils import flush_device_cache, steering_hook
from emotionengine.text_utils import INSTRUCTION_TEMPLATE, make_instruction_prefix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = REPO_ROOT / "activation" / "emotion_rewrites"
OUT_BASE_DIR = Path(__file__).resolve().parent / "results"

TRUNC_PATTERNS = [
    re.compile(r"\bI couldn\s*$", flags=re.IGNORECASE),
    re.compile(r"\bI hadn\s*$", flags=re.IGNORECASE),
    re.compile(r"\bI wasn\s*$", flags=re.IGNORECASE),
    re.compile(r"\bI didn\s*$", flags=re.IGNORECASE),
    re.compile(r"\bcouldn\s*$", flags=re.IGNORECASE),
    re.compile(r"\bhadn\s*$", flags=re.IGNORECASE),
    re.compile(r"\bwasn\s*$", flags=re.IGNORECASE),
    re.compile(r"\bdidn\s*$", flags=re.IGNORECASE),
]


def unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def project_out(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    coeff = X @ g
    return X - np.outer(coeff, g)


def build_per_emotion_residuals(
    emo_act: np.ndarray,
    neu_act: np.ndarray,
    caa_pooled: np.ndarray,
    caa: np.ndarray,
    layer_idx: int,
    intensity_low_idx: int,
    intensity_high_idx: int,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    n, e, _i, _l, d = emo_act.shape
    g_raw = caa_pooled[:, layer_idx, :].mean(axis=0)
    g = unit_norm(g_raw.astype(np.float64)).astype(np.float32)

    intensity_dirs = unit_norm(
        caa[:, intensity_high_idx, layer_idx, :].astype(np.float64)
        - caa[:, intensity_low_idx, layer_idx, :].astype(np.float64)
    ).astype(np.float32)

    delta_resid = np.empty((n, e, d), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        h_emo = emo_act[start:end, :, :, layer_idx, :].mean(axis=2).astype(np.float32)
        h_neu = neu_act[start:end, layer_idx, :].astype(np.float32)
        delta = h_emo - h_neu[:, np.newaxis, :]
        for ei in range(e):
            d_e = project_out(delta[:, ei, :], g)
            d_e = project_out(d_e, intensity_dirs[ei])
            delta_resid[start:end, ei, :] = d_e

    delta_resid = delta_resid - delta_resid.mean(axis=1, keepdims=True)
    return delta_resid, g


def build_decomposed_delta(g: np.ndarray, r_e: np.ndarray, u_ek: np.ndarray, alpha_g: float, alpha_r: float, beta: float) -> np.ndarray:
    return (alpha_g * g + alpha_r * r_e + beta * u_ek).astype(np.float32)


def is_truncation_like(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if any(p.search(t) for p in TRUNC_PATTERNS):
        return True
    # Ends without punctuation and is relatively short -> suspicious cutoff
    if not t.endswith((".", "!", "?", '"', "'")) and len(t.split()) < 18:
        return True
    return False


def generate_with_delta(
    model,
    tokenizer,
    device,
    base_text: str,
    delta_vec: np.ndarray,
    layer_idx: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    prompt = make_instruction_prefix(base_text, tokenizer, INSTRUCTION_TEMPLATE)
    inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    _ = torch.manual_seed(seed)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
    }
    hook_stats = {"hook_calls": 0, "hook_prefill_calls": 0, "hook_decode_calls": 0, "hook_applied_calls": 0}
    try:
        with torch.inference_mode():
            with steering_hook(model, layer_idx, delta_vec, "completion_tokens", hook_stats):
                out_ids = model.generate(**inputs, **gen_kwargs)
        prompt_width = inputs["input_ids"].shape[1]
        new_ids = out_ids[0, prompt_width:]
        return tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    except torch.cuda.OutOfMemoryError:
        flush_device_cache()
        return "[OOM]"
    finally:
        del inputs
        if "out_ids" in locals():
            del out_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--results-jsonl", default=None)
    p.add_argument("--emo-act", default=str(ACT_DIR / "emotion_intensity_residual_stream.npy"))
    p.add_argument("--emo-info", default=str(ACT_DIR / "emotion_intensity_residual_stream_info.json"))
    p.add_argument("--neu-act", default=str(ACT_DIR / "neutral_paraphrase_residual_stream.npy"))
    p.add_argument("--caa-directions", default=str(ACT_DIR / "caa_emotion_directions.npz"))
    p.add_argument("--model-config", default=str(REPO_ROOT / "model.yaml"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--max-rows", type=int, default=0, help="0 means all truncation-like rows")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = OUT_BASE / f"layer{args.layer}"
    results_jsonl = Path(args.results_jsonl) if args.results_jsonl else (out_dir / "results.jsonl")
    regen_jsonl = out_dir / "truncation_regen_samples.jsonl"

    rows = [json.loads(l) for l in open(results_jsonl, encoding="utf-8") if l.strip()]
    trunc_rows = [r for r in rows if is_truncation_like(str(r.get("rewrite_text", "")))]
    if args.max_rows and args.max_rows > 0:
        trunc_rows = trunc_rows[: args.max_rows]
    logger.info("Detected %d truncation-like rows.", len(trunc_rows))
    if not trunc_rows:
        logger.info("Nothing to regenerate.")
        return

    with open(args.emo_info) as f:
        info = json.load(f)
    emotions = info["emotion_order"]
    layer_indices = info["layer_indices"]
    if args.layer not in layer_indices:
        raise ValueError(f"Layer {args.layer} not in available layers: {layer_indices}")
    layer_idx = layer_indices.index(args.layer)
    emotion_to_idx = {e: i for i, e in enumerate(emotions)}

    emo_shape = tuple(info["shape"])
    emo_raw = np.memmap(args.emo_act, dtype=info["dtype"], mode="r", shape=emo_shape)

    neu_info_path = args.neu_act.replace(".npy", "_info.json")
    with open(neu_info_path) as f:
        neu_info = json.load(f)
    neu_shape = tuple(neu_info["shape"])
    neu_raw = np.memmap(args.neu_act, dtype=neu_info["dtype"], mode="r", shape=neu_shape)

    caa_npz = np.load(args.caa_directions)
    caa_pooled = caa_npz["caa_pooled"]
    caa = caa_npz["caa"]

    intensity_order = info["intensity_order"]
    low_idx = intensity_order.index("low")
    high_idx = intensity_order.index("high")

    delta_resid, g = build_per_emotion_residuals(
        emo_raw, neu_raw, caa_pooled, caa,
        layer_idx=layer_idx,
        intensity_low_idx=low_idx,
        intensity_high_idx=high_idx,
    )

    pca_by_emotion: dict[str, Any] = {}
    r_e_by_emotion: dict[str, np.ndarray] = {}
    for emotion in emotions:
        e_idx = emotion_to_idx[emotion]
        r_e_raw = caa_pooled[e_idx, layer_idx, :].astype(np.float64)
        g64 = g.astype(np.float64)
        r_e_resid = r_e_raw - (r_e_raw @ g64) * g64
        r_e = unit_norm(r_e_resid[np.newaxis, :])[0].astype(np.float32)
        r_e_by_emotion[emotion] = r_e

        x_e = delta_resid[:, e_idx, :].astype(np.float64)
        x_c = x_e - x_e.mean(axis=0)
        n_comp = min(args.n_components, x_e.shape[0] - 1, x_e.shape[1])
        pca = PCA(n_components=n_comp, random_state=0)
        pca.fit(x_c)
        pca_by_emotion[emotion] = pca

    model_cfg = load_config(args.model_config)
    model, tokenizer = load_model_and_tokenizer(model_cfg, args.device)
    model_device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    wrote = 0
    with open(regen_jsonl, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(trunc_rows):
            emotion = row["emotion"]
            pc = int(row["pc"])
            beta = float(row["beta"])
            alpha_g = float(row["alpha_g"])
            alpha_r = float(row["alpha_r"])
            base_text = row["base_text"]

            pca = pca_by_emotion[emotion]
            if pc < 1 or pc > pca.components_.shape[0]:
                continue
            u_ek = pca.components_[pc - 1].astype(np.float32)
            r_e = r_e_by_emotion[emotion]
            delta_vec = build_decomposed_delta(g, r_e, u_ek, alpha_g, alpha_r, beta)

            for s in range(args.n_samples):
                seed = 1234 + idx * 100 + s
                rewrite = generate_with_delta(
                    model=model,
                    tokenizer=tokenizer,
                    device=model_device,
                    base_text=base_text,
                    delta_vec=delta_vec,
                    layer_idx=layer_idx,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=seed,
                )
                out = {
                    "layer": args.layer,
                    "emotion": emotion,
                    "pc": pc,
                    "pole": row["pole"],
                    "source_id": row["source_id"],
                    "alpha_g": alpha_g,
                    "alpha_r": alpha_r,
                    "beta": beta,
                    "sample_idx": s,
                    "seed": seed,
                    "base_text": base_text,
                    "original_rewrite_text": row.get("rewrite_text", ""),
                    "regen_rewrite_text": rewrite,
                    "regen_is_truncation_like": is_truncation_like(rewrite),
                }
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                wrote += 1

    logger.info("Wrote %d rows to %s", wrote, regen_jsonl)


if __name__ == "__main__":
    main()
