"""Compute per-emotion neutral baseline scores from neutral_paraphrases.jsonl.

Loads the model and CAA artifacts directly (no API server needed) and runs
each neutral_paraphrase through the emotion scoring pipeline. The per-emotion
mean across all samples becomes the neutral baseline for the bidirectional bar
in the frontend.

Since the input texts are already neutral paraphrases, x_neu = x is used by
default (skipping the OpenAI neutralise call). Pass --neutralise to call
OpenAI for each text instead.

Usage
-----
    cd /path/to/EmotionEngine_v2
    PYTHONPATH=src uv run python emotionengine_experiments/neutral_baseline/compute_neutral_baseline.py

    # Quick sanity check with 20 samples:
    PYTHONPATH=src uv run python emotionengine_experiments/neutral_baseline/compute_neutral_baseline.py --n 20

    # Call OpenAI neutralise for each text (more expensive, closer to live inference):
    PYTHONPATH=src uv run python emotionengine_experiments/neutral_baseline/compute_neutral_baseline.py --neutralise
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA         = REPO_ROOT / "dataset/neutral_paraphrases/neutral_paraphrases.jsonl"
DEFAULT_MODEL_YAML   = REPO_ROOT / "model.yaml"
DEFAULT_NEUTRAL_MEAN = REPO_ROOT / "activation/emotion_rewrites/neutral_mean_layer13.npy"
DEFAULT_NEUTRAL_ACT  = REPO_ROOT / "activation/emotion_rewrites/neutral_paraphrase_residual_stream.npy"
DEFAULT_CAA          = REPO_ROOT / "activation/emotion_rewrites/caa_emotion_directions.npz"
DEFAULT_OUT          = Path(__file__).resolve().parent / "neutral_baseline_results.json"

EMOTIONS = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_stats(all_scores: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    stats = {}
    for emotion in EMOTIONS:
        vals = [s[emotion] for s in all_scores if emotion in s]
        n = len(vals)
        if n == 0:
            stats[emotion] = {"mean": 0.0, "std": 0.0, "n": 0}
            continue
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n
        stats[emotion] = {
            "mean": round(mean, 6),
            "std":  round(variance ** 0.5, 6),
            "n":    n,
        }
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",         type=Path, default=DEFAULT_DATA)
    p.add_argument("--model-yaml",   type=Path, default=DEFAULT_MODEL_YAML)
    p.add_argument("--neutral-mean", type=Path, default=DEFAULT_NEUTRAL_MEAN)
    p.add_argument("--neutral-act",  type=Path, default=DEFAULT_NEUTRAL_ACT)
    p.add_argument("--caa",          type=Path, default=DEFAULT_CAA)
    p.add_argument("--out",          type=Path, default=DEFAULT_OUT)
    p.add_argument("--n",    type=int, default=200, help="Records to sample (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--neutralise", action="store_true",
        help="Call OpenAI neutralise() for each text (requires OPENAI_API_KEY). "
             "Default: use x_neu=x since texts are already neutral paraphrases.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Imports here so PYTHONPATH=src error surfaces cleanly
    from utils.model_utils import load_config, load_model_and_tokenizer
    from emotionengine.emotion_state import (
        compute_emotion_profile,
        compute_neutral_mean,
        extract_current_state,
        load_neutral_mean,
        residualise_caa,
    )

    # Load records
    logger.info("Loading %s …", args.data)
    records = load_records(args.data)
    logger.info("  %d records.", len(records))

    if args.n > 0 and args.n < len(records):
        rng = random.Random(args.seed)
        records = rng.sample(records, args.n)
        logger.info("  Sampled %d records (seed=%d).", len(records), args.seed)

    # Load model
    cfg = load_config(args.model_yaml)
    model, tokenizer = load_model_and_tokenizer(cfg, args.device)

    # Load neutral mean (compute from activations if cached file absent)
    if args.neutral_mean.exists():
        mu = load_neutral_mean(args.neutral_mean)
        logger.info("Loaded μ_neutral from %s", args.neutral_mean)
    else:
        logger.info("neutral_mean_layer13.npy not found — computing from activations …")
        mu = compute_neutral_mean(args.neutral_act, args.neutral_mean)

    # Load CAA directions and remove shared direction g → r̂_e
    caa = residualise_caa(np.load(str(args.caa))["caa_pooled"])
    logger.info("CAA shape (residualised): %s", caa.shape)

    # Optional: OpenAI client for neutralise
    openai_client = None
    if args.neutralise:
        import openai
        openai_client = openai.OpenAI()
        logger.info("OpenAI neutralise enabled.")
    else:
        logger.info("Using x_neu = x (texts are already neutral paraphrases).")

    # Run
    all_scores: list[dict[str, float]] = []
    n_errors = 0

    for i, rec in enumerate(records, 1):
        text = rec["neutral_paraphrase"]
        logger.info("[%d/%d] %s …", i, len(records), text[:70])

        try:
            if args.neutralise:
                from emotionengine.emotion_state import neutralise
                x_neu = neutralise(text, client=openai_client)
            else:
                x_neu = text

            s_raw = extract_current_state(model, tokenizer, text, x_neu, args.device)
            profile = compute_emotion_profile(s_raw, mu, caa)
            all_scores.append(profile)

            scores_str = "  ".join(f"{e[:3]}={profile[e]:+.3f}" for e in EMOTIONS)
            logger.info("  → %s", scores_str)

        except Exception as e:
            logger.warning("  SKIP — %s", e)
            n_errors += 1

    # Stats
    logger.info("\n%d successful / %d errors", len(all_scores), n_errors)
    stats = compute_stats(all_scores)

    print("\nPer-emotion neutral baseline (mean ± std):")
    print(f"  {'emotion':<14}  {'mean':>8}  {'std':>8}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}")
    for emotion in EMOTIONS:
        s = stats[emotion]
        print(f"  {emotion:<14}  {s['mean']:>+8.4f}  {s['std']:>8.4f}")

    # Save
    output = {
        "meta": {
            "n_samples":  len(all_scores),
            "n_errors":   n_errors,
            "seed":       args.seed,
            "neutralise": args.neutralise,
            "data":       str(args.data),
            "caa":        str(args.caa),
        },
        "neutral_baseline": {e: stats[e]["mean"] for e in EMOTIONS},
        "neutral_std":      {e: stats[e]["std"]  for e in EMOTIONS},
        "stats":            stats,
        "all_scores":       all_scores,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info("Results saved → %s", args.out)

    print("\nFrontend neutral_baseline dict (paste into App.tsx):")
    print(json.dumps({e: stats[e]["mean"] for e in EMOTIONS}, indent=2))


if __name__ == "__main__":
    main()
