"""FastAPI server exposing s(x) computation as a REST endpoint."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotionengine.emotion_state import (
    compute_emotion_profile,
    extract_current_state,
    load_neutral_mean,
    neutralise,
)

app = FastAPI(title="EmotionEngine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None = None
_mu: np.ndarray | None = None
_caa: np.ndarray | None = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"

_REPO_ROOT = Path(__file__).resolve().parent.parent  # EmotionEngine_v2/

_MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
_NEUTRAL_MEAN_PATH = Path(os.environ.get(
    "NEUTRAL_MEAN_PATH",
    _REPO_ROOT / "activation/emotion_rewrites/neutral_mean_layer13.npy",
))
_NEUTRAL_ACT_PATH = Path(os.environ.get(
    "NEUTRAL_ACT_PATH",
    _REPO_ROOT / "activation/emotion_rewrites/neutral_paraphrase_residual_stream.npy",
))
_CAA_PATH = Path(os.environ.get(
    "CAA_PATH",
    _REPO_ROOT / "activation/emotion_rewrites/caa_emotion_directions.npz",
))


def _load_globals() -> None:
    global _model, _tokenizer, _mu, _caa
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME, torch_dtype=torch.float16
        ).to(_device)
        _model.eval()
    if _mu is None:
        if not _NEUTRAL_MEAN_PATH.exists():
            from emotionengine.emotion_state import compute_neutral_mean
            _mu = compute_neutral_mean(_NEUTRAL_ACT_PATH, _NEUTRAL_MEAN_PATH)
        else:
            _mu = load_neutral_mean(_NEUTRAL_MEAN_PATH)
    if _caa is None:
        _caa = np.load(str(_CAA_PATH))["caa_pooled"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    text: str
    neutral_text: str
    emotion_profile: dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        _load_globals()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

    try:
        x_neu = neutralise(req.text)
        s_raw = extract_current_state(_model, _tokenizer, req.text, x_neu, _device)
        profile = compute_emotion_profile(s_raw, _mu, _caa)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnalyzeResponse(
        text=req.text,
        neutral_text=x_neu,
        emotion_profile=profile,
    )
