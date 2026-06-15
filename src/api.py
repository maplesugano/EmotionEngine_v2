"""FastAPI server exposing s(x) computation as a REST endpoint."""

from __future__ import annotations

import base64
import logging
import os
import traceback
from pathlib import Path

import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotionengine.emotion_state import (
    compute_emotion_profile,
    compute_meta_projections,
    extract_current_state,
    load_neutral_mean,
    neutralise,
    residualise_caa,
)
from utils.steering_utils import steering_hook
from utils.text_utils import make_instruction_prefix

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
_meta_axes: np.ndarray | None = None
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
_META_AXES_PATH = Path(os.environ.get(
    "META_AXES_PATH",
    _REPO_ROOT / "local_axes_experiments/exp7_meta_axis_extraction/results/layer13_meta_axes_pca.npy",
))


def _load_globals() -> None:
    global _model, _tokenizer, _mu, _caa, _meta_axes
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
        _caa = residualise_caa(np.load(str(_CAA_PATH))["caa_pooled"])
    if _meta_axes is None:
        _meta_axes = np.load(str(_META_AXES_PATH)).astype(np.float32)  # (5, D)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_STEER_LAYER = 13

class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    text: str
    neutral_text: str
    emotion_profile: dict[str, float]
    meta_projections: dict[str, float]


class SteerRequest(BaseModel):
    text: str
    neutral_text: str          # already computed by /analyze — pass it back
    gammas: dict[str, float]   # {"m1": γ1, "m2": γ2, ...}
    max_new_tokens: int = 80


class SteerResponse(BaseModel):
    generated_text: str


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
        meta = compute_meta_projections(s_raw, _mu, _meta_axes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnalyzeResponse(
        text=req.text,
        neutral_text=x_neu,
        emotion_profile=profile,
        meta_projections=meta,
    )


@app.post("/steer", response_model=SteerResponse)
def steer(req: SteerRequest) -> SteerResponse:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        _load_globals()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

    try:
        # Build delta = Σ γ_k * m_k  (shape: D,)
        delta = np.zeros(_meta_axes.shape[1], dtype=np.float32)
        for mk, gamma in req.gammas.items():
            if abs(gamma) < 1e-6:
                continue
            k = int(mk[1:]) - 1  # "m1" → 0, "m2" → 1, ...
            if 0 <= k < len(_meta_axes):
                delta += gamma * _meta_axes[k]

        # Prompt ends at the assistant header — model generates the rewrite.
        # Do NOT append req.text: that would make the model think it already wrote
        # the response and it would immediately emit <|eot_id|>.
        prompt = make_instruction_prefix(req.neutral_text, _tokenizer)
        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=400,  # leave headroom for max_new_tokens within model context
        ).to(_device)

        layer_module = _model.model.layers[_STEER_LAYER]
        layer_dtype  = next(layer_module.parameters()).dtype
        hook_stats: dict[str, int] = {
            "hook_calls": 0, "hook_prefill_calls": 0,
            "hook_decode_calls": 0, "hook_applied_calls": 0,
        }

        with torch.inference_mode():
            with steering_hook(
                _model, _STEER_LAYER,
                delta if np.any(delta != 0) else None,
                "completion_tokens",
                hook_stats,
            ):
                out_ids = _model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    do_sample=False,
                    pad_token_id=_tokenizer.pad_token_id,
                )

        prompt_len = inputs["input_ids"].shape[1]
        new_ids = out_ids[0, prompt_len:]
        if len(new_ids) == 0:
            raise ValueError(
                f"Model generated no new tokens (prompt was {prompt_len} tokens, "
                f"which may have hit the truncation limit). Try a shorter input."
            )
        generated = _tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not generated:
            raise ValueError("Model generated only special tokens or whitespace.")

        _ = layer_dtype  # referenced to satisfy linter

    except Exception as exc:
        logger.error("POST /steer failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SteerResponse(generated_text=generated)


# ---------------------------------------------------------------------------
# Media generation — with disk cache
# ---------------------------------------------------------------------------
import hashlib
import json as _json

_MEDIA_CACHE_DIR = _REPO_ROOT / "media_cache"
_MEDIA_CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(kind: str, text: str) -> str:
    return hashlib.sha256(f"{kind}:{text}".encode()).hexdigest()


def _cache_get(key: str) -> str | None:
    p = _MEDIA_CACHE_DIR / f"{key}.json"
    if p.exists():
        return _json.loads(p.read_text())["data"]
    return None


def _cache_set(key: str, data: str) -> None:
    p = _MEDIA_CACHE_DIR / f"{key}.json"
    p.write_text(_json.dumps({"data": data}))


class MediaRequest(BaseModel):
    text: str  # the passage (input text or steered text)


class ImageResponse(BaseModel):
    image_b64: str  # base64-encoded PNG


class MusicResponse(BaseModel):
    audio_b64: str  # base64-encoded mp3


_IMAGE_PROMPT = (
    'Please paint an oil painting using broad brushstrokes that best captures '
    'the mood of the following passage. Please render it in an abstract and '
    'contemporary style. "{text}"'
)

_MUSIC_PROMPT = (
    'Compose an instrumental piece using one or more of '
    'the following instruments: wind, string, or percussion. Capture the mood '
    'of the following passage in an abstract and contemporary style: \'{text}\''
)


@app.post("/generate-image", response_model=ImageResponse)
async def generate_image(req: MediaRequest) -> ImageResponse:
    key = _cache_key("image", req.text)
    cached = _cache_get(key)
    if cached:
        return ImageResponse(image_b64=cached)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    prompt = _IMAGE_PROMPT.format(text=req.text)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "gpt-image-1", "prompt": prompt, "n": 1, "size": "1024x1024"},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=resp.text)
    data = resp.json()
    b64 = data["data"][0]["b64_json"]
    _cache_set(key, b64)
    return ImageResponse(image_b64=b64)


@app.post("/generate-music", response_model=MusicResponse)
async def generate_music(req: MediaRequest) -> MusicResponse:
    key = _cache_key("music", req.text)
    cached = _cache_get(key)
    if cached:
        return MusicResponse(audio_b64=cached)

    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not set")
    prompt = _MUSIC_PROMPT.format(text=req.text)
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.elevenlabs.io/v1/music/compose",
            headers={"xi-api-key": api_key},
            json={"prompt": prompt, "music_length_ms": 10000, "force_instrumental": True},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=resp.text)
    audio_b64 = base64.b64encode(resp.content).decode()
    _cache_set(key, audio_b64)
    return MusicResponse(audio_b64=audio_b64)
