"""EmotionEngine – shared utilities for residual-stream extraction and analysis."""

from emotionengine.judge_utils import JudgeConfig, judge_many, judge_one
from emotionengine.model_utils import extract_batch, load_config, load_model_and_tokenizer
from emotionengine.steering_utils import (
    build_delta_vector,
    chunk_list,
    cosine_sim_matrix,
    flush_device_cache,
    generation_key,
    steering_hook,
    unit,
    unit_np,
)
from emotionengine.text_utils import extract_rewritten_text, make_instruction_prefix

__all__ = [
    # judge_utils
    "JudgeConfig",
    "judge_one",
    "judge_many",
    # model_utils
    "load_config",
    "load_model_and_tokenizer",
    "extract_batch",
    # steering_utils
    "unit",
    "unit_np",
    "cosine_sim_matrix",
    "build_delta_vector",
    "chunk_list",
    "flush_device_cache",
    "steering_hook",
    "generation_key",
    # text_utils
    "make_instruction_prefix",
    "extract_rewritten_text",
]
