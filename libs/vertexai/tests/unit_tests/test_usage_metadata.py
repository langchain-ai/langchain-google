"""Usage metadata tests for Vertex AI Gemini helpers."""

import importlib.util
from pathlib import Path

USAGE_HELPERS_PATH = (
    Path(__file__).resolve().parents[2] / "langchain_google_vertexai" / "_usage.py"
)

spec = importlib.util.spec_from_file_location("vertex_usage", USAGE_HELPERS_PATH)
assert spec and spec.loader
usage_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(usage_module)

coerce_usage_metadata = usage_module.coerce_usage_metadata
diff_usage_metadata = usage_module.diff_usage_metadata


def test_get_usage_metadata_gemini_details() -> None:
    raw_metadata = {
        "prompt_token_count": 12,
        "tool_use_prompt_token_count": 8,
        "candidates_token_count": 20,
        "thoughts_token_count": 5,
        "cached_content_token_count": 3,
        "total_token_count": 45,
        "prompt_tokens_details": [
            {"modality": "TEXT", "token_count": 12},
            {"modality": "AUDIO", "token_count": 2},
        ],
        "tool_use_prompt_tokens_details": [{"modality": "TEXT", "token_count": 8}],
        "candidates_tokens_details": [{"modality": "TEXT", "token_count": 20}],
    }

    usage = coerce_usage_metadata(raw_metadata)
    assert usage is not None
    assert usage["input_tokens"] == 20
    assert usage["output_tokens"] == 25
    assert usage["total_tokens"] == 45

    input_details = usage.get("input_token_details", {}) or {}
    assert input_details["tool_use_prompt"] == 8
    assert input_details["cache_read"] == 3
    assert input_details["text"] == 12
    assert input_details["audio"] == 2
    assert input_details["tool_use_prompt_text"] == 8

    output_details = usage.get("output_token_details", {}) or {}
    assert output_details["reasoning"] == 5
    assert output_details["text"] == 20


def test_get_usage_metadata_gemini_delta() -> None:
    first_raw = {
        "prompt_token_count": 10,
        "tool_use_prompt_token_count": 2,
        "candidates_token_count": 6,
        "thoughts_token_count": 4,
        "cached_content_token_count": 3,
        "total_token_count": 22,
        "prompt_tokens_details": [{"modality": "TEXT", "token_count": 10}],
        "candidates_tokens_details": [{"modality": "TEXT", "token_count": 6}],
    }

    second_raw = {
        "prompt_token_count": 16,
        "tool_use_prompt_token_count": 4,
        "candidates_token_count": 12,
        "thoughts_token_count": 8,
        "cached_content_token_count": 5,
        "total_token_count": 40,
        "prompt_tokens_details": [{"modality": "TEXT", "token_count": 14}],
        "candidates_tokens_details": [{"modality": "TEXT", "token_count": 12}],
    }

    first_usage = coerce_usage_metadata(first_raw)
    second_usage = coerce_usage_metadata(second_raw)
    assert first_usage is not None
    assert second_usage is not None

    delta = diff_usage_metadata(second_usage, first_usage)
    assert delta is not None

    assert delta["input_tokens"] == 8
    assert delta["output_tokens"] == 10
    assert delta["total_tokens"] == 18

    assert delta["input_token_details"]["tool_use_prompt"] == 2
    assert delta["input_token_details"]["cache_read"] == 2
    assert delta["input_token_details"]["text"] == 4
    assert delta["output_token_details"]["reasoning"] == 4
    assert delta["output_token_details"]["text"] == 6
