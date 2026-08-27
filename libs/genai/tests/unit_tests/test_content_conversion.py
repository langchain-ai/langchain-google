"""Tests for converting LangChain content blocks to Gemini parts."""

import base64
import logging
from typing import Any, cast

import pytest
from google.genai.types import Blob

from langchain_google_genai._compat import (
    _convert_from_v1_to_generativelanguage_v1beta,
)
from langchain_google_genai.chat_models import _convert_to_parts, _decode_signature


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        (base64.b64encode(b"sig").decode("ascii"), b"sig"),
        (b"raw_bytes", b"raw_bytes"),
        ("", None),
        (b"", None),
        (None, None),
        (123, None),
    ],
)
def test_decode_signature(signature: Any, expected: bytes | None) -> None:
    assert _decode_signature(signature) == expected


def test_convert_to_parts_decodes_bytes_thought_signature() -> None:
    parts = _convert_to_parts(
        [{"thought": True, "text": "Thinking.", "thought_signature": b"raw_sig"}],
        allow_v1beta_dicts=True,
    )

    assert len(parts) == 1
    assert parts[0].thought is True
    assert parts[0].thought_signature == b"raw_sig"


def test_convert_to_parts_typed_block_wins_over_thought_key() -> None:
    """Honor an explicit block type before inspecting legacy shape keys."""
    parts = _convert_to_parts(
        [{"type": "text", "text": "Not a thought.", "thought": True}]
    )

    assert len(parts) == 1
    assert parts[0].text == "Not a thought."
    assert parts[0].thought is not True


def test_convert_to_parts_empty_inline_data_is_not_a_file_part() -> None:
    """Do not reinterpret present-but-empty inline data as file data."""
    parts = _convert_to_parts([{"inline_data": {}}], allow_v1beta_dicts=True)

    assert len(parts) == 1
    assert parts[0].inline_data is not None
    assert parts[0].file_data is None


@pytest.mark.parametrize(
    ("block", "attr"),
    [
        ({"inline_data": {"mime_type": None, "data": "aGk="}}, "inline_data"),
        ({"file_data": {"mime_type": None, "file_uri": "gs://b/o"}}, "file_data"),
        ({"executable_code": {"language": None, "code": "x=1"}}, "executable_code"),
        (
            {"code_execution_result": {"outcome": None, "output": "1"}},
            "code_execution_result",
        ),
    ],
)
def test_convert_to_parts_v1beta_shapes_tolerate_none_valued_keys(
    block: dict[str, Any], attr: str
) -> None:
    parts = _convert_to_parts([block], allow_v1beta_dicts=True)

    assert len(parts) == 1
    assert getattr(parts[0], attr) is not None


def test_convert_to_parts_joins_multi_segment_summary() -> None:
    parts = _convert_to_parts(
        [
            {
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "First."},
                    "not a dict",
                    {"type": "summary_text", "text": "Second."},
                ],
            }
        ]
    )

    assert len(parts) == 1
    assert parts[0].text == "First. Second."
    assert parts[0].thought is True


def test_convert_from_v1_drops_foreign_reasoning_with_signature() -> None:
    """Foreign reasoning blocks drop entirely; adjacent text still converts."""
    converted = _convert_from_v1_to_generativelanguage_v1beta(
        [
            {
                "type": "reasoning",
                "reasoning": "Considering Paris.",
                "extras": {"signature": "bedrock-sig"},
            },
            {"type": "text", "text": "Paris."},
        ],
        "bedrock_converse",
    )

    assert converted == [{"text": "Paris."}]


def test_convert_from_v1_drops_foreign_reasoning_summary() -> None:
    """OpenAI-style summary blocks are dropped for foreign providers."""
    converted = _convert_from_v1_to_generativelanguage_v1beta(
        cast(
            "Any",
            [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "I should search."}],
                }
            ],
        ),
        "openai",
    )

    assert converted == []


def test_convert_from_v1_keeps_native_reasoning_with_signature() -> None:
    """Regression guard: native signed reasoning still becomes a thought part."""
    converted = _convert_from_v1_to_generativelanguage_v1beta(
        [
            {
                "type": "reasoning",
                "reasoning": "Thinking.",
                "extras": {"signature": "c2ln"},
            }
        ],
        "google_genai",
    )

    assert converted == [
        {"thought": True, "text": "Thinking.", "thought_signature": "c2ln"}
    ]


def test_convert_from_v1_keeps_unknown_provider_reasoning() -> None:
    """Regression guard: provider-less checkpoints stay lenient."""
    converted = _convert_from_v1_to_generativelanguage_v1beta(
        [
            {
                "type": "reasoning",
                "reasoning": "Thinking.",
                "extras": {"signature": "c2ln"},
            }
        ],
        None,
    )

    assert converted == [
        {"thought": True, "text": "Thinking.", "thought_signature": "c2ln"}
    ]


def test_convert_from_v1_keeps_base64_media_undecoded() -> None:
    """Leave base64 decoding to SDK validation to avoid double encoding."""
    converted = _convert_from_v1_to_generativelanguage_v1beta(
        [{"type": "image", "base64": "aGVsbG8=", "mime_type": "image/png"}],
        "google_genai",
    )

    assert converted == [
        {"inline_data": {"mime_type": "image/png", "data": "aGVsbG8="}}
    ]
    assert Blob(**converted[0]["inline_data"]).data == b"hello"


@pytest.mark.parametrize(
    ("content", "model_provider", "expected_fragment"),
    [
        (
            [{"type": "brand_new_core_block"}],
            "google_genai",
            "no Gemini equivalent",
        ),
        (
            [{"type": "non_standard", "value": {"type": "redacted_thinking"}}],
            "anthropic",
            "non-standard v1 content block",
        ),
        (["not a mapping"], "google_genai", "not a typed mapping"),
    ],
)
def test_convert_from_v1_logs_dropped_blocks(
    content: list[Any],
    model_provider: str,
    expected_fragment: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep every intentional v1 content drop diagnosable."""
    with caplog.at_level(logging.WARNING):
        result = _convert_from_v1_to_generativelanguage_v1beta(
            cast("Any", content), model_provider
        )

    assert result == []
    assert expected_fragment in caplog.text
