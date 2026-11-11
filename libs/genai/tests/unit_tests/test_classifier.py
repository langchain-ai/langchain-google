"""Unit tests for `GoogleGenAIClassifier`."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from google import genai
from langchain_core.decisions import Box, Choice, Noul, Score
from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import SecretStr, ValidationError

from langchain_google_genai.classifier import GoogleGenAIClassifier
from langchain_google_genai.classifier._input import (
    _to_decision_input,
    _to_decision_questions,
)

MODEL_NAME = "gemini-decision-1"

_RESPONSE: dict[str, Any] = {
    "answers": {
        "urgent": {"type": "noul", "noul": 0.9, "confidence": 0.9},
        "department": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"technical": 0.8, "billing": 0.2},
            "confidence": 0.8,
        },
        "frustration": {
            "type": "score",
            "score": 1.4,
            "legend": {"0": "Calm.", "1": "Concerned.", "2": "Angry."},
            "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
            "confidence": 0.5,
        },
    },
    "usage": {
        "total_input_tokens": 12,
        "total_output_tokens": 3,
        "total_tokens": 15,
    },
}

_QUESTIONS: dict[str, Noul | Choice | Score] = {
    "urgent": Noul(instructions="Is this urgent?"),
    "department": Choice(
        instructions="Which team?",
        criteria={"billing": "Payments.", "technical": None},
    ),
    "frustration": Score(
        instructions="How frustrated?",
        criteria=["Calm.", "Concerned.", "Angry."],
    ),
}


def _classifier(response: dict[str, Any] | None = None) -> GoogleGenAIClassifier:
    client = MagicMock(spec=genai.Client)
    client.decisions = MagicMock()
    client.decisions.create.return_value = response or _RESPONSE
    client.aio.decisions.create = AsyncMock(return_value=response or _RESPONSE)
    return GoogleGenAIClassifier(model=MODEL_NAME, client=client)


def test_invoke_sends_request_and_parses_answers() -> None:
    classifier = _classifier()

    response = classifier.invoke({"state": "Help!", "questions": _QUESTIONS})

    classifier.client.decisions.create.assert_called_once_with(  # type: ignore[union-attr]
        model=MODEL_NAME,
        input="Help!",
        questions={
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "department": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Payments.", "technical": ""},
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated?",
                "criteria": ["Calm.", "Concerned.", "Angry."],
            },
        },
        api_version="v1beta",
    )
    assert response.model == MODEL_NAME
    assert response.nouls["urgent"].noul == 0.9
    assert response.choices["department"].choice == "technical"
    assert response.scores["frustration"].legend == {
        0: "Calm.",
        1: "Concerned.",
        2: "Angry.",
    }
    assert response.scores["frustration"].probabilities[2] == 0.5
    assert response.usage == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }


async def test_ainvoke_uses_async_client() -> None:
    classifier = _classifier()

    response = await classifier.ainvoke({"state": "Help!", "questions": _QUESTIONS})

    classifier.client.aio.decisions.create.assert_awaited_once()  # type: ignore[union-attr]
    classifier.client.decisions.create.assert_not_called()  # type: ignore[union-attr]
    assert response.choices["department"].choice == "technical"


def test_sdk_response_objects_are_dumped() -> None:
    sdk_response = MagicMock()
    sdk_response.model_dump.return_value = _RESPONSE
    classifier = _classifier()
    classifier.client.decisions.create.return_value = sdk_response  # type: ignore[union-attr]

    response = classifier.invoke({"state": "Help!", "questions": _QUESTIONS})

    assert response.nouls["urgent"].noul == 0.9


def test_blocked_input_raises_model_refusal_error() -> None:
    classifier = _classifier(
        {
            "prompt_feedback": {
                "block_reason": "safety",
                "block_reason_message": "Input was blocked.",
            },
            "usage": {"total_input_tokens": 5},
        }
    )

    with pytest.raises(ModelRefusalError, match=r"'safety'.*Input was blocked\."):
        classifier.invoke({"state": "text", "questions": _QUESTIONS})


def test_unknown_answer_types_raise() -> None:
    classifier = _classifier(
        {
            "answers": {
                "urgent": {"type": "noul", "noul": 0.2, "confidence": 0.8},
                "region": {"type": "UNKNOWN", "raw": {"type": "box"}},
            }
        }
    )

    with pytest.raises(ValidationError):
        classifier.invoke({"state": "text", "questions": _QUESTIONS})


def test_missing_usage_is_none() -> None:
    classifier = _classifier({"answers": {}})

    response = classifier.invoke({"state": "text", "questions": _QUESTIONS})

    assert response.usage is None


def test_noul_confidence_is_not_kept() -> None:
    classifier = _classifier(_RESPONSE)

    response = classifier.invoke({"state": "text", "questions": _QUESTIONS})

    assert response.nouls["urgent"].model_dump() == {"type": "noul", "noul": 0.9}


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key is required"):
        GoogleGenAIClassifier(model=MODEL_NAME)


def test_api_key_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")

    classifier = GoogleGenAIClassifier(model=MODEL_NAME)

    assert classifier.google_api_key is not None
    assert classifier.google_api_key.get_secret_value() == "env-key"
    assert "env-key" not in repr(classifier)


def test_http_options() -> None:
    classifier = GoogleGenAIClassifier(
        model=MODEL_NAME,
        google_api_key=SecretStr("key"),
        base_url="https://example.com",
        timeout=2.5,
        max_retries=1,
    )

    options = classifier._http_options

    assert options.base_url == "https://example.com"
    assert options.timeout == 2500
    assert options.retry_options is not None
    assert options.retry_options.attempts == 1


def test_traced_config_sets_provider_metadata() -> None:
    config = _classifier()._traced_config({"metadata": {"user": "value"}})

    assert config["metadata"] == {
        "user": "value",
        "ls_provider": "google_genai",
        "ls_model_name": MODEL_NAME,
        "ls_model_type": "chat",
    }


# Questions


def _sent_questions(questions: dict[str, Any]) -> dict[str, Any]:
    classifier = _classifier()
    classifier.invoke({"state": "text", "questions": questions})
    return classifier.client.decisions.create.call_args.kwargs["questions"]  # type: ignore[union-attr]


def test_noul_criteria_mapped_to_descriptions() -> None:
    questions = _to_decision_questions(
        {
            "both": Noul(instructions="q", criteria={"true": "Yes.", "false": "No."}),
            "one": Noul(instructions="q", criteria={"true": None, "false": "No."}),
            "none": Noul(instructions="q", criteria={}),
        }
    )

    assert questions["both"]["criteria"] == {
        "true_description": "Yes.",
        "false_description": "No.",
    }
    assert questions["one"]["criteria"] == {"false_description": "No."}
    assert "criteria" not in questions["none"]


def test_json_content_is_sent_as_json_text() -> None:
    questions = _to_decision_questions(
        {
            "urgent": Noul(
                instructions={"task": "Is this urgent?"},
                criteria={"true": ["pager", "outage"]},
            ),
            "team": Choice(instructions="q", criteria={"a": {"area": "billing"}}),
            "level": Score(instructions="q", criteria=["low", {"level": "high"}]),
        }
    )

    assert questions == {
        "urgent": {
            "type": "noul",
            "instructions": '{"task": "Is this urgent?"}',
            "criteria": {"true_description": '["pager", "outage"]'},
        },
        "team": {
            "type": "choice",
            "instructions": "q",
            "criteria": {"a": '{"area": "billing"}'},
        },
        "level": {
            "type": "score",
            "instructions": "q",
            "criteria": ["low", '{"level": "high"}'],
        },
    }


def test_box_questions_raise() -> None:
    classifier = _classifier()

    with pytest.raises(ValueError, match="does not support Box"):
        classifier.invoke(
            {"state": "text", "questions": {"q": Box(instructions="The dog.")}}
        )

    classifier.client.decisions.create.assert_not_called()  # type: ignore[union-attr]


# State


def test_string_state_is_sent_as_is() -> None:
    assert _to_decision_input("hello") == "hello"


def test_json_state_is_sent_as_json_text() -> None:
    state = {"ticket": {"subject": "Refund", "priority": 2}, "tags": ["a", None]}

    assert json.loads(_to_decision_input(state)) == state  # type: ignore[arg-type]


def test_messages_are_sent_as_role_content_json() -> None:
    state = [SystemMessage("Be helpful."), HumanMessage("Hi"), AIMessage("Hello")]

    assert json.loads(_to_decision_input(state)) == [  # type: ignore[arg-type]
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_media_blocks_in_json_are_sent_in_place() -> None:
    state: dict[str, Any] = {
        "before": "x",
        "photo": {"type": "image", "base64": "AAAA", "mime_type": "image/png"},
        "recording": {"type": "audio", "url": "https://example.com/a.mp3"},
        "after": "y",
    }

    content = _to_decision_input(state)

    assert isinstance(content, list)
    assert [block["type"] for block in content] == [
        "text",
        "image",
        "text",
        "audio",
        "text",
    ]
    assert content[1] == {"type": "image", "data": "AAAA", "mime_type": "image/png"}
    assert content[3] == {"type": "audio", "uri": "https://example.com/a.mp3"}
    text = "".join(block.get("text", "<media>") for block in content)
    assert text == (
        '{"before": "x", "photo": "<media>", "recording": "<media>", "after": "y"}'
    )


def test_media_in_messages_is_sent_in_place() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Look:"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZZ"}},
            {"type": "file", "base64": "UERG", "mime_type": "application/pdf"},
        ]
    )

    content = _to_decision_input(message)

    assert isinstance(content, list)
    media = [block for block in content if block["type"] != "text"]
    assert media == [
        {"type": "image", "data": "ZZ", "mime_type": "image/jpeg"},
        {"type": "document", "data": "UERG", "mime_type": "application/pdf"},
    ]
    assert content[0]["text"].startswith('{"role": "user"')


def test_dicts_without_media_data_are_json() -> None:
    state = {"type": "image", "caption": "no data here"}

    assert json.loads(_to_decision_input(state)) == state  # type: ignore[arg-type]


def test_message_media_without_supported_data_raises() -> None:
    message = HumanMessage(content=[{"type": "image", "file_id": "file-123"}])

    with pytest.raises(ValueError, match="Only `base64` and `url`"):
        _to_decision_input(message)


@pytest.mark.parametrize("state", [None, 1, 1.5, True])
def test_scalar_state_raises(state: Any) -> None:
    with pytest.raises(TypeError, match="state must be"):
        _to_decision_input(state)


def test_unsupported_state_value_raises() -> None:
    with pytest.raises(TypeError, match="Unsupported classifier state value"):
        _to_decision_input({"value": object()})  # type: ignore[dict-item]
