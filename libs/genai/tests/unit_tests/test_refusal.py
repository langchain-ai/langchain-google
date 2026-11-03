"""Gemini safety blocks surface as standard refusal content blocks."""

from typing import Any
from unittest.mock import patch

import pytest
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    GenerateContentResponse,
    GenerateContentResponsePromptFeedback,
    Part,
)
from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RefusalContentBlock,
)
from pydantic import BaseModel, SecretStr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import _parse_chat_history, _response_to_result

MODEL_NAME = "gemini-2.5-flash"


class _Answer(BaseModel):
    answer: str


def _refusal(message: BaseMessage) -> RefusalContentBlock | None:
    for block in message.content_blocks:
        if block["type"] == "refusal":
            return block
    return None


def _blocked_candidate() -> GenerateContentResponse:
    return GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(parts=[]),
                finish_reason=FinishReason.SAFETY,
                finish_message="Blocked for safety.",
            )
        ]
    )


def _blocked_prompt() -> GenerateContentResponse:
    return GenerateContentResponse(
        candidates=[],
        prompt_feedback=GenerateContentResponsePromptFeedback(block_reason="SAFETY"),
    )


@pytest.mark.parametrize("stream", [False, True])
def test_blocked_candidate_is_refusal(*, stream: bool) -> None:
    message = _response_to_result(_blocked_candidate(), stream=stream).generations[0]
    assert _refusal(message.message) == {
        "type": "refusal",
        "refusal": "Blocked for safety.",
        "index": "lc_refusal",
        "extras": {"finish_reason": "SAFETY"},
    }


@pytest.mark.parametrize("stream", [False, True])
def test_blocked_prompt_is_refusal(*, stream: bool) -> None:
    message = _response_to_result(_blocked_prompt(), stream=stream).generations[0]
    assert message.message.response_metadata["model_provider"] == "google_genai"
    refusal = _refusal(message.message)
    assert refusal is not None
    assert refusal.get("extras") == {"block_reason": "SAFETY"}


def test_normal_stop_is_not_refusal() -> None:
    response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(parts=[Part(text="Hi.")], role="model"),
                finish_reason=FinishReason.STOP,
            )
        ]
    )
    message = _response_to_result(response).generations[0].message
    assert _refusal(message) is None


@pytest.mark.parametrize("response", [_blocked_candidate(), _blocked_prompt()])
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_structured_output_raises_on_refusal(
    response: GenerateContentResponse, output_version: str
) -> None:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr("fake-key"),
        output_version=output_version,
    )
    structured = llm.with_structured_output(_Answer)

    def _generate(*_: Any, **__: Any) -> Any:
        return _response_to_result(response)

    with (
        patch.object(ChatGoogleGenerativeAI, "_generate", _generate),
        pytest.raises(ModelRefusalError),
    ):
        structured.invoke("hi")


@pytest.mark.parametrize("model_provider", ["google_genai", "openai", "anthropic"])
def test_refusal_dropped_from_request(model_provider: str) -> None:
    refused = AIMessage(
        content_blocks=[
            {"type": "text", "text": "Here is"},
            {"type": "refusal", "refusal": "No.", "index": "lc_refusal"},
        ],
        response_metadata={"model_provider": model_provider, "output_version": "v1"},
    )
    _, contents = _parse_chat_history(
        [HumanMessage("hi"), refused, HumanMessage("next")]
    )
    sent = str(contents)
    assert "Here is" in sent
    assert "No." not in sent
