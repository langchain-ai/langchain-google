"""Tests for Anthropic Vertex prompt caching middleware."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from google.auth.credentials import AnonymousCredentials
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from langchain_google_vertexai.middleware import (
    AnthropicVertexPromptCachingMiddleware,
)
from langchain_google_vertexai.model_garden import ChatAnthropicVertex


def _model() -> ChatAnthropicVertex:
    return ChatAnthropicVertex(
        model="claude-sonnet-4-6",
        project="test-project",
        location="us-east5",
        credentials=AnonymousCredentials(),
    )


def _request(**kwargs: Any) -> ModelRequest:
    defaults: dict[str, Any] = {
        "model": _model(),
        "messages": [HumanMessage("Hello")],
        "system_message": None,
        "tools": [],
        "model_settings": {},
    }
    defaults.update(kwargs)
    return ModelRequest(**defaults)


def _apply(
    middleware: AnthropicVertexPromptCachingMiddleware,
    request: ModelRequest,
) -> ModelRequest:
    captured: ModelRequest | None = None

    def handler(modified: ModelRequest) -> ModelResponse:
        nonlocal captured
        captured = modified
        return ModelResponse(result=[AIMessage("ok")])

    middleware.wrap_model_call(request, handler)
    assert captured is not None
    return captured


def test_adds_cache_control_to_vertex_request() -> None:
    result = _apply(AnthropicVertexPromptCachingMiddleware(), _request())

    assert result.model_settings["cache_control"] == {
        "type": "ephemeral",
        "ttl": "5m",
    }


def test_respects_message_threshold() -> None:
    request = _request(
        messages=[HumanMessage("one")],
        system_message=SystemMessage("system"),
    )
    result = _apply(
        AnthropicVertexPromptCachingMiddleware(min_messages_to_cache=3),
        request,
    )

    assert result is request


def test_tags_system_message_and_last_tool() -> None:
    @tool
    def first_tool() -> str:
        """Return the first result."""
        return "first"

    @tool
    def last_tool() -> str:
        """Return the last result."""
        return "last"

    request = _request(
        system_message=SystemMessage("stable instructions"),
        tools=[first_tool, last_tool],
    )
    result = _apply(AnthropicVertexPromptCachingMiddleware(ttl="1h"), request)

    assert result.system_message is not None
    system_block = result.system_message.content[-1]
    assert isinstance(system_block, dict)
    assert system_block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert result.tools[0] is first_tool
    result_tool = result.tools[-1]
    request_tool = request.tools[-1]
    assert isinstance(result_tool, BaseTool)
    assert isinstance(request_tool, BaseTool)
    assert result_tool.extras == {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    assert request_tool.extras is None


def test_unsupported_model_behavior() -> None:
    request = _request(model=FakeListChatModel(responses=["ok"]))

    with pytest.raises(ValueError, match="only supports ChatAnthropicVertex"):
        _apply(
            AnthropicVertexPromptCachingMiddleware(unsupported_model_behavior="raise"),
            request,
        )

    with pytest.warns(UserWarning, match="only supports ChatAnthropicVertex"):
        result = _apply(AnthropicVertexPromptCachingMiddleware(), request)
    assert result is request

    result = _apply(
        AnthropicVertexPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        request,
    )
    assert result is request


async def test_async_adds_cache_control() -> None:
    request = _request()
    captured: ModelRequest | None = None

    async def handler(modified: ModelRequest) -> ModelResponse:
        nonlocal captured
        captured = modified
        return ModelResponse(result=[AIMessage("ok")])

    await AnthropicVertexPromptCachingMiddleware().awrap_model_call(request, handler)

    assert captured is not None
    assert captured.model_settings["cache_control"] == {
        "type": "ephemeral",
        "ttl": "5m",
    }


def test_model_is_checked_by_type() -> None:
    spoofed = MagicMock()
    spoofed._llm_type = "anthropic-chat-vertexai"
    request = _request(model=spoofed)

    result = _apply(
        AnthropicVertexPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        request,
    )

    assert result is request
