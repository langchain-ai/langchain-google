"""Tests for Vertex AI Claude prompt caching middleware."""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.runtime import Runtime

from langchain_google_vertexai.middleware import VertexPromptCachingMiddleware
from langchain_google_vertexai.middleware.prompt_caching import (
    ModelRequest,
    ModelResponse,
)
from langchain_google_vertexai.model_garden import ChatAnthropicVertex


def _request(model: Any, **kwargs: Any) -> ModelRequest:
    defaults: dict[str, Any] = {
        "model": model,
        "messages": [HumanMessage("Hello")],
        "system_message": None,
        "tool_choice": None,
        "tools": [],
        "response_format": None,
        "state": {"messages": [HumanMessage("Hello")]},
        "runtime": cast("Runtime", object()),
        "model_settings": {},
    }
    defaults.update(kwargs)
    return ModelRequest(**defaults)


def _vertex_model() -> MagicMock:
    return MagicMock(spec=ChatAnthropicVertex)


def _response() -> ModelResponse:
    return ModelResponse(result=[AIMessage(content="ok")])


def test_tags_system_tools_and_model_settings() -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return location

    request = _request(
        _vertex_model(),
        system_message=SystemMessage("Be concise"),
        tools=[get_weather],
        model_settings={"temperature": 0},
    )
    captured: ModelRequest | None = None

    def handler(value: ModelRequest) -> ModelResponse:
        nonlocal captured
        captured = value
        return _response()

    VertexPromptCachingMiddleware().wrap_model_call(request, handler)

    assert captured is not None
    assert captured.model_settings == {
        "temperature": 0,
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
    }
    assert captured.system_message is not None
    assert captured.system_message.content == [
        {
            "type": "text",
            "text": "Be concise",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }
    ]
    assert captured.tools is not None
    cached_tool = captured.tools[0]
    assert isinstance(cached_tool, BaseTool)
    assert cached_tool.extras == {"cache_control": {"type": "ephemeral", "ttl": "5m"}}
    assert get_weather.extras is None


def test_uses_last_system_block_and_tool() -> None:
    @tool
    def first_tool(value: str) -> str:
        """Return a value."""
        return value

    @tool(extras={"defer_loading": True})
    def last_tool(value: str) -> str:
        """Return a value."""
        return value

    request = _request(
        _vertex_model(),
        system_message=SystemMessage(
            content=[
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Last"},
            ]
        ),
        tools=[first_tool, last_tool],
    )
    captured: ModelRequest | None = None

    def handler(value: ModelRequest) -> ModelResponse:
        nonlocal captured
        captured = value
        return _response()

    VertexPromptCachingMiddleware(ttl="1h").wrap_model_call(request, handler)

    assert captured is not None
    assert captured.system_message is not None
    assert captured.system_message.content == [
        {"type": "text", "text": "First"},
        {
            "type": "text",
            "text": "Last",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    ]
    assert captured.tools is not None
    assert captured.tools[0] is first_tool
    cached_tool = captured.tools[-1]
    assert isinstance(cached_tool, BaseTool)
    assert cached_tool.extras == {
        "defer_loading": True,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }


@pytest.mark.parametrize("behavior", ["ignore", "warn"])
def test_unsupported_model_is_not_modified(behavior: str) -> None:
    request = _request(MagicMock(), model_settings={"temperature": 0})
    captured: ModelRequest | None = None

    def handler(value: ModelRequest) -> ModelResponse:
        nonlocal captured
        captured = value
        return _response()

    middleware = VertexPromptCachingMiddleware(
        unsupported_model_behavior=cast("Any", behavior)
    )
    if behavior == "warn":
        with pytest.warns(UserWarning, match="only supports ChatAnthropicVertex"):
            middleware.wrap_model_call(request, handler)
    else:
        middleware.wrap_model_call(request, handler)

    assert captured is request
    assert captured.model_settings == {"temperature": 0}


def test_unsupported_model_can_raise() -> None:
    def handler(value: ModelRequest) -> ModelResponse:
        return _response()

    with pytest.raises(ValueError, match="only supports ChatAnthropicVertex"):
        VertexPromptCachingMiddleware(
            unsupported_model_behavior="raise"
        ).wrap_model_call(_request(MagicMock()), handler)


async def test_async_applies_caching_and_respects_minimum_message_count() -> None:
    captured: ModelRequest | None = None

    async def handler(value: ModelRequest) -> ModelResponse:
        nonlocal captured
        captured = value
        return _response()

    middleware = VertexPromptCachingMiddleware(min_messages_to_cache=2)
    request = _request(_vertex_model())
    await middleware.awrap_model_call(request, handler)
    assert captured is request

    request = _request(
        _vertex_model(), messages=[HumanMessage("One"), HumanMessage("Two")]
    )
    await middleware.awrap_model_call(request, handler)
    assert captured is not None
    assert captured.model_settings["cache_control"] == {
        "type": "ephemeral",
        "ttl": "5m",
    }
