"""Deterministic unit tests for `betas` propagation in `ChatAnthropicVertex`.

These verify, without any live API calls, that passing `betas` routes the
request through the Anthropic *beta* client for both async invoke
(`_agenerate`) and async streaming (`_astream`).
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from langchain_google_vertexai.model_garden import ChatAnthropicVertex

_BETA = "context-1m-2025-08-07"
_MODEL = "claude-sonnet-4-5@20250929"


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5


class _FakeResponse:
    usage = _FakeUsage()

    def model_dump(self) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": "4"}],
            "role": "assistant",
            "type": "message",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }


class _FakeTextDelta:
    type = "text_delta"
    text = "hello"


class _FakeDeltaEvent:
    type = "content_block_delta"
    index = 0
    delta = _FakeTextDelta()


async def _text_event_stream() -> AsyncIterator[Any]:
    yield _FakeDeltaEvent()


def _build_chat() -> ChatAnthropicVertex:
    with (
        patch("anthropic.AnthropicVertex") as mock_sync_client,
        patch("anthropic.AsyncAnthropicVertex") as mock_async_client,
    ):
        mock_sync_client.return_value = Mock()
        mock_async_client.return_value = Mock()
        return ChatAnthropicVertex(project="test-project", model_name=_MODEL)


async def test_async_invoke_routes_betas_to_beta_client() -> None:
    """`ainvoke` with `betas` must call the async beta client."""
    chat = _build_chat()
    chat.async_client.beta.messages.create = AsyncMock(return_value=_FakeResponse())
    chat.async_client.messages.create = AsyncMock(return_value=_FakeResponse())

    await chat.ainvoke("What is 2+2?", model_name=_MODEL, betas=[_BETA])

    chat.async_client.beta.messages.create.assert_awaited_once()
    chat.async_client.messages.create.assert_not_awaited()
    assert chat.async_client.beta.messages.create.call_args.kwargs["betas"] == [_BETA]


async def test_async_stream_routes_betas_to_beta_client() -> None:
    """`astream` with `betas` must call the async beta client."""
    chat = _build_chat()
    chat.async_client.beta.messages.create = AsyncMock(
        return_value=_text_event_stream()
    )
    chat.async_client.messages.create = AsyncMock(return_value=_text_event_stream())

    async for _ in chat.astream("Say hello", model=_MODEL, betas=[_BETA]):
        pass

    chat.async_client.beta.messages.create.assert_awaited_once()
    chat.async_client.messages.create.assert_not_awaited()
    call_kwargs = chat.async_client.beta.messages.create.call_args.kwargs
    assert call_kwargs["betas"] == [_BETA]
    assert call_kwargs["stream"] is True


async def test_async_invoke_without_betas_uses_standard_client() -> None:
    """Without `betas`, the standard (non-beta) async client is used."""
    chat = _build_chat()
    chat.async_client.beta.messages.create = AsyncMock(return_value=_FakeResponse())
    chat.async_client.messages.create = AsyncMock(return_value=_FakeResponse())

    await chat.ainvoke("What is 2+2?", model_name=_MODEL)

    chat.async_client.messages.create.assert_awaited_once()
    chat.async_client.beta.messages.create.assert_not_awaited()
