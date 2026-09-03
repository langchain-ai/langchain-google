"""Anthropic Vertex prompt caching middleware.

Requires:
    - `langchain`: For agent middleware framework
    - `langchain-google-vertexai[anthropic]`: For `ChatAnthropicVertex`
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal
from warnings import warn

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from langchain_google_vertexai.model_garden import ChatAnthropicVertex

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        ModelCallResult,
        ModelRequest,
        ModelResponse,
    )
except ImportError as e:
    msg = (
        "AnthropicVertexPromptCachingMiddleware requires 'langchain' to be "
        "installed. Install it with: pip install langchain"
    )
    raise ImportError(msg) from e


def _get_trace_policy() -> Any:
    """Use tracing optimizations when supported by the installed LangChain."""
    from langchain.agents.middleware import types

    trace_policy = getattr(types, "TracePolicy", None)
    omit_payload = getattr(types, "omit_payload", None)
    if trace_policy is None or omit_payload is None:
        return None
    return trace_policy(process_inputs=omit_payload)


_TRACE_POLICY = _get_trace_policy()


class AnthropicVertexPromptCachingMiddleware(AgentMiddleware):
    """Add Anthropic prompt caching to `ChatAnthropicVertex` model calls.

    The middleware tags stable system and tool content and passes `cache_control`
    through `model_settings` for the model to place on the message tail.
    """

    trace_policy = _TRACE_POLICY

    def __init__(
        self,
        type: Literal["ephemeral"] = "ephemeral",  # noqa: A002
        ttl: Literal["5m", "1h"] = "5m",
        min_messages_to_cache: int = 0,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "warn",
    ) -> None:
        """Initialize prompt caching.

        Args:
            type: Cache type.
            ttl: Cache lifetime.
            min_messages_to_cache: Minimum message count before caching.
            unsupported_model_behavior: Behavior for non-Vertex Anthropic models.
        """
        self.type = type
        self.ttl = ttl
        self.min_messages_to_cache = min_messages_to_cache
        self.unsupported_model_behavior = unsupported_model_behavior

    @property
    def _cache_control(self) -> dict[str, str]:
        return {"type": self.type, "ttl": self.ttl}

    def _should_apply_caching(self, request: ModelRequest) -> bool:
        if not isinstance(request.model, ChatAnthropicVertex):
            msg = (
                "AnthropicVertexPromptCachingMiddleware only supports "
                f"ChatAnthropicVertex, not instances of {type(request.model)}"
            )
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            return False

        messages_count = len(request.messages) + (request.system_message is not None)
        return messages_count >= self.min_messages_to_cache

    def _apply_caching(self, request: ModelRequest) -> ModelRequest:
        cache_control = self._cache_control
        overrides: dict[str, Any] = {
            "model_settings": {
                **request.model_settings,
                "cache_control": cache_control,
            }
        }

        system_message = _tag_system_message(request.system_message, cache_control)
        if system_message is not request.system_message:
            overrides["system_message"] = system_message

        tools = _tag_tools(request.tools, cache_control)
        if tools is not request.tools:
            overrides["tools"] = tools

        return request.override(**overrides)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Add caching to a synchronous model call.

        Args:
            request: Model request.
            handler: Model call handler.

        Returns:
            Model call result.
        """
        if not self._should_apply_caching(request):
            return handler(request)
        return handler(self._apply_caching(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Add caching to an asynchronous model call.

        Args:
            request: Model request.
            handler: Asynchronous model call handler.

        Returns:
            Model call result.
        """
        if not self._should_apply_caching(request):
            return await handler(request)
        return await handler(self._apply_caching(request))


def _tag_system_message(
    system_message: Any,
    cache_control: dict[str, str],
) -> Any:
    if system_message is None:
        return system_message

    content = system_message.content
    if isinstance(content, str):
        if not content:
            return system_message
        new_content: list[str | dict[str, Any]] = [
            {"type": "text", "text": content, "cache_control": cache_control}
        ]
    elif isinstance(content, list):
        if not content:
            return system_message
        new_content = list(content)
        last = new_content[-1]
        base = last if isinstance(last, dict) else {}
        new_content[-1] = {**base, "cache_control": cache_control}
    else:
        return system_message

    return SystemMessage(content=new_content)


def _tag_tools(
    tools: list[Any] | None,
    cache_control: dict[str, str],
) -> list[Any] | None:
    if not tools:
        return tools

    last = tools[-1]
    if not isinstance(last, BaseTool):
        return tools

    extras = {**(last.extras or {}), "cache_control": cache_control}
    return [*tools[:-1], last.model_copy(update={"extras": extras})]
