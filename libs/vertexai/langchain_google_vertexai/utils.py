from datetime import datetime, timedelta
from typing import Any, cast

from langchain_core.messages import BaseMessage
from vertexai.preview import caching

from langchain_google_vertexai._image_utils import ImageBytesLoader
from langchain_google_vertexai.chat_models import (
    ChatVertexAI,
    _parse_chat_history_gemini,
)
from langchain_google_vertexai.functions_utils import (
    _format_to_gapic_tool,
    _format_tool_config,
    _ToolConfigDict,
    _ToolsType,
)


def create_context_cache(
    model: ChatVertexAI,
    messages: list[BaseMessage],
    expire_time: datetime | None = None,
    time_to_live: timedelta | None = None,
    tools: _ToolsType | None = None,
    tool_config: _ToolConfigDict | None = None,
) -> str:
    """Creates a cache for content in some model.

    Args:
        model: `ChatVertexAI` model. Must be at least `gemini-2.5-pro` or
            `gemini-2.0-flash`.
        messages: List of messages to cache.
        expire_time: Timestamp of when this resource is considered expired.

            At most one of `expire_time` and `time_to_live` can be set. If neither is
            set, default TTL on the API side will be used (currently 1 hour).
        time_to_live: The TTL for this resource. If provided, the expiration time is
            computed as `created_time` + TTL.

            At most one of `expire_time` and `time_to_live` can be set. If neither is
            set, default TTL on the API side will be used (currently 1 hour).
        tools: A list of tool definitions to bind to this chat model.

            Can be a Pydantic model, `Callable`, or `BaseTool`. Pydantic models,
            `Callable`, and `BaseTool` will be automatically converted to their schema
            dictionary representation.
        tool_config: Optional. Immutable. Tool config. This config is shared for all
            tools.

    Raises:
        ValueError: If model doesn't support context catching.

    Returns:
        String with the identificator of the created cache.
    """
    system_instruction, contents = _parse_chat_history_gemini(
        messages, ImageBytesLoader(project=model.project)
    )

    if tool_config:
        tool_config = _format_tool_config(tool_config)

    if tools is not None:
        tools = [_format_to_gapic_tool(tools)]

    if model.full_model_name is None:
        raise ValueError("Model must have a full_model_name to create cached content")

    cached_content = caching.CachedContent.create(
        model_name=model.full_model_name,
        system_instruction=system_instruction,
        contents=cast("list[Any] | None", contents),
        ttl=time_to_live,
        expire_time=expire_time,
        tool_config=tool_config,
        tools=cast("list[Any] | None", tools),
    )

    return cached_content.name
