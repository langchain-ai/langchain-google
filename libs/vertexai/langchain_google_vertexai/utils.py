from datetime import datetime, timedelta
from typing import List, Optional

from langchain_core.messages import BaseMessage
from vertexai.preview import caching  # type: ignore

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
    messages: List[BaseMessage],
    expire_time: Optional[datetime] = None,
    time_to_live: Optional[timedelta] = None,
    tools: Optional[_ToolsType] = None,
    tool_config: Optional[_ToolConfigDict] = None,
) -> str:
    """Creates a cache for content in some model.

    Args:
        model: ChatVertexAI model. Must be at least gemini-2.5-pro or gemini-2.0-flash.
        messages: List of messages to cache.
        expire_time:  Timestamp of when this resource is considered expired.
        At most one of expire_time and ttl can be set. If neither is set, default TTL
            on the API side will be used (currently 1 hour).
        time_to_live:  The TTL for this resource. If provided, the expiration time is
        computed: created_time + TTL.
        At most one of expire_time and ttl can be set. If neither is set, default TTL
            on the API side will be used (currently 1 hour).
        tools:  A list of tool definitions to bind to this chat model.
            Can be a pydantic model, callable, or BaseTool. Pydantic
            models, callables, and BaseTools will be automatically converted to
            their schema dictionary representation.
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

    cached_content = caching.CachedContent.create(
        model_name=model.full_model_name,
        system_instruction=system_instruction,
        contents=contents,
        ttl=time_to_live,
        expire_time=expire_time,
        tool_config=tool_config,
        tools=tools,
    )

    return cached_content.name
