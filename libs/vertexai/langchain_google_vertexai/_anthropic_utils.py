import base64
import re
import urllib
import warnings
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
    cast,
)

import validators
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from langchain_google_vertexai._image_utils import (
    ImageBytesLoader,
)
from langchain_google_vertexai._utils import load_image_from_gcs

if TYPE_CHECKING:
    from anthropic.types import (
        RawMessageStreamEvent,  # type: ignore[unused-ignore, import-not-found]
    )

_message_type_lookups = {
    "human": "user",
    "ai": "assistant",
    "AIMessageChunk": "assistant",
    "HumanMessageChunk": "user",
}


def _create_usage_metadata(anthropic_usage: BaseModel) -> UsageMetadata:
    """Create `UsageMetadata` from Anthropic usage with proper cache token handling.

    This matches the official `langchain_anthropic` implementation exactly.
    """
    input_token_details: dict = {
        "cache_read": getattr(anthropic_usage, "cache_read_input_tokens", None),
        "cache_creation": getattr(anthropic_usage, "cache_creation_input_tokens", None),
    }

    # Anthropic input_tokens exclude cached token counts.
    input_tokens = (
        (getattr(anthropic_usage, "input_tokens", 0) or 0)
        + (input_token_details["cache_read"] or 0)
        + (input_token_details["cache_creation"] or 0)
    )
    output_tokens = getattr(anthropic_usage, "output_tokens", 0) or 0

    # Only add input_token_details if we have non-None cache values
    filtered_details = {k: v for k, v in input_token_details.items() if v is not None}
    if filtered_details:
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_token_details=InputTokenDetails(**filtered_details),
        )
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _format_image(image_url: str, project: str | None) -> dict:
    """Formats a message image to a dict for Anthropic API."""
    regex = r"^data:(?P<media_type>(?:image|application)/.+);base64,(?P<data>.+)$"
    match = re.match(regex, image_url)
    if match:
        return {
            "type": "base64",
            "media_type": match.group("media_type"),
            "data": match.group("data"),
        }
    if validators.url(image_url):
        loader = ImageBytesLoader(project=project)
        image_bytes = loader.load_bytes(image_url)
        path = urllib.parse.urlparse(image_url).path
        raw_mime_type = path.split(".")[-1].lower()
        doc_type = "application" if raw_mime_type == "pdf" else "image"
        mime_type = (
            f"{doc_type}/jpeg"
            if raw_mime_type == "jpg"
            else f"{doc_type}/{raw_mime_type}"
        )
        return {
            "type": "base64",
            "media_type": mime_type,
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }
    if image_url.startswith("gs://"):
        # Gets image and encodes to base64.
        loader = ImageBytesLoader(project=project)
        part = loader.load_part(image_url)
        if part.file_data.mime_type:
            mime_type = part.file_data.mime_type
            image_data = load_image_from_gcs(image_url, project=project).data
        else:
            mime_type = part.inline_data.mime_type
            image_data = part.inline_data.data
        return {
            "type": "base64",
            "media_type": mime_type,
            "data": base64.b64encode(image_data).decode("ascii"),
        }
    msg = (
        "Anthropic only supports base64-encoded images and urls currently."
        " Example: data:image/png;base64,'/9j/4AAQSk'..."
        " Example: https://your-valid-image-url.png"
    )
    raise ValueError(msg)


def _get_cache_control(message: BaseMessage) -> dict[str, Any] | None:
    """Extract cache control from message's `additional_kwargs` or content block."""
    return (
        message.additional_kwargs.get("cache_control")
        if isinstance(message.additional_kwargs, dict)
        else None
    )


def _format_text_content(text: str) -> dict[str, str | dict[str, Any]]:
    """Format text content."""
    content: dict[str, str | dict[str, Any]] = {"type": "text", "text": text}
    return content


def _format_message_anthropic(
    message: HumanMessage | AIMessage | SystemMessage, project: str | None
):
    """Format a message for Anthropic API.

    Args:
        message: The message to format. Can be `HumanMessage`, `AIMessage`, or
            `SystemMessage`.

    Returns:
        A `dict` with the formatted message, or `None` if the message is empty.
    """
    content: list[dict[str, Any]] = []

    if isinstance(message.content, str):
        if not message.content.strip():
            if not (isinstance(message, AIMessage) and message.tool_calls):
                # We still have tool calls to process
                return None
        else:
            message_dict = _format_text_content(message.content)
            if cache_control := _get_cache_control(message):
                message_dict["cache_control"] = cache_control
            content.append(message_dict)
    elif isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, str):
                # Only add non-empty strings for now as empty ones are not
                # accepted.
                # https://github.com/anthropics/anthropic-sdk-python/issues/461
                if not block.strip():
                    continue
                content.append(_format_text_content(block))
            elif isinstance(block, dict):
                if "type" not in block:
                    msg = "Dict content block must have a type key"
                    raise ValueError(msg)

                new_block = {}

                for copy_attr in ["type", "cache_control"]:
                    if copy_attr in block:
                        new_block[copy_attr] = block[copy_attr]

                if block["type"] == "image":
                    if "url" in block:
                        url = block["url"]
                        if url.startswith("data:"):
                            # Data URI
                            formatted_block = {
                                "type": "image",
                                "source": _format_image(url, project),
                            }
                        else:
                            formatted_block = {
                                "type": "image",
                                "source": {"type": "url", "url": url},
                            }
                    elif "base64" in block:
                        formatted_block = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block["mime_type"],
                                "data": block["base64"],
                            },
                        }
                    elif "file_id" in block:
                        formatted_block = {
                            "type": "image",
                            "source": {
                                "type": "file",
                                "file_id": block["file_id"],
                            },
                        }
                    # Backward compatibility for langchain < 1.X
                    # where source_type was used
                    elif "data" in block and block.get("source_type", None) == "base64":
                        formatted_block = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block["mime_type"],
                                "data": block["data"],
                            },
                        }
                    elif "id" in block and block.get("source_type", None) == "id":
                        formatted_block = {
                            "type": "image",
                            "source": {
                                "type": "file",
                                "file_id": block["id"],
                            },
                        }
                    else:
                        msg = (
                            "Image content blocks must have either 'url', 'base64', "
                            "'file_id', 'id' or 'data' field."
                        )
                        raise ValueError(msg)
                    content.append(formatted_block)
                    continue

                if block["type"] == "text":
                    text: str = block.get("text", "")
                    # Only add non-empty strings for now as empty ones are not
                    # accepted.
                    # https://github.com/anthropics/anthropic-sdk-python/issues/461
                    if text.strip():
                        new_block["text"] = text
                        content.append(new_block)
                    continue

                if block["type"] == "thinking":
                    content.append(
                        {
                            k: v
                            for k, v in block.items()
                            if k in ("type", "thinking", "cache_control", "signature")
                        }
                    )
                    continue

                if block["type"] == "redacted_thinking":
                    content.append(
                        {
                            k: v
                            for k, v in block.items()
                            if k in ("type", "cache_control", "data")
                        }
                    )
                    continue

                if block["type"] == "image_url":
                    # convert format
                    source = _format_image(block["image_url"]["url"], project)
                    if source["media_type"] == "application/pdf":
                        doc_type = "document"
                    else:
                        doc_type = "image"
                    content.append({"type": doc_type, "source": source})
                    continue

                if block["type"] == "tool_use":
                    # If a tool_call with the same id as a tool_use content block
                    # exists, the tool_call is preferred.
                    if isinstance(message, AIMessage) and message.tool_calls:
                        is_unique = block["id"] not in [
                            tc["id"] for tc in message.tool_calls
                        ]
                        if not is_unique:
                            continue

                content.append(block)
    else:
        msg = "Message should be a str, list of str or list of dicts"  # type: ignore[unreachable, unused-ignore]
        raise ValueError(msg)

    if isinstance(message, AIMessage) and message.tool_calls:
        for tc in message.tool_calls:
            tu = cast("dict[str, Any]", _lc_tool_call_to_anthropic_tool_use_block(tc))
            content.append(tu)

    if not content:
        return None

    if message.type == "system":
        return content
    return {"role": _message_type_lookups[message.type], "content": content}


def _format_messages_anthropic(
    messages: list[BaseMessage],
    project: str | None,
) -> tuple[dict[str, Any] | None, list[dict]]:
    """Formats messages for Anthropic."""
    system_messages: dict[str, Any] | None = None
    formatted_messages: list[dict] = []

    merged_messages = _merge_messages(messages)
    for i, message in enumerate(merged_messages):
        if message.type == "system":
            if i != 0:
                msg = "System message must be at beginning of message list."
                raise ValueError(msg)
            fm = _format_message_anthropic(message, project)
            if fm:
                system_messages = fm
            continue

        fm = _format_message_anthropic(message, project)
        if not fm:
            continue
        formatted_messages.append(fm)

    return system_messages, formatted_messages


class AnthropicTool(TypedDict):
    name: str
    description: str
    input_schema: dict[str, Any]


def convert_to_anthropic_tool(
    tool: dict[str, Any] | type[BaseModel] | Callable | BaseTool,
) -> AnthropicTool:
    # Already in Anthropic tool format
    if isinstance(tool, dict) and all(
        k in tool for k in ("name", "description", "input_schema")
    ):
        return AnthropicTool(tool)  # type: ignore
    formatted = convert_to_openai_tool(tool)["function"]
    return AnthropicTool(
        name=formatted["name"],
        description=formatted["description"],
        input_schema=formatted["parameters"],
    )


def _clean_content_block(block: Any) -> Any:
    """Remove streaming metadata fields from content blocks.

    Anthropic's streaming API adds `index` and `partial_json` fields to content blocks
    during streaming. These fields must be removed before sending back to the API.

    Args:
        block: Content block (`dict`, `str`, or other type)

    Returns:
        Cleaned content block with streaming metadata removed
    """
    if not isinstance(block, dict):
        return block

    # Remove known streaming metadata fields
    # 'index' - added during streaming to track block position
    # 'partial_json' - added during streaming for incremental JSON parsing
    # Remove known streaming metadata fields
    keys_to_remove = {"index", "partial_json", "caller"}

    # The id field is required for tool_use blocks and some image blocks,
    # but forbidden in text blocks (specifically inside tool_results).
    if block.get("type") not in ("tool_use", "image"):
        keys_to_remove.add("id")

    return {k: v for k, v in block.items() if k not in keys_to_remove}


def _clean_content(content: Any) -> Any:
    """Recursively clean content (`str`, `list`, or `dict`).

    Args:
        content: Content to clean (can be `str`, `list`, `dict`, or other)

    Returns:
        Cleaned content with streaming metadata removed
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_clean_content_block(block) for block in content]
    if isinstance(content, dict):
        return _clean_content_block(content)
    return content


def _merge_messages(
    messages: Sequence[BaseMessage],
) -> list[SystemMessage | AIMessage | HumanMessage]:
    """Merge runs of human/tool messages into single human messages with content blocks."""  # noqa: E501
    merged: list = []
    for curr in messages:
        curr = curr.model_copy(deep=True)
        if isinstance(curr, ToolMessage):
            # Check if already in tool_result format (backward compatibility)
            if isinstance(curr.content, list) and all(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in curr.content
            ):
                # Already formatted - just convert to HumanMessage and clean content
                cleaned_content = _clean_content(curr.content)
                curr = HumanMessage(cleaned_content)
            else:
                # Convert to tool_result format
                tool_result_block = {
                    "type": "tool_result",
                    "content": _clean_content(curr.content),
                    "tool_use_id": curr.tool_call_id,
                }
                # Add error flag if present
                if curr.status == "error":
                    tool_result_block["is_error"] = True

                cache_control = None
                if isinstance(curr.additional_kwargs, dict):
                    cache_control = curr.additional_kwargs.get("cache_control")
                if cache_control:
                    tool_result_block["cache_control"] = cache_control

                curr = HumanMessage([tool_result_block])
        elif isinstance(curr, AIMessage):
            # Clean streaming metadata from AIMessage content blocks
            if isinstance(curr.content, list):
                cleaned_content = _clean_content(curr.content)
                if cleaned_content != curr.content:
                    curr = curr.model_copy(deep=True)
                    curr.content = cleaned_content
        last = merged[-1] if merged else None
        if isinstance(last, HumanMessage) and isinstance(curr, HumanMessage):
            if isinstance(last.content, str):
                new_content: list = [{"type": "text", "text": last.content}]
            else:
                new_content = last.content
            if isinstance(curr.content, str):
                new_content.append({"type": "text", "text": curr.content})
            else:
                new_content.extend(curr.content)
            last.content = new_content
        else:
            merged.append(curr)
    return merged


class _AnthropicToolUse(TypedDict):
    type: Literal["tool_use"]
    name: str
    input: dict
    id: str


def _lc_tool_call_to_anthropic_tool_use_block(
    tool_call: ToolCall,
) -> _AnthropicToolUse:
    return _AnthropicToolUse(
        type="tool_use",
        name=tool_call["name"],
        input=tool_call["args"],
        id=cast("str", tool_call["id"]),
    )


def _make_message_chunk_from_anthropic_event(
    event: "RawMessageStreamEvent",
    *,
    stream_usage: bool = True,
    coerce_content_to_string: bool,
) -> AIMessageChunk | None:
    """Convert Anthropic event to `AIMessageChunk`.

    Note that not all events will result in a message chunk. In these cases we return
    `None`.
    """
    message_chunk: AIMessageChunk | None = None
    # See https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/streaming/_messages.py  # noqa: E501
    if event.type == "message_start" and stream_usage:
        # Follow official langchain_anthropic pattern exactly
        usage_metadata = _create_usage_metadata(event.message.usage)
        # We pick up a cumulative count of output_tokens at the end of the stream,
        # so here we zero out to avoid double counting.
        usage_metadata["total_tokens"] = (
            usage_metadata["total_tokens"] - usage_metadata["output_tokens"]
        )
        usage_metadata["output_tokens"] = 0
        if hasattr(event.message, "model"):
            response_metadata = {"model_name": event.message.model}
        else:
            response_metadata = {}
        message_chunk = AIMessageChunk(
            content="" if coerce_content_to_string else [],
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )
    elif (
        event.type == "content_block_start"
        and event.content_block is not None
        and event.content_block.type == "tool_use"
    ):
        if coerce_content_to_string:
            warnings.warn("Received unexpected tool content block.")
        content_block = event.content_block.model_dump()
        content_block["index"] = event.index
        tool_call_chunk = {
            "index": event.index,
            "id": event.content_block.id,
            "name": event.content_block.name,
            "args": "",
        }
        message_chunk = AIMessageChunk(
            content=[content_block],
            tool_call_chunks=[tool_call_chunk],
        )
    elif event.type == "content_block_delta":
        if event.delta.type == "text_delta":
            if coerce_content_to_string:
                text = event.delta.text
                message_chunk = AIMessageChunk(content=text)
            else:
                content_block = event.delta.model_dump()
                content_block["index"] = event.index
                content_block["type"] = "text"
                message_chunk = AIMessageChunk(content=[content_block])
        elif event.delta.type in {"thinking_delta", "signature_delta"}:
            content_block = event.delta.model_dump()
            if "text" in content_block and content_block["text"] is None:
                content_block.pop("text")
            content_block["index"] = event.index
            content_block["type"] = "thinking"
            message_chunk = AIMessageChunk(content=[content_block])
        elif event.delta.type == "input_json_delta":
            content_block = event.delta.model_dump()
            content_block["index"] = event.index
            content_block["type"] = "tool_use"
            tool_call_chunk = {
                "index": event.index,
                "id": None,
                "name": None,
                "args": event.delta.partial_json,
            }
            message_chunk = AIMessageChunk(
                content=[content_block],
                tool_call_chunks=[tool_call_chunk],
            )
    elif event.type == "message_delta" and stream_usage:
        # Follow official langchain_anthropic pattern - NO cache tokens for delta
        # Only output tokens are provided in message_delta events
        usage_metadata = {
            "input_tokens": 0,
            "output_tokens": event.usage.output_tokens,
            "total_tokens": event.usage.output_tokens,
        }

        message_chunk = AIMessageChunk(
            content="",
            usage_metadata=usage_metadata,
            response_metadata={
                "stop_reason": event.delta.stop_reason,
                "stop_sequence": event.delta.stop_sequence,
            },
        )
    else:
        pass
    return message_chunk


def _tools_in_params(params: dict) -> bool:
    return "tools" in params or (
        "extra_body" in params and params["extra_body"].get("tools")
    )


def _thinking_in_params(params: dict) -> bool:
    return params.get("thinking", {}).get("type") == "enabled"


def _documents_in_params(params: dict) -> bool:
    for message in params.get("messages", []):
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "document"
                    and block.get("citations", {}).get("enabled")
                ):
                    return True
    return False
