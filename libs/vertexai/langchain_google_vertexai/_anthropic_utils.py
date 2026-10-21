import base64
import re
import urllib.parse
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
                    content.append(_format_image_content_block(block, project))
                    continue

                if block["type"] in ("file", "text-plain"):
                    content.append(_format_document_content_block(block))
                    continue

                # Tool results may nest LangChain file/image blocks (e.g. a PDF
                # returned by a tool), which Anthropic rejects unconverted
                if block["type"] == "tool_result":
                    content.append(_clean_content_block(block))
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

                if block["type"] == "reasoning":
                    # LC_OUTPUT_VERSION=v1 standardizes thinking blocks as
                    # "reasoning" type. Convert back to Anthropic-native
                    # "thinking" format before sending to the API.
                    thinking_block: dict[str, Any] = {"type": "thinking"}
                    if "reasoning" in block:
                        thinking_block["thinking"] = block["reasoning"]
                    if signature := block.get("extras", {}).get("signature"):
                        thinking_block["signature"] = signature
                    if "cache_control" in block:
                        thinking_block["cache_control"] = block["cache_control"]
                    content.append(thinking_block)
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
        msg = "Message should be a str, list of str or list of dicts"  # type: ignore[unreachable]  # noqa: E501
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
    for message in merged_messages:
        if message.type == "system":
            if system_messages is not None:
                msg = "Received multiple non-consecutive system messages."
                raise ValueError(msg)
            fm = _format_message_anthropic(message, project)
            if fm:
                system_messages = fm
            continue

        fm = _format_message_anthropic(message, project)
        if not fm:
            continue
        formatted_messages.append(fm)

    # Anthropic treats a trailing assistant message as a "prefill" and rejects
    # requests whose final content ends with whitespace. Mirror langchain-anthropic
    # and rstrip only the last text block of the last assistant message.
    if formatted_messages and formatted_messages[-1]["role"] == "assistant":
        content = formatted_messages[-1]["content"]
        if isinstance(content, str):
            formatted_messages[-1]["content"] = content.rstrip()
        elif (
            isinstance(content, list)
            and content
            and isinstance(content[-1], dict)
            and content[-1].get("type") == "text"
        ):
            content[-1]["text"] = content[-1]["text"].rstrip()

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


def _format_image_content_block(block: dict, project: str | None = None) -> dict:
    """Convert a LangChain image content block to Anthropic wire format.

    LangChain image blocks use `{"type": "image", "base64": ..., "mime_type": ...}`
    but Anthropic expects `{"type": "image", "source": {"type": "base64", ...}}`.

    Raises:
        ValueError: If block has no recognized image data field.
    """
    if "source" in block:
        return block
    if "base64" in block:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block["mime_type"],
                "data": block["base64"],
            },
        }
    if "url" in block:
        url = block["url"]
        if url.startswith("data:"):
            return {
                "type": "image",
                "source": _format_image(url, project),
            }
        return {
            "type": "image",
            "source": {"type": "url", "url": url},
        }
    if "file_id" in block:
        return {
            "type": "image",
            "source": {"type": "file", "file_id": block["file_id"]},
        }
    # Backward compatibility for langchain < 1.X
    if "data" in block and block.get("source_type") == "base64":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block["mime_type"],
                "data": block["data"],
            },
        }
    if "id" in block and block.get("source_type") == "id":
        return {
            "type": "image",
            "source": {"type": "file", "file_id": block["id"]},
        }
    msg = (
        "Image content blocks must have either 'url', 'base64', "
        "'file_id', 'id' or 'data' field."
    )
    raise ValueError(msg)


def _text_document_source(block: dict) -> dict[str, Any]:
    """Build an Anthropic plain-text document source.

    Raises:
        ValueError: If the block has no inline `text`, since Anthropic text
            document sources cannot reference a URL, file ID, or base64 payload.
    """
    if "text" not in block:
        msg = (
            "Anthropic plain-text document sources require an inline 'text' field; "
            f"got keys {sorted(block)}."
        )
        raise ValueError(msg)
    return {
        "type": "text",
        "media_type": block.get("mime_type") or "text/plain",
        "data": block["text"],
    }


def _format_document_source(block: dict) -> dict[str, Any]:
    """Build the Anthropic document source for a LangChain file block.

    Follows `langchain_anthropic`, accepting both standard and legacy
    (`source_type`) block shapes, but validates sources more strictly: non-PDF
    base64 or URL files and empty base64 payloads are rejected locally rather
    than failing at the API.

    `url` and `file_id` sources are still converted for parity, although Claude
    on Vertex AI does not currently support URL or Files API input sources, so
    the API is expected to reject them.

    Raises:
        ValueError: If a `file` block has no supported data field, a base64 or
            URL `file` block is not a PDF, a base64 payload is empty, or a
            plain-text block has no inline `text`.
    """
    if block["type"] == "text-plain":
        return _text_document_source(block)
    if "url" in block:
        # Anthropic URL document sources only accept PDFs; `mime_type` is
        # optional on URL blocks, so only reject an explicit mismatch
        mime_type = block.get("mime_type")
        if mime_type and mime_type != "application/pdf":
            msg = (
                "Anthropic only supports URL file content blocks with mime_type "
                f"'application/pdf', got {mime_type!r}. Use an 'image' block for "
                "image URLs."
            )
            raise ValueError(msg)
        return {"type": "url", "url": block["url"]}
    if "base64" in block or block.get("source_type") == "base64":
        media_type = block.get("mime_type") or "application/pdf"
        # Anthropic base64 document sources only accept PDFs
        if media_type != "application/pdf":
            msg = (
                "Anthropic only supports base64 file content blocks with mime_type "
                f"'application/pdf', got {media_type!r}. Use a 'text-plain' block "
                "for plain text or an 'image' block for images."
            )
            raise ValueError(msg)
        data = block["base64"] if "base64" in block else block.get("data")
        if not data:
            msg = "Base64 file content block is missing its 'base64' or 'data' payload."
            raise ValueError(msg)
        return {"type": "base64", "media_type": media_type, "data": data}
    if block.get("source_type") == "text":
        return _text_document_source(block)
    if "file_id" in block:
        return {"type": "file", "file_id": block["file_id"]}
    if block.get("source_type") == "id" and "id" in block:
        return {"type": "file", "file_id": block["id"]}
    msg = (
        "Anthropic file content blocks require a 'url', 'base64', or 'file_id' "
        "key (or legacy source_type 'base64', 'text', or 'id'). Use a "
        "'text-plain' block for plain text."
    )
    raise ValueError(msg)


def _format_document_content_block(block: dict) -> dict:
    """Convert LangChain `file` and `text-plain` blocks to Anthropic documents.

    Anthropic has no `file` content type, so these blocks must be converted
    before reaching the API.

    Raises:
        ValueError: If the block cannot be converted to an Anthropic document
            source.
    """
    formatted_block: dict[str, Any] = {
        "type": "document",
        "source": _format_document_source(block),
    }
    for key in ("cache_control", "citations", "title", "context"):
        if key in block:
            formatted_block[key] = block[key]
        elif key in (block.get("extras") or {}):
            formatted_block[key] = block["extras"][key]
        elif key in (block.get("metadata") or {}):
            formatted_block[key] = block["metadata"][key]
    return formatted_block


def _clean_content_block(block: Any) -> Any:
    """Prepare a content block to be sent back to the Anthropic API.

    Anthropic's streaming API adds `index` and `partial_json` fields to content blocks
    during streaming. These fields must be removed before sending back to the API.
    LangChain `image`, `file`, and `text-plain` blocks are also converted to Anthropic
    wire format, including when nested inside `tool_result` content.

    Args:
        block: Content block (`dict`, `str`, or other type)

    Returns:
        Cleaned content block with streaming metadata removed

    Raises:
        ValueError: If a LangChain `image`, `file`, or `text-plain` block cannot be
            converted to Anthropic wire format.
    """
    if not isinstance(block, dict):
        return block

    # Convert LangChain image, file, and text-plain blocks to Anthropic wire format
    if block.get("type") == "image":
        return _format_image_content_block(block)
    if block.get("type") in ("file", "text-plain"):
        return _format_document_content_block(block)
    # Tool results may nest LangChain blocks that also need converting
    if block.get("type") == "tool_result" and "content" in block:
        block = {**block, "content": _clean_content(block["content"])}

    # Remove known streaming metadata fields
    # 'index' - added during streaming to track block position
    # 'partial_json' - added during streaming for incremental JSON parsing
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
            # Check if already in tool_result format (backward compatibility).
            # The `and curr.content` guard prevents `all()` from returning True
            # on an empty list, which would silently drop the tool_result (#1722).
            if (
                isinstance(curr.content, list)
                and curr.content
                and all(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in curr.content
                )
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
        if any(
            all(isinstance(m, c) for m in (curr, last))
            for c in (SystemMessage, HumanMessage)
        ):
            if isinstance(cast("BaseMessage", last).content, str):
                new_content: list = [
                    {"type": "text", "text": cast("BaseMessage", last).content}
                ]
            else:
                new_content = cast("list", cast("BaseMessage", last).content)
            if isinstance(curr.content, str):
                new_content.append({"type": "text", "text": curr.content})
            else:
                new_content.extend(curr.content)
            merged[-1] = curr.model_copy(update={"content": new_content})
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
    return params.get("thinking", {}).get("type") in ("enabled", "adaptive")


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
