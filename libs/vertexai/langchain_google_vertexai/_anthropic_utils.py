import re
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

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


def _format_image(image_url: str) -> Dict:
    """Formats a message image to a dict for anthropic api."""
    regex = r"^data:(?P<media_type>image/.+);base64,(?P<data>.+)$"
    match = re.match(regex, image_url)
    if match is None:
        raise ValueError(
            "Anthropic only supports base64-encoded images currently."
            " Example: data:image/png;base64,'/9j/4AAQSk'..."
        )
    return {
        "type": "base64",
        "media_type": match.group("media_type"),
        "data": match.group("data"),
    }


def _get_cache_control(message: BaseMessage) -> Optional[Dict[str, Any]]:
    """Extract cache control from message's additional_kwargs or content block."""
    return (
        message.additional_kwargs.get("cache_control")
        if isinstance(message.additional_kwargs, dict)
        else None
    )


def _format_text_content(text: str) -> Dict[str, Union[str, Dict[str, Any]]]:
    """Format text content."""
    content: Dict[str, Union[str, Dict[str, Any]]] = {"type": "text", "text": text}
    return content


def _format_message_anthropic(message: Union[HumanMessage, AIMessage, SystemMessage]):
    """Format a message for Anthropic API.

    Args:
        message: The message to format. Can be HumanMessage, AIMessage, or SystemMessage.

    Returns:
        A dictionary with the formatted message, or None if the message is empty.
    """  # noqa: E501
    content: List[Dict[str, Any]] = []

    if isinstance(message.content, str):
        if not message.content.strip():
            return None
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
                    raise ValueError("Dict content block must have a type key")

                new_block = {}

                for copy_attr in ["type", "cache_control"]:
                    if copy_attr in block:
                        new_block[copy_attr] = block[copy_attr]

                if block["type"] == "text":
                    text: str = block.get("text", "")
                    # Only add non-empty strings for now as empty ones are not
                    # accepted.
                    # https://github.com/anthropics/anthropic-sdk-python/issues/461
                    if text.strip():
                        new_block["text"] = text
                        content.append(new_block)
                    continue

                if block["type"] == "image_url":
                    # convert format
                    source = _format_image(block["image_url"]["url"])
                    content.append({"type": "image", "source": source})
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
        raise ValueError("Message should be a str, list of str or list of dicts")

    if isinstance(message, AIMessage) and message.tool_calls:
        for tc in message.tool_calls:
            tu = cast(Dict[str, Any], _lc_tool_call_to_anthropic_tool_use_block(tc))
            content.append(tu)

    if message.type == "system":
        return content
    else:
        return {"role": _message_type_lookups[message.type], "content": content}


def _format_messages_anthropic(
    messages: List[BaseMessage],
) -> Tuple[Optional[Dict[str, Any]], List[Dict]]:
    """Formats messages for anthropic."""
    system_messages: Optional[Dict[str, Any]] = None
    formatted_messages: List[Dict] = []

    merged_messages = _merge_messages(messages)
    for i, message in enumerate(merged_messages):
        if message.type == "system":
            if i != 0:
                raise ValueError("System message must be at beginning of message list.")
            fm = _format_message_anthropic(message)
            if fm:
                system_messages = fm
            continue

        fm = _format_message_anthropic(message)
        if not fm:
            continue
        formatted_messages.append(fm)

    return system_messages, formatted_messages


class AnthropicTool(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]


def convert_to_anthropic_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> AnthropicTool:
    # already in Anthropic tool format
    if isinstance(tool, dict) and all(
        k in tool for k in ("name", "description", "input_schema")
    ):
        return AnthropicTool(tool)  # type: ignore
    else:
        formatted = convert_to_openai_tool(tool)["function"]
        return AnthropicTool(
            name=formatted["name"],
            description=formatted["description"],
            input_schema=formatted["parameters"],
        )


def _merge_messages(
    messages: Sequence[BaseMessage],
) -> List[Union[SystemMessage, AIMessage, HumanMessage]]:
    """Merge runs of human/tool messages into single human messages with content blocks."""  # noqa: E501
    merged: list = []
    for curr in messages:
        curr = curr.model_copy(deep=True)
        if isinstance(curr, ToolMessage):
            if isinstance(curr.content, list) and all(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in curr.content
            ):
                curr = HumanMessage(curr.content)
            else:
                curr = HumanMessage(
                    [
                        {
                            "type": "tool_result",
                            "content": curr.content,
                            "tool_use_id": curr.tool_call_id,
                        }
                    ]
                )
        last = merged[-1] if merged else None
        if isinstance(last, HumanMessage) and isinstance(curr, HumanMessage):
            if isinstance(last.content, str):
                new_content: List = [{"type": "text", "text": last.content}]
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
        id=cast(str, tool_call["id"]),
    )


def _make_message_chunk_from_anthropic_event(
    event: "RawMessageStreamEvent",
    *,
    stream_usage: bool = True,
    coerce_content_to_string: bool,
) -> Optional[AIMessageChunk]:
    """Convert Anthropic event to AIMessageChunk.
    Note that not all events will result in a message chunk. In these cases
    we return None.
    """
    message_chunk: Optional[AIMessageChunk] = None
    # See https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/streaming/_messages.py  # noqa: E501
    if event.type == "message_start" and stream_usage:
        input_tokens = event.message.usage.input_tokens
        message_chunk = AIMessageChunk(
            content="" if coerce_content_to_string else [],
            usage_metadata=UsageMetadata(
                input_tokens=input_tokens,
                output_tokens=0,
                total_tokens=input_tokens,
            ),
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
            tool_call_chunks=[tool_call_chunk],  # type: ignore
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
                tool_call_chunks=[tool_call_chunk],  # type: ignore
            )
    elif event.type == "message_delta" and stream_usage:
        output_tokens = event.usage.output_tokens
        message_chunk = AIMessageChunk(
            content="",
            usage_metadata=UsageMetadata(
                input_tokens=0,
                output_tokens=output_tokens,
                total_tokens=output_tokens,
            ),
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
