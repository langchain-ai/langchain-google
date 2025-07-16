"""
Compatibility module for converting between AIMessage output formats.

- "v0": The original format, with a mix of strings and dicts in `content`, and tool
    calls in `additional_kwargs` and `tool_calls`.
- "v1": LangChain's cross-provider standardized format using a list of typed
    `ContentBlock` dicts.

For backwards compatibility, this module provides functions to convert
between these formats.
"""

import json
import uuid
from typing import Dict, List, Union

from langchain_core.messages import (
    AIMessage,
    ToolCall,
)
from langchain_core.messages.content_blocks import (
    make_non_standard_content_block,
    make_reasoning_block,
    make_text_block,
    make_tool_call_block,
)


def _convert_v0_to_v1(message: AIMessage) -> AIMessage:
    """Converts a v0 AIMessage to one with the v1 ContentBlock structure (untyped).

    Processes the `content` field, `tool_calls`, `invalid_tool_calls`,
    and legacy `function_call` from `additional_kwargs`, converting them all into
    a unified list of ContentBlock dictionaries in the new `content` field.

    These dictionaries can then be cast to specific ContentBlock types using the
    `beta_content` property of AIMessage.
    """
    new_content: list[dict] = []
    content = message.content

    # Process the original `content` field (strings and dicts)
    # --------------------------------------------------------------------------
    if isinstance(content, str):
        # Avoid adding empty text blocks
        if content:
            new_content.append(make_text_block(content))
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                if item:
                    new_content.append(make_text_block(item))
            elif isinstance(item, dict):
                block_type = item.get("type")
                # Map known dictionary types to specific ContentBlocks when possible
                if block_type == "thinking":
                    # TODO: this can result in a ContentBlock with an empty reasoning
                    # if the "thinking" key is not present -- need clarity on this
                    new_content.append(
                        make_reasoning_block(reasoning=item.get("thinking", ""))
                    )
                # For other custom types, wrap them in a NonStandardContentBlock
                # to preserve the data within the v1 structure.
                elif block_type in (
                    "executable_code",
                    "code_execution_result",
                    "image_url",
                ):
                    new_content.append(make_non_standard_content_block(value=item))
                else:
                    # Fallback for any other unexpected dicts
                    new_content.append(make_non_standard_content_block(value=item))

    # Process tool and function calls, converting them to ToolCallContentBlocks
    # --------------------------------------------------------------------------
    # Handle legacy `function_call` from older message formats
    if legacy_fc := message.additional_kwargs.get("function_call"):
        try:
            fc_args = json.loads(legacy_fc.get("arguments", "{}"))
        except json.JSONDecodeError:
            fc_args = {
                "error": "Failed to decode arguments",
                "raw_arguments": legacy_fc.get("arguments"),
            }

        new_content.append(
            make_tool_call_block(
                # name=legacy_fc.get("name", ""),
                args=fc_args,
                id=f"tool_call_{uuid.uuid4()}",  # Legacy calls lack ID,
            )  # so generate one
        )

    # Handle standard `tool_calls`
    for tool_call in message.tool_calls:
        new_content.append(
            make_tool_call_block(
                name=tool_call.name,
                args=tool_call.args,
                id=tool_call.id or f"tool_call_{uuid.uuid4()}",
            )
        )

    # Handle `invalid_tool_calls` by wrapping them to preserve the error info
    for invalid_tool_call in message.invalid_tool_calls:
        new_content.append(
            make_non_standard_content_block(
                value={
                    "type": "invalid_tool_call",
                    "name": invalid_tool_call.get("name"),
                    "args": invalid_tool_call.get("args"),
                    "id": invalid_tool_call.get("id"),
                    "error": invalid_tool_call.get("error"),
                },
            )
        )

    # 3. Create the new AIMessage, clearing out the old fields
    # --------------------------------------------------------
    new_additional_kwargs = message.additional_kwargs.copy()
    new_additional_kwargs.pop("function_call", None)  # Clean up the legacy field

    return AIMessage(
        content=new_content,
        # In v1, these are empty because their content is now in the `content` list
        tool_calls=[],
        invalid_tool_calls=[],
        # Pass through remaining data
        additional_kwargs=new_additional_kwargs,
        response_metadata=message.response_metadata,
        id=message.id,
        usage_metadata=message.usage_metadata,
    )


def _convert_v1_to_v0(message: AIMessage) -> AIMessage:
    """
    Converts a v1 AIMessage back to the v0 format.
    (Required for sending v1 messages as input to the model)
    """
    v0_content: Union[str, List[Union[str, Dict]]] = []
    v0_tool_calls: List[ToolCall] = []
    v0_invalid_tool_calls: List[Dict] = []
    v0_additional_kwargs = message.additional_kwargs.copy()

    text_parts = []

    for block in message.content:
        if block["type"] == "text":
            # Just append the text part for v0
            text_parts.append(block["text"])
        elif block["type"] == "tool_call":
            # Find the full tool call from the message's list and rebuild
            # additional_kwargs["function_call"] for v0.
            tc_id = block["tool_call"]["id"]
            found = False
            for tc in message.tool_calls:
                if tc.id == tc_id:
                    v0_tool_calls.append(tc)
                    v0_additional_kwargs["function_call"] = {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args),
                    }
                    found = True
                    break
            if not found:
                for itc in message.invalid_tool_calls:
                    # InvalidToolCall does not have an 'id' attribute directly,
                    # it's part of the dictionary representation.
                    # Assuming itc is a dict or has a dict-like structure for 'id'
                    if isinstance(itc, dict) and itc.get("id") == tc_id:
                        v0_invalid_tool_calls.append(itc)
                        found = True
                        break
                    # If itc is an InvalidToolCall object, access its attributes
                    elif hasattr(itc, "id") and getattr(itc, "id") == tc_id:
                        v0_invalid_tool_calls.append(itc.dict())  # Convert to dict
                        found = True
                        break
        elif block["type"] == "non_standard":
            v0_content.append(block["value"])

    # Consolidate text parts into a single string if that's the only content
    if text_parts and not v0_content:
        v0_content = "\n".join(text_parts)
    else:
        # Ensure v0_content is a list before extending
        if isinstance(v0_content, str):
            v0_content = [v0_content]
        if isinstance(v0_content, list):  # Add this check to satisfy mypy
            v0_content.extend(text_parts)
        else:
            # Fallback for unexpected types, should not happen with current typing
            v0_content = text_parts

    return AIMessage(
        content=v0_content,
        tool_calls=v0_tool_calls,
        invalid_tool_calls=v0_invalid_tool_calls,
        additional_kwargs=v0_additional_kwargs,
        response_metadata=message.response_metadata,
        id=message.id,
    )
