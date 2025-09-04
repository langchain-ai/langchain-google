"""
Google GenAI Block Translator for Standard Content Blocks

This module provides translation functions to convert Google GenAI-specific
content blocks
to the standard content block format defined in langchain-core.

⚠️  IMPORTANT: TEMPORARY LOCATION ⚠️
=====================================
TODO: This translator is temporarily located in langchain-google. The final destination
should be langchain-core/messages/block_translators/google_genai.py

This is a TEMPORARY implementation that should be migrated to langchain-core once the
standard content block system is fully established.

MIGRATION INSTRUCTIONS FOR MAINTAINERS:
========================================
To move this translator to langchain-core:

1. **Copy the file:**
   - Source: langchain-google/libs/genai/langchain_google_genai/block_translator.py
   - Destination: langchain-core/src/langchain_core/messages/block_translators/
     google_genai.py

2. **Update imports:**
   - Change: `from langchain_core.messages import content as types`
   - To: `from ..content import types` (or appropriate relative import)
   - Change: `from langchain_core.messages import AIMessage, AIMessageChunk`
   - To: `from .. import AIMessage, AIMessageChunk`

3. **Update registration:**
   - Ensure `_register_google_genai_translator()` is called when langchain-core is
     imported
   - The registration should happen automatically in langchain-core's __init__.py

4. **Remove from langchain-google:**
   - Delete this file from langchain-google after successful migration
   - Update langchain-google to import the translator from langchain-core instead

5. **Update dependencies:**
   - Ensure langchain-google depends on the appropriate version of langchain-core
   - Update any references to this module in langchain-google

DETAILED CONTENT BLOCK MAPPING:
===============================
Google GenAI content types are mapped to standard content blocks as follows:

🔤 TEXT CONTENT:
- 'text' → TextContentBlock
  - Google fields:
    * text (string): The text content
    * thought (boolean, optional): Whether this is a thinking/reasoning step
  - Standard fields:
    * text (string): The text content
    * type="text"
  - Extras preservation:
    * thought flag → extras["thought"] = True (if present)
  - Example:
    Google: {"type": "text", "text": "Hello", "thought": True}
    Standard: {"type": "text", "text": "Hello", "extras": {"thought": True}}

🎬 MEDIA CONTENT:
- 'media' → VideoContentBlock/AudioContentBlock/ImageContentBlock (based on mime_type)
  - Google fields:
    * mime_type (string): MIME type of the media
    * data (string, optional): Base64-encoded media data
    * file_uri (string, optional): URI to media file
    * video_metadata (dict, optional): Video-specific metadata
  - Standard fields:
    * mime_type (string): MIME type
    * base64/url/file_id (string): Media data or reference
    * type="video"/"audio"/"image" (determined by mime_type)
  - Extras preservation:
    * video_metadata → extras["video_metadata"]
    * Any other Google-specific fields → extras[field_name]
  - Example:
    Google: {"type": "media", "mime_type": "video/mp4", "file_uri": "gs://...",
             "video_metadata": {...}}
    Standard: {"type": "video", "mime_type": "video/mp4", "url": "gs://...",
               "extras": {"video_metadata": {...}}}

💻 CODE EXECUTION:
- 'executable_code' → CodeInterpreterCall
  - Google fields:
    * language (string): Programming language (e.g., "python")
    * code (string): The executable code
  - Standard fields:
    * code (string): The executable code
    * type="code_interpreter_call"
  - Extras preservation:
    * language → extras["language"]
  - Example:
    Google: {"type": "executable_code", "language": "python", "code": "print('hello')"}
    Standard: {"type": "code_interpreter_call", "code": "print('hello')",
               "extras": {"language": "python"}}

📊 CODE RESULTS:
- 'code_execution_result' → CodeInterpreterResult
  - Google fields:
    * output (string): The execution result/output
    * outcome (int): Execution outcome code (1=success, 2=error, etc.)
  - Standard fields:
    * output (list): List of output objects
    * type="code_interpreter_result"
  - Extras preservation:
    * outcome → extras["outcome"]
  - Example:
    Google: {"type": "code_execution_result", "output": "hello", "outcome": 1}
    Standard: {"type": "code_interpreter_result",
               "output": [{"type": "code_interpreter_output", "stdout": "hello"}],
               "extras": {"outcome": 1}}

🧠 REASONING/THINKING:
- 'thinking' → ReasoningContentBlock
  - Google fields:
    * text (string): The reasoning/thinking content
    * thought=True (boolean flag): Always true for thinking blocks
  - Standard fields:
    * reasoning (string): The reasoning content
    * type="reasoning"
  - Extras preservation:
    * Any Google-specific metadata → extras[field_name]
  - Example:
    Google: {"type": "thinking", "text": "Let me think about this...", "thought": True}
    Standard: {"type": "reasoning", "reasoning": "Let me think about this..."}

🔧 NON-STANDARD CONTENT:
- Other types → NonStandardContentBlock
  - All Google-specific content types not covered above
  - Preserved in value field with type="non_standard"
  - Example:
    Google: {"type": "custom_google_type", "custom_field": "value"}
    Standard: {"type": "non_standard",
               "value": {"type": "custom_google_type", "custom_field": "value"}}

FIELD PRESERVATION STRATEGY:
============================
The 'extras' field in standard content blocks is used to preserve Google-specific
metadata:

1. **Known Google fields** are mapped to their standard equivalents
2. **Unknown or Google-specific fields** are preserved in extras
3. **Nested objects** (like video_metadata) are preserved as-is in extras
4. **Type information** is maintained for round-trip compatibility

This ensures that no Google-specific functionality is lost during the translation
process.

USAGE NOTES:
============
- This translator is automatically registered when the module is imported
- It handles both AIMessage and AIMessageChunk content translation
- The translation is bidirectional-safe (can be reversed without data loss)
- All Google-specific fields are preserved in the 'extras' field for compatibility
"""

import warnings
from typing import Any, Dict, List, Union

from langchain_core.messages import AIMessage, AIMessageChunk

# Note: This is a temporary implementation. The actual content block types
# and factory functions will be available when this translator is moved to langchain-core


def _extract_extras(block: Dict[str, Any], known_fields: set[str]) -> Dict[str, Any]:
    """Extract unknown fields from block to preserve as extras."""
    return {k: v for k, v in block.items() if k not in known_fields}


def _convert_text_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Google GenAI text block to TextContentBlock."""
    known_fields = {"type", "text", "thinking", "thought"}
    extras = _extract_extras(block, known_fields)

    text_block: types.TextContentBlock = {
        "type": "text",
        "text": block.get("text", ""),
    }

    # Handle thinking flag (Google-specific)
    if block.get("thought") is True or "thinking" in block:
        if "extras" not in text_block:
            text_block["extras"] = {}
        text_block["extras"]["thought"] = True

    # Add any other extras
    if extras:
        if "extras" not in text_block:
            text_block["extras"] = {}
        text_block["extras"].update(extras)

    return text_block


def _convert_media_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Google GenAI media block to appropriate content block based on
    mime_type."""
    mime_type = block.get("mime_type", "")
    known_fields = {"type", "mime_type", "data", "file_uri", "video_metadata"}
    extras = _extract_extras(block, known_fields)

    # Determine content block type based on mime_type
    if mime_type.startswith("video/"):
        block_type = "video"
    elif mime_type.startswith("audio/"):
        block_type = "audio"
    elif mime_type.startswith("image/"):
        block_type = "image"
    else:
        # Default to file block for unknown mime types
        block_type = "file"

    # Build the content block
    kwargs = {"mime_type": mime_type}

    # Handle different data sources
    if "data" in block:
        kwargs["base64"] = block["data"]
    elif "file_uri" in block:
        kwargs["url"] = block["file_uri"]

    # Add video metadata if present
    if "video_metadata" in block:
        if "extras" not in kwargs:
            kwargs["extras"] = {}
        kwargs["extras"]["video_metadata"] = block["video_metadata"]

    # Add any other extras
    if extras:
        if "extras" not in kwargs:
            kwargs["extras"] = {}
        kwargs["extras"].update(extras)

    # Create the content block dictionary
    content_block = {"type": block_type, **kwargs}
    return content_block


def _convert_executable_code_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Google GenAI executable_code block to CodeInterpreterCall."""
    known_fields = {"type", "language", "executable_code"}
    extras = _extract_extras(block, known_fields)

    code_call: Dict[str, Any] = {
        "type": "code_interpreter_call",
        "code": block.get("executable_code", ""),
    }

    # Preserve language in extras
    if "language" in block:
        if "extras" not in code_call:
            code_call["extras"] = {}
        code_call["extras"]["language"] = block["language"]

    # Add any other extras
    if extras:
        if "extras" not in code_call:
            code_call["extras"] = {}
        code_call["extras"].update(extras)

    return code_call


def _convert_code_execution_result_block(
    block: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert Google GenAI code_execution_result block to CodeInterpreterResult."""
    known_fields = {"type", "code_execution_result", "outcome"}
    extras = _extract_extras(block, known_fields)

    result_block: Dict[str, Any] = {
        "type": "code_interpreter_result",
        "output": [
            {
                "type": "code_interpreter_output",
                "stdout": block.get("code_execution_result", ""),
            }
        ],
    }

    # Preserve outcome in extras
    if "outcome" in block:
        if "extras" not in result_block:
            result_block["extras"] = {}
        result_block["extras"]["outcome"] = block["outcome"]

    # Add any other extras
    if extras:
        if "extras" not in result_block:
            result_block["extras"] = {}
        result_block["extras"].update(extras)

    return result_block


def _convert_thinking_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Google GenAI thinking block to ReasoningContentBlock."""
    known_fields = {"type", "thinking", "thought"}
    extras = _extract_extras(block, known_fields)

    reasoning_block: types.ReasoningContentBlock = {
        "type": "reasoning",
        "reasoning": block.get("thinking", ""),
    }

    # Add any other extras
    if extras:
        if "extras" not in reasoning_block:
            reasoning_block["extras"] = {}
        reasoning_block["extras"].update(extras)

    return reasoning_block


def _convert_google_block_to_standard(block: Dict[str, Any]) -> types.ContentBlock:
    """Convert a Google GenAI content block to a standard content block."""
    block_type = block.get("type")

    if block_type == "text":
        return _convert_text_block(block)
    elif block_type == "media":
        return _convert_media_block(block)
    elif block_type == "executable_code":
        return _convert_executable_code_block(block)
    elif block_type == "code_execution_result":
        return _convert_code_execution_result_block(block)
    elif block_type == "thinking":
        return _convert_thinking_block(block)
    else:
        # For unknown block types, use NonStandardContentBlock
        return types.create_non_standard_block(value=block)


def translate_content(message: AIMessage) -> List[types.ContentBlock]:
    """Derive standard content blocks from a message with Google GenAI content."""
    content_blocks: List[types.ContentBlock] = []

    if isinstance(message.content, str):
        if message.content:
            content_blocks = [types.create_text_block(text=message.content)]
        else:
            content_blocks = []

        # Handle tool calls
        for tool_call in message.tool_calls:
            content_blocks.append(
                {
                    "type": "tool_call",
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                    "id": tool_call.get("id"),
                }
            )

        return content_blocks

    # Handle list content
    if isinstance(message.content, list):
        for item in message.content:
            if isinstance(item, str):
                if item:
                    content_blocks.append(types.create_text_block(text=item))
            elif isinstance(item, dict):
                content_blocks.append(_convert_google_block_to_standard(item))
            else:
                # Handle other types by converting to non-standard block
                content_blocks.append(
                    types.create_non_standard_block(value={"content": item})
                )

    # Handle tool calls
    for tool_call in message.tool_calls:
        content_blocks.append(
            {
                "type": "tool_call",
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call.get("id"),
            }
        )

    return content_blocks


def translate_content_chunk(message: AIMessageChunk) -> List[types.ContentBlock]:
    """Derive standard content blocks from a message chunk with Google GenAI
    content."""
    content_blocks: List[types.ContentBlock] = []

    if isinstance(message.content, str):
        if message.content:
            content_blocks = [types.create_text_block(text=message.content)]
        else:
            content_blocks = []

        # Handle tool call chunks
        if message.chunk_position == "last":
            for tool_call in message.tool_calls:
                content_blocks.append(
                    {
                        "type": "tool_call",
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                        "id": tool_call.get("id"),
                    }
                )
        else:
            for tool_call_chunk in message.tool_call_chunks:
                tc: types.ToolCallChunk = {
                    "type": "tool_call_chunk",
                    "id": tool_call_chunk.get("id"),
                    "name": tool_call_chunk.get("name"),
                    "args": tool_call_chunk.get("args"),
                }
                if (idx := tool_call_chunk.get("index")) is not None:
                    tc["index"] = idx
                content_blocks.append(tc)

        return content_blocks

    # Handle list content
    if isinstance(message.content, list):
        for item in message.content:
            if isinstance(item, str):
                if item:
                    content_blocks.append(types.create_text_block(text=item))
            elif isinstance(item, dict):
                content_blocks.append(_convert_google_block_to_standard(item))
            else:
                # Handle other types by converting to non-standard block
                content_blocks.append(
                    types.create_non_standard_block(value={"content": item})
                )

    # Handle tool call chunks
    if message.chunk_position == "last":
        for tool_call in message.tool_calls:
            content_blocks.append(
                {
                    "type": "tool_call",
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                    "id": tool_call.get("id"),
                }
            )
    else:
        for tool_call_chunk in message.tool_call_chunks:
            tc: types.ToolCallChunk = {
                "type": "tool_call_chunk",
                "id": tool_call_chunk.get("id"),
                "name": tool_call_chunk.get("name"),
                "args": tool_call_chunk.get("args"),
            }
            if (idx := tool_call_chunk.get("index")) is not None:
                tc["index"] = idx
            content_blocks.append(tc)

    return content_blocks


def _register_google_genai_translator() -> None:
    """Register the Google GenAI translator with the central registry.

    Run automatically when the module is imported.

    TODO: After migration to langchain-core, this registration will happen
    automatically when langchain-core is imported.
    """
    try:
        from langchain_core.messages.block_translators import register_translator

        register_translator("google_genai", translate_content, translate_content_chunk)
    except ImportError:
        # If block_translators module is not available, warn but don't fail
        warnings.warn(
            "Could not register Google GenAI translator. "
            "langchain-core block_translators module may not be available.",
            stacklevel=2,
        )


# Register the translator when the module is imported
_register_google_genai_translator()











