"""
Google GenAI Block Translator for Standard Content Blocks

This module provides translation functions to convert Google GenAI-specific content blocks
to the standard content block format defined in langchain-core.

TODO: This translator is temporarily located in langchain-google. The final destination
should be langchain-core/messages/block_translators/google_genai.py

MIGRATION INSTRUCTIONS FOR MAINTAINERS:
========================================
To move this translator to langchain-core:

1. Copy this entire file to: langchain-core/messages/block_translators/google_genai.py
2. Update the imports to use relative imports within langchain-core
3. Ensure the registration function is called when the module is imported
4. Remove this file from langchain-google after migration is complete

CONTENT BLOCK MAPPING:
=====================
Google GenAI content types are mapped to standard content blocks as follows:

- 'text' → TextContentBlock
  - Google fields: text, thinking (boolean flag)
  - Standard fields: text, type="text"
  - Extras: thinking flag preserved in extras["thought"] if present

- 'media' → VideoContentBlock/AudioContentBlock/ImageContentBlock (based on mime_type)
  - Google fields: mime_type, data, file_uri, video_metadata
  - Standard fields: mime_type, base64/url/file_id, type="video"/"audio"/"image"
  - Extras: video_metadata and other Google-specific fields

- 'executable_code' → CodeInterpreterCall
  - Google fields: language, executable_code
  - Standard fields: code, type="code_interpreter_call"
  - Extras: language preserved in extras["language"]

- 'code_execution_result' → CodeInterpreterResult
  - Google fields: code_execution_result, outcome
  - Standard fields: output, type="code_interpreter_result"
  - Extras: outcome preserved in extras["outcome"]

- 'thinking' → ReasoningContentBlock
  - Google fields: thinking (text content), thought=True flag
  - Standard fields: reasoning, type="reasoning"
  - Extras: Google-specific metadata preserved

- Other types → NonStandardContentBlock
  - All Google-specific content types not covered above
  - Preserved in value field with type="non_standard"
"""

from typing import Any, Dict, List, Optional, Union, cast
import warnings

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types


def _extract_extras(block: Dict[str, Any], known_fields: set[str]) -> Dict[str, Any]:
    """Extract unknown fields from block to preserve as extras."""
    return {k: v for k, v in block.items() if k not in known_fields}


def _convert_text_block(block: Dict[str, Any]) -> types.ContentBlock:
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


def _convert_media_block(block: Dict[str, Any]) -> types.ContentBlock:
    """Convert Google GenAI media block to appropriate content block based on mime_type."""
    mime_type = block.get("mime_type", "")
    known_fields = {"type", "mime_type", "data", "file_uri", "video_metadata"}
    extras = _extract_extras(block, known_fields)
    
    # Determine content block type based on mime_type
    if mime_type.startswith("video/"):
        content_type = "video"
        create_func = types.create_video_block
    elif mime_type.startswith("audio/"):
        content_type = "audio"
        create_func = types.create_audio_block
    elif mime_type.startswith("image/"):
        content_type = "image"
        create_func = types.create_image_block
    else:
        # Default to file block for unknown mime types
        content_type = "file"
        create_func = types.create_file_block
    
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
    
    return create_func(**kwargs)


def _convert_executable_code_block(block: Dict[str, Any]) -> types.CodeInterpreterCall:
    """Convert Google GenAI executable_code block to CodeInterpreterCall."""
    known_fields = {"type", "language", "executable_code"}
    extras = _extract_extras(block, known_fields)
    
    code_call: types.CodeInterpreterCall = {
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


def _convert_code_execution_result_block(block: Dict[str, Any]) -> types.CodeInterpreterResult:
    """Convert Google GenAI code_execution_result block to CodeInterpreterResult."""
    known_fields = {"type", "code_execution_result", "outcome"}
    extras = _extract_extras(block, known_fields)
    
    result_block: types.CodeInterpreterResult = {
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


def _convert_thinking_block(block: Dict[str, Any]) -> types.ReasoningContentBlock:
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
            content_blocks.append({
                "type": "tool_call",
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call.get("id"),
            })
        
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
                content_blocks.append(types.create_non_standard_block(value={"content": item}))
    
    # Handle tool calls
    for tool_call in message.tool_calls:
        content_blocks.append({
            "type": "tool_call",
            "name": tool_call["name"],
            "args": tool_call["args"],
            "id": tool_call.get("id"),
        })
    
    return content_blocks


def translate_content_chunk(message: AIMessageChunk) -> List[types.ContentBlock]:
    """Derive standard content blocks from a message chunk with Google GenAI content."""
    content_blocks: List[types.ContentBlock] = []
    
    if isinstance(message.content, str):
        if message.content:
            content_blocks = [types.create_text_block(text=message.content)]
        else:
            content_blocks = []
        
        # Handle tool call chunks
        if message.chunk_position == "last":
            for tool_call in message.tool_calls:
                content_blocks.append({
                    "type": "tool_call",
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                    "id": tool_call.get("id"),
                })
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
                content_blocks.append(types.create_non_standard_block(value={"content": item}))
    
    # Handle tool call chunks
    if message.chunk_position == "last":
        for tool_call in message.tool_calls:
            content_blocks.append({
                "type": "tool_call",
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call.get("id"),
            })
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
            stacklevel=2
        )


# Register the translator when the module is imported
_register_google_genai_translator()
