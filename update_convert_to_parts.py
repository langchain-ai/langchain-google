#!/usr/bin/env python3
"""Script to update _convert_to_parts() function to work with standard content blocks"""

import os

# Read the current file
file_path = "/home/daytona/langchain-google/libs/genai/langchain_google_genai/chat_models.py"
with open(file_path, 'r') as f:
    content = f.read()

# First, add the helper function after _is_lc_content_block
helper_function = '''

def _is_standard_content_block(part: Union[dict, Any]) -> bool:
    """Check if the part is a standard content block from langchain-core."""
    if not isinstance(part, dict):
        return False
    
    # Check if it's one of the standard content block types
    block_type = part.get("type")
    standard_types = {
        "text", "image", "video", "audio", "file", "text-plain",
        "tool_call", "tool_call_chunk", "code_interpreter_call", 
        "code_interpreter_result", "web_search_call", "web_search_result",
        "reasoning", "citation", "non_standard"
    }
    
    return block_type in standard_types


def _convert_standard_content_block_to_part(part: dict, image_loader) -> Part:
    """Convert a standard content block to a Google Part."""
    block_type = part.get("type")
    
    if block_type == "text":
        # Handle standard TextContentBlock
        text_content = part.get("text", "")
        # Check if this was originally a thinking block (preserved in extras)
        if part.get("extras", {}).get("thought") is True:
            return Part(text=text_content, thought=True)
        else:
            return Part(text=text_content)
    
    elif block_type == "image":
        # Handle standard ImageContentBlock
        if "base64" in part:
            bytes_ = base64.b64decode(part["base64"])
            inline_data = {"data": bytes_}
            if "mime_type" in part:
                inline_data["mime_type"] = part["mime_type"]
            return Part(inline_data=inline_data)
        elif "url" in part:
            return image_loader.load_part(part["url"])
        elif "file_id" in part:
            return Part(file_data=FileData(file_uri=part["file_id"]))
        else:
            raise ValueError(f"Image block must have base64, url, or file_id: {part}")
    
    elif block_type in ("video", "audio"):
        # Handle standard VideoContentBlock/AudioContentBlock
        media_part = Part()
        mime_type = part.get("mime_type", "")
        
        if "base64" in part:
            media_part.inline_data = Blob(data=part["base64"], mime_type=mime_type)
        elif "url" in part:
            media_part.file_data = FileData(file_uri=part["url"], mime_type=mime_type)
        elif "file_id" in part:
            media_part.file_data = FileData(file_uri=part["file_id"], mime_type=mime_type)
        else:
            raise ValueError(f"Media block must have base64, url, or file_id: {part}")
        
        # Handle video metadata from extras
        if block_type == "video" and "extras" in part and "video_metadata" in part["extras"]:
            metadata = VideoMetadata(part["extras"]["video_metadata"])
            media_part.video_metadata = metadata
        
        return media_part
    
    elif block_type == "file":
        # Handle standard FileContentBlock
        mime_type = part.get("mime_type", "application/octet-stream")
        
        if "base64" in part:
            return Part(inline_data=Blob(data=part["base64"], mime_type=mime_type))
        elif "url" in part:
            return Part(file_data=FileData(file_uri=part["url"], mime_type=mime_type))
        elif "file_id" in part:
            return Part(file_data=FileData(file_uri=part["file_id"], mime_type=mime_type))
        else:
            raise ValueError(f"File block must have base64, url, or file_id: {part}")
    
    elif block_type == "code_interpreter_call":
        # Handle standard CodeInterpreterCall
        code = part.get("code", "")
        language = part.get("extras", {}).get("language", "python")  # Default to python
        return Part(executable_code=ExecutableCode(language=language, code=code))
    
    elif block_type == "code_interpreter_result":
        # Handle standard CodeInterpreterResult
        output_list = part.get("output", [])
        if output_list and len(output_list) > 0:
            first_output = output_list[0]
            if isinstance(first_output, dict) and first_output.get("type") == "code_interpreter_output":
                result_text = first_output.get("stdout", "")
            else:
                result_text = str(first_output)
        else:
            result_text = ""
        
        outcome = part.get("extras", {}).get("outcome", 1)  # Default to success
        return Part(code_execution_result=CodeExecutionResult(output=result_text, outcome=outcome))
    
    elif block_type == "reasoning":
        # Handle standard ReasoningContentBlock
        reasoning_text = part.get("reasoning", "")
        return Part(text=reasoning_text, thought=True)
    
    elif block_type == "non_standard":
        # Handle NonStandardContentBlock - extract the original Google format
        original_block = part.get("value", {})
        if isinstance(original_block, dict) and "type" in original_block:
            # This is a Google-specific block that was wrapped in NonStandardContentBlock
            # Convert it using the legacy logic
            return _convert_legacy_google_block_to_part(original_block, image_loader)
        else:
            # Unknown format, convert to text
            return Part(text=str(original_block))
    
    else:
        # Unknown standard block type, convert to text
        return Part(text=str(part))


def _convert_legacy_google_block_to_part(part: dict, image_loader) -> Part:
    """Convert legacy Google-specific content blocks to Parts (for backward compatibility)."""
    if part["type"] == "media":
        if "mime_type" not in part:
            raise ValueError(f"Missing mime_type in media part: {part}")
        mime_type = part["mime_type"]
        media_part = Part()

        if "data" in part:
            media_part.inline_data = Blob(data=part["data"], mime_type=mime_type)
        elif "file_uri" in part:
            media_part.file_data = FileData(file_uri=part["file_uri"], mime_type=mime_type)
        else:
            raise ValueError(f"Media part must have either data or file_uri: {part}")
        
        if "video_metadata" in part:
            metadata = VideoMetadata(part["video_metadata"])
            media_part.video_metadata = metadata
        return media_part
    
    elif part["type"] == "executable_code":
        if "executable_code" not in part or "language" not in part:
            raise ValueError(
                "Executable code part must have 'code' and 'language' "
                f"keys, got {part}"
            )
        return Part(executable_code=ExecutableCode(
            language=part["language"], code=part["executable_code"]
        ))
    
    elif part["type"] == "code_execution_result":
        if "code_execution_result" not in part:
            raise ValueError(
                "Code execution result part must have "
                f"'code_execution_result', got {part}"
            )
        outcome = part.get("outcome", 1)  # Default to success if not specified
        return Part(code_execution_result=CodeExecutionResult(
            output=part["code_execution_result"], outcome=outcome
        ))
    
    elif part["type"] == "thinking":
        return Part(text=part["thinking"], thought=True)
    
    else:
        raise ValueError(f"Unrecognized legacy Google block type: {part['type']}")'''

# Insert the helper function after _is_lc_content_block function
insert_point = content.find("def _is_openai_image_block(block: dict) -> bool:")
if insert_point == -1:
    print("Could not find insertion point for helper function")
    exit(1)

content = content[:insert_point] + helper_function + "\n\n" + content[insert_point:]

# Now update the _convert_to_parts function
old_convert_function = '''def _convert_to_parts(
    raw_content: Union[str, Sequence[Union[str, dict]]],
) -> List[Part]:
    """Converts a list of LangChain messages into a Google parts."""
    parts = []
    content = [raw_content] if isinstance(raw_content, str) else raw_content
    image_loader = ImageBytesLoader()
    for part in content:
        if isinstance(part, str):
            parts.append(Part(text=part))
        elif isinstance(part, Mapping):
            if _is_lc_content_block(part):'''

new_convert_function = '''def _convert_to_parts(
    raw_content: Union[str, Sequence[Union[str, dict]]],
) -> List[Part]:
    """Converts a list of LangChain messages into a Google parts.
    
    Supports both standard content blocks from langchain-core and legacy dict-based blocks
    for backward compatibility during the transition period.
    """
    parts = []
    content = [raw_content] if isinstance(raw_content, str) else raw_content
    image_loader = ImageBytesLoader()
    for part in content:
        if isinstance(part, str):
            parts.append(Part(text=part))
        elif isinstance(part, Mapping):
            # Check if it's a standard content block first
            if _is_standard_content_block(part):
                try:
                    converted_part = _convert_standard_content_block_to_part(part, image_loader)
                    parts.append(converted_part)
                    continue
                except Exception as e:
                    # If standard conversion fails, fall back to legacy handling
                    logger.warning(f"Failed to convert standard content block, falling back to legacy: {e}")
            
            # Legacy content block handling (backward compatibility)
            if _is_lc_content_block(part):'''

content = content.replace(old_convert_function, new_convert_function)

# Write the modified content back to the file
with open(file_path, 'w') as f:
    f.write(content)

print("Successfully updated _convert_to_parts() function to handle standard content blocks")
