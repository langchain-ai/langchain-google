#!/usr/bin/env python3
"""
Script to update message conversion functions to handle standard content blocks
"""

import re

def update_convert_tool_message_to_parts():
    """Update _convert_tool_message_to_parts function to handle standard content blocks"""
    
    file_path = "/home/daytona/langchain-google/libs/genai/langchain_google_genai/chat_models.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the docstring
    old_docstring = '    """Converts a tool or function message to a Google part."""'
    new_docstring = '''    """Converts a tool or function message to a Google part.
    
    Supports both standard content blocks from langchain-core and legacy dict-based blocks
    for backward compatibility during the transition period.
    """'''
    
    content = content.replace(old_docstring, new_docstring)
    
    # Update the block processing logic
    old_block_logic = '''        for block in message.content:
            if isinstance(block, dict) and (
                is_data_content_block(block) or _is_openai_image_block(block)
            ):
                media_blocks.append(block)
            else:
                other_blocks.append(block)'''
    
    new_block_logic = '''        for block in message.content:
            if isinstance(block, dict):
                # Check for standard content blocks first
                if _is_standard_content_block(block):
                    # Standard content blocks that represent media/visual content
                    if block.get("type") in ("image", "video", "audio", "file"):
                        media_blocks.append(block)
                    else:
                        other_blocks.append(block)
                # Legacy content block handling (backward compatibility)
                elif is_data_content_block(block) or _is_openai_image_block(block):
                    media_blocks.append(block)
                else:
                    other_blocks.append(block)
            else:
                other_blocks.append(block)'''
    
    content = content.replace(old_block_logic, new_block_logic)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Updated _convert_tool_message_to_parts function")

def find_other_message_conversion_functions():
    """Find other message conversion functions that might need updating"""
    
    file_path = "/home/daytona/langchain-google/libs/genai/langchain_google_genai/chat_models.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Look for functions that process message.content
    functions_with_content = []
    lines = content.split('\n')
    
    current_function = None
    for i, line in enumerate(lines):
        # Check for function definitions
        if line.strip().startswith('def ') and 'message' in line:
            current_function = line.strip()
        
        # Check for message.content usage
        if 'message.content' in line and current_function:
            functions_with_content.append((current_function, i+1, line.strip()))
    
    print("Functions that process message.content:")
    for func, line_num, line in functions_with_content:
        print(f"  {func} (line {line_num}): {line}")
    
    return functions_with_content

if __name__ == "__main__":
    print("Updating message conversion functions...")
    
    # First, find all functions that might need updating
    find_other_message_conversion_functions()
    
    # Update the main function
    update_convert_tool_message_to_parts()
    
    print("Message conversion functions updated successfully!")
