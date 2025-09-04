#!/usr/bin/env python3

import re

def fix_line_length_violations():
    """Fix specific line length violations in chat_models.py"""
    file_path = "/home/daytona/langchain-google/libs/genai/langchain_google_genai/chat_models.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix specific violations identified by the linter
    
    # 1. Line 429: Comment about NonStandardContentBlock
    content = content.replace(
        "            # This is a Google-specific block that was wrapped in NonStandardContentBlock",
        "            # This is a Google-specific block that was wrapped in\n            # NonStandardContentBlock"
    )
    
    # 2. Line 442: Function docstring
    content = content.replace(
        '    """Convert legacy Google-specific content blocks to Parts (for backward compatibility)."""',
        '    """Convert legacy Google-specific content blocks to Parts (for backward\n    compatibility)."""'
    )
    
    # 3. Line 516: Function docstring
    content = content.replace(
        "    Supports both standard content blocks from langchain-core and legacy dict-based blocks",
        "    Supports both standard content blocks from langchain-core and legacy\n    dict-based blocks"
    )
    
    # 4. Line 537: Logger warning message
    content = content.replace(
        '                        f"Failed to convert standard content block, falling back to legacy: {e}"',
        '                        f"Failed to convert standard content block, "\n                        f"falling back to legacy: {e}"'
    )
    
    # 5. Line 653: Another function docstring (similar to line 516)
    # This might be a duplicate, but let's handle it carefully
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if i >= 650 and i <= 655:  # Around line 653
            if "Supports both standard content blocks from langchain-core and legacy dict-based blocks" in line:
                lines[i] = line.replace(
                    "Supports both standard content blocks from langchain-core and legacy dict-based blocks",
                    "Supports both standard content blocks from langchain-core and legacy\n    dict-based blocks"
                )
    
    content = '\n'.join(lines)
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed line length violations in chat_models.py")

if __name__ == "__main__":
    fix_line_length_violations()
