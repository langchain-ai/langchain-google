#!/usr/bin/env python3

import re

def fix_block_translator_types():
    """Fix type annotation issues in block_translator.py"""
    file_path = "/home/daytona/langchain-google/libs/genai/langchain_google_genai/block_translator.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix all remaining type issues
    
    # Replace factory function calls with direct dictionary creation
    content = content.replace(
        'content_blocks = [types.create_text_block(text=message.content)]',
        'content_blocks = [{"type": "text", "text": message.content}]'
    )
    
    # Fix tool call chunk creation
    content = re.sub(
        r'tc: types\.ToolCallChunk = \{',
        'tc: Dict[str, Any] = {',
        content
    )
    
    # Fix text block creation calls
    content = content.replace(
        'types.create_text_block(text=message.content)',
        '{"type": "text", "text": message.content}'
    )
    
    # Fix other factory function calls
    content = content.replace(
        'types.create_text_block(',
        '_create_text_block('
    )
    
    # Add helper function for text block creation
    helper_functions = '''

def _create_text_block(text: str) -> Dict[str, Any]:
    """Helper function to create text blocks."""
    return {"type": "text", "text": text}

'''
    
    # Insert helper functions after imports
    import_end = content.find('from langchain_core.messages import AIMessage, AIMessageChunk')
    if import_end != -1:
        # Find the end of the import section
        next_line = content.find('\n\n', import_end)
        if next_line != -1:
            content = content[:next_line] + helper_functions + content[next_line:]
    
    # Fix remaining type annotations that might be causing issues
    content = re.sub(
        r'List\[types\.ContentBlock\]',
        'List[Dict[str, Any]]',
        content
    )
    
    # Fix chunk_position attribute access (this might not exist)
    content = content.replace(
        'message.chunk_position',
        'getattr(message, "chunk_position", None)'
    )
    
    # Fix variable redefinition by using different variable names
    content = re.sub(
        r'tc: Dict\[str, Any\] = \{([^}]+)\}\s+# Tool call chunk for function calls',
        r'tc_func: Dict[str, Any] = {\1}  # Tool call chunk for function calls',
        content,
        flags=re.DOTALL
    )
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed type annotation issues in block_translator.py")

if __name__ == "__main__":
    fix_block_translator_types()
