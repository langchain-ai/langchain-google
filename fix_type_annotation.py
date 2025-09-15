#!/usr/bin/env python3
"""
Script to fix the type annotation in ChatGoogleGenerativeAI._prepare_request method
"""

import re

def fix_type_annotation():
    file_path = 'libs/genai/langchain_google_genai/chat_models.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the specific type annotation
    # Looking for: ) -> Tuple[GenerateContentRequest, Dict[str, Any]]:
    # Replace with: ) -> GenerateContentRequest:
    pattern = r'\) -> Tuple\[GenerateContentRequest, Dict\[str, Any\]\]:'
    replacement = r') -> GenerateContentRequest:'
    
    new_content = re.sub(pattern, replacement, content)
    
    # Check if the replacement was made
    if new_content != content:
        # Write the modified content back
        with open(file_path, 'w') as f:
            f.write(new_content)
        print("Successfully fixed the type annotation!")
        return True
    else:
        print("No changes made - pattern not found")
        return False

if __name__ == "__main__":
    fix_type_annotation()
