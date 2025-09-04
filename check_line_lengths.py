#!/usr/bin/env python3
"""Check for lines exceeding 88 characters."""

def check_line_lengths(file_path, max_length=88):
    """Check for lines exceeding the specified length."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    violations = []
    for i, line in enumerate(lines, 1):
        if len(line) > max_length:
            violations.append((i, len(line), line.rstrip()))
    
    return violations

if __name__ == "__main__":
    file_path = "/home/daytona/langchain-google/libs/genai/langchain_google_genai/block_translator.py"
    violations = check_line_lengths(file_path)
    
    print(f"Lines exceeding 88 characters in {file_path}:")
    for line_num, length, content in violations:
        print(f"Line {line_num}: {length} chars")
        print(f"  {content}")
        print()
