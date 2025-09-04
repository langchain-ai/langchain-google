#!/usr/bin/env python3

def check_line_lengths(file_path, max_length=88):
    """Check for lines exceeding the specified length."""
    violations = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            # Remove newline for length check but keep original for display
            line_content = line.rstrip('\n\r')
            if len(line_content) > max_length:
                violations.append((line_num, len(line_content), line_content))
        
        return violations
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

if __name__ == "__main__":
    file_path = "/home/daytona/langchain-google/libs/genai/langchain_google_genai/chat_models.py"
    violations = check_line_lengths(file_path)
    
    if violations:
        print(f"Lines exceeding 88 characters in {file_path}:")
        for line_num, length, content in violations:
            print(f"Line {line_num} ({length} chars): {content}")
    else:
        print(f"No lines exceed 88 characters in {file_path}")
