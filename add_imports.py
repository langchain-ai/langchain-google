#!/usr/bin/env python3
"""Script to add standard content block imports to chat_models.py"""

import os

# Read the current file
file_path = "/home/daytona/langchain-google/libs/genai/langchain_google_genai/chat_models.py"
with open(file_path, 'r') as f:
    lines = f.readlines()

# Find the line after the langchain_core.messages.tool import
insert_index = None
for i, line in enumerate(lines):
    if "from langchain_core.messages.tool import invalid_tool_call, tool_call, tool_call_chunk" in line:
        insert_index = i + 1
        break

if insert_index is None:
    print("Could not find the insertion point")
    exit(1)

# Define the imports to add
imports_to_add = [
    "from langchain_core.messages.content import (\n",
    "    AudioContentBlock,\n",
    "    CodeInterpreterCall,\n",
    "    CodeInterpreterResult,\n",
    "    FileContentBlock,\n",
    "    ImageContentBlock,\n",
    "    NonStandardContentBlock,\n",
    "    ReasoningContentBlock,\n",
    "    TextContentBlock,\n",
    "    VideoContentBlock,\n",
    "    create_audio_block,\n",
    "    create_file_block,\n",
    "    create_image_block,\n",
    "    create_non_standard_block,\n",
    "    create_reasoning_block,\n",
    "    create_text_block,\n",
    "    create_video_block,\n",
    ")\n"
]

# Insert the imports
lines[insert_index:insert_index] = imports_to_add

# Write the modified file back
with open(file_path, 'w') as f:
    f.writelines(lines)

print("Successfully added standard content block imports to chat_models.py")
