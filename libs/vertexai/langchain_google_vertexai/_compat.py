"""Go from v1 content blocks to VertexAI format."""

from typing import Any, cast

from langchain_core.messages import content as types


def _convert_from_v1_to_vertex(
    content: list[types.ContentBlock], model_provider: str | None
) -> list[dict[str, Any]]:
    """Convert v1 content blocks to VertexAI content.

    Args:
        content: List of v1 `ContentBlock` objects.
        model_provider: The model provider name that generated the v1 content.

    Returns:
        List of dictionaries in VertexAI content format.
    """
    new_content: list = []
    for block in content:
        block_dict = dict(block)  # (For typing)

        # TextContentBlock
        if block_dict["type"] == "text":
            new_block = {"type": "text", "text": block_dict.get("text", "")}
            if "extras" in block_dict and isinstance(block_dict["extras"], dict):
                extras = block_dict["extras"]
                if "signature" in extras:
                    new_block["thought_signature"] = extras["signature"]
            new_content.append(new_block)
            # Citations are only handled on output. Can't pass them back :/

        # ReasoningContentBlock -> thinking
        elif block_dict["type"] == "reasoning" and model_provider in (
            "google_vertexai",
            "google_genai",
        ):
            # Google requires passing back the thought_signature when available.
            # Signatures are only provided when function calling is enabled.
            new_block = {
                "type": "thinking",
                "thinking": block_dict.get("reasoning", ""),
            }
            if "extras" in block_dict and isinstance(block_dict["extras"], dict):
                extras = block_dict["extras"]
                if "signature" in extras:
                    new_block["thought_signature"] = extras["signature"]

            new_content.append(new_block)

        # ToolCall -> FunctionCall
        elif block_dict["type"] == "tool_call":
            # read from .tool_calls
            continue

        elif block_dict["type"] == "server_tool_call":
            if block_dict.get("name") == "code_interpreter":
                # LangChain v0 format
                args = cast("dict", block_dict.get("args", {}))
                executable_code = {
                    "type": "executable_code",
                    "executable_code": args.get("code", ""),
                    "language": args.get("language", ""),
                    "id": block_dict.get("id", ""),
                }
                # Google generativelanguage format
                new_content.append(executable_code)

        elif block_dict["type"] == "server_tool_result":
            extras = cast("dict", block_dict.get("extras", {}))
            if extras.get("block_type") == "code_execution_result":
                # LangChain v0 format
                code_execution_result = {
                    "type": "code_execution_result",
                    "code_execution_result": block_dict.get("output", ""),
                    "outcome": extras.get("outcome", ""),
                    "tool_call_id": block_dict.get("tool_call_id", ""),
                }
                # Google generativelanguage format
                new_content.append(code_execution_result)

        elif block_dict["type"] == "function_call_signature":
            new_content.append(block_dict)

        elif block_dict["type"] == "non_standard":
            new_content.append(block_dict["value"])

    return new_content
