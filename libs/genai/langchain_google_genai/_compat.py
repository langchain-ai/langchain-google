"""Go from v1 content blocks to generativelanguage_v1beta format."""

import json
from typing import Any, Optional

from langchain_core.messages import content as types


def translate_citations_to_grounding_metadata(
    citations: list[types.Citation], web_search_queries: Optional[list[str]] = None
) -> dict[str, Any]:
    """Translate LangChain Citations to Google AI grounding metadata format.

    Args:
        citations: List of Citation content blocks.
        web_search_queries: Optional list of search queries that generated
            the grounding data.

    Returns:
        Google AI grounding metadata dictionary.

    Example:
        >>> citations = [
        ...     create_citation(
        ...         url="https://uefa.com/euro2024",
        ...         title="UEFA Euro 2024 Results",
        ...         start_index=0,
        ...         end_index=47,
        ...         cited_text="Spain won the UEFA Euro 2024 championship",
        ...     )
        ... ]
        >>> metadata = translate_citations_to_grounding_metadata(citations)
        >>> len(metadata["groundingChunks"])
        1
        >>> metadata["groundingChunks"][0]["web"]["uri"]
        'https://uefa.com/euro2024'
    """
    if not citations:
        return {}

    # Group citations by text segment (start_index, end_index, cited_text)
    segment_to_citations: dict[
        tuple[Optional[int], Optional[int], Optional[str]], list[types.Citation]
    ] = {}

    for citation in citations:
        key = (
            citation.get("start_index"),
            citation.get("end_index"),
            citation.get("cited_text"),
        )
        if key not in segment_to_citations:
            segment_to_citations[key] = []
        segment_to_citations[key].append(citation)

    # Build grounding chunks from unique URLs
    url_to_chunk_index: dict[str, int] = {}
    grounding_chunks: list[dict[str, Any]] = []

    for citation in citations:
        url = citation.get("url")
        if url and url not in url_to_chunk_index:
            url_to_chunk_index[url] = len(grounding_chunks)
            grounding_chunks.append(
                {"web": {"uri": url, "title": citation.get("title", "")}}
            )

    # Build grounding supports
    grounding_supports: list[dict[str, Any]] = []

    for (
        start_index,
        end_index,
        cited_text,
    ), citations_group in segment_to_citations.items():
        if start_index is not None and end_index is not None and cited_text:
            chunk_indices = []
            confidence_scores = []

            for citation in citations_group:
                url = citation.get("url")
                if url and url in url_to_chunk_index:
                    chunk_indices.append(url_to_chunk_index[url])

                    # Extract confidence scores from extras if available
                    extras = citation.get("extras", {})
                    google_metadata = extras.get("google_ai_metadata", {})
                    scores = google_metadata.get("confidence_scores", [])
                    confidence_scores.extend(scores)

            support = {
                "segment": {
                    "startIndex": start_index,
                    "endIndex": end_index,
                    "text": cited_text,
                },
                "groundingChunkIndices": chunk_indices,
            }

            if confidence_scores:
                support["confidenceScores"] = confidence_scores

            grounding_supports.append(support)

    # Extract search queries from extras if not provided
    if web_search_queries is None:
        web_search_queries = []
        for citation in citations:
            extras = citation.get("extras", {})
            google_metadata = extras.get("google_ai_metadata", {})
            queries = google_metadata.get("web_search_queries", [])
            web_search_queries.extend(queries)
        # Remove duplicates while preserving order
        web_search_queries = list(dict.fromkeys(web_search_queries))

    return {
        "webSearchQueries": web_search_queries,
        "groundingChunks": grounding_chunks,
        "groundingSupports": grounding_supports,
    }


def _convert_from_v1_to_generativelanguage_v1beta(
    content: list[types.ContentBlock], model_provider: str | None
) -> list[dict[str, Any]]:
    """Convert from v1 content blocks to generativelanguage_v1beta format.

    Args:
        content: List of v1 `ContentBlock` objects.
        model_provider: The model provider name that generated the v1 content.
    """
    new_content: list = []
    for block in content:
        if not isinstance(block, dict) or "type" not in block:
            # Invalid block, skip or handle as needed
            continue

        block_dict = dict(block)  # For typing

        # TextContentBlock
        if block_dict["type"] == "text":
            new_block = {"type": "text", "text": block_dict.get("text", "")}

            # Handle annotations (citations) -> grounding metadata
            if "annotations" in block_dict:
                if model_provider != "google_genai":
                    msg = (
                        "Annotations to grounding metadata conversion is only "
                        "supported for Google GenAI."
                    )
                    raise ValueError(msg)
                    # TODO: at some point, support other providers if possible

                # TODO: complete this, use `translate_citations_to_grounding_metadata`

                # Note: Google AI doesn't support per-block grounding metadata in input
                # This is typically handled at the message/request level
                pass

            new_content.append(new_block)

        # ImageContentBlock
        elif block_dict["type"] == "image":
            if base64 := block_dict.get("base64"):
                mime_type = block_dict.get("mime_type", "image/jpeg")  # Default to jpeg

                new_block = {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.encode("utf-8")
                        if isinstance(base64, str)
                        else base64,
                    }
                }
                new_content.append(new_block)
            # URL and file-ID based images are not directly supported in Google GenAI
            # They must be made into a Part with inline_data
            else:
                new_content.append({"type": "non_standard", "value": block_dict})

        # FileContentBlock (documents)
        elif block_dict["type"] == "file":
            # Google GenAI typically uses inline_data for file content
            if base64 := block_dict.get("base64"):
                mime_type = block_dict.get("mime_type", "application/octet-stream")

                new_block = {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.encode("utf-8")
                        if isinstance(base64, str)
                        else base64,
                    }
                }
                new_content.append(new_block)
            else:
                new_content.append({"type": "non_standard", "value": block_dict})

        # ToolCall -> FunctionCall
        elif block_dict["type"] == "tool_call":
            function_call = {
                "function_call": {
                    "name": block_dict.get("name", ""),
                    "args": block_dict.get("args", {}),
                }
            }
            new_content.append(function_call)

        # ToolCallChunk -> FunctionCall
        elif block_dict["type"] == "tool_call_chunk":
            try:
                args_str = block_dict.get("args") or "{}"
                input_ = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                input_ = {}

            function_call = {
                "function_call": {
                    "name": block_dict.get("name", "no_tool_name_present"),
                    "args": input_,
                }
            }
            new_content.append(function_call)

        # ReasoningContentBlock -> thinking (if Google GenAI)
        elif block_dict["type"] == "reasoning" and model_provider == "google_genai":
            # If you want to continue a conversation and preserve thought context, you
            # must pass back the thought_signature, not just the text. Thus, we can't
            # just pass the reasoning text as-is.

            # TODO: consider a mode where we interpret ReasoningContentBlock as a
            # TextContentBlock instead (prefixed with "Thought: " or similar?)
            new_block = {
                "type": "thinking",
                "thinking": block_dict.get("reasoning", ""),
            }
            # Signature is required to pass back to Google GenAI
            if "extras" in block_dict and isinstance(block_dict["extras"], dict):
                extras = block_dict["extras"]
                if "signature" in extras:
                    new_block["signature"] = extras["signature"]
            else:
                msg = (
                    "ReasoningContentBlock to thinking conversion requires "
                    "`extras.signature` field for Google GenAI."
                )
                raise ValueError(msg)

            new_content.append(new_block)

        # NonStandardContentBlock
        # TODO: Handle known non-standard types
        elif block_dict["type"] == "non_standard" and "value" in block_dict:
            value = block_dict["value"]
            if isinstance(value, dict):
                value_type = value.get("type")

                if value_type == "executable_code":
                    new_content.append(
                        {
                            "type": "executable_code",
                            "executable_code": value.get("executable_code", ""),
                            "language": value.get("language", ""),
                        }
                    )
                elif value_type == "code_execution_result":
                    new_content.append(
                        {
                            "type": "code_execution_result",
                            "code_execution_result": value.get(
                                "code_execution_result", ""
                            ),
                            "outcome": value.get("outcome", ""),
                        }
                    )
                else:
                    # Unknown non-standard, keep as is
                    new_content.append(value)
            else:
                # Non-dict value, keep as is
                new_content.append(value)

        else:
            # For any other block types, preserve as non-standard wrapper
            new_content.append({"type": "non_standard", "value": block_dict})

    return new_content
