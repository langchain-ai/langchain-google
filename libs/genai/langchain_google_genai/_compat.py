"""Go from v1 content blocks to generativelanguage_v1beta format."""

import json
import logging
from typing import Any, Literal, cast

from langchain_core.messages import content as types

logger = logging.getLogger(__name__)

#: Providers whose content blocks and thought signatures Gemini can consume
#: directly. Vertex AI serves the same Gemini models, so its signatures are valid
#: here and must not be stripped.
#:
#: Defined here because both v1 projection and chat-history replay use the same
#: provider classification rules.
_NATIVE_MODEL_PROVIDERS = frozenset({"google_genai", "google_vertexai"})
_ModelProviderKind = Literal["native", "foreign", "unknown"]


def _classify_model_provider(model_provider: str | None) -> _ModelProviderKind:
    """Classify a model provider for content-block replay.

    Args:
        model_provider: Provider recorded on the source message, if known.

    Returns:
        Whether the message is native to Gemini, known to be foreign, or lacks
        provider metadata.
    """
    if model_provider is None:
        return "unknown"
    if model_provider in _NATIVE_MODEL_PROVIDERS:
        return "native"
    return "foreign"


def translate_citations_to_grounding_metadata(
    citations: list[types.Citation], web_search_queries: list[str] | None = None
) -> dict[str, Any]:
    """Translate LangChain Citations to Google AI grounding metadata format.

    Args:
        citations: List of `Citation` content blocks.
        web_search_queries: Optional list of search queries that generated
            the grounding data.

    Returns:
        Google AI grounding metadata dictionary.

    Example:
        ```python
        citations = [
            create_citation(
                url="https://uefa.com/euro2024",
                title="UEFA Euro 2024 Results",
                start_index=0,
                end_index=47,
                cited_text="Spain won the UEFA Euro 2024 championship",
            )
        ]

        metadata = translate_citations_to_grounding_metadata(citations)
        len(metadata["groundingChunks"])
        # -> 1

        metadata["groundingChunks"][0]["web"]["uri"]
        # -> 'https://uefa.com/euro2024'
        ```
    """
    if not citations:
        return {}

    # Group citations by text segment (start_index, end_index, cited_text)
    segment_to_citations: dict[
        tuple[int | None, int | None, str | None], list[types.Citation]
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
    """Convert v1 content blocks to `generativelanguage_v1beta` `Content`.

    Args:
        content: List of v1 `ContentBlock` objects.
        model_provider: The model provider name that generated the v1 content.

    Returns:
        List of dictionaries in `generativelanguage_v1beta` `Content` format, ready to
            be sent to the API.

    Note:
        Every shape emitted here must be accepted by
        `chat_models._convert_to_parts`, which converts these dicts to `Part`
        objects.
    """
    provider_kind = _classify_model_provider(model_provider)
    new_content: list = []
    for block in content:
        if not isinstance(block, dict) or "type" not in block:
            logger.warning(
                "Dropping v1 content block that is not a typed mapping (got: %s).",
                type(block) if not isinstance(block, dict) else "dict without 'type'",
            )
            continue

        block_dict = dict(block)  # (For typing)

        # TextContentBlock
        if block_dict["type"] == "text":
            new_block = {"text": block_dict.get("text", "")}
            if (
                thought_signature := (block_dict.get("extras") or {}).get("signature")  # type: ignore[attr-defined]
            ) and provider_kind != "foreign":
                new_block["thought_signature"] = thought_signature
            new_content.append(new_block)
            # Citations are only handled on output. Can't pass them back :/

        # ReasoningContentBlock -> thinking
        elif block_dict["type"] == "reasoning":
            extras = block_dict.get("extras")
            signature = extras.get("signature") if isinstance(extras, dict) else None
            if provider_kind == "native" and not signature:
                logger.warning(
                    "Dropping v1 reasoning block with no thought signature; "
                    "its text will not be sent back to the model."
                )
                continue
            reasoning_text = block_dict.get("reasoning")
            summary = block_dict.get("summary")
            if reasoning_text is None and isinstance(summary, list):
                summary_texts = [
                    text.strip()
                    for item in summary
                    if isinstance(item, dict)
                    and isinstance((text := item.get("text")), str)
                    and text.strip()
                ]
                reasoning_text = " ".join(summary_texts)
            if not reasoning_text and not (signature and provider_kind != "foreign"):
                continue
            new_block = {
                "thought": True,
                "text": reasoning_text or "",
            }
            if signature and provider_kind != "foreign":
                new_block["thought_signature"] = signature
            new_content.append(new_block)

        # ImageContentBlock
        elif block_dict["type"] == "image":
            # `Blob.data` accepts a base64 `str` and decodes it (the field sets
            # `val_json_bytes="base64"`), so pass it through untouched. Named
            # `b64_data` rather than `base64`: the latter shadows the stdlib
            # module name and previously caused this payload to be corrupted by
            # a stray `base64.encode("utf-8")` call.
            if b64_data := block_dict.get("base64"):
                new_block = {
                    "inline_data": {
                        "mime_type": block_dict.get("mime_type", "image/jpeg"),
                        "data": b64_data,
                    }
                }
                new_content.append(new_block)
            elif (url := block_dict.get("url")) and provider_kind != "foreign":
                # Google file service
                new_block = {
                    "file_data": {
                        "mime_type": block_dict.get("mime_type", "image/jpeg"),
                        "file_uri": url,
                    }
                }
                new_content.append(new_block)

        # TODO: AudioContentBlock -> audio once models support passing back in

        # FileContentBlock (documents)
        elif block_dict["type"] == "file":
            # `Blob.data` accepts a base64 `str` and decodes it (the field sets
            # `val_json_bytes="base64"`), so pass it through untouched. Named
            # `b64_data` rather than `base64`: the latter shadows the stdlib
            # module name and previously caused this payload to be corrupted by
            # a stray `base64.encode("utf-8")` call.
            if b64_data := block_dict.get("base64"):
                new_block = {
                    "inline_data": {
                        "mime_type": block_dict.get(
                            "mime_type", "application/octet-stream"
                        ),
                        "data": b64_data,
                    }
                }
                new_content.append(new_block)
            elif (file_id := block_dict.get("file_id")) and provider_kind != "foreign":
                # File ID from uploaded file
                new_block = {
                    "file_data": {
                        "mime_type": block_dict.get(
                            "mime_type", "application/octet-stream"
                        ),
                        "file_uri": file_id,
                    }
                }
                new_content.append(new_block)
            elif (url := block_dict.get("url")) and provider_kind != "foreign":
                # Google file service
                new_block = {
                    "file_data": {
                        "mime_type": block_dict.get(
                            "mime_type", "application/octet-stream"
                        ),
                        "file_uri": url,
                    }
                }
                new_content.append(new_block)

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
                new_content.append(
                    {
                        "executable_code": {
                            "language": executable_code["language"],
                            "code": executable_code["executable_code"],
                        }
                    }
                )

        elif block_dict["type"] == "server_tool_result":
            extras = cast("dict", block_dict.get("extras", {}))
            if extras.get("block_type") == "code_execution_result":
                # LangChain v0 format
                raw_outcome = extras.get("outcome", "")
                if isinstance(raw_outcome, int):
                    if raw_outcome == 1:
                        outcome = "OUTCOME_OK"
                    elif raw_outcome == 2:
                        outcome = "OUTCOME_FAILED"
                    else:
                        outcome = "OUTCOME_UNSPECIFIED"
                else:
                    outcome = raw_outcome
                # Google generativelanguage format
                new_content.append(
                    {
                        "code_execution_result": {
                            "outcome": outcome,
                            "output": block_dict.get("output", ""),
                        }
                    }
                )

        elif block_dict["type"] == "non_standard":
            value = block_dict.get("value")
            if provider_kind != "foreign" and isinstance(value, dict):
                # Core wraps provider-native blocks it cannot standardize (for
                # example Gemini's `media` block) as `non_standard`. Preserve the
                # raw value for native and provider-less checkpoints; the caller
                # converts native content strictly and unknown content leniently.
                new_content.append(value)
            else:
                logger.warning(
                    "Dropping non-standard v1 content block that cannot be "
                    "represented as a Gemini part (inner type: %s).",
                    value.get("type") if isinstance(value, dict) else None,
                )

        else:
            # No branch matched. A block type this whitelist does not know is
            # dropped, which is right for another provider's blocks but is a bug
            # signal for a newly added core block type -- hence the warning.
            logger.warning(
                "Dropping v1 content block with no Gemini equivalent (type: %s).",
                block_dict["type"],
            )

    return new_content
