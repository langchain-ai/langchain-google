"""Go from v1 content blocks to generativelanguage_v1beta format."""

import base64
import copy
import json
import logging
from collections.abc import Mapping
from typing import Any, Literal, cast

import langchain_core.messages.block_translators.google_genai as _core_genai_translator
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.block_translators import register_translator

logger = logging.getLogger(__name__)

# Vertex and the Gemini Developer API share content and signature formats.
_NATIVE_MODEL_PROVIDERS = frozenset({"google_genai", "google_vertexai"})
_ModelProviderKind = Literal["native", "foreign", "unknown"]
# Standardized `reasoning` plus the raw shapes providers emit before conversion.
_REASONING_BLOCK_TYPES = frozenset({"reasoning", "thinking", "reasoning_content"})


_MEDIA_PROCESSING_TOOL_NAME = "media_processing"


def _classify_model_provider(model_provider: str | None) -> _ModelProviderKind:
    """Classify a source provider for Gemini content replay."""
    if model_provider is None:
        return "unknown"
    if model_provider in _NATIVE_MODEL_PROVIDERS:
        return "native"
    return "foreign"


def _is_file_uri_supported(file_uri: Any, *, use_vertexai: bool) -> bool:
    """Check whether the target Google backend accepts a file URI.

    Google Cloud Storage references are supported by Vertex AI but not by the
    Gemini Developer API.
    """
    return not (
        isinstance(file_uri, str) and file_uri.startswith("gs://") and not use_vertexai
    )


def _file_data_block(
    file_uri: Any,
    mime_type: Any,
    *,
    use_vertexai: bool,
) -> dict[str, Any] | None:
    """Build file data when the target backend accepts its URI."""
    if not _is_file_uri_supported(file_uri, use_vertexai=use_vertexai):
        logger.warning(
            "Dropping Google Cloud Storage file URI because the target Gemini "
            "Developer API does not support gs:// references."
        )
        return None
    return {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}


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
    content: list[types.ContentBlock],
    model_provider: str | None,
    *,
    use_vertexai: bool = False,
) -> list[dict[str, Any]]:
    """Convert v1 content blocks to `generativelanguage_v1beta` `Content`.

    Args:
        content: List of v1 `ContentBlock` objects.
        model_provider: The model provider name that generated the v1 content.
        use_vertexai: Whether the content will be sent to the Vertex AI backend.

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
            extras = block_dict.get("extras")
            thought_signature = block_dict.get("thought_signature") or (
                extras.get("signature") if isinstance(extras, dict) else None
            )
            if thought_signature and provider_kind != "foreign":
                new_block["thought_signature"] = thought_signature
            new_content.append(new_block)
            # Citations are only handled on output. Can't pass them back :/

        # ReasoningContentBlock -> thinking
        elif block_dict["type"] == "reasoning":
            if provider_kind == "foreign":
                # Another provider's chain-of-thought (for example AWS Bedrock or
                # OpenAI summary blocks) is not a Gemini thought; sending it back
                # would leak foreign reasoning text into Gemini requests.
                logger.warning(
                    "Dropping v1 reasoning block from provider %r; foreign "
                    "reasoning is not replayed as a Gemini thought part.",
                    model_provider,
                )
                continue
            extras = block_dict.get("extras")
            signature = block_dict.get("thought_signature") or (
                extras.get("signature") if isinstance(extras, dict) else None
            )
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
            # Foreign blocks already continued above, and native blocks with no
            # signature were dropped, so an unsigned block here is provider-less.
            if not reasoning_text and not signature:
                continue
            new_block = {
                "thought": True,
                "text": reasoning_text or "",
            }
            if signature:
                new_block["thought_signature"] = signature
            new_content.append(new_block)

        # ImageContentBlock
        elif block_dict["type"] == "image":
            # SDK validation decodes base64 strings; do not pre-encode them.
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
                file_data_block = _file_data_block(
                    url,
                    block_dict.get("mime_type", "image/jpeg"),
                    use_vertexai=use_vertexai,
                )
                if file_data_block is not None:
                    new_content.append(file_data_block)

        # VideoContentBlock / AudioContentBlock
        elif block_dict["type"] in ("video", "audio"):
            default_av_mime = (
                "video/mp4" if block_dict["type"] == "video" else "audio/wav"
            )
            if b64_data := block_dict.get("base64"):
                new_content.append(
                    {
                        "inline_data": {
                            "mime_type": block_dict.get("mime_type", default_av_mime),
                            "data": b64_data,
                        }
                    }
                )
            elif (file_id := block_dict.get("file_id")) and provider_kind != "foreign":
                file_data_block = _file_data_block(
                    file_id,
                    block_dict.get("mime_type", default_av_mime),
                    use_vertexai=use_vertexai,
                )
                if file_data_block is not None:
                    new_content.append(file_data_block)
            elif (url := block_dict.get("url")) and provider_kind != "foreign":
                file_data_block = _file_data_block(
                    url,
                    block_dict.get("mime_type", default_av_mime),
                    use_vertexai=use_vertexai,
                )
                if file_data_block is not None:
                    new_content.append(file_data_block)

        # FileContentBlock (documents)
        elif block_dict["type"] == "file":
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
                file_data_block = _file_data_block(
                    file_id,
                    block_dict.get("mime_type", "application/octet-stream"),
                    use_vertexai=use_vertexai,
                )
                if file_data_block is not None:
                    new_content.append(file_data_block)
            elif (url := block_dict.get("url")) and provider_kind != "foreign":
                # Google file service
                file_data_block = _file_data_block(
                    url,
                    block_dict.get("mime_type", "application/octet-stream"),
                    use_vertexai=use_vertexai,
                )
                if file_data_block is not None:
                    new_content.append(file_data_block)

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

        elif block_dict["type"] == "function_call_signature":
            # The caller extracts this legacy sidecar before v1 projection and
            # attaches it to the function call rebuilt from `message.tool_calls`.
            continue

        elif block_dict["type"] == "server_tool_call":
            if block_dict.get("name") == _MEDIA_PROCESSING_TOOL_NAME:
                # Agentic video steps are informational. The GenerateContent API rejects
                # them when they are echoed back ("Tool type of tool_call part does not
                # match with tool call context"), so they are dropped on replay; the
                # original media part is what preserves video context.
                continue
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
            if extras.get("block_type") == _MEDIA_PROCESSING_TOOL_NAME:
                # Paired with the media processing call dropped above.
                continue
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


_FC_THOUGHT_SIGS_KEY = "__gemini_function_call_thought_signatures__"
_RAW_CALL_NAME_MAP: dict[str, str] = {
    "code_execution_call": "code_interpreter",
    "google_search_call": "google_search",
    "url_context_call": "url_context",
    "google_maps_call": "google_maps",
    "file_search_call": "file_search",
    "retrieval_call": "retrieval",
    "processing_call": _MEDIA_PROCESSING_TOOL_NAME,
}
_RAW_RESULT_TYPES = frozenset(
    {
        "google_search_result",
        "url_context_result",
        "google_maps_result",
        "file_search_result",
        "mcp_server_tool_result",
        "retrieval_result",
        "processing_result",
    }
)
_PASS_THROUGH_V1_TYPES = frozenset(
    {
        "server_tool_call",
        "server_tool_call_chunk",
        "server_tool_result",
        "tool_call",
        "tool_call_chunk",
        "invalid_tool_call",
        "refusal",
    }
)


def _utf8_slice_to_char_span(
    text: str, start_byte: Any, end_byte: Any
) -> tuple[int, int, str]:
    """Convert UTF-8 byte offsets `[start_byte, end_byte)` into Python char offsets."""
    encoded = text.encode("utf-8")
    total = len(encoded)
    sb = max(0, min(int(start_byte) if isinstance(start_byte, int) else 0, total))
    eb = max(sb, min(int(end_byte) if isinstance(end_byte, int) else total, total))
    cs = len(encoded[:sb].decode("utf-8", errors="ignore"))
    ce = len(encoded[:eb].decode("utf-8", errors="ignore"))
    return cs, ce, text[cs:ce]


def _normalize_annotation_to_v1(ann: Any, block_text: str) -> types.Annotation | None:
    """Translate an inline or raw Interactions citation into v1 `Annotation`."""
    if not isinstance(ann, Mapping):
        return None
    atype = ann.get("type")
    if atype in ("citation", "non_standard_annotation"):
        return cast("types.Annotation", copy.deepcopy(dict(ann)))
    if atype in ("url_citation", "place_citation", "file_citation"):
        cs, ce, cited = _utf8_slice_to_char_span(
            block_text, ann.get("start_index"), ann.get("end_index")
        )
        if atype == "url_citation":
            return types.create_citation(
                url=str(ann["url"]) if ann.get("url") else None,
                title=str(ann["title"]) if ann.get("title") else None,
                start_index=cs,
                end_index=ce,
                cited_text=cited,
            )
        if atype == "place_citation":
            extra_p = {"place_id": str(ann["place_id"])} if ann.get("place_id") else {}
            title_p = ann.get("name") or ann.get("title")
            kw_p: dict[str, Any] = {"google_ai_metadata": extra_p} if extra_p else {}
            return types.create_citation(
                url=str(ann["url"]) if ann.get("url") else None,
                title=str(title_p) if title_p else None,
                start_index=cs,
                end_index=ce,
                cited_text=cited,
                **kw_p,
            )
        doc_uri = ann.get("document_uri") or ann.get("url")
        fname = ann.get("file_name") or ann.get("source")
        f_meta: dict[str, Any] = {"type": "file_citation"}
        for fk in ("page_number", "media_id", "source", "custom_metadata"):
            if ann.get(fk) is not None:
                f_meta[fk] = copy.deepcopy(ann[fk])
        return types.create_citation(
            url=str(doc_uri) if doc_uri else None,
            title=str(fname) if fname else None,
            start_index=cs,
            end_index=ce,
            cited_text=cited,
            file_citation_metadata=f_meta,
        )
    return {"type": "non_standard_annotation", "value": copy.deepcopy(dict(ann))}


def _convert_interactions_dict_item(
    item: Mapping[str, Any],
) -> types.ContentBlock | None:
    """Convert an Interactions or standard v1 dict block into a v1 `ContentBlock`."""
    btype = item.get("type")
    idx = item.get("index")

    def _with_idx(blk: dict[str, Any]) -> types.ContentBlock:
        if idx is not None and "index" not in blk:
            blk["index"] = idx
        return cast("types.ContentBlock", blk)

    if btype == "text":
        txt = str(item.get("text", ""))
        out_t: dict[str, Any] = {
            k: copy.deepcopy(v) for k, v in item.items() if k != "annotations"
        }
        out_t["type"] = "text"
        out_t["text"] = txt
        if isinstance(item.get("annotations"), list):
            anns = [
                norm
                for a in item["annotations"]
                if (norm := _normalize_annotation_to_v1(a, txt)) is not None
            ]
            if anns:
                out_t["annotations"] = anns
        return _with_idx(out_t)

    if btype in ("reasoning", "thinking", "thought"):
        out_r: dict[str, Any] = {"type": "reasoning"}
        if btype == "reasoning":
            r_txt = item.get("reasoning")
            if not isinstance(r_txt, str):
                summary = item.get("summary")
                r_txt = (
                    "".join(
                        str(s.get("text", ""))
                        for s in summary
                        if isinstance(s, Mapping)
                    )
                    if isinstance(summary, list)
                    else ""
                )
            out_r["reasoning"] = r_txt
        elif btype == "thinking":
            out_r["reasoning"] = str(item.get("thinking", ""))
        else:
            raw_sum = item.get("summary")
            out_r["reasoning"] = (
                "".join(
                    str(s.get("text", ""))
                    for s in raw_sum
                    if isinstance(s, Mapping) and s.get("type") == "text"
                )
                if isinstance(raw_sum, list)
                else ""
            )
        extras_r: dict[str, Any] = (
            copy.deepcopy(dict(item["extras"]))
            if isinstance(item.get("extras"), Mapping)
            else {}
        )
        if item.get("signature") and "signature" not in extras_r:
            sig_v = item["signature"]
            extras_r["signature"] = (
                base64.b64encode(sig_v).decode("ascii")
                if isinstance(sig_v, bytes)
                else str(sig_v)
            )
        if extras_r:
            out_r["extras"] = extras_r
        if item.get("id"):
            out_r["id"] = str(item["id"])
        return _with_idx(out_r)

    if btype in ("image", "audio", "video", "file", "document"):
        target_type = "file" if btype == "document" else str(btype)
        out_d: dict[str, Any] = {"type": target_type}
        if b64 := (item.get("base64") or item.get("data")):
            out_d["base64"] = (
                base64.b64encode(b64).decode("ascii")
                if isinstance(b64, bytes)
                else str(b64)
            )
        elif u := (item.get("url") or item.get("uri")):
            out_d["url"] = str(u)
        elif item.get("file_id"):
            out_d["file_id"] = str(item["file_id"])
        if item.get("mime_type"):
            out_d["mime_type"] = str(item["mime_type"])
        if item.get("id"):
            out_d["id"] = str(item["id"])
        if isinstance(item.get("extras"), Mapping) and item["extras"]:
            out_d["extras"] = copy.deepcopy(dict(item["extras"]))
        return _with_idx(out_d)

    if btype in _PASS_THROUGH_V1_TYPES:
        return _with_idx(copy.deepcopy(dict(item)))

    if btype in _RAW_CALL_NAME_MAP or btype == "mcp_server_tool_call":
        call_name = (
            str(item.get("name") or "mcp_server_tool")
            if btype == "mcp_server_tool_call"
            else _RAW_CALL_NAME_MAP[str(btype)]
        )
        raw_args = item.get("arguments")
        args_d = copy.deepcopy(dict(raw_args)) if isinstance(raw_args, Mapping) else {}
        if btype == "code_execution_call":
            args_d.setdefault("code", "")
            args_d.setdefault("language", "python")
        extras_sc: dict[str, Any] = {"step_type": str(btype)}
        for ek in ("search_type", "server_name", "retrieval_type", "signature"):
            if ek == "signature" and btype == "mcp_server_tool_call":
                continue
            if item.get(ek):
                extras_sc[ek] = item[ek]
        return _with_idx(
            {
                "type": "server_tool_call",
                "name": call_name,
                "id": str(item.get("id") or ""),
                "args": args_d,
                "extras": extras_sc,
            }
        )

    if btype in _RAW_RESULT_TYPES or (
        btype == "code_execution_result" and "code_execution_result" not in item
    ):
        is_err = bool(item.get("is_error"))
        btag = (
            _MEDIA_PROCESSING_TOOL_NAME if btype == "processing_result" else str(btype)
        )
        extras_sr: dict[str, Any] = {"block_type": btag, "step_type": str(btype)}
        if btype == "code_execution_result":
            extras_sr["outcome"] = 2 if is_err else 1
        for ek in ("name", "server_name", "signature"):
            if ek == "signature" and btype == "mcp_server_tool_result":
                continue
            if item.get(ek):
                extras_sr[ek] = item[ek]
        raw_out = item.get("result")
        return _with_idx(
            {
                "type": "server_tool_result",
                "tool_call_id": str(item.get("call_id") or ""),
                "status": "error" if is_err else "success",
                "output": copy.deepcopy(raw_out) if raw_out is not None else {},
                "extras": extras_sr,
            }
        )

    return None


def _build_tool_blocks_for_message(
    message: AIMessage,
    existing_blocks: list[types.ContentBlock],
    *,
    is_chunk_streaming: bool,
) -> list[types.ContentBlock]:
    """Build v1 `ToolCallChunk` (mid-stream) or `ToolCall` (finalized) blocks."""
    present_ids = {
        b.get("id")
        for b in existing_blocks
        if isinstance(b, Mapping)
        and b.get("type") in ("tool_call", "tool_call_chunk")
        and b.get("id")
    }
    raw_sigs = message.additional_kwargs.get(_FC_THOUGHT_SIGS_KEY)
    sig_map = raw_sigs if isinstance(raw_sigs, Mapping) else {}
    out: list[types.ContentBlock] = []

    if is_chunk_streaming and isinstance(message, AIMessageChunk):
        for tcc in message.tool_call_chunks:
            cid = tcc.get("id")
            if cid and cid in present_ids:
                continue
            tc_blk: dict[str, Any] = {
                "type": "tool_call_chunk",
                "id": cid,
                "name": tcc.get("name"),
                "args": tcc.get("args"),
            }
            tidx = tcc.get("index")
            if tidx is not None:
                tc_blk["index"] = (
                    tidx
                    if isinstance(tidx, str) and tidx.startswith("lc_")
                    else f"lc_tc_{tidx}"
                )
            if cid and sig_map.get(cid):
                tc_blk["extras"] = {"signature": sig_map[cid]}
            out.append(cast("types.ContentBlock", tc_blk))
        return out

    idx_map: dict[str, str] = {}
    for tcc in getattr(message, "tool_call_chunks", None) or []:
        tid = tcc.get("id")
        tidx = tcc.get("index")
        if tid and tidx is not None:
            idx_map[tid] = (
                tidx
                if isinstance(tidx, str) and tidx.startswith("lc_")
                else f"lc_tc_{tidx}"
            )

    for tc in message.tool_calls:
        cid = tc.get("id")
        if cid and cid not in present_ids:
            final_tc: dict[str, Any] = {
                "type": "tool_call",
                "id": cid,
                "name": tc["name"],
                "args": tc["args"],
            }
            if cid in idx_map:
                final_tc["index"] = idx_map[cid]
            if sig_map.get(cid):
                final_tc["extras"] = {"signature": sig_map[cid]}
            out.append(cast("types.ContentBlock", final_tc))
    return out


_REFUSAL_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"})


def _reason_name(reason: object) -> str | None:
    """Return the enum member name of a Gemini finish or block reason."""
    if not reason:
        return None
    return str(getattr(reason, "name", reason)).rsplit(".", 1)[-1]


def _fallback_refusal_from_metadata(
    response_metadata: Mapping[str, Any],
) -> types.ContentBlock | None:
    """Build a refusal block when `langchain_core` lacks `_refusal_from_metadata`."""
    extras: dict[str, Any] = {}
    finish_reason = _reason_name(response_metadata.get("finish_reason"))
    if finish_reason in _REFUSAL_FINISH_REASONS:
        extras["finish_reason"] = finish_reason
    feedback = response_metadata.get("prompt_feedback")
    if isinstance(feedback, Mapping):
        block_reason = _reason_name(feedback.get("block_reason"))
        if block_reason and block_reason != "BLOCKED_REASON_UNSPECIFIED":
            extras["block_reason"] = block_reason
    if not extras:
        return None
    finish_message = response_metadata.get("finish_message")
    refusal_text = (
        finish_message if isinstance(finish_message, str) else str(finish_message or "")
    )
    return cast(
        "types.ContentBlock",
        {
            "type": "refusal",
            "refusal": refusal_text,
            "extras": extras,
        },
    )


def _refusal_from_metadata(
    response_metadata: dict[str, Any],
) -> types.ContentBlock | None:
    """Resolve a v1 refusal block via `langchain_core` with a pure-Python fallback."""
    core_fn = getattr(_core_genai_translator, "_refusal_from_metadata", None)
    if callable(core_fn):
        return cast("types.ContentBlock | None", core_fn(response_metadata))
    return _fallback_refusal_from_metadata(response_metadata)


def _core_convert_to_v1(message: AIMessage) -> list[types.ContentBlock]:
    """Delegate legacy/`v0` block conversion to `langchain_core`'s GenAI translator."""
    core_conv = getattr(_core_genai_translator, "_convert_to_v1_from_genai", None)
    if callable(core_conv):
        return cast("list[types.ContentBlock]", core_conv(message))
    if isinstance(message, AIMessageChunk):
        chunk_fn = getattr(_core_genai_translator, "translate_content_chunk", None)
        if callable(chunk_fn):
            return cast("list[types.ContentBlock]", chunk_fn(message))
    content_fn = getattr(_core_genai_translator, "translate_content", None)
    if callable(content_fn):
        return cast("list[types.ContentBlock]", content_fn(message))
    return _translate_interactions_v1_message(message, is_chunk_streaming=False)


def _translate_interactions_v1_message(
    message: AIMessage, *, is_chunk_streaming: bool
) -> list[types.ContentBlock]:
    """Translate an `output_version='interactions/v1'` message or chunk to v1 blocks."""
    blocks: list[types.ContentBlock] = []
    if isinstance(message.content, str):
        if message.content:
            blocks.append({"type": "text", "text": message.content})
    elif isinstance(message.content, list):
        for raw in message.content:
            if isinstance(raw, str):
                blocks.append({"type": "text", "text": raw})
            elif isinstance(raw, Mapping):
                if (
                    is_chunk_streaming
                    and isinstance(message, AIMessageChunk)
                    and raw.get("type") == "tool_call"
                ):
                    # Streamed incrementally via message.tool_call_chunks during SSE;
                    # kept in raw content so sum(chunks) matches unary interactions/v1.
                    continue
                conv = _convert_interactions_dict_item(raw)
                if conv is not None:
                    blocks.append(conv)
                else:
                    ns: dict[str, Any] = {
                        "type": "non_standard",
                        "value": copy.deepcopy(dict(raw)),
                    }
                    if raw.get("index") is not None:
                        ns["index"] = raw["index"]
                    blocks.append(cast("types.ContentBlock", ns))

    gm = message.response_metadata.get("grounding_metadata")
    cit_fn = getattr(
        _core_genai_translator, "translate_grounding_metadata_to_citations", None
    )
    if (
        gm
        and callable(cit_fn)
        and not any(b.get("type") == "text" and b.get("annotations") for b in blocks)
    ):
        cits = cit_fn(gm)
        if cits:
            for i, blk in enumerate(blocks):
                if blk.get("type") == "text":
                    blocks[i] = cast("types.ContentBlock", {**blk, "annotations": cits})
                    break

    aud = message.additional_kwargs.get("audio")
    if (
        isinstance(aud, bytes)
        and aud
        and not any(b.get("type") == "audio" for b in blocks)
    ):
        blocks.append(
            {
                "type": "audio",
                "base64": base64.b64encode(aud).decode("ascii"),
                "mime_type": "audio/wav",
            }
        )

    blocks.extend(
        _build_tool_blocks_for_message(
            message, blocks, is_chunk_streaming=is_chunk_streaming
        )
    )
    if not any(b.get("type") == "refusal" for b in blocks) and (
        refusal := _refusal_from_metadata(message.response_metadata)
    ):
        blocks.append(refusal)
    return blocks


def _is_interactions_message(message: AIMessage) -> bool:
    """Return True when `message` was produced by the Gemini Interactions API."""
    meta = message.response_metadata or {}
    return (
        "interaction_id" in meta
        or "status" in meta
        or meta.get("output_version") == "interactions/v1"
    )


def _relocate_multistep_citations(
    blocks: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Attach byte-converted citations to the matching text block in multi-step runs."""
    text_indices = [
        i
        for i, b in enumerate(blocks)
        if isinstance(b, Mapping) and b.get("type") == "text"
    ]
    if len(text_indices) < 2:
        return blocks
    first_idx = text_indices[0]
    first_blk = cast("dict[str, Any]", blocks[first_idx])
    raw_anns = first_blk.get("annotations")
    if not isinstance(raw_anns, list) or not raw_anns:
        return blocks

    first_txt = str(first_blk.get("text", ""))
    per_block_anns: dict[int, list[Any]] = {ti: [] for ti in text_indices}
    moved = False
    for ann in raw_anns:
        if not isinstance(ann, Mapping) or ann.get("type") != "citation":
            per_block_anns[first_idx].append(ann)
            continue
        cs = ann.get("start_index")
        ce = ann.get("end_index")
        cited = ann.get("cited_text")
        if (
            isinstance(cs, int)
            and isinstance(ce, int)
            and isinstance(cited, str)
            and cited
            and first_txt[cs:ce] != cited
        ):
            placed = False
            for cand_i in text_indices[1:]:
                cand_txt = str(blocks[cand_i].get("text", ""))
                if cand_txt[cs:ce] == cited:
                    per_block_anns[cand_i].append(ann)
                    moved = True
                    placed = True
                    break
            if not placed:
                per_block_anns[first_idx].append(ann)
        else:
            per_block_anns[first_idx].append(ann)

    if not moved:
        return blocks

    out = list(blocks)
    for ti in text_indices:
        orig_b = dict(cast("dict[str, Any]", out[ti]))
        anns_for_ti = per_block_anns[ti]
        if ti == first_idx:
            if anns_for_ti:
                orig_b["annotations"] = anns_for_ti
            else:
                orig_b.pop("annotations", None)
        elif anns_for_ti:
            existing_a = list(orig_b.get("annotations") or [])
            orig_b["annotations"] = [*existing_a, *anns_for_ti]
        out[ti] = cast("types.ContentBlock", orig_b)
    return out


def _translate_legacy_genai_message(message: AIMessage) -> list[types.ContentBlock]:
    """Delegate default `generateContent` messages/chunks unchanged to core."""
    base_blocks = _core_convert_to_v1(message)
    if not any(b.get("type") == "refusal" for b in base_blocks) and (
        refusal := _refusal_from_metadata(message.response_metadata)
    ):
        base_blocks = [*base_blocks, refusal]
    return base_blocks


def _translate_interactions_v0_message(
    message: AIMessage, *, is_chunk_streaming: bool
) -> list[types.ContentBlock]:
    """Translate `v0` Interactions messages/chunks to v1 blocks with tool/sig parity."""
    msg_for_core = message
    if (
        is_chunk_streaming
        and isinstance(message, AIMessageChunk)
        and message.tool_call_chunks
    ):
        # Prevent core from promoting streaming deltas to finalized `tool_call` blocks
        # (and dropping `id=None` `arguments_delta` chunks).
        msg_for_core = message.model_copy(
            update={"tool_calls": [], "tool_call_chunks": []}
        )

    base_blocks = _core_convert_to_v1(msg_for_core)

    raw_sigs = message.additional_kwargs.get(_FC_THOUGHT_SIGS_KEY)
    sig_map = raw_sigs if isinstance(raw_sigs, Mapping) else {}

    # Upgrade any Interactions-emitted blocks core wrapped as `non_standard`, preserve
    # `executable_code`/`code_execution_result` extras, merge inline `file_citation`s
    # that `grounding_metadata` replaced, and enrich `tool_call` blocks with signatures.
    upgraded: list[types.ContentBlock] = []
    src_exec_extras: dict[str, dict[str, Any]] = {}
    src_cer_extras: list[dict[str, Any]] = []
    src_text_file_cits: dict[int, list[types.Annotation]] = {}
    if isinstance(message.content, list):
        src_t_ord = 0
        for raw_c in message.content:
            if isinstance(raw_c, str):
                src_t_ord += 1
                continue
            if not isinstance(raw_c, Mapping):
                continue
            rc_type = raw_c.get("type")
            if rc_type == "text":
                raw_fc_anns = raw_c.get("annotations")
                if isinstance(raw_fc_anns, list):
                    rc_txt = str(raw_c.get("text", ""))
                    kept_fc: list[types.Annotation] = []
                    for ra in raw_fc_anns:
                        norm_a = _normalize_annotation_to_v1(ra, rc_txt)
                        if (
                            norm_a is not None
                            and norm_a.get("type") == "citation"
                            and isinstance(
                                cast("Mapping[str, Any]", norm_a).get("extras"), Mapping
                            )
                            and "file_citation_metadata"
                            in cast("Mapping[str, Any]", norm_a)["extras"]
                        ):
                            kept_fc.append(norm_a)
                    if kept_fc:
                        src_text_file_cits[src_t_ord] = kept_fc
                src_t_ord += 1
            elif (
                rc_type == "executable_code"
                and isinstance(raw_c.get("extras"), Mapping)
                and raw_c["extras"]
            ):
                src_exec_extras[str(raw_c.get("id", ""))] = copy.deepcopy(
                    dict(raw_c["extras"])
                )
            elif rc_type == "code_execution_result":
                cer_ex = (
                    copy.deepcopy(dict(raw_c["extras"]))
                    if isinstance(raw_c.get("extras"), Mapping) and raw_c["extras"]
                    else {}
                )
                src_cer_extras.append(cer_ex)

    cer_ord = 0
    dst_t_ord = 0
    for blk in base_blocks:
        raw_blk = cast("dict[str, Any]", blk)
        if raw_blk.get("type") == "non_standard" and isinstance(
            raw_blk.get("value"), Mapping
        ):
            inner = cast("Mapping[str, Any]", raw_blk["value"])
            # Leave legacy `image_url` fallback error cases untouched
            if inner.get("type") not in ("image_url", None):
                conv = _convert_interactions_dict_item(inner)
                if conv is not None:
                    if raw_blk.get("index") is not None and "index" not in conv:
                        conv = cast(
                            "types.ContentBlock", {**conv, "index": raw_blk["index"]}
                        )
                    upgraded.append(conv)
                    continue
        elif (
            raw_blk.get("type") == "server_tool_call"
            and raw_blk.get("name") == "code_interpreter"
            and str(raw_blk.get("id", "")) in src_exec_extras
            and "extras" not in raw_blk
        ):
            blk = cast(
                "types.ContentBlock",
                {**raw_blk, "extras": src_exec_extras[str(raw_blk.get("id", ""))]},
            )
        elif (
            raw_blk.get("type") == "server_tool_result"
            and isinstance(raw_blk.get("extras"), Mapping)
            and raw_blk["extras"].get("block_type") == "code_execution_result"
        ):
            extra_cer = src_cer_extras[cer_ord] if cer_ord < len(src_cer_extras) else {}
            cer_ord += 1
            if extra_cer:
                blk = cast(
                    "types.ContentBlock",
                    {
                        **raw_blk,
                        "extras": {**dict(raw_blk["extras"]), **extra_cer},
                    },
                )
        elif raw_blk.get("type") == "tool_call":
            tc_id = raw_blk.get("id")
            if tc_id and sig_map.get(tc_id):
                existing_ex = (
                    dict(raw_blk["extras"])
                    if isinstance(raw_blk.get("extras"), Mapping)
                    else {}
                )
                if "signature" not in existing_ex:
                    blk = cast(
                        "types.ContentBlock",
                        {
                            **raw_blk,
                            "extras": {**existing_ex, "signature": sig_map[tc_id]},
                        },
                    )
        elif raw_blk.get("type") == "text":
            cur_t_ord = dst_t_ord
            dst_t_ord += 1
            norm_anns: list[types.Annotation] = []
            if isinstance(raw_blk.get("annotations"), list):
                norm_anns = [
                    n
                    for a in cast("list[Any]", raw_blk["annotations"])
                    if (
                        n := _normalize_annotation_to_v1(
                            a, str(raw_blk.get("text", ""))
                        )
                    )
                    is not None
                ]
            extra_fcs = src_text_file_cits.get(cur_t_ord, [])
            for fc_ann in extra_fcs:
                if fc_ann not in norm_anns:
                    norm_anns.append(fc_ann)
            if norm_anns:
                blk = cast("types.ContentBlock", {**raw_blk, "annotations": norm_anns})
        upgraded.append(blk)

    upgraded = _relocate_multistep_citations(upgraded)

    if (
        is_chunk_streaming
        and isinstance(message, AIMessageChunk)
        and message.tool_call_chunks
    ):
        upgraded.extend(
            _build_tool_blocks_for_message(message, upgraded, is_chunk_streaming=True)
        )
    if not any(b.get("type") == "refusal" for b in upgraded) and (
        refusal := _refusal_from_metadata(message.response_metadata)
    ):
        upgraded.append(refusal)
    return upgraded


def translate_genai_content(message: AIMessage) -> list[types.ContentBlock]:
    """Translate a `google_genai` `AIMessage` into standard v1 content blocks.

    Args:
        message: The `AIMessage` to translate.

    Returns:
        Derived standard v1 `ContentBlock` dictionaries.
    """
    if message.response_metadata.get("output_version") == "interactions/v1":
        return _translate_interactions_v1_message(message, is_chunk_streaming=False)
    if _is_interactions_message(message):
        return _translate_interactions_v0_message(message, is_chunk_streaming=False)
    return _translate_legacy_genai_message(message)


def translate_genai_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Translate a `google_genai` `AIMessageChunk` into standard v1 content blocks.

    Args:
        message: The streaming `AIMessageChunk` to translate.

    Returns:
        Derived standard v1 `ContentBlock` (and `ToolCallChunk`) dictionaries.
    """
    if message.response_metadata.get("output_version") == "interactions/v1":
        return _translate_interactions_v1_message(
            message, is_chunk_streaming=message.chunk_position != "last"
        )
    if _is_interactions_message(message):
        return _translate_interactions_v0_message(
            message, is_chunk_streaming=message.chunk_position != "last"
        )
    return _translate_legacy_genai_message(message)


def _register_genai_translator() -> None:
    """Register the Interactions-aware `google_genai` content block translator."""
    register_translator(
        "google_genai", translate_genai_content, translate_genai_content_chunk
    )


_register_genai_translator()
