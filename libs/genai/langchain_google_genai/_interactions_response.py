"""Response parsing and sync/async execution for the Gemini Interactions API."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import uuid
import warnings
import wave
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, cast

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.messages.content import Citation, create_citation
from langchain_core.messages.tool import invalid_tool_call, tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

from langchain_google_genai._common import SafetySettingDict
from langchain_google_genai._function_utils import _ToolChoiceType
from langchain_google_genai._interactions_request import (
    _aprepare_interactions_request,
    _normalize_signature_b64,
    _prepare_interactions_request,
)
from langchain_google_genai.chat_models import (
    _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY,
    GoogleAuthenticationError,
    GoogleInteractionsAPIError,
    GoogleInteractionsContextOverflowError,
    GoogleInvalidRequestError,
    GoogleModelNotFoundError,
    GooglePermissionDeniedError,
    GoogleRateLimitError,
    GoogleTimeoutError,
    _acall_with_interactions_retry,
    _append_to_content,
    _call_with_interactions_retry,
    _convert_integer_like_floats,
    _finish_reason_fields,
    _get_scoped_interactions_client,
    _is_context_overflow_text,
    _is_gemini_3_or_later,
)

logger = logging.getLogger(__name__)

_NON_TERMINAL_STATUSES = frozenset({"in_progress", "queued"})


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read an attribute or dictionary key uniformly."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _dump_plain(obj: Any) -> Any:
    """Convert Pydantic SDK models into plain JSON-compatible structures."""
    if getattr(obj, "is_unknown", False) is True and hasattr(obj, "raw"):
        return _dump_plain(obj.raw)
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump(exclude_none=True)
        # Speakeasy `construct_unvalidated` wraps typeless JSON dicts inside a list as
        # the first all-optional `*Content` union arm (e.g. `ImageContent` in 2.25,
        # `AudioContent` in 2.34) with only default `type` + `model_extra` populated.
        model_fields = getattr(type(obj), "model_fields", None)
        declared_non_type = (
            set(model_fields) - {"type"} if isinstance(model_fields, Mapping) else set()
        )
        if (
            isinstance(dumped, dict)
            and type(obj).__name__.endswith("Content")
            and declared_non_type
            and not declared_non_type.intersection(dumped)
            and bool(getattr(obj, "model_extra", None))
        ):
            dumped.pop("type", None)
        return dumped
    if isinstance(obj, list):
        return [_dump_plain(x) for x in obj]
    if isinstance(obj, Mapping):
        return {k: _dump_plain(v) for k, v in obj.items() if v is not None}
    return obj


def _utf8_byte_offsets_to_char_offsets(
    text: str, start_byte: int | None, end_byte: int | None
) -> tuple[int, int, str]:
    """Convert UTF-8 byte offsets `[start_byte, end_byte)` into Python `str` offsets."""
    encoded = text.encode("utf-8")
    total_bytes = len(encoded)
    sb = max(0, min(int(start_byte) if start_byte is not None else 0, total_bytes))
    eb = max(
        sb,
        min(int(end_byte) if end_byte is not None else total_bytes, total_bytes),
    )
    char_start = len(encoded[:sb].decode("utf-8", errors="ignore"))
    char_end = len(encoded[:eb].decode("utf-8", errors="ignore"))
    return char_start, char_end, text[char_start:char_end]


def _format_status_errors(interaction: Any) -> tuple[str, list[str]]:
    """Extract readable messages and error codes from `interaction.errors` / steps."""
    messages: list[str] = []
    codes: list[str] = []
    for err in _get_field(interaction, "errors") or []:
        code = _get_field(err, "code")
        msg = _get_field(err, "message")
        if code:
            codes.append(str(code).lower())
        if code and msg:
            messages.append(f"[{code}] {msg}")
        elif msg:
            messages.append(str(msg))
        elif code:
            messages.append(str(code))
    for step in _get_field(interaction, "steps") or []:
        if _get_field(step, "type") == "model_output":
            step_err = _get_field(step, "error")
            if step_err is not None:
                s_code = _get_field(step_err, "code")
                s_msg = _get_field(step_err, "message")
                if s_code is not None:
                    codes.append(str(s_code).lower())
                if s_msg:
                    messages.append(str(s_msg))
    return "; ".join(messages), codes


def _raise_classified_interactions_error(
    codes: Sequence[str],
    summary: str,
    *,
    status: str = "failed",
    err_detail: str = "",
) -> NoReturn:
    """Raise a classified LangChain model exception from status/SSE error codes."""
    if _is_context_overflow_text(summary):
        raise GoogleInteractionsContextOverflowError(
            code=400,
            response_json={
                "error": {
                    "code": 400,
                    "message": summary,
                    "status": "INVALID_ARGUMENT",
                }
            },
            response=None,
        )

    joined_codes = " ".join(c.lower() for c in codes if c)
    if any(
        tok in joined_codes
        for tok in (
            "invalid_argument",
            "invalid_request",
            "bad_request",
            "unprocessable",
            "400",
            "422",
        )
    ):
        raise GoogleInvalidRequestError(summary)
    if any(
        tok in joined_codes
        for tok in ("unauthenticated", "unauthorized", "authentication", "401")
    ):
        raise GoogleAuthenticationError(summary)
    if any(tok in joined_codes for tok in ("permission_denied", "forbidden", "403")):
        raise GooglePermissionDeniedError(summary)
    if any(tok in joined_codes for tok in ("not_found", "404")):
        raise GoogleModelNotFoundError(summary)
    if any(
        tok in joined_codes
        for tok in (
            "resource_exhausted",
            "rate_limit",
            "too_many_requests",
            "quota",
            "429",
        )
    ):
        raise GoogleRateLimitError(summary)
    if any(
        tok in joined_codes
        for tok in ("deadline_exceeded", "timeout", "timed_out", "504", "408")
    ):
        raise GoogleTimeoutError(summary)

    if status == "cancelled" or any(
        tok in joined_codes for tok in ("cancelled", "canceled", "499")
    ):
        raise GoogleInteractionsAPIError(
            code=499,
            response_json={
                "error": {
                    "code": 499,
                    "message": summary,
                    "status": "CANCELLED",
                }
            },
            response=None,
        )

    lowered_detail = err_detail.lower()
    if any(
        tok in joined_codes
        for tok in (
            "safety",
            "prohibited_content",
            "blocklist",
            "recitation",
            "content_filter",
            "spii",
            "jailbreak",
        )
    ) or any(
        phrase in lowered_detail
        for phrase in (
            "blocked due to safety",
            "blocked for safety",
            "prohibited_content",
            "prohibited content",
        )
    ):
        raise GoogleInteractionsAPIError(
            code=400,
            response_json={
                "error": {
                    "code": 400,
                    "message": summary,
                    "status": "SAFETY",
                }
            },
            response=None,
        )

    if any(tok in joined_codes for tok in ("failed_precondition", "out_of_range")):
        raise GoogleInteractionsAPIError(
            code=400,
            response_json={
                "error": {
                    "code": 400,
                    "message": summary,
                    "status": "FAILED_PRECONDITION",
                }
            },
            response=None,
        )

    if any(tok in joined_codes for tok in ("unimplemented", "501")):
        raise GoogleInteractionsAPIError(
            code=501,
            response_json={
                "error": {
                    "code": 501,
                    "message": summary,
                    "status": "UNIMPLEMENTED",
                }
            },
            response=None,
        )

    raise GoogleInteractionsAPIError(
        code=500,
        response_json={
            "error": {
                "code": 500,
                "message": summary,
                "status": status.upper(),
            }
        },
        response=None,
    )


def _raise_for_interaction_status(interaction: Any, model_name: str) -> NoReturn:
    """Raise a classified LangChain model error for `failed`/`cancelled` status."""
    status = str(_get_field(interaction, "status") or "failed")
    iid = _get_field(interaction, "id") or "<stateless>"
    err_detail, codes = _format_status_errors(interaction)
    summary = (
        f"Interaction '{iid}' on model '{model_name}' {status}: {err_detail}"
        if err_detail
        else f"Interaction '{iid}' on model '{model_name}' {status}."
    )
    _raise_classified_interactions_error(
        codes, summary, status=status, err_detail=err_detail
    )


def _status_to_finish_reason(status: str | None) -> str:
    """Map an `Interaction.status` value to a standard Gemini `finish_reason`."""
    if status in ("incomplete", "budget_exceeded", "requires_continuation"):
        return "MAX_TOKENS"
    if status in _NON_TERMINAL_STATUSES:
        return str(status).upper()
    return "STOP"


def _parse_usage_metadata(usage: Any) -> UsageMetadata | None:
    """Translate `Interaction.usage` into LangChain `UsageMetadata`."""
    if usage is None:
        return None
    raw_in = _get_field(usage, "total_input_tokens")
    raw_tool = _get_field(usage, "total_tool_use_tokens")
    raw_out = _get_field(usage, "total_output_tokens")
    raw_thought = _get_field(usage, "total_thought_tokens")
    raw_cached = _get_field(usage, "total_cached_tokens")
    raw_total = _get_field(usage, "total_tokens")

    if all(
        v is None
        for v in (raw_in, raw_tool, raw_out, raw_thought, raw_cached, raw_total)
    ):
        return None

    prompt_tokens = int(raw_in or 0)
    input_tokens = prompt_tokens + int(raw_tool or 0)
    thought_tokens = int(raw_thought or 0)
    output_tokens = int(raw_out or 0) + thought_tokens
    # Cached tokens are a subset of `total_input_tokens`. With grounding (for
    # example `google_search`), Gemini can report a cached count above the
    # prompt. That count can't be trusted, and clamping it would report a full
    # cache hit, so drop it instead.
    cached_tokens = int(raw_cached or 0)
    if cached_tokens > prompt_tokens:
        cached_tokens = 0
    computed_sum = input_tokens + output_tokens
    total_tokens = (
        int(raw_total)
        if raw_total is not None and (int(raw_total) > 0 or computed_sum == 0)
        else computed_sum
    )

    in_details: InputTokenDetails = {"cache_read": cached_tokens}
    if thought_tokens > 0 or raw_thought is not None:
        out_details: OutputTokenDetails = {"reasoning": thought_tokens}
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_token_details=in_details,
            output_token_details=out_details,
        )
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details=in_details,
    )


def _get_audio_sample_rate(item: Any) -> Any:
    """Read `sample_rate`, falling back to `rate` without Speakeasy warnings."""
    sr = _get_field(item, "sample_rate")
    if sr is not None:
        return sr
    if isinstance(item, Mapping):
        return item.get("rate")
    d = getattr(item, "__dict__", None)
    if isinstance(d, Mapping) and d.get("rate") is not None:
        return d["rate"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return getattr(item, "rate", None)


def _decode_v0_audio_bytes(item: Any) -> bytes | None:
    """Decode an `AudioContent` block for `additional_kwargs['audio']`."""
    data_b64 = _get_field(item, "data")
    if not data_b64:
        return None
    raw_bytes = base64.b64decode(data_b64)
    mime = str(_get_field(item, "mime_type") or "audio/wav").lower()
    # Unary Gemini 3.8 TTS returns full WAV (`RIFF`), while raw PCM (`l16`/`pcm`)
    # needs a WAV container header for backwards compatibility with v0 audio output.
    if raw_bytes.startswith(b"RIFF") or mime not in ("audio/l16", "audio/pcm"):
        return raw_bytes
    sample_rate = int(_get_audio_sample_rate(item) or 24000)
    channels = int(_get_field(item, "channels") or 1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    return buf.getvalue()


def _merge_v0_audio_items(audio_items: Sequence[Any]) -> bytes | None:
    """Decode and concatenate one or more `AudioContent` items into WAV/raw bytes."""
    decoded = [b for it in audio_items if (b := _decode_v0_audio_bytes(it)) is not None]
    if not decoded:
        return None
    if len(decoded) == 1:
        return decoded[0]

    pcm_frames: list[bytes] = []
    nchannels = 1
    sampwidth = 2
    framerate = 24000
    all_wav = True
    for idx_a, wav_or_raw in enumerate(decoded):
        if not wav_or_raw.startswith(b"RIFF"):
            all_wav = False
            break
        try:
            with wave.open(io.BytesIO(wav_or_raw), "rb") as rf:
                if idx_a == 0:
                    nchannels = rf.getnchannels()
                    sampwidth = rf.getsampwidth()
                    framerate = rf.getframerate()
                pcm_frames.append(rf.readframes(rf.getnframes()))
        except wave.Error:
            all_wav = False
            break

    if not all_wav:
        return b"".join(decoded)

    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(b"".join(pcm_frames))
    return out_buf.getvalue()


def _parse_text_annotations(
    text: str,
    annotations: Sequence[Any] | None,
    *,
    chunk_index_map: dict[tuple[str, str, str], int],
    grounding_chunks: list[dict[str, Any]],
    grounding_supports: list[dict[str, Any]],
) -> list[Citation]:
    """Convert byte-indexed Interactions annotations into char-indexed `Citation`s."""
    if not annotations:
        return []
    citations: list[Citation] = []
    seg_groups: dict[tuple[int, int, str], list[int]] = {}

    def _register_chunk(kind: str, uri: str, title: str, place_id: str = "") -> int:
        key = (kind, uri, place_id or title)
        if key not in chunk_index_map:
            chunk_index_map[key] = len(grounding_chunks)
            if kind == "maps":
                m_obj: dict[str, Any] = {"uri": uri, "title": title}
                if place_id:
                    m_obj["placeId"] = place_id
                grounding_chunks.append({"maps": m_obj})
            else:
                grounding_chunks.append({"web": {"uri": uri, "title": title}})
        return chunk_index_map[key]

    for ann in annotations:
        atype = _get_field(ann, "type")
        if atype not in ("url_citation", "place_citation", "file_citation"):
            continue
        c_start, c_end, cited = _utf8_byte_offsets_to_char_offsets(
            text, _get_field(ann, "start_index"), _get_field(ann, "end_index")
        )
        if atype == "url_citation":
            url = str(_get_field(ann, "url") or "")
            title = str(_get_field(ann, "title") or "")
            citations.append(
                create_citation(
                    url=url or None,
                    title=title or None,
                    start_index=c_start,
                    end_index=c_end,
                    cited_text=cited,
                )
            )
            if url:
                cidx = _register_chunk("web", url, title)
                seg_groups.setdefault((c_start, c_end, cited), []).append(cidx)
        elif atype == "place_citation":
            p_url = str(_get_field(ann, "url") or "")
            p_name = str(_get_field(ann, "name") or "")
            p_id = str(_get_field(ann, "place_id") or "")
            extra_p: dict[str, Any] = {}
            if p_id:
                extra_p["place_id"] = p_id
            p_cit = (
                create_citation(
                    url=p_url or None,
                    title=p_name or None,
                    start_index=c_start,
                    end_index=c_end,
                    cited_text=cited,
                    google_ai_metadata=extra_p,
                )
                if extra_p
                else create_citation(
                    url=p_url or None,
                    title=p_name or None,
                    start_index=c_start,
                    end_index=c_end,
                    cited_text=cited,
                )
            )
            citations.append(p_cit)
            if p_url or p_id:
                cidx = _register_chunk("maps", p_url, p_name, p_id)
                seg_groups.setdefault((c_start, c_end, cited), []).append(cidx)
        elif atype == "file_citation":
            doc_uri = str(_get_field(ann, "document_uri") or "")
            fname = str(_get_field(ann, "file_name") or _get_field(ann, "source") or "")
            f_meta: dict[str, Any] = {"type": "file_citation"}
            for fk in ("page_number", "media_id", "source", "custom_metadata"):
                fv = _get_field(ann, fk)
                if fv is not None:
                    f_meta[fk] = _dump_plain(fv)
            citations.append(
                create_citation(
                    url=doc_uri or None,
                    title=fname or None,
                    start_index=c_start,
                    end_index=c_end,
                    cited_text=cited,
                    file_citation_metadata=f_meta,
                )
            )

    for (cs, ce, ctext), indices in seg_groups.items():
        dedup_indices = list(dict.fromkeys(indices))
        grounding_supports.append(
            {
                "segment": {
                    "start_index": cs,
                    "end_index": ce,
                    "text": ctext,
                },
                "grounding_chunk_indices": dedup_indices,
            }
        )
    return citations


def _parse_function_call_step(
    step: Any,
    *,
    tool_calls: list[Any],
    invalid_tool_calls: list[Any],
    additional_kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    """Parse a `function_call` step, routing malformed JSON to `invalid_tool_calls`."""
    raw_id = _get_field(step, "id")
    call_id = str(raw_id) if raw_id else str(uuid.uuid4())
    name = str(_get_field(step, "name") or "")
    raw_args = _get_field(step, "arguments")

    if raw_args is None:
        clean_args: dict[str, Any] = {}
    elif isinstance(raw_args, str):
        try:
            loaded = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            invalid_tool_calls.append(
                invalid_tool_call(name=name, args=raw_args, id=call_id, error=str(exc))
            )
            return call_id, None
        if not isinstance(loaded, Mapping):
            err_msg = f"Expected JSON object arguments, got {type(loaded).__name__}"
            invalid_tool_calls.append(
                invalid_tool_call(
                    name=name,
                    args=raw_args,
                    id=call_id,
                    error=err_msg,
                )
            )
            return call_id, None
        clean_args = _convert_integer_like_floats(dict(loaded))
    elif isinstance(raw_args, Mapping):
        clean_args = _convert_integer_like_floats(_dump_plain(dict(raw_args)))
    else:
        invalid_tool_calls.append(
            invalid_tool_call(
                name=name,
                args=str(raw_args),
                id=call_id,
                error=f"Expected dict arguments, got {type(raw_args).__name__}",
            )
        )
        return call_id, None

    try:
        dumped_args = json.dumps(clean_args, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        invalid_tool_calls.append(
            invalid_tool_call(name=name, args=str(raw_args), id=call_id, error=str(exc))
        )
        return call_id, None

    additional_kwargs["function_call"] = {"name": name, "arguments": dumped_args}
    tool_calls.append(tool_call(name=name, args=clean_args, id=call_id))
    return call_id, clean_args


def _slice_output_steps(steps: Sequence[Any]) -> list[Any]:
    """Return only assistant output steps following any echoed input steps."""
    cut = 0
    for idx, step in enumerate(steps):
        if _get_field(step, "type") in ("user_input", "function_result"):
            cut = idx + 1
    return list(steps[cut:])


def _interaction_to_result(
    interaction: Any,
    *,
    model_name: str | None = None,
    requested_store: bool = False,
    output_version: str | None = None,
    pydantic_schema: type | None = None,
    previous_interaction_id: str | None = None,
) -> ChatResult:
    """Convert a completed `Interaction` into a LangChain `ChatResult`."""
    eff_model = str(
        _get_field(interaction, "model")
        or _get_field(interaction, "agent")
        or model_name
        or ""
    )
    status = _get_field(interaction, "status")
    if status in ("failed", "cancelled"):
        _raise_for_interaction_status(interaction, eff_model or "unknown")

    is_ia_v1 = output_version == "interactions/v1"
    steps = _slice_output_steps(_get_field(interaction, "steps") or [])

    content: str | list[Any] | None = [] if is_ia_v1 else None
    additional_kwargs: dict[str, Any] = {}
    tool_calls: list[Any] = []
    invalid_tool_calls: list[Any] = []
    all_output_texts: list[str] = []

    web_search_queries: list[str] = []
    grounding_chunks: list[dict[str, Any]] = []
    grounding_supports: list[dict[str, Any]] = []
    chunk_index_map: dict[tuple[str, str, str], int] = {}
    maps_widget_token: str | None = None
    saw_grounding_step = False

    pending_sig: str | None = None
    current_segment_sig: str | None = None
    audio_items: list[Any] = []

    def _add_block(blk: Any) -> None:
        nonlocal content
        content = _append_to_content(content, blk)

    def _record_fc_sig(cid: str, sig: str) -> None:
        sig_map = additional_kwargs.setdefault(
            _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY, {}
        )
        sig_map[cid] = sig

    def _emit_pending_thought_for_server_call() -> None:
        nonlocal pending_sig, current_segment_sig
        if pending_sig:
            if is_ia_v1:
                _add_block(
                    {
                        "type": "reasoning",
                        "reasoning": "",
                        "extras": {"signature": pending_sig},
                    }
                )
            else:
                _add_block(
                    {"type": "thinking", "thinking": "", "signature": pending_sig}
                )
            pending_sig = None
        current_segment_sig = None

    for step in steps:
        stype = _get_field(step, "type")
        step_sig = _normalize_signature_b64(_get_field(step, "signature"))

        if stype == "thought":
            summary_list = _get_field(step, "summary") or []
            summary_texts: list[str] = []
            summary_images: list[dict[str, Any]] = []
            for s_item in summary_list:
                s_type = _get_field(s_item, "type")
                if s_type == "text":
                    t_val = _get_field(s_item, "text")
                    if isinstance(t_val, str) and t_val:
                        summary_texts.append(t_val)
                elif s_type == "image" and is_ia_v1:
                    img_d = _get_field(s_item, "data")
                    if img_d:
                        summary_images.append(
                            {
                                "type": "image",
                                "base64": img_d,
                                "mime_type": _get_field(s_item, "mime_type")
                                or "image/png",
                            }
                        )
            joined_thought = "".join(summary_texts)
            current_segment_sig = step_sig
            if joined_thought:
                if is_ia_v1:
                    r_blk: dict[str, Any] = {
                        "type": "reasoning",
                        "reasoning": joined_thought,
                    }
                    if step_sig:
                        r_blk["extras"] = {"signature": step_sig}
                    _add_block(r_blk)
                else:
                    th_blk: dict[str, Any] = {
                        "type": "thinking",
                        "thinking": joined_thought,
                    }
                    if step_sig:
                        th_blk["signature"] = step_sig
                    _add_block(th_blk)
                pending_sig = None
            else:
                pending_sig = step_sig
            for s_img in summary_images:
                _add_block(s_img)
            continue

        if stype == "model_output":
            for item in _get_field(step, "content") or []:
                itype = _get_field(item, "type")
                if itype == "text":
                    txt = str(_get_field(item, "text") or "")
                    all_output_texts.append(txt)
                    block_cits = _parse_text_annotations(
                        txt,
                        _get_field(item, "annotations"),
                        chunk_index_map=chunk_index_map,
                        grounding_chunks=grounding_chunks,
                        grounding_supports=grounding_supports,
                    )
                    attached_sig = pending_sig
                    pending_sig = None
                    if is_ia_v1:
                        ia_txt: dict[str, Any] = {"type": "text", "text": txt}
                        if block_cits:
                            ia_txt["annotations"] = block_cits
                        if attached_sig:
                            ia_txt["extras"] = {"signature": attached_sig}
                        _add_block(ia_txt)
                    else:
                        file_cits = [
                            cit
                            for cit in block_cits
                            if isinstance(cit.get("extras"), Mapping)
                            and "file_citation_metadata" in cit["extras"]
                        ]
                        if txt or attached_sig or file_cits:
                            v0_txt: dict[str, Any] = {"type": "text", "text": txt}
                            if file_cits:
                                v0_txt["annotations"] = file_cits
                            if attached_sig:
                                v0_txt["extras"] = {"signature": attached_sig}
                            if (
                                attached_sig
                                or file_cits
                                or _is_gemini_3_or_later(eff_model)
                                or isinstance(content, list)
                            ):
                                _add_block(v0_txt)
                            else:
                                _add_block(txt)

                elif itype == "image":
                    mime = str(_get_field(item, "mime_type") or "image/png")
                    img_b64 = _get_field(item, "data")
                    img_uri = _get_field(item, "uri")
                    attached_sig = pending_sig
                    pending_sig = None
                    if is_ia_v1:
                        ia_img: dict[str, Any] = {"type": "image", "mime_type": mime}
                        if img_b64:
                            ia_img["base64"] = img_b64
                        elif img_uri:
                            ia_img["url"] = str(img_uri)
                        if attached_sig:
                            ia_img["extras"] = {"signature": attached_sig}
                        _add_block(ia_img)
                    elif img_b64:
                        v0_img: dict[str, Any] = {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                        }
                        if attached_sig:
                            v0_img["extras"] = {"signature": attached_sig}
                        _add_block(v0_img)
                    elif img_uri:
                        _add_block(
                            {
                                "type": "file_data",
                                "file_uri": str(img_uri),
                                "mime_type": mime,
                            }
                        )

                elif itype == "audio":
                    aud_mime = str(_get_field(item, "mime_type") or "audio/wav")
                    aud_b64 = _get_field(item, "data")
                    aud_uri = _get_field(item, "uri")
                    if aud_b64:
                        audio_items.append(item)
                    if is_ia_v1:
                        ia_aud: dict[str, Any] = {
                            "type": "audio",
                            "mime_type": aud_mime,
                        }
                        if aud_b64:
                            ia_aud["base64"] = aud_b64
                        elif aud_uri:
                            ia_aud["url"] = str(aud_uri)
                        extras_a: dict[str, Any] = {}
                        raw_a_sr = _get_audio_sample_rate(item)
                        if raw_a_sr is not None:
                            extras_a["sample_rate"] = raw_a_sr
                        if _get_field(item, "channels") is not None:
                            extras_a["channels"] = _get_field(item, "channels")
                        if pending_sig:
                            extras_a["signature"] = pending_sig
                            pending_sig = None
                        if extras_a:
                            ia_aud["extras"] = extras_a
                        _add_block(ia_aud)
                    elif aud_uri and not aud_b64:
                        _add_block(
                            {
                                "type": "file_data",
                                "file_uri": str(aud_uri),
                                "mime_type": aud_mime,
                            }
                        )

                elif itype == "video":
                    vid_mime = str(_get_field(item, "mime_type") or "video/mp4")
                    vid_b64 = _get_field(item, "data")
                    vid_uri = _get_field(item, "uri")
                    if is_ia_v1 or vid_b64:
                        v_blk: dict[str, Any] = {"type": "video", "mime_type": vid_mime}
                        if vid_b64:
                            v_blk["base64"] = vid_b64
                        elif vid_uri:
                            v_blk["url"] = str(vid_uri)
                        _add_block(v_blk)
                    elif vid_uri:
                        _add_block(
                            {
                                "type": "file_data",
                                "file_uri": str(vid_uri),
                                "mime_type": vid_mime,
                            }
                        )

                elif itype == "document":
                    doc_mime = str(_get_field(item, "mime_type") or "application/pdf")
                    doc_b64 = _get_field(item, "data")
                    doc_uri = _get_field(item, "uri")
                    if is_ia_v1 or doc_b64:
                        f_blk: dict[str, Any] = {"type": "file", "mime_type": doc_mime}
                        if doc_b64:
                            f_blk["base64"] = doc_b64
                        elif doc_uri:
                            f_blk["url"] = str(doc_uri)
                        _add_block(f_blk)
                    elif doc_uri:
                        _add_block(
                            {
                                "type": "file_data",
                                "file_uri": str(doc_uri),
                                "mime_type": doc_mime,
                            }
                        )
            continue

        if stype == "function_call":
            fc_id, fc_args = _parse_function_call_step(
                step,
                tool_calls=tool_calls,
                invalid_tool_calls=invalid_tool_calls,
                additional_kwargs=additional_kwargs,
            )
            sig_for_fc = pending_sig or current_segment_sig
            pending_sig = None
            if sig_for_fc:
                _record_fc_sig(fc_id, sig_for_fc)
            if is_ia_v1 and fc_args is not None:
                tc_blk: dict[str, Any] = {
                    "type": "tool_call",
                    "id": fc_id,
                    "name": str(_get_field(step, "name") or ""),
                    "args": fc_args,
                }
                if sig_for_fc:
                    tc_blk["extras"] = {"signature": sig_for_fc}
                _add_block(tc_blk)
            continue

        if stype == "code_execution_call":
            _emit_pending_thought_for_server_call()
            args_ce = _get_field(step, "arguments")
            code = str(_get_field(args_ce, "code") or "") if args_ce else ""
            lang = (
                str(_get_field(args_ce, "language") or "python")
                if args_ce
                else "python"
            )
            cid = str(_get_field(step, "id") or uuid.uuid4())
            if is_ia_v1:
                stc: dict[str, Any] = {
                    "type": "server_tool_call",
                    "name": "code_interpreter",
                    "id": cid,
                    "args": {"code": code, "language": lang},
                    "extras": {"step_type": "code_execution_call"},
                }
                if step_sig:
                    stc["extras"]["signature"] = step_sig
                _add_block(stc)
            else:
                v0_ce: dict[str, Any] = {
                    "type": "executable_code",
                    "executable_code": code,
                    "language": lang,
                    "id": cid,
                }
                if step_sig:
                    v0_ce["extras"] = {"signature": step_sig}
                _add_block(v0_ce)
            continue

        if stype == "code_execution_result":
            current_segment_sig = None
            cid = str(_get_field(step, "call_id") or "")
            out_ce = str(_get_field(step, "result") or "")
            is_err = bool(_get_field(step, "is_error"))
            outcome = 2 if is_err else 1
            if is_ia_v1:
                str_ce: dict[str, Any] = {
                    "type": "server_tool_result",
                    "tool_call_id": cid,
                    "status": "error" if is_err else "success",
                    "output": out_ce,
                    "extras": {
                        "block_type": "code_execution_result",
                        "step_type": "code_execution_result",
                        "outcome": outcome,
                    },
                }
                if step_sig:
                    str_ce["extras"]["signature"] = step_sig
                _add_block(str_ce)
            else:
                v0_cer: dict[str, Any] = {
                    "type": "code_execution_result",
                    "code_execution_result": out_ce,
                    "outcome": outcome,
                    "tool_call_id": cid,
                }
                if step_sig:
                    v0_cer["extras"] = {"signature": step_sig}
                _add_block(v0_cer)
            continue

        if stype == "google_search_call":
            _emit_pending_thought_for_server_call()
            saw_grounding_step = True
            gs_args = _get_field(step, "arguments")
            queries = list(_get_field(gs_args, "queries") or []) if gs_args else []
            for q in queries:
                if isinstance(q, str) and q not in web_search_queries:
                    web_search_queries.append(q)
            gs_blk: dict[str, Any] = {
                "type": "server_tool_call",
                "name": "google_search",
                "id": str(_get_field(step, "id") or uuid.uuid4()),
                "args": {"queries": queries},
                "extras": {"step_type": "google_search_call"},
            }
            if _get_field(step, "search_type"):
                gs_blk["extras"]["search_type"] = _get_field(step, "search_type")
            if step_sig:
                gs_blk["extras"]["signature"] = step_sig
            if is_ia_v1:
                _add_block(gs_blk)
            continue

        if stype == "google_search_result":
            current_segment_sig = None
            saw_grounding_step = True
            gs_res = _dump_plain(_get_field(step, "result") or [])
            if is_ia_v1:
                gsr_blk: dict[str, Any] = {
                    "type": "server_tool_result",
                    "tool_call_id": str(_get_field(step, "call_id") or ""),
                    "status": "error" if _get_field(step, "is_error") else "success",
                    "output": gs_res,
                    "extras": {
                        "block_type": "google_search_result",
                        "step_type": "google_search_result",
                    },
                }
                if step_sig:
                    gsr_blk["extras"]["signature"] = step_sig
                _add_block(gsr_blk)
            continue

        if stype in (
            "url_context_call",
            "google_maps_call",
            "file_search_call",
            "mcp_server_tool_call",
            "retrieval_call",
            "processing_call",
        ):
            _emit_pending_thought_for_server_call()
            if stype == "google_maps_call":
                saw_grounding_step = True
            name_map = {
                "url_context_call": "url_context",
                "google_maps_call": "google_maps",
                "file_search_call": "file_search",
                "retrieval_call": "retrieval",
                "processing_call": "media_processing",
            }
            call_name = (
                str(_get_field(step, "name") or "mcp_server_tool")
                if stype == "mcp_server_tool_call"
                else name_map[stype]
            )
            raw_a = _get_field(step, "arguments")
            args_d = _dump_plain(raw_a) if raw_a is not None else {}
            extras_c: dict[str, Any] = {"step_type": stype}
            if stype == "mcp_server_tool_call" and _get_field(step, "server_name"):
                extras_c["server_name"] = _get_field(step, "server_name")
            if stype == "retrieval_call" and _get_field(step, "retrieval_type"):
                extras_c["retrieval_type"] = _get_field(step, "retrieval_type")
            if step_sig and stype != "mcp_server_tool_call":
                extras_c["signature"] = step_sig
            stc_other: dict[str, Any] = {
                "type": "server_tool_call",
                "name": call_name,
                "id": str(_get_field(step, "id") or uuid.uuid4()),
                "args": args_d if isinstance(args_d, dict) else {},
                "extras": extras_c,
            }
            if is_ia_v1 or stype in ("processing_call", "mcp_server_tool_call"):
                _add_block(stc_other)
            continue

        if stype in (
            "url_context_result",
            "google_maps_result",
            "file_search_result",
            "mcp_server_tool_result",
            "retrieval_result",
            "processing_result",
        ):
            current_segment_sig = None
            raw_r = _get_field(step, "result")
            res_dumped = _dump_plain(raw_r) if raw_r is not None else {}
            if stype == "google_maps_result":
                saw_grounding_step = True
                for gm_entry in raw_r or []:
                    w_tok = _get_field(gm_entry, "widget_context_token")
                    if w_tok and not maps_widget_token:
                        maps_widget_token = str(w_tok)
                    for pl in _get_field(gm_entry, "places") or []:
                        p_u = str(_get_field(pl, "url") or "")
                        p_n = str(_get_field(pl, "name") or "")
                        p_id = str(_get_field(pl, "place_id") or "")
                        key_m = ("maps", p_u, p_id or p_n)
                        if (p_u or p_id) and key_m not in chunk_index_map:
                            chunk_index_map[key_m] = len(grounding_chunks)
                            m_chunk: dict[str, Any] = {"uri": p_u, "title": p_n}
                            if p_id:
                                m_chunk["placeId"] = p_id
                            grounding_chunks.append({"maps": m_chunk})

            b_tag = "media_processing" if stype == "processing_result" else str(stype)
            extras_r: dict[str, Any] = {"block_type": b_tag, "step_type": stype}
            if stype == "mcp_server_tool_result":
                if _get_field(step, "name"):
                    extras_r["name"] = _get_field(step, "name")
                if _get_field(step, "server_name"):
                    extras_r["server_name"] = _get_field(step, "server_name")
            if step_sig and stype != "mcp_server_tool_result":
                extras_r["signature"] = step_sig
            str_other: dict[str, Any] = {
                "type": "server_tool_result",
                "tool_call_id": str(_get_field(step, "call_id") or ""),
                "status": "error" if _get_field(step, "is_error") else "success",
                "output": res_dumped,
                "extras": extras_r,
            }
            if is_ia_v1 or stype in ("processing_result", "mcp_server_tool_result"):
                _add_block(str_other)

    if audio_items:
        merged_audio = _merge_v0_audio_items(audio_items)
        if merged_audio is not None:
            additional_kwargs["audio"] = merged_audio

    # Preserve a trailing signature-only turn if no content/tool_calls consumed it
    if pending_sig and not tool_calls and content in (None, []):
        _add_block({"type": "text", "text": "", "extras": {"signature": pending_sig}})

    if content is None:
        content = [] if _is_gemini_3_or_later(eff_model) else ""

    if (
        not is_ia_v1
        and isinstance(content, list)
        and any(isinstance(it, Mapping) and "executable_code" in it for it in content)
    ):
        warnings.warn(
            "Output may vary each run: 'executable_code' is present; "
            "'code_execution_result' & 'image_url' may be absent for some queries.",
            stacklevel=2,
        )

    # Structured Pydantic parsing into additional_kwargs["parsed"] when no tool calls
    if pydantic_schema is not None and not tool_calls and not invalid_tool_calls:
        joined_out = "".join(all_output_texts).strip()
        if joined_out:
            try:
                if issubclass(pydantic_schema, BaseModel):
                    additional_kwargs["parsed"] = pydantic_schema.model_validate_json(
                        joined_out
                    )
                elif issubclass(pydantic_schema, BaseModelV1):
                    additional_kwargs["parsed"] = pydantic_schema.parse_raw(joined_out)
            except Exception as exc:
                logger.debug(
                    "Failed to parse structured Pydantic output for %s: %s",
                    pydantic_schema,
                    exc,
                )

    raw_iid = _get_field(interaction, "id")
    clean_iid = raw_iid if isinstance(raw_iid, str) and raw_iid else None
    stored_flag = bool(requested_store and clean_iid)
    finish_reason = _status_to_finish_reason(status)
    finish_reason_fields = _finish_reason_fields(
        finish_reason,
        has_tool_calls=bool(tool_calls or invalid_tool_calls),
    )

    response_metadata: dict[str, Any] = {
        "model_provider": "google_genai",
        "model_name": eff_model,
        "id": clean_iid,
        "interaction_id": clean_iid,
        "stored": stored_flag,
        "status": status or "completed",
        "finish_reason": finish_reason,
        **finish_reason_fields,
    }
    service_tier = _get_field(interaction, "service_tier")
    if service_tier is not None:
        response_metadata["service_tier"] = service_tier
    env_id = _get_field(interaction, "environment_id")
    if env_id is not None:
        response_metadata["environment_id"] = env_id
    raw_prev_iid = previous_interaction_id or _get_field(
        interaction, "previous_interaction_id"
    )
    if isinstance(raw_prev_iid, str) and raw_prev_iid.strip():
        response_metadata["previous_interaction_id"] = raw_prev_iid.strip()
    cont_ids = _get_field(interaction, "continuation_ids")
    if isinstance(cont_ids, list) and len(cont_ids) > 1:
        response_metadata["continuation_ids"] = [
            str(x) for x in cont_ids if isinstance(x, str) and x
        ]
    cont_tok = _get_field(interaction, "continuation_token")
    if (
        isinstance(cont_tok, str)
        and cont_tok
        and status in ("requires_continuation", "incomplete")
    ):
        response_metadata["continuation_token"] = cont_tok
    if is_ia_v1:
        response_metadata["output_version"] = "interactions/v1"

    generation_info: dict[str, Any] = {
        "finish_reason": finish_reason,
        **finish_reason_fields,
    }
    if eff_model:
        generation_info["model_name"] = eff_model

    if (
        saw_grounding_step
        or grounding_chunks
        or grounding_supports
        or web_search_queries
        or maps_widget_token
    ):
        gm: dict[str, Any] = {
            "web_search_queries": web_search_queries,
            "grounding_chunks": grounding_chunks,
            "grounding_supports": grounding_supports,
        }
        if maps_widget_token:
            gm["google_maps_widget_context_token"] = maps_widget_token
        response_metadata["grounding_metadata"] = gm
        generation_info["grounding_metadata"] = gm

    usage_meta = _parse_usage_metadata(_get_field(interaction, "usage"))
    msg_id = clean_iid or f"lc_{uuid.uuid4()}"

    ai_message = AIMessage(
        content=cast("str | list[str | dict[Any, Any]]", content),
        additional_kwargs=additional_kwargs,
        response_metadata=response_metadata,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
        usage_metadata=usage_meta,
        id=msg_id,
    )
    if output_version == "v1":
        v1_blocks = ai_message.content_blocks
        response_metadata_v1 = {**response_metadata, "output_version": "v1"}
        ai_message = ai_message.model_copy(
            update={
                "content": cast("list[str | dict[Any, Any]]", v1_blocks),
                "response_metadata": response_metadata_v1,
            }
        )
    return ChatResult(
        generations=[
            ChatGeneration(message=ai_message, generation_info=generation_info)
        ],
        llm_output={},
    )


def generate_interactions(
    model_instance: Any,
    messages: Sequence[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    *,
    tools: Sequence[Any] | None = None,
    functions: Sequence[Any] | None = None,
    safety_settings: SafetySettingDict | Sequence[Any] | None = None,
    tool_config: Mapping[str, Any] | Any = None,
    generation_config: Mapping[str, Any] | None = None,
    cached_content: str | None = None,
    tool_choice: _ToolChoiceType | bool | None = None,
    **kwargs: Any,
) -> ChatResult:
    """Execute a synchronous request (including background polling) via Interactions."""
    prep = _prepare_interactions_request(
        model_instance,
        messages,
        stop=stop,
        tools=tools,
        functions=functions,
        safety_settings=safety_settings,
        tool_config=tool_config,
        generation_config=generation_config,
        cached_content=cached_content,
        tool_choice=tool_choice,
        stream=False,
        **kwargs,
    )
    interactions_client, ephemeral_http = _get_scoped_interactions_client(
        model_instance,
        is_async=False,
        base_url_override=prep.base_url_override,
        client_args_override=prep.client_args_override,
        httpx_client_override=prep.httpx_client_override,
    )

    def _create_unary_sync() -> Any:
        created = interactions_client.create(**prep.create_kwargs)
        if not prep.background and _get_field(created, "status") in (
            "failed",
            "cancelled",
        ):
            eff_name = str(
                _get_field(created, "model")
                or _get_field(created, "agent")
                or prep.target_name
                or "unknown"
            )
            _raise_for_interaction_status(created, eff_name)
        return created

    try:
        interaction = _call_with_interactions_retry(
            _create_unary_sync,
            model_name=prep.target_name,
            max_retries=prep.max_retries,
            run_manager=run_manager,
        )

        if prep.poll and _get_field(interaction, "status") in _NON_TERMINAL_STATUSES:
            raw_iid = _get_field(interaction, "id")
            iid = raw_iid.strip() if isinstance(raw_iid, str) else ""
            if not iid:
                msg = f"Background interaction on '{prep.target_name}' returned no id."
                raise GoogleInteractionsAPIError(
                    code=500,
                    response_json={
                        "error": {"code": 500, "message": msg, "status": "INTERNAL"}
                    },
                    response=None,
                )
            deadline = (
                time.monotonic() + prep.timeout_seconds
                if prep.timeout_seconds is not None and prep.timeout_seconds > 0
                else None
            )
            while _get_field(interaction, "status") in _NON_TERMINAL_STATUSES:
                remaining: float | None = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        msg_t = (
                            f"Timed out waiting for background interaction '{iid}' "
                            f"on '{prep.target_name}' after {prep.timeout_seconds}s."
                        )
                        raise GoogleTimeoutError(msg_t)
                sleep_for = (
                    min(prep.polling_interval, remaining)
                    if remaining is not None
                    else prep.polling_interval
                )
                time.sleep(sleep_for)
                poll_kwargs: dict[str, Any] = {"id": str(iid)}
                if prep.api_version is not None:
                    poll_kwargs["api_version"] = prep.api_version
                if prep.extra_headers is not None:
                    poll_kwargs["extra_headers"] = prep.extra_headers
                if remaining is not None:
                    poll_kwargs["timeout"] = max(1.0, deadline - time.monotonic())  # type: ignore[operator]
                interaction = _call_with_interactions_retry(
                    interactions_client.get,
                    model_name=prep.target_name,
                    max_retries=prep.max_retries,
                    run_manager=run_manager,
                    **poll_kwargs,
                )
    finally:
        if ephemeral_http is not None:
            ephemeral_http.close()

    return _interaction_to_result(
        interaction,
        model_name=prep.target_name,
        requested_store=prep.requested_store,
        output_version=prep.output_version,
        pydantic_schema=prep.pydantic_schema,
        previous_interaction_id=prep.previous_interaction_id,
    )


async def agenerate_interactions(
    model_instance: Any,
    messages: Sequence[BaseMessage],
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    *,
    tools: Sequence[Any] | None = None,
    functions: Sequence[Any] | None = None,
    safety_settings: SafetySettingDict | Sequence[Any] | None = None,
    tool_config: Mapping[str, Any] | Any = None,
    generation_config: Mapping[str, Any] | None = None,
    cached_content: str | None = None,
    tool_choice: _ToolChoiceType | bool | None = None,
    **kwargs: Any,
) -> ChatResult:
    """Execute an async request (including background polling) via Interactions."""
    prep = await _aprepare_interactions_request(
        model_instance,
        messages,
        stop=stop,
        tools=tools,
        functions=functions,
        safety_settings=safety_settings,
        tool_config=tool_config,
        generation_config=generation_config,
        cached_content=cached_content,
        tool_choice=tool_choice,
        stream=False,
        **kwargs,
    )
    interactions_client, ephemeral_http = _get_scoped_interactions_client(
        model_instance,
        is_async=True,
        base_url_override=prep.base_url_override,
        client_args_override=prep.async_client_args_override,
        httpx_client_override=prep.httpx_async_client_override,
    )

    async def _create_unary_async() -> Any:
        created = await interactions_client.create(**prep.create_kwargs)
        if not prep.background and _get_field(created, "status") in (
            "failed",
            "cancelled",
        ):
            eff_name = str(
                _get_field(created, "model")
                or _get_field(created, "agent")
                or prep.target_name
                or "unknown"
            )
            _raise_for_interaction_status(created, eff_name)
        return created

    try:
        interaction = await _acall_with_interactions_retry(
            _create_unary_async,
            model_name=prep.target_name,
            max_retries=prep.max_retries,
            run_manager=run_manager,
        )

        if prep.poll and _get_field(interaction, "status") in _NON_TERMINAL_STATUSES:
            raw_iid = _get_field(interaction, "id")
            iid = raw_iid.strip() if isinstance(raw_iid, str) else ""
            if not iid:
                msg = f"Background interaction on '{prep.target_name}' returned no id."
                raise GoogleInteractionsAPIError(
                    code=500,
                    response_json={
                        "error": {"code": 500, "message": msg, "status": "INTERNAL"}
                    },
                    response=None,
                )
            deadline = (
                time.monotonic() + prep.timeout_seconds
                if prep.timeout_seconds is not None and prep.timeout_seconds > 0
                else None
            )
            while _get_field(interaction, "status") in _NON_TERMINAL_STATUSES:
                remaining: float | None = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        msg_t = (
                            f"Timed out waiting for background interaction '{iid}' "
                            f"on '{prep.target_name}' after {prep.timeout_seconds}s."
                        )
                        raise GoogleTimeoutError(msg_t)
                sleep_for = (
                    min(prep.polling_interval, remaining)
                    if remaining is not None
                    else prep.polling_interval
                )
                await asyncio.sleep(sleep_for)
                poll_kwargs: dict[str, Any] = {"id": str(iid)}
                if prep.api_version is not None:
                    poll_kwargs["api_version"] = prep.api_version
                if prep.extra_headers is not None:
                    poll_kwargs["extra_headers"] = prep.extra_headers
                if remaining is not None:
                    poll_kwargs["timeout"] = max(1.0, deadline - time.monotonic())  # type: ignore[operator]
                interaction = await _acall_with_interactions_retry(
                    interactions_client.get,
                    model_name=prep.target_name,
                    max_retries=prep.max_retries,
                    run_manager=run_manager,
                    **poll_kwargs,
                )
    finally:
        if ephemeral_http is not None:
            await ephemeral_http.aclose()

    return _interaction_to_result(
        interaction,
        model_name=prep.target_name,
        requested_store=prep.requested_store,
        output_version=prep.output_version,
        pydantic_schema=prep.pydantic_schema,
        previous_interaction_id=prep.previous_interaction_id,
    )


__all__ = [
    "_interaction_to_result",
    "_parse_usage_metadata",
    "_utf8_byte_offsets_to_char_offsets",
    "agenerate_interactions",
    "generate_interactions",
]
