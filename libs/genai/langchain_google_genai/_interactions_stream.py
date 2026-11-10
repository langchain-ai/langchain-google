"""SSE streaming state machine and executors for the Gemini Interactions API."""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import json
import logging
import os
import time
import uuid
import wave
from collections.abc import AsyncGenerator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

from langchain_google_genai._common import SafetySettingDict
from langchain_google_genai._function_utils import _ToolChoiceType
from langchain_google_genai._interactions_request import (
    _aprepare_interactions_request,
    _InteractionsPreparedRequest,
    _normalize_signature_b64,
    _prepare_interactions_request,
)
from langchain_google_genai._interactions_response import (
    _NON_TERMINAL_STATUSES,
    _dump_plain,
    _get_audio_sample_rate,
    _get_field,
    _interaction_to_result,
    _parse_function_call_step,
    _parse_text_annotations,
    _parse_usage_metadata,
    _raise_classified_interactions_error,
    _raise_for_interaction_status,
    _slice_output_steps,
    _status_to_finish_reason,
    agenerate_interactions,
    generate_interactions,
)
from langchain_google_genai.chat_models import (
    _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY,
    _HTTPX_HTTP_ERRORS,
    _HTTPX_TIMEOUT_ERRORS,
    _INTERACTIONS_CATCHABLE_ERRORS,
    GoogleConnectionError,
    GoogleInteractionsAPIError,
    GoogleTimeoutError,
    _acall_with_interactions_retry,
    _call_with_interactions_retry,
    _convert_integer_like_floats,
    _finish_reason_fields,
    _get_scoped_interactions_client,
    _handle_interactions_error,
)

logger = logging.getLogger(__name__)

_STREAM_CHUNK_TIMEOUT_ENV = "LANGCHAIN_GOOGLE_STREAM_CHUNK_TIMEOUT_S"
_DEFAULT_STREAM_CHUNK_TIMEOUT_S = 300.0

_SERVER_CALL_STEP_TYPES = frozenset(
    {
        "code_execution_call",
        "google_search_call",
        "url_context_call",
        "google_maps_call",
        "file_search_call",
        "mcp_server_tool_call",
        "retrieval_call",
        "processing_call",
    }
)
_SERVER_RESULT_STEP_TYPES = frozenset(
    {
        "code_execution_result",
        "google_search_result",
        "url_context_result",
        "google_maps_result",
        "file_search_result",
        "mcp_server_tool_result",
        "retrieval_result",
        "processing_result",
    }
)
_SERVER_NAME_MAP: dict[str, str] = {
    "url_context_call": "url_context",
    "google_maps_call": "google_maps",
    "file_search_call": "file_search",
    "retrieval_call": "retrieval",
    "processing_call": "media_processing",
}


def _resolve_stream_chunk_timeout_s() -> float | None:
    """Read async silence timeout from `LANGCHAIN_GOOGLE_STREAM_CHUNK_TIMEOUT_S`."""
    raw = os.environ.get(_STREAM_CHUNK_TIMEOUT_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_STREAM_CHUNK_TIMEOUT_S
    try:
        parsed = float(raw.strip())
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to %ss.",
            _STREAM_CHUNK_TIMEOUT_ENV,
            raw,
            _DEFAULT_STREAM_CHUNK_TIMEOUT_S,
        )
        return _DEFAULT_STREAM_CHUNK_TIMEOUT_S
    return parsed if parsed > 0 else None


def _wrap_audio_frames(
    raw_bytes: bytes, *, mime_type: str, sample_rate: int, channels: int
) -> bytes:
    """Wrap raw PCM/L16 bytes in a single WAV header when needed."""
    mime_l = mime_type.lower()
    if raw_bytes.startswith(b"RIFF") or mime_l not in ("audio/l16", "audio/pcm"):
        return raw_bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    return buf.getvalue()


@dataclass
class _StepSlot:
    """Mutable per-`event.index` step tracking slot during SSE streaming."""

    kind: str
    step_type: str
    reserved_idx: int | None = None
    emitted_block: bool = False
    signature: str | None = None
    # model_output tracking
    text_block_idx: int | None = None
    text_has_sig: bool = False
    text_buf: str = ""
    img_block_idx: int | None = None
    img_header_emitted: bool = False
    vid_block_idx: int | None = None
    vid_header_emitted: bool = False
    doc_block_idx: int | None = None
    doc_header_emitted: bool = False
    audio_b64_parts: list[str] = field(default_factory=list)
    audio_uri: str | None = None
    audio_mime: str = "audio/wav"
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    audio_sig: str | None = None
    audio_block_emitted: bool = False
    # function_call tracking
    tc_index: int = 0
    call_id: str = ""
    name: str = ""
    thought_event_idx: int | None = None
    saw_args: bool = False
    raw_args_buf: str = ""
    ia_v1_tc_emitted: bool = False
    # server tool tracking
    server_name: str = ""
    search_type: str | None = None
    retrieval_type: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    is_error: bool = False


class _InteractionsStreamProcessor:
    """Deterministic state machine translating Interactions SSE events into chunks."""

    def __init__(
        self,
        *,
        target_name: str,
        requested_store: bool,
        output_version: str | None,
        pydantic_schema: type | None,
        previous_interaction_id: str | None = None,
    ) -> None:
        self._target_name = target_name
        self._model_name = target_name
        self._requested_store = requested_store
        self._output_version = output_version
        self._is_ia_v1 = output_version == "interactions/v1"
        self._is_v1 = output_version == "v1"
        self._pydantic_schema = pydantic_schema
        self._previous_interaction_id = self._clean_id(previous_interaction_id)

        self._interaction_id: str | None = None
        self._fallback_id: str = f"lc_{uuid.uuid4()}"
        self._status: str | None = None
        self._service_tier: Any = None
        self._environment_id: Any = None

        self._emitted_model_name = False
        self._completed = False
        self._hop_started = False
        self._hop_index = 0
        self._hop_ids: list[str] = []
        self._pending_continuation_inter: Any = None
        self._continuation_token: str | None = None
        self._continuation_token_status: str | None = None
        self._accumulated_usage: UsageMetadata | None = None

        self._next_block_idx = 0
        self._next_tc_idx = 0
        self._slots: dict[int, _StepSlot] = {}
        self._fc_slots: list[_StepSlot] = []
        self._emitted_fc_sigs: set[str] = set()
        self._pending_fc_sigs: dict[str, str] = {}

        self._pending_sig: str | None = None
        self._current_segment_thought_sig: str | None = None
        self._latest_thought_event_idx: int | None = None
        self._last_nonempty_text_event_idx: int | None = None

        self._all_output_texts: list[str] = []
        self._audio_raw_chunks: list[bytes] = []
        self._audio_mime = "audio/wav"
        self._audio_sample_rate = 24000
        self._audio_channels = 1

        self._web_search_queries: list[str] = []
        self._grounding_chunks: list[dict[str, Any]] = []
        self._grounding_supports: list[dict[str, Any]] = []
        self._chunk_index_map: dict[tuple[str, str, str], int] = {}
        self._maps_widget_token: str | None = None
        self._saw_grounding_step = False
        self._latest_step_usage: Any = None

    def _alloc_block_idx(self) -> int:
        idx = self._next_block_idx
        self._next_block_idx += 1
        return idx

    def _clean_id(self, raw_id: Any) -> str | None:
        if isinstance(raw_id, str) and raw_id.strip():
            return raw_id.strip()
        return None

    def _make_chunk(
        self,
        *,
        content: list[dict[str, Any]] | None = None,
        tool_call_chunks: list[Any] | None = None,
        additional_kwargs: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
        generation_info: dict[str, Any] | None = None,
        usage_metadata: Any = None,
        chunk_position: Literal["last"] | None = None,
        include_header_meta: bool = False,
    ) -> ChatGenerationChunk:
        """Construct an `AIMessageChunk` avoiding duplicate string concatenation."""
        meta: dict[str, Any] = {
            "model_provider": "google_genai",
            "interaction_id": None,
        }
        if self._is_ia_v1:
            meta["output_version"] = "interactions/v1"

        if include_header_meta or not self._emitted_model_name:
            if not self._emitted_model_name and self._model_name:
                meta["model_name"] = self._model_name
                self._emitted_model_name = True

        if extra_meta:
            meta.update(extra_meta)

        add_kw = dict(additional_kwargs) if additional_kwargs else {}
        if self._pending_fc_sigs:
            sig_bucket = dict(
                add_kw.get(_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY) or {}
            )
            for cid, csig in self._pending_fc_sigs.items():
                if cid not in self._emitted_fc_sigs:
                    sig_bucket[cid] = csig
                    self._emitted_fc_sigs.add(cid)
            self._pending_fc_sigs.clear()
            if sig_bucket:
                add_kw[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY] = sig_bucket

        chunk_id = (
            (self._interaction_id or self._fallback_id)
            if chunk_position == "last"
            else self._fallback_id
        )
        msg = AIMessageChunk(
            content=cast("list[str | dict[Any, Any]]", list(content or [])),
            tool_call_chunks=list(tool_call_chunks or []),
            additional_kwargs=add_kw,
            response_metadata=meta,
            usage_metadata=usage_metadata,
            id=chunk_id,
            chunk_position=chunk_position,
        )
        if self._is_v1:
            v1_blocks: list[Any] = []
            for vb in msg.content_blocks:
                if (
                    isinstance(vb, Mapping)
                    and "index" not in vb
                    and vb.get("type") not in ("tool_call", "invalid_tool_call")
                ):
                    v1_blocks.append({**vb, "index": self._alloc_block_idx()})
                else:
                    v1_blocks.append(vb)
            msg = msg.model_copy(
                update={
                    "content": cast("list[str | dict[Any, Any]]", v1_blocks),
                    "response_metadata": {
                        **msg.response_metadata,
                        "output_version": "v1",
                    },
                }
            )
        return ChatGenerationChunk(message=msg, generation_info=generation_info or {})

    def _queue_fc_sig(self, call_id: str, sig: str) -> None:
        if call_id and sig and call_id not in self._emitted_fc_sigs:
            self._pending_fc_sigs[call_id] = sig

    def _emit_thought_text(self, slot: _StepSlot, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        bidx = (
            slot.reserved_idx
            if slot.reserved_idx is not None
            else self._alloc_block_idx()
        )
        slot.reserved_idx = bidx
        attach_sig: str | None = None
        if not slot.emitted_block and slot.signature:
            attach_sig = slot.signature
            if self._pending_sig == attach_sig:
                self._pending_sig = None
        slot.emitted_block = True
        if self._is_ia_v1:
            r_blk: dict[str, Any] = {
                "type": "reasoning",
                "reasoning": text,
                "index": bidx,
            }
            if attach_sig:
                r_blk["extras"] = {"signature": attach_sig}
            return r_blk
        th_blk: dict[str, Any] = {
            "type": "thinking",
            "thinking": text,
            "index": bidx,
        }
        if attach_sig:
            th_blk["signature"] = attach_sig
        return th_blk

    def _emit_signature_only_thought(self, sig: str, bidx: int) -> dict[str, Any]:
        """Build a summary-less signed thought block preceding a server tool step."""
        if self._is_ia_v1:
            return {
                "type": "reasoning",
                "reasoning": "",
                "extras": {"signature": sig},
                "index": bidx,
            }
        return {
            "type": "thinking",
            "thinking": "",
            "signature": sig,
            "index": bidx,
        }

    def _consume_pending_sig_for_server_call(self) -> list[dict[str, Any]]:
        """Flush a summary-less `ThoughtStep` signature before opening a server call."""
        if not self._pending_sig:
            self._current_segment_thought_sig = None
            self._latest_thought_event_idx = None
            return []
        sig = self._pending_sig
        self._pending_sig = None
        bidx: int | None = None
        if self._latest_thought_event_idx is not None:
            t_slot = self._slots.get(self._latest_thought_event_idx)
            if (
                t_slot is not None
                and t_slot.kind == "thought"
                and not t_slot.emitted_block
            ):
                t_slot.emitted_block = True
                bidx = t_slot.reserved_idx
        if bidx is None:
            bidx = self._alloc_block_idx()
        self._current_segment_thought_sig = None
        self._latest_thought_event_idx = None
        return [self._emit_signature_only_thought(sig, bidx)]

    @property
    def interaction_id(self) -> str | None:
        """Return the latest non-empty interaction ID observed on the stream."""
        return self._interaction_id

    @property
    def is_non_terminal_eof(self) -> bool:
        """Return True if SSE ended while status was still `in_progress`/`queued`."""
        return (
            not self._completed
            and self._pending_continuation_inter is None
            and self._status in _NON_TERMINAL_STATUSES
        )

    def _emit_model_content_item(
        self, event_idx: int, slot: _StepSlot, item: Any
    ) -> list[dict[str, Any]]:
        """Translate a `model_output` item or media delta into indexed blocks."""
        itype = _get_field(item, "type")
        blocks: list[dict[str, Any]] = []

        if itype == "text":
            txt = str(_get_field(item, "text") or "")
            raw_anns = _get_field(item, "annotations")
            if not txt and not raw_anns and not self._pending_sig:
                return blocks
            # Reset active non-text delta cursors so interleaved media -> text -> media
            # allocates fresh sequential block indices instead of merging back.
            slot.img_block_idx = None
            slot.img_header_emitted = False
            slot.vid_block_idx = None
            slot.vid_header_emitted = False
            slot.doc_block_idx = None
            slot.doc_header_emitted = False
            if slot.text_block_idx is None:
                slot.text_block_idx = self._alloc_block_idx()
            if txt:
                slot.text_buf += txt
                self._all_output_texts.append(txt)
                self._last_nonempty_text_event_idx = event_idx
            cits = _parse_text_annotations(
                slot.text_buf,
                raw_anns,
                chunk_index_map=self._chunk_index_map,
                grounding_chunks=self._grounding_chunks,
                grounding_supports=self._grounding_supports,
            )
            attached_sig = self._pending_sig
            self._pending_sig = None
            t_blk: dict[str, Any] = {
                "type": "text",
                "text": txt,
                "index": slot.text_block_idx,
            }
            if attached_sig and not slot.text_has_sig:
                t_blk["extras"] = {"signature": attached_sig}
                slot.text_has_sig = True
            if cits:
                if self._is_ia_v1 or self._is_v1:
                    t_blk["annotations"] = cits
                else:
                    file_cits = [
                        cit
                        for cit in cits
                        if isinstance(cit.get("extras"), Mapping)
                        and "file_citation_metadata" in cit["extras"]
                    ]
                    if file_cits:
                        t_blk["annotations"] = file_cits
            blocks.append(t_blk)
            return blocks

        # Any non-text item closes an open text block inside this model_output step so
        # `[text("A"), image, text("B")]` never collapses `"B"` into `"A"`.
        if itype in ("image", "audio", "video", "document"):
            slot.text_block_idx = None
            slot.text_buf = ""
            slot.text_has_sig = False

        if itype == "image":
            mime = str(_get_field(item, "mime_type") or "image/png")
            img_b64 = _get_field(item, "data")
            img_uri = _get_field(item, "uri")
            attached_sig = self._pending_sig
            self._pending_sig = None
            if img_b64:
                if slot.img_block_idx is None:
                    slot.img_block_idx = self._alloc_block_idx()
                first = not slot.img_header_emitted
                slot.img_header_emitted = True
                if self._is_ia_v1:
                    ia_img: dict[str, Any] = {
                        "type": "image",
                        "base64": str(img_b64),
                        "index": slot.img_block_idx,
                    }
                    if first:
                        ia_img["mime_type"] = mime
                    if attached_sig:
                        ia_img["extras"] = {"signature": attached_sig}
                    blocks.append(ia_img)
                else:
                    url_val = f"data:{mime};base64,{img_b64}" if first else str(img_b64)
                    v0_img: dict[str, Any] = {
                        "type": "image_url",
                        "image_url": {"url": url_val},
                        "index": slot.img_block_idx,
                    }
                    if attached_sig:
                        v0_img["extras"] = {"signature": attached_sig}
                    blocks.append(v0_img)
            elif img_uri:
                uidx = self._alloc_block_idx()
                if self._is_ia_v1:
                    ia_u: dict[str, Any] = {
                        "type": "image",
                        "url": str(img_uri),
                        "mime_type": mime,
                        "index": uidx,
                    }
                    if attached_sig:
                        ia_u["extras"] = {"signature": attached_sig}
                    blocks.append(ia_u)
                else:
                    blocks.append(
                        {
                            "type": "file_data",
                            "file_uri": str(img_uri),
                            "mime_type": mime,
                            "index": uidx,
                        }
                    )
            return blocks

        if itype == "audio":
            if _get_field(item, "mime_type"):
                slot.audio_mime = str(_get_field(item, "mime_type"))
                self._audio_mime = slot.audio_mime
            raw_sr = _get_audio_sample_rate(item)
            if raw_sr is not None:
                sr = int(raw_sr)
                slot.audio_sample_rate = sr
                self._audio_sample_rate = sr
            if _get_field(item, "channels") is not None:
                ch = int(_get_field(item, "channels"))
                slot.audio_channels = ch
                self._audio_channels = ch
            if self._pending_sig and not slot.audio_sig:
                slot.audio_sig = self._pending_sig
                self._pending_sig = None
            aud_b64 = _get_field(item, "data")
            aud_uri = _get_field(item, "uri")
            if aud_b64:
                s_b64 = str(aud_b64)
                slot.audio_b64_parts.append(s_b64)
                self._audio_raw_chunks.append(base64.b64decode(s_b64))
            elif aud_uri:
                slot.audio_uri = str(aud_uri)
            return blocks

        if itype in ("video", "document"):
            is_vid = itype == "video"
            def_m = "video/mp4" if is_vid else "application/pdf"
            blk_t = "video" if is_vid else "file"
            mime_v = str(_get_field(item, "mime_type") or def_m)
            d_b64 = _get_field(item, "data")
            d_uri = _get_field(item, "uri")
            if d_b64:
                attr_idx = "vid_block_idx" if is_vid else "doc_block_idx"
                attr_hdr = "vid_header_emitted" if is_vid else "doc_header_emitted"
                cur_idx = getattr(slot, attr_idx)
                if cur_idx is None:
                    cur_idx = self._alloc_block_idx()
                    setattr(slot, attr_idx, cur_idx)
                first_v = not getattr(slot, attr_hdr)
                setattr(slot, attr_hdr, True)
                vb: dict[str, Any] = {
                    "type": blk_t,
                    "base64": str(d_b64),
                    "index": cur_idx,
                }
                if first_v:
                    vb["mime_type"] = mime_v
                blocks.append(vb)
            elif d_uri:
                uidx_v = self._alloc_block_idx()
                if self._is_ia_v1:
                    blocks.append(
                        {
                            "type": blk_t,
                            "url": str(d_uri),
                            "mime_type": mime_v,
                            "index": uidx_v,
                        }
                    )
                else:
                    blocks.append(
                        {
                            "type": "file_data",
                            "file_uri": str(d_uri),
                            "mime_type": mime_v,
                            "index": uidx_v,
                        }
                    )
        return blocks

    def _open_server_slot(self, stype: str, step: Any) -> _StepSlot:
        raw_step_sig = _normalize_signature_b64(_get_field(step, "signature"))
        step_sig = (
            None
            if stype in ("mcp_server_tool_call", "mcp_server_tool_result")
            else raw_step_sig
        )
        self._current_segment_thought_sig = None
        self._latest_thought_event_idx = None
        raw_a = _get_field(step, "arguments")
        args_d = (
            _dump_plain(dict(raw_a))
            if isinstance(raw_a, Mapping)
            else (_dump_plain(raw_a) if raw_a is not None else {})
        )
        raw_id = _get_field(step, "id") or _get_field(step, "call_id")
        cid = (
            str(raw_id)
            if raw_id
            else (str(uuid.uuid4()) if stype in _SERVER_CALL_STEP_TYPES else "")
        )
        return _StepSlot(
            kind="server_tool",
            step_type=stype,
            signature=step_sig,
            call_id=cid,
            name=str(_get_field(step, "name") or ""),
            server_name=str(_get_field(step, "server_name") or ""),
            search_type=_get_field(step, "search_type"),
            retrieval_type=_get_field(step, "retrieval_type"),
            arguments=args_d if isinstance(args_d, dict) else {},
            result=_get_field(step, "result"),
            is_error=bool(_get_field(step, "is_error")),
        )

    def _apply_server_delta(self, slot: _StepSlot, delta: Any) -> None:
        dtype = str(_get_field(delta, "type") or slot.step_type)
        dsig = _normalize_signature_b64(_get_field(delta, "signature"))
        if dsig and slot.step_type not in (
            "mcp_server_tool_call",
            "mcp_server_tool_result",
        ):
            slot.signature = dsig
        if _get_field(delta, "name"):
            slot.name = str(_get_field(delta, "name"))
        if _get_field(delta, "server_name"):
            slot.server_name = str(_get_field(delta, "server_name"))
        if _get_field(delta, "is_error") is not None:
            slot.is_error = bool(_get_field(delta, "is_error"))

        raw_args = _get_field(delta, "arguments")
        if raw_args is not None:
            dumped_a = _dump_plain(raw_args)
            if isinstance(dumped_a, dict):
                if dtype == "code_execution_call" and "code" in dumped_a:
                    prev_code = str(slot.arguments.get("code") or "")
                    slot.arguments["code"] = prev_code + str(dumped_a.get("code") or "")
                    if dumped_a.get("language"):
                        slot.arguments["language"] = str(dumped_a["language"])
                else:
                    slot.arguments.update(dumped_a)

        raw_res = _get_field(delta, "result")
        if raw_res is not None:
            if dtype == "code_execution_result" and isinstance(raw_res, str):
                prev_r = str(slot.result or "")
                slot.result = prev_r + raw_res
            elif isinstance(raw_res, list) and isinstance(slot.result, list):
                slot.result = [*slot.result, *raw_res]
            else:
                slot.result = raw_res

    def _build_server_block(self, slot: _StepSlot) -> dict[str, Any] | None:
        """Format a completed server tool call/result slot matching unary output."""
        stype = slot.step_type
        sig = slot.signature

        if stype == "code_execution_call":
            code = str(slot.arguments.get("code") or "")
            lang = str(slot.arguments.get("language") or "python")
            bidx = self._alloc_block_idx()
            if self._is_ia_v1:
                stc_ce: dict[str, Any] = {
                    "type": "server_tool_call",
                    "name": "code_interpreter",
                    "id": slot.call_id or str(uuid.uuid4()),
                    "args": {"code": code, "language": lang},
                    "extras": {"step_type": stype},
                    "index": bidx,
                }
                if sig:
                    stc_ce["extras"]["signature"] = sig
                return stc_ce
            v0_ce: dict[str, Any] = {
                "type": "executable_code",
                "executable_code": code,
                "language": lang,
                "id": slot.call_id or str(uuid.uuid4()),
                "index": bidx,
            }
            if sig:
                v0_ce["extras"] = {"signature": sig}
            return v0_ce

        if stype == "code_execution_result":
            out_ce = str(slot.result or "")
            outcome = 2 if slot.is_error else 1
            bidx = self._alloc_block_idx()
            if self._is_ia_v1:
                str_ce: dict[str, Any] = {
                    "type": "server_tool_result",
                    "tool_call_id": slot.call_id,
                    "status": "error" if slot.is_error else "success",
                    "output": out_ce,
                    "extras": {
                        "block_type": stype,
                        "step_type": stype,
                        "outcome": outcome,
                    },
                    "index": bidx,
                }
                if sig:
                    str_ce["extras"]["signature"] = sig
                return str_ce
            v0_cer: dict[str, Any] = {
                "type": "code_execution_result",
                "code_execution_result": out_ce,
                "outcome": outcome,
                "tool_call_id": slot.call_id,
                "index": bidx,
            }
            if sig:
                v0_cer["extras"] = {"signature": sig}
            return v0_cer

        if stype == "google_search_call":
            self._saw_grounding_step = True
            queries = list(slot.arguments.get("queries") or [])
            for q in queries:
                if isinstance(q, str) and q not in self._web_search_queries:
                    self._web_search_queries.append(q)
            if not self._is_ia_v1:
                return None
            gs_blk: dict[str, Any] = {
                "type": "server_tool_call",
                "name": "google_search",
                "id": slot.call_id or str(uuid.uuid4()),
                "args": {"queries": queries},
                "extras": {"step_type": stype},
                "index": self._alloc_block_idx(),
            }
            if slot.search_type:
                gs_blk["extras"]["search_type"] = slot.search_type
            if sig:
                gs_blk["extras"]["signature"] = sig
            return gs_blk

        if stype == "google_search_result":
            self._saw_grounding_step = True
            if not self._is_ia_v1:
                return None
            gs_res = _dump_plain(slot.result or [])
            gsr_blk: dict[str, Any] = {
                "type": "server_tool_result",
                "tool_call_id": slot.call_id,
                "status": "error" if slot.is_error else "success",
                "output": gs_res,
                "extras": {"block_type": stype, "step_type": stype},
                "index": self._alloc_block_idx(),
            }
            if sig:
                gsr_blk["extras"]["signature"] = sig
            return gsr_blk

        if stype in _SERVER_CALL_STEP_TYPES:
            if stype == "google_maps_call":
                self._saw_grounding_step = True
            if not (
                self._is_ia_v1 or stype in ("processing_call", "mcp_server_tool_call")
            ):
                return None
            call_name = (
                slot.name or "mcp_server_tool"
                if stype == "mcp_server_tool_call"
                else _SERVER_NAME_MAP[stype]
            )
            extras_c: dict[str, Any] = {"step_type": stype}
            if stype == "mcp_server_tool_call" and slot.server_name:
                extras_c["server_name"] = slot.server_name
            if stype == "retrieval_call" and slot.retrieval_type:
                extras_c["retrieval_type"] = slot.retrieval_type
            if sig:
                extras_c["signature"] = sig
            return {
                "type": "server_tool_call",
                "name": call_name,
                "id": slot.call_id or str(uuid.uuid4()),
                "args": dict(slot.arguments),
                "extras": extras_c,
                "index": self._alloc_block_idx(),
            }

        if stype in _SERVER_RESULT_STEP_TYPES:
            raw_r = slot.result
            if stype == "google_maps_result":
                self._saw_grounding_step = True
                for gm_entry in raw_r or []:
                    w_tok = _get_field(gm_entry, "widget_context_token")
                    if w_tok and not self._maps_widget_token:
                        self._maps_widget_token = str(w_tok)
                    for pl in _get_field(gm_entry, "places") or []:
                        p_u = str(_get_field(pl, "url") or "")
                        p_n = str(_get_field(pl, "name") or "")
                        p_id = str(_get_field(pl, "place_id") or "")
                        key_m = ("maps", p_u, p_id or p_n)
                        if (p_u or p_id) and key_m not in self._chunk_index_map:
                            self._chunk_index_map[key_m] = len(self._grounding_chunks)
                            m_c: dict[str, Any] = {"uri": p_u, "title": p_n}
                            if p_id:
                                m_c["placeId"] = p_id
                            self._grounding_chunks.append({"maps": m_c})

            if not (
                self._is_ia_v1
                or stype in ("processing_result", "mcp_server_tool_result")
            ):
                return None
            res_dumped = _dump_plain(raw_r) if raw_r is not None else {}
            b_tag = "media_processing" if stype == "processing_result" else stype
            extras_r: dict[str, Any] = {"block_type": b_tag, "step_type": stype}
            if stype == "mcp_server_tool_result":
                if slot.name:
                    extras_r["name"] = slot.name
                if slot.server_name:
                    extras_r["server_name"] = slot.server_name
            if sig:
                extras_r["signature"] = sig
            return {
                "type": "server_tool_result",
                "tool_call_id": slot.call_id,
                "status": "error" if slot.is_error else "success",
                "output": res_dumped,
                "extras": extras_r,
                "index": self._alloc_block_idx(),
            }
        return None

    def _finalize_slot(self, slot: _StepSlot) -> list[ChatGenerationChunk]:
        """Flush buffered 0-arg tool calls, inline `ia/v1` tools, or server/audio."""
        chunks: list[ChatGenerationChunk] = []
        if slot.kind == "function_call":
            if not slot.saw_args:
                slot.saw_args = True
                slot.raw_args_buf = "{}"
                chunks.append(
                    self._make_chunk(
                        tool_call_chunks=[
                            tool_call_chunk(
                                name=None, args="{}", id=None, index=slot.tc_index
                            )
                        ]
                    )
                )
            if self._is_ia_v1 and not slot.ia_v1_tc_emitted:
                slot.ia_v1_tc_emitted = True
                try:
                    loaded_fc = json.loads(slot.raw_args_buf or "{}")
                except json.JSONDecodeError:
                    loaded_fc = None
                if isinstance(loaded_fc, Mapping):
                    clean_fc = _convert_integer_like_floats(dict(loaded_fc))
                    bidx_fc = (
                        slot.reserved_idx
                        if slot.reserved_idx is not None
                        else self._alloc_block_idx()
                    )
                    tc_blk: dict[str, Any] = {
                        "type": "tool_call",
                        "id": slot.call_id,
                        "name": slot.name,
                        "args": clean_fc,
                        "index": bidx_fc,
                    }
                    if slot.signature:
                        tc_blk["extras"] = {"signature": slot.signature}
                    chunks.append(self._make_chunk(content=[tc_blk]))
            return chunks

        if slot.kind == "server_tool" and not slot.emitted_block:
            slot.emitted_block = True
            srv_blk = self._build_server_block(slot)
            if srv_blk is not None:
                chunks.append(self._make_chunk(content=[srv_blk]))
        elif (
            slot.kind == "model_output"
            and not slot.audio_block_emitted
            and (slot.audio_b64_parts or slot.audio_uri)
        ):
            slot.audio_block_emitted = True
            if self._is_ia_v1:
                ia_aud: dict[str, Any] = {
                    "type": "audio",
                    "mime_type": slot.audio_mime,
                    "index": self._alloc_block_idx(),
                }
                if slot.audio_b64_parts:
                    merged_raw = b"".join(
                        base64.b64decode(p) for p in slot.audio_b64_parts
                    )
                    ia_aud["base64"] = base64.b64encode(merged_raw).decode("ascii")
                elif slot.audio_uri:
                    ia_aud["url"] = slot.audio_uri
                extras_a: dict[str, Any] = {}
                if slot.audio_sample_rate is not None:
                    extras_a["sample_rate"] = slot.audio_sample_rate
                if slot.audio_channels is not None:
                    extras_a["channels"] = slot.audio_channels
                if slot.audio_sig:
                    extras_a["signature"] = slot.audio_sig
                if extras_a:
                    ia_aud["extras"] = extras_a
                chunks.append(self._make_chunk(content=[ia_aud]))
            elif slot.audio_uri and not slot.audio_b64_parts:
                chunks.append(
                    self._make_chunk(
                        content=[
                            {
                                "type": "file_data",
                                "file_uri": slot.audio_uri,
                                "mime_type": slot.audio_mime,
                                "index": self._alloc_block_idx(),
                            }
                        ]
                    )
                )
        return chunks

    def _accumulate_hop_usage(self, raw_usage: Any) -> None:
        parsed = _parse_usage_metadata(raw_usage)
        if parsed is not None:
            self._accumulated_usage = (
                parsed
                if self._accumulated_usage is None
                else add_usage(self._accumulated_usage, parsed)
            )

    def _reset_hop_slots(self) -> list[ChatGenerationChunk]:
        """Finalize and clear per-hop step slots before a new continuation hop."""
        flushed: list[ChatGenerationChunk] = []
        for s_idx in sorted(self._slots):
            flushed.extend(self._finalize_slot(self._slots[s_idx]))
        self._slots.clear()
        self._latest_thought_event_idx = None
        self._last_nonempty_text_event_idx = None
        self._current_segment_thought_sig = None
        self._pending_sig = None
        return flushed

    def process_event(self, event: Any) -> list[ChatGenerationChunk]:
        """Process one `InteractionSSEEvent` and return `ChatGenerationChunk`s."""
        etype = _get_field(event, "event_type")

        if etype == "interaction.created":
            flushed_prev: list[ChatGenerationChunk] = []
            was_started = self._hop_started
            if was_started:
                self._hop_index += 1
                self._pending_continuation_inter = None
                self._continuation_token = None
                self._continuation_token_status = None
                if self._slots:
                    flushed_prev.extend(self._reset_hop_slots())
                if self._latest_step_usage is not None:
                    self._accumulate_hop_usage(self._latest_step_usage)
                    self._latest_step_usage = None
            self._hop_started = True
            self._completed = False

            inter = _get_field(event, "interaction")
            cid = self._clean_id(_get_field(inter, "id"))
            if cid:
                self._interaction_id = cid
                if cid not in self._hop_ids:
                    self._hop_ids.append(cid)
            if self._hop_index == 0 and self._previous_interaction_id is None:
                prev_from_ev = self._clean_id(
                    _get_field(inter, "previous_interaction_id")
                )
                if prev_from_ev:
                    self._previous_interaction_id = prev_from_ev
            m_name = _get_field(inter, "model") or _get_field(inter, "agent")
            if isinstance(m_name, str) and m_name:
                self._model_name = m_name
            st = _get_field(inter, "status")
            if isinstance(st, str) and st:
                self._status = st
            if st in ("failed", "cancelled"):
                _raise_for_interaction_status(inter, self._model_name)
            if _get_field(inter, "service_tier") is not None:
                self._service_tier = _get_field(inter, "service_tier")
            if _get_field(inter, "environment_id") is not None:
                self._environment_id = _get_field(inter, "environment_id")
            if was_started:
                return flushed_prev
            return [self._make_chunk(include_header_meta=True)]

        if etype == "interaction.status_update":
            self._hop_started = True
            upd_id = self._clean_id(_get_field(event, "interaction_id"))
            if upd_id:
                self._interaction_id = upd_id
                if upd_id not in self._hop_ids:
                    self._hop_ids.append(upd_id)
            st_u = _get_field(event, "status")
            if isinstance(st_u, str) and st_u:
                self._status = st_u
            tok_u = self._clean_id(_get_field(event, "continuation_token"))
            if tok_u:
                self._continuation_token = tok_u
                self._continuation_token_status = self._status
            usage_u = _get_field(event, "usage")
            if usage_u is not None:
                self._latest_step_usage = usage_u
            if st_u in ("failed", "cancelled"):
                _raise_for_interaction_status(
                    {
                        "id": self._interaction_id,
                        "status": st_u,
                        "errors": _get_field(event, "errors"),
                    },
                    self._model_name,
                )
            return []

        if etype == "error":
            err = _get_field(event, "error")
            e_code = str(_get_field(err, "code") or "")
            e_msg = str(_get_field(err, "message") or "Unknown stream error")
            iid_lbl = self._interaction_id or "<stateless>"
            detail = f"[{e_code}] {e_msg}" if e_code else e_msg
            summary = (
                f"Interaction '{iid_lbl}' on model '{self._model_name}' "
                f"failed: {detail}"
            )
            _raise_classified_interactions_error(
                [e_code] if e_code else [],
                summary,
                status="failed",
                err_detail=detail,
            )

        if etype == "step.start":
            self._hop_started = True
            idx = int(_get_field(event, "index") or 0)
            step = _get_field(event, "step")
            stype = str(_get_field(step, "type") or "")

            if stype in ("user_input", "function_result"):
                self._slots[idx] = _StepSlot(
                    kind="input", step_type=stype, emitted_block=True
                )
                return []

            if stype == "thought":
                self._current_segment_thought_sig = None
                self._latest_thought_event_idx = idx
                step_sig = _normalize_signature_b64(_get_field(step, "signature"))
                t_slot = _StepSlot(
                    kind="thought",
                    step_type="thought",
                    reserved_idx=self._alloc_block_idx(),
                    signature=step_sig,
                )
                self._slots[idx] = t_slot
                init_blocks: list[dict[str, Any]] = []
                for s_item in _get_field(step, "summary") or []:
                    s_t = _get_field(s_item, "type")
                    if s_t == "text":
                        tb = self._emit_thought_text(
                            t_slot, str(_get_field(s_item, "text") or "")
                        )
                        if tb is not None:
                            init_blocks.append(tb)
                    elif (
                        s_t == "image" and self._is_ia_v1 and _get_field(s_item, "data")
                    ):
                        init_blocks.append(
                            {
                                "type": "image",
                                "base64": str(_get_field(s_item, "data")),
                                "mime_type": str(
                                    _get_field(s_item, "mime_type") or "image/png"
                                ),
                                "index": self._alloc_block_idx(),
                            }
                        )
                if step_sig:
                    self._current_segment_thought_sig = step_sig
                    if not t_slot.emitted_block:
                        self._pending_sig = step_sig
                return [self._make_chunk(content=init_blocks)] if init_blocks else []

            if stype == "model_output":
                m_slot = _StepSlot(kind="model_output", step_type="model_output")
                self._slots[idx] = m_slot
                out_blocks: list[dict[str, Any]] = []
                for item in _get_field(step, "content") or []:
                    out_blocks.extend(self._emit_model_content_item(idx, m_slot, item))
                return [self._make_chunk(content=out_blocks)] if out_blocks else []

            if stype == "function_call":
                tc_i = self._next_tc_idx
                self._next_tc_idx += 1
                raw_fid = _get_field(step, "id")
                call_id = str(raw_fid) if raw_fid else f"call_{uuid.uuid4().hex[:16]}"
                fname = str(_get_field(step, "name") or "")
                raw_a = _get_field(step, "arguments")
                init_args = ""
                saw_a = False
                if isinstance(raw_a, str) and raw_a:
                    init_args = raw_a
                    saw_a = True
                elif isinstance(raw_a, Mapping) and len(raw_a) > 0:
                    clean_a = _convert_integer_like_floats(_dump_plain(dict(raw_a)))
                    init_args = json.dumps(clean_a, ensure_ascii=False)
                    saw_a = True

                enclosing_sig = self._pending_sig or self._current_segment_thought_sig
                self._pending_sig = None
                if enclosing_sig:
                    self._queue_fc_sig(call_id, enclosing_sig)

                fc_slot = _StepSlot(
                    kind="function_call",
                    step_type="function_call",
                    reserved_idx=self._alloc_block_idx() if self._is_ia_v1 else None,
                    emitted_block=True,
                    signature=enclosing_sig,
                    tc_index=tc_i,
                    call_id=call_id,
                    name=fname,
                    thought_event_idx=self._latest_thought_event_idx,
                    saw_args=saw_a,
                    raw_args_buf=init_args,
                )
                self._slots[idx] = fc_slot
                self._fc_slots.append(fc_slot)
                return [
                    self._make_chunk(
                        tool_call_chunks=[
                            tool_call_chunk(
                                name=fname,
                                args=init_args,
                                id=call_id,
                                index=tc_i,
                            )
                        ]
                    )
                ]

            if stype in _SERVER_CALL_STEP_TYPES or stype in _SERVER_RESULT_STEP_TYPES:
                pre_blks = (
                    self._consume_pending_sig_for_server_call()
                    if stype in _SERVER_CALL_STEP_TYPES
                    else []
                )
                self._slots[idx] = self._open_server_slot(stype, step)
                return [self._make_chunk(content=pre_blks)] if pre_blks else []
            return []

        if etype == "step.delta":
            self._hop_started = True
            idx = int(_get_field(event, "index") or 0)
            existing_slot = self._slots.get(idx)
            if existing_slot is not None and existing_slot.kind == "input":
                return []
            delta = _get_field(event, "delta")
            dtype = str(_get_field(delta, "type") or "")

            if dtype == "text":
                slot = self._slots.get(idx)
                if slot is None:
                    slot = _StepSlot(kind="model_output", step_type="model_output")
                    self._slots[idx] = slot
                t_blocks = self._emit_model_content_item(idx, slot, delta)
                return [self._make_chunk(content=t_blocks)] if t_blocks else []

            if dtype == "text_annotation_delta":
                slot = self._slots.get(idx)
                target_slot = slot
                if (target_slot is None or not target_slot.text_buf) and (
                    self._last_nonempty_text_event_idx is not None
                ):
                    target_slot = self._slots.get(self._last_nonempty_text_event_idx)
                ref_text = target_slot.text_buf if target_slot is not None else ""
                cits = _parse_text_annotations(
                    ref_text,
                    _get_field(delta, "annotations"),
                    chunk_index_map=self._chunk_index_map,
                    grounding_chunks=self._grounding_chunks,
                    grounding_supports=self._grounding_supports,
                )
                emit_cits = (
                    cits
                    if (self._is_ia_v1 or self._is_v1)
                    else [
                        cit
                        for cit in cits
                        if isinstance(cit.get("extras"), Mapping)
                        and "file_citation_metadata" in cit["extras"]
                    ]
                )
                if (
                    emit_cits
                    and target_slot is not None
                    and target_slot.text_block_idx is not None
                ):
                    return [
                        self._make_chunk(
                            content=[
                                {
                                    "type": "text",
                                    "text": "",
                                    "annotations": emit_cits,
                                    "index": target_slot.text_block_idx,
                                }
                            ]
                        )
                    ]
                return []

            if dtype == "thought_summary":
                slot = self._slots.get(idx)
                if slot is None:
                    slot = _StepSlot(
                        kind="thought",
                        step_type="thought",
                        reserved_idx=self._alloc_block_idx(),
                    )
                    self._slots[idx] = slot
                d_content = _get_field(delta, "content")
                c_type = _get_field(d_content, "type")
                if c_type == "text":
                    tb = self._emit_thought_text(
                        slot, str(_get_field(d_content, "text") or "")
                    )
                    return [self._make_chunk(content=[tb])] if tb is not None else []
                if (
                    c_type == "image"
                    and self._is_ia_v1
                    and _get_field(d_content, "data")
                ):
                    return [
                        self._make_chunk(
                            content=[
                                {
                                    "type": "image",
                                    "base64": str(_get_field(d_content, "data")),
                                    "mime_type": str(
                                        _get_field(d_content, "mime_type")
                                        or "image/png"
                                    ),
                                    "index": self._alloc_block_idx(),
                                }
                            ]
                        )
                    ]
                return []

            if dtype == "thought_signature":
                sig = _normalize_signature_b64(_get_field(delta, "signature"))
                if not sig:
                    return []
                slot = self._slots.get(idx)
                if slot is None:
                    slot = _StepSlot(
                        kind="thought",
                        step_type="thought",
                        reserved_idx=self._alloc_block_idx(),
                    )
                    self._slots[idx] = slot
                slot.signature = sig
                if self._latest_thought_event_idx in (None, idx):
                    self._current_segment_thought_sig = sig

                # Backfill any already-opened function_calls scoped to this thought step
                matched_fc = False
                for fcs in self._fc_slots:
                    if fcs.thought_event_idx == idx:
                        matched_fc = True
                        fcs.signature = sig
                        self._queue_fc_sig(fcs.call_id, sig)

                if slot.emitted_block and slot.reserved_idx is not None:
                    sig_blk: dict[str, Any] = (
                        {
                            "type": "reasoning",
                            "reasoning": "",
                            "extras": {"signature": sig},
                            "index": slot.reserved_idx,
                        }
                        if self._is_ia_v1
                        else {
                            "type": "thinking",
                            "thinking": "",
                            "signature": sig,
                            "index": slot.reserved_idx,
                        }
                    )
                    return [self._make_chunk(content=[sig_blk])]

                if matched_fc:
                    return [self._make_chunk()] if self._pending_fc_sigs else []

                # Late signature for an already-started text step without a signature
                later_text_slot: _StepSlot | None = None
                for s_idx, cand in self._slots.items():
                    if (
                        s_idx > idx
                        and cand.kind == "model_output"
                        and cand.text_block_idx is not None
                        and not cand.text_has_sig
                    ):
                        later_text_slot = cand
                        break
                if (
                    later_text_slot is not None
                    and later_text_slot.text_block_idx is not None
                ):
                    later_text_slot.text_has_sig = True
                    return [
                        self._make_chunk(
                            content=[
                                {
                                    "type": "text",
                                    "text": "",
                                    "extras": {"signature": sig},
                                    "index": later_text_slot.text_block_idx,
                                }
                            ]
                        )
                    ]

                if any(
                    s_idx > idx and cand.kind == "server_tool"
                    for s_idx, cand in self._slots.items()
                ):
                    bidx_late = (
                        slot.reserved_idx
                        if slot.reserved_idx is not None
                        else self._alloc_block_idx()
                    )
                    slot.reserved_idx = bidx_late
                    slot.emitted_block = True
                    return [
                        self._make_chunk(
                            content=[self._emit_signature_only_thought(sig, bidx_late)]
                        )
                    ]

                self._pending_sig = sig
                return []

            if dtype == "arguments_delta":
                slot = self._slots.get(idx)
                if slot is None:
                    tc_i = self._next_tc_idx
                    self._next_tc_idx += 1
                    slot = _StepSlot(
                        kind="function_call",
                        step_type="function_call",
                        emitted_block=True,
                        tc_index=tc_i,
                    )
                    self._slots[idx] = slot
                    self._fc_slots.append(slot)
                arg_str = _get_field(delta, "arguments")
                if isinstance(arg_str, str) and arg_str:
                    slot.saw_args = True
                    slot.raw_args_buf += arg_str
                    return [
                        self._make_chunk(
                            tool_call_chunks=[
                                tool_call_chunk(
                                    name=None,
                                    args=arg_str,
                                    id=None,
                                    index=slot.tc_index,
                                )
                            ]
                        )
                    ]
                return []

            if dtype in ("image", "audio", "video", "document"):
                slot = self._slots.get(idx)
                if slot is None:
                    slot = _StepSlot(kind="model_output", step_type="model_output")
                    self._slots[idx] = slot
                m_blocks = self._emit_model_content_item(idx, slot, delta)
                return [self._make_chunk(content=m_blocks)] if m_blocks else []

            if dtype in _SERVER_CALL_STEP_TYPES or dtype in _SERVER_RESULT_STEP_TYPES:
                slot = self._slots.get(idx)
                if slot is None:
                    pre_srv = (
                        self._consume_pending_sig_for_server_call()
                        if dtype in _SERVER_CALL_STEP_TYPES
                        else []
                    )
                    slot = self._open_server_slot(dtype, delta)
                    self._slots[idx] = slot
                    return [self._make_chunk(content=pre_srv)] if pre_srv else []
                self._apply_server_delta(slot, delta)
                return []

            return []

        if etype == "step.stop":
            if _get_field(event, "usage") is not None:
                self._latest_step_usage = _get_field(event, "usage")
            idx = int(_get_field(event, "index") or 0)
            slot = self._slots.get(idx)
            return self._finalize_slot(slot) if slot is not None else []

        if etype == "interaction.completed":
            inter_ev = _get_field(event, "interaction")
            if self._clean_id(
                _get_field(inter_ev, "continuation_token")
            ) is None and self._continuation_token_status not in (
                "requires_continuation",
                "incomplete",
            ):
                self._continuation_token = None
                self._continuation_token_status = None
            return self._complete_from_interaction(inter_ev)

        return []

    def _complete_from_interaction(
        self, inter: Any, *, force_terminal: bool = False
    ) -> list[ChatGenerationChunk]:
        status = str(_get_field(inter, "status") or self._status or "completed")
        self._status = status
        eff_m = _get_field(inter, "model") or _get_field(inter, "agent")
        if isinstance(eff_m, str) and eff_m:
            self._model_name = eff_m
        if status in ("failed", "cancelled"):
            _raise_for_interaction_status(inter, self._model_name)

        cid = self._clean_id(_get_field(inter, "id"))
        if cid:
            self._interaction_id = cid
            if cid not in self._hop_ids:
                self._hop_ids.append(cid)
        if self._hop_index == 0 and self._previous_interaction_id is None:
            prev_from_end = self._clean_id(_get_field(inter, "previous_interaction_id"))
            if prev_from_end:
                self._previous_interaction_id = prev_from_end
        if _get_field(inter, "service_tier") is not None:
            self._service_tier = _get_field(inter, "service_tier")
        if _get_field(inter, "environment_id") is not None:
            self._environment_id = _get_field(inter, "environment_id")
        cont_tok = self._clean_id(_get_field(inter, "continuation_token"))
        if cont_tok:
            self._continuation_token = cont_tok
            self._continuation_token_status = status

        # When status is `requires_continuation` during SSE streaming, SDK>=2.34 with
        # default `automatic_continuation=True` yields the next hop right on the same
        # Stream. Defer `chunk_position='last'`/metadata/audio to `finalize_eof()` so a
        # multi-hop stream never emits premature finish events or concatenates statuses,
        # while `automatic_continuation=False` still closes cleanly at stream EOF!
        if status == "requires_continuation" and not force_terminal:
            hop_usage = _get_field(inter, "usage") or self._latest_step_usage
            self._latest_step_usage = None
            self._accumulate_hop_usage(hop_usage)
            pending_sig_backup = self._pending_sig
            flushed_hop = self._reset_hop_slots()
            if (
                pending_sig_backup
                and not self._fc_slots
                and self._next_block_idx == 0
                and not self._audio_raw_chunks
            ):
                self._pending_sig = pending_sig_backup
            self._pending_continuation_inter = inter
            self._hop_started = True
            return flushed_hop

        flushed: list[ChatGenerationChunk] = []
        for s_idx in sorted(self._slots):
            flushed.extend(self._finalize_slot(self._slots[s_idx]))
        self._slots.clear()

        if (
            self._pending_sig
            and not self._fc_slots
            and self._next_block_idx == 0
            and not self._audio_raw_chunks
        ):
            flushed.append(
                self._make_chunk(
                    content=[
                        {
                            "type": "text",
                            "text": "",
                            "extras": {"signature": self._pending_sig},
                            "index": self._alloc_block_idx(),
                        }
                    ]
                )
            )
            self._pending_sig = None

        add_kw: dict[str, Any] = {}
        if self._audio_raw_chunks:
            add_kw["audio"] = _wrap_audio_frames(
                b"".join(self._audio_raw_chunks),
                mime_type=self._audio_mime,
                sample_rate=self._audio_sample_rate,
                channels=self._audio_channels,
            )

        for fc_s in self._fc_slots:
            try:
                parsed_fc = json.loads(fc_s.raw_args_buf or "{}")
                if isinstance(parsed_fc, dict) and fc_s.name:
                    clean_fc_args = _convert_integer_like_floats(parsed_fc)
                    add_kw["function_call"] = {
                        "name": fc_s.name,
                        "arguments": json.dumps(clean_fc_args, ensure_ascii=False),
                    }
            except json.JSONDecodeError:
                continue

        if self._pydantic_schema is not None and not self._fc_slots:
            joined_out = "".join(self._all_output_texts).strip()
            if joined_out:
                try:
                    if issubclass(self._pydantic_schema, BaseModel):
                        add_kw["parsed"] = self._pydantic_schema.model_validate_json(
                            joined_out
                        )
                    elif issubclass(self._pydantic_schema, BaseModelV1):
                        add_kw["parsed"] = self._pydantic_schema.parse_raw(joined_out)
                except Exception as exc:
                    logger.debug(
                        "Failed to parse streamed structured output for %s: %s",
                        self._pydantic_schema,
                        exc,
                    )

        finish_reason = _status_to_finish_reason(status)
        finish_reason_fields = _finish_reason_fields(
            finish_reason,
            has_tool_calls=bool(self._fc_slots),
        )
        end_meta: dict[str, Any] = {
            "id": self._interaction_id,
            "interaction_id": self._interaction_id,
            "stored": bool(self._requested_store and self._interaction_id),
            "status": status,
            "finish_reason": finish_reason,
            **finish_reason_fields,
        }
        if self._previous_interaction_id:
            end_meta["previous_interaction_id"] = self._previous_interaction_id
        if len(self._hop_ids) > 1:
            end_meta["continuation_ids"] = list(self._hop_ids)
        if (
            status in ("requires_continuation", "incomplete")
            and self._continuation_token
        ):
            end_meta["continuation_token"] = self._continuation_token

        stier = (
            _get_field(inter, "service_tier")
            if _get_field(inter, "service_tier") is not None
            else self._service_tier
        )
        if stier is not None:
            end_meta["service_tier"] = stier
        env_id = (
            _get_field(inter, "environment_id")
            if _get_field(inter, "environment_id") is not None
            else self._environment_id
        )
        if env_id is not None:
            end_meta["environment_id"] = env_id

        gen_info: dict[str, Any] = {
            "finish_reason": finish_reason,
            **finish_reason_fields,
        }
        if (
            self._saw_grounding_step
            or self._grounding_chunks
            or self._grounding_supports
            or self._web_search_queries
            or self._maps_widget_token
        ):
            gm: dict[str, Any] = {
                "web_search_queries": self._web_search_queries,
                "grounding_chunks": self._grounding_chunks,
                "grounding_supports": self._grounding_supports,
            }
            if self._maps_widget_token:
                gm["google_maps_widget_context_token"] = self._maps_widget_token
            end_meta["grounding_metadata"] = gm
            gen_info["grounding_metadata"] = gm

        if self._pending_continuation_inter is None:
            usage_raw = _get_field(inter, "usage") or self._latest_step_usage
            self._latest_step_usage = None
            self._accumulate_hop_usage(usage_raw)
        self._pending_continuation_inter = None
        usage_meta = self._accumulated_usage

        self._completed = True
        flushed.append(
            self._make_chunk(
                additional_kwargs=add_kw or None,
                extra_meta=end_meta,
                generation_info=gen_info,
                usage_metadata=usage_meta,
                chunk_position="last",
            )
        )
        return flushed

    def finalize_eof(
        self, *, allow_non_terminal: bool = False
    ) -> list[ChatGenerationChunk]:
        """Emit a closing `chunk_position='last'` chunk on EOF or uncontinued hop."""
        if self._completed:
            return []
        if self._pending_continuation_inter is not None:
            return self._complete_from_interaction(
                self._pending_continuation_inter, force_terminal=True
            )
        if self._status in _NON_TERMINAL_STATUSES and not allow_non_terminal:
            iid_lbl = self._interaction_id or "<stateless>"
            msg = (
                f"Stream for interaction '{iid_lbl}' on '{self._model_name}' ended "
                f"prematurely while still in non-terminal status '{self._status}'."
            )
            raise GoogleInteractionsAPIError(
                code=500,
                response_json={
                    "error": {
                        "code": 500,
                        "message": msg,
                        "status": "UNAVAILABLE",
                    }
                },
                response=None,
            )
        return self._complete_from_interaction(
            {"id": self._interaction_id, "status": self._status or "completed"},
            force_terminal=True,
        )

    def reconcile_polled_interaction(
        self, interaction: Any
    ) -> list[ChatGenerationChunk]:
        """Reconcile a polled terminal `Interaction` after a premature SSE drop."""
        status = str(_get_field(interaction, "status") or "completed")
        if status in ("failed", "cancelled"):
            _raise_for_interaction_status(interaction, self._model_name)

        flushed: list[ChatGenerationChunk] = []
        for s_idx in sorted(self._slots):
            flushed.extend(self._finalize_slot(self._slots[s_idx]))
        self._slots.clear()

        had_streamed_payload = bool(
            self._all_output_texts
            or self._fc_slots
            or self._audio_raw_chunks
            or self._next_block_idx > 0
        )
        if not had_streamed_payload:
            polled_res = _interaction_to_result(
                interaction,
                model_name=self._model_name,
                requested_store=self._requested_store,
                output_version=self._output_version,
                pydantic_schema=self._pydantic_schema,
                previous_interaction_id=self._previous_interaction_id,
            )
            self._completed = True
            return [*flushed, _result_to_stream_chunk(polled_res)]

        # Reconcile un-emitted tail text, tool calls, grounding, and terminal usage
        already_txt = "".join(self._all_output_texts)
        polled_v0 = _interaction_to_result(
            interaction,
            model_name=self._model_name,
            requested_store=self._requested_store,
            output_version="interactions/v1" if self._is_ia_v1 else None,
            pydantic_schema=self._pydantic_schema,
            previous_interaction_id=self._previous_interaction_id,
        )
        polled_msg = cast("AIMessage", polled_v0.generations[0].message)
        full_txt = polled_msg.text
        rem_txt = (
            full_txt[len(already_txt) :]
            if full_txt.startswith(already_txt)
            else (full_txt if not already_txt else "")
        )
        if rem_txt:
            self._all_output_texts.append(rem_txt)
            flushed.append(
                self._make_chunk(
                    content=[
                        {
                            "type": "text",
                            "text": rem_txt,
                            "index": self._alloc_block_idx(),
                        }
                    ]
                )
            )

        emitted_fc_ids = {fcs.call_id for fcs in self._fc_slots if fcs.call_id}
        out_steps = _slice_output_steps(_get_field(interaction, "steps") or [])
        polled_sigs = polled_msg.additional_kwargs.get(
            _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY, {}
        )
        for st in out_steps:
            if _get_field(st, "type") != "function_call":
                continue
            tmp_tcs: list[Any] = []
            tmp_inv: list[Any] = []
            tmp_kw: dict[str, Any] = {}
            fc_id, fc_args = _parse_function_call_step(
                st,
                tool_calls=tmp_tcs,
                invalid_tool_calls=tmp_inv,
                additional_kwargs=tmp_kw,
            )
            if fc_id in emitted_fc_ids:
                continue
            emitted_fc_ids.add(fc_id)
            tc_i = self._next_tc_idx
            self._next_tc_idx += 1
            raw_s = (
                json.dumps(fc_args, ensure_ascii=False)
                if fc_args is not None
                else str(_get_field(st, "arguments") or "{}")
            )
            fc_sig = (
                polled_sigs.get(fc_id) if isinstance(polled_sigs, Mapping) else None
            )
            if isinstance(fc_sig, str) and fc_sig:
                self._queue_fc_sig(fc_id, fc_sig)
            fname = str(_get_field(st, "name") or "")
            f_slot = _StepSlot(
                kind="function_call",
                step_type="function_call",
                emitted_block=True,
                signature=fc_sig if isinstance(fc_sig, str) else None,
                tc_index=tc_i,
                call_id=fc_id,
                name=fname,
                saw_args=True,
                raw_args_buf=raw_s,
            )
            self._fc_slots.append(f_slot)
            flushed.append(
                self._make_chunk(
                    tool_call_chunks=[
                        tool_call_chunk(name=fname, args=raw_s, id=fc_id, index=tc_i)
                    ]
                )
            )
            if self._is_ia_v1 and fc_args is not None:
                tc_b: dict[str, Any] = {
                    "type": "tool_call",
                    "id": fc_id,
                    "name": fname,
                    "args": fc_args,
                    "index": self._alloc_block_idx(),
                }
                if isinstance(fc_sig, str) and fc_sig:
                    tc_b["extras"] = {"signature": fc_sig}
                flushed.append(self._make_chunk(content=[tc_b]))

        polled_gm = polled_msg.response_metadata.get("grounding_metadata")
        if isinstance(polled_gm, Mapping) and not self._grounding_chunks:
            self._saw_grounding_step = True
            self._web_search_queries = list(polled_gm.get("web_search_queries") or [])
            self._grounding_chunks = list(polled_gm.get("grounding_chunks") or [])
            self._grounding_supports = list(polled_gm.get("grounding_supports") or [])
            if polled_gm.get("google_maps_widget_context_token"):
                self._maps_widget_token = str(
                    polled_gm["google_maps_widget_context_token"]
                )

        if _get_field(interaction, "usage") is not None:
            self._accumulated_usage = None
            self._latest_step_usage = None
        flushed.extend(
            self._complete_from_interaction(interaction, force_terminal=True)
        )
        return flushed


def _result_to_stream_chunk(result: ChatResult) -> ChatGenerationChunk:
    """Convert a polled `ChatResult` into a single `chunk_position='last'` chunk."""
    gen = result.generations[0]
    ai_msg = cast("AIMessage", gen.message)
    content: str | list[str | dict[Any, Any]] = ai_msg.content
    next_idx = 0
    if isinstance(content, list):
        indexed_list: list[str | dict[Any, Any]] = []
        for item in content:
            if isinstance(item, Mapping):
                copied = dict(item)
                if "index" not in copied and copied.get("type") != "tool_call":
                    copied["index"] = next_idx
                    next_idx += 1
                indexed_list.append(copied)
            else:
                indexed_list.append(item)
        content = indexed_list

    tc_chunks = [
        tool_call_chunk(
            name=tc["name"],
            args=json.dumps(tc["args"], ensure_ascii=False),
            id=tc.get("id"),
            index=i,
        )
        for i, tc in enumerate(ai_msg.tool_calls)
    ]
    for j, itc in enumerate(ai_msg.invalid_tool_calls):
        tc_chunks.append(
            tool_call_chunk(
                name=itc.get("name"),
                args=itc.get("args"),
                id=itc.get("id"),
                index=len(ai_msg.tool_calls) + j,
            )
        )
    chunk_msg = AIMessageChunk(
        content=content,
        additional_kwargs=dict(ai_msg.additional_kwargs),
        response_metadata=dict(ai_msg.response_metadata),
        tool_call_chunks=tc_chunks,
        usage_metadata=ai_msg.usage_metadata,
        id=ai_msg.id,
        chunk_position="last",
    )
    gen_info = {
        k: v for k, v in (gen.generation_info or {}).items() if k != "model_name"
    }
    return ChatGenerationChunk(message=chunk_msg, generation_info=gen_info)


def _handle_midstream_transport_error(exc: BaseException, model_name: str) -> None:
    """Classify transport/SDK exceptions raised during SSE event iteration."""
    if isinstance(exc, _INTERACTIONS_CATCHABLE_ERRORS):
        _handle_interactions_error(exc, model_name=model_name)
    if isinstance(exc, (TimeoutError, *_HTTPX_TIMEOUT_ERRORS)):
        msg = f"Timeout streaming from model '{model_name}': {exc}"
        raise GoogleTimeoutError(msg) from exc
    if isinstance(exc, _HTTPX_HTTP_ERRORS):
        msg = f"Connection error streaming from model '{model_name}': {exc}"
        raise GoogleConnectionError(msg) from exc
    raise exc


def _poll_until_terminal_sync(
    interactions_client: Any,
    interaction: Any,
    prep: _InteractionsPreparedRequest,
    run_manager: CallbackManagerForLLMRun | None,
) -> Any:
    """Poll `interactions.get` synchronously until `interaction.status` is terminal."""
    raw_iid = _get_field(interaction, "id")
    iid = raw_iid.strip() if isinstance(raw_iid, str) else ""
    if not iid:
        msg_id = f"Background interaction on '{prep.target_name}' returned no id."
        raise GoogleInteractionsAPIError(
            code=500,
            response_json={
                "error": {"code": 500, "message": msg_id, "status": "INTERNAL"}
            },
            response=None,
        )
    deadline = (
        time.monotonic() + prep.timeout_seconds
        if prep.timeout_seconds is not None and prep.timeout_seconds > 0
        else None
    )
    cur = interaction
    while _get_field(cur, "status") in _NON_TERMINAL_STATUSES:
        remaining = deadline - time.monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            msg = (
                f"Timed out waiting for background interaction '{iid}' "
                f"on '{prep.target_name}' after {prep.timeout_seconds}s."
            )
            raise GoogleTimeoutError(msg)
        time.sleep(
            min(prep.polling_interval, remaining)
            if remaining is not None
            else prep.polling_interval
        )
        poll_kw: dict[str, Any] = {"id": iid}
        if prep.api_version is not None:
            poll_kw["api_version"] = prep.api_version
        if prep.extra_headers is not None:
            poll_kw["extra_headers"] = prep.extra_headers
        if deadline is not None:
            poll_kw["timeout"] = max(1.0, deadline - time.monotonic())
        cur = _call_with_interactions_retry(
            interactions_client.get,
            model_name=prep.target_name,
            max_retries=prep.max_retries,
            run_manager=run_manager,
            **poll_kw,
        )
    return cur


async def _poll_until_terminal_async(
    interactions_client: Any,
    interaction: Any,
    prep: _InteractionsPreparedRequest,
    run_manager: AsyncCallbackManagerForLLMRun | None,
) -> Any:
    """Poll `interactions.get` asynchronously until `interaction.status` is terminal."""
    raw_iid = _get_field(interaction, "id")
    iid = raw_iid.strip() if isinstance(raw_iid, str) else ""
    if not iid:
        msg_id = f"Background interaction on '{prep.target_name}' returned no id."
        raise GoogleInteractionsAPIError(
            code=500,
            response_json={
                "error": {"code": 500, "message": msg_id, "status": "INTERNAL"}
            },
            response=None,
        )
    deadline = (
        time.monotonic() + prep.timeout_seconds
        if prep.timeout_seconds is not None and prep.timeout_seconds > 0
        else None
    )
    cur = interaction
    while _get_field(cur, "status") in _NON_TERMINAL_STATUSES:
        rem = deadline - time.monotonic() if deadline is not None else None
        if rem is not None and rem <= 0:
            msg_t = (
                f"Timed out waiting for background interaction '{iid}' "
                f"on '{prep.target_name}' after {prep.timeout_seconds}s."
            )
            raise GoogleTimeoutError(msg_t)
        await asyncio.sleep(
            min(prep.polling_interval, rem)
            if rem is not None
            else prep.polling_interval
        )
        poll_kw_a: dict[str, Any] = {"id": iid}
        if prep.api_version is not None:
            poll_kw_a["api_version"] = prep.api_version
        if prep.extra_headers is not None:
            poll_kw_a["extra_headers"] = prep.extra_headers
        if deadline is not None:
            poll_kw_a["timeout"] = max(1.0, deadline - time.monotonic())
        cur = await _acall_with_interactions_retry(
            interactions_client.get,
            model_name=prep.target_name,
            max_retries=prep.max_retries,
            run_manager=run_manager,
            **poll_kw_a,
        )
    return cur


def _poll_non_stream_interaction_sync(
    interactions_client: Any,
    interaction: Any,
    prep: _InteractionsPreparedRequest,
    run_manager: CallbackManagerForLLMRun | None,
) -> ChatGenerationChunk:
    """Poll a non-streaming `Interaction` returned on `stream()` until terminal."""
    if prep.poll and _get_field(interaction, "status") in _NON_TERMINAL_STATUSES:
        interaction = _poll_until_terminal_sync(
            interactions_client, interaction, prep, run_manager
        )
    res = _interaction_to_result(
        interaction,
        model_name=prep.target_name,
        requested_store=prep.requested_store,
        output_version=prep.output_version,
        pydantic_schema=prep.pydantic_schema,
        previous_interaction_id=prep.previous_interaction_id,
    )
    return _result_to_stream_chunk(res)


def stream_interactions(
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
) -> Iterator[ChatGenerationChunk]:
    """Stream `ChatGenerationChunk`s synchronously from the Gemini Interactions API."""
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
        stream=True,
        **kwargs,
    )

    if prep.background and "agent" not in prep.create_kwargs:
        polled = generate_interactions(
            model_instance,
            messages,
            stop=stop,
            run_manager=run_manager,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content,
            tool_choice=tool_choice,
            **kwargs,
        )
        gen_c = _result_to_stream_chunk(polled)
        if run_manager:
            run_manager.on_llm_new_token(gen_c.text, chunk=gen_c)
        yield gen_c
        return

    interactions_client, ephemeral_http = _get_scoped_interactions_client(
        model_instance,
        is_async=False,
        base_url_override=prep.base_url_override,
        client_args_override=prep.client_args_override,
        httpx_client_override=prep.httpx_client_override,
    )

    def _create_stream_sync() -> Any:
        created = interactions_client.create(**prep.create_kwargs)
        if (
            not prep.background
            and (not hasattr(created, "__iter__") or hasattr(created, "steps"))
            and _get_field(created, "status") in ("failed", "cancelled")
        ):
            eff_n = str(
                _get_field(created, "model")
                or _get_field(created, "agent")
                or prep.target_name
                or "unknown"
            )
            _raise_for_interaction_status(created, eff_n)
        return created

    raw_stream: Any = None
    try:
        raw_stream = _call_with_interactions_retry(
            _create_stream_sync,
            model_name=prep.target_name,
            max_retries=prep.max_retries,
            run_manager=run_manager,
        )

        if not hasattr(raw_stream, "__iter__") or hasattr(raw_stream, "steps"):
            gen_f = _poll_non_stream_interaction_sync(
                interactions_client, raw_stream, prep, run_manager
            )
            if run_manager:
                run_manager.on_llm_new_token(gen_f.text, chunk=gen_f)
            yield gen_f
            return

        processor = _InteractionsStreamProcessor(
            target_name=prep.target_name,
            requested_store=prep.requested_store,
            output_version=prep.output_version,
            pydantic_schema=prep.pydantic_schema,
            previous_interaction_id=prep.previous_interaction_id,
        )
        iterator = iter(raw_stream)
        while True:
            try:
                event = next(iterator)
            except StopIteration:
                break
            except Exception as exc:
                _handle_midstream_transport_error(exc, prep.target_name)
                raise  # pragma: no cover

            for gen in processor.process_event(event):
                if run_manager:
                    run_manager.on_llm_new_token(gen.text, chunk=gen)
                yield gen

        if processor.is_non_terminal_eof:
            if not prep.poll:
                eof_chunks = processor.finalize_eof(allow_non_terminal=True)
            elif (prep.background or prep.requested_store) and processor.interaction_id:
                polled_s = _poll_until_terminal_sync(
                    interactions_client,
                    {"id": processor.interaction_id, "status": "in_progress"},
                    prep,
                    run_manager,
                )
                eof_chunks = processor.reconcile_polled_interaction(polled_s)
            else:
                eof_chunks = processor.finalize_eof()
        else:
            eof_chunks = processor.finalize_eof()

        for gen in eof_chunks:
            if run_manager:
                run_manager.on_llm_new_token(gen.text, chunk=gen)
            yield gen
    finally:
        try:
            close_fn = (
                getattr(raw_stream, "close", None) if raw_stream is not None else None
            )
            if callable(close_fn):
                close_fn()
            inner_gen_close = getattr(
                getattr(raw_stream, "generator", None), "close", None
            )
            if callable(inner_gen_close):
                inner_gen_close()
        finally:
            if ephemeral_http is not None:
                ephemeral_http.close()


async def astream_interactions(
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
) -> AsyncGenerator[ChatGenerationChunk, None]:
    """Stream `ChatGenerationChunk`s asynchronously from the Gemini Interactions API."""
    # Background model runs delegate directly to `agenerate_interactions` so any
    # external HTTP URLs are fetched asynchronously exactly once.
    raw_bg = kwargs.get("background")
    if raw_bg is None:
        raw_bg = (
            model_instance.background
            if model_instance.background is not None
            else (model_instance.model_kwargs or {}).get("background")
        )
    has_agent = any(
        bool(v)
        for v in (
            kwargs.get("agent"),
            model_instance.agent,
            (model_instance.model_kwargs or {}).get("agent"),
            kwargs.get("agent_config"),
            model_instance.agent_config,
            (model_instance.model_kwargs or {}).get("agent_config"),
        )
    ) or _get_field(model_instance, "model", "").rsplit("/", 1)[-1].lower().startswith(
        ("deep-research-", "antigravity-")
    )
    if (
        bool(raw_bg)
        and not has_agent
        and not str(kwargs.get("model") or "")
        .rsplit("/", 1)[-1]
        .lower()
        .startswith(("deep-research-", "antigravity-"))
    ):
        polled_fast = await agenerate_interactions(
            model_instance,
            messages,
            stop=stop,
            run_manager=run_manager,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content,
            tool_choice=tool_choice,
            **kwargs,
        )
        gen_cf = _result_to_stream_chunk(polled_fast)
        if run_manager:
            await run_manager.on_llm_new_token(gen_cf.text, chunk=gen_cf)
        yield gen_cf
        return

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
        stream=True,
        **kwargs,
    )

    if prep.background and "agent" not in prep.create_kwargs:  # pragma: no cover
        polled = await agenerate_interactions(
            model_instance,
            messages,
            stop=stop,
            run_manager=run_manager,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content,
            tool_choice=tool_choice,
            **kwargs,
        )
        gen_c = _result_to_stream_chunk(polled)
        if run_manager:
            await run_manager.on_llm_new_token(gen_c.text, chunk=gen_c)
        yield gen_c
        return

    interactions_client, ephemeral_http = _get_scoped_interactions_client(
        model_instance,
        is_async=True,
        base_url_override=prep.base_url_override,
        client_args_override=prep.async_client_args_override,
        httpx_client_override=prep.httpx_async_client_override,
    )

    async def _create_stream_async() -> Any:
        created = await interactions_client.create(**prep.create_kwargs)
        if (
            not prep.background
            and (not hasattr(created, "__aiter__") or hasattr(created, "steps"))
            and _get_field(created, "status") in ("failed", "cancelled")
        ):
            eff_n = str(
                _get_field(created, "model")
                or _get_field(created, "agent")
                or prep.target_name
                or "unknown"
            )
            _raise_for_interaction_status(created, eff_n)
        return created

    raw_stream: Any = None
    try:
        raw_stream = await _acall_with_interactions_retry(
            _create_stream_async,
            model_name=prep.target_name,
            max_retries=prep.max_retries,
            run_manager=run_manager,
        )

        if not hasattr(raw_stream, "__aiter__") or hasattr(raw_stream, "steps"):
            cur = raw_stream
            if prep.poll and _get_field(cur, "status") in _NON_TERMINAL_STATUSES:
                cur = await _poll_until_terminal_async(
                    interactions_client, cur, prep, run_manager
                )
            res_a = _interaction_to_result(
                cur,
                model_name=prep.target_name,
                requested_store=prep.requested_store,
                output_version=prep.output_version,
                pydantic_schema=prep.pydantic_schema,
                previous_interaction_id=prep.previous_interaction_id,
            )
            gen_p = _result_to_stream_chunk(res_a)
            if run_manager:
                await run_manager.on_llm_new_token(gen_p.text, chunk=gen_p)
            yield gen_p
            return

        chunk_timeout_s = _resolve_stream_chunk_timeout_s()
        processor = _InteractionsStreamProcessor(
            target_name=prep.target_name,
            requested_store=prep.requested_store,
            output_version=prep.output_version,
            pydantic_schema=prep.pydantic_schema,
            previous_interaction_id=prep.previous_interaction_id,
        )
        aiterator = raw_stream.__aiter__()
        while True:
            try:
                if chunk_timeout_s is not None:
                    event = await asyncio.wait_for(
                        aiterator.__anext__(), timeout=chunk_timeout_s
                    )
                else:
                    event = await aiterator.__anext__()
            except StopAsyncIteration:
                break
            except TimeoutError as exc:
                if type(exc) is not TimeoutError and (
                    isinstance(exc, _INTERACTIONS_CATCHABLE_ERRORS)
                    or isinstance(exc, _HTTPX_TIMEOUT_ERRORS)
                ):
                    _handle_midstream_transport_error(exc, prep.target_name)
                msg = (
                    f"Async stream from model '{prep.target_name}' timed out after "
                    f"{chunk_timeout_s}s waiting for the next SSE event."
                )
                raise GoogleTimeoutError(msg) from exc
            except Exception as exc:
                _handle_midstream_transport_error(exc, prep.target_name)
                raise  # pragma: no cover

            for gen in processor.process_event(event):
                if run_manager:
                    await run_manager.on_llm_new_token(gen.text, chunk=gen)
                yield gen

        if processor.is_non_terminal_eof:
            if not prep.poll:
                eof_chunks_a = processor.finalize_eof(allow_non_terminal=True)
            elif (prep.background or prep.requested_store) and processor.interaction_id:
                polled_a = await _poll_until_terminal_async(
                    interactions_client,
                    {"id": processor.interaction_id, "status": "in_progress"},
                    prep,
                    run_manager,
                )
                eof_chunks_a = processor.reconcile_polled_interaction(polled_a)
            else:
                eof_chunks_a = processor.finalize_eof()
        else:
            eof_chunks_a = processor.finalize_eof()

        for gen in eof_chunks_a:
            if run_manager:
                await run_manager.on_llm_new_token(gen.text, chunk=gen)
            yield gen
    finally:
        try:
            close_fn = (
                getattr(raw_stream, "close", None)
                or getattr(raw_stream, "aclose", None)
                if raw_stream is not None
                else None
            )
            if callable(close_fn):
                maybe_coro = close_fn()
                if inspect.isawaitable(maybe_coro):
                    await maybe_coro
            inner_agen_close = getattr(
                getattr(raw_stream, "generator", None), "aclose", None
            )
            if callable(inner_agen_close):
                maybe_inner_coro = inner_agen_close()
                if inspect.isawaitable(maybe_inner_coro):
                    await maybe_inner_coro
        finally:
            if ephemeral_http is not None:
                await ephemeral_http.aclose()


__all__ = [
    "_InteractionsStreamProcessor",
    "astream_interactions",
    "stream_interactions",
]
