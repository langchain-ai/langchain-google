"""Unit tests for Gemini Interactions API SSE streaming and v1 block translation."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import io
import json
import warnings
import wave
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import httpx
import httpx2
import langchain_core.messages.block_translators.google_genai as _core_genai_translator
import pytest
from google.genai._gaos import models as gaos_models
from google.genai._gaos import utils as gaos_utils
from google.genai._gaos.types.interactions import Interaction, InteractionSSEEvent
from google.genai.types import FinishReason
from langchain_core._api import LangChainBetaWarning
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from pydantic import BaseModel, TypeAdapter

import langchain_google_genai._compat as _compat_mod
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai._interactions_request import _messages_to_interaction_steps
from langchain_google_genai.chat_models import (
    _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY,
    GoogleAuthenticationError,
    GoogleConnectionError,
    GoogleInteractionsAPIError,
    GoogleInteractionsContextOverflowError,
    GoogleInvalidRequestError,
    GoogleModelNotFoundError,
    GooglePermissionDeniedError,
    GoogleRateLimitError,
    GoogleTimeoutError,
)

MODEL_NAME = "gemini-3.8-flash"
FAKE_KEY = "fake-api-key"

_SSE_ADAPTER: TypeAdapter[Any] = TypeAdapter(InteractionSSEEvent)


def _sse(payload: dict[str, Any]) -> Any:
    """Validate a raw dict through the real Speakeasy `InteractionSSEEvent` union."""
    return _SSE_ADAPTER.validate_python(payload)


def _serialize_sdk_body(create_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Run real `_gaos` Speakeasy Pydantic unmarshaling + JSON serialization."""
    body_kw = {
        k: v
        for k, v in create_kwargs.items()
        if k
        not in {"api_version", "extra_headers", "extra_query", "extra_body", "timeout"}
    }
    req = gaos_utils.unmarshal(
        {"api_version": create_kwargs.get("api_version"), "body": body_kw},
        gaos_models.CreateInteractionRequest,
    )
    serialized = gaos_utils.serialize_request_body(
        req.body,
        False,
        False,
        "json",
        gaos_models.CreateInteractionRequestBody,
        extra_body=create_kwargs.get("extra_body"),
    )
    assert serialized is not None and serialized.content is not None
    return json.loads(serialized.content)


class _FakeSyncStream:
    """Mock matching `google.genai._gaos.utils.eventstreaming.Stream`."""

    def __init__(
        self,
        events: Sequence[Any],
        *,
        error_after: BaseException | None = None,
    ) -> None:
        self._events = list(events)
        self._error_after = error_after
        self._cursor = 0
        self.closed = False

    def __iter__(self) -> _FakeSyncStream:
        return self

    def __next__(self) -> Any:
        if self.closed:
            raise StopIteration
        if self._cursor < len(self._events):
            ev = self._events[self._cursor]
            self._cursor += 1
            return ev
        if self._error_after is not None:
            err = self._error_after
            self._error_after = None
            raise err
        raise StopIteration

    def close(self) -> None:
        self.closed = True


class _FakeAsyncStream:
    """Mock matching `google.genai._gaos.utils.eventstreaming.AsyncStream`."""

    def __init__(
        self,
        events: Sequence[Any],
        *,
        hang_after_events: bool = False,
        error_after: BaseException | None = None,
    ) -> None:
        self._events = list(events)
        self._hang_after = hang_after_events
        self._error_after = error_after
        self._cursor = 0
        self.closed = False

    def __aiter__(self) -> _FakeAsyncStream:
        return self

    async def __anext__(self) -> Any:
        if self.closed:
            raise StopAsyncIteration
        if self._cursor < len(self._events):
            ev = self._events[self._cursor]
            self._cursor += 1
            return ev
        if self._hang_after:
            await asyncio.sleep(60.0)
        if self._error_after is not None:
            err = self._error_after
            self._error_after = None
            raise err
        raise StopAsyncIteration

    async def close(self) -> None:
        self.closed = True


class _FakeInteractionsClient:
    def __init__(
        self,
        sync_factory: Any = None,
        async_factory: Any = None,
        get_items: Sequence[Any] = (),
    ) -> None:
        self._sync_factory = sync_factory
        self._async_factory = async_factory
        self._get_items = list(get_items)
        self.create_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return (
            self._sync_factory() if callable(self._sync_factory) else self._sync_factory
        )

    def get(self, **kwargs: Any) -> Any:
        self.get_calls.append(kwargs)
        return self._get_items.pop(0)

    async def acreate(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return (
            self._async_factory()
            if callable(self._async_factory)
            else self._async_factory
        )

    async def aget(self, **kwargs: Any) -> Any:
        self.get_calls.append(kwargs)
        return self._get_items.pop(0)


class _AsyncClientAdapter:
    def __init__(self, inner: _FakeInteractionsClient) -> None:
        self._inner = inner

    async def create(self, **kwargs: Any) -> Any:
        return await self._inner.acreate(**kwargs)

    async def get(self, **kwargs: Any) -> Any:
        return await self._inner.aget(**kwargs)


def _wired_llm(
    events: Sequence[Any] = (),
    *,
    sync_stream: Any = None,
    async_stream: Any = None,
    get_items: Sequence[Any] = (),
    **model_kwargs: Any,
) -> tuple[
    ChatGoogleGenerativeAI, _FakeInteractionsClient, _FakeSyncStream, _FakeAsyncStream
]:
    s_stream = sync_stream if sync_stream is not None else _FakeSyncStream(events)
    a_stream = async_stream if async_stream is not None else _FakeAsyncStream(events)
    fake_ia = _FakeInteractionsClient(
        sync_factory=lambda: s_stream,
        async_factory=lambda: a_stream,
        get_items=get_items,
    )
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_KEY,
        use_interactions_api=True,
        **model_kwargs,
    )
    object.__setattr__(llm.client, "_interactions", fake_ia)
    object.__setattr__(llm.async_client, "_interactions", _AsyncClientAdapter(fake_ia))
    return llm, fake_ia, s_stream, a_stream


def assert_valid_event_stream(events: Sequence[Mapping[str, Any]]) -> None:
    """Validate strict LangChain v3 protocol event lifecycle invariants."""
    assert len(events) >= 2, f"Expected >=2 events, got {events}"
    assert events[0]["event"] == "message-start"
    assert events[0]["role"] == "ai"
    assert isinstance(events[0].get("id"), str) and events[0]["id"]
    start_meta = events[0].get("metadata") or {}
    assert start_meta.get("provider") == "google_genai"
    assert start_meta.get("model")

    assert events[-1]["event"] == "message-finish"
    assert sum(1 for e in events if e["event"] == "message-start") == 1
    assert sum(1 for e in events if e["event"] == "message-finish") == 1

    expected_next_idx = 0
    open_blocks: set[int] = set()
    finished_blocks: set[int] = set()
    valid_deltas = {
        "text-delta",
        "reasoning-delta",
        "tool-call-delta",
        "data-delta",
        "block-delta",
    }
    unfinalized_types = {"tool_call_chunk", "server_tool_call_chunk"}

    for ev in events[1:-1]:
        kind = ev["event"]
        idx = ev.get("index")
        assert isinstance(idx, int) and idx >= 0
        if kind == "content-block-start":
            assert idx == expected_next_idx
            expected_next_idx += 1
            assert idx not in open_blocks and idx not in finished_blocks
            content = ev.get("content")
            assert isinstance(content, Mapping) and "type" in content
            open_blocks.add(idx)
        elif kind == "content-block-delta":
            assert idx in open_blocks, f"Delta for non-open block {idx}: {ev}"
            delta = ev.get("delta")
            assert isinstance(delta, Mapping) and delta.get("type") in valid_deltas
        elif kind == "content-block-finish":
            assert idx in open_blocks, f"Finish for non-open block {idx}: {ev}"
            open_blocks.remove(idx)
            finished_blocks.add(idx)
            fin_c = ev.get("content")
            assert isinstance(fin_c, Mapping)
            assert fin_c.get("type") not in unfinalized_types, (
                f"Unfinalized chunk leaked into content-block-finish: {fin_c}"
            )
        else:
            pytest.fail(f"Unexpected v3 protocol event: {ev}")

    assert not open_blocks, f"Unclosed content blocks at message-finish: {open_blocks}"


# ---------------------------------------------------------------------------
# 1. Text streaming, metadata non-duplication, single-chunk usage, callbacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_version", [None, "v0", "interactions/v1", "v1"])
async def test_text_stream_sync_and_async_metadata_and_single_usage(
    output_version: str | None,
) -> None:
    sig_b64 = base64.b64encode(b"hello-sig").decode("ascii")
    usage_dict = {
        "total_input_tokens": 12,
        "total_tool_use_tokens": 3,
        "total_output_tokens": 8,
        "total_thought_tokens": 4,
        "total_cached_tokens": 2,
        "total_tokens": 27,
    }
    events = [
        _sse(
            {
                "event_type": "interaction.created",
                "interaction": {
                    "id": "v1_inter_123",
                    "status": "in_progress",
                    "model": MODEL_NAME,
                },
            }
        ),
        _sse(
            {
                "event_type": "interaction.status_update",
                "interaction_id": "v1_inter_123",
                "status": "in_progress",
            }
        ),
        # Signature-only thought step (live Vertex/AI Studio shape for "Hello")
        _sse({"event_type": "step.start", "index": 0, "step": {"type": "thought"}}),
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_signature", "signature": sig_b64},
            }
        ),
        _sse({"event_type": "step.stop", "index": 0}),
        _sse(
            {"event_type": "step.start", "index": 1, "step": {"type": "model_output"}}
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 1,
                "delta": {"type": "text", "text": "Hello"},
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 1,
                "delta": {"type": "text", "text": " world!"},
            }
        ),
        # StepStop echoes cumulative usage in live Vertex SSE; must NOT double-count!
        {
            "event_type": "step.stop",
            "index": 1,
            "usage": usage_dict,
        },
        _sse(
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "v1_inter_123",
                    "status": "completed",
                    "model": MODEL_NAME,
                    "service_tier": "standard",
                    "usage": usage_dict,
                },
            }
        ),
    ]

    sync_tokens: list[str] = []
    async_tokens: list[str] = []

    class _SyncRec(BaseCallbackHandler):
        def on_llm_new_token(self, token: Any, **kw: Any) -> None:
            chunk = kw.get("chunk")
            if chunk is not None and chunk.text:
                sync_tokens.append(chunk.text)

    class _AsyncRec(AsyncCallbackHandler):
        async def on_llm_new_token(self, token: Any, **kw: Any) -> None:
            chunk = kw.get("chunk")
            if chunk is not None and chunk.text:
                async_tokens.append(chunk.text)

    extra_kw: dict[str, Any] = {"store": True}
    if output_version is not None:
        extra_kw["output_version"] = output_version

    llm, fake_ia, s_stream, a_stream = _wired_llm(events, **extra_kw)

    chunks: list[AIMessageChunk] = []
    full: AIMessageChunk | None = None
    non_zero_input_chunks = 0
    for c in llm.stream("Hello", config={"callbacks": [_SyncRec()]}):
        assert isinstance(c, AIMessageChunk)
        assert isinstance(c.content, (str, list))
        if c.usage_metadata and c.usage_metadata.get("input_tokens"):
            non_zero_input_chunks += 1
        chunks.append(c)
        full = c if full is None else full + c

    assert s_stream.closed is True
    assert fake_ia.create_calls[0]["stream"] is True
    assert sync_tokens == ["Hello", " world!"]
    assert non_zero_input_chunks == 1
    assert chunks[-1].chunk_position == "last"

    assert full is not None
    assert full.id == "v1_inter_123"
    assert full.text == "Hello world!"
    # Exact metadata strings (NEVER duplicated/concatenated!)
    assert full.response_metadata["model_provider"] == "google_genai"
    assert full.response_metadata["model_name"] == MODEL_NAME
    assert full.response_metadata["id"] == "v1_inter_123"
    assert full.response_metadata["interaction_id"] == "v1_inter_123"
    assert full.response_metadata["stored"] is True
    assert full.response_metadata["finish_reason"] == "STOP"
    assert full.response_metadata["service_tier"] == "standard"
    assert full.usage_metadata == {
        "input_tokens": 15,
        "output_tokens": 12,
        "total_tokens": 27,
        "input_token_details": {"cache_read": 2},
        "output_token_details": {"reasoning": 4},
    }
    # test_stream invariant: signature-only thought does NOT emit a 2nd content_block!
    assert len(full.content_blocks) == 1
    assert full.content_blocks[0]["type"] == "text"
    assert full.content_blocks[0]["text"] == "Hello world!"
    assert full.content_blocks[0]["extras"]["signature"] == sig_b64

    # Async parity
    achunks = [
        c async for c in llm.astream("Hello", config={"callbacks": [_AsyncRec()]})
    ]
    assert a_stream.closed is True
    assert async_tokens == ["Hello", " world!"]
    assert achunks[-1].chunk_position == "last"
    afull = achunks[0]
    for ac in achunks[1:]:
        afull = afull + ac
    assert afull.text == "Hello world!"
    assert afull.response_metadata["model_name"] == MODEL_NAME
    assert afull.response_metadata["interaction_id"] == "v1_inter_123"
    assert afull.usage_metadata == full.usage_metadata


# ---------------------------------------------------------------------------
# 2. Late out-of-order thought_signature, parallel + 0-arg tools, replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_version", [None, "interactions/v1", "v1"])
def test_late_thought_signature_parallel_and_zero_arg_tools_and_replay(
    output_version: str | None,
) -> None:
    late_sig = base64.b64encode(b"late-thought-sig-bytes").decode("ascii")
    events = [
        _sse(
            {
                "event_type": "interaction.created",
                "interaction": {"id": "", "status": "in_progress", "model": MODEL_NAME},
            }
        ),
        # Step 0: thought starts with summary pieces, NO signature yet!
        _sse({"event_type": "step.start", "index": 0, "step": {"type": "thought"}}),
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {
                    "type": "thought_summary",
                    "content": {"type": "text", "text": "Checking weather "},
                },
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {
                    "type": "thought_summary",
                    "content": {"type": "text", "text": "and clock."},
                },
            }
        ),
        # Step 1: model_output text arrives BEFORE step 0 closes
        _sse(
            {"event_type": "step.start", "index": 1, "step": {"type": "model_output"}}
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 1,
                "delta": {"type": "text", "text": "Checking now!"},
            }
        ),
        _sse({"event_type": "step.stop", "index": 1}),
        # Step 3 (non-contiguous index): 1-arg tool split across interleaved deltas
        _sse(
            {
                "event_type": "step.start",
                "index": 3,
                "step": {
                    "type": "function_call",
                    "id": "call_weather",
                    "name": "get_weather",
                    "arguments": {},
                },
            }
        ),
        # Step 5: ZERO-arg tool with step.start arguments={} and NO arguments_delta!
        _sse(
            {
                "event_type": "step.start",
                "index": 5,
                "step": {
                    "type": "function_call",
                    "id": "call_clock",
                    "name": "get_server_time",
                    "arguments": {},
                },
            }
        ),
        # Step 7: Parallel 2-arg tool interleaved with step 3
        _sse(
            {
                "event_type": "step.start",
                "index": 7,
                "step": {
                    "type": "function_call",
                    "id": "call_calc",
                    "name": "multiply",
                    "arguments": {},
                },
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 3,
                "delta": {"type": "arguments_delta", "arguments": '{"ci'},
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 7,
                "delta": {"type": "arguments_delta", "arguments": '{"a": 6,'},
            }
        ),
        # LATE out-of-order thought_signature for index=0 arrives mid-tool-streaming!
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_signature", "signature": late_sig},
            }
        ),
        _sse({"event_type": "step.stop", "index": 0}),
        _sse(
            {
                "event_type": "step.delta",
                "index": 3,
                "delta": {"type": "arguments_delta", "arguments": 'ty": "Tokyo"}'},
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 7,
                "delta": {"type": "arguments_delta", "arguments": ' "b": 7}'},
            }
        ),
        _sse({"event_type": "step.stop", "index": 3}),
        _sse({"event_type": "step.stop", "index": 5}),
        _sse({"event_type": "step.stop", "index": 7}),
        _sse(
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "",
                    "status": "requires_action",
                    "model": MODEL_NAME,
                },
            }
        ),
    ]

    kw: dict[str, Any] = {"output_version": output_version} if output_version else {}
    llm, _, _, _ = _wired_llm(events, **kw)
    chunks = list(llm.stream("Check Tokyo weather, server time, and 6*7"))
    assert chunks[-1].chunk_position == "last"
    full = chunks[0]
    for c in chunks[1:]:
        full = full + c

    # Signature NEVER leaks into .text!
    assert full.text == "Checking now!"
    assert late_sig not in full.text
    assert full.id is not None and full.id.startswith("lc_")
    assert full.response_metadata["stored"] is False
    assert full.response_metadata["id"] is None
    assert full.response_metadata["finish_reason"] == "STOP"

    # Sequential tool_call_chunk indices 0, 1, 2 (NOT raw event indices 3, 5, 7!)
    assert [tcc["index"] for tcc in full.tool_call_chunks] == [0, 1, 2]
    assert full.invalid_tool_calls == []
    assert full.tool_calls == [
        {
            "name": "get_weather",
            "args": {"city": "Tokyo"},
            "id": "call_weather",
            "type": "tool_call",
        },
        {
            "name": "get_server_time",
            "args": {},
            "id": "call_clock",
            "type": "tool_call",
        },
        {
            "name": "multiply",
            "args": {"a": 6, "b": 7},
            "id": "call_calc",
            "type": "tool_call",
        },
    ]

    # Late thought_signature routed back to block 0 AND recorded in map cleanly
    cbs = full.content_blocks
    assert cbs[0]["type"] == "reasoning"
    assert cbs[0]["reasoning"] == "Checking weather and clock."
    assert cbs[0]["extras"]["signature"] == late_sig
    fc_sigs = full.additional_kwargs[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY]
    assert fc_sigs == {
        "call_weather": late_sig,
        "call_clock": late_sig,
        "call_calc": late_sig,
    }

    # Multi-turn replay into _messages_to_interaction_steps + real Speakeasy SDK!
    _, steps = _messages_to_interaction_steps(
        [
            HumanMessage("go"),
            full,
            ToolMessage("Sunny 22C", tool_call_id="call_weather"),
            ToolMessage("10:00Z", tool_call_id="call_clock"),
            ToolMessage("42", tool_call_id="call_calc"),
        ],
        effective_model=MODEL_NAME,
    )
    assert [s["type"] for s in steps] == [
        "user_input",
        "thought",
        "model_output",
        "function_call",
        "function_call",
        "function_call",
        "function_result",
        "function_result",
        "function_result",
    ]
    assert steps[1] == {
        "type": "thought",
        "signature": late_sig,
        "summary": [{"type": "text", "text": "Checking weather and clock."}],
    }
    assert steps[3]["arguments"] == {"city": "Tokyo"}
    assert steps[4]["arguments"] == {}
    assert steps[5]["arguments"] == {"a": 6, "b": 7}
    _serialize_sdk_body({"model": MODEL_NAME, "input": steps, "store": False})


def test_malformed_arguments_delta_routes_to_invalid_tool_calls() -> None:
    events = [
        _sse(
            {
                "event_type": "interaction.created",
                "interaction": {"id": "", "status": "in_progress", "model": MODEL_NAME},
            }
        ),
        _sse(
            {
                "event_type": "step.start",
                "index": 0,
                "step": {
                    "type": "function_call",
                    "id": "bad_1",
                    "name": "broken_fn",
                    "arguments": {},
                },
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "arguments_delta", "arguments": "not-valid-json!"},
            }
        ),
        _sse({"event_type": "step.stop", "index": 0}),
        _sse(
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "",
                    "status": "requires_action",
                    "model": MODEL_NAME,
                },
            }
        ),
    ]
    llm, _, _, _ = _wired_llm(events)
    chunks = list(llm.stream("run"))
    full = chunks[0]
    for c in chunks[1:]:
        full = full + c
    assert full.tool_calls == []
    assert len(full.invalid_tool_calls) == 1
    assert full.invalid_tool_calls[0]["id"] == "bad_1"
    assert full.invalid_tool_calls[0]["args"] == "not-valid-json!"


# ---------------------------------------------------------------------------
# 3. UTF-8 URLCitation / PlaceCitation / FileCitation & server tools + media
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("post_hoc_annotation_step", [False, True])
@pytest.mark.parametrize("output_version", [None, "interactions/v1", "v1"])
def test_utf8_url_citation_streaming_same_and_posthoc_steps(
    post_hoc_annotation_step: bool, output_version: str | None
) -> None:
    # Multi-byte UTF-8: 🚀 (4B), é (2B), U+2013 en-dash (3B), 東京 (6B)
    prefix = "🚀 Café 40\u201322 in "
    target = "東京 (Tokyo)"
    suffix = " won!"
    full_text = prefix + target + suffix
    sb = len(prefix.encode("utf-8"))
    eb = sb + len(target.encode("utf-8"))
    assert sb != len(prefix)  # Confirm multi-byte drift is exercised!

    events: list[Any] = [
        _sse(
            {
                "event_type": "interaction.created",
                "interaction": {"id": "", "status": "in_progress", "model": MODEL_NAME},
            }
        ),
        _sse(
            {
                "event_type": "step.start",
                "index": 0,
                "step": {
                    "type": "google_search_call",
                    "id": "gs_1",
                    "search_type": "web_search",
                    "arguments": {},
                },
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {
                    "type": "google_search_call",
                    "arguments": {"queries": ["Tokyo cafe score"]},
                },
            }
        ),
        _sse({"event_type": "step.stop", "index": 0}),
        _sse(
            {
                "event_type": "step.start",
                "index": 1,
                "step": {
                    "type": "google_search_result",
                    "call_id": "gs_1",
                    "result": [],
                },
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 1,
                "delta": {
                    "type": "google_search_result",
                    "is_error": False,
                    "result": [
                        {"url": "https://tokyo.example/news", "title": "Tokyo News"}
                    ],
                },
            }
        ),
        _sse({"event_type": "step.stop", "index": 1}),
        _sse(
            {"event_type": "step.start", "index": 2, "step": {"type": "model_output"}}
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 2,
                "delta": {"type": "text", "text": prefix},
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 2,
                "delta": {"type": "text", "text": target + suffix},
            }
        ),
    ]
    ann_event = _sse(
        {
            "event_type": "step.delta",
            "index": 3 if post_hoc_annotation_step else 2,
            "delta": {
                "type": "text_annotation_delta",
                "annotations": [
                    {
                        "type": "url_citation",
                        "start_index": sb,
                        "end_index": eb,
                        "url": "https://tokyo.example/news",
                        "title": "Tokyo News",
                    }
                ],
            },
        }
    )
    if post_hoc_annotation_step:
        # Exact live Vertex AI shape: text in index=2, annotation in empty index=3!
        events.extend(
            [
                _sse({"event_type": "step.stop", "index": 2}),
                _sse(
                    {
                        "event_type": "step.start",
                        "index": 3,
                        "step": {"type": "model_output"},
                    }
                ),
                ann_event,
                _sse({"event_type": "step.stop", "index": 3}),
            ]
        )
    else:
        events.extend([ann_event, _sse({"event_type": "step.stop", "index": 2})])

    events.append(
        _sse(
            {
                "event_type": "interaction.completed",
                "interaction": {"id": "", "status": "completed", "model": MODEL_NAME},
            }
        )
    )

    kw: dict[str, Any] = {"output_version": output_version} if output_version else {}
    llm, _, _, _ = _wired_llm(events, **kw)
    chunks = list(llm.stream("news"))
    full = chunks[0]
    for c in chunks[1:]:
        full = full + c

    assert full.text == full_text
    gm = full.response_metadata["grounding_metadata"]
    assert gm["web_search_queries"] == ["Tokyo cafe score"]
    assert gm["grounding_supports"] == [
        {
            "segment": {
                "start_index": len(prefix),
                "end_index": len(prefix) + len(target),
                "text": target,
            },
            "grounding_chunk_indices": [0],
        }
    ]
    text_blocks = [b for b in full.content_blocks if b["type"] == "text"]
    assert len(text_blocks) == 1
    cit = text_blocks[0]["annotations"][0]
    assert cit["start_index"] == len(prefix)  # type: ignore[typeddict-item]
    assert cit["end_index"] == len(prefix) + len(target)  # type: ignore[typeddict-item]
    assert cit["cited_text"] == target  # type: ignore[typeddict-item]
    assert full_text[cit["start_index"] : cit["end_index"]] == target  # type: ignore[typeddict-item]

    if output_version == "interactions/v1":
        # Google search server tool steps preserved & replayable!
        assert [b["type"] for b in full.content_blocks] == [
            "server_tool_call",
            "server_tool_result",
            "text",
        ]
        _, steps = _messages_to_interaction_steps(
            [HumanMessage("q"), full, HumanMessage("followup")],
            effective_model=MODEL_NAME,
        )
        assert [s["type"] for s in steps] == [
            "user_input",
            "google_search_call",
            "google_search_result",
            "model_output",
            "user_input",
        ]


@pytest.mark.parametrize("output_version", [None, "interactions/v1", "v1"])
def test_server_tools_media_audio_and_structured_output_streaming(
    output_version: str | None,
) -> None:
    class CalcAnswer(BaseModel):
        total: int

    pcm1 = b"\x01\x02" * 20
    pcm2 = b"\x03\x04" * 20
    img1 = "QUFB"  # 3 bytes "AAA" -> clean base64 chunk 1
    img2 = "QkJC"  # 3 bytes "BBB" -> clean base64 chunk 2

    events = [
        _sse(
            {
                "event_type": "interaction.created",
                "interaction": {
                    "id": "int_srv",
                    "status": "in_progress",
                    "model": MODEL_NAME,
                },
            }
        ),
        _sse(
            {
                "event_type": "step.start",
                "index": 0,
                "step": {"type": "processing_call", "id": "proc_1"},
            }
        ),
        _sse({"event_type": "step.stop", "index": 0}),
        _sse(
            {
                "event_type": "step.start",
                "index": 1,
                "step": {"type": "processing_result", "call_id": "proc_1"},
            }
        ),
        _sse({"event_type": "step.stop", "index": 1}),
        _sse(
            {
                "event_type": "step.start",
                "index": 2,
                "step": {"type": "code_execution_call", "id": "ce_1", "arguments": {}},
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 2,
                "delta": {
                    "type": "code_execution_call",
                    "arguments": {"code": "print(385)", "language": "python"},
                },
            }
        ),
        _sse({"event_type": "step.stop", "index": 2}),
        _sse(
            {
                "event_type": "step.start",
                "index": 3,
                "step": {
                    "type": "code_execution_result",
                    "call_id": "ce_1",
                    "result": "",
                },
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 3,
                "delta": {
                    "type": "code_execution_result",
                    "result": "385\n",
                    "is_error": False,
                },
            }
        ),
        _sse({"event_type": "step.stop", "index": 3}),
        _sse(
            {
                "event_type": "step.start",
                "index": 4,
                "step": {
                    "type": "mcp_server_tool_call",
                    "id": "mcp_1",
                    "name": "lookup",
                    "server_name": "docs",
                    "arguments": {"q": "385"},
                },
            }
        ),
        _sse({"event_type": "step.stop", "index": 4}),
        _sse(
            {
                "event_type": "step.start",
                "index": 5,
                "step": {
                    "type": "mcp_server_tool_result",
                    "call_id": "mcp_1",
                    "name": "lookup",
                    "server_name": "docs",
                    "result": "verified",
                },
            }
        ),
        _sse({"event_type": "step.stop", "index": 5}),
        _sse(
            {"event_type": "step.start", "index": 6, "step": {"type": "model_output"}}
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 6,
                "delta": {"type": "text", "text": '{"tot'},
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 6,
                "delta": {"type": "text", "text": 'al": 385}'},
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 6,
                "delta": {"type": "image", "data": img1, "mime_type": "image/png"},
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 6,
                "delta": {"type": "image", "data": img2, "mime_type": "image/png"},
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 6,
                "delta": {
                    "type": "audio",
                    "data": base64.b64encode(pcm1).decode("ascii"),
                    "mime_type": "audio/l16",
                    "sample_rate": 24000,
                    "channels": 1,
                },
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 6,
                "delta": {
                    "type": "audio",
                    "data": base64.b64encode(pcm2).decode("ascii"),
                    "mime_type": "audio/l16",
                },
            }
        ),
        _sse({"event_type": "step.stop", "index": 6}),
        _sse(
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "int_srv",
                    "status": "incomplete",
                    "model": MODEL_NAME,
                },
            }
        ),
    ]

    kw: dict[str, Any] = {"output_version": output_version} if output_version else {}
    llm, _, _, _ = _wired_llm(events, **kw)
    chunks = list(llm.stream("calc", response_format=CalcAnswer))
    full = chunks[0]
    for c in chunks[1:]:
        full = full + c

    assert full.response_metadata["finish_reason"] == "MAX_TOKENS"
    assert full.additional_kwargs["parsed"] == CalcAnswer(total=385)
    # Multi-chunk PCM audio wrapped into a SINGLE valid RIFF WAV without TypeError!
    wav_bytes = full.additional_kwargs["audio"]
    assert wav_bytes.startswith(b"RIFF")
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.readframes(wf.getnframes()) == pcm1 + pcm2

    # Multi-chunk image mime_type is NOT concatenated ("image/png" stays clean)
    img_blocks = [b for b in full.content_blocks if b["type"] == "image"]
    assert len(img_blocks) == 1
    assert img_blocks[0]["mime_type"] == "image/png"
    assert img_blocks[0]["base64"] == img1 + img2

    # Replay strips media_processing while preserving code_execution & mcp_server!
    _, steps = _messages_to_interaction_steps(
        [HumanMessage("calc"), full, HumanMessage("next")],
        effective_model=MODEL_NAME,
    )
    step_types = [s["type"] for s in steps]
    assert "code_execution_call" in step_types
    assert "code_execution_result" in step_types
    assert "mcp_server_tool_call" in step_types
    assert "mcp_server_tool_result" in step_types
    assert "processing_call" not in step_types
    assert "processing_result" not in step_types


# ---------------------------------------------------------------------------
# 4. stream_events(version="v3") & astream_events(version="v3") validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_version", [None, "v0", "interactions/v1", "v1"])
async def test_stream_events_v3_sync_and_async_valid_protocol_and_replay(
    output_version: str | None,
) -> None:
    sig_b64 = base64.b64encode(b"v3-sig").decode("ascii")
    events = [
        _sse(
            {
                "event_type": "interaction.created",
                "interaction": {
                    "id": "v1_v3_run",
                    "status": "in_progress",
                    "model": MODEL_NAME,
                },
            }
        ),
        _sse({"event_type": "step.start", "index": 0, "step": {"type": "thought"}}),
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {
                    "type": "thought_summary",
                    "content": {"type": "text", "text": "Reasoning step."},
                },
            }
        ),
        _sse(
            {"event_type": "step.start", "index": 1, "step": {"type": "model_output"}}
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 1,
                "delta": {"type": "text", "text": "Hi "},
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 1,
                "delta": {"type": "text", "text": "there!"},
            }
        ),
        _sse({"event_type": "step.stop", "index": 1}),
        _sse(
            {
                "event_type": "step.start",
                "index": 2,
                "step": {
                    "type": "function_call",
                    "id": "call_v3_1",
                    "name": "ping_tool",
                    "arguments": {},
                },
            }
        ),
        _sse(
            {
                "event_type": "step.start",
                "index": 3,
                "step": {
                    "type": "function_call",
                    "id": "call_v3_0arg",
                    "name": "noop_tool",
                    "arguments": {},
                },
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 2,
                "delta": {"type": "arguments_delta", "arguments": '{"target":'},
            }
        ),
        # Late signature arrives mid-stream!
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_signature", "signature": sig_b64},
            }
        ),
        _sse({"event_type": "step.stop", "index": 0}),
        _sse(
            {
                "event_type": "step.delta",
                "index": 2,
                "delta": {"type": "arguments_delta", "arguments": '"db"}'},
            }
        ),
        _sse({"event_type": "step.stop", "index": 2}),
        _sse({"event_type": "step.stop", "index": 3}),
        _sse(
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "v1_v3_run",
                    "status": "requires_action",
                    "model": MODEL_NAME,
                    "usage": {
                        "total_input_tokens": 10,
                        "total_output_tokens": 15,
                        "total_thought_tokens": 5,
                        "total_tokens": 30,
                    },
                },
            }
        ),
    ]

    kw: dict[str, Any] = {"store": True}
    if output_version is not None:
        kw["output_version"] = output_version
    llm, _, s_stream, a_stream = _wired_llm(events, **kw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangChainBetaWarning)
        v3_sync: Any = llm.stream_events("check", version="v3")
        raw_sync_events = list(v3_sync)

    assert s_stream.closed is True
    assert_valid_event_stream(raw_sync_events)
    assert list(v3_sync.text) == ["Hi ", "there!"]
    assert str(v3_sync.reasoning) == "Reasoning step."
    tc_deltas = list(v3_sync.tool_calls)
    assert len(tc_deltas) >= 2
    out_msg = v3_sync.output
    assert isinstance(out_msg, AIMessage)
    assert out_msg.response_metadata["output_version"] == "v1"
    assert out_msg.response_metadata["model_name"] == MODEL_NAME
    assert out_msg.response_metadata["interaction_id"] == "v1_v3_run"
    assert out_msg.response_metadata["stored"] is True
    assert out_msg.tool_calls == [
        {
            "id": "call_v3_1",
            "name": "ping_tool",
            "args": {"target": "db"},
            "type": "tool_call",
        },
        {"id": "call_v3_0arg", "name": "noop_tool", "args": {}, "type": "tool_call"},
    ]
    assert out_msg.additional_kwargs[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY] == {
        "call_v3_1": sig_b64,
        "call_v3_0arg": sig_b64,
    }

    # Verify v3 assembled AIMessage replays cleanly into _messages_to_interaction_steps!
    _, steps_v3 = _messages_to_interaction_steps(
        [
            HumanMessage("check"),
            out_msg,
            ToolMessage("ok", tool_call_id="call_v3_1"),
            ToolMessage("done", tool_call_id="call_v3_0arg"),
        ],
        effective_model=MODEL_NAME,
    )
    assert steps_v3[1] == {
        "type": "thought",
        "signature": sig_b64,
        "summary": [{"type": "text", "text": "Reasoning step."}],
    }
    _serialize_sdk_body({"model": MODEL_NAME, "input": steps_v3, "store": False})

    # Async astream_events(version="v3") parity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangChainBetaWarning)
        v3_async: Any = await llm.astream_events("check", version="v3")
        raw_async_events = [ev async for ev in v3_async]

    assert a_stream.closed is True
    assert_valid_event_stream(raw_async_events)
    a_out = await v3_async.output
    assert a_out.tool_calls == out_msg.tool_calls
    assert a_out.usage_metadata == {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "input_token_details": {"cache_read": 0},
        "output_token_details": {"reasoning": 5},
    }


# ---------------------------------------------------------------------------
# 5. Mid-stream errors, timeout, resource cleanup, finish_reasons, background
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("err_code", "err_msg", "expected_exc"),
    [
        ("invalid_request", "Thinking level is unsupported", GoogleInvalidRequestError),
        ("invalid_argument", "Bad schema", GoogleInvalidRequestError),
        (
            "invalid_argument",
            "The input token count exceeds maximum context length",
            GoogleInteractionsContextOverflowError,
        ),
        ("unauthenticated", "Bad key", GoogleAuthenticationError),
        ("permission_denied", "No access", GooglePermissionDeniedError),
        ("not_found", "Unknown model", GoogleModelNotFoundError),
        ("resource_exhausted", "Quota hit", GoogleRateLimitError),
        ("deadline_exceeded", "Backend timed out", GoogleTimeoutError),
        ("internal", "Backend crash", GoogleInteractionsAPIError),
    ],
)
def test_midstream_sse_error_event_classification_and_cleanup(
    err_code: str, err_msg: str, expected_exc: type[Exception]
) -> None:
    events = [
        _sse(
            {
                "event_type": "interaction.created",
                "interaction": {"id": "", "status": "in_progress", "model": MODEL_NAME},
            }
        ),
        _sse({"event_type": "error", "error": {"code": err_code, "message": err_msg}}),
    ]
    llm, _, s_stream, _ = _wired_llm(events)
    with pytest.raises(expected_exc):
        list(llm.stream("fail"))
    assert s_stream.closed is True


@pytest.mark.parametrize("bad_status", ["failed", "cancelled"])
def test_midstream_status_update_and_completed_failure_raises(bad_status: str) -> None:
    ev_upd = [
        _sse(
            {
                "event_type": "interaction.status_update",
                "interaction_id": "id_1",
                "status": bad_status,
            }
        )
    ]
    llm1, _, s1, _ = _wired_llm(ev_upd)
    with pytest.raises(GoogleInteractionsAPIError):
        list(llm1.stream("x"))
    assert s1.closed is True

    ev_comp = [
        _sse(
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "id_2",
                    "status": bad_status,
                    "model": MODEL_NAME,
                },
            }
        )
    ]
    llm2, _, s2, _ = _wired_llm(ev_comp)
    with pytest.raises(GoogleInteractionsAPIError):
        list(llm2.stream("x"))
    assert s2.closed is True


@pytest.mark.parametrize(
    ("terminal_status", "expected_reason"),
    [
        ("completed", "STOP"),
        ("requires_action", "STOP"),
        ("incomplete", "MAX_TOKENS"),
        ("budget_exceeded", "MAX_TOKENS"),
        ("requires_continuation", "MAX_TOKENS"),
    ],
)
def test_finish_reason_mapping_and_eof_fallback(
    terminal_status: str, expected_reason: str
) -> None:
    # Unrecognized status strings like "requires_continuation" parse cleanly
    events = [
        _sse(
            {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}}
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "text", "text": "ok"},
            }
        ),
        _sse(
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "",
                    "status": terminal_status,
                    "model": MODEL_NAME,
                },
            }
        ),
    ]
    llm, _, _, _ = _wired_llm(events)
    chunks = list(llm.stream("hi"))
    full = chunks[0]
    for c in chunks[1:]:
        full = full + c
    assert full.response_metadata["finish_reason"] == expected_reason
    assert chunks[-1].chunk_position == "last"

    # Early EOF without interaction.completed still sets chunk_position='last' + usage
    eof_events = [
        _sse(
            {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}}
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "text", "text": "partial"},
            }
        ),
        {
            "event_type": "step.stop",
            "index": 0,
            "usage": {"total_input_tokens": 4, "total_output_tokens": 2},
        },
    ]
    llm_eof, _, _, _ = _wired_llm(eof_events)
    eof_chunks = list(llm_eof.stream("hi"))
    assert eof_chunks[-1].chunk_position == "last"
    assert eof_chunks[-1].usage_metadata is not None
    assert eof_chunks[-1].usage_metadata["input_tokens"] == 4


async def test_stream_close_on_early_break_silence_timeout_and_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        _sse(
            {
                "event_type": "interaction.created",
                "interaction": {"id": "", "status": "in_progress", "model": MODEL_NAME},
            }
        ),
        _sse(
            {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}}
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "text", "text": "one"},
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "text", "text": "two"},
            }
        ),
    ]
    llm, _, s_stream, a_stream = _wired_llm(
        events,
        async_stream=_FakeAsyncStream(events[:2], hang_after_events=True),
    )

    # Early break closes underlying sync stream when generator is closed
    gen = llm.stream("hi")
    _ = next(gen)
    cast("Any", gen).close()
    assert s_stream.closed is True

    # Configurable LANGCHAIN_GOOGLE_STREAM_CHUNK_TIMEOUT_S raises GoogleTimeoutError
    monkeypatch.setenv("LANGCHAIN_GOOGLE_STREAM_CHUNK_TIMEOUT_S", "0.02")
    with pytest.raises(GoogleTimeoutError, match=r"0\.02s"):
        _ = [c async for c in llm.astream("hi")]
    assert a_stream.closed is True

    # Mid-stream httpx transport exceptions map cleanly to GoogleTimeout/ConnectionError
    t_stream = _FakeSyncStream(
        events[:1], error_after=httpx.ReadTimeout("read timeout")
    )
    llm_t, _, _, _ = _wired_llm(sync_stream=t_stream)
    with pytest.raises(GoogleTimeoutError):
        list(llm_t.stream("hi"))
    assert t_stream.closed is True

    c_stream = _FakeAsyncStream(events[:1], error_after=httpx.ReadError("socket reset"))
    llm_c, _, _, _ = _wired_llm(async_stream=c_stream)
    with pytest.raises(GoogleConnectionError):
        _ = [c async for c in llm_c.astream("hi")]
    assert c_stream.closed is True


async def test_background_stream_and_astream_polls_and_yields_last_chunk() -> None:
    in_prog = Interaction.model_validate(
        {"id": "bg_1", "model": MODEL_NAME, "status": "in_progress", "steps": []}
    )
    done = Interaction.model_validate(
        {
            "id": "bg_1",
            "model": MODEL_NAME,
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Done bg!"}],
                }
            ],
            "usage": {
                "total_input_tokens": 5,
                "total_output_tokens": 3,
                "total_tokens": 8,
            },
        }
    )

    llm_s, ia_s, _, _ = _wired_llm(
        sync_stream=in_prog, get_items=[done], background=True, polling_interval=0.001
    )
    s_chunks = list(llm_s.stream("research"))
    assert len(s_chunks) == 1
    assert s_chunks[0].text == "Done bg!"
    assert s_chunks[0].chunk_position == "last"
    assert s_chunks[0].response_metadata["stored"] is True
    assert len(ia_s.get_calls) == 1

    llm_a, ia_a, _, _ = _wired_llm(
        async_stream=in_prog, get_items=[done], background=True, polling_interval=0.001
    )
    a_chunks = [c async for c in llm_a.astream("research")]
    assert len(a_chunks) == 1
    assert a_chunks[0].text == "Done bg!"
    assert a_chunks[0].chunk_position == "last"
    assert len(ia_a.get_calls) == 1

    # `poll=False` yields initial in_progress chunk immediately without calling .get
    llm_np_s, ia_np_s, _, _ = _wired_llm(
        sync_stream=in_prog,
        agent="deep-research-preview-04-2026",
        background=True,
        store=True,
    )
    np_s = list(llm_np_s.stream("research", poll=False))
    assert len(np_s) == 1
    assert np_s[0].chunk_position == "last"
    assert np_s[0].response_metadata["status"] == "in_progress"
    assert ia_np_s.get_calls == []

    llm_np_a, ia_np_a, _, _ = _wired_llm(
        async_stream=in_prog,
        agent="deep-research-preview-04-2026",
        background=True,
        store=True,
    )
    np_a = [c async for c in llm_np_a.astream("research", poll=False)]
    assert len(np_a) == 1
    assert np_a[0].chunk_position == "last"
    assert np_a[0].response_metadata["status"] == "in_progress"
    assert ia_np_a.get_calls == []


def test_maps_url_context_file_search_late_id_and_translator_annotations() -> None:
    class SchemaOut(BaseModel):
        val: str

    # 1) Maps + url_context + file_search + late ID + tool call suppresses parsed
    events: list[Any] = [
        {
            "event_type": "interaction.created",
            "interaction": {"id": None, "status": None, "model": MODEL_NAME},
        },
        _sse(
            {
                "event_type": "step.start",
                "index": 0,
                "step": {
                    "type": "url_context_call",
                    "id": "uc_1",
                    "arguments": {"urls": ["https://a.example"]},
                },
            }
        ),
        _sse({"event_type": "step.stop", "index": 0}),
        _sse(
            {
                "event_type": "step.start",
                "index": 1,
                "step": {
                    "type": "url_context_result",
                    "call_id": "uc_1",
                    "result": [{"url": "https://a.example", "status": "success"}],
                },
            }
        ),
        _sse({"event_type": "step.stop", "index": 1}),
        _sse(
            {
                "event_type": "step.start",
                "index": 2,
                "step": {"type": "google_maps_call", "id": "gm_1"},
            }
        ),
        _sse({"event_type": "step.stop", "index": 2}),
        _sse(
            {
                "event_type": "step.start",
                "index": 3,
                "step": {
                    "type": "google_maps_result",
                    "call_id": "gm_1",
                    "result": [],
                },
            }
        ),
        _sse(
            {
                "event_type": "step.delta",
                "index": 3,
                "delta": {
                    "type": "google_maps_result",
                    "result": [
                        {
                            "widget_context_token": "tok_xyz",
                            "places": [
                                {
                                    "name": "Cafe Luna",
                                    "place_id": "pl_1",
                                    "url": "https://maps.example/1",
                                }
                            ],
                        }
                    ],
                },
            }
        ),
        _sse({"event_type": "step.stop", "index": 3}),
        _sse(
            {
                "event_type": "step.start",
                "index": 4,
                "step": {"type": "file_search_call", "id": "fs_1"},
            }
        ),
        _sse({"event_type": "step.stop", "index": 4}),
        _sse(
            {
                "event_type": "step.start",
                "index": 5,
                "step": {"type": "file_search_result", "call_id": "fs_1"},
            }
        ),
        _sse({"event_type": "step.stop", "index": 5}),
        _sse(
            {
                "event_type": "step.start",
                "index": 6,
                "step": {
                    "type": "function_call",
                    "id": "fc_x",
                    "name": "act",
                    "arguments": {"k": 1.0},
                },
            }
        ),
        _sse({"event_type": "step.stop", "index": 6}),
        _sse(
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "late_id_99",
                    "status": "requires_action",
                    "model": MODEL_NAME,
                },
            }
        ),
    ]
    llm, _, _, _ = _wired_llm(events, store=True, output_version="interactions/v1")
    chunks = list(llm.stream("maps", response_format=SchemaOut))
    full = chunks[0]
    for c in chunks[1:]:
        full = full + c

    assert full.id == "late_id_99"
    assert full.response_metadata["id"] == "late_id_99"
    assert full.response_metadata["interaction_id"] == "late_id_99"
    assert full.response_metadata["stored"] is True
    assert "parsed" not in full.additional_kwargs
    assert full.tool_calls == [
        {"name": "act", "args": {"k": 1}, "id": "fc_x", "type": "tool_call"}
    ]
    gm = full.response_metadata["grounding_metadata"]
    assert gm["google_maps_widget_context_token"] == "tok_xyz"
    assert gm["grounding_chunks"] == [
        {
            "maps": {
                "uri": "https://maps.example/1",
                "title": "Cafe Luna",
                "placeId": "pl_1",
            }
        }
    ]
    _, steps = _messages_to_interaction_steps(
        [HumanMessage("maps"), full, ToolMessage("done", tool_call_id="fc_x")],
        effective_model=MODEL_NAME,
    )
    stypes = [s["type"] for s in steps]
    assert "url_context_call" in stypes
    assert "google_maps_call" in stypes
    assert "file_search_call" in stypes

    # 2) Direct translator normalizes raw UTF-8 place_citation & file_citation
    txt = "Visit ☕ Luna and page 3."
    sb_l = len("Visit ☕ ".encode())
    eb_l = sb_l + len(b"Luna")
    msg = AIMessage(
        content=[
            {
                "type": "text",
                "text": txt,
                "annotations": [
                    {
                        "type": "place_citation",
                        "start_index": sb_l,
                        "end_index": eb_l,
                        "name": "Luna",
                        "place_id": "pl_1",
                        "url": "https://maps.example/1",
                    },
                    {
                        "type": "file_citation",
                        "start_index": 0,
                        "end_index": 5,
                        "document_uri": "files/doc1",
                        "file_name": "guide.pdf",
                        "page_number": 3,
                    },
                ],
            }
        ],
        response_metadata={
            "model_provider": "google_genai",
            "output_version": "interactions/v1",
        },
    )
    anns = msg.content_blocks[0]["annotations"]  # type: ignore[typeddict-item]
    assert anns[0]["cited_text"] == "Luna"  # type: ignore[typeddict-item]
    assert anns[0]["extras"]["google_ai_metadata"] == {"place_id": "pl_1"}  # type: ignore[typeddict-item]
    assert anns[1]["cited_text"] == "Visit"  # type: ignore[typeddict-item]
    assert anns[1]["extras"]["file_citation_metadata"]["page_number"] == 3  # type: ignore[typeddict-item]


def test_compat_translator_without_core_private_refusal_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_compat` import and refusal/v1 translation work on core 1.7.4+."""
    # Simulate `langchain-core==1.7.4` where `_refusal_from_metadata` does not exist.
    monkeypatch.delattr(_core_genai_translator, "_refusal_from_metadata", raising=False)
    reloaded = importlib.reload(_compat_mod)
    assert reloaded is _compat_mod

    # 1) Interactions v1 blocked candidate (enum and string finish reasons) + chunk
    blocked_ia = AIMessage(
        content=[{"type": "text", "text": "Partial."}],
        response_metadata={
            "model_provider": "google_genai",
            "output_version": "interactions/v1",
            "finish_reason": FinishReason.SAFETY,
            "finish_message": "Blocked for safety.",
        },
    )
    assert blocked_ia.content_blocks == [
        {"type": "text", "text": "Partial."},
        {
            "type": "refusal",
            "refusal": "Blocked for safety.",
            "extras": {"finish_reason": "SAFETY"},
        },
    ]

    blocked_chunk = AIMessageChunk(
        content="",
        chunk_position="last",
        response_metadata={
            "model_provider": "google_genai",
            "output_version": "interactions/v1",
            "finish_reason": "PROHIBITED_CONTENT",
            "prompt_feedback": {"block_reason": "BLOCKLIST"},
        },
    )
    assert blocked_chunk.content_blocks == [
        {
            "type": "refusal",
            "refusal": "",
            "extras": {
                "finish_reason": "PROHIBITED_CONTENT",
                "block_reason": "BLOCKLIST",
            },
        }
    ]

    # 2) Non-refusal finish/block reasons and pre-existing refusal blocks do not emit
    # spurious or duplicate refusal blocks.
    normal_ia = AIMessage(
        content=[{"type": "text", "text": "Ok."}],
        response_metadata={
            "model_provider": "google_genai",
            "output_version": "interactions/v1",
            "finish_reason": "STOP",
            "prompt_feedback": {"block_reason": "BLOCKED_REASON_UNSPECIFIED"},
        },
    )
    assert normal_ia.content_blocks == [{"type": "text", "text": "Ok."}]

    zero_block_ia = AIMessage(
        content="Fine.",
        response_metadata={
            "model_provider": "google_genai",
            "output_version": "interactions/v1",
            "finish_reason": "STOP",
            "prompt_feedback": {"block_reason": 0},
        },
    )
    assert zero_block_ia.content_blocks == [{"type": "text", "text": "Fine."}]

    existing_refusal = AIMessage(
        content=[
            {
                "type": "refusal",
                "refusal": "Already present.",
                "extras": {"finish_reason": "SAFETY"},
            }
        ],
        response_metadata={
            "model_provider": "google_genai",
            "output_version": "interactions/v1",
            "finish_reason": "SAFETY",
            "finish_message": "Duplicate.",
        },
    )
    assert existing_refusal.content_blocks == [
        {
            "type": "refusal",
            "refusal": "Already present.",
            "extras": {"finish_reason": "SAFETY"},
        }
    ]

    # 3) Legacy/v0 translation on core 1.7.4 (where `_convert_to_v1_from_genai` returns
    # content blocks without appending a refusal block).
    monkeypatch.setattr(
        _core_genai_translator,
        "_convert_to_v1_from_genai",
        lambda msg: [{"type": "text", "text": msg.content}] if msg.content else [],
    )
    blocked_v0 = AIMessage(
        content="",
        response_metadata={
            "model_provider": "google_genai",
            "prompt_feedback": {"block_reason": "SAFETY"},
        },
    )
    assert blocked_v0.content_blocks == [
        {
            "type": "refusal",
            "refusal": "",
            "extras": {"block_reason": "SAFETY"},
        }
    ]

    # 4) Fallback works even if core refactors away private `_convert_to_v1_from_genai`
    # and module-level `translate_content` / `translate_content_chunk`.
    monkeypatch.delattr(
        _core_genai_translator, "_convert_to_v1_from_genai", raising=False
    )
    monkeypatch.delattr(_core_genai_translator, "translate_content", raising=False)
    monkeypatch.delattr(
        _core_genai_translator, "translate_content_chunk", raising=False
    )
    refactored_v0 = AIMessage(
        content="Fallback text",
        response_metadata={
            "model_provider": "google_genai",
            "finish_reason": "BLOCKLIST",
            "finish_message": "Blocked term.",
        },
    )
    assert refactored_v0.content_blocks == [
        {"type": "text", "text": "Fallback text"},
        {
            "type": "refusal",
            "refusal": "Blocked term.",
            "extras": {"finish_reason": "BLOCKLIST"},
        },
    ]


# ---------------------------------------------------------------------------
# 10. Default streaming interop, multi-hop continuation, and edge-case suites
# ---------------------------------------------------------------------------


async def test_default_generate_content_streaming_tool_calls_zero_regression() -> None:
    """Default `generateContent` streaming keeps finalized `tool_call` blocks in v1."""
    from google.genai.types import (
        Candidate,
        Content,
        FunctionCall,
        GenerateContentResponse,
        Part,
    )

    chunk_1 = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            text="Planning call.",
                            thought=True,
                            thought_signature=b"gc_sig",
                        )
                    ],
                )
            )
        ]
    )
    chunk_2 = GenerateContentResponse(
        candidates=[
            Candidate(
                finish_reason=FinishReason.STOP,
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                name="lookup", args={"a": 1}, id="call_gc_1"
                            ),
                            thought_signature=b"gc_sig",
                        )
                    ],
                ),
            )
        ]
    )

    for ov in (None, "v1"):
        kw: dict[str, Any] = {"output_version": ov} if ov else {}
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=FAKE_KEY, **kw)
        assert llm.client is not None
        with patch.object(
            llm.client.models,
            "generate_content_stream",
            return_value=iter([chunk_1, chunk_2]),
        ):
            s_chunks = list(llm.stream("hi"))
        agg = s_chunks[0]
        for sc in s_chunks[1:]:
            agg = agg + sc
        blocks = agg.content if ov == "v1" else agg.content_blocks
        assert isinstance(blocks, list)
        assert not any(
            isinstance(b, Mapping) and b.get("type") == "tool_call_chunk"
            for b in blocks
        )
        tc_blocks: list[dict[str, Any]] = [
            dict(b)
            for b in blocks
            if isinstance(b, Mapping) and b.get("type") == "tool_call"
        ]
        assert len(tc_blocks) == 1
        assert tc_blocks[0]["id"] == "call_gc_1"
        assert tc_blocks[0]["name"] == "lookup"
        assert tc_blocks[0]["args"] == {"a": 1}

        # Async `astream` check
        async def _agen() -> Any:
            yield chunk_1
            yield chunk_2

        aio = llm.async_client
        with patch.object(
            aio.models, "generate_content_stream", AsyncMock(return_value=_agen())
        ):
            a_chunks = [c async for c in llm.astream("hi")]
        a_agg = a_chunks[0]
        for ac in a_chunks[1:]:
            a_agg = a_agg + ac
        a_blocks = a_agg.content if ov == "v1" else a_agg.content_blocks
        a_tc: list[dict[str, Any]] = [
            dict(b)
            for b in a_blocks
            if isinstance(b, Mapping) and b.get("type") == "tool_call"
        ]
        assert len(a_tc) == 1 and a_tc[0]["args"] == {"a": 1}


@pytest.mark.parametrize("output_version", [None, "interactions/v1", "v1"])
async def test_multi_hop_automatic_continuation_audio_v3_and_interrupted_error(
    output_version: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-hop `requires_continuation` stream merges ID, status, audio, and v3."""
    import wave

    from google.genai import errors as genai_errors
    from langchain_core._api import LangChainBetaWarning

    from langchain_google_genai import chat_models as cm_mod
    from langchain_google_genai.chat_models import GoogleRateLimitError

    pcm_h1 = b"\x11\x22" * 24
    pcm_h2 = b"\x33\x44" * 24
    # Use plain dict SSE payloads so SDK 2.25 accepts `"requires_continuation"` too!
    multi_hop_events: list[Any] = [
        {
            "event_type": "interaction.created",
            "interaction": {
                "id": "hop_1",
                "status": "in_progress",
                "model": MODEL_NAME,
            },
        },
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "thought"},
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {
                "type": "thought_summary",
                "content": {"type": "text", "text": "Thinking hop 1."},
            },
        },
        {"event_type": "step.stop", "index": 0},
        {"event_type": "step.start", "index": 1, "step": {"type": "model_output"}},
        {
            "event_type": "step.delta",
            "index": 1,
            "delta": {"type": "text", "text": "Hello from hop 1, "},
        },
        {
            "event_type": "step.delta",
            "index": 1,
            "delta": {
                "type": "audio",
                "data": base64.b64encode(pcm_h1).decode(),
                "mime_type": "audio/l16",
                "sample_rate": 24000,
                "channels": 1,
            },
        },
        {"event_type": "step.stop", "index": 1},
        {
            "event_type": "interaction.completed",
            "interaction": {
                "id": "hop_1",
                "status": "requires_continuation",
                "continuation_token": "tok_hop_1",
                "model": MODEL_NAME,
                "usage": {
                    "total_input_tokens": 20,
                    "total_output_tokens": 30,
                    "total_thought_tokens": 5,
                    "total_tokens": 55,
                },
            },
        },
        # Hop 2 starts immediately on same Stream with reset index=0 and parent hop_1
        {
            "event_type": "interaction.created",
            "interaction": {
                "id": "hop_2",
                "previous_interaction_id": "hop_1",
                "status": "in_progress",
                "model": MODEL_NAME,
            },
        },
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "text", "text": "and hop 2 done!"},
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {
                "type": "audio",
                "data": base64.b64encode(pcm_h2).decode(),
                "mime_type": "audio/l16",
                "sample_rate": 24000,
                "channels": 1,
            },
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "interaction.completed",
            "interaction": {
                "id": "hop_2",
                "previous_interaction_id": "hop_1",
                "status": "completed",
                "model": MODEL_NAME,
                "usage": {
                    "total_input_tokens": 50,
                    "total_output_tokens": 25,
                    "total_thought_tokens": 0,
                    "total_tokens": 75,
                },
            },
        },
    ]

    kw: dict[str, Any] = {"store": True, "use_previous_interaction_id": True}
    if output_version:
        kw["output_version"] = output_version
    llm, _, _, _ = _wired_llm(multi_hop_events, **kw)

    prior_turn = AIMessage(
        "Turn 0 reply",
        response_metadata={
            "model_provider": "google_genai",
            "stored": True,
            "interaction_id": "turn_0_root",
        },
    )
    chunks = list(llm.stream([HumanMessage("t0"), prior_turn, HumanMessage("t1")]))
    assert sum(1 for c in chunks if c.chunk_position == "last") == 1
    assert chunks[-1].chunk_position == "last"

    full = chunks[0]
    for ch in chunks[1:]:
        full = full + ch
    assert full.id == "hop_2"
    assert full.response_metadata["id"] == "hop_2"
    assert full.response_metadata["interaction_id"] == "hop_2"
    # Preserves caller's Turn 0 ID (NOT Hop 2's internal `hop_1` parent)!
    assert full.response_metadata["previous_interaction_id"] == "turn_0_root"
    assert full.response_metadata["continuation_ids"] == ["hop_1", "hop_2"]
    assert full.response_metadata["stored"] is True
    assert full.response_metadata["status"] == "completed"
    assert full.response_metadata["finish_reason"] == "STOP"
    assert full.text == "Hello from hop 1, and hop 2 done!"
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] == 70
    assert full.usage_metadata["output_tokens"] == 60
    assert full.usage_metadata["total_tokens"] == 130
    with wave.open(io.BytesIO(full.additional_kwargs["audio"]), "rb") as wf:
        assert wf.readframes(wf.getnframes()) == pcm_h1 + pcm_h2

    # Next stateful turn chains onto `hop_2`, NOT `hop_1`!
    from langchain_google_genai._interactions_request import (
        _get_last_interaction_messages,
    )

    next_id, next_tail = _get_last_interaction_messages(
        [HumanMessage("t1"), full, HumanMessage("t2")]
    )
    assert next_id == "hop_2" and len(next_tail) == 1

    # v3 `stream_events` validates strictly (single message-finish, monotonic blocks)
    llm_v3, _, _, _ = _wired_llm(multi_hop_events, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangChainBetaWarning)
        v3_events = list(llm_v3.stream_events("go", version="v3"))
    assert_valid_event_stream(v3_events)

    # Uncontinued `automatic_continuation=False` closes cleanly on EOF with token
    llm_uncont, _, _, _ = _wired_llm(multi_hop_events[:9], store=True)
    u_chunks = list(llm_uncont.stream("go", automatic_continuation=False))
    u_agg = u_chunks[0]
    for uc in u_chunks[1:]:
        u_agg = u_agg + uc
    assert u_agg.response_metadata["interaction_id"] == "hop_1"
    assert u_agg.response_metadata["status"] == "requires_continuation"
    assert u_agg.response_metadata["finish_reason"] == "MAX_TOKENS"
    assert u_agg.response_metadata["continuation_token"] == "tok_hop_1"

    # Mid-stream `ContinuationInterruptedError` classified cleanly
    from google.genai._gaos.lib.compat_errors import RateLimitError

    from langchain_google_genai import _interactions_stream as s_mod

    cont_cls = getattr(genai_errors, "ContinuationInterruptedError", None)
    if cont_cls is None:

        class _FakeStreamContErr(Exception):
            pass

        cont_cls = _FakeStreamContErr
        monkeypatch.setattr(cm_mod, "_ContinuationInterruptedError", cont_cls)
        monkeypatch.setattr(
            s_mod,
            "_INTERACTIONS_CATCHABLE_ERRORS",
            (*cm_mod._INTERACTIONS_CATCHABLE_ERRORS, cont_cls),
        )

    req_h = httpx.Request("POST", "https://example.com")
    resp_429 = httpx.Response(429, json={"error": {"message": "429"}}, request=req_h)
    mid_err = cont_cls("Hop 2 failed")
    mid_err.__cause__ = RateLimitError("429", response=resp_429, body=None)

    class _InterruptingStream:
        def __iter__(self) -> Any:
            yield multi_hop_events[0]
            raise mid_err

        def close(self) -> None:
            pass

    llm_err = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=FAKE_KEY, use_interactions_api=True
    )
    fake_err_ia = Mock(create=Mock(return_value=_InterruptingStream()))
    object.__setattr__(llm_err.client, "_interactions", fake_err_ia)
    with pytest.raises(GoogleRateLimitError) as ce_info:
        list(llm_err.stream("go"))
    assert ce_info.value.__cause__ is mid_err


@pytest.mark.parametrize("output_version", [None, "v1", "interactions/v1"])
def test_stream_and_v1_multi_segment_parity_persisted_blocks_and_server_boundary(
    output_version: str | None,
) -> None:
    """Multi-segment `[thought_1, fc_1, thought_2, model_output, fc_2]` + v1 blocks."""
    from langchain_core._api import LangChainBetaWarning

    from langchain_google_genai._interactions_response import _interaction_to_result

    s1 = base64.b64encode(b"sig-s1").decode("ascii")
    s2 = base64.b64encode(b"sig-s2").decode("ascii")

    # 1) Unary & Stream parity + exact step order across v0, v1, interactions/v1
    steps_raw = [
        {
            "type": "thought",
            "signature": s1,
            "summary": [{"type": "text", "text": "T1"}],
        },
        {"type": "function_call", "id": "fc1", "name": "f1", "arguments": {"a": 1.0}},
        {
            "type": "thought",
            "signature": s2,
            "summary": [{"type": "text", "text": "T2"}],
        },
        {"type": "model_output", "content": [{"type": "text", "text": "Halfway"}]},
        {"type": "function_call", "id": "fc2", "name": "f2", "arguments": {"b": 2.0}},
    ]
    ai_unary = (
        _interaction_to_result(
            {
                "id": "i_p1",
                "model": MODEL_NAME,
                "status": "requires_action",
                "steps": steps_raw,
            },
            output_version=output_version,
        )
        .generations[0]
        .message
    )

    sse_events: list[Any] = []
    for idx_s, st_obj in enumerate(steps_raw):
        sse_events.append({"event_type": "step.start", "index": idx_s, "step": st_obj})
        sse_events.append({"event_type": "step.stop", "index": idx_s})
    sse_events.append(
        {
            "event_type": "interaction.completed",
            "interaction": {
                "id": "i_p1",
                "model": MODEL_NAME,
                "status": "requires_action",
            },
        }
    )

    kw: dict[str, Any] = {"output_version": output_version} if output_version else {}
    llm, _, _, _ = _wired_llm(sse_events, **kw)
    s_chunks = list(llm.stream("go"))
    ai_stream = s_chunks[0]
    for sc in s_chunks[1:]:
        ai_stream = ai_stream + sc

    # Parity on legacy function_call + int-float coercion
    assert (
        ai_stream.additional_kwargs["function_call"]
        == ai_unary.additional_kwargs["function_call"]
    )
    assert ai_stream.additional_kwargs["function_call"] == {
        "name": "f2",
        "arguments": '{"b": 2}',
    }

    if output_version == "interactions/v1":
        assert [b["type"] for b in ai_stream.content if isinstance(b, Mapping)] == [
            "reasoning",
            "tool_call",
            "reasoning",
            "text",
            "tool_call",
        ]
        llm_v3, _, _, _ = _wired_llm(sse_events, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LangChainBetaWarning)
            assert_valid_event_stream(list(llm_v3.stream_events("go", version="v3")))

    for candidate in (ai_unary, ai_stream):
        _, rep = _messages_to_interaction_steps(
            [
                HumanMessage("go"),
                candidate,
                ToolMessage("1", tool_call_id="fc1"),
                ToolMessage("2", tool_call_id="fc2"),
            ],
            effective_model=MODEL_NAME,
        )
        assert [s["type"] for s in rep[1:-2]] == [
            "thought",
            "function_call",
            "thought",
            "model_output",
            "function_call",
        ]
        assert rep[1]["signature"] == s1 and rep[3]["signature"] == s2

    # 2) Unary `v1` includes `extras.signature` on `tool_call` blocks & replays from
    # persisted `content_blocks` alone (without private `additional_kwargs`)!
    if output_version == "v1":
        v1_sig_only = (
            _interaction_to_result(
                {
                    "id": "i_v1_persist",
                    "model": MODEL_NAME,
                    "status": "requires_action",
                    "steps": [
                        {"type": "thought", "signature": s1},
                        {
                            "type": "function_call",
                            "id": "ca",
                            "name": "fa",
                            "arguments": {},
                        },
                        {"type": "thought", "signature": s2},
                        {
                            "type": "function_call",
                            "id": "cb",
                            "name": "fb",
                            "arguments": {},
                        },
                    ],
                },
                output_version="v1",
            )
            .generations[0]
            .message
        )
        assert isinstance(v1_sig_only, AIMessage)
        tc_v1: list[dict[str, Any]] = [
            dict(b)
            for b in v1_sig_only.content
            if isinstance(b, Mapping) and b.get("type") == "tool_call"
        ]
        assert tc_v1[0]["extras"]["signature"] == s1
        assert tc_v1[1]["extras"]["signature"] == s2
        restored = AIMessage(
            content=v1_sig_only.content,
            tool_calls=v1_sig_only.tool_calls,
            response_metadata={"model_provider": "google_genai"},
        )
        _, rep_p = _messages_to_interaction_steps(
            [
                HumanMessage("q"),
                restored,
                ToolMessage("a", tool_call_id="ca"),
                ToolMessage("b", tool_call_id="cb"),
            ],
            effective_model=MODEL_NAME,
        )
        assert rep_p[1:-2] == [
            {"type": "thought", "signature": s1},
            {"type": "function_call", "id": "ca", "name": "fa", "arguments": {}},
            {"type": "thought", "signature": s2},
            {"type": "function_call", "id": "cb", "name": "fb", "arguments": {}},
        ]

    # 3) Streamed `[thought(s1), code_exec, code_res, model_output, fc]` never
    # duplicates `thought(s1)`!
    ce_events: list[Any] = [
        {
            "event_type": "step.start",
            "index": 0,
            "step": {
                "type": "thought",
                "signature": s1,
                "summary": [{"type": "text", "text": "Run python"}],
            },
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "step.start",
            "index": 1,
            "step": {
                "type": "code_execution_call",
                "id": "ce_s",
                "arguments": {"code": "1+1", "language": "python"},
            },
        },
        {"event_type": "step.stop", "index": 1},
        {
            "event_type": "step.start",
            "index": 2,
            "step": {
                "type": "code_execution_result",
                "call_id": "ce_s",
                "result": "2",
            },
        },
        {"event_type": "step.stop", "index": 2},
        {
            "event_type": "step.start",
            "index": 3,
            "step": {
                "type": "model_output",
                "content": [{"type": "text", "text": "2"}],
            },
        },
        {"event_type": "step.stop", "index": 3},
        {
            "event_type": "step.start",
            "index": 4,
            "step": {
                "type": "function_call",
                "id": "fc_fin",
                "name": "fin",
                "arguments": {},
            },
        },
        {"event_type": "step.stop", "index": 4},
        {
            "event_type": "interaction.completed",
            "interaction": {
                "id": "i_ce",
                "model": MODEL_NAME,
                "status": "requires_action",
            },
        },
    ]
    llm_ce, _, _, _ = _wired_llm(ce_events, **kw)
    ce_chunks = list(llm_ce.stream("calc"))
    ce_msg = ce_chunks[0]
    for cc in ce_chunks[1:]:
        ce_msg = ce_msg + cc
    assert _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY not in ce_msg.additional_kwargs
    _, rep_ce = _messages_to_interaction_steps(
        [HumanMessage("calc"), ce_msg, ToolMessage("ok", tool_call_id="fc_fin")],
        effective_model=MODEL_NAME,
    )
    from langchain_google_genai._interactions_request import _SKIP_THOUGHT_SIGNATURE_B64

    assert [s for s in rep_ce if s["type"] == "thought"] == [
        {
            "type": "thought",
            "signature": s1,
            "summary": [{"type": "text", "text": "Run python"}],
        },
        {"type": "thought", "signature": _SKIP_THOUGHT_SIGNATURE_B64},
    ]


async def test_echoed_input_interleaved_media_poll_timeout_and_citations() -> None:
    """Echoed input drop, `[text, image, text]` indices, poll timeout, & citations."""
    from langchain_core._api import LangChainBetaWarning

    from langchain_google_genai._interactions_response import _interaction_to_result

    img_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
    events: list[Any] = [
        # Echoed user_input + function_result deltas MUST NOT leak into output!
        {"event_type": "step.start", "index": 0, "step": {"type": "user_input"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "text", "text": "SECRET USER PROMPT "},
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "image", "data": img_b64, "mime_type": "image/png"},
        },
        {"event_type": "step.stop", "index": 0},
        {"event_type": "step.start", "index": 1, "step": {"type": "function_result"}},
        {
            "event_type": "step.delta",
            "index": 1,
            "delta": {"type": "text", "text": "SECRET TOOL OUTPUT "},
        },
        {"event_type": "step.stop", "index": 1},
        # Single model_output with interleaved [text("A"), image, text("B")]
        {"event_type": "step.start", "index": 2, "step": {"type": "model_output"}},
        {
            "event_type": "step.delta",
            "index": 2,
            "delta": {"type": "text", "text": "A"},
        },
        {
            "event_type": "step.delta",
            "index": 2,
            "delta": {"type": "image", "data": img_b64, "mime_type": "image/png"},
        },
        {
            "event_type": "step.delta",
            "index": 2,
            "delta": {"type": "text", "text": "B"},
        },
        {"event_type": "step.stop", "index": 2},
        {
            "event_type": "interaction.completed",
            "interaction": {"id": "i_mix", "model": MODEL_NAME, "status": "completed"},
        },
    ]

    llm, _, _, _ = _wired_llm(events, output_version="v1")
    chunks = list(llm.stream("prompt"))
    full = chunks[0]
    for c in chunks[1:]:
        full = full + c
    assert "SECRET" not in full.text
    assert [
        (b["type"], b.get("text")) for b in full.content if isinstance(b, Mapping)
    ] == [("text", "A"), ("image", None), ("text", "B")]

    llm_v3, _, _, _ = _wired_llm(events, output_version="v1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangChainBetaWarning)
        assert_valid_event_stream(list(llm_v3.stream_events("prompt", version="v3")))

    # Non-SSE polled stream passes bounded `timeout` to `.get()` in sync & async
    queued = Interaction.model_validate({"id": "bg_s", "status": "queued"})
    done = Interaction.model_validate(
        {
            "id": "bg_s",
            "agent": "deep-research-preview-04-2026",
            "status": "completed",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "ok"}]}
            ],
        }
    )
    bg_llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_KEY,
        agent="deep-research-preview-04-2026",
        use_interactions_api=True,
        polling_interval=0.01,
        timeout=30.0,
    )
    sync_m = Mock(create=Mock(return_value=queued), get=Mock(return_value=done))
    async_m = Mock(
        create=AsyncMock(return_value=queued), get=AsyncMock(return_value=done)
    )
    object.__setattr__(bg_llm.client, "_interactions", sync_m)
    object.__setattr__(bg_llm.async_client, "_interactions", async_m)
    with patch("langchain_google_genai._interactions_stream.time.sleep"):
        assert next(iter(bg_llm.stream("q"))).text == "ok"
    assert sync_m.get.call_args.kwargs["timeout"] >= 1.0
    with patch(
        "langchain_google_genai._interactions_stream.asyncio.sleep", new=AsyncMock()
    ):
        polled_async = [c async for c in bg_llm.astream("q")]
        assert len(polled_async) == 1 and polled_async[0].text == "ok"
    assert async_m.get.call_args.kwargs["timeout"] >= 1.0

    # Async `astream` external URL fetching uses `httpx2.AsyncClient`, never `requests`
    async def _fake_aget(self_c: Any, url: str, **_: Any) -> httpx2.Response:
        return httpx2.Response(
            200,
            content=b"\x89PNG\r\n\x1a\n\x00",
            headers={"Content-Type": "image/png"},
            request=httpx2.Request("GET", url),
        )

    llm_au, fake_au, _, _ = _wired_llm(events[7:])
    with (
        patch("requests.get", side_effect=AssertionError("Sync requests.get used!")),
        patch.object(httpx2.AsyncClient, "get", _fake_aget),
    ):
        au_out = [
            c
            async for c in llm_au.astream(
                [HumanMessage([{"type": "image", "url": "https://cdn.example/pic"}])]
            )
        ]
    assert au_out
    assert fake_au.create_calls[0]["input"][0]["content"][0]["mime_type"] == "image/png"

    # Multi-step v0 `.content_blocks` attaches citation to the matching 2nd block
    multi_cit = (
        _interaction_to_result(
            {
                "id": "i_mcit",
                "model": MODEL_NAME,
                "status": "completed",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "Let me check.\n"}],
                    },
                    {
                        "type": "google_search_call",
                        "id": "gs_1",
                        "arguments": {"queries": ["python creator"]},
                    },
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": "Guido created Python.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "start_index": 0,
                                        "end_index": 5,
                                        "url": "https://python.org",
                                        "title": "Python",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            }
        )
        .generations[0]
        .message
    )
    v1_blks = multi_cit.content_blocks
    assert "annotations" not in v1_blks[0]
    assert v1_blks[1]["annotations"][0]["cited_text"] == "Guido"  # type: ignore[typeddict-item]


async def test_stream_ephemeral_http_cleanup_on_close_error_and_httpx2_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ephemeral_http` closes even on stream close error; `httpx2` errors classify."""
    import asyncio

    import langchain_google_genai._interactions_stream as st_mod

    # 1) Sync & Async ephemeral httpx2 clients close even when raw_stream.close() raises
    class _BrokenSyncStream:
        def __iter__(self) -> Any:
            return iter(
                [
                    {
                        "event_type": "interaction.completed",
                        "interaction": {
                            "id": "i_ok",
                            "model": MODEL_NAME,
                            "status": "completed",
                        },
                    }
                ]
            )

        def close(self) -> None:
            msg = "sync close failed"
            raise RuntimeError(msg)

    class _BrokenAsyncStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self) -> _BrokenAsyncStream:
            return self

        async def __anext__(self) -> Any:
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "i_ok_a",
                    "model": MODEL_NAME,
                    "status": "completed",
                },
            }

        async def close(self) -> None:
            msg = "async close failed"
            raise RuntimeError(msg)

    m = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=FAKE_KEY, use_interactions_api=True
    )
    eph_s = httpx2.Client()
    fake_s = Mock(create=Mock(return_value=_BrokenSyncStream()))
    with (
        patch.object(
            st_mod, "_get_scoped_interactions_client", return_value=(fake_s, eph_s)
        ),
        pytest.raises(RuntimeError, match="sync close failed"),
    ):
        list(m.stream("hi"))
    assert eph_s.is_closed

    eph_a = httpx2.AsyncClient()
    fake_a = Mock(create=AsyncMock(return_value=_BrokenAsyncStream()))
    with (
        patch.object(
            st_mod, "_get_scoped_interactions_client", return_value=(fake_a, eph_a)
        ),
        pytest.raises(RuntimeError, match="async close failed"),
    ):
        _ = [c async for c in m.astream("hi")]
    assert eph_a.is_closed

    # 2) Per-chunk `asyncio.wait_for` `TimeoutError` + mid-stream `httpx2` errors
    async def _fake_wait_for(coro: Any, timeout: float | None = None) -> Any:
        coro.close()
        msg_exp = "wait_for expired"
        raise TimeoutError(msg_exp)

    monkeypatch.setattr(asyncio, "wait_for", _fake_wait_for)
    monkeypatch.setenv("LANGCHAIN_GOOGLE_STREAM_CHUNK_TIMEOUT_S", "0.05")

    llm_t, _, _, _ = _wired_llm(
        async_stream=_FakeAsyncStream([], hang_after_events=True)
    )
    with pytest.raises(GoogleTimeoutError, match=r"0\.05s"):
        _ = [c async for c in llm_t.astream("hi")]

    monkeypatch.undo()
    llm_v2_to, _, _, _ = _wired_llm(
        async_stream=_FakeAsyncStream([], error_after=httpx2.ReadTimeout("v2 read"))
    )
    with pytest.raises(GoogleTimeoutError):
        _ = [c async for c in llm_v2_to.astream("hi")]

    llm_v2_conn, _, _, _ = _wired_llm(
        async_stream=_FakeAsyncStream([], error_after=httpx2.ReadError("v2 reset"))
    )
    with pytest.raises(GoogleConnectionError):
        _ = [c async for c in llm_v2_conn.astream("hi")]

    await m.aclose()


@pytest.mark.parametrize("output_version", [None, "v1", "interactions/v1"])
def test_stream_and_v1_server_tool_thought_signatures_round_trip(
    output_version: str | None,
) -> None:
    """Streaming & `v1` keep summary-less server-tool `thought(s1)` & `thought(s2)`."""
    from langchain_google_genai._interactions_response import _interaction_to_result

    s1 = base64.b64encode(b"sig-srv-1").decode("ascii")
    s2 = base64.b64encode(b"sig-fc-2").decode("ascii")
    ce_sig = base64.b64encode(b"ce-own-sig").decode("ascii")
    kw: dict[str, Any] = {"output_version": output_version} if output_version else {}

    # 1) Unary `v1` + Stream (`None`, `v1`, `interactions/v1`) code_execution + fc
    raw_steps = [
        {"type": "thought", "signature": s1},
        {
            "type": "code_execution_call",
            "id": "ce1",
            "arguments": {"code": "1+1", "language": "python"},
            "signature": ce_sig,
        },
        {"type": "code_execution_result", "call_id": "ce1", "result": "2"},
        {"type": "thought", "signature": s2},
        {"type": "function_call", "id": "fc1", "name": "fn", "arguments": {}},
    ]
    ai_u = (
        _interaction_to_result(
            {
                "id": "i_u",
                "status": "requires_action",
                "model": MODEL_NAME,
                "steps": raw_steps,
            },
            output_version=output_version,
        )
        .generations[0]
        .message
    )

    # Stream using realistic `thought_signature` deltas (summary-less)
    sse_ev: list[Any] = [
        {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "thought_signature", "signature": s1},
        },
        {"event_type": "step.stop", "index": 0},
        {"event_type": "step.start", "index": 1, "step": raw_steps[1]},
        {"event_type": "step.stop", "index": 1},
        {"event_type": "step.start", "index": 2, "step": raw_steps[2]},
        {"event_type": "step.stop", "index": 2},
        {"event_type": "step.start", "index": 3, "step": {"type": "thought"}},
        {
            "event_type": "step.delta",
            "index": 3,
            "delta": {"type": "thought_signature", "signature": s2},
        },
        {"event_type": "step.stop", "index": 3},
        {"event_type": "step.start", "index": 4, "step": raw_steps[4]},
        {"event_type": "step.stop", "index": 4},
        {
            "event_type": "interaction.completed",
            "interaction": {
                "id": "i_s",
                "model": MODEL_NAME,
                "status": "requires_action",
            },
        },
    ]
    llm_s, _, _, _ = _wired_llm(sse_ev, **kw)
    chunks = list(llm_s.stream("q"))
    ai_s = chunks[0]
    for ch in chunks[1:]:
        ai_s = ai_s + ch

    for cand in (ai_u, ai_s):
        _, rep = _messages_to_interaction_steps(
            [HumanMessage("q"), cand, ToolMessage("r", tool_call_id="fc1")],
            effective_model=MODEL_NAME,
        )
        assert [(st["type"], st.get("signature")) for st in rep[1:-1]] == [
            ("thought", s1),
            ("code_execution_call", ce_sig),
            ("code_execution_result", None),
            ("thought", s2),
            ("function_call", None),
        ]

    # 2) Streamed MCP call + result preceded by summary-less `thought(s1)` + `fc1(s2)`
    mcp_ev: list[Any] = [
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "thought", "signature": s1},
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "step.start",
            "index": 1,
            "step": {
                "type": "mcp_server_tool_call",
                "id": "m1",
                "name": "get_doc",
                "server_name": "docs",
                "arguments": {},
            },
        },
        {"event_type": "step.stop", "index": 1},
        {
            "event_type": "step.start",
            "index": 2,
            "step": {
                "type": "mcp_server_tool_result",
                "call_id": "m1",
                "name": "get_doc",
                "server_name": "docs",
                "result": {"k": "v"},
            },
        },
        {"event_type": "step.stop", "index": 2},
        {
            "event_type": "step.start",
            "index": 3,
            "step": {"type": "thought", "signature": s2},
        },
        {"event_type": "step.stop", "index": 3},
        {
            "event_type": "step.start",
            "index": 4,
            "step": {
                "type": "function_call",
                "id": "fc1",
                "name": "fn",
                "arguments": {},
            },
        },
        {"event_type": "step.stop", "index": 4},
        {
            "event_type": "interaction.completed",
            "interaction": {
                "id": "i_m",
                "model": MODEL_NAME,
                "status": "requires_action",
            },
        },
    ]
    llm_m, _, _, _ = _wired_llm(mcp_ev, **kw)
    m_chunks = list(llm_m.stream("q"))
    ai_m = m_chunks[0]
    for ch in m_chunks[1:]:
        ai_m = ai_m + ch
    _, rep_m = _messages_to_interaction_steps(
        [HumanMessage("q"), ai_m, ToolMessage("r", tool_call_id="fc1")],
        effective_model=MODEL_NAME,
    )
    assert [(st["type"], st.get("signature")) for st in rep_m[1:-1]] == [
        ("thought", s1),
        ("mcp_server_tool_call", None),
        ("mcp_server_tool_result", None),
        ("thought", s2),
        ("function_call", None),
    ]


@pytest.mark.parametrize("output_version", [None, "v0", "v1", "interactions/v1"])
def test_mid_conversation_toggle_interactions_to_generate_content(
    output_version: str | None,
) -> None:
    """Replay of Interactions `AIMessage`s to `generateContent` never fails/corrupts."""
    from langchain_google_genai._interactions_response import _interaction_to_result

    m = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=FAKE_KEY)
    ov_arg = None if output_version == "v0" else output_version

    # 1) URI-backed image, video, audio, and document (`files/...`) replay as FileData
    ai_media = (
        _interaction_to_result(
            {
                "id": "i_med",
                "status": "completed",
                "model": "gemini-3-flash-preview",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "image",
                                "uri": "files/img123",
                                "mime_type": "image/png",
                            },
                            {
                                "type": "video",
                                "uri": "files/vid123",
                                "mime_type": "video/mp4",
                            },
                            {
                                "type": "document",
                                "uri": "files/doc123",
                                "mime_type": "application/pdf",
                            },
                        ],
                    }
                ],
            },
            output_version=ov_arg,
        )
        .generations[0]
        .message
    )
    req_med = m._prepare_request(
        [HumanMessage("q"), ai_media, HumanMessage("follow")],
        use_interactions_api=False,
    )
    parts_med = req_med["contents"][1].parts
    assert [p.file_data.file_uri for p in parts_med] == [
        "files/img123",
        "files/vid123",
        "files/doc123",
    ]

    # 2) Non-code server tools (MCP, Search, Maps, URL context, File Search, Retrieval)
    # NEVER fabricate invalid orphaned `CodeExecutionResult` parts!
    ai_srv = (
        _interaction_to_result(
            {
                "id": "i_srv",
                "status": "completed",
                "model": "gemini-3-flash-preview",
                "steps": [
                    {
                        "type": "mcp_server_tool_call",
                        "id": "m1",
                        "name": "get_doc",
                        "server_name": "s",
                        "arguments": {},
                    },
                    {
                        "type": "mcp_server_tool_result",
                        "call_id": "m1",
                        "result": {"k": "v"},
                    },
                    {
                        "type": "google_search_call",
                        "id": "g1",
                        "arguments": {"queries": ["x"]},
                    },
                    {"type": "google_search_result", "call_id": "g1", "result": []},
                    {
                        "type": "url_context_call",
                        "id": "u1",
                        "arguments": {"urls": ["https://a.example"]},
                    },
                    {"type": "url_context_result", "call_id": "u1", "result": []},
                    {"type": "google_maps_call", "id": "gm1"},
                    {"type": "google_maps_result", "call_id": "gm1", "result": []},
                    {"type": "file_search_call", "id": "fs1"},
                    {"type": "file_search_result", "call_id": "fs1"},
                    {
                        "type": "retrieval_call",
                        "id": "r1",
                        "arguments": {"queries": ["y"]},
                    },
                    {"type": "retrieval_result", "call_id": "r1"},
                    # Genuine code execution IS still preserved cleanly!
                    {
                        "type": "code_execution_call",
                        "id": "ce1",
                        "arguments": {"code": "print(7)", "language": "python"},
                    },
                    {
                        "type": "code_execution_result",
                        "call_id": "ce1",
                        "result": "7\n",
                    },
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "done"}],
                    },
                ],
            },
            output_version=ov_arg,
        )
        .generations[0]
        .message
    )
    req_srv = m._prepare_request(
        [HumanMessage("q"), ai_srv, HumanMessage("follow")],
        use_interactions_api=False,
    )
    parts_srv = req_srv["contents"][1].parts
    ce_results = [p for p in parts_srv if p.code_execution_result is not None]
    assert len(ce_results) == 1
    assert ce_results[0].code_execution_result.output == "7\n"
    assert any(p.executable_code is not None for p in parts_srv)
    assert parts_srv[-1].text == "done"


@pytest.mark.parametrize("output_version", [None, "v1", "interactions/v1"])
def test_mixed_file_place_and_url_citations_unary_and_stream_parity(
    output_version: str | None,
) -> None:
    """Unary & stream keep `file_citation`, `place_citation`, & `url_citation`."""
    from langchain_google_genai._interactions_response import _interaction_to_result

    anns_raw = [
        {
            "type": "url_citation",
            "url": "https://web.example/1",
            "title": "Web Doc",
            "start_index": 0,
            "end_index": 3,
        },
        {
            "type": "place_citation",
            "url": "https://maps.google.com/?cid=9",
            "name": "Cafe Nine",
            "place_id": "ChIJ_9",
            "start_index": 4,
            "end_index": 8,
        },
        {
            "type": "file_citation",
            "document_uri": "files/spec_1",
            "file_name": "spec.pdf",
            "page_number": 7,
            "start_index": 9,
            "end_index": 13,
        },
    ]
    text_val = "Web Cafe Spec!"
    unary_msg = (
        _interaction_to_result(
            {
                "id": "i_cit",
                "status": "completed",
                "model": MODEL_NAME,
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": text_val,
                                "annotations": anns_raw,
                            }
                        ],
                    }
                ],
            },
            output_version=output_version,
        )
        .generations[0]
        .message
    )

    sse_events: list[Any] = [
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "text", "text": text_val},
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "text_annotation_delta", "annotations": anns_raw},
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "interaction.completed",
            "interaction": {
                "id": "i_cit",
                "model": MODEL_NAME,
                "status": "completed",
            },
        },
    ]
    kw: dict[str, Any] = {"output_version": output_version} if output_version else {}
    llm_s, _, _, _ = _wired_llm(sse_events, **kw)
    s_chunks = list(llm_s.stream("cite"))
    stream_msg = s_chunks[0]
    for sc in s_chunks[1:]:
        stream_msg = stream_msg + sc

    for candidate in (unary_msg, stream_msg):
        v1_b = candidate.content_blocks
        txt_b = next(b for b in v1_b if b.get("type") == "text")
        cits = txt_b.get("annotations")
        assert isinstance(cits, list) and len(cits) == 3
        urls = {c["url"]: c for c in cits}
        assert urls["https://web.example/1"]["cited_text"] == "Web"
        assert urls["https://maps.google.com/?cid=9"]["cited_text"] == "Cafe"
        assert (
            urls["https://maps.google.com/?cid=9"]["extras"]["google_ai_metadata"][
                "place_id"
            ]
            == "ChIJ_9"
        )
        assert urls["files/spec_1"]["cited_text"] == "Spec"
        assert (
            urls["files/spec_1"]["extras"]["file_citation_metadata"]["page_number"] == 7
        )


async def test_premature_sse_eof_polling_status_update_metadata_and_audio_rate() -> (
    None
):
    """Premature `in_progress` SSE EOF polls/raises; `status_update` keeps metadata."""
    from langchain_google_genai._interactions_stream import _InteractionsStreamProcessor

    # 1) Direct processor `finalize_eof()` on `in_progress` raises classified API error
    p1 = _InteractionsStreamProcessor(
        target_name="deep-research-preview-04-2026",
        requested_store=True,
        output_version=None,
        pydantic_schema=None,
    )
    p1.process_event(
        {
            "event_type": "interaction.created",
            "interaction": {
                "id": "dr_1",
                "agent": "deep-research-preview-04-2026",
                "status": "in_progress",
            },
        }
    )
    with pytest.raises(GoogleInteractionsAPIError, match="ended prematurely"):
        p1.finalize_eof()

    # 2) `status_update` preserves `continuation_token` + `usage` on uncontinued hop
    p2 = _InteractionsStreamProcessor(
        target_name=MODEL_NAME,
        requested_store=True,
        output_version=None,
        pydantic_schema=None,
    )
    p2.process_event(
        {
            "event_type": "interaction.created",
            "interaction": {"id": "h1", "model": MODEL_NAME, "status": "in_progress"},
        }
    )
    p2.process_event(
        {
            "event_type": "interaction.status_update",
            "interaction_id": "h1",
            "status": "requires_continuation",
            "continuation_token": "tok_1",
            "usage": {
                "total_input_tokens": 5,
                "total_output_tokens": 10,
                "total_tokens": 15,
            },
        }
    )
    end2 = cast("AIMessageChunk", p2.finalize_eof()[-1].message)
    assert end2.response_metadata["status"] == "requires_continuation"
    assert end2.response_metadata["finish_reason"] == "MAX_TOKENS"
    assert end2.response_metadata["continuation_token"] == "tok_1"
    assert end2.usage_metadata is not None
    assert end2.usage_metadata["total_tokens"] == 15

    # 3) Deep Research SSE stream dropping mid-run seamlessly polls `.get(id=...)`
    drop_events: list[Any] = [
        {
            "event_type": "interaction.created",
            "interaction": {
                "id": "dr_live",
                "agent": "deep-research-preview-04-2026",
                "status": "in_progress",
            },
        },
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "text", "text": "Part 1. "},
        },
    ]
    polled_done = Interaction.model_validate(
        {
            "id": "dr_live",
            "agent": "deep-research-preview-04-2026",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Part 1. Part 2."}],
                }
            ],
            "usage": {
                "total_input_tokens": 20,
                "total_output_tokens": 40,
                "total_tokens": 60,
            },
        }
    )
    dr_llm, fake_dr, _, _ = _wired_llm(
        drop_events,
        get_items=[
            Interaction.model_validate({"id": "dr_live", "status": "in_progress"}),
            polled_done,
        ],
        agent="deep-research-preview-04-2026",
        polling_interval=0.01,
    )
    with patch("langchain_google_genai._interactions_stream.time.sleep"):
        dr_chunks = list(dr_llm.stream("research quantum"))
    dr_full = dr_chunks[0]
    for dc in dr_chunks[1:]:
        dr_full = dr_full + dc
    assert fake_dr.create_calls[0]["background"] is True
    assert len(fake_dr.get_calls) == 2
    assert dr_full.text == "Part 1. Part 2."
    assert dr_full.response_metadata["status"] == "completed"
    assert dr_full.response_metadata["finish_reason"] == "STOP"
    assert dr_full.usage_metadata is not None
    assert dr_full.usage_metadata["total_tokens"] == 60

    dr_llm_a, fake_dr_a, _, _ = _wired_llm(
        drop_events,
        get_items=[polled_done],
        agent="deep-research-preview-04-2026",
        polling_interval=0.01,
    )
    with patch(
        "langchain_google_genai._interactions_stream.asyncio.sleep", new=AsyncMock()
    ):
        a_dr_chunks = [c async for c in dr_llm_a.astream("research quantum")]
    a_dr_full = a_dr_chunks[0]
    for dc in a_dr_chunks[1:]:
        a_dr_full = a_dr_full + dc
    assert a_dr_full.text == "Part 1. Part 2."
    assert len(fake_dr_a.get_calls) == 1

    # 4) Stateless `store=False` premature EOF raises `GoogleInteractionsAPIError`
    drop_stateless, _, _, _ = _wired_llm(
        [
            {
                "event_type": "interaction.created",
                "interaction": {
                    "id": "",
                    "model": MODEL_NAME,
                    "status": "in_progress",
                },
            }
        ],
        store=False,
    )
    with pytest.raises(GoogleInteractionsAPIError, match="ended prematurely"):
        list(drop_stateless.stream("hi"))

    # 5) Non-SSE background stream with missing `id` raises GoogleInteractionsAPIError
    noid_inter = Interaction.model_validate({"id": "", "status": "in_progress"})
    llm_noid, _, _, _ = _wired_llm(
        sync_stream=noid_inter,
        async_stream=noid_inter,
        agent="deep-research-preview-04-2026",
    )
    with pytest.raises(GoogleInteractionsAPIError, match="returned no id"):
        list(llm_noid.stream("q"))
    with pytest.raises(GoogleInteractionsAPIError, match="returned no id"):
        _ = [c async for c in llm_noid.astream("q")]

    # 6) `AudioDelta.rate` fallback sets WAV framerate when `sample_rate` is absent,
    # and async `automatic_continuation` inspects `async_client` signature!
    import io
    import wave

    pcm_raw = b"\x03\x04" * 16
    aud_ev: list[Any] = [
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {
                "type": "audio",
                "data": base64.b64encode(pcm_raw).decode("ascii"),
                "mime_type": "audio/pcm",
                "rate": 16000,
                "channels": 1,
            },
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "interaction.completed",
            "interaction": {"id": "i_ar", "model": MODEL_NAME, "status": "completed"},
        },
    ]
    llm_ar, _, _, _ = _wired_llm(aud_ev)
    ar_chunks = list(llm_ar.stream("say hi"))
    ar_full = ar_chunks[0]
    for ch in ar_chunks[1:]:
        ar_full = ar_full + ch
    with wave.open(io.BytesIO(ar_full.additional_kwargs["audio"]), "rb") as wf:
        assert wf.getframerate() == 16000

    # Custom async mock accepting `automatic_continuation` while sync mock lacks `**kw`
    from langchain_google_genai._interactions_request import (
        _aprepare_interactions_request,
    )

    llm_ac = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=FAKE_KEY, use_interactions_api=True
    )

    def _sync_no_ac(*, input: Any, model: str, store: bool, timeout: Any) -> Any:
        return (input, model, store, timeout)

    async def _async_with_ac(
        *,
        input: Any,
        model: str,
        store: bool,
        timeout: Any,
        automatic_continuation: bool = True,
    ) -> Any:
        return (input, model, store, timeout, automatic_continuation)

    object.__setattr__(llm_ac.client, "_interactions", Mock(create=_sync_no_ac))
    object.__setattr__(
        llm_ac.async_client, "_interactions", Mock(create=_async_with_ac)
    )
    aprep = await _aprepare_interactions_request(
        llm_ac, [HumanMessage("hi")], automatic_continuation=False
    )
    assert aprep.create_kwargs["automatic_continuation"] is False
    await llm_ac.aclose()


async def test_stream_ai_url_zero_fetch_manual_continuation_and_httpx_parity() -> None:
    """Stream honors AIMessage zero-fetch, manual resume, `incomplete`, & httpx."""
    from langchain_google_genai._interactions_stream import (
        _InteractionsStreamProcessor,
    )
    from langchain_google_genai.chat_models import (
        GoogleConnectionError,
        GoogleTimeoutError,
    )

    events: list[Any] = [
        {
            "event_type": "interaction.created",
            "interaction": {"id": "s_h2", "model": MODEL_NAME, "status": "in_progress"},
        },
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "text", "text": "done"},
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "interaction.completed",
            "interaction": {"id": "s_h2", "model": MODEL_NAME, "status": "completed"},
        },
    ]

    # 1) `stream` & `astream` never fetch AIMessage HTTP(S) URLs and prefetch `HTTPS://`
    llm_s, fake_s, _, _ = _wired_llm(events)
    png_bytes = b"\x89PNG\r\n\x1a\n\x00"
    msgs_ssrf: list[Any] = [
        HumanMessage("q1"),
        AIMessage(
            content=[
                {
                    "type": "text-plain",
                    "url": "http://169.254.169.254/latest/meta-data/",
                },
                {"type": "image", "url": "http://internal.corp/chart.png"},
            ],
            response_metadata={"model_provider": "openai"},
        ),
        HumanMessage([{"type": "image", "url": "HTTPS://cdn.example.com/upper.png"}]),
    ]

    def _img_ok(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == "https://cdn.example.com/upper.png"
        return httpx.Response(
            200, headers={"content-type": "image/png"}, content=png_bytes, request=req
        )

    with (
        httpx.Client(transport=httpx.MockTransport(_img_ok)) as sc,
        patch("requests.get", side_effect=AssertionError("requests.get used!")),
    ):
        s_chunks = list(llm_s.stream(msgs_ssrf, http_options={"httpx_client": sc}))
        assert not sc.is_closed
    assert "".join(c.text for c in s_chunks) == "done"

    async with httpx.AsyncClient(transport=httpx.MockTransport(_img_ok)) as ac:
        with (
            patch("requests.get", side_effect=AssertionError("requests.get used!")),
            patch.object(
                httpx.Client,
                "get",
                side_effect=AssertionError("Sync httpx.Client.get used in astream!"),
            ),
        ):
            a_chunks = [
                c
                async for c in llm_s.astream(
                    msgs_ssrf, http_options={"httpx_async_client": ac}
                )
            ]
        assert not ac.is_closed
    assert "".join(c.text for c in a_chunks) == "done"
    for call_kw in fake_s.create_calls:
        assert "169.254.169.254" not in json.dumps(call_kw["input"])
        assert "internal.corp" not in json.dumps(call_kw["input"])

    # 2) Sync `stream` & async `astream` classify httpx timeout/connection errors
    def _to_err(req: httpx.Request) -> httpx.Response:
        msg = "connect timeout"
        raise httpx.ConnectTimeout(msg, request=req)

    def _conn_err(req: httpx.Request) -> httpx.Response:
        msg = "network unreachable"
        raise httpx.ConnectError(msg, request=req)

    user_img = [HumanMessage([{"type": "image", "url": "https://cdn.example/x.png"}])]
    for fn, expected in (
        (_to_err, GoogleTimeoutError),
        (_conn_err, GoogleConnectionError),
    ):
        with (
            httpx.Client(transport=httpx.MockTransport(fn)) as sc_e,
            pytest.raises(expected),
        ):
            list(llm_s.stream(user_img, http_options={"httpx_client": sc_e}))
        async with httpx.AsyncClient(transport=httpx.MockTransport(fn)) as ac_e:
            with pytest.raises(expected):
                _ = [
                    c
                    async for c in llm_s.astream(
                        user_img, http_options={"httpx_async_client": ac_e}
                    )
                ]

    # 3) Stream preserves `continuation_token` on `status="incomplete"` (without leaking
    # stale mid-hop `in_progress` checkpoint tokens when completed clears token), and
    # supports resuming via `stream([], ...)` / `stream("", ...)` / `[prompt, hop1]`!
    inc_events: list[Any] = [
        {
            "event_type": "interaction.created",
            "interaction": {
                "id": "inc_1",
                "model": MODEL_NAME,
                "status": "in_progress",
            },
        },
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "text", "text": "cut short"},
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "interaction.completed",
            "interaction": {
                "id": "inc_1",
                "model": MODEL_NAME,
                "status": "incomplete",
                "continuation_token": "tok_inc_stream",
            },
        },
    ]
    llm_inc, _, _, _ = _wired_llm(
        inc_events, use_previous_interaction_id=True, store=True
    )
    hop1_ch = list(llm_inc.stream("long prompt"))
    hop1_agg = hop1_ch[0]
    for c in hop1_ch[1:]:
        hop1_agg = hop1_agg + c
    assert hop1_agg.response_metadata["status"] == "incomplete"
    assert hop1_agg.response_metadata["continuation_token"] == "tok_inc_stream"
    assert hop1_agg.response_metadata["stored"] is True

    # Mid-hop `in_progress` checkpoint token is NOT leaked if `incomplete` clears token
    p_stale = _InteractionsStreamProcessor(
        target_name=MODEL_NAME,
        requested_store=True,
        output_version=None,
        pydantic_schema=None,
    )
    p_stale.process_event(
        {
            "event_type": "interaction.status_update",
            "interaction_id": "inc_2",
            "status": "in_progress",
            "continuation_token": "mid_checkpoint",
        }
    )
    end_stale = p_stale.process_event(
        {
            "event_type": "interaction.completed",
            "interaction": {"id": "inc_2", "model": MODEL_NAME, "status": "incomplete"},
        }
    )[-1].message
    assert "continuation_token" not in end_stale.response_metadata

    # Resume `hop1_agg` across `[]`, `""`, `[SystemMessage]`, and `[prompt, hop1_ai]`!
    llm_res, fake_res, _, _ = _wired_llm(
        events, use_previous_interaction_id=True, store=True
    )
    hop1_ai = AIMessage(
        content=hop1_agg.content,
        response_metadata=hop1_agg.response_metadata,
    )
    list(
        llm_res.stream(
            [], previous_interaction_id="inc_1", continuation_token="tok_inc_stream"
        )
    )
    assert "input" not in fake_res.create_calls[-1]
    assert fake_res.create_calls[-1]["previous_interaction_id"] == "inc_1"

    _ = [
        c
        async for c in llm_res.astream(
            [HumanMessage("long prompt"), hop1_ai],
            continuation_token="tok_inc_stream",
            max_output_tokens=2048,
        )
    ]
    assert "input" not in fake_res.create_calls[-1]
    assert fake_res.create_calls[-1]["previous_interaction_id"] == "inc_1"
    await llm_s.aclose()
    await llm_inc.aclose()
    await llm_res.aclose()


async def test_astream_aclosing_early_break_closes_raw_sse_and_ephemeral_http() -> None:
    """Breaking out of `aclosing(...)` closes `raw_stream` & `AsyncClient` at once."""
    closed_sse = 0
    created_clients: list[httpx.AsyncClient] = []

    class TrackedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created_clients.append(self)

    def _as_agen(it: Any) -> AsyncGenerator[Any, None]:
        return cast("AsyncGenerator[Any, None]", it)

    def make_slow_transport(
        *, hang_after_first: bool = True
    ) -> httpx.AsyncBaseTransport:
        class SlowSSETransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, request=request)

            async def handle_async_request(
                self, request: httpx.Request
            ) -> httpx.Response:
                class SlowStream(httpx.AsyncByteStream):
                    async def __aiter__(self) -> AsyncIterator[bytes]:
                        ev = {
                            "event_type": "step.delta",
                            "index": 0,
                            "delta": {"type": "text", "text": "hello"},
                        }
                        yield f"data: {json.dumps(ev)}\n\n".encode()
                        if hang_after_first:
                            await asyncio.sleep(30)
                        else:
                            done = {
                                "event_type": "interaction.completed",
                                "interaction": {
                                    "id": "",
                                    "model": MODEL_NAME,
                                    "status": "completed",
                                },
                            }
                            yield f"data: {json.dumps(done)}\n\n".encode()

                    async def aclose(self) -> None:
                        nonlocal closed_sse
                        closed_sse += 1

                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=SlowStream(),
                )

        return SlowSSETransport()

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, api_key=FAKE_KEY, use_interactions_api=True
    )
    try:
        # 1) Public `llm.astream(...)` with per-request `ephemeral_http`
        with patch("httpx.AsyncClient", TrackedAsyncClient):
            async with contextlib.aclosing(
                _as_agen(
                    llm.astream(
                        "hi",
                        http_options={
                            "async_client_args": {"transport": make_slow_transport()}
                        },
                    )
                )
            ) as agen:
                async for chunk in agen:
                    if chunk.text == "hello":
                        break
            assert closed_sse == 1
            assert len(created_clients) == 1
            assert created_clients[0].is_closed is True

        # 2) Direct `_astream` and `_astream_interactions` via `contextlib.aclosing`
        closed_sse = 0
        created_clients.clear()
        with patch("httpx.AsyncClient", TrackedAsyncClient):
            async with contextlib.aclosing(
                _as_agen(
                    llm._astream(
                        [HumanMessage("hi")],
                        http_options={
                            "async_client_args": {"transport": make_slow_transport()}
                        },
                    )
                )
            ) as raw_agen:
                async for gchunk in raw_agen:
                    if gchunk.text == "hello":
                        break
            assert closed_sse == 1
            assert len(created_clients) == 1
            assert created_clients[0].is_closed is True

        closed_sse = 0
        created_clients.clear()
        with patch("httpx.AsyncClient", TrackedAsyncClient):
            async with contextlib.aclosing(
                _as_agen(
                    llm._astream_interactions(
                        [HumanMessage("hi")],
                        http_options={
                            "async_client_args": {"transport": make_slow_transport()}
                        },
                    )
                )
            ) as ia_agen:
                async for ichunk in ia_agen:
                    if ichunk.text == "hello":
                        break
            assert closed_sse == 1
            assert len(created_clients) == 1
            assert created_clients[0].is_closed is True

        # 3) `astream_events(..., version="v2")` breaking early closes immediately
        closed_sse = 0
        created_clients.clear()
        with patch("httpx.AsyncClient", TrackedAsyncClient):
            async with contextlib.aclosing(
                _as_agen(
                    llm.astream_events(
                        "hi",
                        version="v2",
                        http_options={
                            "async_client_args": {"transport": make_slow_transport()}
                        },
                    )
                )
            ) as ev_gen:
                async for ev in ev_gen:
                    if ev["event"] == "on_chat_model_stream":
                        break
            assert closed_sse == 1
            assert len(created_clients) == 1
            assert created_clients[0].is_closed is True

        # 4) Concurrent streams: one breaks early, sibling completes undisturbed
        closed_sse = 0
        created_clients.clear()
        with patch("httpx.AsyncClient", TrackedAsyncClient):

            async def _break_early() -> str:
                got = ""
                async with contextlib.aclosing(
                    _as_agen(
                        llm.astream(
                            "early",
                            http_options={
                                "async_client_args": {
                                    "transport": make_slow_transport()
                                }
                            },
                        )
                    )
                ) as g1:
                    async for c in g1:
                        if c.text:
                            got = c.text
                            break
                return got

            async def _read_all() -> str:
                async with contextlib.aclosing(
                    _as_agen(
                        llm.astream(
                            "full",
                            http_options={
                                "async_client_args": {
                                    "transport": make_slow_transport(
                                        hang_after_first=False
                                    )
                                }
                            },
                        )
                    )
                ) as g2:
                    parts = [c.text async for c in g2 if c.text]
                return "".join(parts)

            r_early, r_full = await asyncio.gather(_break_early(), _read_all())
            assert r_early == "hello"
            assert r_full == "hello"
            assert closed_sse == 2
            assert len(created_clients) == 2
            assert all(c.is_closed for c in created_clients)
    finally:
        await llm.aclose()

    # 5) Shared (non-ephemeral) async transport SSE stream closes immediately too
    closed_sse = 0
    llm_shared = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_KEY,
        use_interactions_api=True,
        client_args={"transport": make_slow_transport()},
    )
    try:
        async with contextlib.aclosing(_as_agen(llm_shared.astream("hi"))) as shared_it:
            async for chunk in shared_it:
                if chunk.text == "hello":
                    break
        assert closed_sse == 1
    finally:
        await llm_shared.aclose()


def test_streamed_mcp_json_array_result_replays_cleanly_through_real_sdk() -> None:
    """Aggregated SSE `mcp_server_tool_result` JSON arrays replay on Turn 2 cleanly."""
    captured: list[dict[str, Any]] = []
    sse_events = [
        {
            "event_type": "interaction.start",
            "interaction": {"id": "", "model": MODEL_NAME, "status": "in_progress"},
        },
        {
            "event_type": "step.start",
            "index": 0,
            "step": {
                "type": "mcp_server_tool_call",
                "id": "m_stream",
                "name": "get_jobs",
                "server_name": "ci",
                "arguments": {},
            },
        },
        {"event_type": "step.complete", "index": 0},
        {
            "event_type": "step.start",
            "index": 1,
            "step": {
                "type": "mcp_server_tool_result",
                "call_id": "m_stream",
                "name": "get_jobs",
                "server_name": "ci",
                "result": [{"job": "lint", "passed": True}],
            },
        },
        {"event_type": "step.complete", "index": 1},
        {"event_type": "step.start", "index": 2, "step": {"type": "model_output"}},
        {
            "event_type": "step.delta",
            "index": 2,
            "delta": {"type": "text", "text": "Lint passed."},
        },
        {"event_type": "step.complete", "index": 2},
        {
            "event_type": "interaction.completed",
            "interaction": {"id": "", "model": MODEL_NAME, "status": "completed"},
        },
    ]
    sse_bytes = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in sse_events)

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        captured.append(body)
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_bytes,
            )
        return httpx.Response(
            200,
            json={
                "id": "",
                "model": MODEL_NAME,
                "status": "completed",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "All green."}],
                    }
                ],
            },
        )

    for ov in (None, "v1", "interactions/v1"):
        kw: dict[str, Any] = {
            "model": MODEL_NAME,
            "api_key": FAKE_KEY,
            "use_interactions_api": True,
            "client_args": {"transport": httpx.MockTransport(handler)},
        }
        if ov is not None:
            kw["output_version"] = ov
        llm = ChatGoogleGenerativeAI(**kw)
        chunks = list(llm.stream("Check CI"))
        agg = chunks[0]
        for ch in chunks[1:]:
            agg = agg + ch
        turn1_ai = AIMessage(
            content=agg.content,
            additional_kwargs=agg.additional_kwargs,
            response_metadata=agg.response_metadata,
        )
        turn2 = llm.invoke(
            [HumanMessage("Check CI"), turn1_ai, HumanMessage("Summary?")]
        )
        assert turn2.text == "All green."
        sent_mcp = next(
            s for s in captured[-1]["input"] if s["type"] == "mcp_server_tool_result"
        )
        assert sent_mcp["result"] == '[{"job": "lint", "passed": true}]'


async def test_stream_normalized_finish_reason_and_cache_read_guard() -> None:
    """Streaming sets `normalized_finish_reason` and drops bogus `cache_read`."""

    def _make_sse(events: list[dict[str, Any]]) -> bytes:
        return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)

    # 1. Text stream with completed status + bogus cached tokens > prompt tokens
    text_sse = _make_sse(
        [
            {"event_type": "interaction.created", "interaction": {"id": "int_s_txt"}},
            {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "text", "text": "Hello"},
            },
            {"event_type": "step.complete", "index": 0},
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "int_s_txt",
                    "model": MODEL_NAME,
                    "status": "completed",
                    "usage": {
                        "total_input_tokens": 30,
                        "total_output_tokens": 10,
                        "total_cached_tokens": 99,
                        "total_tokens": 40,
                    },
                },
            },
        ]
    )

    # 2. Split tool call stream: function call in earlier chunk, completed later
    tool_sse = _make_sse(
        [
            {"event_type": "interaction.created", "interaction": {"id": "int_s_tc"}},
            {
                "event_type": "step.start",
                "index": 0,
                "step": {
                    "type": "function_call",
                    "id": "call_1",
                    "name": "get_weather",
                },
            },
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "arguments", "arguments": '{"city":"Paris"}'},
            },
            {"event_type": "step.complete", "index": 0},
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "int_s_tc",
                    "model": MODEL_NAME,
                    "status": "completed",
                },
            },
        ]
    )

    # 3. Truncated stream (incomplete -> MAX_TOKENS / length)
    trunc_sse = _make_sse(
        [
            {"event_type": "interaction.created", "interaction": {"id": "int_s_len"}},
            {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "text", "text": "Part"},
            },
            {"event_type": "step.complete", "index": 0},
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "int_s_len",
                    "model": MODEL_NAME,
                    "status": "incomplete",
                },
            },
        ]
    )

    def _run_sync(payload: bytes) -> list[AIMessageChunk]:
        def _handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=payload
            )

        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=FAKE_KEY,
            use_interactions_api=True,
            client_args={"transport": httpx.MockTransport(_handler)},
        )
        return list(llm.stream("hi"))

    # Text stream assertions + cache_read guard
    txt_chunks = _run_sync(text_sse)
    assert len(txt_chunks) >= 2
    assert "normalized_finish_reason" not in txt_chunks[0].response_metadata
    assert txt_chunks[-1].response_metadata["finish_reason"] == "STOP"
    assert txt_chunks[-1].response_metadata["normalized_finish_reason"] == "stop"
    full_txt = txt_chunks[0]
    for ch in txt_chunks[1:]:
        full_txt = full_txt + ch
    assert full_txt.response_metadata["normalized_finish_reason"] == "stop"
    assert full_txt.usage_metadata is not None
    assert full_txt.usage_metadata["input_tokens"] == 30
    assert full_txt.usage_metadata["output_tokens"] == 10
    assert full_txt.usage_metadata["input_token_details"] == {"cache_read": 0}

    # Tool call stream (sync) upgrades STOP -> tool_call
    tc_chunks = _run_sync(tool_sse)
    assert "normalized_finish_reason" not in tc_chunks[0].response_metadata
    full_tc = tc_chunks[0]
    for ch in tc_chunks[1:]:
        full_tc = full_tc + ch
    assert full_tc.tool_calls
    assert full_tc.response_metadata["finish_reason"] == "STOP"
    assert full_tc.response_metadata["normalized_finish_reason"] == "tool_call"

    # Tool call stream (async astream) upgrades STOP -> tool_call
    def _async_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=tool_sse
        )

    async_llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_KEY,
        use_interactions_api=True,
        client_args={"transport": httpx.MockTransport(_async_handler)},
    )
    atc_chunks = [ch async for ch in async_llm.astream("hi")]
    full_atc = atc_chunks[0]
    for ch in atc_chunks[1:]:
        full_atc = full_atc + ch
    assert full_atc.tool_calls
    assert full_atc.response_metadata["finish_reason"] == "STOP"
    assert full_atc.response_metadata["normalized_finish_reason"] == "tool_call"

    # Truncated stream maps incomplete -> MAX_TOKENS / length
    len_chunks = _run_sync(trunc_sse)
    full_len = len_chunks[0]
    for ch in len_chunks[1:]:
        full_len = full_len + ch
    assert full_len.response_metadata["finish_reason"] == "MAX_TOKENS"
    assert full_len.response_metadata["normalized_finish_reason"] == "length"
