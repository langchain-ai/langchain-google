"""Unit tests for ``stream_function_call_arguments`` (Gemini 3+ partial args).

Covers:

1. Constructor flag threads ``stream_function_call_arguments=True`` onto the
   request's ``ToolConfig.function_calling_config`` on the streaming path.
2. ``_parse_response_candidate`` surfaces ``PartialArg`` typed leaf updates
   via ``additional_kwargs["gemini_partial_args"]`` for advanced consumers,
   and continues emitting standard atomic ``tool_call_chunks`` for
   SDK-assembled ``fc.args``.
3. The unary endpoint (``_generate``/``_agenerate``) does NOT send the flag —
   Vertex ``:generateContent`` rejects it (vercel/ai#14314 → vercel/ai#14352).
4. With the flag off, behavior is byte-identical to the legacy atomic chunk —
   regression guard.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from google.genai.types import (
    Candidate,
    Content,
    FunctionCall,
    GenerateContentResponse,
    Part,
    PartialArg,
)
from langchain_core.messages import AIMessageChunk
from pydantic import SecretStr

from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
    _parse_response_candidate,
)

MODEL_NAME = "gemini-3-pro-preview"
FAKE_API_KEY = "fake-api-key"


def _streaming_candidate(part: Part) -> GenerateContentResponse:
    return GenerateContentResponse(
        candidates=[Candidate(content=Content(role="model", parts=[part]))]
    )


def test_streaming_partial_args_threads_flag_sync() -> None:
    """``_stream`` sends ``stream_function_call_arguments=True`` when configured."""
    captured: dict[str, object] = {}

    def fake_stream(**request: object) -> Iterator[GenerateContentResponse]:
        captured.update(request)
        yield _streaming_candidate(Part(text="ok"))

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        stream_function_call_arguments=True,
    )
    assert llm.client is not None

    def fake_tool(query: str) -> str:
        """Search."""
        return query

    bound = llm.bind_tools([fake_tool])

    with patch.object(
        llm.client.models, "generate_content_stream", side_effect=fake_stream
    ):
        list(bound.stream("hi"))

    config = captured["config"]
    fcc = config.tool_config.function_calling_config  # type: ignore[union-attr]
    assert fcc is not None
    assert fcc.stream_function_call_arguments is True


@pytest.mark.asyncio
async def test_unary_path_does_not_send_flag() -> None:
    """Async ``_agenerate`` must NOT carry the streaming-only flag."""
    captured: dict[str, object] = {}

    async def fake_generate(**request: object) -> GenerateContentResponse:
        captured.update(request)
        return GenerateContentResponse(
            candidates=[
                Candidate(content=Content(role="model", parts=[Part(text="done")]))
            ]
        )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        stream_function_call_arguments=True,  # On — should still NOT thread on unary.
    )
    assert llm.client is not None

    def fake_tool(query: str) -> str:
        """Search."""
        return query

    bound = llm.bind_tools([fake_tool])

    with patch.object(
        llm.client.aio.models, "generate_content", side_effect=fake_generate
    ):
        await bound.ainvoke("hi")

    config = captured["config"]
    tool_config = getattr(config, "tool_config", None)
    fcc_flag = None
    if tool_config is not None and tool_config.function_calling_config is not None:
        fcc_flag = tool_config.function_calling_config.stream_function_call_arguments
    assert fcc_flag in (None, False), (
        "Vertex :generateContent rejects stream_function_call_arguments — "
        "must not be set on unary requests; got "
        f"tool_config={tool_config!r}, fcc_flag={fcc_flag!r}"
    )


def test_parse_response_candidate_surfaces_structured_partial_args() -> None:
    """``PartialArg`` deltas surface structured via ``additional_kwargs``.

    ``_parse_response_candidate`` emits no ``tool_call_chunks`` for chunks
    that carry only ``partial_args`` — typed leaf updates surface via the
    ``gemini_partial_args`` side channel; the SDK-assembled ``fc.args``
    drives ``tool_call_chunks`` once present.
    """
    candidate = Candidate(
        content=Content(
            role="model",
            parts=[
                Part(
                    function_call=FunctionCall(
                        id="call-1",
                        name="search",
                        partial_args=[
                            PartialArg(
                                json_path="$.query",
                                string_value="hello",
                                will_continue=True,
                            ),
                            PartialArg(
                                json_path="$.query",
                                string_value=" world",
                                will_continue=False,
                            ),
                        ],
                    )
                )
            ],
        )
    )

    msg = _parse_response_candidate(candidate, streaming=True)

    assert isinstance(msg, AIMessageChunk)
    # No tool_call_chunks at this layer — buffer downstream emits them.
    assert msg.tool_call_chunks == []

    structured = msg.additional_kwargs["gemini_partial_args"]
    assert len(structured) == 2
    assert structured[0]["json_path"] == "$.query"
    assert structured[0]["value"] == "hello"
    assert structured[0]["tool_call_id"] == "call-1"
    assert structured[0]["sdk_call_id"] == "call-1"
    assert structured[0]["name"] == "search"
    assert structured[1]["value"] == " world"
    assert structured[1]["will_continue"] is False
    assert structured[1]["sdk_call_id"] == "call-1"


def test_parse_response_candidate_seal_chunk_is_full_json() -> None:
    """Final part with ``args`` (no ``partial_args``) seals with full JSON.

    Guarantees ``tool_call_chunks[-1]["args"]`` is a parseable complete object,
    preserving downstream ``parse_partial_json`` compatibility.
    """
    seal = Candidate(
        content=Content(
            role="model",
            parts=[
                Part(
                    function_call=FunctionCall(
                        id="call-1",
                        name="search",
                        args={"query": "hello world"},
                    )
                )
            ],
        )
    )

    msg = _parse_response_candidate(seal, streaming=True)
    assert isinstance(msg, AIMessageChunk)
    assert len(msg.tool_call_chunks) == 1
    chunk = msg.tool_call_chunks[0]
    assert chunk["id"] == "call-1"
    assert json.loads(chunk["args"] or "{}") == {"query": "hello world"}
    # No structured deltas on a seal chunk.
    assert "gemini_partial_args" not in msg.additional_kwargs


def test_flag_off_regression_emits_single_atomic_chunk() -> None:
    """With flag off + no ``partial_args``, behavior matches legacy code path."""
    candidate = Candidate(
        content=Content(
            role="model",
            parts=[
                Part(
                    function_call=FunctionCall(
                        id="call-7",
                        name="search",
                        args={"query": "hello world"},
                    )
                )
            ],
        )
    )

    msg = _parse_response_candidate(candidate, streaming=True)
    assert isinstance(msg, AIMessageChunk)
    assert len(msg.tool_call_chunks) == 1
    assert msg.tool_call_chunks[0]["id"] == "call-7"
    assert json.loads(msg.tool_call_chunks[0]["args"] or "{}") == {
        "query": "hello world"
    }
    assert "gemini_partial_args" not in msg.additional_kwargs


@pytest.mark.asyncio
async def test_astream_chunks_merge_to_full_args() -> None:
    """End-to-end: streaming several chunks and ``+``-merging yields full args."""

    async def fake_astream(
        **_kwargs: object,
    ) -> AsyncIterator[GenerateContentResponse]:
        async def gen() -> AsyncIterator[GenerateContentResponse]:
            yield _streaming_candidate(
                Part(
                    function_call=FunctionCall(
                        id="call-1",
                        name="search",
                        partial_args=[
                            PartialArg(
                                json_path="$.query",
                                string_value="hello",
                                will_continue=True,
                            )
                        ],
                    )
                )
            )
            yield _streaming_candidate(
                Part(
                    function_call=FunctionCall(
                        id="call-1",
                        name="search",
                        partial_args=[
                            PartialArg(
                                json_path="$.query",
                                string_value=" world",
                                will_continue=False,
                            )
                        ],
                    )
                )
            )
            yield _streaming_candidate(
                Part(
                    function_call=FunctionCall(
                        id="call-1",
                        name="search",
                        args={"query": "hello world"},
                    )
                )
            )

        return gen()

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        stream_function_call_arguments=True,
    )
    assert llm.client is not None

    def fake_tool(query: str) -> str:
        """Search."""
        return query

    bound = llm.bind_tools([fake_tool])

    with patch.object(
        llm.client.aio.models,
        "generate_content_stream",
        new=AsyncMock(side_effect=fake_astream),
    ):
        chunks: list[AIMessageChunk] = []
        async for chunk in bound.astream("hi"):
            assert isinstance(chunk, AIMessageChunk)
            chunks.append(chunk)

    assert len(chunks) >= 3
    merged = chunks[0]
    for c in chunks[1:]:
        merged = merged + c  # type: ignore[assignment]

    # parse_partial_json on the merged AIMessageChunk yields the complete dict.
    assert merged.tool_calls, "Expected at least one fully-parsed tool_call"
    assert merged.tool_calls[0]["name"] == "search"
    assert merged.tool_calls[0]["args"] == {"query": "hello world"}
