"""Live integration tests for Gemini Interactions API capabilities."""

from __future__ import annotations

import base64
import importlib
import io
import math
import os
import struct
import wave
import zlib
from typing import Any, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI

pytestmark = pytest.mark.flaky(retries=3, delay=1)

_FLASH_MODEL = "gemini-3.8-flash"
_OVERRIDE_MODEL = "gemini-3.6-flash"
_DEEP_RESEARCH_AGENT = "deep-research-preview-04-2026"
_VERTEX_INTERACTIONS_LOCATIONS = frozenset({"global", "us", "eu"})
_YOUTUBE_URL = "https://www.youtube.com/watch?v=GbzEDgcuGJU"
_HTTPS_IMAGE_URL = (
    "https://raw.githubusercontent.com/langchain-ai/docs/"
    "4d11d08b6b0e210bd456943f7a22febbd168b543/src/images/agentic-rag-output.png"
)


def _get_vertex_interactions_location() -> str:
    """Return a Vertex AI location supported by the Interactions API.

    Vertex Interactions rejects regional locations such as `us-central1`, so
    default to `'global'` unless `GOOGLE_CLOUD_LOCATION` is explicitly one of
    the supported endpoints (`'global'`, `'us'`, `'eu'`).
    """
    loc = (os.environ.get("GOOGLE_CLOUD_LOCATION") or "").strip().lower()
    return loc if loc in _VERTEX_INTERACTIONS_LOCATIONS else "global"


@pytest.fixture
def backend_config(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Provide Interactions-compatible backend configuration.

    Overrides `tests/conftest.py`'s `backend_config` fixture so Vertex AI runs
    always specify a supported Interactions location (`'global'`, `'us'`, or
    `'eu'`) even when CI exports a regional `GOOGLE_CLOUD_LOCATION` such as
    `'us-central1'`.
    """
    if request.param == "vertex_ai":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            pytest.skip("Vertex AI requires GOOGLE_CLOUD_PROJECT env var")
        return {
            "vertexai": True,
            "project": project,
            "location": _get_vertex_interactions_location(),
            "api_key": None,
        }
    return {}


def _make_solid_png_b64(r: int = 220, g: int = 20, b: int = 20, size: int = 24) -> str:
    """Build a valid uncompressed-valid RGB PNG with exact CRC32 checksums."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw_rows = b"".join(b"\x00" + bytes([r, g, b]) * size for _ in range(size))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw_rows))
        + _chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


_RED_PNG_B64 = _make_solid_png_b64(230, 15, 15)


def _make_pdf_b64(label: str) -> str:
    """Build a minimal valid single-page PDF containing `label` as visible text."""
    stream_bytes = f"BT /F1 24 Tf 72 700 Td ({label}) Tj ET".encode("ascii")
    objects: list[bytes] = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        ),
        (
            f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        ),
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    buf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for obj in objects:
        offsets.append(len(buf))
        buf.extend(obj)
    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    buf.extend(b"0000000000 65535 f \n")
    for off in offsets:
        buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    buf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    return base64.b64encode(bytes(buf)).decode("ascii")


def _make_wav_b64(freq_hz: float = 440.0, duration_s: float = 0.6) -> str:
    """Generate a valid 16-bit mono PCM WAV sine tone encoded as base64."""
    sample_rate = 16000
    n_samples = int(sample_rate * duration_s)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            val = int(16000 * math.sin(2.0 * math.pi * freq_hz * i / sample_rate))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))
    return base64.b64encode(bio.getvalue()).decode("ascii")


@tool
def compute_vault_code(a: int, b: int) -> str:
    """Compute the deterministic vault code from two integers `a` and `b`."""
    return f"VAULT-{a * b + 19}"


def test_previous_interaction_id_multiturn_conversation(backend_config: dict) -> None:
    """Verify server-side stateful multi-turn via `use_previous_interaction_id`."""
    llm = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        use_interactions_api=True,
        use_previous_interaction_id=True,
        timeout=90,
        **backend_config,
    )

    q1 = HumanMessage(
        "Remember the secret project token KRYPTON-731. Confirm with 'STORED'."
    )
    ai1 = llm.invoke([q1])
    assert isinstance(ai1, AIMessage)
    iid1 = ai1.response_metadata.get("interaction_id")
    assert isinstance(iid1, str) and iid1
    assert ai1.response_metadata.get("stored") is True

    # Auto-chained Turn 2: history contains ai1; client slices delta + chains iid1
    q2 = HumanMessage("What was the secret project token? Reply with just the token.")
    ai2 = llm.invoke([q1, ai1, q2])
    assert "KRYPTON-731" in ai2.text.upper()
    iid2 = ai2.response_metadata.get("interaction_id")
    assert isinstance(iid2, str) and iid2 and iid2 != iid1

    # Explicit `previous_interaction_id` Turn 3 passing ONLY the new HumanMessage
    ai3 = llm.invoke(
        [HumanMessage("What are the three digits in that token?")],
        previous_interaction_id=iid2,
    )
    assert "731" in ai3.text or all(d in ai3.text for d in ("7", "3", "1"))


async def test_previous_interaction_id_tool_loop_sync_async_stream(
    backend_config: dict,
) -> None:
    """Verify `use_previous_interaction_id=True` tool loops in sync, async, stream."""
    llm = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        use_interactions_api=True,
        use_previous_interaction_id=True,
        timeout=90,
        **backend_config,
    )
    bound_force = llm.bind_tools([compute_vault_code], tool_choice="any")
    bound_auto = llm.bind_tools([compute_vault_code])
    prompt = HumanMessage(
        "Call compute_vault_code with a=12 and b=11, then report the exact VAULT code."
    )

    # 1. Synchronous tool loop
    ai_call_sync = bound_force.invoke([prompt])
    assert ai_call_sync.tool_calls
    assert ai_call_sync.response_metadata.get("stored") is True
    assert ai_call_sync.response_metadata.get("interaction_id")
    tool_msg_sync = compute_vault_code.invoke(ai_call_sync.tool_calls[0])
    final_sync = bound_auto.invoke([prompt, ai_call_sync, tool_msg_sync])
    assert "VAULT-151" in final_sync.text.upper() or "151" in final_sync.text
    assert final_sync.response_metadata.get(
        "interaction_id"
    ) != ai_call_sync.response_metadata.get("interaction_id")

    # 2. Asynchronous tool loop
    ai_call_async = await bound_force.ainvoke([prompt])
    assert ai_call_async.tool_calls
    assert ai_call_async.response_metadata.get("stored") is True
    tool_msg_async = compute_vault_code.invoke(ai_call_async.tool_calls[0])
    final_async = await bound_auto.ainvoke([prompt, ai_call_async, tool_msg_async])
    assert "VAULT-151" in final_async.text.upper() or "151" in final_async.text

    # 3. Streaming tool loop
    stream_call: AIMessageChunk | None = None
    for chunk in bound_force.stream([prompt]):
        c_call = cast("AIMessageChunk", chunk)
        stream_call = c_call if stream_call is None else stream_call + c_call
    assert stream_call is not None
    assert stream_call.tool_calls
    assert stream_call.response_metadata.get("stored") is True
    assert stream_call.response_metadata.get("interaction_id")
    tool_msg_stream = compute_vault_code.invoke(stream_call.tool_calls[0])
    ai_from_stream = AIMessage(
        content=stream_call.content,
        tool_calls=stream_call.tool_calls,
        additional_kwargs=stream_call.additional_kwargs,
        response_metadata=stream_call.response_metadata,
        id=stream_call.id,
    )
    stream_final: AIMessageChunk | None = None
    for chunk in bound_auto.stream([prompt, ai_from_stream, tool_msg_stream]):
        c_fin = cast("AIMessageChunk", chunk)
        stream_final = c_fin if stream_final is None else stream_final + c_fin
    assert stream_final is not None
    assert "VAULT-151" in stream_final.text.upper() or "151" in stream_final.text


def test_env_var_routing_and_per_call_fallback(
    backend_config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify `LANGCHAIN_GOOGLE_USE_INTERACTIONS_API` and per-call fallback."""
    monkeypatch.setenv("LANGCHAIN_GOOGLE_USE_INTERACTIONS_API", "true")
    llm = ChatGoogleGenerativeAI(model=_FLASH_MODEL, timeout=90, **backend_config)
    assert llm.use_interactions_api is None

    res_int = llm.invoke("Reply with the single word PONG.")
    assert "interaction_id" in res_int.response_metadata
    assert res_int.response_metadata.get("stored") is False
    assert res_int.response_metadata.get("status") == "completed"
    assert "PONG" in res_int.text.upper()

    # Per-call fallback to generateContent overrides the env var
    res_gc = llm.invoke("Reply with the single word PING.", use_interactions_api=False)
    assert "interaction_id" not in res_gc.response_metadata
    assert "PING" in res_gc.text.upper()

    # Env var false overridden per-call back to Interactions API (with store=True)
    monkeypatch.setenv("LANGCHAIN_GOOGLE_USE_INTERACTIONS_API", "false")
    res_int2 = llm.invoke(
        "Reply with the single word ZING.", use_interactions_api=True, store=True
    )
    assert res_int2.response_metadata.get("interaction_id")
    assert res_int2.response_metadata.get("stored") is True
    assert "ZING" in res_int2.text.upper()


def test_configurable_fields_and_init_chat_model(backend_config: dict) -> None:
    """Verify `configurable_fields` runtime routing and `init_chat_model`."""
    configurable = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL, store=True, timeout=90, **backend_config
    ).configurable_fields(
        use_interactions_api=ConfigurableField(id="use_interactions_api"),
        model=ConfigurableField(id="model"),
    )

    r_int = configurable.invoke(
        "Reply with ALPHA.",
        config={
            "configurable": {
                "use_interactions_api": True,
                "model": _OVERRIDE_MODEL,
            }
        },
    )
    assert r_int.response_metadata.get("interaction_id")
    assert r_int.response_metadata.get("stored") is True
    assert _OVERRIDE_MODEL in str(r_int.response_metadata.get("model_name"))

    r_gc = configurable.invoke(
        "Reply with BETA.",
        config={"configurable": {"use_interactions_api": False}},
    )
    assert "interaction_id" not in r_gc.response_metadata
    assert "BETA" in r_gc.text.upper()

    try:
        lc_cm = importlib.import_module("langchain.chat_models")
    except ImportError:
        return

    init_llm = lc_cm.init_chat_model(
        f"google_genai:{_FLASH_MODEL}",
        use_interactions_api=True,
        store=True,
        timeout=90,
        **backend_config,
    )
    r_init = init_llm.invoke("Reply with GAMMA.")
    assert r_init.response_metadata.get("interaction_id")
    assert "GAMMA" in r_init.text.upper()


@pytest.mark.parametrize("output_version", [None, "v0", "interactions/v1", "v1"])
def test_output_versions_invoke_and_stream(
    backend_config: dict, output_version: str | None
) -> None:
    """Verify non-streaming and streaming across all 4 `output_version` modes."""
    kwargs: dict[str, Any] = {
        "model": _FLASH_MODEL,
        "use_interactions_api": True,
        "include_thoughts": True,
        "timeout": 90,
        **backend_config,
    }
    if output_version is not None:
        kwargs["output_version"] = output_version

    llm = ChatGoogleGenerativeAI(**kwargs)
    prompt = "Compute 17 * 19 step by step and state the final number."

    inv = llm.invoke(prompt)
    assert "323" in inv.text
    assert "interaction_id" in inv.response_metadata
    assert inv.response_metadata.get("stored") is False
    assert inv.response_metadata.get("status") == "completed"
    assert inv.usage_metadata is not None
    assert inv.usage_metadata["total_tokens"] > 0

    full: AIMessageChunk | None = None
    chunks: list[AIMessageChunk] = []
    for ch in llm.stream(prompt):
        chunks.append(ch)
        full = ch if full is None else full + ch

    assert full is not None
    assert chunks[-1].chunk_position == "last"
    assert "323" in full.text
    assert "interaction_id" in full.response_metadata
    assert full.response_metadata.get("status") == "completed"
    assert full.usage_metadata is not None
    assert full.usage_metadata["total_tokens"] > 0

    if output_version == "v1":
        assert inv.response_metadata.get("output_version") == "v1"
        assert isinstance(inv.content, list)
        assert any(isinstance(b, dict) and b.get("type") == "text" for b in inv.content)
        assert any(
            isinstance(b, dict) and b.get("type") == "text" for b in full.content
        )
    elif output_version == "interactions/v1":
        assert inv.response_metadata.get("output_version") == "interactions/v1"
        assert isinstance(inv.content, list)
        assert any(isinstance(b, dict) and b.get("type") == "text" for b in inv.content)


async def test_stream_events_v3(backend_config: dict) -> None:
    """Verify LangChain `stream_events` & `astream_events` with `version='v3'`."""
    llm = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        use_interactions_api=True,
        output_version="v1",
        timeout=90,
        **backend_config,
    )

    sync_events = list(
        llm.stream_events("Say hello in exactly three words.", version="v3")
    )
    sync_names = [e.get("event") for e in sync_events if isinstance(e, dict)]
    assert "message-start" in sync_names
    assert "content-block-start" in sync_names
    assert "content-block-delta" in sync_names
    assert "content-block-finish" in sync_names
    assert "message-finish" in sync_names

    async_iter = await llm.astream_events(
        "Say welcome in exactly three words.", version="v3"
    )
    async_events = [e async for e in async_iter]
    async_names = [e.get("event") for e in async_events if isinstance(e, dict)]
    assert "message-start" in async_names
    assert "content-block-delta" in async_names
    assert "message-finish" in async_names


def test_builtin_google_search_citations(backend_config: dict) -> None:
    """Verify `google_search` populates grounding metadata, v1, and interactions/v1."""
    query = (
        "Use Google Search to check what the current latest major Python 3 release "
        "series is and who created Python, citing official web sources."
    )
    llm_v1 = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        use_interactions_api=True,
        output_version="v1",
        timeout=120,
        **backend_config,
    )
    resp = llm_v1.invoke(query, tools=[{"type": "google_search"}])
    assert "Guido" in resp.text
    gm = resp.response_metadata.get("grounding_metadata")
    assert isinstance(gm, dict)
    assert "web_search_queries" in gm
    assert "grounding_chunks" in gm

    blocks = resp.content_blocks
    block_types = {b.get("type") for b in blocks}
    assert "text" in block_types
    for blk in blocks:
        if blk["type"] == "text":
            annotations = blk.get("annotations") or []
            for ann in annotations:
                if ann.get("type") == "citation":
                    assert "url" in ann or "title" in ann

    llm_ia = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        use_interactions_api=True,
        output_version="interactions/v1",
        timeout=120,
        **backend_config,
    )
    resp_ia = llm_ia.invoke(query, tools=[{"type": "google_search"}])
    assert isinstance(resp_ia.content, list)
    ia_calls = [
        b
        for b in resp_ia.content
        if isinstance(b, dict)
        and b.get("type") == "server_tool_call"
        and b.get("name") == "google_search"
    ]
    assert ia_calls, f"Expected google_search server_tool_call in {resp_ia.content}"


def test_builtin_code_execution(backend_config: dict) -> None:
    """Verify `code_execution` emits v0 code blocks and v1 server tool calls."""
    llm = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        use_interactions_api=True,
        timeout=120,
        **backend_config,
    )
    prompt = (
        "You MUST write and execute Python code to compute "
        "sum(i * i for i in range(1, 101)) and report the exact integer."
    )

    resp_v0 = llm.invoke(prompt, tools=[{"type": "code_execution"}])
    assert "338350" in resp_v0.text.replace(",", "")
    assert isinstance(resp_v0.content, list)
    v0_types = {b.get("type") for b in resp_v0.content if isinstance(b, dict)}
    assert "executable_code" in v0_types
    assert "code_execution_result" in v0_types

    # Verify v1 translation onto server_tool_call / server_tool_result
    v1_blocks = resp_v0.content_blocks
    stc = [
        b
        for b in v1_blocks
        if b.get("type") == "server_tool_call" and b.get("name") == "code_interpreter"
    ]
    str_res = [b for b in v1_blocks if b.get("type") == "server_tool_result"]
    assert stc, f"Expected code_interpreter server_tool_call in {v1_blocks}"
    assert str_res, f"Expected server_tool_result in {v1_blocks}"


def test_builtin_url_context(backend_config: dict) -> None:
    """Verify built-in `url_context` fetches page content and emits v1 server steps."""
    llm = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        use_interactions_api=True,
        output_version="v1",
        timeout=120,
        **backend_config,
    )
    resp = llm.invoke(
        "Use url_context to inspect https://example.com and tell me its main heading.",
        tools=[{"type": "url_context"}],
    )
    assert "example domain" in resp.text.lower()
    btypes = {b.get("type") for b in resp.content_blocks}
    assert "text" in btypes


def test_multimodal_image_pdf_audio_and_tool_messages(backend_config: dict) -> None:
    """Verify image (base64 + HTTPS URL), PDF, WAV audio, and media ToolMessages."""
    llm = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        use_interactions_api=True,
        timeout=120,
        **backend_config,
    )

    # 1. Base64 image
    r_img_b64 = llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "What color is this square? One word."},
                    {
                        "type": "image",
                        "base64": _RED_PNG_B64,
                        "mime_type": "image/png",
                    },
                ]
            )
        ]
    )
    assert "red" in r_img_b64.text.lower()

    # 2. HTTPS image URL
    r_img_url = llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this diagram in one sentence."},
                    {"type": "image", "url": _HTTPS_IMAGE_URL},
                ]
            )
        ]
    )
    assert r_img_url.text.strip()

    # 3. Inline PDF containing known marker text
    pdf_b64 = _make_pdf_b64("INTERACTIONS-PDF-42")
    r_pdf = llm.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Read the exact text written in this PDF verbatim.",
                    },
                    {
                        "type": "file",
                        "base64": pdf_b64,
                        "mime_type": "application/pdf",
                    },
                ]
            )
        ]
    )
    assert "INTERACTIONS-PDF-42" in r_pdf.text.upper()

    # 4. Inline WAV audio
    wav_b64 = _make_wav_b64(440.0, 0.6)
    r_wav = llm.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Confirm whether this audio contains a tone or speech.",
                    },
                    {"type": "audio", "base64": wav_b64, "mime_type": "audio/wav"},
                ]
            )
        ]
    )
    assert r_wav.text.strip()

    # 5. Image ToolMessage
    def fetch_swatch() -> str:
        """Fetch color swatch image."""
        return ""

    r_tm_img = llm.bind_tools([fetch_swatch]).invoke(
        [
            HumanMessage("Call fetch_swatch and tell me the color of the image."),
            AIMessage(
                "",
                tool_calls=[{"id": "img_1", "name": "fetch_swatch", "args": {}}],
            ),
            ToolMessage(
                content=[
                    {
                        "type": "image",
                        "base64": _RED_PNG_B64,
                        "mime_type": "image/png",
                    }
                ],
                tool_call_id="img_1",
                name="fetch_swatch",
            ),
        ]
    )
    assert "red" in r_tm_img.text.lower()

    # 6. PDF ToolMessage
    pdf_tm_b64 = _make_pdf_b64("BADGE-917")

    def fetch_badge_pdf() -> str:
        """Fetch badge PDF document."""
        return ""

    r_tm_pdf = llm.bind_tools([fetch_badge_pdf]).invoke(
        [
            HumanMessage(
                "Call fetch_badge_pdf and repeat the exact badge text in the PDF."
            ),
            AIMessage(
                "",
                tool_calls=[{"id": "pdf_1", "name": "fetch_badge_pdf", "args": {}}],
            ),
            ToolMessage(
                content=[
                    {
                        "type": "file",
                        "base64": pdf_tm_b64,
                        "mime_type": "application/pdf",
                    }
                ],
                tool_call_id="pdf_1",
                name="fetch_badge_pdf",
            ),
        ]
    )
    assert "BADGE-917" in r_tm_pdf.text.upper()


def test_multimodal_youtube_static_and_agentic(backend_config: dict) -> None:
    """Verify YouTube URL understanding in both STATIC and AGENTIC modes."""
    llm = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        use_interactions_api=True,
        timeout=180,
        **backend_config,
    )

    # Static YouTube video understanding
    res_static = llm.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "video",
                        "url": _YOUTUBE_URL,
                        "mime_type": "video/mp4",
                        "media_processing": "STATIC",
                    },
                    {"type": "text", "text": "What is deepagents in one sentence?"},
                ]
            )
        ]
    )
    assert res_static.text.strip()
    assert res_static.usage_metadata is not None

    # Agentic YouTube video understanding (streamed to inspect server tool steps)
    full: AIMessageChunk | None = None
    for chunk in llm.stream(
        [
            HumanMessage(
                content=[
                    {
                        "type": "video",
                        "url": _YOUTUBE_URL,
                        "mime_type": "video/mp4",
                        "media_processing": "AGENTIC",
                    },
                    {"type": "text", "text": "What is deepagents in one sentence?"},
                ]
            )
        ],
        output_version="v1",
    ):
        full = chunk if full is None else full + chunk

    assert full is not None
    assert full.text.strip()
    btypes = [b.get("type") for b in full.content_blocks]
    assert "server_tool_call" in btypes
    assert "server_tool_result" in btypes
    assert full.usage_metadata is not None
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )


def test_managed_background_agent_create_and_status(backend_config: dict) -> None:
    """Verify live `CreateAgentInteraction` wire format + status check + cancel."""
    dr_llm = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        agent=_DEEP_RESEARCH_AGENT,
        agent_config={"thinking_summaries": "auto"},
        background=True,
        store=True,
        timeout=90,
        **backend_config,
    )
    submitted = dr_llm.invoke(
        "Give a brief two-sentence overview of LangChain.", poll=False
    )
    iid = submitted.response_metadata.get("interaction_id")
    assert isinstance(iid, str) and iid
    assert submitted.response_metadata.get("stored") is True
    assert submitted.response_metadata.get("status") in (
        "in_progress",
        "completed",
    )

    assert dr_llm.client is not None
    try:
        fetched = dr_llm.client.interactions.get(id=iid)
        assert getattr(fetched, "id", None) == iid
        assert getattr(fetched, "status", None) in (
            "in_progress",
            "completed",
            "cancelled",
        )
    finally:
        if submitted.response_metadata.get("status") == "in_progress":
            try:
                dr_llm.client.interactions.cancel(id=iid)
            except Exception:
                pass


@pytest.mark.extended
def test_managed_background_agent_full_completion(backend_config: dict) -> None:
    """Full polled end-to-end run of managed `deep-research-preview-04-2026` agent."""
    dr_llm = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        agent=_DEEP_RESEARCH_AGENT,
        background=True,
        store=True,
        polling_interval=5.0,
        timeout=600,
        **backend_config,
    )
    result = dr_llm.invoke("Write a concise 1-paragraph summary of Python PEP 8.")
    assert result.text.strip()
    assert result.response_metadata.get("status") == "completed"
    assert result.response_metadata.get("interaction_id")


def test_stateful_empty_json_list_tool_result_and_prev_id_metadata(
    backend_config: dict,
) -> None:
    """Vertex/AI Studio stateful tool-calling handles empty JSON lists `[]` cleanly."""
    import json

    @tool
    def get_order_record(oid: str) -> str:
        """Look up an order record by order ID."""
        return oid

    llm = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        use_interactions_api=True,
        use_previous_interaction_id=True,
        **backend_config,
    ).bind_tools([get_order_record])

    msgs: list = [HumanMessage("Call get_order_record for oid 'W1'.")]
    t1 = llm.invoke(msgs)
    assert t1.response_metadata["stored"] is True
    assert "previous_interaction_id" not in t1.response_metadata
    assert len(t1.tool_calls) == 1

    tool_json = json.dumps(
        {"oid": "W1", "fulfillments": [], "nested": {"tags": [], "meta": {}}}
    )
    msgs.extend(
        [
            t1,
            ToolMessage(
                content=tool_json,
                tool_call_id=t1.tool_calls[0]["id"],
                name="get_order_record",
            ),
        ]
    )
    t2 = llm.invoke(msgs)
    assert (
        t2.response_metadata.get("previous_interaction_id")
        == (t1.response_metadata["interaction_id"])
    )
    assert "W1" in t2.text

    msgs.extend([t2, HumanMessage("Repeat the order ID from earlier.")])
    t3 = llm.invoke(msgs)
    assert (
        t3.response_metadata.get("previous_interaction_id")
        == (t2.response_metadata["interaction_id"])
    )
    assert "W1" in t3.text


def test_empty_and_signed_empty_ai_message_stateless_replay(
    backend_config: dict,
) -> None:
    """Stateless replay omits empty and bare-signed-stop `AIMessage` turns cleanly."""
    llm = ChatGoogleGenerativeAI(
        model=_FLASH_MODEL,
        use_interactions_api=True,
        store=False,
        **backend_config,
    )

    for empty_ai in (
        AIMessage(""),
        AIMessage([]),
        AIMessage([{"type": "text", "text": ""}]),
        AIMessage([{"type": "text", "text": "", "extras": {"signature": "..."}}]),
        AIMessage(
            [{"type": "text", "text": "", "extras": {"signature": "..."}}],
            response_metadata={"model_provider": "google_genai"},
        ),
    ):
        res = llm.invoke(
            [HumanMessage("Hi"), empty_ai, HumanMessage("Reply with the word OK.")]
        )
        assert res.response_metadata.get("status") == "completed"
        assert "OK" in res.text.upper()
