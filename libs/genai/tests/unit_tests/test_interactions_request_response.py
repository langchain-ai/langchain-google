"""Unit tests for Gemini Interactions API request and response translation."""

from __future__ import annotations

import base64
import copy
import json
import warnings
from typing import Annotated, Any, Literal, cast
from unittest.mock import AsyncMock, Mock, patch

import httpx
import httpx2
import pytest
from google.genai._gaos import models as gaos_models
from google.genai._gaos import utils as gaos_utils
from google.genai._gaos.types.interactions import AllowedTools, Interaction
from google.genai.types import (
    FunctionCallingConfigMode,
    HarmBlockThreshold,
    HarmCategory,
    HttpOptions,
    LatLng,
    RetrievalConfig,
    SafetyPolicy,
    ServiceTier,
    ThinkingLevel,
    ToolConfig,
    UrlContext,
)
from google.genai.types import Tool as GoogleTool
from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI, Environment, Modality
from langchain_google_genai._interactions_request import (
    _SKIP_THOUGHT_SIGNATURE_B64,
    _get_last_interaction_messages,
    _messages_to_interaction_steps,
    _prepare_interactions_request,
)
from langchain_google_genai._interactions_response import (
    _interaction_to_result,
    _parse_usage_metadata,
    _utf8_byte_offsets_to_char_offsets,
)
from langchain_google_genai.chat_models import (
    _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY,
    SKIP_THOUGHT_SIGNATURE_VALIDATOR,
    GoogleInteractionsAPIError,
    GoogleInteractionsContextOverflowError,
    GoogleInvalidRequestError,
    GoogleRateLimitError,
    GoogleTimeoutError,
)

MODEL_NAME = "gemini-3.8-flash"
FAKE_KEY = "fake-api-key"


def _llm(**kwargs: Any) -> ChatGoogleGenerativeAI:
    params: dict[str, Any] = {
        "model": MODEL_NAME,
        "google_api_key": FAKE_KEY,
        "use_interactions_api": True,
        **kwargs,
    }
    return ChatGoogleGenerativeAI(**params)


def _serialize_sdk_body(create_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Run real `_gaos` Speakeasy Pydantic unmarshaling + JSON serialization."""
    body_kw = {
        k: v
        for k, v in create_kwargs.items()
        if k
        not in {
            "api_version",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
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


# ---------------------------------------------------------------------------
# 1. Smart store & stateful chaining
# ---------------------------------------------------------------------------


def test_smart_store_defaults_priorities_and_conflicts() -> None:
    base = _llm()
    p0 = _prepare_interactions_request(base, [HumanMessage("hi")])
    assert p0.create_kwargs["store"] is False

    # Auto True when stateful, background, or agent is active
    assert (
        _prepare_interactions_request(
            _llm(use_previous_interaction_id=True), [HumanMessage("hi")]
        ).create_kwargs["store"]
        is True
    )
    assert (
        _prepare_interactions_request(
            base, [HumanMessage("hi")], previous_interaction_id="int_1"
        ).create_kwargs["store"]
        is True
    )
    assert (
        _prepare_interactions_request(
            base, [HumanMessage("hi")], background=True
        ).create_kwargs["store"]
        is True
    )
    assert (
        _prepare_interactions_request(
            _llm(agent="deep-research-preview-04-2026"), [HumanMessage("hi")]
        ).create_kwargs["store"]
        is True
    )

    # Explicit overrides: call kwarg > self.store
    s_true = _llm(store=True)
    assert (
        _prepare_interactions_request(s_true, [HumanMessage("hi")]).create_kwargs[
            "store"
        ]
        is True
    )
    assert (
        _prepare_interactions_request(
            s_true, [HumanMessage("hi")], store=False
        ).create_kwargs["store"]
        is False
    )

    # Invalid combinations raise GoogleInvalidRequestError
    with pytest.raises(GoogleInvalidRequestError) as exc_prev:
        _prepare_interactions_request(
            base, [HumanMessage("hi")], store=False, previous_interaction_id="int_1"
        )
    assert isinstance(exc_prev.value, ModelInvalidRequestError)

    with pytest.raises(GoogleInvalidRequestError):
        _prepare_interactions_request(
            base, [HumanMessage("hi")], store=False, background=True
        )


def test_vertex_store_true_replay_downgrades_with_warning() -> None:
    v_llm = _llm(vertexai=True, project="test-proj", location="global", store=True)
    msgs = [
        HumanMessage("compute"),
        AIMessage("Calling add", tool_calls=[{"id": "c1", "name": "add", "args": {}}]),
        ToolMessage("3", tool_call_id="c1", name="add"),
    ]
    with pytest.warns(UserWarning, match="downgrading store=False"):
        prep = _prepare_interactions_request(v_llm, msgs)
    assert prep.create_kwargs["store"] is False
    assert prep.requested_store is False


def test_stateful_slicing_resends_scoped_config_and_backfills_tool_name() -> None:
    @tool
    def lookup(q: str) -> str:
        """Lookup query."""
        return q

    msgs: list[Any] = [
        SystemMessage("Be concise."),
        HumanMessage("Turn 1"),
        AIMessage(
            "Old stored",
            response_metadata={
                "model_provider": "google_genai",
                "stored": True,
                "interaction_id": "int_old",
            },
        ),
        HumanMessage("Turn 2"),
        AIMessage(
            "Unstored checkpoint",
            id="lc_123",
            response_metadata={
                "model_provider": "google_genai",
                "stored": False,
                "interaction_id": None,
            },
        ),
        AIMessage(
            "",
            tool_calls=[{"id": "tc_9", "name": "lookup", "args": {"q": "x"}}],
            response_metadata={
                "model_provider": "google_genai",
                "stored": True,
                "id": "int_latest",
            },
        ),
        ToolMessage('{"answer": 42}', tool_call_id="tc_9"),
    ]
    iid, tail = _get_last_interaction_messages(msgs)
    assert iid == "int_latest"
    assert len(tail) == 1 and isinstance(tail[0], ToolMessage)

    stateful_llm = _llm(
        use_previous_interaction_id=True,
        max_output_tokens=256,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH
        },
    )
    prep = _prepare_interactions_request(
        stateful_llm,
        msgs,
        tools=[lookup],
        response_format={"type": "json_object"},
    )
    wire = _serialize_sdk_body(prep.create_kwargs)
    assert wire["previous_interaction_id"] == "int_latest"
    assert wire["store"] is True
    assert wire["system_instruction"] == "Be concise."
    assert wire["tools"][0]["name"] == "lookup"
    assert wire["generation_config"]["max_output_tokens"] == 256
    assert wire["response_format"] == {"type": "text", "mime_type": "application/json"}
    assert wire["safety_settings"] == [
        {"type": "hate_speech", "threshold": "block_only_high"}
    ]
    assert wire["input"] == [
        {
            "type": "function_result",
            "call_id": "tc_9",
            "name": "lookup",
            "result": '{"answer": 42}',
        }
    ]


# ---------------------------------------------------------------------------
# 2. Message -> step conversion, thought signatures, ToolMessage media
# ---------------------------------------------------------------------------


def test_system_join_promotion_empty_human_double_messages_and_prefill() -> None:
    # System-only promotes to user_input
    sys_, steps = _messages_to_interaction_steps(
        [SystemMessage("Part A"), SystemMessage([{"type": "text", "text": "Part B"}])],
        effective_model=MODEL_NAME,
    )
    assert sys_ is None
    assert steps == [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "Part A\n\nPart B"}],
        }
    ]

    # Double messages + empty HumanMessage warning + name attribute ignored cleanly
    with pytest.warns(UserWarning, match="HumanMessage with empty content"):
        sys2, steps2 = _messages_to_interaction_steps(
            [
                SystemMessage("sys1"),
                SystemMessage("sys2"),
                HumanMessage(""),
                HumanMessage("h1", name="example_user"),
                HumanMessage("h2"),
                AIMessage("a1"),
                AIMessage("a2"),
                HumanMessage("final"),
            ],
            effective_model=MODEL_NAME,
        )
    assert sys2 == "sys1\n\nsys2"
    assert len(steps2) == 5
    wire = _serialize_sdk_body(
        _prepare_interactions_request(
            _llm(), [SystemMessage("s"), HumanMessage("hi", name="u")]
        ).create_kwargs
    )
    assert wire["system_instruction"] == "s"

    # Fixed-sampling model rejects trailing assistant prefill
    with pytest.raises(GoogleInvalidRequestError, match="prefilling"):
        _messages_to_interaction_steps(
            [HumanMessage("hi"), AIMessage("partial")],
            effective_model="gemini-3.6-flash",
        )


def test_assistant_step_order_interleaved_segments_and_signatures() -> None:
    sig_a = base64.b64encode(b"real-sig-A").decode("ascii")
    sig_b = base64.b64encode(b"real-sig-B").decode("ascii")

    # Out-of-order [text, thinking] + unsigned thought dropped + single segment order
    ai_reordered = AIMessage(
        content=[
            {"type": "thinking", "thinking": "unsigned MUST drop"},
            {"type": "text", "text": "Checking now"},
            {"type": "thinking", "thinking": "Signed plan", "signature": sig_a},
        ],
        tool_calls=[{"id": "c1", "name": "fn", "args": {"x": 1}}],
        response_metadata={"model_provider": "google_genai"},
    )
    _, steps_single = _messages_to_interaction_steps(
        [HumanMessage("go"), ai_reordered, ToolMessage("ok", tool_call_id="c1")],
        effective_model=MODEL_NAME,
    )
    ai_steps = steps_single[1:-1]
    assert [s["type"] for s in ai_steps] == ["thought", "model_output", "function_call"]
    assert ai_steps[0] == {
        "type": "thought",
        "signature": sig_a,
        "summary": [{"type": "text", "text": "Signed plan"}],
    }
    assert "signature" not in ai_steps[2]

    # Multi-segment round-trip: [thought(A), model_output, fc(a), thought(B), fc(b)]
    raw_interaction = Interaction.model_validate(
        {
            "id": "int_multi",
            "model": MODEL_NAME,
            "status": "requires_action",
            "steps": [
                {
                    "type": "thought",
                    "signature": sig_a,
                    "summary": [{"type": "text", "text": "Step A"}],
                },
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Let me call A then B"}],
                },
                {
                    "type": "function_call",
                    "id": "call_a",
                    "name": "tool_a",
                    "arguments": {"n": 1},
                },
                {"type": "thought", "signature": sig_b},
                {
                    "type": "function_call",
                    "id": "call_b",
                    "name": "tool_b",
                    "arguments": {"n": 2},
                },
            ],
        }
    )
    for ov in (None, "v1", "interactions/v1"):
        res = _interaction_to_result(
            raw_interaction, requested_store=True, output_version=ov
        )
        ai_msg = res.generations[0].message
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.additional_kwargs[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY] == {
            "call_a": sig_a,
            "call_b": sig_b,
        }
        _, replayed = _messages_to_interaction_steps(
            [
                HumanMessage("start"),
                ai_msg,
                ToolMessage("ra", tool_call_id="call_a"),
                ToolMessage("rb", tool_call_id="call_b"),
            ],
            effective_model=MODEL_NAME,
        )
        mid = replayed[1:-2]
        assert [s["type"] for s in mid] == [
            "thought",
            "model_output",
            "function_call",
            "thought",
            "function_call",
        ], f"Failed segment order for output_version={ov}: {mid}"
        assert mid[0]["signature"] == sig_a  # never double-encoded!
        assert mid[2]["id"] == "call_a" and "signature" not in mid[2]
        assert mid[3] == {"type": "thought", "signature": sig_b}
        assert mid[4]["id"] == "call_b" and "signature" not in mid[4]


def test_foreign_signature_stripping_and_bypass_injection_gemini3_vs_25() -> None:
    foreign_ai = AIMessage(
        content=[
            {"type": "thinking", "thinking": "claude thought", "signature": "YW50aA=="},
            {"type": "text", "text": "Using tool", "extras": {"signature": "YW50aA=="}},
        ],
        tool_calls=[{"id": "t1", "name": "f", "args": {}}],
        additional_kwargs={
            _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY: {"t1": "YW50aA=="}
        },
        response_metadata={"model_provider": "anthropic"},
    )
    # On Gemini 3+, foreign signature stripped AND active-loop bypass injected
    _, steps_g3 = _messages_to_interaction_steps(
        [HumanMessage("q"), foreign_ai, ToolMessage("r", tool_call_id="t1")],
        effective_model="gemini-3.8-flash",
    )
    assert steps_g3[1] == {
        "type": "thought",
        "signature": _SKIP_THOUGHT_SIGNATURE_B64,
    }
    assert steps_g3[2] == {
        "type": "model_output",
        "content": [{"type": "text", "text": "Using tool"}],
    }
    assert steps_g3[3]["type"] == "function_call"

    # On Gemini 2.5, NO bypass thought step injected
    _, steps_g25 = _messages_to_interaction_steps(
        [HumanMessage("q"), foreign_ai, ToolMessage("r", tool_call_id="t1")],
        effective_model="gemini-2.5-flash",
    )
    assert [s["type"] for s in steps_g25] == [
        "user_input",
        "model_output",
        "function_call",
        "function_result",
    ]

    # Few-shot historical tool call BEFORE final HumanMessage is not active-loop
    _, steps_fewshot = _messages_to_interaction_steps(
        [
            HumanMessage("1+2"),
            AIMessage("", tool_calls=[{"id": "fs1", "name": "add", "args": {"a": 1}}]),
            ToolMessage("3", tool_call_id="fs1"),
            AIMessage("It is 3"),
            HumanMessage("3+4"),
        ],
        effective_model="gemini-4-argon",
    )
    assert "thought" not in [s["type"] for s in steps_fewshot]

    # Raw SKIP_THOUGHT_SIGNATURE_VALIDATOR sentinel string encodes cleanly once
    ai_raw_skip = AIMessage(
        "",
        tool_calls=[{"id": "k1", "name": "f", "args": {}}],
        additional_kwargs={
            _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY: {
                "k1": SKIP_THOUGHT_SIGNATURE_VALIDATOR
            }
        },
    )
    _, s_skip = _messages_to_interaction_steps(
        [HumanMessage("q"), ai_raw_skip, ToolMessage("ok", tool_call_id="k1")],
        effective_model="gemini-3.8-flash",
    )
    assert s_skip[1] == {"type": "thought", "signature": _SKIP_THOUGHT_SIGNATURE_B64}


def test_tool_message_id_matching_error_image_and_pdf_split() -> None:
    b64_img = base64.b64encode(b"\x89PNG\r\n\x1a\nfakepng").decode("ascii")
    b64_pdf = base64.b64encode(b"%PDF-1.4 fake").decode("ascii")

    # Missing IDs deterministically paired + error status + image subcontent + PDF split
    msgs: list[Any] = [
        HumanMessage("run"),
        AIMessage(
            "",
            tool_calls=[
                {"id": None, "name": "err_fn", "args": {}},
                {"id": "img_1", "name": "get_img", "args": {}},
                {"id": "pdf_1", "name": "get_pdf", "args": {}},
            ],
        ),
        ToolMessage("Boom!", tool_call_id="", name="err_fn", status="error"),
        ToolMessage(
            content=[
                {"type": "text", "text": "diagram:"},
                {"type": "image", "base64": b64_img, "mime_type": "image/png"},
            ],
            tool_call_id="img_1",
        ),
        ToolMessage(
            content=[
                {"type": "file", "base64": b64_pdf, "mime_type": "application/pdf"}
            ],
            tool_call_id="pdf_1",
        ),
    ]
    _, steps = _messages_to_interaction_steps(msgs, effective_model=MODEL_NAME)
    fc_err = next(
        s for s in steps if s["type"] == "function_call" and s["name"] == "err_fn"
    )
    fr_err = next(
        s for s in steps if s["type"] == "function_result" and s.get("name") == "err_fn"
    )
    assert fr_err["call_id"] == fc_err["id"]
    assert fr_err["is_error"] is True
    assert fr_err["result"] == "Boom!"

    fr_img = next(
        s for s in steps if s["type"] == "function_result" and s["call_id"] == "img_1"
    )
    assert fr_img["name"] == "get_img"
    assert fr_img["result"] == [
        {"type": "text", "text": "diagram:"},
        {"type": "image", "data": b64_img, "mime_type": "image/png"},
    ]

    pdf_idx = next(
        i
        for i, s in enumerate(steps)
        if s["type"] == "function_result" and s["call_id"] == "pdf_1"
    )
    assert steps[pdf_idx] == {
        "type": "function_result",
        "call_id": "pdf_1",
        "name": "get_pdf",
        "result": "See attached media.",
    }
    assert steps[pdf_idx + 1] == {
        "type": "user_input",
        "content": [
            {"type": "document", "data": b64_pdf, "mime_type": "application/pdf"}
        ],
    }
    # Validate real SDK Pydantic unmarshaling on full step sequence
    _serialize_sdk_body({"model": MODEL_NAME, "input": steps, "store": False})


# ---------------------------------------------------------------------------
# 3. Multimodal content & URL loading
# ---------------------------------------------------------------------------


def test_multimodal_formats_urls_gcs_and_local_file_security(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    b64_data = base64.b64encode(b"rawbytes").decode("ascii")
    msg = HumanMessage(
        content=[
            {"cachePoint": {"type": "default"}},
            {"type": "text-plain", "text": "plain doc"},
            {"type": "input_audio", "input_audio": {"data": b64_data, "format": "wav"}},
            {"type": "input_audio", "input_audio": {"data": b64_data, "format": "mp3"}},
            {
                "type": "input_audio",
                "input_audio": {"data": b64_data, "format": "pcm16"},
            },
            {
                "type": "file",
                "file": {
                    "filename": "doc.pdf",
                    "file_data": f"data:application/pdf;base64,{b64_data}",
                },
            },
            {
                "type": "video",
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "media_resolution": "low",
                "media_processing": "AGENTIC",
            },
            {
                "type": "file",
                "file_id": "https://generativelanguage.googleapis.com/v1beta/files/f1",
                "mime_type": "text/csv",
            },
            {
                "type": "image",
                "url": "gs://my-bucket/pic.png",
                "mime_type": "image/png",
            },
        ]
    )

    # AI Studio warns & drops gs:// while keeping Files API and YouTube URIs
    with pytest.warns(UserWarning, match="gs://"):
        _, steps_ais = _messages_to_interaction_steps(
            [msg], effective_model=MODEL_NAME, use_vertexai=False
        )
    c_ais = steps_ais[0]["content"]
    assert c_ais[0] == {"type": "text", "text": "plain doc"}
    assert c_ais[1]["mime_type"] == "audio/wav"
    assert c_ais[2]["mime_type"] == "audio/mp3"
    assert c_ais[3]["mime_type"] == "audio/pcm"
    assert c_ais[4] == {
        "type": "document",
        "data": b64_data,
        "mime_type": "application/pdf",
    }
    assert c_ais[5] == {
        "type": "video",
        "uri": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "resolution": "low",
        "processing": "agentic",
    }
    assert c_ais[6] == {
        "type": "document",
        "uri": "https://generativelanguage.googleapis.com/v1beta/files/f1",
        "mime_type": "text/csv",
    }
    assert len(c_ais) == 7

    # Vertex keeps gs:// as uri
    _, steps_vtx = _messages_to_interaction_steps(
        [msg], effective_model=MODEL_NAME, use_vertexai=True
    )
    assert steps_vtx[0]["content"][-1] == {
        "type": "image",
        "uri": "gs://my-bucket/pic.png",
        "mime_type": "image/png",
    }

    # External URL loading uses httpx2.Client (never requests.get) and forwards
    # LANGCHAIN_TESTS_USER_AGENT
    monkeypatch.setenv("LANGCHAIN_TESTS_USER_AGENT", "LC-Test-Agent/1.0")
    seen_sync_calls: list[tuple[str, dict[str, Any], Any]] = []

    def _fake_sync_get(self_c: httpx2.Client, url: str, **kw: Any) -> httpx2.Response:
        seen_sync_calls.append((url, kw, self_c.timeout))
        return httpx2.Response(
            200,
            content=b"\x89PNG\r\n\x1a\n\x00",
            headers={"Content-Type": "image/png"},
            request=httpx2.Request("GET", url),
        )

    with (
        patch("requests.get", side_effect=AssertionError("requests.get must not run")),
        patch.object(httpx2.Client, "get", _fake_sync_get),
    ):
        _, s_http = _messages_to_interaction_steps(
            [HumanMessage([{"type": "image", "url": "https://example.com/a.png"}])],
            effective_model=MODEL_NAME,
        )
    assert len(seen_sync_calls) == 1
    assert seen_sync_calls[0][0] == "https://example.com/a.png"
    assert seen_sync_calls[0][1] == {"headers": {"User-Agent": "LC-Test-Agent/1.0"}}
    assert seen_sync_calls[0][2] == httpx2.Timeout(60.0)
    assert s_http[0]["content"][0]["type"] == "image"
    assert s_http[0]["content"][0]["mime_type"] == "image/png"

    # Local file path rejected for security
    secret_file = tmp_path / "secret.png"
    secret_file.write_bytes(b"secret")
    with pytest.raises(ValueError, match="local files is no longer supported"):
        _messages_to_interaction_steps(
            [HumanMessage([{"type": "image", "url": str(secret_file)}])],
            effective_model=MODEL_NAME,
        )


# ---------------------------------------------------------------------------
# 4. Tools, tool_choice, thinking, sampling (2.25.0 extra_body), outputs, agents
# ---------------------------------------------------------------------------


def test_tools_preserve_json_schema_without_mutation_and_lift_maps_config() -> None:
    class SubItem(BaseModel):
        kind: Literal["a", "b"]

    class ComplexArgs(BaseModel):
        count: int = Field(..., gt=0, lt=100)
        item: SubItem | None = None

    @tool(args_schema=ComplexArgs)
    def run_complex(count: int, item: SubItem | None = None) -> str:
        """Run complex."""
        return str(count)

    @tool
    def zero_arg() -> int:
        """No args."""
        return 1

    class V1Tool(BaseModelV1):
        """V1 schema tool."""

        q: str = FieldV1(description="query")

    class TDTool(TypedDict):
        """TypedDict tool."""

        city: Annotated[str, ..., "city name"]

    chain_tool = (
        ChatPromptTemplate.from_messages([("human", "{topic}")])
        | GenericFakeChatModel(messages=iter(["ok"]))
        | StrOutputParser()
    ).as_tool(name="gen_topic", description="Generate")

    caller_mcp = {
        "type": "mcp_server",
        "name": "docs",
        "url": "https://mcp.example.com",
        "headers": {"Authorization": "Bearer secret"},
    }
    caller_snapshot = copy.deepcopy(caller_mcp)

    schema_with_defs: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "tree_fn",
            "description": "Tree input",
            "parameters": {
                "type": "object",
                "$defs": {
                    "Leaf": {
                        "type": "object",
                        "properties": {"tag": {"type": "string", "enum": ["x", "y"]}},
                    }
                },
                "properties": {"root": {"$ref": "#/$defs/Leaf"}},
                "required": ["root"],
            },
        },
    }
    defs_snapshot = copy.deepcopy(schema_with_defs)

    bound = _llm().bind_tools(
        [
            run_complex,
            schema_with_defs,
            zero_arg,
            V1Tool,
            TDTool,
            chain_tool,
            {"googleSearch": {}},
            {"google_maps": {"enable_widget": True}},
            GoogleTool(url_context=UrlContext()),
            caller_mcp,
        ],
        tool_choice="run_complex",
        tool_config=ToolConfig(
            retrieval_config=RetrievalConfig(
                lat_lng=LatLng(latitude=37.7749, longitude=-122.4194)
            )
        ),
    )
    prep = _prepare_interactions_request(
        _llm(),
        [HumanMessage("hi")],
        **bound.kwargs,  # type: ignore[attr-defined]
    )
    assert caller_mcp == caller_snapshot
    assert schema_with_defs == defs_snapshot
    wire = _serialize_sdk_body(prep.create_kwargs)
    by_type_or_name = {t.get("name") or t["type"]: t for t in wire["tools"]}
    complex_params = by_type_or_name["run_complex"]["parameters"]
    assert "anyOf" in complex_params["properties"]["item"]
    assert complex_params["properties"]["count"]["exclusiveMinimum"] == 0
    assert complex_params["properties"]["count"]["exclusiveMaximum"] == 100
    tree_params = by_type_or_name["tree_fn"]["parameters"]
    assert "$defs" in tree_params
    assert tree_params["properties"]["root"] == {"$ref": "#/$defs/Leaf"}
    assert by_type_or_name["zero_arg"]["parameters"] == {
        "type": "object",
        "properties": {},
    }
    assert "V1Tool" in by_type_or_name
    assert "TDTool" in by_type_or_name
    assert "gen_topic" in by_type_or_name
    assert by_type_or_name["google_search"] == {"type": "google_search"}
    assert by_type_or_name["google_maps"] == {
        "type": "google_maps",
        "enable_widget": True,
        "latitude": 37.7749,
        "longitude": -122.4194,
    }
    assert by_type_or_name["url_context"] == {"type": "url_context"}
    assert by_type_or_name["docs"]["type"] == "mcp_server"
    assert wire["generation_config"]["tool_choice"] == {
        "allowed_tools": {"mode": "any", "tools": ["run_complex"]}
    }

    # Server-only tools skip function tool_choice cleanly
    srv_prep = _prepare_interactions_request(
        _llm(),
        [HumanMessage("hi")],
        tools=[{"google_maps": {}}],
        tool_choice="required",
    )
    assert "generation_config" not in srv_prep.create_kwargs


def test_thinking_clamping_sampling_extra_body_and_api_version() -> None:
    # minimal clamped -> low on gemini-3.8-flash and gemini-4-argon, kept on 3.5-flash
    p38 = _prepare_interactions_request(
        _llm(thinking_level="minimal"), [HumanMessage("hi")]
    )
    assert p38.create_kwargs["generation_config"]["thinking_level"] == "low"

    p4 = _prepare_interactions_request(
        _llm(model="gemini-4-argon", thinking_budget=0), [HumanMessage("hi")]
    )
    assert p4.create_kwargs["generation_config"] == {
        "thinking_level": "low",
        "thinking_summaries": "none",
    }

    p35 = _prepare_interactions_request(
        _llm(model="gemini-3.5-flash", thinking_budget=0), [HumanMessage("hi")]
    )
    assert p35.create_kwargs["generation_config"] == {
        "thinking_level": "minimal",
        "thinking_summaries": "none",
    }

    # Budget buckets + include_thoughts=True + dynamic -1
    assert _prepare_interactions_request(
        _llm(thinking_budget=512, include_thoughts=True), [HumanMessage("hi")]
    ).create_kwargs["generation_config"] == {
        "thinking_level": "low",
        "thinking_summaries": "auto",
    }
    assert (
        _prepare_interactions_request(
            _llm(thinking_budget=4096), [HumanMessage("hi")]
        ).create_kwargs["generation_config"]["thinking_level"]
        == "medium"
    )
    assert (
        _prepare_interactions_request(
            _llm(thinking_budget=16384), [HumanMessage("hi")]
        ).create_kwargs["generation_config"]["thinking_level"]
        == "high"
    )
    assert (
        "generation_config"
        not in _prepare_interactions_request(
            _llm(thinking_budget=-1), [HumanMessage("hi")]
        ).create_kwargs
    )

    # SDK 2.25.0: temperature & top_p sent via extra_body keeping max_output_tokens
    with pytest.warns(UserWarning, match="top_k"):
        p_samp = _prepare_interactions_request(
            _llm(
                temperature=0.25,
                top_p=0.8,
                top_k=20,
                max_output_tokens=128,
                seed=7,
                stop=["END"],
                api_version="v1",
            ),
            [HumanMessage("hi")],
            http_options=HttpOptions(timeout=15000, headers={"X-Req": "1"}),
            cached_content="cachedContents/abc",
        )
    assert p_samp.create_kwargs["api_version"] == "v1beta"
    assert p_samp.create_kwargs["timeout"] == 15.0
    assert p_samp.create_kwargs["extra_headers"] == {"X-Req": "1"}
    wire = _serialize_sdk_body(p_samp.create_kwargs)
    assert wire["generation_config"] == {
        "max_output_tokens": 128,
        "seed": 7,
        "stop_sequences": ["END"],
        "temperature": 0.25,
        "top_p": 0.8,
    }

    # Vertex api_version="v1" guarded to v1beta1/projects/.../locations/global
    v_v1 = _llm(vertexai=True, project="my-gcp", api_version="v1")
    pv = _prepare_interactions_request(
        v_v1, [HumanMessage("hi")], model="models/gemini-3.8-flash"
    )
    assert pv.create_kwargs["model"] == "gemini-3.8-flash"
    assert pv.create_kwargs["api_version"] == "v1beta1/projects/my-gcp/locations/global"
    assert pv.create_kwargs["timeout"] == 600.0
    assert "extra_body" not in pv.create_kwargs  # Gemini 3+ default temperature is None

    # CreateAgentInteraction omits model and generation_config, infers agent_config.type
    pa = _prepare_interactions_request(
        _llm(
            agent="deep-research-preview-04-2026",
            agent_config={"thinking_summaries": "auto"},
            environment="env_1",
            max_output_tokens=500,
        ),
        [HumanMessage("research TPUs")],
    )
    wire_a = _serialize_sdk_body(pa.create_kwargs)
    assert "model" not in wire_a
    assert "generation_config" not in wire_a
    assert wire_a["agent"] == "deep-research-preview-04-2026"
    assert wire_a["agent_config"] == {
        "type": "deep-research",
        "thinking_summaries": "auto",
    }
    assert wire_a["environment"] == "env_1"


def test_multimodal_image_and_tts_response_formats() -> None:
    p_img = _prepare_interactions_request(
        _llm(
            model="gemini-3.1-flash-image",
            response_modalities=[Modality.TEXT, Modality.IMAGE],
            image_config={"aspect_ratio": "16:9", "image_size": "2K"},
        ),
        [HumanMessage("draw")],
    )
    wire_img = _serialize_sdk_body(p_img.create_kwargs)
    assert wire_img["response_format"] == [
        {"type": "text"},
        {"type": "image", "aspect_ratio": "16:9", "image_size": "2K"},
    ]

    p_tts = _prepare_interactions_request(
        _llm(model="gemini-3.8-flash-tts"),
        [HumanMessage("Hello world")],
    )
    wire_tts = _serialize_sdk_body(p_tts.create_kwargs)
    assert wire_tts["response_format"] == {"type": "audio"}
    assert wire_tts["generation_config"]["speech_config"] == [{"voice": "Kore"}]


# ---------------------------------------------------------------------------
# 5. Response parsing, UTF-8 citations, parsed Pydantic, errors, polling
# ---------------------------------------------------------------------------


def test_utf8_citation_offsets_grounding_usage_and_parsed_pydantic() -> None:
    # Multi-byte UTF-8: emoji (4B) + CJK (3B each) + en-dash (3B)
    text = "🏆 日本代表 won 2\u20131 today."
    target = "日本代表 won 2\u20131"
    b_start = len("🏆 ".encode())  # 5 bytes (char index 2)
    b_end = b_start + len(target.encode("utf-8"))
    cs, ce, seg = _utf8_byte_offsets_to_char_offsets(text, b_start, b_end)
    assert (cs, ce, seg) == (2, 2 + len(target), target)

    inter = Interaction.model_validate(
        {
            "id": "int_search_1",
            "model": MODEL_NAME,
            "status": "completed",
            "service_tier": "priority",
            "steps": [
                {"type": "thought", "signature": "c2lnX3RleHQ="},
                {
                    "type": "google_search_call",
                    "id": "gs_1",
                    "arguments": {"queries": ["Japan score"]},
                    "search_type": "web_search",
                },
                {
                    "type": "google_search_result",
                    "call_id": "gs_1",
                    "result": [{"search_suggestions": "<div>chip</div>"}],
                },
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://fifa.example/match",
                                    "title": "Match Report",
                                    "start_index": b_start,
                                    "end_index": b_end,
                                }
                            ],
                        }
                    ],
                },
            ],
            "usage": {
                "total_input_tokens": 100,
                "total_tool_use_tokens": 25,
                "total_output_tokens": 30,
                "total_thought_tokens": 45,
                "total_cached_tokens": 40,
                "total_tokens": 200,
            },
        }
    )

    msg_v0 = _interaction_to_result(inter, requested_store=True).generations[0].message
    assert isinstance(msg_v0, AIMessage)
    assert msg_v0.id == "int_search_1"
    assert msg_v0.response_metadata["stored"] is True
    assert msg_v0.response_metadata["service_tier"] == "priority"
    gm = msg_v0.response_metadata["grounding_metadata"]
    assert gm["web_search_queries"] == ["Japan score"]
    assert gm["grounding_chunks"] == [
        {"web": {"uri": "https://fifa.example/match", "title": "Match Report"}}
    ]
    assert gm["grounding_supports"] == [
        {
            "segment": {"start_index": cs, "end_index": ce, "text": target},
            "grounding_chunk_indices": [0],
        }
    ]
    # Core v1 translator preserves preceding thought sig AND grounding citations
    cbs = msg_v0.content_blocks
    assert cbs[0] == {
        "type": "reasoning",
        "reasoning": "",
        "extras": {"signature": "c2lnX3RleHQ="},
    }
    txt_cbs = [b for b in cbs if b["type"] == "text"]
    assert txt_cbs[0]["annotations"][0]["start_index"] == cs  # type: ignore[typeddict-item]
    assert txt_cbs[0]["annotations"][0]["cited_text"] == target  # type: ignore[typeddict-item]

    assert msg_v0.usage_metadata == {
        "input_tokens": 125,
        "output_tokens": 75,
        "total_tokens": 200,
        "input_token_details": {"cache_read": 40},
        "output_token_details": {"reasoning": 45},
    }

    # interactions/v1 exposes server_tool_call/result + inline annotations
    msg_ia = (
        _interaction_to_result(
            inter, requested_store=False, output_version="interactions/v1"
        )
        .generations[0]
        .message
    )
    assert msg_ia.response_metadata["stored"] is False
    assert [b["type"] for b in msg_ia.content] == [  # type: ignore[index]
        "reasoning",
        "server_tool_call",
        "server_tool_result",
        "text",
    ]

    # Pydantic response_format populates additional_kwargs["parsed"] iff no tool_calls
    class Answer(BaseModel):
        value: int

    json_inter = Interaction.model_validate(
        {
            "id": "",
            "model": MODEL_NAME,
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": '{"value": 99}'}],
                }
            ],
        }
    )
    parsed_msg = (
        _interaction_to_result(json_inter, pydantic_schema=Answer)
        .generations[0]
        .message
    )
    assert parsed_msg.additional_kwargs["parsed"] == Answer(value=99)
    assert parsed_msg.id is not None and parsed_msg.id.startswith("lc_")


def test_invalid_tool_call_and_failed_cancelled_status_errors() -> None:
    from google.genai._gaos.types.interactions import FunctionCallStep

    # Unvalidated malformed arguments string -> invalid_tool_calls
    bad_step = FunctionCallStep.model_construct(
        type="function_call",
        id="bad_1",
        name="broken",
        arguments="{not valid json",  # type: ignore[arg-type]
    )
    bad_fc = BaseModel.model_construct.__func__(  # type: ignore[attr-defined]
        Interaction,
        id="int_bad_fc",
        model=MODEL_NAME,
        status="requires_action",
        steps=[bad_step],
    )
    msg = _interaction_to_result(bad_fc).generations[0].message
    assert isinstance(msg, AIMessage)
    assert msg.tool_calls == []
    assert len(msg.invalid_tool_calls) == 1
    assert msg.invalid_tool_calls[0]["id"] == "bad_1"
    assert msg.invalid_tool_calls[0]["args"] == "{not valid json"

    # Failed status with context overflow vs rate limit vs generic failure vs cancelled
    overflow_i = Interaction.model_validate(
        {
            "status": "failed",
            "errors": [
                {
                    "code": "invalid_argument",
                    "message": "Input token count exceeds context window",
                }
            ],
        }
    )
    with pytest.raises(GoogleInteractionsContextOverflowError):
        _interaction_to_result(overflow_i, model_name=MODEL_NAME)

    quota_i = Interaction.model_validate(
        {
            "status": "failed",
            "errors": [{"code": "resource_exhausted", "message": "Quota hit"}],
        }
    )
    with pytest.raises(GoogleRateLimitError, match="Quota hit"):
        _interaction_to_result(quota_i, model_name=MODEL_NAME)

    cancelled_i = Interaction.model_validate(
        {
            "id": "int_c",
            "status": "cancelled",
            "errors": [{"code": "cancelled", "message": "User cancelled"}],
        }
    )
    with pytest.raises(GoogleInteractionsAPIError, match="User cancelled"):
        _interaction_to_result(cancelled_i, model_name=MODEL_NAME)


async def test_sync_and_async_background_polling_and_timeout() -> None:
    queued = Interaction.model_validate({"id": "int_bg", "status": "queued"})
    running = Interaction.model_validate({"id": "int_bg", "status": "in_progress"})
    done = Interaction.model_validate(
        {
            "id": "int_bg",
            "agent": "deep-research-preview-04-2026",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Research report."}],
                }
            ],
        }
    )

    llm = _llm(
        agent="deep-research-preview-04-2026", background=True, polling_interval=0.01
    )
    sync_ia = Mock()
    sync_ia.create = Mock(return_value=queued)
    sync_ia.get = Mock(side_effect=[running, done])

    async_ia = Mock()
    async_ia.create = AsyncMock(return_value=queued)
    async_ia.get = AsyncMock(side_effect=[running, done])

    with (
        patch.object(type(llm.client), "interactions", sync_ia, create=True),
        patch("langchain_google_genai._interactions_response.time.sleep"),
    ):
        res_sync = llm.invoke("Research TPUs")
    assert res_sync.text == "Research report."
    assert res_sync.response_metadata["stored"] is True
    assert res_sync.response_metadata["interaction_id"] == "int_bg"
    assert sync_ia.get.call_count == 2

    aio_client = llm.async_client
    with (
        patch.object(type(aio_client), "interactions", async_ia, create=True),
        patch(
            "langchain_google_genai._interactions_response.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        res_async = await llm.ainvoke("Research TPUs")
    assert res_async.text == "Research report."
    assert async_ia.get.call_count == 2

    # Deadline exceeded raises classified GoogleTimeoutError
    stuck_ia = Mock()
    stuck_ia.create = Mock(return_value=queued)
    stuck_ia.get = Mock(return_value=running)
    ticks = 0.0

    def _fake_mono() -> float:
        nonlocal ticks
        val = ticks
        ticks += 0.6
        return val

    with (
        patch.object(type(llm.client), "interactions", stuck_ia, create=True),
        patch(
            "langchain_google_genai._interactions_response.time.monotonic",
            side_effect=_fake_mono,
        ),
        patch("langchain_google_genai._interactions_response.time.sleep"),
        pytest.raises(GoogleTimeoutError, match="Timed out waiting"),
    ):
        llm.invoke("Research TPUs", timeout=1.0)


def test_all_server_tool_and_multimodal_output_steps_and_replay() -> None:
    raw_pcm = b"\x00\x01" * 100
    b64_pcm = base64.b64encode(raw_pcm).decode("ascii")
    b64_png = base64.b64encode(b"\x89PNG\r\n\x1a\nfakepng").decode("ascii")

    inter = Interaction.model_validate(
        {
            "id": "int_all_steps",
            "model": MODEL_NAME,
            "status": "incomplete",
            "steps": [
                {
                    "type": "code_execution_call",
                    "id": "ce_1",
                    "arguments": {"code": "print(3**3)", "language": "python"},
                    "signature": "c2lnX2Nl",
                },
                {
                    "type": "code_execution_result",
                    "call_id": "ce_1",
                    "result": "27\n",
                    "signature": "c2lnX2Nlcg==",
                },
                {
                    "type": "url_context_call",
                    "id": "uc_1",
                    "arguments": {"urls": ["https://example.com"]},
                },
                {
                    "type": "url_context_result",
                    "call_id": "uc_1",
                    "result": [{"url": "https://example.com", "status": "success"}],
                },
                {
                    "type": "google_maps_call",
                    "id": "gm_1",
                    "arguments": {"queries": ["coffee"]},
                },
                {
                    "type": "google_maps_result",
                    "call_id": "gm_1",
                    "result": [
                        {
                            "widget_context_token": "w_tok_1",
                            "places": [
                                {
                                    "name": "Blue Bottle",
                                    "place_id": "places/p1",
                                    "url": "https://maps.google.com/?cid=1",
                                }
                            ],
                        }
                    ],
                },
                {"type": "file_search_call", "id": "fs_1"},
                {"type": "file_search_result", "call_id": "fs_1"},
                {
                    "type": "mcp_server_tool_call",
                    "id": "mcp_1",
                    "name": "search_docs",
                    "server_name": "docs_srv",
                    "arguments": {"q": "genai"},
                },
                {
                    "type": "mcp_server_tool_result",
                    "call_id": "mcp_1",
                    "name": "search_docs",
                    "server_name": "docs_srv",
                    "result": "found",
                },
                {
                    "type": "retrieval_call",
                    "id": "ret_1",
                    "arguments": {"queries": ["rag"]},
                    "retrieval_type": "rag_store",
                },
                {"type": "retrieval_result", "call_id": "ret_1"},
                {"type": "processing_call", "id": "pr_1"},
                {"type": "processing_result", "call_id": "pr_1"},
                {
                    "type": "model_output",
                    "content": [
                        {"type": "text", "text": "Result is 27."},
                        {
                            "type": "image",
                            "data": b64_png,
                            "mime_type": "image/png",
                        },
                        {
                            "type": "audio",
                            "data": b64_pcm,
                            "mime_type": "audio/l16",
                            "sample_rate": 24000,
                            "channels": 1,
                        },
                        {
                            "type": "video",
                            "uri": "https://generativelanguage.googleapis.com/v1beta/files/v1",
                            "mime_type": "video/mp4",
                        },
                        {
                            "type": "document",
                            "uri": "https://generativelanguage.googleapis.com/v1beta/files/d1",
                            "mime_type": "application/pdf",
                        },
                    ],
                },
            ],
        }
    )

    with pytest.warns(UserWarning, match="executable_code"):
        res_v0 = _interaction_to_result(inter, requested_store=True)
    msg_v0 = res_v0.generations[0].message
    assert isinstance(msg_v0, AIMessage)
    assert msg_v0.response_metadata["finish_reason"] == "MAX_TOKENS"
    assert msg_v0.response_metadata["status"] == "incomplete"
    assert msg_v0.additional_kwargs["audio"].startswith(b"RIFF")
    gm = msg_v0.response_metadata["grounding_metadata"]
    assert gm["google_maps_widget_context_token"] == "w_tok_1"
    assert gm["grounding_chunks"] == [
        {
            "maps": {
                "uri": "https://maps.google.com/?cid=1",
                "title": "Blue Bottle",
                "placeId": "places/p1",
            }
        }
    ]

    # Replay msg_v0 into steps: processing dropped, code_exec & mcp preserved
    _, replayed = _messages_to_interaction_steps(
        [HumanMessage("calc"), msg_v0, HumanMessage("next")],
        effective_model=MODEL_NAME,
    )
    rep_types = [s["type"] for s in replayed]
    assert "code_execution_call" in rep_types
    assert "code_execution_result" in rep_types
    assert "mcp_server_tool_call" in rep_types
    assert "mcp_server_tool_result" in rep_types
    assert "processing_call" not in rep_types
    assert "processing_result" not in rep_types
    _serialize_sdk_body({"model": MODEL_NAME, "input": replayed, "store": False})

    # Also verify interactions/v1 output format covers all blocks cleanly
    res_ia = _interaction_to_result(inter, output_version="interactions/v1")
    ia_types = [b["type"] for b in res_ia.generations[0].message.content]  # type: ignore[index]
    assert "audio" in ia_types and "image" in ia_types and "video" in ia_types
    assert "file" in ia_types and "server_tool_call" in ia_types


@pytest.mark.asyncio
async def test_unstamped_anthropic_signature_stripping_and_poll_false() -> None:
    """Regression coverage for unstamped Anthropic signatures and `poll=False`."""
    claude_sig = "ErUBCkYICBgCIkDCTQUXPc3O7nHXd302Zercaz8WrrpddpOqHITxBih5ze"
    explicit_gemini_sig = base64.b64encode(b"explicit-gemini-sig").decode("ascii")

    # 1) Historical unstamped Anthropic thinking block drops untrusted signature
    _, steps_hist = _messages_to_interaction_steps(
        [
            HumanMessage([{"type": "text", "text": "Hello"}]),
            AIMessage(
                [
                    {
                        "type": "thinking",
                        "thinking": "Simple greeting.",
                        "signature": claude_sig,
                    },
                    {"type": "text", "text": "Hello, how are you?"},
                ]
            ),
            HumanMessage([{"type": "text", "text": "Well, thanks."}]),
        ],
        effective_model="gemini-3.8-flash",
    )
    assert [s["type"] for s in steps_hist] == [
        "user_input",
        "model_output",
        "user_input",
    ]
    assert all("signature" not in str(s) for s in steps_hist)

    # 2) Active-loop unstamped Anthropic thinking + tool_use strips Claude sig and
    # injects official Gemini 3/4 bypass signature
    _, steps_loop = _messages_to_interaction_steps(
        [
            HumanMessage("Pick color"),
            AIMessage(
                [
                    {
                        "type": "thinking",
                        "thinking": "Picking purple",
                        "signature": claude_sig,
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "color_picker",
                        "input": {"fav_color": "purple"},
                    },
                ],
                tool_calls=[
                    {
                        "name": "color_picker",
                        "args": {"fav_color": "purple"},
                        "id": "tu_1",
                    }
                ],
            ),
            ToolMessage("Great!", tool_call_id="tu_1"),
        ],
        effective_model="gemini-3.8-flash",
    )
    assert steps_loop[1] == {
        "type": "thought",
        "signature": _SKIP_THOUGHT_SIGNATURE_B64,
    }
    assert steps_loop[2]["type"] == "function_call"

    # 3) Explicit `thought_signature` on an unstamped block IS preserved
    _, steps_explicit = _messages_to_interaction_steps(
        [
            HumanMessage("Hi"),
            AIMessage(
                [
                    {
                        "type": "thinking",
                        "thinking": "Gemini explicit",
                        "thought_signature": explicit_gemini_sig,
                    },
                    {"type": "text", "text": "Hi there"},
                ]
            ),
            HumanMessage("Follow up"),
        ],
        effective_model="gemini-3.8-flash",
    )
    assert steps_explicit[1] == {
        "type": "thought",
        "signature": explicit_gemini_sig,
        "summary": [{"type": "text", "text": "Gemini explicit"}],
    }

    # 4) `poll=False` returns initial `in_progress` Interaction without calling `.get`
    in_prog = Interaction.model_validate(
        {
            "id": "int_bg_nopoll",
            "agent": "deep-research-preview-04-2026",
            "status": "in_progress",
            "steps": [],
        }
    )
    llm = _llm(
        agent="deep-research-preview-04-2026",
        background=True,
        store=True,
    )
    sync_ic = Mock()
    sync_ic.create = Mock(return_value=in_prog)
    sync_ic.get = Mock()
    with patch.object(type(llm.client), "interactions", sync_ic, create=True):
        sync_res = llm.invoke("Summarize PEP 8", poll=False)
    assert sync_res.response_metadata["interaction_id"] == "int_bg_nopoll"
    assert sync_res.response_metadata["status"] == "in_progress"
    sync_ic.get.assert_not_called()

    async_ic = Mock()
    async_ic.create = AsyncMock(return_value=in_prog)
    async_ic.get = AsyncMock()
    aio_client = llm.async_client
    with patch.object(type(aio_client), "interactions", async_ic, create=True):
        async_res = await llm.ainvoke("Summarize PEP 8", poll=False)
    assert async_res.response_metadata["interaction_id"] == "int_bg_nopoll"
    assert async_res.response_metadata["status"] == "in_progress"
    async_ic.get.assert_not_called()


# ---------------------------------------------------------------------------
# 9. Stateful multi-turn, schema fidelity, ordering, and resilience
# ---------------------------------------------------------------------------


def test_empty_list_tool_results_and_parallel_multimodal_batching() -> None:
    """Vertex stateful empty-list `ToolMessage` safety + parallel media batching."""
    pdf_a = base64.b64encode(b"%PDF-1.4 A").decode("ascii")
    pdf_b = base64.b64encode(b"%PDF-1.4 B").decode("ascii")
    aud_c = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVE").decode("ascii")

    json_str_empty = '{"oid": "W1", "fulfillments": [], "nested": {"tags": []}}'
    dict_payload: Any = {
        "oid": "W2",
        "fulfillments": [],
        "nested": {"tags": [], "meta": {}},
    }
    dict_empty_tm = ToolMessage.model_construct(
        content=dict_payload,
        tool_call_id="c_dict",
        name="get_order",
        type="tool",
    )

    msgs: list[Any] = [
        HumanMessage("Process all"),
        AIMessage(
            "",
            tool_calls=[
                {"id": "c_str", "name": "get_order", "args": {"oid": "W1"}},
                {"id": "c_dict", "name": "get_order", "args": {"oid": "W2"}},
                {"id": "c_pdf1", "name": "fetch_pdf", "args": {"n": 1}},
                {"id": "c_pdf2", "name": "fetch_pdf", "args": {"n": 2}},
            ],
        ),
        ToolMessage(json_str_empty, tool_call_id="c_str", name="get_order"),
        dict_empty_tm,
        ToolMessage(
            [{"type": "file", "base64": pdf_a, "mime_type": "application/pdf"}],
            tool_call_id="c_pdf1",
        ),
        ToolMessage(
            [
                {"type": "text", "text": "Second doc:"},
                {"type": "file", "base64": pdf_b, "mime_type": "application/pdf"},
            ],
            tool_call_id="c_pdf2",
        ),
    ]
    _, steps = _messages_to_interaction_steps(
        msgs, effective_model=MODEL_NAME, use_vertexai=True
    )
    # All 4 function_result steps sit contiguously back-to-back BEFORE a single
    # combined user_input media step (never splitting parallel tool responses).
    assert [s["type"] for s in steps] == [
        "user_input",
        "thought",
        "function_call",
        "function_call",
        "function_call",
        "function_call",
        "function_result",
        "function_result",
        "function_result",
        "function_result",
        "user_input",
    ]
    assert steps[6]["result"] == json_str_empty
    assert isinstance(steps[6]["result"], str)
    assert isinstance(steps[7]["result"], str)
    assert json.loads(steps[7]["result"])["fulfillments"] == []
    assert json.loads(steps[7]["result"])["nested"] == {"tags": [], "meta": {}}
    assert steps[8]["result"] == "See attached media."
    assert steps[9]["result"] == [{"type": "text", "text": "Second doc:"}]
    assert steps[10] == {
        "type": "user_input",
        "content": [
            {"type": "document", "data": pdf_a, "mime_type": "application/pdf"},
            {"type": "document", "data": pdf_b, "mime_type": "application/pdf"},
        ],
    }
    _serialize_sdk_body({"model": MODEL_NAME, "input": steps, "store": True})

    # Also check embedded Anthropic parallel `tool_result` blocks followed by user text
    _, anth_steps = _messages_to_interaction_steps(
        [
            HumanMessage("Run"),
            AIMessage(
                "",
                tool_calls=[
                    {"id": "a1", "name": "f", "args": {}},
                    {"id": "a2", "name": "g", "args": {}},
                ],
            ),
            HumanMessage(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "a1",
                        "content": [
                            {
                                "type": "file",
                                "base64": pdf_a,
                                "mime_type": "application/pdf",
                            }
                        ],
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "a2",
                        "content": [
                            {"type": "audio", "base64": aud_c, "mime_type": "audio/wav"}
                        ],
                    },
                    {"type": "text", "text": "Summarize both attachments"},
                ]
            ),
        ],
        effective_model=MODEL_NAME,
    )
    assert [s["type"] for s in anth_steps] == [
        "user_input",
        "function_call",
        "function_call",
        "function_result",
        "function_result",
        "user_input",
        "user_input",
    ]
    assert len(anth_steps[5]["content"]) == 2
    assert anth_steps[6]["content"] == [
        {"type": "text", "text": "Summarize both attachments"}
    ]


def test_previous_interaction_id_metadata_and_v1_extras_media() -> None:
    """`previous_interaction_id` in `response_metadata` + v1 `extras` media helpers."""
    from langchain_core.messages.content import (
        create_image_block,
        create_video_block,
    )

    b64_img = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
    vid_blk = create_video_block(
        url="https://generativelanguage.googleapis.com/v1beta/files/vid_1",
        mime_type="video/mp4",
        media_processing="AGENTIC",
        media_resolution="MEDIA_RESOLUTION_LOW",
        name="demo_clip",
    )
    img_blk = create_image_block(
        base64=b64_img,
        mime_type="image/png",
        media_resolution="high",
    )
    aud_blk = {
        "type": "audio",
        "base64": b64_img,
        "mime_type": "audio/pcm",
        "extras": {"sample_rate": 16000, "channels": 2},
    }
    _, steps = _messages_to_interaction_steps(
        [HumanMessage(content=[dict(vid_blk), dict(img_blk), aud_blk])],
        effective_model=MODEL_NAME,
    )
    assert steps[0]["content"] == [
        {
            "type": "video",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/vid_1",
            "mime_type": "video/mp4",
            "processing": "agentic",
            "resolution": "low",
            "name": "demo_clip",
        },
        {
            "type": "image",
            "data": b64_img,
            "mime_type": "image/png",
            "resolution": "high",
        },
        {
            "type": "audio",
            "data": b64_img,
            "mime_type": "audio/pcm",
            "sample_rate": 16000,
            "channels": 2,
        },
    ]

    # Turn 1 omits previous_interaction_id; Turn 2 stamps previous_interaction_id
    done_t1 = Interaction.model_validate(
        {
            "id": "int_t1",
            "model": MODEL_NAME,
            "status": "completed",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "Hi"}]}
            ],
        }
    )
    r1 = _interaction_to_result(done_t1, requested_store=True).generations[0].message
    assert "previous_interaction_id" not in r1.response_metadata

    stateful_llm = _llm(use_previous_interaction_id=True)
    done_t2 = Interaction.model_validate(
        {
            "id": "int_t2",
            "model": MODEL_NAME,
            "status": "completed",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "T2"}]}
            ],
        }
    )
    mock_ia = Mock()
    mock_ia.create = Mock(return_value=done_t2)
    with patch.object(type(stateful_llm.client), "interactions", mock_ia, create=True):
        out_t2 = stateful_llm.invoke([HumanMessage("hi"), r1, HumanMessage("next")])
    assert out_t2.response_metadata["previous_interaction_id"] == "int_t1"
    assert out_t2.response_metadata["interaction_id"] == "int_t2"


def test_bind_on_default_then_toggle_interactions_lossless_schema() -> None:
    """Binding tools on a default LLM preserves lossless JSON Schema on runtime flip."""
    from google.genai.types import FunctionDeclaration, Schema, Type

    from langchain_google_genai._interactions_tools import _normalize_genai_schema_dump

    class TagSpec(BaseModel):
        label: str

    class ComplexInput(BaseModel):
        tags: list[TagSpec] = Field(min_length=1, max_length=4)
        threshold: int = Field(ge=0, le=10)
        note: str | None = Field(default=None, description="Optional note")

    @tool(args_schema=ComplexInput)
    def analyze(tags: list[TagSpec], threshold: int, note: str | None = None) -> str:
        """Analyze tags."""
        return str(threshold)

    @tool
    def ping_zero() -> str:
        """Zero-arg ping."""
        return "pong"

    defs_fn = {
        "type": "function",
        "function": {
            "name": "tree_op",
            "parameters": {
                "type": "object",
                "$defs": {
                    "Node": {"type": "object", "properties": {"id": {"type": "string"}}}
                },
                "properties": {"root": {"$ref": "#/$defs/Node"}},
            },
        },
    }
    default_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=FAKE_KEY)
    bound = default_llm.bind_tools([analyze, defs_fn, ping_zero, {"google_search": {}}])
    prep = _prepare_interactions_request(
        default_llm,
        [HumanMessage("run")],
        use_interactions_api=True,
        **bound.kwargs,  # type: ignore[attr-defined]
    )
    tools_out = prep.create_kwargs["tools"]
    # Exact order preserved (`google_search` NOT hoisted ahead of functions)
    assert [t.get("name") or t["type"] for t in tools_out] == [
        "analyze",
        "tree_op",
        "ping_zero",
        "google_search",
    ]
    a_params = tools_out[0]["parameters"]
    assert a_params["properties"]["tags"]["items"]["required"] == ["label"]
    assert a_params["properties"]["tags"]["minItems"] == 1
    assert a_params["properties"]["tags"]["maxItems"] == 4
    assert a_params["properties"]["threshold"]["minimum"] == 0
    assert a_params["properties"]["threshold"]["maximum"] == 10
    assert "nullable" not in json.dumps(a_params)
    assert any(b.get("type") == "null" for b in a_params["properties"]["note"]["anyOf"])
    assert "$defs" in tools_out[1]["parameters"]
    assert tools_out[1]["parameters"]["properties"]["root"] == {"$ref": "#/$defs/Node"}
    assert tools_out[2]["parameters"] == {"type": "object", "properties": {}}

    # Direct FunctionDeclaration(Schema) normalizes nullable, snake_case, & empty req
    fd = FunctionDeclaration(
        name="fd_test",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "items": Schema(
                    type=Type.ARRAY,
                    items=Schema(type=Type.STRING, nullable=True),
                    min_items=2,
                    max_items=5,
                )
            },
        ),
    )
    prep_fd = _prepare_interactions_request(_llm(), [HumanMessage("hi")], tools=[fd])
    fd_params = prep_fd.create_kwargs["tools"][0]["parameters"]
    assert "nullable" not in json.dumps(fd_params)
    assert fd_params["properties"]["items"]["minItems"] == 2
    assert fd_params["properties"]["items"]["maxItems"] == 5
    assert fd_params["properties"]["items"]["items"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }
    assert _normalize_genai_schema_dump(
        {"type": "OBJECT", "properties": None, "required": []}
    ) == {"type": "object", "properties": {}}


async def test_per_request_http_options_base_url_and_gateway_version() -> None:
    """Per-request `http_options.base_url` scopes client; custom gateway keeps `v1`."""
    # 1) Custom gateway instance preserves api_version="v1" (both dev and vertex)
    gw_dev = _llm(
        base_url="https://my-gateway.example.com/api/gemini", api_version="v1"
    )
    assert gw_dev.client is not None
    p_gw_dev = _prepare_interactions_request(gw_dev, [HumanMessage("hi")])
    assert "api_version" not in p_gw_dev.create_kwargs
    assert gw_dev.client.interactions.sdk_configuration.globals.api_version == "v1"

    gw_vtx = _llm(
        vertexai=True,
        base_url="https://my-gateway.example.com/api/gemini",
        api_version="v1",
    )
    p_gw_vtx = _prepare_interactions_request(gw_vtx, [HumanMessage("hi")])
    assert "api_version" not in p_gw_vtx.create_kwargs

    # 2) Per-request base_url + headers + api_version="v1" never hits googleapis.com!
    llm = _llm()
    assert llm.client is not None
    captured_urls: list[str] = []
    captured_headers: list[dict[str, str]] = []

    def _mock_send(request: httpx.Request, **_: Any) -> httpx.Response:
        captured_urls.append(str(request.url))
        captured_headers.append(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "id": "int_proxy",
                "model": MODEL_NAME,
                "status": "completed",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "ok"}],
                    }
                ],
            },
            request=request,
        )

    transport = httpx.MockTransport(_mock_send)
    orig_client = llm.client._api_client._httpx_client
    mock_http = httpx.Client(transport=transport)
    llm.client._api_client._httpx_client = mock_http
    try:
        res = llm.invoke(
            "secret prompt",
            http_options={
                "base_url": "https://proxy.corp.internal/gemini",
                "api_version": "v1",
                "headers": {"Authorization": "Bearer corp-secret"},
            },
        )
    finally:
        llm.client._api_client._httpx_client = orig_client
        mock_http.close()

    assert res.text == "ok"
    assert captured_urls == ["https://proxy.corp.internal/gemini/v1/interactions"]
    assert captured_headers[0].get("authorization") == "Bearer corp-secret"


def test_unary_multi_segment_ordering_parallel_sigs_and_server_step_boundary() -> None:
    """Exact replay of `[thought_1, fc_1, thought_2, model_output, fc_2, fc_3]`."""
    s1 = base64.b64encode(b"sig-seg-1").decode("ascii")
    s2 = base64.b64encode(b"sig-seg-2").decode("ascii")

    # 1) Summary thoughts + interleaved model_output + parallel fc_2, fc_3
    inter = Interaction.model_validate(
        {
            "id": "int_p1_1",
            "model": MODEL_NAME,
            "status": "requires_action",
            "steps": [
                {
                    "type": "thought",
                    "signature": s1,
                    "summary": [{"type": "text", "text": "Plan 1"}],
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "name": "tool_1",
                    "arguments": {"a": 1},
                },
                {
                    "type": "thought",
                    "signature": s2,
                    "summary": [{"type": "text", "text": "Plan 2"}],
                },
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Running 2 and 3"}],
                },
                {
                    "type": "function_call",
                    "id": "fc_2",
                    "name": "tool_2",
                    "arguments": {"b": 2},
                },
                {
                    "type": "function_call",
                    "id": "fc_3",
                    "name": "tool_3",
                    "arguments": {"c": 3},
                },
            ],
        }
    )
    for ov in (None, "interactions/v1"):
        ai = _interaction_to_result(inter, output_version=ov).generations[0].message
        assert ai.additional_kwargs[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY] == {
            "fc_1": s1,
            "fc_2": s2,
            "fc_3": s2,
        }
        _, rep = _messages_to_interaction_steps(
            [
                HumanMessage("go"),
                ai,
                ToolMessage("r1", tool_call_id="fc_1"),
                ToolMessage("r2", tool_call_id="fc_2"),
                ToolMessage("r3", tool_call_id="fc_3"),
            ],
            effective_model=MODEL_NAME,
        )
        mid = rep[1:-3]
        assert [s["type"] for s in mid] == [
            "thought",
            "function_call",
            "thought",
            "model_output",
            "function_call",
            "function_call",
        ]
        assert mid[0]["signature"] == s1
        assert mid[1]["id"] == "fc_1"
        assert mid[2]["signature"] == s2
        assert mid[4]["id"] == "fc_2"
        assert mid[5]["id"] == "fc_3"

    # 2) Signature-only `[thought_1(s1), fc_1, thought_2(s2), model_output, fc_2]`
    inter_sig_only = Interaction.model_validate(
        {
            "id": "int_sig_only",
            "model": MODEL_NAME,
            "status": "requires_action",
            "steps": [
                {"type": "thought", "signature": s1},
                {"type": "function_call", "id": "fc_1", "name": "t1", "arguments": {}},
                {"type": "thought", "signature": s2},
                {"type": "model_output", "content": [{"type": "text", "text": "Mid"}]},
                {"type": "function_call", "id": "fc_2", "name": "t2", "arguments": {}},
            ],
        }
    )
    for ov in (None, "interactions/v1"):
        ai_so = (
            _interaction_to_result(inter_sig_only, output_version=ov)
            .generations[0]
            .message
        )
        _, rep_so = _messages_to_interaction_steps(
            [
                HumanMessage("go"),
                ai_so,
                ToolMessage("1", tool_call_id="fc_1"),
                ToolMessage("2", tool_call_id="fc_2"),
            ],
            effective_model=MODEL_NAME,
        )
        assert [s["type"] for s in rep_so[1:-2]] == [
            "thought",
            "function_call",
            "thought",
            "model_output",
            "function_call",
        ]

    # 3) Server tool step boundary clears `current_segment_sig` so no duplicate thought
    inter_ce = Interaction.model_validate(
        {
            "id": "int_ce",
            "model": MODEL_NAME,
            "status": "requires_action",
            "steps": [
                {
                    "type": "thought",
                    "signature": s1,
                    "summary": [{"type": "text", "text": "Compute first"}],
                },
                {
                    "type": "code_execution_call",
                    "id": "ce_1",
                    "arguments": {"code": "print(1)", "language": "python"},
                },
                {"type": "code_execution_result", "call_id": "ce_1", "result": "1\n"},
                {"type": "model_output", "content": [{"type": "text", "text": "Done"}]},
                {
                    "type": "function_call",
                    "id": "fc_after",
                    "name": "save",
                    "arguments": {},
                },
            ],
        }
    )
    ai_ce = _interaction_to_result(inter_ce).generations[0].message
    assert _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY not in ai_ce.additional_kwargs
    _, rep_ce = _messages_to_interaction_steps(
        [HumanMessage("q"), ai_ce, ToolMessage("ok", tool_call_id="fc_after")],
        effective_model=MODEL_NAME,
    )
    thought_steps = [s for s in rep_ce if s["type"] == "thought"]
    assert [t["signature"] for t in thought_steps] == [s1, _SKIP_THOUGHT_SIGNATURE_B64]


def test_stateful_foreign_ids_branching_trailing_system_and_convert_sys() -> None:
    """Foreign ID filtering, branching slice, trailing SystemMessage, & convert_sys."""
    stateful = _llm(use_previous_interaction_id=True)

    # 1) Foreign and unstamped stored AIMessages are never chained onto Gemini
    foreign_msgs: list[Any] = [
        HumanMessage("h1"),
        AIMessage(
            "from oai",
            response_metadata={
                "model_provider": "openai",
                "stored": True,
                "id": "resp_openai_1",
            },
        ),
        AIMessage("unstamped", response_metadata={"stored": True, "id": "resp_unk_2"}),
        HumanMessage("h2"),
    ]
    p_for = _prepare_interactions_request(stateful, foreign_msgs)
    assert "previous_interaction_id" not in p_for.create_kwargs
    assert len(p_for.create_kwargs["input"]) == 4

    # 2) Explicit previous_interaction_id="int_1" branches without duplicating Turn 1
    ai1 = AIMessage(
        "r1",
        response_metadata={
            "model_provider": "google_genai",
            "stored": True,
            "interaction_id": "int_1",
        },
    )
    ai2 = AIMessage(
        "r2",
        response_metadata={
            "model_provider": "google_genai",
            "stored": True,
            "interaction_id": "int_2",
        },
    )
    branch_msgs: list[Any] = [
        HumanMessage("turn 1"),
        ai1,
        HumanMessage("turn 2"),
        ai2,
        HumanMessage("turn 3"),
    ]
    p_branch = _prepare_interactions_request(
        stateful, branch_msgs, previous_interaction_id="int_1"
    )
    assert p_branch.create_kwargs["previous_interaction_id"] == "int_1"
    assert [s["content"][0]["text"] for s in p_branch.create_kwargs["input"]] == [
        "turn 2",
        "r2",
        "turn 3",
    ]

    # 3) Explicit external previous_interaction_id with trailing stored AIMessage never
    # drops intermediate messages via the `earlier_id` fallback!
    p_ext = _prepare_interactions_request(
        _llm(model="gemini-2.5-flash", use_previous_interaction_id=True),
        [HumanMessage("t1"), ai1, HumanMessage("t2"), ai2],
        previous_interaction_id="external_root_id",
    )
    assert p_ext.create_kwargs["previous_interaction_id"] == "external_root_id"
    assert len(p_ext.create_kwargs["input"]) == 4

    # 4) Trailing SystemMessage chains onto `int_1` while preserving initial system text
    p_tsys = _prepare_interactions_request(
        stateful,
        [
            SystemMessage("Initial persona."),
            HumanMessage("hello"),
            ai1,
            SystemMessage("Now reply in 3 words."),
        ],
    )
    assert p_tsys.create_kwargs["previous_interaction_id"] == "int_1"
    assert p_tsys.create_kwargs["system_instruction"] == "Initial persona."
    assert p_tsys.create_kwargs["input"] == [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "Now reply in 3 words."}],
        }
    ]

    # 5) convert_system_message_to_human=True does NOT re-prepend Turn 1's SystemMessage
    conv_llm = _llm(
        use_previous_interaction_id=True, convert_system_message_to_human=True
    )
    with pytest.warns(DeprecationWarning):
        p_conv = _prepare_interactions_request(
            conv_llm,
            [
                SystemMessage("Turn 1 rule."),
                HumanMessage("turn 1"),
                ai1,
                HumanMessage("turn 2"),
            ],
        )
    assert p_conv.create_kwargs["previous_interaction_id"] == "int_1"
    assert "system_instruction" not in p_conv.create_kwargs
    assert p_conv.create_kwargs["input"] == [
        {"type": "user_input", "content": [{"type": "text", "text": "turn 2"}]}
    ]


async def test_unary_multi_audio_async_url_fetch_timeout_and_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-part unary audio merge, bounded URL fetch, non-blocking async, & guards."""
    import io
    import pathlib
    import wave

    # 1) Unary multi-item AudioContent concatenates PCM frames instead of overwriting
    pcm1 = b"\x01\x02" * 20
    pcm2 = b"\x03\x04" * 30
    inter_aud = Interaction.model_validate(
        {
            "id": "int_aud2",
            "model": "gemini-2.5-flash-preview-tts",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "audio",
                            "data": base64.b64encode(pcm1).decode(),
                            "mime_type": "audio/l16",
                            "sample_rate": 24000,
                            "channels": 1,
                        },
                        {
                            "type": "audio",
                            "data": base64.b64encode(pcm2).decode(),
                            "mime_type": "audio/l16",
                            "sample_rate": 24000,
                            "channels": 1,
                        },
                    ],
                }
            ],
        }
    )
    aud_msg = _interaction_to_result(inter_aud).generations[0].message
    with wave.open(io.BytesIO(aud_msg.additional_kwargs["audio"]), "rb") as wf:
        assert wf.readframes(wf.getnframes()) == pcm1 + pcm2

    # 2) Sync URL fetch WITHOUT LANGCHAIN_TESTS_USER_AGENT uses httpx2.Client with
    # timeout=60.0 and reads Content-Type header for extensionless URLs!
    monkeypatch.delenv("LANGCHAIN_TESTS_USER_AGENT", raising=False)
    seen_ext_calls: list[tuple[str, dict[str, Any], Any]] = []

    def _fake_ext_get(self_c: httpx2.Client, url: str, **kw: Any) -> httpx2.Response:
        seen_ext_calls.append((url, kw, self_c.timeout))
        return httpx2.Response(
            200,
            content=b"not-magic-bytes",
            headers={"Content-Type": "application/pdf; charset=binary"},
            request=httpx2.Request("GET", url),
        )

    with (
        patch("requests.get", side_effect=AssertionError("requests.get must not run")),
        patch.object(httpx2.Client, "get", _fake_ext_get),
    ):
        _, s_ext = _messages_to_interaction_steps(
            [HumanMessage([{"type": "file", "url": "https://cdn.example.com/blob"}])],
            effective_model=MODEL_NAME,
        )
    assert seen_ext_calls == [
        ("https://cdn.example.com/blob", {}, httpx2.Timeout(60.0))
    ]
    assert s_ext[0]["content"][0]["type"] == "document"
    assert s_ext[0]["content"][0]["mime_type"] == "application/pdf"

    # 3) Async `ainvoke` fetches external URL via `httpx2.AsyncClient` without EVER
    # touching blocking `requests.get`!
    llm = _llm()
    done_ia = Interaction.model_validate(
        {
            "id": "int_async_u",
            "model": MODEL_NAME,
            "status": "completed",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "seen"}]}
            ],
        }
    )
    async_ic = Mock()
    async_ic.create = AsyncMock(return_value=done_ia)

    async def _fake_async_get(self_client: Any, url: str, **kw: Any) -> httpx2.Response:
        return httpx2.Response(
            200,
            content=b"\x89PNG\r\n\x1a\n\x00",
            headers={"Content-Type": "image/png"},
            request=httpx2.Request("GET", url),
        )

    aio_c = llm.async_client
    with (
        patch(
            "requests.get",
            side_effect=AssertionError("Blocking requests.get called in async!"),
        ),
        patch.object(httpx2.AsyncClient, "get", _fake_async_get),
        patch.object(type(aio_c), "interactions", async_ic, create=True),
    ):
        a_out = await llm.ainvoke(
            [HumanMessage([{"type": "image", "url": "https://img.example.com/noext"}])]
        )
    assert a_out.text == "seen"
    sent_part = async_ic.create.call_args.kwargs["input"][0]["content"][0]
    assert sent_part["type"] == "image" and sent_part["mime_type"] == "image/png"

    # 4) Negative / zero per-call polling_interval raises ValueError immediately
    for bad_pi in (0, 0.0, -1.5):
        with pytest.raises(ValueError, match="polling_interval must be positive"):
            _prepare_interactions_request(
                llm, [HumanMessage("hi")], polling_interval=bad_pi
            )

    # 5) Scheme & PathLike defensive guards
    for bad_url in (
        "ftp://youtube.com/watch?v=abc",
        "file://generativelanguage.googleapis.com/v1beta/files/secret",
    ):
        with pytest.raises(ValueError):
            _messages_to_interaction_steps(
                [HumanMessage([{"type": "video", "url": bad_url}])],
                effective_model=MODEL_NAME,
            )
    bad_path_val: Any = pathlib.Path("/etc/passwd")
    with pytest.raises(ValueError, match="local files is no longer supported"):
        _messages_to_interaction_steps(
            [
                HumanMessage(
                    [
                        {
                            "type": "image",
                            "base64": bad_path_val,
                            "mime_type": "image/png",
                        }
                    ]
                )
            ],
            effective_model=MODEL_NAME,
        )


def test_structured_output_include_raw_continuation_and_unknown_kw_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`with_structured_output(..., include_raw=True)` populates `raw.parsed`."""

    class AnswerCard(BaseModel):
        answer: str
        confidence: int

    llm = _llm()
    done = Interaction.model_validate(
        {
            "id": "int_so",
            "model": MODEL_NAME,
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": '{"answer": "Paris", "confidence": 99}',
                        }
                    ],
                }
            ],
        }
    )
    mock_ia = Mock()
    mock_ia.create = Mock(return_value=done)
    structured = llm.with_structured_output(AnswerCard, include_raw=True)
    with patch.object(type(llm.client), "interactions", mock_ia, create=True):
        out = cast("dict[str, Any]", structured.invoke("Capital of France?"))
    assert out["parsing_error"] is None
    assert out["parsed"] == AnswerCard(answer="Paris", confidence=99)
    assert out["raw"].additional_kwargs["parsed"] == AnswerCard(
        answer="Paris", confidence=99
    )

    # continuation_token forwarding + background=True + automatic_continuation=False err
    p_ct = _prepare_interactions_request(
        llm, [HumanMessage("resume")], continuation_token="ct_abc"
    )
    wire_ct = p_ct.create_kwargs.get("continuation_token") or p_ct.create_kwargs.get(
        "extra_body", {}
    ).get("continuation_token")
    assert wire_ct == "ct_abc"
    with pytest.raises(GoogleInvalidRequestError, match="automatic_continuation=False"):
        _prepare_interactions_request(
            llm,
            [HumanMessage("bg")],
            background=True,
            automatic_continuation=False,
        )

    # Unrecognized kwarg emits a warning instead of failing silently
    with caplog.at_level("WARNING"):
        _prepare_interactions_request(llm, [HumanMessage("hi")], max_output_token=50)
    assert "max_output_token" in caplog.text


async def test_unary_http200_failed_status_retries_and_fails_fast_on_fatal() -> None:
    """Unary `invoke`/`ainvoke` retries transient HTTP-200 `status='failed'` runs."""
    from langchain_google_genai.chat_models import (
        GoogleAuthenticationError,
        GoogleModelNotFoundError,
        GooglePermissionDeniedError,
    )

    llm = _llm(max_retries=4)
    transient_empty = Interaction.model_validate(
        {"status": "failed", "id": None, "errors": None}
    )
    transient_429 = Interaction.model_validate(
        {
            "status": "failed",
            "errors": [{"code": "resource_exhausted", "message": "Quota spike"}],
        }
    )
    transient_408 = Interaction.model_validate(
        {
            "status": "failed",
            "errors": [{"code": "deadline_exceeded", "message": "Backend timeout"}],
        }
    )
    completed = Interaction.model_validate(
        {
            "id": "int_ok",
            "model": MODEL_NAME,
            "status": "completed",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "ok"}]}
            ],
        }
    )

    # Sync recovers after 3 transient HTTP-200 failures on attempt 4
    sync_ic = Mock(
        create=Mock(
            side_effect=[transient_empty, transient_429, transient_408, completed]
        )
    )
    with (
        patch("tenacity.nap.time.sleep"),
        patch.object(type(llm.client), "interactions", sync_ic, create=True),
    ):
        res_s = llm.invoke("hello")
    assert res_s.text == "ok"
    assert sync_ic.create.call_count == 4

    # Async recovers from Vertex `Interaction(status='failed', id=None, errors=None)`
    async_ic = Mock(
        create=AsyncMock(side_effect=[transient_empty, transient_empty, completed])
    )
    aio_c = llm.async_client
    with (
        patch("asyncio.sleep", new=AsyncMock()),
        patch.object(type(aio_c), "interactions", async_ic, create=True),
    ):
        res_a = await llm.ainvoke("hello")
    assert res_a.text == "ok"
    assert async_ic.create.call_count == 3

    # Persistent HTTP-200 `status='failed'` exhausts all `max_retries=4` attempts
    exhaust_ic = Mock(create=Mock(return_value=transient_empty))
    with (
        patch("tenacity.nap.time.sleep"),
        patch.object(type(llm.client), "interactions", exhaust_ic, create=True),
        pytest.raises(GoogleInteractionsAPIError),
    ):
        llm.invoke("boom")
    assert exhaust_ic.create.call_count == 4

    # Non-retryable 400/401/403/404/safety/context-overflow/cancelled fail on attempt 1
    fatal_cases: list[tuple[Interaction, type[Exception]]] = [
        (
            Interaction.model_validate(
                {
                    "status": "failed",
                    "errors": [{"code": "invalid_argument", "message": "bad field"}],
                }
            ),
            GoogleInvalidRequestError,
        ),
        (
            Interaction.model_validate(
                {
                    "status": "failed",
                    "errors": [{"code": "unauthenticated", "message": "bad key"}],
                }
            ),
            GoogleAuthenticationError,
        ),
        (
            Interaction.model_validate(
                {
                    "status": "failed",
                    "errors": [{"code": "permission_denied", "message": "denied"}],
                }
            ),
            GooglePermissionDeniedError,
        ),
        (
            Interaction.model_validate(
                {
                    "status": "failed",
                    "errors": [{"code": "not_found", "message": "missing"}],
                }
            ),
            GoogleModelNotFoundError,
        ),
        (
            Interaction.model_validate(
                {
                    "status": "failed",
                    "errors": [{"code": "safety", "message": "Blocked for safety"}],
                }
            ),
            GoogleInteractionsAPIError,
        ),
        (
            Interaction.model_validate(
                {
                    "status": "failed",
                    "errors": [
                        {
                            "code": "invalid_argument",
                            "message": "Input token count exceeds context window",
                        }
                    ],
                }
            ),
            GoogleInteractionsContextOverflowError,
        ),
        (
            Interaction.model_validate({"id": "c1", "status": "cancelled"}),
            GoogleInteractionsAPIError,
        ),
    ]
    for fatal_inter, expected_cls in fatal_cases:
        fatal_s = Mock(create=Mock(return_value=fatal_inter))
        with (
            patch("tenacity.nap.time.sleep"),
            patch.object(type(llm.client), "interactions", fatal_s, create=True),
            pytest.raises(expected_cls),
        ):
            llm.invoke("x")
        assert fatal_s.create.call_count == 1


async def test_inbound_media_url_collection_prevents_ssrf_and_slices_stateful() -> None:
    """Async prefetch avoids SSRF on AI/tool JSON, slices first, & uses client_args."""
    from langchain_google_genai._interactions_request import (
        _aprepare_interactions_request,
        _collect_external_fetch_urls,
    )

    # 1) Prompt-injected tool_call args, MCP results, search/Maps/URL results,
    # citations, Anthropic tool_result dict content, & domain JSON trigger 0 fetches!
    ai_tc = (
        _interaction_to_result(
            {
                "id": "i_tc",
                "status": "requires_action",
                "model": MODEL_NAME,
                "steps": [
                    {
                        "type": "function_call",
                        "id": "c1",
                        "name": "open_url",
                        "arguments": {
                            "url": "http://169.254.169.254/latest/meta-data/",
                            "uri": "https://internal.corp/secret",
                            "image_url": "https://internal.corp/img.png",
                        },
                    }
                ],
            },
            output_version="interactions/v1",
        )
        .generations[0]
        .message
    )
    ai_mcp_and_search = (
        _interaction_to_result(
            {
                "id": "i_srv",
                "status": "completed",
                "model": MODEL_NAME,
                "steps": [
                    {
                        "type": "mcp_server_tool_call",
                        "id": "m1",
                        "name": "fetch_ticket",
                        "server_name": "jira",
                        "arguments": {"uri": "https://jira.internal.corp/PROJ-1"},
                    },
                    {
                        "type": "mcp_server_tool_result",
                        "call_id": "m1",
                        "result": {"url": "https://jira.internal.corp/PROJ-1"},
                    },
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": "Ticket ready.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://jira.internal.corp/PROJ-1",
                                        "start_index": 0,
                                        "end_index": 6,
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
    anth_tr = HumanMessage(
        content=[
            {
                "type": "tool_result",
                "tool_use_id": "c1",
                "content": {"url": "https://private.internal/admin"},
            },
            {
                "type": "tool_result",
                "tool_use_id": "c2",
                "content": [
                    {
                        "id": "tx_1",
                        "type": "credit",
                        "url": "https://bank.internal/tx/1",
                    }
                ],
            },
        ]
    )
    domain_tm = ToolMessage(
        content=[
            {"type": "credit", "url": "https://bank.internal/tx/2"},
            {"type": "image", "sku": "IMG-1"},
        ],
        tool_call_id="c1",
    )
    assert (
        _collect_external_fetch_urls(
            [HumanMessage("q"), ai_tc, domain_tm, ai_mcp_and_search, anth_tr]
        )
        == []
    )

    # 2) Stateful Turn 2 slices BEFORE collecting URLs (Turn 1 URL is never fetched)
    stateful_llm = _llm(
        use_previous_interaction_id=True,
        client_args={"trust_env": False, "headers": {"X-Proxy-Auth": "tok"}},
    )
    t1_u = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": "https://expired-s3.example/t1.png"},
            }
        ]
    )
    t1_a = AIMessage(
        "seen",
        response_metadata={
            "model_provider": "google_genai",
            "interaction_id": "int_turn_1",
            "stored": True,
        },
    )
    with patch.object(
        httpx.AsyncClient,
        "get",
        side_effect=AssertionError("Expired Turn 1 URL must not be fetched!"),
    ):
        prep_t2 = await _aprepare_interactions_request(
            stateful_llm, [t1_u, t1_a, ai_mcp_and_search, HumanMessage("turn 2")]
        )
    assert prep_t2.previous_interaction_id == "int_turn_1"

    # 3) Active-turn multimodal URL honors `client_args` + `httpx_async_client` & blocks
    # non-HTTP redirect scheme downgrades
    seen_headers: list[dict[str, str]] = []

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        return httpx.Response(
            200,
            content=b"\x89PNG\r\n\x1a\n\x00",
            headers={"Content-Type": "image/png"},
            request=request,
        )

    transport = httpx.MockTransport(_mock_handler)
    async with httpx.AsyncClient(transport=transport) as custom_async:
        prep_media = await _aprepare_interactions_request(
            stateful_llm,
            [
                t1_u,
                t1_a,
                HumanMessage(
                    [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://cdn.example/t2.png",
                            },
                        }
                    ]
                ),
            ],
            http_options={"httpx_async_client": custom_async},
        )
        assert not custom_async.is_closed
    assert len(seen_headers) == 1
    assert prep_media.httpx_async_client_override is custom_async
    assert (
        prep_media.create_kwargs["input"][0]["content"][0]["mime_type"] == "image/png"
    )

    # Merged ephemeral AsyncClient inherits model `client_args` + per-req overrides
    from langchain_google_genai._interactions_request import (
        _build_async_prefetch_client_kwargs,
    )

    borrowed_c, merged_kw = _build_async_prefetch_client_kwargs(
        stateful_llm,
        {"async_client_args": {"verify": False}},
    )
    assert borrowed_c is None
    assert merged_kw["trust_env"] is False
    assert merged_kw["verify"] is False
    assert merged_kw["headers"] == {"X-Proxy-Auth": "tok"}
    assert merged_kw["follow_redirects"] is True


def test_server_tools_preserve_distinct_thought_signatures_and_segment_order() -> None:
    """Server-tool `thought(SIG_1)` & post-server `thought(SIG_2)` stay ordered."""
    s1 = base64.b64encode(b"sig-before-srv").decode("ascii")
    s2 = base64.b64encode(b"sig-before-fc").decode("ascii")
    ce_sig = base64.b64encode(b"ce-step-own-sig").decode("ascii")
    gs_sig = base64.b64encode(b"gs-step-own-sig").decode("ascii")

    # 1) `[thought(s1), code_exec(sig=ce_sig), code_res, thought(s2), fc1]` lossless!
    for ov in (None, "interactions/v1"):
        ai_ce = (
            _interaction_to_result(
                {
                    "id": "i_ce_ord",
                    "status": "requires_action",
                    "model": "gemini-3-flash-preview",
                    "steps": [
                        {"type": "thought", "signature": s1},
                        {
                            "type": "code_execution_call",
                            "id": "ce1",
                            "arguments": {"code": "1+1", "language": "python"},
                            "signature": ce_sig,
                        },
                        {
                            "type": "code_execution_result",
                            "call_id": "ce1",
                            "result": "2",
                        },
                        {"type": "thought", "signature": s2},
                        {
                            "type": "function_call",
                            "id": "fc1",
                            "name": "fn",
                            "arguments": {},
                        },
                    ],
                },
                output_version=ov,
            )
            .generations[0]
            .message
        )
        prep = _prepare_interactions_request(
            _llm(model="gemini-3-flash-preview"),
            [HumanMessage("q"), ai_ce, ToolMessage("r", tool_call_id="fc1")],
        )
        wire = _serialize_sdk_body(prep.create_kwargs)
        mid = wire["input"][1:-1]
        assert [(st["type"], st.get("signature")) for st in mid] == [
            ("thought", s1),
            ("code_execution_call", ce_sig),
            ("code_execution_result", None),
            ("thought", s2),
            ("function_call", None),
        ]

    # 2) `[thought(s1), mcp_call, mcp_result, thought(s2), fc1]` preserves s1 as thought
    # and never injects illegal `signature` onto MCPServerToolCallStepParam!
    for ov in (None, "interactions/v1"):
        ai_mcp = (
            _interaction_to_result(
                {
                    "id": "i_mcp_ord",
                    "status": "requires_action",
                    "model": "gemini-3-flash-preview",
                    "steps": [
                        {"type": "thought", "signature": s1},
                        {
                            "type": "mcp_server_tool_call",
                            "id": "m1",
                            "name": "lookup",
                            "server_name": "docs",
                            "arguments": {"q": "x"},
                        },
                        {
                            "type": "mcp_server_tool_result",
                            "call_id": "m1",
                            "name": "lookup",
                            "server_name": "docs",
                            "result": {"ok": True},
                        },
                        {"type": "thought", "signature": s2},
                        {
                            "type": "function_call",
                            "id": "fc1",
                            "name": "done",
                            "arguments": {},
                        },
                    ],
                },
                output_version=ov,
            )
            .generations[0]
            .message
        )
        prep_m = _prepare_interactions_request(
            _llm(model="gemini-3-flash-preview"),
            [HumanMessage("q"), ai_mcp, ToolMessage("r", tool_call_id="fc1")],
        )
        wire_m = _serialize_sdk_body(prep_m.create_kwargs)
        mid_m = wire_m["input"][1:-1]
        assert [(st["type"], st.get("signature")) for st in mid_m] == [
            ("thought", s1),
            ("mcp_server_tool_call", None),
            ("mcp_server_tool_result", None),
            ("thought", s2),
            ("function_call", None),
        ]

    # 3) Non-code server calls (`google_search`, `url_context`, `google_maps`,
    # `file_search`, `retrieval`) preserve summary-less `thought(s1)` + own signature;
    # and signed `[google_search_call(sig=gs_sig), result, thought(s2), fc1]` gets NO
    # spurious `skip_thought_signature_validator` thought injected at index 0!
    ai_gs = (
        _interaction_to_result(
            {
                "id": "i_gs_ord",
                "status": "requires_action",
                "model": "gemini-3-flash-preview",
                "steps": [
                    {
                        "type": "google_search_call",
                        "id": "gs1",
                        "arguments": {"queries": ["langchain"]},
                        "signature": gs_sig,
                    },
                    {"type": "google_search_result", "call_id": "gs1", "result": []},
                    {"type": "thought", "signature": s2},
                    {
                        "type": "function_call",
                        "id": "fc1",
                        "name": "fn",
                        "arguments": {},
                    },
                ],
            },
            output_version="interactions/v1",
        )
        .generations[0]
        .message
    )
    mid_gs = _prepare_interactions_request(
        _llm(model="gemini-3-flash-preview"),
        [HumanMessage("q"), ai_gs, ToolMessage("r", tool_call_id="fc1")],
    ).create_kwargs["input"][1:-1]
    assert [(st["type"], st.get("signature")) for st in mid_gs] == [
        ("google_search_call", gs_sig),
        ("google_search_result", None),
        ("thought", s2),
        ("function_call", None),
    ]

    for st_name, res_name, st_args in [
        ("url_context_call", "url_context_result", {"urls": ["https://a.example"]}),
        ("google_maps_call", "google_maps_result", {"queries": ["cafe"]}),
        ("file_search_call", "file_search_result", None),
        ("retrieval_call", "retrieval_result", {"queries": ["policy"]}),
    ]:
        call_step: dict[str, Any] = {"type": st_name, "id": "s1", "signature": gs_sig}
        if st_args is not None:
            call_step["arguments"] = st_args
        res_step: dict[str, Any] = {"type": res_name, "call_id": "s1"}
        if res_name in ("url_context_result", "google_maps_result"):
            res_step["result"] = []
        for ov_srv in (None, "interactions/v1"):
            ai_srv = (
                _interaction_to_result(
                    {
                        "id": f"i_{st_name}",
                        "status": "completed",
                        "model": "gemini-3-flash-preview",
                        "steps": [
                            {"type": "thought", "signature": s1},
                            call_step,
                            res_step,
                            {
                                "type": "model_output",
                                "content": [{"type": "text", "text": "Answer"}],
                            },
                        ],
                    },
                    output_version=ov_srv,
                )
                .generations[0]
                .message
            )
            assert ai_srv.text == "Answer"
            rep_srv = _prepare_interactions_request(
                _llm(model="gemini-3-flash-preview"),
                [HumanMessage("q"), ai_srv, HumanMessage("follow")],
            ).create_kwargs["input"][1:-1]
            assert rep_srv[0] == {"type": "thought", "signature": s1}
            if ov_srv == "interactions/v1":
                assert rep_srv[1]["type"] == st_name
                assert rep_srv[1]["signature"] == gs_sig


def test_tool_message_domain_json_dicts_and_unary_file_citations() -> None:
    """`ToolMessage` list of domain JSON dicts + unary `file_citation` preservation."""
    llm = _llm()
    tm = ToolMessage(
        content=[
            {"id": "tx_1", "type": "credit", "amount": 50.0},
            {"id": "tx_2", "type": "image", "sku": "IMG-1"},
            {"id": "d_1", "type": "document", "data": {"nested": True}},
            {"id": "m_1", "type": "text", "text": "note", "author": "alice"},
        ],
        tool_call_id="c1",
        name="list_tx",
    )
    prep = _prepare_interactions_request(
        llm,
        [
            HumanMessage("q"),
            AIMessage("", tool_calls=[{"name": "list_tx", "args": {}, "id": "c1"}]),
            tm,
        ],
    )
    wire = _serialize_sdk_body(prep.create_kwargs)
    fr_step = wire["input"][-1]
    assert fr_step["type"] == "function_result"
    parsed_items = [json.loads(sub["text"]) for sub in fr_step["result"]]
    assert parsed_items[0] == {"id": "tx_1", "type": "credit", "amount": 50.0}
    assert parsed_items[1] == {"id": "tx_2", "type": "image", "sku": "IMG-1"}
    assert parsed_items[2] == {
        "id": "d_1",
        "type": "document",
        "data": {"nested": True},
    }
    assert parsed_items[3]["author"] == "alice"

    # Unary `file_citation` kept in `v0` `.content_blocks`, `v1`, & `interactions/v1`
    fc_payload = {
        "id": "i_fcit",
        "status": "completed",
        "model": MODEL_NAME,
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello spec",
                        "annotations": [
                            {
                                "type": "file_citation",
                                "document_uri": "files/d1",
                                "file_name": "spec.pdf",
                                "page_number": 3,
                                "start_index": 0,
                                "end_index": 5,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    msg_v0 = _interaction_to_result(fc_payload).generations[0].message
    msg_v1 = (
        _interaction_to_result(fc_payload, output_version="v1").generations[0].message
    )
    msg_ia = (
        _interaction_to_result(fc_payload, output_version="interactions/v1")
        .generations[0]
        .message
    )
    for blocks in (msg_v0.content_blocks, msg_v1.content_blocks, msg_ia.content_blocks):
        anns = blocks[0].get("annotations")
        assert isinstance(anns, list) and len(anns) == 1
        assert anns[0]["url"] == "files/d1"
        assert anns[0]["title"] == "spec.pdf"
        assert anns[0]["cited_text"] == "Hello"
        assert anns[0]["extras"]["file_citation_metadata"]["page_number"] == 3


def test_deep_research_background_default_store_false_override_and_audio_rate() -> None:
    """Deep Research defaults `background=True`; `store=False` skips auto-slicing."""
    # 1) `agent="deep-research-*"` and `model="deep-research-*"` default background=True
    p_dr = _prepare_interactions_request(
        _llm(agent="deep-research-preview-04-2026"), [HumanMessage("topic")]
    )
    assert p_dr.background is True
    assert p_dr.create_kwargs["background"] is True
    assert p_dr.create_kwargs["store"] is True

    p_dr_m = _prepare_interactions_request(
        _llm(model="deep-research-preview-04-2026"), [HumanMessage("topic")]
    )
    assert p_dr_m.create_kwargs["agent"] == "deep-research-preview-04-2026"
    assert p_dr_m.create_kwargs["background"] is True

    # Explicit `background=False` is honored, and unary still polls `in_progress`
    llm_dr_fg = _llm(agent="deep-research-preview-04-2026", polling_interval=0.01)
    mock_dr = Mock(
        create=Mock(
            return_value=Interaction.model_validate(
                {"id": "dr_fg", "status": "in_progress"}
            )
        ),
        get=Mock(
            return_value=Interaction.model_validate(
                {
                    "id": "dr_fg",
                    "agent": "deep-research-preview-04-2026",
                    "status": "completed",
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": "Report done"}],
                        }
                    ],
                }
            )
        ),
    )
    with (
        patch("langchain_google_genai._interactions_response.time.sleep"),
        patch.object(type(llm_dr_fg.client), "interactions", mock_dr, create=True),
    ):
        dr_res = llm_dr_fg.invoke("topic", background=False)
    assert mock_dr.create.call_args.kwargs["background"] is False
    assert dr_res.text == "Report done"
    assert mock_dr.get.call_count == 1

    # Missing id on non-terminal unary response raises GoogleInteractionsAPIError
    mock_noid = Mock(
        create=Mock(
            return_value=Interaction.model_validate({"id": "", "status": "in_progress"})
        )
    )
    with (
        patch.object(type(llm_dr_fg.client), "interactions", mock_noid, create=True),
        pytest.raises(GoogleInteractionsAPIError, match="returned no id"),
    ):
        llm_dr_fg.invoke("topic")

    # 2) Per-call `store=False` on `use_previous_interaction_id=True` sends full history
    stateful = _llm(use_previous_interaction_id=True)
    ai1 = AIMessage(
        "hi",
        response_metadata={
            "model_provider": "google_genai",
            "interaction_id": "int_1",
            "stored": True,
        },
    )
    p_zdr = _prepare_interactions_request(
        stateful,
        [HumanMessage("h1"), ai1, HumanMessage("h2")],
        store=False,
    )
    assert p_zdr.create_kwargs["store"] is False
    assert "previous_interaction_id" not in p_zdr.create_kwargs
    assert len(p_zdr.create_kwargs["input"]) == 3

    # Explicit `store=False` + explicit `previous_interaction_id` STILL raises!
    with pytest.raises(GoogleInvalidRequestError, match="store=False"):
        _prepare_interactions_request(
            stateful,
            [HumanMessage("h1"), ai1, HumanMessage("h2")],
            store=False,
            previous_interaction_id="int_1",
        )

    # 3) Audio `rate` fallback when `sample_rate` is absent
    pcm_b64 = base64.b64encode(b"\x01\x02" * 20).decode("ascii")
    aud_msg = (
        _interaction_to_result(
            {
                "id": "i_rate",
                "status": "completed",
                "model": MODEL_NAME,
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "audio",
                                "data": pcm_b64,
                                "mime_type": "audio/pcm",
                                "rate": 16000,
                                "channels": 1,
                            }
                        ],
                    }
                ],
            },
            output_version="interactions/v1",
        )
        .generations[0]
        .message
    )
    assert aud_msg.content[0]["extras"]["sample_rate"] == 16000  # type: ignore[index]


async def test_ai_message_never_fetches_external_urls_and_schemes_ignore_case() -> None:
    """`AIMessage` replay never downloads external URLs; uppercase schemes prefetch."""
    from langchain_google_genai._image_utils import ImageBytesLoader
    from langchain_google_genai._interactions_request import (
        _aprepare_interactions_request,
        _collect_external_fetch_urls,
    )

    png_bytes = b"\x89PNG\r\n\x1a\n\x00"
    png_b64 = base64.b64encode(png_bytes).decode("ascii")

    ssrf_blocks: list[str | dict[Any, Any]] = [
        {"type": "text-plain", "url": "http://169.254.169.254/latest/meta-data/"},
        {"type": "image", "url": "http://internal.corp/chart.png"},
        {"type": "video", "url": "HTTPS://internal.corp/clip.mp4"},
        {"type": "audio", "url": "https://internal.corp/voice.wav"},
        {"type": "file", "url": "https://internal.corp/doc.pdf"},
        {"type": "document", "uri": "https://internal.corp/spec.pdf"},
        {"type": "image_url", "image_url": {"url": "https://internal.corp/iu.png"}},
        {"type": "image_url", "image_url": "HTTP://internal.corp/iu2.png"},
        {"type": "media", "file_uri": "https://internal.corp/m.png"},
        {"type": "file_data", "file_uri": "https://internal.corp/fd.pdf"},
        {
            "type": "file",
            "file": {
                "filename": "leak.txt",
                "file_data": "http://169.254.169.254/secret.txt",
            },
        },
        {"type": "file", "file": {"file_id": "https://internal.corp/fid.pdf"}},
        {
            "type": "image",
            "source": {"type": "url", "url": "https://internal.corp/anth.png"},
        },
        {
            "type": "document",
            "source": {"type": "url", "url": "https://internal.corp/anth.pdf"},
        },
        {
            "type": "non_standard",
            "value": {
                "type": "text-plain",
                "url": "http://169.254.169.254/wrapped-secret",
            },
        },
        # Dual block with external URL + inline base64 falls back to inline base64!
        {
            "type": "image",
            "url": "https://oaidalle.example/ephemeral.png",
            "base64": png_b64,
            "mime_type": "image/png",
        },
        # Uppercase DATA: URI decodes strictly in-memory without any HTTP request
        {
            "type": "image_url",
            "image_url": {"url": f"DATA:IMAGE/PNG;BASE64,{png_b64}"},
        },
    ]

    # 1) Every provider_kind ("foreign", "unknown", "native") triggers ZERO sync or
    # async HTTP fetches on AIMessage blocks and injects zero SSRF response bytes.
    llm = _llm()
    done_ia = Interaction.model_validate(
        {
            "id": "i_zero_ssrf",
            "model": MODEL_NAME,
            "status": "completed",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "safe"}]}
            ],
        }
    )
    sync_ic = Mock(create=Mock(return_value=done_ia))
    async_ic = Mock(create=AsyncMock(return_value=done_ia))

    for meta in (
        {"model_provider": "openai"},
        {},
        {"model_provider": "google_genai"},
    ):
        history = [
            HumanMessage("q1"),
            AIMessage(content=ssrf_blocks, response_metadata=meta),
            HumanMessage("q2"),
        ]
        with (
            patch(
                "requests.get",
                side_effect=AssertionError("requests.get called on AIMessage!"),
            ),
            patch.object(
                httpx.Client,
                "get",
                side_effect=AssertionError(
                    "Sync httpx.Client.get called on AIMessage!"
                ),
            ),
            patch.object(
                httpx.AsyncClient,
                "get",
                side_effect=AssertionError(
                    "Async httpx.AsyncClient.get called on AIMessage!"
                ),
            ),
            patch.object(
                ImageBytesLoader,
                "_bytes_from_url",
                side_effect=AssertionError("ImageBytesLoader fetched AIMessage URL!"),
            ),
            patch.object(type(llm.client), "interactions", sync_ic, create=True),
            patch.object(type(llm.async_client), "interactions", async_ic, create=True),
        ):
            llm.invoke(history)
            await llm.ainvoke(history)

        for sent in (
            sync_ic.create.call_args.kwargs["input"],
            async_ic.create.call_args.kwargs["input"],
        ):
            dumped = json.dumps(sent)
            assert "169.254.169.254" not in dumped
            assert "internal.corp" not in dumped
            # Inline base64 fallback + DATA: URI survived with 0 network calls
            model_step = sent[1]
            assert model_step["type"] == "model_output"
            assert len(model_step["content"]) == 2
            assert all(
                c == {"type": "image", "data": png_b64, "mime_type": "image/png"}
                for c in model_step["content"]
            )

    # 2) Native/unstamped AIMessage preserves provider URIs (including uppercase
    # schemes) without fetching, whereas foreign AIMessage drops them all cleanly.
    prov_blocks: list[str | dict[Any, Any]] = [
        {"type": "file", "file_id": "files/gem_123", "mime_type": "application/pdf"},
        {
            "type": "image",
            "url": "HTTPS://generativelanguage.googleapis.com/v1beta/files/ImgAbC",
            "mime_type": "image/png",
        },
        {"type": "video", "url": "HTTPS://WWW.YOUTUBE.COM/watch?v=dQw4w9WgXcQ"},
        {"type": "document", "uri": "GS://MyBucket/Folder/Report.PDF"},
    ]
    with (
        patch.object(
            httpx.Client, "get", side_effect=AssertionError("Must not fetch!")
        ),
        patch("requests.get", side_effect=AssertionError("Must not fetch!")),
    ):
        _, steps_nat = _messages_to_interaction_steps(
            [
                HumanMessage("u"),
                AIMessage(
                    content=prov_blocks,
                    response_metadata={"model_provider": "google_vertexai"},
                ),
                HumanMessage("u2"),
            ],
            effective_model=MODEL_NAME,
            use_vertexai=True,
        )
        _, steps_for = _messages_to_interaction_steps(
            [
                HumanMessage("u"),
                AIMessage(
                    content=prov_blocks,
                    response_metadata={"model_provider": "openai"},
                ),
                HumanMessage("u2"),
            ],
            effective_model=MODEL_NAME,
            use_vertexai=True,
        )
    assert [c["uri"] for c in steps_nat[1]["content"]] == [
        "files/gem_123",
        "https://generativelanguage.googleapis.com/v1beta/files/ImgAbC",
        "https://WWW.YOUTUBE.COM/watch?v=dQw4w9WgXcQ",
        "gs://MyBucket/Folder/Report.PDF",
    ]
    assert steps_for == [
        {"type": "user_input", "content": [{"type": "text", "text": "u"}]},
        {"type": "user_input", "content": [{"type": "text", "text": "u2"}]},
    ]

    # 3) Uppercase/mixed-case HTTP(S) schemes on HumanMessage / ToolMessage / OpenAI
    # file / Anthropic tool_result prefetch asynchronously and NEVER block asyncio!
    inbound_upper: list[Any] = [
        HumanMessage(
            content=[
                {"type": "image", "url": "HTTPS://cdn.example.com/Upper.PNG"},
                {
                    "type": "file",
                    "file": {
                        "filename": "note.txt",
                        "file_data": "HTTP://cdn.example.com/note.txt",
                    },
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tc_1",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": "  HTTPS://cdn.example.com/tr.png",
                        }
                    ],
                },
                {"type": "image", "url": "GS://skip-bucket/a.png"},
                {
                    "type": "image_url",
                    "image_url": f"DATA:image/png;base64,{png_b64}",
                },
                {
                    "type": "video",
                    "url": "HTTPS://youtu.be/dQw4w9WgXcQ",
                },
            ]
        ),
        AIMessage(
            content=[
                {
                    "type": "text-plain",
                    "url": "http://169.254.169.254/latest/meta-data/",
                }
            ],
            response_metadata={"model_provider": "openai"},
            tool_calls=[{"id": "tc_2", "name": "f", "args": {}}],
        ),
        ToolMessage(
            content=[{"type": "image", "url": "HttpS://cdn.example.com/tm.png"}],
            tool_call_id="tc_2",
        ),
    ]
    collected = _collect_external_fetch_urls(inbound_upper)
    assert collected == [
        "HTTPS://cdn.example.com/Upper.PNG",
        "HTTP://cdn.example.com/note.txt",
        "  HTTPS://cdn.example.com/tr.png",
        "HttpS://cdn.example.com/tm.png",
    ]

    async_fetched: list[str] = []

    def _upper_handler(req: httpx.Request) -> httpx.Response:
        async_fetched.append(str(req.url))
        if str(req.url).endswith(".txt"):
            return httpx.Response(
                200, headers={"content-type": "text/plain"}, content=b"hello txt"
            )
        return httpx.Response(
            200, headers={"content-type": "image/png"}, content=png_bytes
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_upper_handler)) as ac:
        with (
            patch(
                "requests.get",
                side_effect=AssertionError("Sync requests.get called!"),
            ),
            patch.object(
                httpx.Client,
                "get",
                side_effect=AssertionError("Sync httpx.Client.get called in async!"),
            ),
            pytest.warns(UserWarning, match="gs://"),
        ):
            aprep = await _aprepare_interactions_request(
                llm, inbound_upper, http_options={"httpx_async_client": ac}
            )
    assert sorted(async_fetched) == [
        "http://cdn.example.com/note.txt",
        "https://cdn.example.com/Upper.PNG",
        "https://cdn.example.com/tm.png",
        "https://cdn.example.com/tr.png",
    ]
    assert "169.254.169.254" not in json.dumps(aprep.create_kwargs["input"])


async def test_manual_stateful_continuation_all_shapes_and_incomplete_token(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Manual `continuation_token` omits `input` across shapes; keeps `incomplete`."""
    # 1) Unary `_interaction_to_result` preserves `continuation_token` on BOTH
    # `requires_continuation` and `incomplete`, and omits it on `completed`.
    for ov in (None, "v1", "interactions/v1"):
        for status, expect_tok in (
            ("requires_continuation", "tok_req"),
            ("incomplete", "tok_inc"),
            ("completed", None),
        ):
            res_msg = (
                _interaction_to_result(
                    {
                        "id": f"id_{status}",
                        "model": MODEL_NAME,
                        "status": status,
                        "continuation_token": expect_tok or "stale_tok",
                        "steps": [
                            {
                                "type": "model_output",
                                "content": [{"type": "text", "text": "partial"}],
                            }
                        ],
                    },
                    requested_store=True,
                    output_version=ov,
                )
                .generations[0]
                .message
            )
            assert res_msg.response_metadata.get("continuation_token") == expect_tok

    # 2) Test all manual continuation shapes through real SDK wire serialization!
    hop1_stored = AIMessage(
        "first half ",
        response_metadata={
            "model_provider": "google_genai",
            "interaction_id": "hop_1",
            "stored": True,
            "status": "requires_continuation",
            "continuation_token": "tok_1",
        },
    )
    llm_prev = _llm(use_previous_interaction_id=True)
    llm_store = _llm(store=True)
    llm_no_prefill = _llm(model="gemini-3.5-flash-lite", store=True)

    wire_payloads: list[dict[str, Any]] = []

    def _wire_handler(req: httpx.Request) -> httpx.Response:
        wire_payloads.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "hop_2",
                "model": MODEL_NAME,
                "status": "completed",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "second half"}],
                    }
                ],
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(_wire_handler)) as sync_hc,
    ):
        http_opt = {"httpx_client": sync_hc}

        # (a) Empty list `[]` + explicit `previous_interaction_id`
        r_a = llm_prev.invoke(
            [],
            previous_interaction_id="hop_1",
            continuation_token="tok_1",
            http_options=http_opt,
        )
        assert r_a.text == "second half"
        assert "input" not in wire_payloads[-1]
        assert wire_payloads[-1]["previous_interaction_id"] == "hop_1"
        assert wire_payloads[-1]["continuation_token"] == "tok_1"
        assert wire_payloads[-1]["store"] is True

        # (b) Empty string `""` emits ZERO empty-HumanMessage warnings & omits `input`
        before_warns = len(recwarn)
        llm_prev.invoke(
            "",
            previous_interaction_id="hop_1",
            continuation_token="tok_1",
            http_options=http_opt,
        )
        empty_hm_warns = [
            w
            for w in recwarn.list[before_warns:]
            if "HumanMessage with empty content" in str(w.message)
        ]
        assert empty_hm_warns == []
        assert "input" not in wire_payloads[-1]
        assert wire_payloads[-1]["previous_interaction_id"] == "hop_1"

        # (c) `[SystemMessage]` keeps `system_instruction` intact & omits `input`
        llm_prev.invoke(
            [SystemMessage("Be concise")],
            previous_interaction_id="hop_1",
            continuation_token="tok_1",
            http_options=http_opt,
        )
        assert "input" not in wire_payloads[-1]
        assert wire_payloads[-1]["system_instruction"] == "Be concise"
        assert wire_payloads[-1]["previous_interaction_id"] == "hop_1"

        # (d) `[SystemMessage, HumanMessage, hop1_ai]` + explicit previous ID
        # (including on no-prefill models & with unstamped partial AIMessage!)
        for candidate_ai in (hop1_stored, AIMessage("partial")):
            llm_no_prefill.invoke(
                [SystemMessage("Sys rule"), HumanMessage("Write essay"), candidate_ai],
                previous_interaction_id="hop_1",
                continuation_token="tok_1",
                max_output_tokens=4096,
                http_options=http_opt,
            )
            assert "input" not in wire_payloads[-1]
            assert wire_payloads[-1]["system_instruction"] == "Sys rule"
            assert wire_payloads[-1]["previous_interaction_id"] == "hop_1"
            assert wire_payloads[-1]["continuation_token"] == "tok_1"
            assert wire_payloads[-1]["generation_config"]["max_output_tokens"] == 4096

        # (e) `[HumanMessage, hop1_stored]` without explicit `previous_interaction_id`
        # works on both `use_previous_interaction_id=True` and `store=True`, plus 1-msg
        for runner in (llm_prev, llm_store):
            runner.invoke(
                [HumanMessage("Write essay"), hop1_stored],
                continuation_token="tok_1",
                http_options=http_opt,
            )
            assert "input" not in wire_payloads[-1]
            assert wire_payloads[-1]["previous_interaction_id"] == "hop_1"
            assert wire_payloads[-1]["continuation_token"] == "tok_1"
            assert wire_payloads[-1]["store"] is True

        llm_prev.invoke(
            [hop1_stored], continuation_token="tok_1", http_options=http_opt
        )
        assert "input" not in wire_payloads[-1]
        assert wire_payloads[-1]["previous_interaction_id"] == "hop_1"

    # (f) Async `ainvoke([HumanMessage(image), hop1_stored], continuation_token=...)`
    # slices Turn 1 cleanly without re-fetching Turn 1's external image URL!
    async with httpx.AsyncClient(transport=httpx.MockTransport(_wire_handler)) as ac:
        with patch.object(
            httpx.AsyncClient,
            "get",
            side_effect=AssertionError("Turn 1 URL must not be re-fetched on resume!"),
        ):
            a_res = await llm_prev.ainvoke(
                [
                    HumanMessage(
                        [{"type": "image", "url": "https://expired.example/t1.png"}]
                    ),
                    hop1_stored,
                ],
                continuation_token="tok_1",
                http_options={"httpx_async_client": ac},
            )
    assert a_res.text == "second half"
    assert "input" not in wire_payloads[-1]
    assert wire_payloads[-1]["previous_interaction_id"] == "hop_1"

    # Non-continuation empty input still raises GoogleInvalidRequestError!
    with pytest.raises(GoogleInvalidRequestError, match="empty input"):
        llm_prev.invoke([], previous_interaction_id="hop_1")
    with pytest.raises(GoogleInvalidRequestError, match="empty input"):
        llm_prev.invoke([], previous_interaction_id="hop_1", continuation_token="   ")


async def test_sync_and_async_httpx_media_fetch_parity_and_errors() -> None:
    """Sync & async media fetches honor `httpx` clients/args/timeouts & error types."""
    from langchain_google_genai._interactions_request import (
        _build_async_prefetch_client_kwargs,
        _build_sync_prefetch_client_kwargs,
    )
    from langchain_google_genai.chat_models import (
        GoogleConnectionError,
        GoogleTimeoutError,
    )

    png_bytes = b"\x89PNG\r\n\x1a\n\x00"
    img_msgs = [HumanMessage([{"type": "image", "url": "https://cdn.example/pic.png"}])]
    done_ia = Interaction.model_validate(
        {
            "id": "i_sync_h",
            "model": MODEL_NAME,
            "status": "completed",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "ok"}]}
            ],
        }
    )

    # 1) `http_options.httpx_client` is used on sync `invoke` (never `requests.get`),
    # follows safe HTTP redirects even when `follow_redirects=False` on client, and is
    # NOT closed after `invoke`!
    seen_sync_urls: list[str] = []

    def _sync_handler(req: httpx.Request) -> httpx.Response:
        seen_sync_urls.append(str(req.url))
        if req.url.path == "/pic.png":
            return httpx.Response(302, headers={"location": "/final.png"}, request=req)
        if req.url.path == "/final.png":
            return httpx.Response(
                200,
                headers={"content-type": "image/png"},
                content=png_bytes,
                request=req,
            )
        return httpx.Response(
            200,
            json=done_ia.model_dump(mode="json", exclude_none=True),
            request=req,
        )

    with (
        httpx.Client(transport=httpx.MockTransport(_sync_handler)) as custom_sync,
        patch("requests.get", side_effect=AssertionError("requests.get ignored!")),
    ):
        llm = _llm()
        out = llm.invoke(img_msgs, http_options={"httpx_client": custom_sync})
        assert out.text == "ok"
        assert not custom_sync.is_closed
    assert seen_sync_urls[:2] == [
        "https://cdn.example/pic.png",
        "https://cdn.example/final.png",
    ]

    # 2) `model.client_args` + `http_options.client_args` + request timeouts merge
    # identically across sync and async helpers, always keeping finite timeout.
    custom_sync_tr = httpx.MockTransport(_sync_handler)
    configured = _llm(
        timeout=12.5,
        client_args={
            "trust_env": False,
            "verify": False,
            "headers": {"X-Proxy": "corp"},
            "transport": custom_sync_tr,
        },
    )
    b_s, kw_s = _build_sync_prefetch_client_kwargs(
        configured, {"timeout": 4500, "client_args": {"headers": {"X-Req": "1"}}}
    )
    b_a, kw_a = _build_async_prefetch_client_kwargs(
        configured, {"timeout": 4500, "client_args": {"headers": {"X-Req": "1"}}}
    )
    assert b_s is None and b_a is None
    assert kw_s["trust_env"] is False and kw_a["trust_env"] is False
    assert kw_s["verify"] is False and kw_a["verify"] is False
    assert kw_s["transport"] is custom_sync_tr
    assert kw_a["transport"] is custom_sync_tr  # MockTransport is sync+async!

    class _SyncOnlyTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return _sync_handler(request)

    sync_only_opts = {"client_args": {"transport": _SyncOnlyTransport()}}
    _, kw_so_s = _build_sync_prefetch_client_kwargs(llm, sync_only_opts)
    _, kw_so_a = _build_async_prefetch_client_kwargs(llm, sync_only_opts)
    assert isinstance(kw_so_s["transport"], _SyncOnlyTransport)
    assert "transport" not in kw_so_a
    assert kw_s["timeout"] == 4.5 and kw_a["timeout"] == 4.5
    assert kw_s["headers"] == {"X-Req": "1"} and kw_a["headers"] == {"X-Req": "1"}

    # `client_args={"timeout": None}`, `httpx.Timeout(None)`, & `httpx2.Timeout(None)`
    # clamp to finite 60s
    llm_inf = _llm(client_args={"timeout": None})
    _, kw_clamped = _build_sync_prefetch_client_kwargs(llm_inf, None)
    assert kw_clamped["timeout"] == 60.0
    _, kw_hto = _build_sync_prefetch_client_kwargs(
        llm_inf, None, call_timeout=httpx.Timeout(None)
    )
    assert isinstance(kw_hto["timeout"], httpx.Timeout)
    assert kw_hto["timeout"].read == 60.0
    _, kw_hto2 = _build_sync_prefetch_client_kwargs(
        llm_inf, None, call_timeout=httpx2.Timeout(None)
    )
    assert isinstance(kw_hto2["timeout"], httpx2.Timeout)
    assert kw_hto2["timeout"].read == 60.0

    # Sync `invoke` honors `model.client_args` custom transport end-to-end!
    mock_ia = Mock(create=Mock(return_value=done_ia))
    with patch.object(type(configured.client), "interactions", mock_ia, create=True):
        res_cfg = configured.invoke(img_msgs, timeout=3.0)
    assert res_cfg.text == "ok"

    # 3) Timeout, connection, redirect-SSRF, and HTTP status error parity (sync & async)
    def _raise_timeout(req: httpx.Request) -> httpx.Response:
        msg = "media read timed out"
        raise httpx.ReadTimeout(msg, request=req)

    def _raise_conn(req: httpx.Request) -> httpx.Response:
        msg = "dns lookup failed"
        raise httpx.ConnectError(msg, request=req)

    def _redir_file(req: httpx.Request) -> httpx.Response:
        if req.url.scheme == "https":
            return httpx.Response(
                302, headers={"location": "file:///etc/passwd"}, request=req
            )
        return httpx.Response(200, content=b"secret", request=req)

    def _http_404(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"missing", request=req)

    llm_err = _llm()
    for handler_fn, exc_cls in (
        (_raise_timeout, GoogleTimeoutError),
        (_raise_conn, GoogleConnectionError),
        (_redir_file, ValueError),
        (_http_404, httpx.HTTPStatusError),
    ):
        with (
            httpx.Client(transport=httpx.MockTransport(handler_fn)) as sc,
            pytest.raises(exc_cls),
        ):
            llm_err.invoke(img_msgs, http_options={"httpx_client": sc})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_fn)) as ac:
            with pytest.raises(exc_cls):
                await llm_err.ainvoke(img_msgs, http_options={"httpx_async_client": ac})

    # Same error parity when callers pass `httpx2.Client` / `httpx2.AsyncClient`
    def _raise_v2_timeout(req: httpx2.Request) -> httpx2.Response:
        msg = "v2 media read timed out"
        raise httpx2.ReadTimeout(msg, request=req)

    def _raise_v2_conn(req: httpx2.Request) -> httpx2.Response:
        msg = "v2 dns lookup failed"
        raise httpx2.ConnectError(msg, request=req)

    for v2_handler, exc_cls in (
        (_raise_v2_timeout, GoogleTimeoutError),
        (_raise_v2_conn, GoogleConnectionError),
    ):
        with (
            httpx2.Client(transport=httpx2.MockTransport(v2_handler)) as sc2,
            pytest.raises(exc_cls),
        ):
            llm_err.invoke(img_msgs, http_options={"httpx_client": sc2})
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(v2_handler)
        ) as ac2:
            with pytest.raises(exc_cls):
                await llm_err.ainvoke(
                    img_msgs, http_options={"httpx_async_client": ac2}
                )


def test_mcp_server_allowed_tools_normalizes_through_real_sdk() -> None:
    """`mcp_server.allowed_tools` normalizes convenience shapes via real Speakeasy."""
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "",
                "status": "completed",
                "model": MODEL_NAME,
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "ok"}],
                    }
                ],
            },
        )

    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_KEY,
        output_version="interactions/v1",
        service_tier=ServiceTier.PRIORITY,
        client_args={"transport": httpx.MockTransport(handler)},
    )

    # 1) Exact published docs snippet: flat list[str] -> [{"tools": [...]}]
    caller_flat: dict[str, Any] = {
        "type": "mcp_server",
        "name": "deployment-tracker",
        "url": "https://mcp.example.com/mcp",
        "headers": {"Authorization": "Bearer your-token"},
        "allowed_tools": ["get_deployment_status", "list_releases"],
    }
    flat_snapshot = copy.deepcopy(caller_flat)
    resp = model.bind_tools([caller_flat]).invoke("Check deployment status.")
    assert resp.text == "ok"
    assert caller_flat == flat_snapshot
    assert captured[-1]["service_tier"] == "priority"
    assert captured[-1]["tools"] == [
        {
            "type": "mcp_server",
            "name": "deployment-tracker",
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer your-token"},
            "allowed_tools": [
                {"tools": ["get_deployment_status", "list_releases"]},
            ],
        }
    ]

    # 2) Single AllowedTools dict (including Enum mode) -> [dict], without mutation
    caller_single: dict[str, Any] = {
        "type": "mcp_server",
        "name": "ops",
        "url": "https://mcp.example.com/ops",
        "allowed_tools": {
            "mode": FunctionCallingConfigMode.ANY,
            "tools": ["restart_pod"],
        },
    }
    single_snapshot = copy.deepcopy(caller_single)
    assert model.invoke("Restart", tools=[caller_single]).text == "ok"
    assert caller_single == single_snapshot
    assert captured[-1]["tools"][0]["allowed_tools"] == [
        {"mode": "any", "tools": ["restart_pod"]}
    ]

    # 3) Valid list[dict], AllowedTools BaseModel, and empty [] pass through cleanly
    caller_list: dict[str, Any] = {
        "type": "mcp_server",
        "name": "multi",
        "url": "https://mcp.example.com/multi",
        "allowed_tools": [
            {"mode": "validated", "tools": ["read_doc"]},
            AllowedTools(mode="auto", tools=["search_docs"]),
        ],
    }
    assert model.bind_tools([caller_list]).invoke("Read").text == "ok"
    assert captured[-1]["tools"][0]["allowed_tools"] == [
        {"mode": "validated", "tools": ["read_doc"]},
        {"mode": "auto", "tools": ["search_docs"]},
    ]
    assert isinstance(caller_list["allowed_tools"][1], AllowedTools)

    assert (
        model.invoke(
            "Empty",
            tools=[
                {
                    "type": "mcp_server",
                    "name": "empty",
                    "url": "https://mcp.example.com/e",
                    "allowed_tools": [],
                }
            ],
        ).text
        == "ok"
    )
    assert captured[-1]["tools"][0]["allowed_tools"] == []

    # 4) Legacy {"mcp_servers": [...]} shorthand works via both bind_tools and invoke
    caller_legacy: dict[str, Any] = {
        "mcp_servers": [
            {
                "name": "short",
                "url": "https://mcp.example.com/short",
                "allowed_tools": ["ping"],
            },
            {
                "name": "short2",
                "url": "https://mcp.example.com/short2",
                "allowed_tools": {"mode": "auto", "tools": ["pong"]},
            },
        ]
    }
    legacy_snapshot = copy.deepcopy(caller_legacy)
    assert model.bind_tools([caller_legacy]).invoke("Ping").text == "ok"
    assert caller_legacy == legacy_snapshot
    assert captured[-1]["tools"] == [
        {
            "type": "mcp_server",
            "name": "short",
            "url": "https://mcp.example.com/short",
            "allowed_tools": [{"tools": ["ping"]}],
        },
        {
            "type": "mcp_server",
            "name": "short2",
            "url": "https://mcp.example.com/short2",
            "allowed_tools": [{"mode": "auto", "tools": ["pong"]}],
        },
    ]


def test_mcp_server_tool_result_list_outputs_replay_through_real_sdk() -> None:
    """Non-subcontent MCP list outputs JSON-serialize cleanly across output_versions."""
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "",
                "status": "completed",
                "model": MODEL_NAME,
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "done"}],
                    }
                ],
            },
        )

    img_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
    output_versions: tuple[Any, ...] = (None, "v0", "v1", "interactions/v1")

    for ov in output_versions:
        llm_kwargs: dict[str, Any] = {
            "model": MODEL_NAME,
            "api_key": FAKE_KEY,
            "use_interactions_api": True,
            "client_args": {"transport": httpx.MockTransport(handler)},
        }
        if ov is not None:
            llm_kwargs["output_version"] = ov
        llm = ChatGoogleGenerativeAI(**llm_kwargs)

        for raw_result, expect_serialized in (
            ([{"id": "dep_1", "active": True}], True),
            (["alpha", "beta"], True),
            ([], True),
            ([{"type": "audio", "data": img_b64}], True),
            ([{"type": "text"}], True),
            ([{"type": "image"}], True),
            ([{"type": "text", "text": "rich ok"}], False),
            (
                [
                    {"type": "text", "text": "chart:"},
                    {"type": "image", "data": img_b64, "mime_type": "image/png"},
                    {"type": "image", "uri": "https://example.com/chart.png"},
                ],
                False,
            ),
            ({"id": "dep_dict", "ok": True}, False),
        ):
            turn1_ai = (
                _interaction_to_result(
                    {
                        "id": "",
                        "status": "completed",
                        "model": MODEL_NAME,
                        "steps": [
                            {
                                "type": "mcp_server_tool_call",
                                "id": "m1",
                                "name": "list_deployments",
                                "server_name": "tracker",
                                "arguments": {},
                            },
                            {
                                "type": "mcp_server_tool_result",
                                "call_id": "m1",
                                "name": "list_deployments",
                                "server_name": "tracker",
                                "result": copy.deepcopy(raw_result),
                            },
                            {
                                "type": "model_output",
                                "content": [{"type": "text", "text": "1 active"}],
                            },
                        ],
                    },
                    model_name=MODEL_NAME,
                    output_version=ov,
                )
                .generations[0]
                .message
            )
            turn2 = llm.invoke(
                [HumanMessage("List deployments"), turn1_ai, HumanMessage("Failed?")]
            )
            assert turn2.text == "done"
            sent_mcp = next(
                s
                for s in captured[-1]["input"]
                if s["type"] == "mcp_server_tool_result"
            )
            if expect_serialized:
                assert sent_mcp["result"] == json.dumps(raw_result, ensure_ascii=False)
            elif isinstance(raw_result, dict):
                assert sent_mcp["result"] == raw_result
            else:
                assert isinstance(sent_mcp["result"], list)
                assert len(sent_mcp["result"]) == len(raw_result)

    # Raw mcp_server_tool_result step dict in AIMessage.content is also normalized
    llm_raw = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_KEY,
        use_interactions_api=True,
        client_args={"transport": httpx.MockTransport(handler)},
    )
    raw_ai = AIMessage(
        content=[
            {
                "type": "mcp_server_tool_call",
                "id": "r1",
                "name": "fetch_rows",
                "server_name": "db",
                "arguments": {},
            },
            {
                "type": "mcp_server_tool_result",
                "call_id": "r1",
                "name": "fetch_rows",
                "server_name": "db",
                "result": [{"row": 1}],
            },
            {"type": "text", "text": "1 row"},
        ],
        response_metadata={"model_provider": "google_genai"},
    )
    assert llm_raw.invoke([HumanMessage("go"), raw_ai, HumanMessage("next")]).text == (
        "done"
    )
    sent_raw = next(
        s for s in captured[-1]["input"] if s["type"] == "mcp_server_tool_result"
    )
    assert sent_raw["result"] == '[{"row": 1}]'


def test_server_tool_dicts_normalize_enums_without_caller_mutation() -> None:
    """Direct Interactions server tool dicts normalize enums identically to legacy."""
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "",
                "status": "completed",
                "model": MODEL_NAME,
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "ok"}],
                    }
                ],
            },
        )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_KEY,
        use_interactions_api=True,
        client_args={"transport": httpx.MockTransport(handler)},
    )

    caller_tools: list[dict[str, Any]] = [
        {
            "type": "computer_use",
            "environment": Environment.ENVIRONMENT_BROWSER,
            "disabled_safety_policies": [
                SafetyPolicy.SAFETY_POLICY_UNSPECIFIED,
                SafetyPolicy.FINANCIAL_TRANSACTIONS,
                "ACCOUNT_CREATION",
            ],
            "excludedPredefinedFunctions": ["drag_and_drop"],
            "enablePromptInjectionDetection": True,
        },
        {
            "type": "google_search",
            "search_types": ["WEB_SEARCH", "IMAGE_SEARCH"],
        },
        {
            "type": "retrieval",
            "retrieval_types": ["RAG_STORE", "VERTEX_AI_SEARCH"],
        },
    ]
    caller_snapshot = copy.deepcopy(caller_tools)

    assert llm.bind_tools(caller_tools).invoke("Open browser").text == "ok"
    assert caller_tools == caller_snapshot
    assert captured[-1]["tools"] == [
        {
            "type": "computer_use",
            "environment": "browser",
            "disabled_safety_policies": [
                "financial_transactions",
                "account_creation",
            ],
            "excluded_predefined_functions": ["drag_and_drop"],
            "enable_prompt_injection_detection": True,
        },
        {
            "type": "google_search",
            "search_types": ["web_search", "image_search"],
        },
        {
            "type": "retrieval",
            "retrieval_types": ["rag_store", "vertex_ai_search"],
        },
    ]

    # ENVIRONMENT_UNSPECIFIED is stripped rather than sent as "unspecified"
    llm.invoke(
        "Check",
        tools=[
            {
                "type": "computer_use",
                "environment": Environment.ENVIRONMENT_UNSPECIFIED,
            }
        ],
    )
    assert captured[-1]["tools"] == [{"type": "computer_use"}]


@pytest.mark.parametrize(
    ("case_id", "empty_ai"),
    [
        ("empty_str", AIMessage("")),
        ("empty_list", AIMessage([])),
        ("list_with_empty_str", AIMessage([""])),
        ("empty_text_block", AIMessage([{"type": "text", "text": ""}])),
        ("missing_text_key", AIMessage([{"type": "text"}])),
        (
            "empty_text_with_speech_ann",
            AIMessage(
                [
                    {
                        "type": "text",
                        "text": "",
                        "annotations": [{"type": "speech_metadata"}],
                    }
                ]
            ),
        ),
        ("empty_untyped_text_dict", AIMessage([{"text": ""}])),
        ("empty_text_plain_bare", AIMessage([{"type": "text-plain"}])),
        ("empty_text_plain_str", AIMessage([{"type": "text-plain", "text": ""}])),
        (
            "empty_data_uri_text_plain",
            AIMessage(
                [{"type": "file_data", "file_uri": "data:text/plain;base64,"}],
            ),
        ),
        (
            "empty_anthropic_doc_text",
            AIMessage(
                [
                    {
                        "type": "document",
                        "source": {"type": "text", "data": ""},
                    }
                ]
            ),
        ),
        (
            "signed_empty_unstamped",
            AIMessage(
                [{"type": "text", "text": "", "extras": {"signature": "..."}}],
            ),
        ),
        (
            "signed_empty_native_genai",
            AIMessage(
                [{"type": "text", "text": "", "extras": {"signature": "..."}}],
                response_metadata={"model_provider": "google_genai"},
            ),
        ),
        (
            "signed_empty_native_vertex",
            AIMessage(
                [
                    {
                        "type": "text",
                        "text": "",
                        "extras": {
                            "signature": base64.b64encode(b"stop-sig").decode("ascii")
                        },
                    }
                ],
                response_metadata={"model_provider": "google_vertexai"},
            ),
        ),
        (
            "signed_empty_explicit_thought_signature",
            AIMessage(
                [
                    {
                        "type": "text",
                        "text": "",
                        "thought_signature": base64.b64encode(b"ts").decode("ascii"),
                    }
                ]
            ),
        ),
    ],
)
def test_empty_and_signed_empty_ai_message_stateless_replay_wire(
    case_id: str, empty_ai: AIMessage
) -> None:
    """Empty and bare-signed-stop `AIMessage` turns never emit `{"text": ""}`."""
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": f"int_{case_id}",
                "status": "completed",
                "model": MODEL_NAME,
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "OK"}],
                    }
                ],
            },
        )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_KEY,
        use_interactions_api=True,
        client_args={"transport": httpx.MockTransport(handler)},
    )
    out = llm.invoke([HumanMessage("Hi"), empty_ai, HumanMessage("Reply OK")])
    assert out.text == "OK"

    wire_steps = captured_bodies[-1]["input"]
    assert wire_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]},
        {"type": "user_input", "content": [{"type": "text", "text": "Reply OK"}]},
    ]
    assert '"text": ""' not in json.dumps(captured_bodies[-1])


def test_empty_text_ai_preserves_tools_thoughts_media_and_server_steps() -> None:
    """`AIMessage` with empty text keeps valid steps without empty `model_output`."""
    sig_1 = base64.b64encode(b"gemini-sig-1").decode("ascii")
    sig_2 = base64.b64encode(b"gemini-sig-2").decode("ascii")
    png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
    wav_bytes = b"RIFF$\x00\x00\x00WAVEfmt "

    def _assert_no_empty_text_content(steps: list[dict[str, Any]]) -> None:
        for st in steps:
            if st["type"] in ("user_input", "model_output"):
                assert st["content"], f"Empty content list in step: {st}"
                for blk in st["content"]:
                    if blk.get("type") == "text":
                        assert blk.get("text") != "", f"Empty text block in {st}"
        _serialize_sdk_body({"model": MODEL_NAME, "input": steps, "store": False})

    # 1) Round-trip a real signature-only Interaction stop turn -> stateless replay
    bare_stop_ia = Interaction.model_validate(
        {
            "id": "int_deg_stop",
            "model": MODEL_NAME,
            "status": "completed",
            "steps": [{"type": "thought", "signature": sig_1}],
        }
    )
    bare_ai = _interaction_to_result(bare_stop_ia).generations[0].message
    assert bare_ai.content == [
        {"type": "text", "text": "", "extras": {"signature": sig_1}}
    ]
    _, s_round = _messages_to_interaction_steps(
        [HumanMessage("Hi"), bare_ai, HumanMessage("Reply OK")],
        effective_model=MODEL_NAME,
    )
    assert s_round == [
        {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]},
        {"type": "user_input", "content": [{"type": "text", "text": "Reply OK"}]},
    ]
    _assert_no_empty_text_content(s_round)

    # 2) Signed empty-text carrier alongside `tool_calls` promotes signature cleanly,
    # and post-tool empty AIMessage("") before next user turn is omitted cleanly.
    _, s_tc_sig = _messages_to_interaction_steps(
        [
            HumanMessage("Run check"),
            AIMessage(
                content=[{"type": "text", "text": "", "extras": {"signature": sig_1}}],
                tool_calls=[{"id": "tc_1", "name": "ping", "args": {"host": "a"}}],
                response_metadata={"model_provider": "google_genai"},
            ),
            ToolMessage(
                content=["", {"type": "text", "text": ""}, "pong"],
                tool_call_id="tc_1",
                name="ping",
            ),
            AIMessage(""),
            HumanMessage("Summarize"),
        ],
        effective_model=MODEL_NAME,
    )
    assert s_tc_sig == [
        {"type": "user_input", "content": [{"type": "text", "text": "Run check"}]},
        {"type": "thought", "signature": sig_1},
        {
            "type": "function_call",
            "id": "tc_1",
            "name": "ping",
            "arguments": {"host": "a"},
        },
        {
            "type": "function_result",
            "call_id": "tc_1",
            "name": "ping",
            "result": [{"type": "text", "text": "pong"}],
        },
        {"type": "user_input", "content": [{"type": "text", "text": "Summarize"}]},
    ]
    _assert_no_empty_text_content(s_tc_sig)

    # 3) If empty-text carrier has sig_1 but tool_call has its own distinct sig_2,
    # unattached sig_1 is discarded rather than emitting duplicate back-to-back thoughts
    _, s_tc_distinct = _messages_to_interaction_steps(
        [
            HumanMessage("Run"),
            AIMessage(
                content=[{"type": "text", "text": "", "extras": {"signature": sig_1}}],
                tool_calls=[{"id": "tc_d", "name": "ping", "args": {}}],
                additional_kwargs={
                    _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY: {"tc_d": sig_2}
                },
                response_metadata={"model_provider": "google_genai"},
            ),
            ToolMessage([{"type": "text", "text": ""}], tool_call_id="tc_d"),
        ],
        effective_model=MODEL_NAME,
    )
    assert [s["type"] for s in s_tc_distinct] == [
        "user_input",
        "thought",
        "function_call",
        "function_result",
    ]
    assert s_tc_distinct[1] == {"type": "thought", "signature": sig_2}
    assert s_tc_distinct[3]["result"] == ""
    _assert_no_empty_text_content(s_tc_distinct)

    # 4) Unsigned empty text (`""` / `[]` / `[{"text": ""}]`) + active-loop tool_calls
    empty_variants: tuple[str | list[str | dict[Any, Any]], ...] = (
        "",
        [],
        [{"type": "text", "text": ""}],
    )
    for empty_c in empty_variants:
        _, s_active = _messages_to_interaction_steps(
            [
                HumanMessage("Go"),
                AIMessage(
                    content=empty_c,
                    tool_calls=[{"id": "c_u", "name": "act", "args": {}}],
                ),
                ToolMessage("done", tool_call_id="c_u"),
            ],
            effective_model=MODEL_NAME,
        )
        assert [s["type"] for s in s_active] == [
            "user_input",
            "thought",
            "function_call",
            "function_result",
        ]
        assert s_active[1]["signature"] == _SKIP_THOUGHT_SIGNATURE_B64
        _assert_no_empty_text_content(s_active)

    # 5) Trailing signed empty-text chunk after real text (`generateContent` streaming)
    _, s_stream_tail = _messages_to_interaction_steps(
        [
            HumanMessage("Greet"),
            AIMessage(
                content=[
                    {"type": "text", "text": "Hello!"},
                    {"type": "text", "text": "", "extras": {"signature": sig_1}},
                ],
                response_metadata={"model_provider": "google_genai"},
            ),
            HumanMessage("Thanks"),
        ],
        effective_model=MODEL_NAME,
    )
    assert s_stream_tail[1:3] == [
        {"type": "thought", "signature": sig_1},
        {"type": "model_output", "content": [{"type": "text", "text": "Hello!"}]},
    ]
    _assert_no_empty_text_content(s_stream_tail)

    # 6) Explicit signed thinking block + empty text emits ONLY thought step
    _, s_thought = _messages_to_interaction_steps(
        [
            HumanMessage("Think"),
            AIMessage(
                content=[
                    {
                        "type": "thinking",
                        "thinking": "Evaluating safety constraints",
                        "signature": sig_1,
                    },
                    {"type": "text", "text": ""},
                ],
                response_metadata={"model_provider": "google_genai"},
            ),
            HumanMessage("Continue"),
        ],
        effective_model=MODEL_NAME,
    )
    assert s_thought == [
        {"type": "user_input", "content": [{"type": "text", "text": "Think"}]},
        {
            "type": "thought",
            "signature": sig_1,
            "summary": [{"type": "text", "text": "Evaluating safety constraints"}],
        },
        {"type": "user_input", "content": [{"type": "text", "text": "Continue"}]},
    ]
    _assert_no_empty_text_content(s_thought)

    # 7) Empty text + inline image AND v0 `additional_kwargs["audio"]` preserve media
    _, s_media = _messages_to_interaction_steps(
        [
            HumanMessage("Draw and speak"),
            AIMessage(
                content=[
                    {"type": "text", "text": "", "extras": {"signature": sig_1}},
                    {"type": "image", "base64": png_b64, "mime_type": "image/png"},
                ],
                response_metadata={"model_provider": "google_genai"},
            ),
            AIMessage(
                content=[{"type": "text", "text": "", "extras": {"signature": sig_2}}],
                additional_kwargs={"audio": wav_bytes},
                response_metadata={"model_provider": "google_genai"},
            ),
            HumanMessage("Nice"),
        ],
        effective_model=MODEL_NAME,
    )
    assert [s["type"] for s in s_media] == [
        "user_input",
        "thought",
        "model_output",
        "thought",
        "model_output",
        "user_input",
    ]
    assert s_media[2]["content"] == [
        {"type": "image", "data": png_b64, "mime_type": "image/png"}
    ]
    assert s_media[4]["content"] == [
        {
            "type": "audio",
            "data": base64.b64encode(wav_bytes).decode("ascii"),
            "mime_type": "audio/wav",
        }
    ]
    _assert_no_empty_text_content(s_media)

    # 8) Empty text + server tool blocks (including MCP empty-text subcontent)
    _, s_srv = _messages_to_interaction_steps(
        [
            HumanMessage("Calc"),
            AIMessage(
                content=[
                    {"type": "text", "text": ""},
                    {"type": "executable_code", "id": "ce_x", "executable_code": "1+1"},
                    {
                        "type": "code_execution_result",
                        "tool_call_id": "ce_x",
                        "code_execution_result": "2",
                    },
                    {
                        "type": "mcp_server_tool_call",
                        "id": "m_1",
                        "name": "q",
                        "server_name": "srv",
                        "arguments": {},
                    },
                    {
                        "type": "mcp_server_tool_result",
                        "call_id": "m_1",
                        "name": "q",
                        "server_name": "srv",
                        "result": [{"type": "text", "text": ""}],
                    },
                ],
                response_metadata={"model_provider": "google_genai"},
            ),
            HumanMessage("Next"),
        ],
        effective_model=MODEL_NAME,
    )
    assert [s["type"] for s in s_srv] == [
        "user_input",
        "code_execution_call",
        "code_execution_result",
        "mcp_server_tool_call",
        "mcp_server_tool_result",
        "user_input",
    ]
    assert s_srv[4]["result"] == ""
    _assert_no_empty_text_content(s_srv)


def test_empty_message_boundaries_human_blocks_whitespace_and_stateful_delta() -> None:
    """Leading/trailing/all-empty `AIMessage`, empty `HumanMessage` blocks, & `" "`."""
    # Leading + consecutive + trailing empty AIMessage around a valid HumanMessage
    _, s_bounds = _messages_to_interaction_steps(
        [
            AIMessage(""),
            HumanMessage(["", {"type": "text", "text": ""}, "Hello"]),
            AIMessage([]),
            AIMessage([{"type": "text", "text": ""}]),
            AIMessage(" "),  # Non-empty whitespace IS preserved untouched
            HumanMessage("Reply OK"),
            AIMessage(""),
        ],
        effective_model="gemini-3.6-flash",
    )
    assert s_bounds == [
        {"type": "user_input", "content": [{"type": "text", "text": "Hello"}]},
        {"type": "model_output", "content": [{"type": "text", "text": " "}]},
        {"type": "user_input", "content": [{"type": "text", "text": "Reply OK"}]},
    ]

    # Empty HumanMessage([{"type": "text", "text": ""}]) warns & does NOT shift
    # active_loop_start away from a preceding Gemini 3 unsigned tool call
    with pytest.warns(UserWarning, match="HumanMessage with empty content"):
        _, s_loop = _messages_to_interaction_steps(
            [
                HumanMessage("Run tool"),
                AIMessage("", tool_calls=[{"id": "t_1", "name": "fn", "args": {}}]),
                ToolMessage("ok", tool_call_id="t_1"),
                HumanMessage([{"type": "text", "text": ""}]),
            ],
            effective_model=MODEL_NAME,
        )
    assert [s["type"] for s in s_loop] == [
        "user_input",
        "thought",
        "function_call",
        "function_result",
    ]

    # [SystemMessage, AIMessage("")] promotes SystemMessage into valid user_input
    sys_none, s_sys_only = _messages_to_interaction_steps(
        [SystemMessage("Act concisely"), AIMessage("")],
        effective_model=MODEL_NAME,
    )
    assert sys_none is None
    assert s_sys_only == [
        {"type": "user_input", "content": [{"type": "text", "text": "Act concisely"}]}
    ]

    # Stateful delta [SystemMessage("Delta"), AIMessage("")] promotes only "Delta"
    prior = [
        SystemMessage("Base sys"),
        HumanMessage("Turn 1"),
        AIMessage(
            "Done",
            response_metadata={
                "model_provider": "google_genai",
                "id": "int_p1",
                "stored": True,
            },
        ),
    ]
    sys_keep, s_delta = _messages_to_interaction_steps(
        [SystemMessage("Delta instruction"), AIMessage("")],
        effective_model=MODEL_NAME,
        prior_messages=prior,
    )
    assert sys_keep == "Base sys"
    assert s_delta == [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "Delta instruction"}],
        }
    ]

    # Completely empty history or stateful delta [AIMessage("")] raises clean error
    for empty_only, priors in (
        ([AIMessage("")], None),
        ([AIMessage([]), AIMessage([{"type": "text", "text": ""}])], None),
        ([AIMessage("")], prior),
    ):
        with pytest.raises(GoogleInvalidRequestError, match="empty input"):
            _messages_to_interaction_steps(
                empty_only,
                effective_model=MODEL_NAME,
                prior_messages=priors,
            )


def test_interactions_xhigh_thinking_level_supported_and_fallback() -> None:
    """`xhigh` thinking level passes through when supported and falls back otherwise."""
    # Supported model (`gemini-4-hydrogen`) sends `xhigh` without warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prep_h = _prepare_interactions_request(
            _llm(model="gemini-4-hydrogen", thinking_level="xhigh"),
            [HumanMessage("Solve this")],
        )
    assert prep_h.create_kwargs["generation_config"]["thinking_level"] == "xhigh"

    # Unsupported models warn and fall back to `high`
    for unsupported_model in (
        "gemini-4-carbon",
        "gemini-3.5-flash",
        "models/gemini-4-carbon",
        "gemini-x",
    ):
        with pytest.warns(UserWarning, match="using 'high' instead"):
            prep_fb = _prepare_interactions_request(
                _llm(model=unsupported_model, thinking_level="xhigh"),
                [HumanMessage("Solve this")],
            )
        assert prep_fb.create_kwargs["generation_config"]["thinking_level"] == "high"

    # Case-insensitive and ThinkingLevel enum fallback via call-time kwarg
    carbon_llm = _llm(model="gemini-4-carbon")
    for raw_level in ("XHIGH", "xHigh", ThinkingLevel.XHIGH):
        with pytest.warns(UserWarning, match="using 'high' instead"):
            prep_ci = _prepare_interactions_request(
                carbon_llm, [HumanMessage("hi")], thinking_level=raw_level
            )
        assert prep_ci.create_kwargs["generation_config"]["thinking_level"] == "high"

    # Call-time `reasoning_effort="xhigh"` respects model profile
    hydrogen_llm = _llm(model="gemini-4-hydrogen")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prep_call_h = _prepare_interactions_request(
            hydrogen_llm, [HumanMessage("hi")], reasoning_effort="xhigh"
        )
    assert prep_call_h.create_kwargs["generation_config"]["thinking_level"] == "xhigh"

    # Explicit profile override enables `xhigh` on unknown models
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prep_override = _prepare_interactions_request(
            _llm(
                model="gemini-x",
                thinking_level="xhigh",
                profile={"reasoning_effort_levels": ["low", "high", "xhigh"]},
            ),
            [HumanMessage("hi")],
        )
    assert prep_override.create_kwargs["generation_config"]["thinking_level"] == "xhigh"

    # Explicit empty profile clears support even on `gemini-4-hydrogen`
    with pytest.warns(UserWarning, match="using 'high' instead"):
        prep_cleared = _prepare_interactions_request(
            _llm(model="gemini-4-hydrogen", thinking_level="xhigh", profile={}),
            [HumanMessage("hi")],
        )
    assert prep_cleared.create_kwargs["generation_config"]["thinking_level"] == "high"

    # Per-call model override looks up the override model's default profile
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prep_model_ovr = _prepare_interactions_request(
            carbon_llm,
            [HumanMessage("hi")],
            model="gemini-4-hydrogen",
            thinking_level="xhigh",
        )
    assert (
        prep_model_ovr.create_kwargs["generation_config"]["thinking_level"] == "xhigh"
    )


def test_interactions_usage_drops_cache_read_exceeding_prompt_tokens() -> None:
    """Drop `cache_read` to 0 when `total_cached_tokens > total_input_tokens`."""
    bogus_usage = _parse_usage_metadata(
        {
            "total_input_tokens": 50,
            "total_output_tokens": 20,
            "total_cached_tokens": 120,
            "total_tokens": 70,
        }
    )
    assert bogus_usage is not None
    assert bogus_usage["input_tokens"] == 50
    assert bogus_usage["output_tokens"] == 20
    assert bogus_usage["total_tokens"] == 70
    assert bogus_usage["input_token_details"] == {"cache_read": 0}

    # Compared against prompt tokens (`total_input_tokens`), not prompt + tool_use
    bogus_with_tools = _parse_usage_metadata(
        {
            "total_input_tokens": 40,
            "total_tool_use_tokens": 100,
            "total_output_tokens": 15,
            "total_cached_tokens": 80,
            "total_tokens": 155,
        }
    )
    assert bogus_with_tools is not None
    assert bogus_with_tools["input_tokens"] == 140
    assert bogus_with_tools["input_token_details"] == {"cache_read": 0}

    # Valid cache_read (including equal to prompt tokens) is preserved
    exact_usage = _parse_usage_metadata(
        {
            "total_input_tokens": 50,
            "total_output_tokens": 10,
            "total_cached_tokens": 50,
            "total_tokens": 60,
        }
    )
    assert exact_usage is not None
    assert exact_usage["input_token_details"] == {"cache_read": 50}


def test_interactions_unary_normalized_finish_reason() -> None:
    """Populate `normalized_finish_reason` across statuses, tool calls, and v1."""
    # Completed text -> STOP / stop
    res_stop = _interaction_to_result(
        {
            "id": "int_nfr_stop",
            "status": "completed",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "ok"}]}
            ],
        },
        model_name=MODEL_NAME,
    )
    gen_stop = res_stop.generations[0]
    assert gen_stop.generation_info is not None
    assert gen_stop.generation_info["finish_reason"] == "STOP"
    assert gen_stop.generation_info["normalized_finish_reason"] == "stop"
    assert gen_stop.message.response_metadata["normalized_finish_reason"] == "stop"

    # Completed with tool call -> STOP / tool_call (including output_version='v1')
    for out_ver in (None, "v1"):
        res_tc = _interaction_to_result(
            {
                "id": "int_nfr_tc",
                "status": "completed",
                "steps": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    }
                ],
            },
            model_name=MODEL_NAME,
            output_version=out_ver,
        )
        gen_tc = res_tc.generations[0]
        assert gen_tc.generation_info is not None
        assert gen_tc.generation_info["finish_reason"] == "STOP"
        assert gen_tc.generation_info["normalized_finish_reason"] == "tool_call"
        assert (
            gen_tc.message.response_metadata["normalized_finish_reason"] == "tool_call"
        )

    # Completed with invalid tool call -> STOP / tool_call
    res_itc = _interaction_to_result(
        {
            "id": "int_nfr_itc",
            "status": "completed",
            "steps": [
                {
                    "type": "function_call",
                    "id": "fc_bad",
                    "name": "broken_tool",
                    "arguments": "{not-valid-json",
                }
            ],
        },
        model_name=MODEL_NAME,
    )
    gen_itc = res_itc.generations[0]
    assert gen_itc.generation_info is not None
    assert gen_itc.generation_info["normalized_finish_reason"] == "tool_call"
    assert gen_itc.message.response_metadata["normalized_finish_reason"] == "tool_call"

    # Truncated statuses -> MAX_TOKENS / length
    for trunc_status in ("incomplete", "budget_exceeded", "requires_continuation"):
        res_len = _interaction_to_result(
            {
                "id": f"int_nfr_{trunc_status}",
                "status": trunc_status,
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "partial"}],
                    }
                ],
            },
            model_name=MODEL_NAME,
        )
        gen_len = res_len.generations[0]
        assert gen_len.generation_info is not None
        assert gen_len.generation_info["finish_reason"] == "MAX_TOKENS"
        assert gen_len.generation_info["normalized_finish_reason"] == "length"
        assert gen_len.message.response_metadata["normalized_finish_reason"] == "length"

    # Non-terminal poll=False status -> IN_PROGRESS / other
    res_prog = _interaction_to_result(
        {"id": "int_nfr_prog", "status": "in_progress", "steps": []},
        model_name=MODEL_NAME,
    )
    gen_prog = res_prog.generations[0]
    assert gen_prog.generation_info is not None
    assert gen_prog.generation_info["finish_reason"] == "IN_PROGRESS"
    assert gen_prog.generation_info["normalized_finish_reason"] == "other"
    assert gen_prog.message.response_metadata["normalized_finish_reason"] == "other"

    # End-to-end invoke() exposes normalized_finish_reason in response_metadata
    llm = _llm()
    sync_ia = Mock()
    sync_ia.create = Mock(
        return_value={
            "id": "int_nfr_invoke",
            "status": "incomplete",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "cut"}]}
            ],
        }
    )
    with patch.object(type(llm.client), "interactions", sync_ia, create=True):
        msg = llm.invoke("hi")
    assert msg.response_metadata["finish_reason"] == "MAX_TOKENS"
    assert msg.response_metadata["normalized_finish_reason"] == "length"
