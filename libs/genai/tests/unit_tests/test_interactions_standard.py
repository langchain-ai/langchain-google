import base64
import json
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_NAME = "gemini-3.8-flash"

FAKE_API_KEY = "fake-api-key"


class TestGeminiInteractionsStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": MODEL_NAME,
            "google_api_key": FAKE_API_KEY,
            "use_interactions_api": True,
        }

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {"GOOGLE_API_KEY": "api_key"},
            {"model": MODEL_NAME, "use_interactions_api": True},
            {"google_api_key": "api_key", "use_interactions_api": True},
        )

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        return True

    @property
    def supports_pdf_tool_message(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True


def test_stream_replay_omits_empty_and_bare_signed_ai_messages() -> None:
    """SSE `stream()` replay never emits empty `{"type": "text", "text": ""}`."""
    captured: list[dict[str, Any]] = []
    sig_b64 = base64.b64encode(b"bare-stop-sig").decode("ascii")
    sse_events = [
        {
            "event_type": "interaction.start",
            "interaction": {
                "id": "i_s",
                "model": MODEL_NAME,
                "status": "in_progress",
            },
        },
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "text", "text": "OK"},
        },
        {"event_type": "step.complete", "index": 0},
        {
            "event_type": "interaction.completed",
            "interaction": {
                "id": "i_s",
                "model": MODEL_NAME,
                "status": "completed",
            },
        },
    ]
    sse_bytes = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in sse_events)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_bytes,
        )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_API_KEY,
        use_interactions_api=True,
        output_version="v1",
        client_args={"transport": httpx.MockTransport(handler)},
    )
    history = [
        HumanMessage("Turn 1"),
        AIMessage(""),
        HumanMessage("Turn 2"),
        AIMessage([]),
        HumanMessage("Turn 3"),
        AIMessage(
            [{"type": "text", "text": "", "extras": {"signature": sig_b64}}],
            response_metadata={"model_provider": "google_genai"},
        ),
        HumanMessage("Use tool"),
        AIMessage(
            [{"type": "text", "text": "", "extras": {"signature": sig_b64}}],
            tool_calls=[{"id": "tc_s", "name": "lookup", "args": {"k": "v"}}],
            response_metadata={"model_provider": "google_genai"},
        ),
        ToolMessage("found", tool_call_id="tc_s", name="lookup"),
    ]
    chunks = list(llm.stream(history))
    merged: AIMessageChunk = chunks[0]
    for ch in chunks[1:]:
        merged = merged + ch
    assert merged.text == "OK"

    sent_steps = captured[-1]["input"]
    assert [s["type"] for s in sent_steps] == [
        "user_input",
        "user_input",
        "user_input",
        "user_input",
        "thought",
        "function_call",
        "function_result",
    ]
    assert sent_steps[4] == {"type": "thought", "signature": sig_b64}
    assert '"text": ""' not in json.dumps(captured[-1])
