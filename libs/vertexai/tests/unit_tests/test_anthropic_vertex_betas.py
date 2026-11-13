import json
from typing import Any

import httpx
import pytest

from langchain_google_vertexai.model_garden import ChatAnthropicVertex

MODEL = "claude-test-model"
BETAS = ["beta-one-2025-01-01", "beta-two-2025-01-01"]
THINKING = {
    "type": "adaptive",
    "block_binding": {"prefix_mismatch_behavior": "drop_block"},
}
MESSAGE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": MODEL,
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 2, "output_tokens": 1},
}
EVENTS = (
    "event: content_block_delta\ndata: "
    '{"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"hello"}}\n\n'
)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("betas", [BETAS, [], None])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_beta_request_serialization(
    streaming: bool, betas: list[str] | None, asynchronous: bool
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(500, json={"error": {"message": "retry"}})
        if streaming:
            return httpx.Response(
                200, text=EVENTS, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=MESSAGE)

    transport = httpx.MockTransport(handle)
    async with httpx.AsyncClient(transport=transport) as async_http_client:
        with httpx.Client(transport=transport) as http_client:
            model = ChatAnthropicVertex(
                project="test-project",
                location="eu",
                model=MODEL,
                access_token="test-token",
                http_client=http_client,
                async_http_client=async_http_client,
                max_retries=2,
                wait_exponential_kwargs={"min": 0, "max": 0},
                model_kwargs={
                    "betas": betas,
                    "thinking": THINKING,
                    "extra_body": {"custom_field": "kept"},
                },
            )
            if asynchronous and streaming:
                assert [chunk async for chunk in model.astream("hi")]
            elif asynchronous:
                assert await model.ainvoke("hi")
            elif streaming:
                assert list(model.stream("hi"))
            else:
                assert model.invoke("hi")

    assert len(requests) == 2
    for request in requests:
        body: dict[str, Any] = json.loads(request.content)
        assert request.url.path.endswith(
            ":streamRawPredict" if streaming else ":rawPredict"
        )
        assert request.headers.get("anthropic-beta") == (
            ",".join(betas) if betas else None
        )
        assert "betas" not in body
        assert "anthropic_beta" not in body
        assert body["custom_field"] == "kept"
        assert body["thinking"] == THINKING
        assert body.get("stream", False) is streaming


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("betas", [BETAS, [], None])
@pytest.mark.asyncio
async def test_invocation_betas_override_model_betas(
    asynchronous: bool, betas: list[str] | None
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=MESSAGE)

    transport = httpx.MockTransport(handle)
    async with httpx.AsyncClient(transport=transport) as async_http_client:
        with httpx.Client(transport=transport) as http_client:
            model = ChatAnthropicVertex(
                project="test-project",
                location="eu",
                model=MODEL,
                access_token="test-token",
                http_client=http_client,
                async_http_client=async_http_client,
                model_kwargs={"betas": ["default-beta"]},
                additional_headers={"x-custom": "kept"},
            )
            if asynchronous:
                await model.ainvoke("hi", betas=betas)
            else:
                model.invoke("hi", betas=betas)

    assert requests[0].headers.get("anthropic-beta") == (
        ",".join(betas) if betas else None
    )
    assert requests[0].headers["x-custom"] == "kept"


def test_additional_beta_header_without_betas() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=MESSAGE)

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        model = ChatAnthropicVertex(
            project="test-project",
            location="eu",
            model=MODEL,
            access_token="test-token",
            http_client=http_client,
            additional_headers={"anthropic-beta": BETAS[0]},
        )
        model.invoke("hi")

    assert requests[0].headers["anthropic-beta"] == BETAS[0]
    assert "anthropic_beta" not in json.loads(requests[0].content)
