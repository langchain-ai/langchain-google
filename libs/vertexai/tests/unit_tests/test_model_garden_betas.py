import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from langchain_google_vertexai.model_garden import ChatAnthropicVertex

MODEL = "claude-sonnet-4-6"
BETAS = ["thinking-binding-controls-2026-08-01", "another-beta"]


@pytest.mark.parametrize("mode", ["generate", "agenerate", "stream", "astream"])
@pytest.mark.parametrize(
    "beta_source",
    ["model_kwargs", "invocation", "empty", "override_empty", "absent", "header"],
)
async def test_vertex_beta_request(mode: str, beta_source: str) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if mode in {"stream", "astream"}:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=""
            )
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    transport = httpx.MockTransport(respond)
    async with httpx.AsyncClient(transport=transport) as async_client:
        with httpx.Client(transport=transport) as sync_client:
            model_kwargs: dict[str, Any] = {"extra_body": {"custom_field": "value"}}
            if beta_source == "model_kwargs":
                model_kwargs["betas"] = BETAS
            elif beta_source in {"invocation", "override_empty"}:
                model_kwargs["betas"] = ["overridden-beta"]
            elif beta_source == "empty":
                model_kwargs["betas"] = []
            headers = {"x-custom-header": "value"}
            if beta_source == "header":
                headers["anthropic-beta"] = "explicit-beta"
            model = ChatAnthropicVertex(
                model=MODEL,
                project="test-project",
                location="eu",
                access_token="test-token",
                http_client=sync_client,
                async_http_client=async_client,
                additional_headers=headers,
                model_kwargs=model_kwargs,
            )
            kwargs: dict[str, Any] = {}
            if beta_source == "invocation":
                kwargs["betas"] = BETAS
            elif beta_source == "override_empty":
                kwargs["betas"] = []
            messages: list[BaseMessage] = [HumanMessage(content="hello")]
            if mode == "generate":
                model._generate(messages, **kwargs)
            elif mode == "agenerate":
                await model._agenerate(messages, **kwargs)
            elif mode == "stream":
                list(model._stream(messages, **kwargs))
            else:
                _ = [chunk async for chunk in model._astream(messages, **kwargs)]

    assert len(requests) == 1
    request = requests[0]
    body = json.loads(request.content)
    assert body["custom_field"] == "value"
    assert "anthropic_beta" not in body
    assert "betas" not in body
    assert request.headers["x-custom-header"] == "value"
    if beta_source in {"model_kwargs", "invocation"}:
        assert request.headers["anthropic-beta"] == ",".join(BETAS)
    elif beta_source == "header":
        assert request.headers["anthropic-beta"] == "explicit-beta"
    else:
        assert "anthropic-beta" not in request.headers
