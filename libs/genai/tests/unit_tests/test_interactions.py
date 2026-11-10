"""Unit tests for Gemini Interactions API routing, fields, and error classification."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai._gaos.lib.compat_errors import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    Candidate,
    Content,
    GenerateContentResponse,
    McpServer,
    Part,
    ServiceTier,
    StreamableHttpTransport,
)
from google.genai.types import Tool as GoogleTool
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAIError,
    GoogleAPIError,
    GoogleAuthenticationError,
    GoogleConnectionError,
    GoogleContextOverflowError,
    GoogleInteractionsAPIError,
    GoogleInteractionsContextOverflowError,
    GoogleInvalidRequestError,
    GoogleModelNotFoundError,
    GooglePermissionDeniedError,
    GoogleRateLimitError,
    GoogleTimeoutError,
    _acall_with_interactions_retry,
    _call_with_interactions_retry,
    _disable_sdk_internal_interactions_retries,
    _get_scoped_interactions_client,
    _handle_interactions_error,
)

MODEL_NAME = "gemini-3.8-flash"
FAKE_API_KEY = "fake-api-key"


def _make_httpx_response(status_code: int, body: dict[str, Any]) -> httpx.Response:
    req = httpx.Request(
        "POST", "https://generativelanguage.googleapis.com/v1beta/interactions"
    )
    return httpx.Response(status_code=status_code, json=body, request=req)


def _sample_generate_content_response(text: str = "pong") -> GenerateContentResponse:
    return GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(role="model", parts=[Part(text=text)]),
                finish_reason="STOP",
            )
        ]
    )


def test_interactions_fields_defaults_and_validation() -> None:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=FAKE_API_KEY)
    assert llm.use_interactions_api is None
    assert llm.use_previous_interaction_id is False
    assert llm.store is None
    assert llm.service_tier is None
    assert llm.agent is None
    assert llm.agent_config is None
    assert llm.environment is None
    assert llm.background is None
    assert llm.polling_interval is None

    configured = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_API_KEY,
        use_interactions_api=True,
        use_previous_interaction_id=True,
        store=True,
        service_tier="priority",
        agent="deep-research-pro-preview-12-2025",
        agent_config={"thinking_summaries": "auto"},
        environment="default",
        background=True,
        polling_interval=1.5,
    )
    assert configured.use_interactions_api is True
    assert configured.use_previous_interaction_id is True
    assert configured.store is True
    assert configured.service_tier == "priority"
    assert configured.agent == "deep-research-pro-preview-12-2025"
    assert configured.agent_config == {"thinking_summaries": "auto"}
    assert configured.environment == "default"
    assert configured.background is True
    assert configured.polling_interval == 1.5
    assert "use_interactions_api" not in configured.model_kwargs

    with pytest.raises(ValueError, match="polling_interval must be positive"):
        ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=FAKE_API_KEY,
            polling_interval=0.0,
        )


def test_use_interactions_api_priority_cascade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGCHAIN_GOOGLE_USE_INTERACTIONS_API", raising=False)
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=FAKE_API_KEY)
    assert llm._use_interactions_api({}) is False

    # Priority 4: output_version == "interactions/v1"
    llm_ov = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_API_KEY,
        output_version="interactions/v1",
    )
    assert llm_ov._use_interactions_api({}) is True

    # Priority 3: env var overrides output_version and auto-routing
    monkeypatch.setenv("LANGCHAIN_GOOGLE_USE_INTERACTIONS_API", "FALSE")
    assert llm_ov._use_interactions_api({}) is False
    llm_prev = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_API_KEY,
        use_previous_interaction_id=True,
    )
    assert llm_prev._use_interactions_api({}) is False

    for truthy in ("1", "true", "YES", " True "):
        monkeypatch.setenv("LANGCHAIN_GOOGLE_USE_INTERACTIONS_API", truthy)
        assert llm._use_interactions_api({}) is True

    # Priority 2: self.use_interactions_api overrides env var
    llm_explicit_false = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_API_KEY,
        use_interactions_api=False,
        agent="deep-research-pro-preview-12-2025",
    )
    assert llm_explicit_false._use_interactions_api({}) is False

    monkeypatch.setenv("LANGCHAIN_GOOGLE_USE_INTERACTIONS_API", "0")
    llm_explicit_true = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_API_KEY,
        use_interactions_api=True,
    )
    assert llm_explicit_true._use_interactions_api({}) is True

    # Priority 1: per-call kwargs and model_kwargs
    assert llm_explicit_true._use_interactions_api({"use_interactions_api": False}) is (
        False
    )
    assert llm_explicit_false._use_interactions_api({"use_interactions_api": True}) is (
        True
    )
    with pytest.warns(UserWarning, match="use_interactions_api"):
        llm_mk = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=FAKE_API_KEY,
            model_kwargs={"use_interactions_api": True},
        )
    assert llm_mk._use_interactions_api({}) is True
    assert llm_mk._use_interactions_api({"use_interactions_api": False}) is False


@pytest.mark.parametrize(
    ("init_kwargs", "call_kwargs", "expected"),
    [
        ({"use_previous_interaction_id": True}, {}, True),
        ({}, {"previous_interaction_id": "int_123"}, True),
        ({"agent": "deep-research-pro-preview-12-2025"}, {}, True),
        ({}, {"agent": "custom-agent"}, True),
        ({"agent_config": {"type": "deep-research"}}, {}, True),
        ({"background": True}, {}, True),
        ({}, {"background": True}, True),
        ({"environment": "browser"}, {}, True),
        ({"service_tier": "flex"}, {}, False),
        ({}, {"service_tier": "priority"}, False),
        ({}, {"webhook_config": {"url": "https://example.com/hook"}}, True),
        ({"model": "deep-research-pro-preview-12-2025"}, {}, True),
        ({"model": "models/deep-research-pro-preview-12-2025"}, {}, True),
        ({"model": "antigravity-agent-1"}, {}, True),
        ({}, {"model": "models/antigravity-preview"}, True),
        (
            {},
            {
                "tools": [
                    {
                        "type": "file_search",
                        "file_search_store_names": ["fileSearchStores/s1"],
                    }
                ]
            },
            True,
        ),
        (
            {},
            {"tools": [{"type": "mcp_server", "name": "s", "url": "https://mcp.test"}]},
            True,
        ),
        ({}, {"tools": [{"type": "retrieval"}]}, True),
        ({}, {"tools": [{"type": "google_search"}]}, False),
        ({"output_version": "v1"}, {}, False),
    ],
)
def test_use_interactions_api_auto_routing(
    monkeypatch: pytest.MonkeyPatch,
    init_kwargs: dict[str, Any],
    call_kwargs: dict[str, Any],
    *,
    expected: bool,
) -> None:
    monkeypatch.delenv("LANGCHAIN_GOOGLE_USE_INTERACTIONS_API", raising=False)
    params: dict[str, Any] = {
        "model": MODEL_NAME,
        "google_api_key": FAKE_API_KEY,
        **init_kwargs,
    }
    llm = ChatGoogleGenerativeAI(**params)
    assert llm._use_interactions_api(call_kwargs) is expected


async def test_use_interactions_api_false_never_leaks_into_generate_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGCHAIN_GOOGLE_USE_INTERACTIONS_API", "true")
    mock_client_cls = Mock()
    mock_models = Mock()
    mock_aio_models = Mock()
    mock_models.generate_content = Mock(
        return_value=_sample_generate_content_response()
    )
    mock_models.generate_content_stream = Mock(
        return_value=iter([_sample_generate_content_response("chunk")])
    )
    mock_aio_models.generate_content = AsyncMock(
        return_value=_sample_generate_content_response("async")
    )

    async def _agen() -> Any:
        yield _sample_generate_content_response("achunk")

    mock_aio_models.generate_content_stream = AsyncMock(return_value=_agen())
    mock_client_cls.return_value.models = mock_models
    mock_client_cls.return_value.aio.models = mock_aio_models

    with patch("langchain_google_genai.chat_models.Client", mock_client_cls):
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=FAKE_API_KEY,
            use_interactions_api=True,
        )
        res = llm.invoke("ping", use_interactions_api=False)
        assert res.text == "pong"
        ares = await llm.ainvoke("ping", use_interactions_api=False)
        assert ares.text == "async"
        chunks = list(llm.stream("ping", use_interactions_api=False))
        assert chunks[0].text == "chunk"
        achunks = [c async for c in llm.astream("ping", use_interactions_api=False)]
        assert achunks[0].text == "achunk"

        # Direct _prepare_request call must also drop use_interactions_api cleanly
        from langchain_core.messages import HumanMessage

        req = llm._prepare_request(
            [HumanMessage(content="ping")], use_interactions_api=False
        )
        assert "use_interactions_api" not in req
        assert not hasattr(req["config"], "use_interactions_api")


async def test_dispatch_routes_to_interactions_methods() -> None:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_API_KEY,
        use_interactions_api=True,
    )
    fake_result = ChatResult(generations=[ChatGeneration(message=AIMessage("via_ia"))])
    fake_chunk = ChatGenerationChunk(message=AIMessageChunk(content="delta"))

    with (
        patch.object(
            ChatGoogleGenerativeAI,
            "_generate_interactions",
            return_value=fake_result,
        ) as mock_gen,
        patch.object(
            ChatGoogleGenerativeAI,
            "_agenerate_interactions",
            new=AsyncMock(return_value=fake_result),
        ) as mock_agen,
        patch.object(
            ChatGoogleGenerativeAI,
            "_stream_interactions",
            return_value=iter([fake_chunk]),
        ) as mock_stream,
    ):
        assert llm.invoke("hi").text == "via_ia"
        assert "use_interactions_api" not in mock_gen.call_args.kwargs
        assert (await llm.ainvoke("hi", use_interactions_api=True)).text == "via_ia"
        assert "use_interactions_api" not in mock_agen.call_args.kwargs
        assert next(llm.stream("hi")).text == "delta"
        assert mock_stream.called

    # Also verify bound MCP tool triggers routing via explicit `tools` parameter
    llm_auto = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=FAKE_API_KEY)
    bound = llm_auto.bind_tools(
        [{"type": "mcp_server", "name": "docs", "url": "https://mcp.example.com"}]
    )
    with patch.object(
        ChatGoogleGenerativeAI,
        "_generate_interactions",
        return_value=fake_result,
    ) as mock_auto_gen:
        assert bound.invoke("search docs").text == "via_ia"
        assert mock_auto_gen.called


def test_mcp_server_headers_redacted_without_mutating_caller() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    raw_headers = {"Authorization": "Bearer super-secret-token"}
    mcp_tool: dict[str, Any] = {
        "type": "mcp_server",
        "name": "internal",
        "url": "https://mcp.internal/sse",
        "headers": raw_headers,
    }
    caller_tools: list[Any] = [add, mcp_tool]

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=FAKE_API_KEY)
    inv_params = llm._get_invocation_params(tools=caller_tools)
    redacted_mcp = inv_params["tools"][1]
    assert redacted_mcp["headers"] == "**REDACTED**"
    # Caller structures MUST remain completely unmutated
    assert mcp_tool["headers"] == {"Authorization": "Bearer super-secret-token"}
    assert raw_headers["Authorization"] == "Bearer super-secret-token"
    assert caller_tools[1]["headers"]["Authorization"] == "Bearer super-secret-token"

    # Also check bind_tools preserves MCP server tool for runtime while tracing redacts
    bound = llm.bind_tools(caller_tools)
    assert bound.kwargs["tools"][1]["headers"] == {  # type: ignore[attr-defined]
        "Authorization": "Bearer super-secret-token"
    }
    bound_inv = llm._get_invocation_params(**bound.kwargs)  # type: ignore[attr-defined]
    assert bound_inv["tools"][1]["headers"] == "**REDACTED**"
    assert bound.kwargs["tools"][1]["headers"] == {  # type: ignore[attr-defined]
        "Authorization": "Bearer super-secret-token"
    }

    # Also check model_kwargs tools redaction + agent ls_model_name
    llm_mk = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_API_KEY,
        agent="deep-research-pro-preview-12-2025",
        model_kwargs={"tools": [mcp_tool]},
    )
    mk_params = llm_mk._get_invocation_params()
    assert "super-secret-token" not in str(mk_params)
    assert llm_mk.model_kwargs["tools"][0]["headers"] == {
        "Authorization": "Bearer super-secret-token"
    }
    ls = llm_mk._get_ls_params()
    assert ls["ls_model_name"] == "deep-research-pro-preview-12-2025"
    assert ls["ls_provider"] == "google_genai"


@pytest.mark.parametrize(
    ("exc_factory", "expected_types", "is_retryable"),
    [
        (
            lambda: BadRequestError(
                "The input token count exceeds the maximum number of tokens allowed.",
                response=_make_httpx_response(
                    400, {"error": {"message": "token limit exceeded"}}
                ),
                body={"error": {"message": "token limit exceeded"}},
            ),
            (
                GoogleInteractionsContextOverflowError,
                GoogleContextOverflowError,
                ContextOverflowError,
                ClientError,
                ChatGoogleGenerativeAIError,
            ),
            False,
        ),
        (
            lambda: BadRequestError(
                "Invalid generation_config",
                response=_make_httpx_response(400, {"error": {"message": "bad"}}),
                body={"error": {"message": "bad"}},
            ),
            (
                GoogleInvalidRequestError,
                ModelInvalidRequestError,
                ChatGoogleGenerativeAIError,
            ),
            False,
        ),
        (
            lambda: UnprocessableEntityError(
                "Unprocessable schema",
                response=_make_httpx_response(422, {"error": {"message": "bad"}}),
                body={"error": {"message": "bad"}},
            ),
            (
                GoogleInvalidRequestError,
                ModelInvalidRequestError,
                ChatGoogleGenerativeAIError,
            ),
            False,
        ),
        (
            lambda: AuthenticationError(
                "Invalid API key",
                response=_make_httpx_response(401, {"error": {"message": "unauth"}}),
                body={"error": {"message": "unauth"}},
            ),
            (
                GoogleAuthenticationError,
                ModelAuthenticationError,
                ChatGoogleGenerativeAIError,
            ),
            False,
        ),
        (
            lambda: PermissionDeniedError(
                "Forbidden",
                response=_make_httpx_response(403, {"error": {"message": "denied"}}),
                body={"error": {"message": "denied"}},
            ),
            (
                GooglePermissionDeniedError,
                ModelPermissionDeniedError,
                ChatGoogleGenerativeAIError,
            ),
            False,
        ),
        (
            lambda: NotFoundError(
                "Model not found",
                response=_make_httpx_response(404, {"error": {"message": "missing"}}),
                body={"error": {"message": "missing"}},
            ),
            (
                GoogleModelNotFoundError,
                ModelNotFoundError,
                ChatGoogleGenerativeAIError,
            ),
            False,
        ),
        (
            lambda: RateLimitError(
                "Quota exceeded",
                response=_make_httpx_response(429, {"error": {"message": "quota"}}),
                body={"error": {"message": "quota"}},
            ),
            (
                GoogleRateLimitError,
                ModelRateLimitError,
                ChatGoogleGenerativeAIError,
            ),
            True,
        ),
        (
            lambda: InternalServerError(
                "Backend unavailable",
                response=_make_httpx_response(503, {"error": {"message": "503"}}),
                body="plain string error body",
            ),
            (
                GoogleInteractionsAPIError,
                GoogleAPIError,
                ServerError,
                ModelAPIError,
                ChatGoogleGenerativeAIError,
            ),
            True,
        ),
        (
            lambda: APITimeoutError(
                request=httpx.Request("POST", "https://example.com")
            ),
            (
                GoogleTimeoutError,
                ModelTimeoutError,
                ModelAPIError,
                ChatGoogleGenerativeAIError,
            ),
            True,
        ),
        (
            lambda: APIConnectionError(
                message="DNS failed",
                request=httpx.Request("POST", "https://example.com"),
            ),
            (
                GoogleConnectionError,
                ModelConnectionError,
                ModelAPIError,
                ChatGoogleGenerativeAIError,
            ),
            True,
        ),
        (
            lambda: APIError(
                "Unexpected non-status failure",
                httpx.Request("POST", "https://example.com"),
                body=None,
            ),
            (
                GoogleInteractionsAPIError,
                ModelAPIError,
                ChatGoogleGenerativeAIError,
            ),
            True,
        ),
    ],
)
def test_handle_interactions_error_taxonomy(
    exc_factory: Any,
    expected_types: tuple[type[BaseException], ...],
    *,
    is_retryable: bool,
) -> None:
    raw_exc = exc_factory()
    with pytest.raises(ChatGoogleGenerativeAIError) as exc_info:
        _handle_interactions_error(raw_exc, model_name=MODEL_NAME)
    err = exc_info.value
    for cls in expected_types:
        assert isinstance(err, cls), f"Expected {type(err)} to be instance of {cls}"
    assert isinstance(err, ModelError)
    assert err.is_retryable is is_retryable
    assert err.__cause__ is raw_exc


def test_handle_interactions_conflict_stays_unclassified() -> None:
    conflict = ConflictError(
        "Aborted",
        response=_make_httpx_response(409, {"error": {"message": "conflict"}}),
        body={"error": {"message": "conflict"}},
    )
    with pytest.raises(ChatGoogleGenerativeAIError) as exc_info:
        _handle_interactions_error(conflict, model_name=MODEL_NAME)
    assert not isinstance(exc_info.value, ModelError)


async def test_interactions_retry_sync_and_async() -> None:
    req = httpx.Request("POST", "https://example.com")
    rate_err = RateLimitError(
        "429",
        response=_make_httpx_response(429, {"error": {"message": "slow down"}}),
        body=None,
    )
    bad_err = BadRequestError(
        "400",
        response=_make_httpx_response(400, {"error": {"message": "invalid"}}),
        body=None,
    )

    sync_fn = Mock(side_effect=[rate_err, APITimeoutError(request=req), "ok"])
    with patch("tenacity.nap.time.sleep"):
        result = _call_with_interactions_retry(
            sync_fn, model_name=MODEL_NAME, max_retries=3
        )
    assert result == "ok"
    assert sync_fn.call_count == 3

    # Non-retryable 400 fails immediately on attempt 1
    non_retry_fn = Mock(side_effect=bad_err)
    with pytest.raises(GoogleInvalidRequestError):
        _call_with_interactions_retry(
            non_retry_fn, model_name=MODEL_NAME, max_retries=4
        )
    assert non_retry_fn.call_count == 1

    # Async retry exhaustion surfaces classified error
    async_fn = AsyncMock(
        side_effect=[
            APIConnectionError(request=req),
            InternalServerError(
                "500",
                response=_make_httpx_response(500, {"error": {"message": "boom"}}),
                body=None,
            ),
        ]
    )
    with (
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(GoogleInteractionsAPIError),
    ):
        await _acall_with_interactions_retry(
            async_fn, model_name=MODEL_NAME, max_retries=2
        )
    assert async_fn.call_count == 2


def test_disable_sdk_internal_interactions_retries_and_private_scoping() -> None:
    """Private Interactions wrapper never mutates shared `llm.client.interactions`."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=FAKE_API_KEY, max_retries=1
    )
    assert llm.client is not None
    public_ia = llm.client.interactions
    orig_strategy = public_ia.sdk_configuration.retry_config.strategy  # type: ignore[union-attr]
    assert orig_strategy != "none"

    # Concurrent calls across worker threads return a single private wrapper, preserve
    # Speakeasy `_hooks`, and leave `llm.client.interactions` untouched.
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: _get_scoped_interactions_client(llm, is_async=False)[0],
                range(16),
            )
        )
    first_priv = results[0]
    assert all(r is first_priv for r in results)
    assert first_priv is not public_ia
    assert first_priv.sdk_configuration.retry_config.strategy == "none"
    assert "_hooks" in first_priv.sdk_configuration.__dict__
    assert public_ia.sdk_configuration.retry_config.strategy == orig_strategy  # type: ignore[union-attr]

    # Per-request base_url + client_args produce an isolated ephemeral client
    scoped, eph = _get_scoped_interactions_client(
        llm,
        is_async=False,
        base_url_override="https://proxy.corp/gemini/",
        client_args_override={"follow_redirects": True},
    )
    try:
        assert scoped is not first_priv
        assert scoped._get_url(None, None) == "https://proxy.corp/gemini"
        assert "_hooks" in scoped.sdk_configuration.__dict__
        assert (
            first_priv._get_url(None, None)
            == "https://generativelanguage.googleapis.com"
        )
        assert eph is not None
    finally:
        if eph is not None:
            eph.close()

    # Direct helper still preserves `_hooks` on cloned config
    clone_ia, _ = _get_scoped_interactions_client(
        llm, is_async=False, base_url_override="https://gw.example.com"
    )
    _disable_sdk_internal_interactions_retries(clone_ia)
    assert clone_ia.sdk_configuration.retry_config.strategy == "none"
    assert "_hooks" in clone_ia.sdk_configuration.__dict__


async def test_http_408_retry_and_continuation_interrupted_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 408 retries cleanly and `ContinuationInterruptedError` classifies causes."""
    from langchain_google_genai import chat_models as cm_mod

    err_408 = APIStatusError(
        "Request Timeout",
        response=_make_httpx_response(408, {"error": {"message": "slow"}}),
        body={"error": {"message": "slow"}},
    )
    fn_408 = Mock(side_effect=[err_408, "recovered"])
    with patch("tenacity.nap.time.sleep"):
        assert (
            _call_with_interactions_retry(fn_408, model_name=MODEL_NAME, max_retries=2)
            == "recovered"
        )
    assert fn_408.call_count == 2

    with pytest.raises(GoogleTimeoutError):
        _handle_interactions_error(err_408, model_name=MODEL_NAME)

    # Simulate SDK>=2.34 `ContinuationInterruptedError` on both 2.25 and 2.34+ envs
    cont_cls = getattr(genai_errors, "ContinuationInterruptedError", None)
    if cont_cls is None:

        class _FakeContinuationInterruptedError(Exception):
            def __init__(self, msg: str, *, continuation_token: str = "tok") -> None:
                super().__init__(msg)
                self.continuation_token = continuation_token

        cont_cls = _FakeContinuationInterruptedError
        monkeypatch.setattr(cm_mod, "_ContinuationInterruptedError", cont_cls)
        monkeypatch.setattr(
            cm_mod,
            "_INTERACTIONS_CATCHABLE_ERRORS",
            (*cm_mod._INTERACTIONS_CATCHABLE_ERRORS, cont_cls),
        )

    # 1) Deadline timeout cause -> GoogleTimeoutError (not retried after hop progress!)
    c_to = cont_cls("Interrupted after 1 hop", continuation_token="tok_t")
    c_to.__cause__ = TimeoutError("budget exhausted")
    one_shot = Mock(side_effect=c_to)
    with pytest.raises(GoogleTimeoutError) as to_info:
        _call_with_interactions_retry(one_shot, model_name=MODEL_NAME, max_retries=3)
    assert one_shot.call_count == 1
    assert to_info.value.__cause__ is c_to
    assert getattr(to_info.value.__cause__, "continuation_token", None) == "tok_t"

    # 2) Rate-limit cause -> GoogleRateLimitError
    c_rl = cont_cls("Hop 2 rate limited", continuation_token="tok_rl")
    c_rl.__cause__ = RateLimitError(
        "429",
        response=_make_httpx_response(429, {"error": {"message": "quota"}}),
        body=None,
    )
    a_shot = AsyncMock(side_effect=c_rl)
    with pytest.raises(GoogleRateLimitError) as rl_info:
        await _acall_with_interactions_retry(
            a_shot, model_name=MODEL_NAME, max_retries=3
        )
    assert a_shot.call_count == 1
    assert rl_info.value.__cause__ is c_rl

    # 3) Context-overflow cause -> GoogleInteractionsContextOverflowError
    c_ov = cont_cls("Hop 2 overflow", continuation_token="tok_ov")
    c_ov.__cause__ = BadRequestError(
        "exceeds the maximum number of tokens allowed",
        response=_make_httpx_response(
            400, {"error": {"message": "exceeds the maximum number of tokens allowed"}}
        ),
        body=None,
    )
    with pytest.raises(GoogleInteractionsContextOverflowError) as ov_info:
        _handle_interactions_error(c_ov, model_name=MODEL_NAME)
    assert ov_info.value.__cause__ is c_ov

    # 4) Auto-routing activates on continuation_token & automatic_continuation
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=FAKE_API_KEY)
    assert llm._use_interactions_api({"continuation_token": "tok_resume"}) is True
    assert llm._use_interactions_api({"automatic_continuation": False}) is True


def test_bind_tools_cross_mode_generate_content_safety() -> None:
    """Interactions tool dicts translate cleanly (or reject Interactions-only) on GC."""
    from langchain_core.messages import HumanMessage

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=FAKE_API_KEY)

    # 1) Shared server tool dicts on default generateContent never produce MISSING_NAME
    flat_fn: dict[str, Any] = {
        "type": "function",
        "name": "get_weather",
        "description": "Lookup weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    flat_snapshot = {**flat_fn, "parameters": dict(flat_fn["parameters"])}
    bound = llm.bind_tools(
        [
            {"type": "google_search"},
            {
                "type": "google_maps",
                "enable_widget": True,
                "latitude": 48.8588,
                "longitude": 2.2943,
            },
            {"type": "code_execution"},
            {"type": "url_context"},
            {"type": "computer_use", "environment": "browser"},
            flat_fn,
            {
                "name": "anth_fn",
                "description": "Anthropic schema",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        ]
    )
    req = llm._prepare_request([HumanMessage(content="hi")], **bound.kwargs)  # type: ignore[attr-defined]
    assert flat_fn == flat_snapshot  # caller dict never mutated
    cfg_tools = req["config"].tools
    assert cfg_tools[0].google_search is not None
    assert cfg_tools[1].google_maps is not None
    assert cfg_tools[1].google_maps.enable_widget is True
    assert cfg_tools[2].code_execution is not None
    assert cfg_tools[3].url_context is not None
    assert cfg_tools[4].computer_use is not None
    fn_names = [fd.name for fd in cfg_tools[5].function_declarations]
    assert fn_names == ["get_weather", "anth_fn"]
    assert "MISSING_NAME" not in fn_names
    lat_lng = req["config"].tool_config.retrieval_config.lat_lng
    assert lat_lng.latitude == pytest.approx(48.8588)
    assert lat_lng.longitude == pytest.approx(2.2943)

    # 2) Interactions-only tools forced onto use_interactions_api=False raise cleanly
    llm_ia = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=FAKE_API_KEY, use_interactions_api=True
    )
    ia_only_items: list[dict[str, Any]] = [
        {"type": "mcp_server", "name": "m", "url": "https://mcp.example"},
        {"type": "file_search", "file_search_store_names": ["fileSearchStores/s1"]},
        {"type": "retrieval"},
    ]
    for ia_only in ia_only_items:
        bound_only = llm_ia.bind_tools([ia_only])
        with pytest.raises(GoogleInvalidRequestError, match=str(ia_only["type"])):
            bound_only.invoke("hi", use_interactions_api=False)

    # 3) GoogleTool(mcp_servers=[...]) headers are also redacted in invocation params
    g_mcp = GoogleTool(
        mcp_servers=[
            McpServer(
                name="corp",
                streamable_http_transport=StreamableHttpTransport(
                    url="https://mcp.corp",
                    headers={"Authorization": "Bearer TOPSEC"},
                ),
            )
        ]
    )
    bound_gmcp = llm.bind_tools([g_mcp])
    inv_params = llm._get_invocation_params(**bound_gmcp.kwargs)  # type: ignore[attr-defined]
    assert "TOPSEC" not in str(inv_params)
    assert "**REDACTED**" in str(inv_params)

    # 4) Direct per-call GoogleTool(mcp_servers=[...]) is redacted without mutating it
    direct_params = llm._get_invocation_params(tools=[g_mcp])
    assert "TOPSEC" not in str(direct_params)
    assert "**REDACTED**" in str(direct_params)
    assert g_mcp.mcp_servers is not None
    assert g_mcp.mcp_servers[0].streamable_http_transport is not None
    assert g_mcp.mcp_servers[0].streamable_http_transport.headers == {
        "Authorization": "Bearer TOPSEC"
    }

    # 5) Shorthand {"mcp_servers": [{"name": ..., "url": ...}]} binds & redacts cleanly
    shorthand_mcp = {
        "mcp_servers": [
            {
                "name": "short-srv",
                "url": "https://mcp.short/sse",
                "headers": {"Authorization": "Bearer SHORT_SECRET"},
            }
        ]
    }
    bound_short = llm.bind_tools([shorthand_mcp])
    short_params = llm._get_invocation_params(**bound_short.kwargs)  # type: ignore[attr-defined]
    assert "SHORT_SECRET" not in str(short_params)
    assert "**REDACTED**" in str(short_params)
    assert shorthand_mcp["mcp_servers"][0]["headers"] == {
        "Authorization": "Bearer SHORT_SECRET"
    }


def test_service_tier_preserves_generate_content_on_both_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`service_tier` stays on `generateContent` by default and applies on both APIs."""
    monkeypatch.delenv("LANGCHAIN_GOOGLE_USE_INTERACTIONS_API", raising=False)
    gc_resp = _sample_generate_content_response("ok")

    # 1) Unset use_interactions_api + service_tier stays on generateContent (including
    # regional Vertex AI us-central1)
    llm_default = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=FAKE_API_KEY)
    assert llm_default.client is not None
    mock_gc_def = Mock(return_value=gc_resp)
    with patch.object(llm_default.client.models, "generate_content", mock_gc_def):
        res_def = llm_default.invoke("hi", service_tier="priority")
    assert res_def.text == "ok"
    assert mock_gc_def.call_args.kwargs["config"].service_tier == ServiceTier.PRIORITY

    llm_vtx_regional = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
        project="test-proj",
        location="us-central1",
        vertexai=True,
    )
    assert llm_vtx_regional.client is not None
    mock_gc_reg = Mock(return_value=gc_resp)
    with patch.object(llm_vtx_regional.client.models, "generate_content", mock_gc_reg):
        llm_vtx_regional.invoke("hi", service_tier="priority")
    reg_cfg = mock_gc_reg.call_args.kwargs["config"]
    assert reg_cfg.service_tier is None
    assert reg_cfg.http_options is not None
    assert reg_cfg.http_options.headers == {
        "X-Vertex-AI-LLM-Shared-Request-Type": "priority"
    }

    # 2) Constructor service_tier (str or ServiceTier enum) sent on generateContent,
    # including when overriding use_interactions_api=True -> False per-call
    llm_flex = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_API_KEY,
        service_tier="flex",
        use_interactions_api=False,
    )
    assert llm_flex.client is not None
    mock_gc_flex = Mock(return_value=gc_resp)
    with patch.object(llm_flex.client.models, "generate_content", mock_gc_flex):
        llm_flex.invoke("Summarize transcript.")
        assert mock_gc_flex.call_args.kwargs["config"].service_tier == ServiceTier.FLEX

        # Per-call override beats constructor value
        llm_flex.invoke("Urgent!", service_tier=ServiceTier.PRIORITY)
        assert (
            mock_gc_flex.call_args.kwargs["config"].service_tier == ServiceTier.PRIORITY
        )

    llm_ia_flex = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=FAKE_API_KEY,
        service_tier=ServiceTier.FLEX,
        use_interactions_api=True,
    )
    assert llm_ia_flex.client is not None
    mock_gc_fallback = Mock(return_value=gc_resp)
    with patch.object(llm_ia_flex.client.models, "generate_content", mock_gc_fallback):
        llm_ia_flex.invoke("Fallback call", use_interactions_api=False)
    assert mock_gc_fallback.call_args.kwargs["config"].service_tier == ServiceTier.FLEX

    # 3) Vertex AI constructor service_tier preserves timeout/retries and yields to
    # explicit caller X-Vertex-AI-LLM-Shared-Request-Type header
    llm_vtx = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
        project="test-proj",
        location="global",
        vertexai=True,
        service_tier=ServiceTier.FLEX,
        timeout=7,
        max_retries=3,
    )
    vtx_cfg = llm_vtx._prepare_request([])["config"]
    assert vtx_cfg.service_tier is None
    assert vtx_cfg.http_options is not None
    assert vtx_cfg.http_options.headers == {
        "X-Vertex-AI-LLM-Shared-Request-Type": "flex"
    }
    assert vtx_cfg.http_options.timeout == 7000
    assert vtx_cfg.http_options.retry_options is not None
    assert vtx_cfg.http_options.retry_options.attempts == 3

    vtx_override = llm_vtx._prepare_request(
        [],
        http_options={
            "headers": {
                "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
                "X-Custom": "keep-me",
            }
        },
    )["config"]
    assert vtx_override.http_options.headers == {
        "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
        "X-Custom": "keep-me",
    }

    # Unset service_tier adds no Vertex header
    llm_vtx_plain = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
        project="test-proj",
        vertexai=True,
    )
    plain_cfg = llm_vtx_plain._prepare_request([])["config"]
    assert plain_cfg.service_tier is None
    assert plain_cfg.http_options is None or plain_cfg.http_options.headers is None

    # Forward-compatible unknown tier passes through cleanly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        llm_future = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=FAKE_API_KEY,
            service_tier="tier-from-the-future",
        )
        assert llm_future._prepare_request([])["config"].service_tier == (
            "tier-from-the-future"
        )


def test_guarded_gaos_imports_do_not_break_generate_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`chat_models` imports cleanly even if private `google.genai._gaos` moves."""
    import importlib.util
    import sys

    from langchain_google_genai import chat_models as cm_mod

    assert cm_mod.__file__ is not None
    spec = importlib.util.spec_from_file_location(
        "_isolated_chat_models_no_gaos", cm_mod.__file__
    )
    assert spec is not None and spec.loader is not None
    isolated_mod = importlib.util.module_from_spec(spec)

    for mod_name in list(sys.modules):
        if mod_name == "google.genai._gaos" or mod_name.startswith(
            "google.genai._gaos."
        ):
            monkeypatch.setitem(sys.modules, mod_name, None)

    sys.modules[spec.name] = isolated_mod
    try:
        spec.loader.exec_module(isolated_mod)
        assert isolated_mod._HAS_GAOS_SDK is False
        llm = isolated_mod.ChatGoogleGenerativeAI(
            model=MODEL_NAME, google_api_key=FAKE_API_KEY
        )
        assert llm._use_interactions_api({}) is False
    finally:
        sys.modules.pop(spec.name, None)


async def test_interactions_retry_predicates_transient_vs_non_retryable() -> None:
    """Transient 5xx/429/timeout errors retry; 4xx/safety/cancelled/501 fail fast."""
    transient_500 = GoogleInteractionsAPIError(
        code=500,
        response_json={"error": {"code": 500, "message": "failed", "status": "FAILED"}},
        response=None,
    )
    fn_recover = Mock(
        side_effect=[
            transient_500,
            GoogleRateLimitError("429 quota"),
            GoogleConnectionError("reset"),
            "ok",
        ]
    )
    with patch("tenacity.nap.time.sleep"):
        assert (
            _call_with_interactions_retry(
                fn_recover, model_name=MODEL_NAME, max_retries=4
            )
            == "ok"
        )
    assert fn_recover.call_count == 4

    raw_503 = ServerError(
        503,
        {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}},
        None,
    )
    afn_recover = AsyncMock(side_effect=[raw_503, "async-ok"])
    with patch("asyncio.sleep", new=AsyncMock()):
        assert (
            await _acall_with_interactions_retry(
                afn_recover, model_name=MODEL_NAME, max_retries=3
            )
            == "async-ok"
        )
    assert afn_recover.call_count == 2

    non_retryable_errors: list[Exception] = [
        GoogleInteractionsAPIError(
            code=499,
            response_json={
                "error": {"code": 499, "message": "cancelled", "status": "CANCELLED"}
            },
            response=None,
        ),
        GoogleInteractionsAPIError(
            code=400,
            response_json={
                "error": {"code": 400, "message": "blocked", "status": "SAFETY"}
            },
            response=None,
        ),
        GoogleInteractionsAPIError(
            code=501,
            response_json={
                "error": {
                    "code": 501,
                    "message": "unsupported",
                    "status": "UNIMPLEMENTED",
                }
            },
            response=None,
        ),
        GoogleInteractionsContextOverflowError(
            code=400,
            response_json={
                "error": {
                    "code": 400,
                    "message": "context window",
                    "status": "INVALID_ARGUMENT",
                }
            },
            response=None,
        ),
        GoogleInvalidRequestError("400 bad"),
        GoogleAuthenticationError("401 auth"),
        GooglePermissionDeniedError("403 perm"),
        GoogleModelNotFoundError("404 missing"),
    ]
    for fatal_err in non_retryable_errors:
        sync_fatal = Mock(side_effect=fatal_err)
        with (
            patch("tenacity.nap.time.sleep"),
            pytest.raises(type(fatal_err)),
        ):
            _call_with_interactions_retry(
                sync_fatal, model_name=MODEL_NAME, max_retries=4
            )
        assert sync_fatal.call_count == 1

        async_fatal = AsyncMock(side_effect=fatal_err)
        with (
            patch("asyncio.sleep", new=AsyncMock()),
            pytest.raises(type(fatal_err)),
        ):
            await _acall_with_interactions_retry(
                async_fatal, model_name=MODEL_NAME, max_retries=4
            )
        assert async_fatal.call_count == 1


async def test_client_swap_invalidation_owned_gaos_aclose_and_httpx_override() -> None:
    """Client reassignment invalidates cache; `aclose()` shuts owned `_gaos` pools."""
    from google.genai import Client

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key="key-1",
        use_interactions_api=True,
        client_args={"trust_env": False, "headers": {"X-Model-Cfg": "1"}},
    )
    ia1, _ = _get_scoped_interactions_client(llm, is_async=False)
    ia1_async, _ = _get_scoped_interactions_client(llm, is_async=True)

    # Reassigning `llm.client` invalidates BOTH sync & async cached wrappers
    replacement = Client(api_key="key-2")
    llm.client = replacement
    ia2, _ = _get_scoped_interactions_client(llm, is_async=False)
    ia2_async, _ = _get_scoped_interactions_client(llm, is_async=True)
    assert ia2 is not ia1
    assert ia2_async is not ia1_async
    sec_fn_or_obj = ia2.sdk_configuration.security
    sec_val = sec_fn_or_obj() if callable(sec_fn_or_obj) else sec_fn_or_obj
    assert getattr(sec_val, "api_key", None) == "key-2"

    # Simulate Vertex mTLS non-supplied `_gaos` fallback sync & async httpx clients
    owned_sync = httpx.Client()
    owned_async = httpx.AsyncClient()
    ia2.sdk_configuration.client = owned_sync
    ia2.sdk_configuration.client_supplied = False
    ia2_async.sdk_configuration.async_client = owned_async
    ia2_async.sdk_configuration.async_client_supplied = False

    # Merge `model.client_args` into `client_args_override` + borrow `httpx_client`
    scoped_eph, eph_client = _get_scoped_interactions_client(
        llm,
        is_async=False,
        client_args_override={"headers": {"X-Per-Req": "2"}},
    )
    try:
        assert eph_client is not None
        assert scoped_eph.sdk_configuration.client is eph_client
        assert eph_client._trust_env is False
        assert eph_client.follow_redirects is True
        assert eph_client.headers["x-per-req"] == "2"
    finally:
        if eph_client is not None:
            eph_client.close()

    borrowed = httpx.AsyncClient()
    try:
        scoped_b, none_eph = _get_scoped_interactions_client(
            llm, is_async=True, httpx_client_override=borrowed
        )
        assert none_eph is None
        assert scoped_b.sdk_configuration.async_client is borrowed
        assert not borrowed.is_closed
    finally:
        await borrowed.aclose()

    await llm.aclose()
    assert owned_sync.is_closed
    assert owned_async.is_closed

    # `model_construct` caches private wrapper cleanly across calls without rebuilding
    raw_c = Client(api_key="key-mc")
    try:
        constructed = ChatGoogleGenerativeAI.model_construct(
            model=MODEL_NAME, client=raw_c
        )
        mc1, _ = _get_scoped_interactions_client(constructed, is_async=True)
        mc2, _ = _get_scoped_interactions_client(constructed, is_async=True)
        assert mc1 is mc2
        await constructed.aclose()
    finally:
        raw_c.close()


def test_httpx2_and_timeout_error_classification_and_scoped_client() -> None:
    """`httpx2` and `TimeoutError` classify cleanly and work with scoped clients."""
    import httpx2

    from langchain_google_genai import chat_models as cm_mod

    with pytest.raises(GoogleTimeoutError):
        _handle_interactions_error(TimeoutError("hung"), MODEL_NAME)
    with pytest.raises(GoogleTimeoutError):
        _handle_interactions_error(httpx2.ReadTimeout("v2 read timeout"), MODEL_NAME)
    with pytest.raises(GoogleConnectionError):
        _handle_interactions_error(httpx2.ConnectError("v2 conn reset"), MODEL_NAME)

    cont_cls = getattr(genai_errors, "ContinuationInterruptedError", None)
    assert cont_cls is not None
    interrupted_to = cont_cls("hop timeout")
    interrupted_to.__cause__ = httpx2.ReadTimeout("v2 hop timeout")
    with pytest.raises(GoogleTimeoutError):
        _handle_interactions_error(interrupted_to, MODEL_NAME)

    interrupted_conn = cont_cls("hop conn")
    interrupted_conn.__cause__ = httpx2.ConnectError("v2 hop conn")
    with pytest.raises(GoogleConnectionError):
        _handle_interactions_error(interrupted_conn, MODEL_NAME)

    # Scoped Interactions client with `httpx2.MockTransport` produces an `httpx2.Client`
    def _v2_handler(req: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"id": "i_v2"}, request=req)

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=FAKE_API_KEY, use_interactions_api=True
    )
    scoped, eph = cm_mod._get_scoped_interactions_client(
        llm,
        is_async=False,
        client_args_override={"transport": httpx2.MockTransport(_v2_handler)},
    )
    try:
        assert isinstance(eph, httpx2.Client)
        assert scoped.sdk_configuration.client is eph
    finally:
        if eph is not None:
            eph.close()
        if llm.client is not None:
            llm.client.close()


def test_generate_content_handles_file_data_and_skips_non_code_server_results() -> None:
    """Replay `file_data`, `interactions/v1`, and MCP results to `generateContent`."""
    from langchain_core.messages import HumanMessage

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=FAKE_API_KEY)

    ai_v0_uri = AIMessage(
        content=[
            {"type": "file_data", "file_uri": "files/img123", "mime_type": "image/png"}
        ],
        response_metadata={"model_provider": "google_genai"},
    )
    req_v0 = llm._prepare_request(
        [HumanMessage("q"), ai_v0_uri, HumanMessage("f")], use_interactions_api=False
    )
    p0 = req_v0["contents"][1].parts
    assert p0 is not None and len(p0) == 1
    assert p0[0].file_data is not None and p0[0].file_data.file_uri == "files/img123"

    ai_ia_v1 = AIMessage(
        content=[
            {"type": "image", "url": "files/img456", "mime_type": "image/png"},
            {"type": "video", "url": "files/vid789", "mime_type": "video/mp4"},
            {
                "type": "server_tool_call",
                "name": "get_doc",
                "id": "m1",
                "args": {},
                "extras": {"step_type": "mcp_server_tool_call", "server_name": "docs"},
            },
            {
                "type": "server_tool_result",
                "tool_call_id": "m1",
                "status": "success",
                "output": {"k": "v"},
                "extras": {
                    "block_type": "mcp_server_tool_result",
                    "step_type": "mcp_server_tool_result",
                },
            },
            {"type": "text", "text": "All set"},
        ],
        response_metadata={
            "model_provider": "google_genai",
            "output_version": "interactions/v1",
        },
    )
    # Note: full `_compat` video conversion runs once `_compat` is loaded; check MCP
    # server_tool_call/result skipping directly on v0 too:
    ai_v0_mcp = AIMessage(
        content=[
            {
                "type": "server_tool_call",
                "name": "get_doc",
                "id": "m1",
                "args": {},
                "extras": {"step_type": "mcp_server_tool_call", "server_name": "docs"},
            },
            {
                "type": "server_tool_result",
                "tool_call_id": "m1",
                "status": "success",
                "output": {"k": "v"},
                "extras": {
                    "block_type": "mcp_server_tool_result",
                    "step_type": "mcp_server_tool_result",
                },
            },
            {"type": "text", "text": "done"},
        ],
        response_metadata={"model_provider": "google_genai"},
    )
    req_mcp = llm._prepare_request(
        [HumanMessage("q"), ai_v0_mcp, HumanMessage("f")], use_interactions_api=False
    )
    mcp_parts = req_mcp["contents"][1].parts
    assert mcp_parts is not None
    assert all(p.code_execution_result is None for p in mcp_parts)
    assert [p.text for p in mcp_parts] == ["done"]
    assert ai_ia_v1.response_metadata["output_version"] == "interactions/v1"
