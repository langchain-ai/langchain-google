"""Unit tests for Model Armor Middleware."""

from typing import Any, Optional, cast
from unittest.mock import MagicMock, patch

import langchain.agents as lc_agents
import pytest
from google.cloud.modelarmor_v1 import FilterMatchState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from langchain_google_community.model_armor.middleware import ModelArmorMiddleware
from langchain_google_community.model_armor.runnable import (
    ModelArmorSanitizePromptRunnable,
    ModelArmorSanitizeResponseRunnable,
)


class DummySanitizationResult:
    """Dummy sanitization result for testing."""

    def __init__(self, match_found: bool = False, sdp_text: Optional[str] = None):
        self.filter_match_state = (
            FilterMatchState.MATCH_FOUND
            if match_found
            else FilterMatchState.NO_MATCH_FOUND
        )
        self.filter_results: dict[str, str] = {}


def _agent_state(messages: list[Any]) -> lc_agents.AgentState:
    return cast(lc_agents.AgentState, {"messages": messages})


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_middleware_initialization_with_both_sanitizers(
    mock_get_client: MagicMock,
) -> None:
    """Test middleware initialization with both sanitizers."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    prompt_sanitizer = ModelArmorSanitizePromptRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
    )
    response_sanitizer = ModelArmorSanitizeResponseRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
    )

    middleware = ModelArmorMiddleware(
        prompt_sanitizer=prompt_sanitizer,
        response_sanitizer=response_sanitizer,
    )

    assert middleware.prompt_sanitizer is prompt_sanitizer
    assert middleware.response_sanitizer is response_sanitizer


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_middleware_initialization_prompt_only(mock_get_client: MagicMock) -> None:
    """Test middleware initialization with only prompt sanitizer."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    prompt_sanitizer = ModelArmorSanitizePromptRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
    )

    middleware = ModelArmorMiddleware(prompt_sanitizer=prompt_sanitizer)

    assert middleware.prompt_sanitizer is prompt_sanitizer
    assert middleware.response_sanitizer is None


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_middleware_initialization_response_only(mock_get_client: MagicMock) -> None:
    """Test middleware initialization with only response sanitizer."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    response_sanitizer = ModelArmorSanitizeResponseRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
    )

    middleware = ModelArmorMiddleware(response_sanitizer=response_sanitizer)

    assert middleware.prompt_sanitizer is None
    assert middleware.response_sanitizer is response_sanitizer


def test_middleware_initialization_no_sanitizers_raises() -> None:
    """Test middleware raises ValueError when no sanitizers provided."""
    with pytest.raises(ValueError, match="At least one of"):
        ModelArmorMiddleware()


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_before_model_safe_content(mock_get_client: MagicMock) -> None:
    """Test before_model hook with safe content."""
    mock_client = MagicMock()
    mock_client.sanitize_user_prompt.return_value = type(
        "Result",
        (),
        {"sanitization_result": DummySanitizationResult(match_found=False)},
    )()
    mock_get_client.return_value = mock_client

    prompt_sanitizer = ModelArmorSanitizePromptRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
        fail_open=False,
    )
    middleware = ModelArmorMiddleware(prompt_sanitizer=prompt_sanitizer)

    state = _agent_state([HumanMessage(content="What is the capital of France?")])
    runtime = MagicMock(spec=Runtime)

    result = middleware.before_model(state, runtime)
    assert result is None


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_before_model_unsafe_content_strict(mock_get_client: MagicMock) -> None:
    """Test before_model hook with unsafe content in strict mode."""
    mock_client = MagicMock()
    mock_client.sanitize_user_prompt.return_value = type(
        "Result",
        (),
        {"sanitization_result": DummySanitizationResult(match_found=True)},
    )()
    mock_get_client.return_value = mock_client

    prompt_sanitizer = ModelArmorSanitizePromptRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
        fail_open=False,
    )
    middleware = ModelArmorMiddleware(prompt_sanitizer=prompt_sanitizer)

    state = _agent_state([HumanMessage(content="malicious content")])
    runtime = MagicMock(spec=Runtime)

    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "content policy violations" in result["messages"][0].content


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_before_model_empty_messages(mock_get_client: MagicMock) -> None:
    """Test before_model hook with empty messages."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    prompt_sanitizer = ModelArmorSanitizePromptRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
    )
    middleware = ModelArmorMiddleware(prompt_sanitizer=prompt_sanitizer)

    state = _agent_state([])
    runtime = MagicMock(spec=Runtime)

    result = middleware.before_model(state, runtime)
    assert result is None


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_before_model_skipped_when_no_prompt_sanitizer(
    mock_get_client: MagicMock,
) -> None:
    """Test before_model returns None when no prompt sanitizer configured."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    response_sanitizer = ModelArmorSanitizeResponseRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
    )
    middleware = ModelArmorMiddleware(response_sanitizer=response_sanitizer)

    state = _agent_state([HumanMessage(content="test message")])
    runtime = MagicMock(spec=Runtime)

    result = middleware.before_model(state, runtime)
    assert result is None


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_after_model_safe_content(mock_get_client: MagicMock) -> None:
    """Test after_model hook with safe content."""
    mock_client = MagicMock()
    mock_client.sanitize_model_response.return_value = type(
        "Result",
        (),
        {"sanitization_result": DummySanitizationResult(match_found=False)},
    )()
    mock_get_client.return_value = mock_client

    response_sanitizer = ModelArmorSanitizeResponseRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
        fail_open=False,
    )
    middleware = ModelArmorMiddleware(response_sanitizer=response_sanitizer)

    state = _agent_state([AIMessage(content="The capital of France is Paris.")])
    runtime = MagicMock(spec=Runtime)

    result = middleware.after_model(state, runtime)
    assert result is None


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_after_model_unsafe_content_strict(mock_get_client: MagicMock) -> None:
    """Test after_model hook with unsafe content in strict mode."""
    mock_client = MagicMock()
    mock_client.sanitize_model_response.return_value = type(
        "Result",
        (),
        {"sanitization_result": DummySanitizationResult(match_found=True)},
    )()
    mock_get_client.return_value = mock_client

    response_sanitizer = ModelArmorSanitizeResponseRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
        fail_open=False,
    )
    middleware = ModelArmorMiddleware(response_sanitizer=response_sanitizer)

    state = _agent_state([AIMessage(content="malicious response")])
    runtime = MagicMock(spec=Runtime)

    # Should return jump_to="end" instead of raising exception
    result = middleware.after_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "content policy violations" in result["messages"][0].content


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_after_model_empty_content(mock_get_client: MagicMock) -> None:
    """Test after_model hook with empty content."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    response_sanitizer = ModelArmorSanitizeResponseRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
    )
    middleware = ModelArmorMiddleware(response_sanitizer=response_sanitizer)

    state = _agent_state([AIMessage(content="")])
    runtime = MagicMock(spec=Runtime)

    result = middleware.after_model(state, runtime)
    assert result is None


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_after_model_skipped_when_no_response_sanitizer(
    mock_get_client: MagicMock,
) -> None:
    """Test after_model returns None when no response sanitizer configured."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    prompt_sanitizer = ModelArmorSanitizePromptRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
    )
    middleware = ModelArmorMiddleware(prompt_sanitizer=prompt_sanitizer)

    state = _agent_state([AIMessage(content="test response")])
    runtime = MagicMock(spec=Runtime)

    result = middleware.after_model(state, runtime)
    assert result is None


@patch("langchain_google_community.model_armor._client_utils._get_model_armor_client")
def test_middleware_fail_open_mode(mock_get_client: MagicMock) -> None:
    """Test middleware in fail-open mode."""
    mock_client = MagicMock()
    mock_client.sanitize_user_prompt.return_value = type(
        "Result",
        (),
        {"sanitization_result": DummySanitizationResult(match_found=True)},
    )()
    mock_get_client.return_value = mock_client

    prompt_sanitizer = ModelArmorSanitizePromptRunnable(
        project="test-project",
        location="us-central1",
        template_id="test-template",
        fail_open=True,
    )
    middleware = ModelArmorMiddleware(prompt_sanitizer=prompt_sanitizer)

    state = _agent_state([HumanMessage(content="malicious content")])
    runtime = MagicMock(spec=Runtime)

    # Should not raise in fail-open mode
    result = middleware.before_model(state, runtime)
    assert result is None
