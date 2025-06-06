import uuid
from unittest.mock import MagicMock

import pytest
from google.cloud.modelarmor_v1 import FilterMatchState, SanitizationResult
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_google_community.model_armor.runnable import (
    ModelArmorSanitizePromptRunnable,
    ModelArmorSanitizeResponseRunnable,
)


class DummySanitizationResult:
    def __init__(self, match_found=False):
        self.sanitization_result = SanitizationResult(
            filter_match_state=(
                FilterMatchState.MATCH_FOUND
                if match_found
                else FilterMatchState.NO_MATCH_FOUND
            ),
            filter_results=[],
        )


@pytest.fixture
def mock_client():
    client = MagicMock()
    return client


def test_prompt_safe(mock_client):
    mock_client.sanitize_user_prompt.return_value = DummySanitizationResult(
        match_found=False
    )
    runnable = ModelArmorSanitizePromptRunnable(
        client=mock_client,
        template_id="test-template",
        fail_open=False,
    )
    assert runnable.invoke("hello") == "hello"


def test_prompt_unsafe_fail_open_false(mock_client):
    mock_client.sanitize_user_prompt.return_value = DummySanitizationResult(
        match_found=True
    )
    runnable = ModelArmorSanitizePromptRunnable(
        client=mock_client,
        template_id="test-template",
        fail_open=False,
    )
    with pytest.raises(ValueError):
        runnable.invoke("unsafe prompt")


def test_prompt_unsafe_fail_open_true(mock_client):
    mock_client.sanitize_user_prompt.return_value = DummySanitizationResult(
        match_found=True
    )
    runnable = ModelArmorSanitizePromptRunnable(
        client=mock_client,
        template_id="test-template",
        fail_open=True,
    )
    assert runnable.invoke("unsafe prompt") == "unsafe prompt"


def test_prompt_return_findings(mock_client):
    mock_client.sanitize_user_prompt.return_value = DummySanitizationResult(
        match_found=False
    )
    runnable = ModelArmorSanitizePromptRunnable(
        client=mock_client,
        template_id="test-template",
        fail_open=True,
        return_findings=True,
    )
    result = runnable.invoke("hello")
    assert result["prompt"] == "hello"
    assert hasattr(result["findings"], "filter_match_state")


def test_response_safe(mock_client):
    mock_client.sanitize_model_response.return_value = DummySanitizationResult(
        match_found=False
    )
    runnable = ModelArmorSanitizeResponseRunnable(
        client=mock_client,
        template_id="test-template",
        fail_open=False,
    )
    assert runnable.invoke("response") == "response"


def test_response_unsafe_fail_open_false(mock_client):
    mock_client.sanitize_model_response.return_value = DummySanitizationResult(
        match_found=True
    )
    runnable = ModelArmorSanitizeResponseRunnable(
        client=mock_client,
        template_id="test-template",
        fail_open=False,
    )
    with pytest.raises(ValueError):
        runnable.invoke("unsafe response")


def test_response_unsafe_fail_open_true(mock_client):
    mock_client.sanitize_model_response.return_value = DummySanitizationResult(
        match_found=True
    )
    runnable = ModelArmorSanitizeResponseRunnable(
        client=mock_client,
        template_id="test-template",
        fail_open=True,
    )
    assert runnable.invoke("unsafe response") == "unsafe response"


def test_response_return_findings(mock_client):
    mock_client.sanitize_model_response.return_value = DummySanitizationResult(
        match_found=False
    )
    runnable = ModelArmorSanitizeResponseRunnable(
        client=mock_client,
        template_id="test-template",
        fail_open=True,
        return_findings=True,
    )
    result = runnable.invoke("response")
    assert result["response"] == "response"
    assert hasattr(result["findings"], "filter_match_state")


class EventCatcher(BaseCallbackHandler):
    def __init__(self):
        self.events = []

    def on_custom_event(self, event_name, payload, **kwargs):
        self.events.append((event_name, payload))


def test_prompt_event_dispatch(mock_client):
    mock_client.sanitize_user_prompt.return_value = DummySanitizationResult(
        match_found=True
    )
    catcher = EventCatcher()
    runnable = ModelArmorSanitizePromptRunnable(
        client=mock_client,
        template_id="test-template",
        fail_open=True,
    )
    config = RunnableConfig(run_id=str(uuid.uuid4()), callbacks=[catcher])
    # Pass config through the lambda!
    wrapper = RunnableLambda(
        lambda x, config=None: runnable.invoke(x, config=config)
    )
    wrapper.invoke("unsafe prompt", config=config)
    assert any(e[0] == "on_model_armor_finding" for e in catcher.events)


def test_response_event_dispatch(mock_client):
    mock_client.sanitize_model_response.return_value = DummySanitizationResult(
        match_found=True
    )
    catcher = EventCatcher()
    runnable = ModelArmorSanitizeResponseRunnable(
        client=mock_client,
        template_id="test-template",
        fail_open=True,
    )
    config = RunnableConfig(
        run_name="test_runner", run_id=str(uuid.uuid4()), callbacks=[catcher]
    )
    # Pass config through the lambda!
    wrapper = RunnableLambda(
        lambda x, config=None: runnable.invoke(x, config=config)
    )
    wrapper.invoke("unsafe response", config=config)
    assert any(e[0] == "on_model_armor_finding" for e in catcher.events)
