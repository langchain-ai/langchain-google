"""Unit tests for `langchain_google_genai.utils`."""

from unittest.mock import MagicMock

import pytest
from google.genai import types
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import SecretStr

from langchain_google_genai import ChatGoogleGenerativeAI, create_context_cache

FAKE_API_KEY = "fake-api-key"


@tool
def search_database(query: str) -> str:
    """Search the database."""
    return f"Results for: {query}"


@tool
def send_email(to: str) -> str:
    """Send an email."""
    return f"Sent to {to}"


def _model_with_fake_client() -> tuple[ChatGoogleGenerativeAI, MagicMock]:
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash", google_api_key=SecretStr(FAKE_API_KEY)
    )
    client = MagicMock()
    client.caches.create.return_value = MagicMock(name="cachedContents/abc")
    client.caches.create.return_value.name = "cachedContents/abc"
    object.__setattr__(model, "client", client)
    return model, client


def _created_config(client: MagicMock) -> types.CreateCachedContentConfig:
    return client.caches.create.call_args.kwargs["config"]


def _function_calling_config(client: MagicMock) -> types.FunctionCallingConfig:
    tool_config = _created_config(client).tool_config
    assert tool_config is not None
    assert tool_config.function_calling_config is not None
    return tool_config.function_calling_config


def test_create_context_cache_tool_choice_sets_tool_config() -> None:
    """`tool_choice` must be stored on the cache as its `tool_config`."""
    model, client = _model_with_fake_client()

    name = create_context_cache(
        model,
        messages=[SystemMessage(content="sys"), HumanMessage(content="ctx")],
        tools=[search_database, send_email],
        tool_choice="search_database",
    )

    assert name == "cachedContents/abc"
    fcc = _function_calling_config(client)
    assert fcc.mode == "ANY"
    assert fcc.allowed_function_names == ["search_database"]


def test_create_context_cache_tool_choice_any_allows_all_tools() -> None:
    model, client = _model_with_fake_client()

    create_context_cache(
        model,
        messages=[HumanMessage(content="ctx")],
        tools=[search_database, send_email],
        tool_choice="any",
    )

    fcc = _function_calling_config(client)
    assert fcc.mode == "ANY"
    assert fcc.allowed_function_names == ["search_database", "send_email"]


def test_create_context_cache_without_tool_choice_has_no_tool_config() -> None:
    model, client = _model_with_fake_client()

    create_context_cache(
        model, messages=[HumanMessage(content="ctx")], tools=[search_database]
    )

    assert _created_config(client).tool_config is None


def test_create_context_cache_tool_choice_requires_tools() -> None:
    model, client = _model_with_fake_client()

    with pytest.raises(ValueError, match="'tool_choice' can only be specified"):
        create_context_cache(
            model, messages=[HumanMessage(content="ctx")], tool_choice="any"
        )

    client.caches.create.assert_not_called()
