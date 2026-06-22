"""Test that __del__ does not corrupt the thread's event loop state."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import SecretStr

from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_NAME = "models/gemini-2.0-flash"
FAKE_API_KEY = "fake-api-key"


def _create_llm_with_mock_client() -> ChatGoogleGenerativeAI:
    """Create a ChatGoogleGenerativeAI with a mock client that has aio.aclose()."""
    mock_client = Mock()
    mock_client.return_value.models = Mock()
    # Set up async client mock
    mock_aio = Mock()
    mock_aio.aclose = AsyncMock()
    mock_client.return_value.aio = mock_aio

    with patch("langchain_google_genai.chat_models.Client", mock_client):
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=SecretStr(FAKE_API_KEY),
        )
    return llm


def test_del_preserves_existing_idle_event_loop() -> None:
    """__del__ must not destroy an existing idle event loop on the thread.

    This is the primary bug scenario: GC triggers __del__ during sync teardown
    (e.g., pytest with a session-scoped event loop). The loop exists on the
    thread but is not running. The original code used get_running_loop(), missed
    the idle loop, and called set_event_loop(None) which destroyed it.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        llm = _create_llm_with_mock_client()
        llm.__del__()

        # The thread's event loop must still be the same one
        assert asyncio.get_event_loop() is loop
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_del_without_any_event_loop() -> None:
    """__del__ must not leave a stale event loop when none existed before."""
    asyncio.set_event_loop(None)

    llm = _create_llm_with_mock_client()
    llm.__del__()

    with pytest.raises(RuntimeError):
        asyncio.get_event_loop()


def test_del_with_closed_event_loop() -> None:
    """__del__ should still clean up async resources when the loop is closed."""
    loop = asyncio.new_event_loop()
    loop.close()
    asyncio.set_event_loop(loop)
    try:
        llm = _create_llm_with_mock_client()
        llm.__del__()

        # aclose should still have been called via a throwaway loop
        llm.client.aio.aclose.assert_awaited_once()
    finally:
        asyncio.set_event_loop(None)
