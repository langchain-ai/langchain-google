from unittest.mock import Mock, patch

import httpx
import pytest

from langchain_google_vertexai.model_garden import ChatAnthropicVertex

TIMEOUT_TEST_CASES = [
    pytest.param(30.0, id="float_timeout"),
    pytest.param(httpx.Timeout(timeout=30.0), id="httpx_timeout"),
    pytest.param(None, id="none_timeout"),
    pytest.param(
        ...,  # No timeout specified in constructor
        id="default_timeout",
    ),
]


@pytest.mark.parametrize("timeout_value", TIMEOUT_TEST_CASES)
def test_timeout_configuration(timeout_value) -> None:
    """Test that different timeout values are correctly handled."""
    with (
        patch("anthropic.AnthropicVertex") as mock_sync_client,
        patch("anthropic.AsyncAnthropicVertex") as mock_async_client,
    ):
        mock_sync_instance = Mock()
        mock_sync_instance.timeout = None if timeout_value is ... else timeout_value
        mock_sync_client.return_value = mock_sync_instance

        mock_async_instance = Mock()
        mock_async_instance.timeout = None if timeout_value is ... else timeout_value
        mock_async_client.return_value = mock_async_instance

        # Create chat instance with or without timeout parameter
        chat_kwargs = {"project": "test-project"}
        if timeout_value is not ...:
            chat_kwargs["timeout"] = timeout_value

        chat = ChatAnthropicVertex(**chat_kwargs)

        # Verify initialization
        mock_sync_client.assert_called_once()
        expected_timeout = None if timeout_value is ... else timeout_value
        assert mock_sync_client.call_args.kwargs["timeout"] == expected_timeout, (
            "Synchronous Anthropic instance not initialized with correct timeout"
        )

        mock_async_client.assert_called_once()
        assert mock_async_client.call_args.kwargs["timeout"] == expected_timeout, (
            "Asynchronous Anthropic instance not initialized with correct timeout"
        )

        # Verify the clients have the correct timeout after initialization
        assert chat.client.timeout == expected_timeout, (
            "Sync client timeout not set correctly after initialization"
        )
        assert chat.async_client.timeout == expected_timeout, (
            "Async client timeout not set correctly after initialization"
        )


def test_timeout_invalid() -> None:
    """Test that invalid timeout values raise appropriate errors."""
    with pytest.raises(ValueError) as exc_info:
        ChatAnthropicVertex(
            project="test-project",
            timeout="invalid",
        )
    assert "Input should be a valid number" in str(exc_info.value)
